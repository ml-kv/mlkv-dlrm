import os
import yaml
from typing import List, Union

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics

from persia.ctx import TrainCtx, eval_ctx, EmbeddingConfig
from persia.embedding.optim import Adam
from persia.embedding.data import PersiaBatch
from persia.env import get_rank, get_local_rank, get_world_size
from persia.logger import get_default_logger
from persia.data import DataLoader, IterableDataset, StreamingDataset
from persia.utils import setup_seed

from model import DNN
from data_generator import make_dataloader


logger = get_default_logger("nn_worker")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = str(False)
setup_seed(1)
config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config/criteo_ad_config.yml"
)
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class Meter:
    def __init__(self):
        self.all_pred = []
        self.all_target = []

    def add_item(self, pred, target):
        self.all_pred.append(pred.cpu().detach().numpy())
        self.all_target.append(target.cpu().detach().numpy())

    def calc_metric(self) -> float:
        raise NotImplementedError

    def get_metric_name(self) -> str:
        raise NotImplementedError


class AUCMeter(Meter):
    def __init__(self, label_num: int = 2):
        super().__init__()

        self.label_num = label_num

    def add_item(self, pred, target):
        output = torch.nn.functional.softmax(pred)
        # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/metrics/auc_op.cc

        if self.label_num == 2:
            output = output[:, -1]
            output = torch.reshape(output, (-1,))

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.all_pred.append(output)
        self.all_target.append(target)

    def calc_metric(self) -> Union[List[float], float]:
        all_pred, all_target = np.concatenate(
            self.all_pred), np.concatenate(self.all_target)

        if self.label_num == 2:
            test_auc = metrics.roc_auc_score(all_target, all_pred)
            return test_auc
        else:
            auc_set = []
            for label in range(self.label_num):
                cnt_class_label = label == all_target
                cnt_class_auc = metrics.roc_auc_score(
                    cnt_class_label, all_pred[:, label])
                auc_set.append(cnt_class_auc)
            logger.info(f"label auc: {auc_set}")
            return auc_set[1]

    def get_metric_name(self) -> str:
        return "auc"


class TestDataset(IterableDataset):
    def __init__(self, test_dir: str, batch_size: int = 128):
        super(TestDataset, self).__init__(buffer_size=10)
        self.loader = make_dataloader(test_dir, batch_size)
        logger.info(f"test dataset size is {len(self.loader)}")

    def __iter__(self):
        logger.info("test loader start to generating data...")
        for non_id_type_feature, id_type_features, label in self.loader:
            yield PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
                requires_grad=False,
            )


def test(model: torch.nn.Module, data_loader: DataLoader, cuda: bool):
    logger.info("start to test...")
    model.eval()

    with eval_ctx(model=model) as ctx:
        if loss_function == "cse":
            meter = AUCMeter(label_num=config.get("label_num", 2))
        else:
            raise NotImplementedError

        for (batch_idx, batch_data) in enumerate(tqdm(data_loader, desc="test...")):
            (pred, labels) = ctx.forward(batch_data)
            meter.add_item(pred, labels[0])
        result = meter.calc_metric()

    model.train()

    return meter.get_metric_name(), result


if __name__ == "__main__":
    model = DNN()
    logger.info("init Simple DNN model...")

    rank, device_id, world_size = get_rank(), get_local_rank(), get_world_size()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        mixed_precision = False
        torch.cuda.set_device(device_id)
        model.cuda(device_id)
    else:
        mixed_precision = False
        device_id = None
    logger.info(f"device_id is {device_id}")

    batch_size = config['batch_size']
    dense_lr = config['dense_lr']
    embedding_staleness = config['embedding_staleness']
    forward_buffer_size = config['forward_buffer_size']
    loss_function = config["loss_function"]
    sample_num = config['sample_num']
    sparse_lr = config['sparse_lr']
    sparses_dim = config["sparses_dim"]
    sparses_field = config["sparses_field"]
    test_dir = config['test_data_dir']
    test_interval = config["test_interval"]
    weight_bound = config['weight_bound']

    if config['dense_optimizer'] == 'Adam':
        dense_optimizer = torch.optim.Adam(model.parameters(), lr=dense_lr)
    else:
        raise NotImplementedError

    if config['sparse_optimizer'] == 'Adam':
        embedding_optimizer = Adam(lr=sparse_lr)
    else:
        raise NotImplementedError

    test_dataset = TestDataset(test_dir, batch_size=2560)

    embedding_config = EmbeddingConfig(
        emb_initialization=(-0.1, 0.1),
        weight_bound=weight_bound
    )
    epoch_idx = 0
    available_permits_upper_bound = embedding_staleness # / 10

    with TrainCtx(
        model=model,
        embedding_optimizer=embedding_optimizer,
        dense_optimizer=dense_optimizer,
        mixed_precision=mixed_precision,
        device_id=device_id,
        embedding_config=embedding_config,
    ) as ctx:
        train_dataloader = DataLoader(
            dataset=StreamingDataset(forward_buffer_size),
            forward_buffer_size=forward_buffer_size,
            embedding_staleness=embedding_staleness,
        )
        test_loader = DataLoader(test_dataset)

        logger.info("start to training...")
        for (batch_idx, data) in enumerate(train_dataloader):
            if available_permits_upper_bound \
                < train_dataloader.forward_engine.get_available_permits():
                import time
                time.sleep(1)

            (output, labels) = ctx.forward(data)
            label = labels[0]
            accuracy = (torch.round(output) == label).sum() / label.shape[0]

            if loss_function == "cse":
                label = torch.reshape(label, (-1,)).long()
                loss = torch.nn.functional.cross_entropy(output, label)
            else:
                raise NotImplementedError

            scaled_loss = ctx.backward(loss)

            logger.info(
                f"current batch idx: {batch_idx} loss: {float(loss)} scaled_loss: {float(scaled_loss)} accuracy: {float(accuracy)}"
            )
            if batch_idx % test_interval == 0 and batch_idx != 0:
                if rank == 0:
                    epoch_idx = (batch_size * batch_idx * world_size) // sample_num
                    metric_name, result = test(model, test_loader, use_cuda)
                    logger.info(
                        f"current epoch idx {epoch_idx}, current batch idx: {batch_idx}, test_{metric_name}: {result}")
                break
