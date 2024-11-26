import os
import yaml
from tqdm import tqdm

from persia.embedding.data import PersiaBatch
from persia.logger import get_logger
from persia.ctx import DataCtx

from data_generator import make_dataloader

logger = get_logger("data_loader")

config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config/criteo_terabyte_config.yml"
)
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
train_filepath = config['train_data_dir']
batch_size = config['batch_size']

if __name__ == "__main__":
    with DataCtx() as ctx:
        loader = make_dataloader(train_filepath, batch_size)
        for (non_id_type_feature, id_type_features, label) in tqdm(
            loader, desc="gen batch data..."
        ):
            persia_batch = PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
                requires_grad=True,
            )
            ctx.send_data(persia_batch)
