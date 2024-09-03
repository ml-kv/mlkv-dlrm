from typing import List

import torch
import torch.nn as nn

import numpy as np

import yaml
import os

# config_path = os.environ['PERSIA_MODEL_CONFIG']
config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config/criteo_terabyte_config.yml"
)
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sparses_dim = config["sparses_dim"]
sparses_field = config["sparses_field"]
dense_dim = config["dense_dim"]

loss_function = config["loss_function"]
label_num = config["label_num"]
last_layer_size = [label_num] if loss_function == "cse" else [1]

class DNN(nn.Module):
    def __init__(
        self
    ):
        super(DNN, self).__init__()

        ln = config['ln']
        ln = [sparses_dim * sparses_field + dense_dim] + ln + last_layer_size
        ln_size = len(ln)
        layers = nn.ModuleList()
        for i in range(ln_size - 1):
            m = ln[i]
            n = ln[i + 1]
            LL = nn.Linear(int(m), int(n), bias=True)

            mean = 0.0
            std_dev = np.sqrt(2.0 / (m + n))
            W = np.random.normal(mean, std_dev, size=(n, m)).astype(np.float32)

            std_dev = np.sqrt(1.0 / n)
            bt = np.random.normal(mean, std_dev, size=(n)).astype(np.float32)

            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            layers.append(LL)

            if i < ln_size - 2:
                layers.append(nn.ReLU())

        self.fc = torch.nn.Sequential(*layers)

    def forward(
        self, non_id_tensors: List[torch.Tensor], embedding_tensors: List[torch.Tensor]
    ):
        dense_x = non_id_tensors[0]
        sparse = torch.cat(embedding_tensors, dim=1)
        x = torch.cat((sparse, dense_x), dim=1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DNN()
    print(model)

    total = 0
    for name, params in model.named_parameters():
        params_size = 1
        for s in params.shape:
            params_size = params_size * s
        total = total + params_size

    print(total)
    # batch_size = 4
    # dense = torch.ones(batch_size, dense_dim)
    # print(dense)
    # sparses = [torch.ones(batch_size, sparses_dim) for _ in range(sparses_field)]
    # target = torch.ones((batch_size,), dtype=torch.int64)
    # output = model(dense, sparses)
    # loss = torch.nn.functional.cross_entropy(output, target)
    # print(loss)
