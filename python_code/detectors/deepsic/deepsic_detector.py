import torch
from torch import nn

from python_code import conf
from python_code.utils.constants import N_ANTS, N_USERS

HIDDEN_BASE_SIZE = 64


class DeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self):
        super(DeepSICDetector, self).__init__()
        classes_num = 2
        hidden_size = HIDDEN_BASE_SIZE * classes_num
        base_rx_size = N_ANTS
        linear_input = base_rx_size + (classes_num - 1) * (N_USERS - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        out0 = self.activation(self.fc1(rx))
        out1 = self.fc2(out0)
        return out1
