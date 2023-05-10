import torch

from python_code.utils.config_singleton import Config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conf = Config()
