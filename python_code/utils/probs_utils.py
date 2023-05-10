import torch

from python_code import DEVICE
from python_code.utils.constants import HALF


def calculate_mimo_states(n_user: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    Calculates mimo states vector for the transmitted words. Number of states is 2/4/8 ** n_user depending on the
    constellation size.
    """
    states_enumerator = (2 ** torch.arange(n_user)).to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    """
    return torch.sign(p - HALF)
