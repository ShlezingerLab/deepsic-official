import torch


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    ber = 1 - torch.mean(torch.all(prediction == target, dim=1).float()).item()
    return ber
