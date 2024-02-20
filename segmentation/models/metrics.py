import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, tolerance: float = 1e-2) -> torch.Tensor:
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)

        return (torch.abs(y_true - y_pred) < tolerance).sum() / y_true.shape[0]


class Dice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)

        intersection = (y_true * y_pred).sum()

        dice = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

        return dice
