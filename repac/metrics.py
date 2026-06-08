"""Metrics for target-region reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_values(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if mask.shape != prediction.shape:
        mask = mask.expand_as(prediction)
    return (prediction - target)[mask.bool()]


def masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """MAE on the evaluation target region Omega."""

    values = _masked_values(prediction, target, mask).abs()
    if values.numel() == 0:
        raise ValueError("mask selects no entries")
    return values.mean()


def masked_rmse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """RMSE on the evaluation target region Omega."""

    values = _masked_values(prediction, target, mask)
    if values.numel() == 0:
        raise ValueError("mask selects no entries")
    return torch.sqrt((values**2).mean())
