"""Lightweight plug-in baselines used for paper comparisons."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import MLP


class GenericAdapter(nn.Module):
    """Bounded generic residual correction without prior promotion."""

    def __init__(self, feature_dim: int, channels: int, hidden_dim: int = 64, bound: float = 0.25) -> None:
        super().__init__()
        self.head = MLP(feature_dim, channels, hidden_dim)
        self.register_buffer("bound", torch.tensor(float(bound)))

    def forward(self, base: Tensor, features: Tensor) -> Tensor:
        return base + self.bound * torch.tanh(self.head(features))


class DoRAAdapter(nn.Module):
    """Direction-and-scale residual adapter inspired by DoRA-style decomposition."""

    def __init__(self, feature_dim: int, channels: int, hidden_dim: int = 64, bound: float = 0.25) -> None:
        super().__init__()
        self.direction = MLP(feature_dim, channels, hidden_dim)
        self.scale = MLP(feature_dim, channels, hidden_dim)
        self.register_buffer("bound", torch.tensor(float(bound)))

    def forward(self, base: Tensor, features: Tensor) -> Tensor:
        direction = torch.tanh(self.direction(features))
        scale = torch.sigmoid(self.scale(features))
        return base + self.bound * scale * direction


class CalibrationGuard(nn.Module):
    """Generic correction gated by learned local calibration evidence."""

    def __init__(self, feature_dim: int, evidence_dim: int, channels: int, hidden_dim: int = 64, bound: float = 0.25) -> None:
        super().__init__()
        self.residual = MLP(feature_dim, channels, hidden_dim)
        self.gate = MLP(evidence_dim, 1, hidden_dim)
        self.register_buffer("bound", torch.tensor(float(bound)))

    def forward(self, base: Tensor, features: Tensor, evidence: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(evidence))
        return base + gate * self.bound * torch.tanh(self.residual(features))


class FailureAnomalyGuard(CalibrationGuard):
    """Calibration guard variant with explicit failure/anomaly evidence inputs."""

