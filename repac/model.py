"""Core RePAC module.

The module implements the post-backbone correction used in the paper:

    x_hat = x0 + Delta_g + alpha * rho * Delta_p.

The prior-aligned direction is treated as a candidate correction direction, not
as a guaranteed physical update. Conflict and failure cues should be included in
the evidence vector and learned by the reliability network; no fixed hard
suppression multiplier is applied here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


class MLP(nn.Sequential):
    """Small MLP used for lightweight post-backbone correction heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )


@dataclass
class RePACOutput:
    prediction: Tensor
    raw_prediction: Tensor
    delta_generic: Tensor
    delta_prior: Tensor
    rho: Tensor
    alpha: Tensor


def preserve_reliable_observations(
    raw_prediction: Tensor,
    observed: Tensor,
    reliable_mask: Tensor,
) -> Tensor:
    """Anchor reliable observed entries after correction.

    The mask is protocol-dependent. For missing-value imputation it can be the
    observation mask. For noisy or perturbed observations it should contain only
    the reliable observed subset, or be omitted entirely.
    """

    return reliable_mask * observed + (1.0 - reliable_mask) * raw_prediction


class RePAC(nn.Module):
    """Reliability-conditioned Prior-Aligned Correction.

    Parameters
    ----------
    channels:
        Number of output channels in the reconstructed signal.
    generic_dim:
        Feature dimension for the generic residual branch. In practice this can
        be a backbone hidden state, local representation, or handcrafted local
        residual feature.
    evidence_dim:
        Dimension of the reliability evidence vector.
    hidden_dim:
        Hidden size of the lightweight correction heads.
    generic_bound:
        Bound for the generic correction branch.
    prior_bound:
        Bound for the prior-aligned candidate correction branch.
    alpha_init:
        Initial global promotion strength. The learned value is exp(log_alpha).
    channelwise_rho:
        If false, rho is node-level and broadcast to all channels. If true, rho
        is produced channel-wise.
    """

    def __init__(
        self,
        channels: int,
        generic_dim: int,
        evidence_dim: int,
        hidden_dim: int = 64,
        generic_bound: float = 0.25,
        prior_bound: float = 0.25,
        alpha_init: float = 1.5,
        channelwise_rho: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")

        self.channels = channels
        self.channelwise_rho = channelwise_rho
        self.generic_head = MLP(generic_dim, channels, hidden_dim)
        self.rho_head = MLP(evidence_dim, channels if channelwise_rho else 1, hidden_dim)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))
        self.register_buffer("generic_bound", torch.tensor(float(generic_bound)))
        self.register_buffer("prior_bound", torch.tensor(float(prior_bound)))

    @property
    def alpha(self) -> Tensor:
        return torch.exp(self.log_alpha)

    def forward(
        self,
        base: Tensor,
        generic_features: Tensor,
        evidence: Tensor,
        prior_direction: Tensor,
        observed: Optional[Tensor] = None,
        reliable_mask: Optional[Tensor] = None,
    ) -> RePACOutput:
        """Apply RePAC to a backbone prediction.

        Shapes are expected to be broadcast-compatible. A common setting is
        ``base`` with shape ``[T, N, C]``, generic/evidence features with shape
        ``[T, N, D]``, and ``prior_direction`` with shape ``[T, N, C]``.
        """

        if base.shape[-1] != self.channels:
            raise ValueError(f"base last dimension must be {self.channels}")
        if prior_direction.shape[-1] != self.channels:
            raise ValueError(f"prior_direction last dimension must be {self.channels}")

        delta_generic = self.generic_bound * torch.tanh(self.generic_head(generic_features))
        delta_prior = self.prior_bound * torch.tanh(prior_direction)
        rho = torch.sigmoid(self.rho_head(evidence))
        raw_prediction = base + delta_generic + self.alpha * rho * delta_prior

        if observed is not None and reliable_mask is not None:
            prediction = preserve_reliable_observations(raw_prediction, observed, reliable_mask)
        else:
            prediction = raw_prediction

        return RePACOutput(
            prediction=prediction,
            raw_prediction=raw_prediction,
            delta_generic=delta_generic,
            delta_prior=delta_prior,
            rho=rho,
            alpha=self.alpha,
        )
