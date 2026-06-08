"""Local reliability evidence helpers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def temporal_residual(x: Tensor) -> Tensor:
    """Absolute one-step temporal inconsistency, padded at the first step."""

    diff = torch.zeros_like(x)
    diff[1:] = (x[1:] - x[:-1]).abs()
    return diff


def relation_residual(x: Tensor, relation: Tensor) -> Tensor:
    """Residual to a relation-neighborhood estimate.

    ``x`` has shape ``[T, N, C]`` and ``relation`` has shape ``[N, N]``. Rows of
    ``relation`` are normalized internally.
    """

    weights = relation / relation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    neighbor = torch.einsum("ij,tjc->tic", weights, x)
    return (x - neighbor).abs()


def local_gap_statistics(mask: Tensor, window: int = 3) -> Tensor:
    """Local sparsity cue from a centered temporal window."""

    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    pad = window // 2
    # Average over a local temporal window. Missingness is 1 - mask.
    missing = 1.0 - mask.float()
    padded = torch.nn.functional.pad(missing.permute(1, 2, 0), (pad, pad), mode="replicate")
    pooled = torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1)
    return pooled.permute(2, 0, 1)


def build_evidence(
    mask: Tensor,
    base: Tensor,
    relation: Optional[Tensor] = None,
    prior_residual: Optional[Tensor] = None,
    failure_score: Optional[Tensor] = None,
    conflict_score: Optional[Tensor] = None,
) -> Tensor:
    """Build a modular evidence vector.

    The returned tensor concatenates available cues along the channel dimension:
    mask pattern, local gap statistics, relation residual, temporal residual,
    optional prior residual score, failure-like structure, and conflict cue.
    """

    cues = [mask.float(), local_gap_statistics(mask), temporal_residual(base)]
    if relation is not None:
        cues.append(relation_residual(base, relation))
    if prior_residual is not None:
        cues.append(prior_residual.abs())
    if failure_score is not None:
        cues.append(failure_score)
    if conflict_score is not None:
        cues.append(conflict_score)
    return torch.cat(cues, dim=-1)
