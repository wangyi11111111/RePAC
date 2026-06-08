"""Prior-aligned candidate directions."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def relation_direction(base: Tensor, relation: Tensor) -> Tensor:
    """Direction from the base estimate toward a relation-neighborhood estimate."""

    weights = relation / relation.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    neighbor = torch.einsum("ij,tjc->tic", weights, base)
    return neighbor - base


def temporal_direction(base: Tensor) -> Tensor:
    """Direction toward the midpoint of adjacent temporal estimates."""

    prev_x = torch.cat([base[:1], base[:-1]], dim=0)
    next_x = torch.cat([base[1:], base[-1:]], dim=0)
    return 0.5 * (prev_x + next_x) - base


def combine_prior_directions(
    relation_dir: Optional[Tensor] = None,
    temporal_dir: Optional[Tensor] = None,
    structural_dir: Optional[Tensor] = None,
    lambda_relation: float = 1.0,
    lambda_temporal: float = 1.0,
    lambda_structural: float = 1.0,
) -> Tensor:
    """Combine modular candidate directions.

    At least one direction must be provided. Domain-specific structural
    residuals are optional; graph/correlation-neighborhood and temporal
    consistency cues form the default prior interface.
    """

    terms = []
    if relation_dir is not None:
        terms.append(lambda_relation * relation_dir)
    if temporal_dir is not None:
        terms.append(lambda_temporal * temporal_dir)
    if structural_dir is not None:
        terms.append(lambda_structural * structural_dir)
    if not terms:
        raise ValueError("at least one candidate direction must be provided")
    return torch.stack(terms, dim=0).sum(dim=0)
