"""Run a minimal RePAC forward pass."""

from __future__ import annotations

import torch

from repac import RePAC, masked_mae
from repac.evidence import build_evidence
from repac.priors import combine_prior_directions, relation_direction, temporal_direction


def main() -> None:
    torch.manual_seed(7)
    t, n, c = 12, 8, 2
    base = torch.randn(t, n, c)
    target = base + 0.1 * torch.randn(t, n, c)
    observed = target + 0.02 * torch.randn(t, n, c)
    mask = (torch.rand(t, n, c) > 0.4).float()
    omega = 1.0 - mask
    relation = torch.rand(n, n)
    relation.fill_diagonal_(0.0)

    relation_dir = relation_direction(base, relation)
    temporal_dir = temporal_direction(base)
    prior_dir = combine_prior_directions(relation_dir, temporal_dir, lambda_relation=0.7, lambda_temporal=0.3)
    evidence = build_evidence(mask, base, relation=relation)
    generic_features = torch.cat([base, mask], dim=-1)

    model = RePAC(
        channels=c,
        generic_dim=generic_features.shape[-1],
        evidence_dim=evidence.shape[-1],
        hidden_dim=32,
        generic_bound=0.2,
        prior_bound=0.2,
    )
    out = model(base, generic_features, evidence, prior_dir, observed=observed, reliable_mask=mask)
    print("prediction", tuple(out.prediction.shape))
    print("rho", tuple(out.rho.shape), "alpha", float(out.alpha.detach()))
    print("masked_mae", float(masked_mae(out.prediction, target, omega).detach()))


if __name__ == "__main__":
    main()
