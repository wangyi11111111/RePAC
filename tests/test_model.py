import torch

from repac import RePAC, preserve_reliable_observations


def test_repac_shapes_and_bounds():
    torch.manual_seed(0)
    base = torch.zeros(4, 3, 2)
    features = torch.randn(4, 3, 5)
    evidence = torch.randn(4, 3, 7)
    prior = torch.randn(4, 3, 2)
    model = RePAC(2, 5, 7, hidden_dim=8, generic_bound=0.1, prior_bound=0.2)
    out = model(base, features, evidence, prior)
    assert out.prediction.shape == base.shape
    assert out.delta_generic.abs().max() <= 0.1001
    assert out.delta_prior.abs().max() <= 0.2001
    assert out.rho.shape == (4, 3, 1)
    assert out.alpha.item() > 0


def test_preserve_reliable_observations():
    raw = torch.zeros(2, 2, 1)
    observed = torch.ones(2, 2, 1)
    mask = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    anchored = preserve_reliable_observations(raw, observed, mask)
    assert torch.equal(anchored, mask)
