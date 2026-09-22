"""Analytic tests: y = w*.x + eps, x ~ N(0, I_d), eps ~ N(0, s^2), loss_i = 0.5 (w.x_i - y_i)^2.

With delta = w - w*:  |G|^2 = |delta|^2,  tr Sigma = (d + 1) |delta|^2 + d s^2,
so the gradient noise scale B_simple = tr Sigma / |G|^2 is known exactly.
"""

import statistics

import pytest
import torch

from gns_pytorch import (
    GnsEma,
    GnsStats,
    compute_gns,
    gns_from_microbatch_grads,
    gns_per_example,
    stats_from_sqnorms,
)

D, DELTA, SIGMA, BATCH = 8, 1.0, 0.1, 16
TR_TRUE = (D + 1) * DELTA**2 + D * SIGMA**2
G2_TRUE = DELTA**2
B_TRUE = TR_TRUE / G2_TRUE  # 9.08


def make_problem(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w_star = torch.randn(D, generator=gen)
    direction = torch.randn(D, generator=gen)
    model = torch.nn.Linear(D, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_((w_star + DELTA * direction / direction.norm())[None])
    return model, w_star, gen


def per_example_losses(model, w_star, gen, n=BATCH, dtype=torch.float32):
    x = torch.randn(n, D, generator=gen)
    y = x @ w_star + SIGMA * torch.randn(n, generator=gen)
    pred = model(x.to(dtype)).squeeze(-1)
    return 0.5 * (pred - y.to(dtype)) ** 2


def ratio_of_means(stats: list[GnsStats]) -> float:
    return statistics.mean(s.tr_sigma for s in stats) / statistics.mean(s.g2 for s in stats)


def test_stats_from_sqnorms_matches_paper_formulas():
    # E|G_b|^2 = |G|^2 + tr Sigma / b, so plugging the expectations in must return the truth.
    s = stats_from_sqnorms(G2_TRUE + TR_TRUE / 8, G2_TRUE + TR_TRUE / 16, 8, 16)
    assert s.tr_sigma == pytest.approx(TR_TRUE)
    assert s.g2 == pytest.approx(G2_TRUE)
    assert s.b_simple == pytest.approx(B_TRUE)


@pytest.mark.parametrize("use_vmap", [False, True])
def test_per_example_estimator_is_unbiased(use_vmap):
    model, w_star, gen = make_problem()
    stats = [
        gns_per_example(per_example_losses(model, w_star, gen), model, use_vmap=use_vmap)
        for _ in range(300)
    ]
    assert all(s.b_small == 1 and s.b_big == BATCH for s in stats)
    assert ratio_of_means(stats) == pytest.approx(B_TRUE, rel=0.10)
    assert statistics.mean(s.tr_sigma for s in stats) == pytest.approx(TR_TRUE, rel=0.10)
    assert statistics.mean(s.g2 for s in stats) == pytest.approx(G2_TRUE, rel=0.10)


def test_microbatch_estimator_is_unbiased():
    model, w_star, gen = make_problem()
    params = list(model.parameters())
    stats = []
    for _ in range(300):
        losses = per_example_losses(model, w_star, gen)
        micro = [
            [g.detach() for g in torch.autograd.grad(losses[i : i + 4].mean(), params, retain_graph=True)]
            for i in range(0, BATCH, 4)
        ]
        stats.append(gns_from_microbatch_grads(micro, 4))
    assert all(s.b_small == 4 and s.b_big == BATCH for s in stats)
    assert ratio_of_means(stats) == pytest.approx(B_TRUE, rel=0.10)


def test_ema_of_separate_terms_converges():
    model, w_star, gen = make_problem()
    ema = GnsEma(0.99)
    for _ in range(300):
        ema.update(gns_per_example(per_example_losses(model, w_star, gen), model))
    assert ema.value == pytest.approx(B_TRUE, rel=0.15)


def test_all_parameters_are_used_deterministically():
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(6, 6) for _ in range(12)])  # 24 tensors
    x = torch.randn(BATCH, 6)
    losses = model(x).pow(2).mean(dim=1)
    first = gns_per_example(losses, model)
    again = gns_per_example(losses, model)
    explicit = gns_per_example(losses, list(model.parameters()))
    assert first == again == explicit
    subset = gns_per_example(losses, list(model.parameters())[:10])
    assert subset != first


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_grads_are_accumulated_in_fp32(dtype):
    model, w_star, gen = make_problem()
    state = gen.get_state()
    reference = gns_per_example(per_example_losses(model, w_star, gen, n=64), model)
    gen.set_state(state)
    low = gns_per_example(per_example_losses(model.to(dtype), w_star, gen, n=64, dtype=dtype), model)
    assert isinstance(low.tr_sigma, float) and isinstance(low.g2, float)
    assert low.tr_sigma == pytest.approx(reference.tr_sigma, rel=0.05)
    assert low.g2 == pytest.approx(reference.g2, rel=0.05)


def test_compute_gns_compatibility_wrapper():
    model, w_star, gen = make_problem()
    value = compute_gns(per_example_losses(model, w_star, gen, n=64), model)
    assert isinstance(value, float)
    assert value == pytest.approx(B_TRUE, rel=1.0)  # single-batch point estimate: only a sanity check
