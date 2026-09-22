"""Gradient noise scale (McCandlish et al. 2018): unbiased tr(Sigma) and |G|^2 returned separately,
full batch by default, deterministic parameter set, fp32 accumulation,
correct DDP reduction, and helpers for the two-batch-size (microbatch) estimator."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence

import torch


class GnsStats(NamedTuple):
    tr_sigma: float  # unbiased estimate of tr(Sigma)  (per-example gradient covariance)
    g2: float  # unbiased estimate of |G|^2 (true gradient squared norm)
    b_small: int
    b_big: int

    @property
    def b_simple(self) -> float:
        """Point estimate. Prefer GnsEma: this ratio of two noisy numbers is biased and can be negative."""
        return self.tr_sigma / self.g2 if self.g2 > 0 else float("inf")


def stats_from_sqnorms(sq_small: float, sq_big: float, b_small: int, b_big: int) -> GnsStats:
    """McCandlish et al. 2018 App. A: sq_small = E|G_{b_small}|^2, sq_big = E|G_{b_big}|^2."""
    assert b_big > b_small >= 1
    tr_sigma = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
    g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
    return GnsStats(float(tr_sigma), float(g2), b_small, b_big)


def _per_example_grads(
    loss_per_example: torch.Tensor, params: Sequence[torch.Tensor], use_vmap: bool
) -> list[torch.Tensor]:
    n = loss_per_example.size(0)
    if use_vmap:
        eye = torch.eye(n, device=loss_per_example.device, dtype=loss_per_example.dtype)

        def grads_for_vec(v: torch.Tensor) -> list[torch.Tensor]:
            g = torch.autograd.grad(loss_per_example, params, v, retain_graph=True, allow_unused=True)
            return [x for x in g if x is not None]

        return [g.detach() for g in torch.vmap(grads_for_vec)(eye)]
    per_example = []
    for i in range(n):
        g = torch.autograd.grad(loss_per_example[i], params, retain_graph=True, allow_unused=True)
        per_example.append([x.detach() for x in g if x is not None])
    return [torch.stack(gs, dim=0) for gs in zip(*per_example)]


def gns_per_example(
    loss_per_example: torch.Tensor,
    params: Iterable[torch.Tensor] | torch.nn.Module,
    *,
    use_vmap: bool = False,
) -> GnsStats:
    """Per-example estimator (b_small = 1, b_big = n, or n * world_size under DDP).

    `loss_per_example` must be the per-example loss of this rank's batch, shape [n], n >= 2,
    with each entry depending only on its own example (no BatchNorm-style coupling).
    Pass the same `params` on every rank.
    """
    assert loss_per_example.ndim == 1 and loss_per_example.size(0) >= 2
    if isinstance(params, torch.nn.Module):
        params = params.parameters()
    params = [p for p in params if p.requires_grad]
    grads = _per_example_grads(loss_per_example, params, use_vmap)
    n = loss_per_example.size(0)
    with torch.no_grad():
        sq_small = loss_per_example.new_zeros((), dtype=torch.float32)
        sq_big = loss_per_example.new_zeros((), dtype=torch.float32)
        means = []
        for g in grads:
            g = g.float()
            sq_small += g.pow(2).sum() / n  # mean over examples of |g_i|^2
            means.append(g.mean(dim=0))
        b_big = n
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            torch.distributed.all_reduce(sq_small, op=torch.distributed.ReduceOp.AVG)
            flat = torch.cat([m.reshape(-1) for m in means])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.AVG)
            sq_big = flat.pow(2).sum()
            b_big = n * world
        else:
            for m in means:
                sq_big += m.pow(2).sum()
    return stats_from_sqnorms(sq_small.item(), sq_big.item(), 1, b_big)


def gns_from_microbatch_grads(
    microbatch_grads: Sequence[Sequence[torch.Tensor]], b_micro: int
) -> GnsStats:
    """Two-batch-size estimator from the C microbatch gradients of one optimizer step
    (each a list of per-parameter grads, all of batch size b_micro): b_small = b_micro,
    b_big = C * b_micro. Free when already accumulating gradients."""
    c = len(microbatch_grads)
    assert c >= 2
    with torch.no_grad():
        sq_small = 0.0
        acc = [torch.zeros_like(g, dtype=torch.float32) for g in microbatch_grads[0]]
        for grads in microbatch_grads:
            for a, g in zip(acc, grads):
                g = g.float()
                sq_small += g.pow(2).sum().item() / c
                a += g / c
        sq_big = sum(a.pow(2).sum().item() for a in acc)
    return stats_from_sqnorms(sq_small, sq_big, b_micro, c * b_micro)


class GnsEma:
    """EMA of numerator and denominator separately (the paper's recommendation), ratio at read time."""

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay
        self.tr_sigma: float | None = None
        self.g2: float | None = None

    def update(self, s: GnsStats) -> None:
        if self.tr_sigma is None or self.g2 is None:
            self.tr_sigma, self.g2 = s.tr_sigma, s.g2
        else:
            self.tr_sigma = self.decay * self.tr_sigma + (1 - self.decay) * s.tr_sigma
            self.g2 = self.decay * self.g2 + (1 - self.decay) * s.g2

    @property
    def value(self) -> float:
        if self.g2 is None or self.tr_sigma is None or self.g2 <= 0:
            return float("nan")
        return self.tr_sigma / self.g2


def compute_gns(loss_per_example: torch.Tensor, model: torch.nn.Module, use_vmap: bool = False) -> float:
    """Backwards-compatible point estimate. Prefer gns_per_example + GnsEma."""
    return gns_per_example(loss_per_example, model, use_vmap=use_vmap).b_simple
