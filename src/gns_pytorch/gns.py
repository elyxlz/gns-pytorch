import gc
import math
import random

import torch


def compute_gns(
    loss_per_example: torch.Tensor,
    model: torch.nn.Module,
    param_percentage: float = 0.01,
    batch_percentage: float = 0.25,
    use_vmap: bool = False,
) -> float:
    assert loss_per_example.ndim == 1
    assert loss_per_example.size(0) > 1
    all_params = list(model.parameters())
    k = max(min(len(all_params), 10), int(len(all_params) * param_percentage))
    all_params = random.sample(all_params, k)
    new_batch_size = max(2, int(loss_per_example.size(0) * batch_percentage))
    loss_per_example = loss_per_example[:new_batch_size]
    bsz = loss_per_example.size(0)
    dev = loss_per_example.device

    g = None
    if use_vmap:
        eye = torch.eye(bsz, device=dev)

        def grads_for_vec(v):
            g = torch.autograd.grad(
                loss_per_example, all_params, v, retain_graph=True, allow_unused=True
            )
            return [
                x if x is not None else torch.zeros_like(p)
                for x, p in zip(g, all_params)
            ]

        batched_grads = torch.vmap(grads_for_vec)(eye)
    else:
        batched_grads = []
        for i in range(bsz):
            g = torch.autograd.grad(
                loss_per_example[i], all_params, retain_graph=True, allow_unused=True
            )
            batched_grads.append(
                [
                    x if x is not None else torch.zeros_like(p)
                    for x, p in zip(g, all_params)
                ]
            )

        per_param = list(zip(*batched_grads))
        batched_grads = [
            torch.stack(grads_for_param, dim=0) for grads_for_param in per_param
        ]

    with torch.no_grad():
        sqnorm_per_ex = torch.zeros(bsz, device=dev)
        sq_large = 0.0
        for g in batched_grads:
            sqnorm_per_ex += g.pow(2).reshape(bsz, -1).sum(dim=1)
            sq_large += g.mean(dim=0).pow(2).sum()

        sq_small = sqnorm_per_ex.mean()
        m, n = 1, bsz
        var = (sq_small - sq_large) / (1 / m - 1 / n)
        mean_sq = (n * sq_large - m * sq_small) / (n - m)
        val = var / mean_sq
        val = val.item()

        if math.isnan(val) or val < 0.0:
            val = 0.0

    del batched_grads, sqnorm_per_ex, g, sq_large, sq_small
    gc.collect()
    torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        t_val = torch.tensor(val, device=dev)
        torch.distributed.all_reduce(t_val, op=torch.distributed.ReduceOp.AVG)
        val = t_val.item()

    return val
