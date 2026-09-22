# GNS PyTorch

Gradient Noise Scale (GNS) for PyTorch models, from per-example losses. No hooks, no second batch size, no multi-GPU setup needed.

## What's GNS?

The simple noise scale of McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162), is

    B_simple = tr(Σ) / |G|²

where `G` is the true gradient and `Σ` the per-example gradient covariance. It predicts the critical batch size: below it, doubling the batch nearly halves the steps needed; above it, extra examples are wasted. See also <https://openreview.net/forum?id=xINTMAvPQA>.

## Install

```bash
pip install gns-pytorch
```

## Usage

```python
from gns_pytorch import GnsEma, gns_per_example

model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
gns = GnsEma(decay=0.99)

def training_step(batch, global_step):
    x, y = batch
    logits = model(x)
    per_example_losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")

    if global_step % 100 == 0:
        gns.update(gns_per_example(per_example_losses, model))  # before loss.backward()
        print(f"GNS: {gns.value:.1f}")

    loss = per_example_losses.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`gns_per_example` must be called **before `loss.backward()`**: it differentiates the same graph (with `retain_graph=True`), so the activations are needed once more.

## API

- `GnsStats(tr_sigma, g2, b_small, b_big)` — one measurement: unbiased estimates of `tr(Σ)` and `|G|²`, and the two batch sizes they came from. `.b_simple` is the single-batch point estimate; do not average it (see below).
- `gns_per_example(loss_per_example, params_or_model, *, use_vmap=False) -> GnsStats` — the per-example estimator (`b_small = 1`, `b_big = n`, or `n * world_size` under DDP). Uses the whole batch and every parameter with `requires_grad`, in a deterministic order; pass an explicit list of parameters to measure a subset. `use_vmap=True` replaces the `n` backward passes by one vmapped pass (not compatible with flex attention or `torch.compile`). Squared norms are accumulated in fp32 whatever the gradient dtype.
- `gns_from_microbatch_grads(microbatch_grads, b_micro) -> GnsStats` — the two-batch-size estimator (`b_small = b_micro`, `b_big = C * b_micro`) from the `C` microbatch gradients of one optimizer step. Free when you already accumulate gradients: snapshot `p.grad` after each microbatch's backward.
- `stats_from_sqnorms(sq_small, sq_big, b_small, b_big) -> GnsStats` — the paper's unbiased formulas for any pair of batch sizes, given `E|G_{b_small}|²` and `E|G_{b_big}|²`.
- `GnsEma(decay)` — keeps an EMA of `tr(Σ)` and of `|G|²` separately; `.value` divides them.
- `compute_gns(loss_per_example, model, use_vmap=False) -> float` — compatibility wrapper returning the single-batch point estimate. Prefer the above.

## Average the numerator and the denominator, not the ratio

Both `tr_sigma` and `g2` are unbiased, but each single-batch `|G|²` estimate is very noisy — its relative standard deviation is roughly `(B_simple / n) * sqrt(2 / d_eff)` for a batch of `n` — and often negative when the batch is far below the noise scale. The ratio of two such numbers is biased and heavy-tailed, so an EMA of per-batch GNS values converges to the wrong number (50–70 % off in the synthetic test below, or clamped towards zero if negatives are dropped). Accumulate `tr_sigma` and `g2` separately (`GnsEma`, or plain sums over a window of consecutive steps) and divide once. Use a slow decay (0.99 or slower); if `g2` is still negative, the window is too short for how far the batch is below the noise scale.

## Requirements on the per-example losses

`loss_per_example[i]` must depend only on example `i`. Anything that couples examples inside the batch — BatchNorm in training mode, contrastive or batch-coupled losses, a critic with batch statistics — silently invalidates the estimator. Under DDP, call it on every rank with the same parameter list; the reduction is done for you.

## Tips

- Call it every N steps (100+) to keep the overhead negligible; at a coarse cadence, measure on several consecutive steps and sum the stats.
- GNS approximates the critical batch size: if it reads 64 and your global batch is 32, doubling the gradient accumulation steps is roughly free in samples.

## Synthetic check

`tests/test_gns.py` uses a linear model `y = w*·x + ε` where `tr(Σ)` and `|G|²` are known in closed form, and asserts that the ratio of window means matches the true GNS within 10 % for the per-example, vmap and microbatch paths.
