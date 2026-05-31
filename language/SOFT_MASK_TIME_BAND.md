# Soft-Masking Time Band (`t ∈ [0.2, 0.8]`)

## Background

In the original soft-masking paper, the soft-mask feedback mechanism is **only
applied for diffusion timesteps in a middle band** — by default `t ∈ [0.2, 0.8]`
— and a plain forward pass is used elsewhere. The motivation:

- Near `t = 0` very little is masked, so the input is already nearly clean and
  the feedback signal adds noise/compute for almost no benefit.
- Near `t = 1` almost everything is masked, so the previous-step prediction fed
  back as "soft-mask" content is essentially uninformative.

Before these changes, this codebase applied soft-masking **across the entire
timestep range**. Whether the two-pass soft-mask path ran was gated only by a
Bernoulli probability `optim.sm_prob` (training) and by the presence of a cached
prediction (sampling) — neither of which depended on `t`.

This change adds the `t`-band restriction while preserving the existing
`sm_prob` Bernoulli gate. Setting the band to `[0.0, 1.0]` exactly recovers the
old behavior.

## New config knobs

Added to the `optim:` block in
[`configs/config.yaml`](configs/config.yaml):

```yaml
optim:
  ...
  sm_prob: 0.5
  # Soft-masking is only applied for timesteps inside [sm_t_min, sm_t_max].
  sm_t_min: 0.2   # below this, use a standard forward pass
  sm_t_max: 0.8   # above this, use a standard forward pass
```

To disable the band (old behavior — soft-mask at all timesteps):

```bash
optim.sm_t_min=0.0 optim.sm_t_max=1.0
```

`run_modal.py` now passes `optim.sm_t_min=0.2` / `optim.sm_t_max=0.8` explicitly
so Modal runs use the paper defaults regardless of any future config edits.

## Code changes

All changes are in [`algo.py`](algo.py), the `Diffusion`/soft-mask subclass,
plus the config and Modal launcher.

### 1. Training: `nll(...)`

The base repo gates soft-masking with a single whole-batch Bernoulli coin flip
(`sm_prob`), applied at every timestep. We keep that exact structure and simply
**add the time-band condition to the same single decision** — i.e. a *batch-level*
gate, not a per-sample one.

`t` is sampled per-example (shape `[B]`), so the band is evaluated on the batch's
representative timestep `t.mean()`. The soft-mask path is taken only when the
Bernoulli gate fires **and** the batch's mean `t` lies in `[sm_t_min, sm_t_max]`:

| Case | Behavior |
|------|----------|
| Gate off, or batch out of band | Standard single forward. |
| Gate on **and** batch in band | Original two-pass soft-mask (detached pass-1 feedback → gradient-carrying pass-2). |

This is single-path and has exactly the same memory/compute profile as the base
repo's soft-mask step — there is no batch splitting, no `torch.where`, and no
extra retained activation graph (an earlier per-sample version that combined two
gradient-carrying full-batch forwards OOMed on a 40 GB A100, which is why the
batch-level gate is used).

```python
sm_gate = (not train_mode) or (torch.rand(1).item() < self.config.optim.sm_prob)
in_band = (self.config.optim.sm_t_min <= t.mean().item()
           <= self.config.optim.sm_t_max)
use_soft_mask = sm_gate and in_band

if use_soft_mask:
    log_x_theta_pass1 = self.forward(xt, sigma=sigma).detach()
    log_x_theta = self.forward(xt, sigma=sigma, log_p_x0=log_x_theta_pass1)
else:
    log_x_theta = self.forward(xt, sigma=sigma)
```

> Trade-off: because the decision is whole-batch, a batch is either fully
> soft-masked or not. Over training, batches are soft-masked roughly in
> proportion to how often their mean `t` falls in-band, which matches how the
> base repo already makes one soft-mask decision per batch. The sampling-side
> gates below remain per-timestep (at inference all positions share one scalar
> `t`, so there is no ambiguity there).

### 2. Sampling: `_ddpm_caching_update(...)`

At inference all positions share one scalar timestep. The cached previous
prediction (`log_p_x0_sm`) is now fed back **only when that timestep is in band**;
otherwise a standard forward is used:

```python
time_val = t.reshape(-1)[0].item()
in_band = (self.config.optim.sm_t_min <= time_val <= self.config.optim.sm_t_max)
feedback = log_p_x0_sm if in_band else None
log_p_x0_sm = self.forward(x, sigma_t, feedback)
```

### 3. Sampling: final noise-removal pass in `generate_samples(...)`

The noise-removal step runs at `min_t ≈ eps` (effectively `t ≈ 0`), which is
below `sm_t_min`. It is now gated the same way, so by default it performs a
standard forward instead of a soft-mask feedback pass:

```python
nr_time = t.reshape(-1)[0].item()
nr_in_band = (self.config.optim.sm_t_min <= nr_time <= self.config.optim.sm_t_max)
final_feedback = log_p_x0_cache_sm if nr_in_band else None
final_log_p_x0 = self.forward(x, unet_conditioning, log_p_x0=final_feedback)
```

## Paths intentionally **not** changed

- `_analytic_update` / `_denoiser_update` (the analytic samplers in
  `trainer_base.py`) call the model through `_get_score(...)` without a
  `log_p_x0` argument, so they never invoke soft-masking. No gating is needed.

## Verifying / tuning

- **Disable the band** (reproduce pre-change behavior):
  `optim.sm_t_min=0.0 optim.sm_t_max=1.0`.
- **Match a different paper band**: override `optim.sm_t_min` / `optim.sm_t_max`.
- The existing `transparency/lambda_*` W&B logs let you confirm that the
  realized soft-mask interpolation only fires for in-band steps.

## Summary of files touched

| File | Change |
|------|--------|
| `configs/config.yaml` | Added `optim.sm_t_min` / `optim.sm_t_max` (defaults `0.2` / `0.8`). |
| `algo.py` | Time-band gating in `nll(...)` (training), `_ddpm_caching_update(...)` and the noise-removal pass in `generate_samples(...)` (sampling). |
| `run_modal.py` | Pass `optim.sm_t_min=0.2` / `optim.sm_t_max=0.8` to Modal runs. |
