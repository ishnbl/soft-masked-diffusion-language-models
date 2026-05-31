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

The single batch-level Bernoulli gate was replaced with a Bernoulli gate **plus**
a per-sample time-band gate. `t` is sampled per-example (shape `[B]`), so a batch
can contain a mix of in-band and out-of-band timesteps. Three cases:

| Case | Behavior | Forward passes |
|------|----------|----------------|
| Gate off, or **no** sample in band | Standard forward only | 1 |
| Gate on, **all** samples in band | Original two-pass soft-mask (detached pass-1 feedback → pass-2) | 2 |
| Gate on, **mixed** batch | Standard pass (gradient-carrying) for out-of-band samples; soft-mask pass for in-band samples, selected per-sample with `torch.where`. The standard pass doubles as the detached feedback for the soft-mask pass. | 2 |

Key points:

- The mixed case stays at **two** forward passes (no extra cost vs. the original)
  because the gradient-carrying standard pass is reused (detached) as the
  feedback input for the soft-mask pass.
- Gradients still flow only through the loss-producing output for each sample:
  pass-2 for in-band samples, the standard pass for out-of-band samples. The
  soft-mask feedback input remains detached, exactly as before.

```python
sm_gate = (not train_mode) or (torch.rand(1).item() < self.config.optim.sm_prob)
in_band = (t >= self.config.optim.sm_t_min) & (t <= self.config.optim.sm_t_max)
use_soft_mask = sm_gate and bool(in_band.any())

if use_soft_mask and bool(in_band.all()):
    log_x_theta_pass1 = self.forward(xt, sigma=sigma).detach()
    log_x_theta = self.forward(xt, sigma=sigma, log_p_x0=log_x_theta_pass1)
elif use_soft_mask:
    log_x_theta_std = self.forward(xt, sigma=sigma)
    log_x_theta_sm = self.forward(xt, sigma=sigma, log_p_x0=log_x_theta_std.detach())
    sel = in_band.view(-1, *([1] * (log_x_theta_sm.ndim - 1)))
    log_x_theta = torch.where(sel, log_x_theta_sm, log_x_theta_std)
else:
    log_x_theta = self.forward(xt, sigma=sigma)
```

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
