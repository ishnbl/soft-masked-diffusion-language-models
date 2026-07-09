# Reliability-Conditioned SLERP Soft-Masking for Scratch Training

## Summary

This proposal adapts soft-masked diffusion language model training for a
from-scratch SLERP soft-masking setup. The goal is to preserve the paper's
central idea, namely feeding model predictions back into masked positions, while
changing the interpolation geometry from linear interpolation to spherical
interpolation in embedding space.

The key training issue is that early from-scratch predictions are unreliable. A
randomly initialized backbone produces high-entropy or incorrect Pass-1 logits,
so using those predictions as feedback can inject noise into the input. A
schedule should therefore be tied to feedback reliability, not only to training
step.

This differs from a simple linear warmup. A step-based ramp assumes feedback
becomes useful because enough steps have passed. A reliability-conditioned ramp
turns on feedback when the model's own denoising predictions become informative.

The soft-masking paper supports probabilistic SM exposure and learned
confidence-based lambda. It does not specify a full from-scratch schedule for
SLERP feedback. This proposal is an adaptation for our spherical-interpolation
variant.

References:

- Soft-Masked Diffusion Language Models: https://arxiv.org/abs/2510.17206
- Project repository: https://github.com/IBM/soft-masked-diffusion-language-models

## Terms

`x0` is the original clean token sequence.

`xt` is the corrupted diffusion input at time `t`, where some tokens have been
replaced by the mask token.

`p_theta(x0 | xt)` is the model probability assigned to the correct clean token
at a masked position.

`R_batch` is the current batch feedback-reliability score:

```text
R_batch = mean p_theta(x0 | xt)
```

The mean is computed over masked positions only. A low value means feedback is
mostly unreliable. A high value means the model is assigning meaningful
probability to the true token. Equivalently, this can be viewed as:

```text
R_batch ~= exp(-masked_nll)
```

`R_ema` is an exponential moving average of `R_batch`:

```text
R_ema[t] = beta * R_ema[t - 1] + (1 - beta) * R_batch[t]
```

`beta` is the EMA smoothing factor. Values such as `0.99` or `0.995` make the
schedule stable and prevent it from reacting too strongly to a single batch.

`R_start` is the reliability value where soft feedback begins turning on. Below
this threshold, feedback is treated as too noisy.

`R_full` is the reliability value where soft feedback reaches its full scheduled
strength.

`u` is normalized reliability progress:

```text
u = clamp((R_ema - R_start) / (R_full - R_start), 0, 1)
```

`u = 0` means feedback is not reliable enough yet. `u = 1` means feedback is
reliable enough for normal soft-masked training.

`r` is the final schedule multiplier. The proposed default is smoothstep:

```text
r = u^2 * (3 - 2u)
```

This is the lowest-degree polynomial satisfying:

```text
r(0) = 0
r(1) = 1
r'(0) = 0
r'(1) = 0
```

The zero slopes make feedback turn on gently near `R_start` and saturate gently
near `R_full`. This matters because the main scratch-training danger is early
noisy feedback. A linear schedule, `r = u`, is also valid and simpler, but it
starts increasing at full slope as soon as `R_ema` crosses `R_start`.

`sm_prob` is the probability that a training step uses the two-pass soft-masking
path instead of the standard single-pass binary-mask path.

`p_max` is the maximum soft-masking probability. It should remain below `1.0` so
the model continues to see ordinary binary-mask inputs.

`lambda` is the strength of soft feedback at masked positions. In the SLERP
variant, lambda controls how far the input embedding moves along the sphere from
the normalized mask embedding toward the spherical mean of predicted token
embeddings.

`lambda = 0` means no feedback; the position stays at the mask embedding.

`lambda = 1` means full movement to the predicted feedback direction.

`lambda_learned` is the learned confidence-based lambda:

```text
lambda_learned = scale * sigmoid(steepness * (neg_entropy - centre))
```

`neg_entropy` is the negative entropy of the prediction distribution. It is near
`0` for confident predictions and very negative for near-uniform predictions.

`scale` is a learned upper bound on lambda.

`steepness` controls how sharply lambda turns on as confidence improves.

`centre` controls the confidence threshold where lambda begins activating.

## SLERP Soft-Masking Adaptation

Normal soft masking linearly blends feedback into masked positions. Our variant
keeps the same two-pass training concept but changes the geometry:

```text
mask embedding -> SLERP -> spherical mean of top-k predicted token embeddings
```

For masked positions:

1. Run Pass 1 and obtain logits.
2. Select the top-k predicted tokens.
3. Normalize their embeddings onto the unit sphere.
4. Compute a probability-weighted Frechet/Karcher mean on the sphere.
5. Normalize the mask-token embedding.
6. Use SLERP from the mask embedding to the spherical mean with strength
   `lambda`.

Unmasked positions keep their normal token embeddings.

The schedule should not decide whether SLERP is mathematically valid. SLERP is
the geometry. The schedule decides when the model's own predictions are reliable
enough to use as feedback.

## Learned-Lambda Schedule

The learned-lambda setup already has an internal confidence gate:

```text
lambda_learned = scale * sigmoid(steepness * (neg_entropy - centre))
```

When entropy is high, `lambda_learned` becomes small. This suppresses noisy
feedback at the token level.

However, the training dynamics can still fail. If early noisy soft-masking steps
repeatedly hurt the loss, the transparency head can learn to suppress itself by
driving `scale` downward. Then, when the backbone later becomes useful, lambda
may remain too small for SLERP and linear SM to meaningfully differ.

Therefore, in learned-lambda training, schedule the exposure frequency, not the
lambda strength:

```text
sm_prob_eff = p_max * r
lambda_eff = lambda_learned
```

Behavior:

```text
early:
  R_ema is low
  sm_prob_eff ~= 0
  training is mostly vanilla MDLM
  the transparency head is protected from noisy feedback gradients

middle:
  R_ema rises
  sm_prob_eff increases smoothly
  the transparency head starts learning from more meaningful feedback

late:
  R_ema >= R_full
  sm_prob_eff -> p_max
  training behaves like normal learned-lambda SLERP-SM
```

This keeps the learned lambda mechanism intact while avoiding premature
overexposure to noisy self-feedback.

## Fixed-Lambda Schedule

Fixed lambda has no entropy-based protection. If lambda is constant from step
zero, early soft-masked inputs contain fixed-strength random feedback.

Therefore, fixed-lambda training should schedule both feedback strength and
feedback frequency:

```text
lambda_eff = lambda_target * r
sm_prob_eff = p_max * r
```

Behavior:

```text
early:
  R_ema is low
  lambda_eff ~= 0
  sm_prob_eff ~= 0
  fixed-lambda SM reduces to vanilla MDLM

middle:
  R_ema rises
  lambda_eff and sm_prob_eff increase smoothly

late:
  R_ema >= R_full
  lambda_eff -> lambda_target
  sm_prob_eff -> p_max
```

Scheduling only `sm_prob` is weaker for fixed lambda. Whenever SM is active,
the feedback would still be full-strength. Scheduling `lambda_eff` is the direct
way to prevent early random predictions from becoming strong input noise.

## Contradiction Checks

If learned lambda already goes to zero under high entropy, why schedule
`sm_prob`?

Because learned lambda protects individual inputs, but it does not fully protect
parameter dynamics. Repeated early noisy SM steps can teach the head that
feedback is harmful and push `scale` down before the backbone is competent.

If `sm_prob` is zero early, how does the transparency head learn?

It should not learn from random feedback early. The backbone first learns basic
denoising through vanilla MDLM. Once predictions become reliable, soft masking
turns on and the head learns from useful feedback.

If reliability is computed from Pass-1 logits, do we need an extra Pass 1 when
`sm_prob_eff` is zero?

No. Use the logits from the normal single forward already computed for the loss
as the reliability proxy. Do not add an extra forward only to update the
schedule.

Why use correctness-based reliability instead of entropy only?

Entropy measures confidence, but confident predictions can be wrong.
`mean p_theta(x0 | xt)` measures whether the model assigns probability to the
true clean token. During training, `x0` is available, so this is a better
feedback-reliability signal than entropy alone.

Why keep `p_max < 1`?

The model still needs exposure to binary-mask inputs. The soft-masking paper
reports that always using SM can hurt because the model must handle binary-mask
states during decoding and initial denoising.

## Recommended Defaults

For learned-lambda SLERP-SM from scratch:

```text
lambda_eff = lambda_learned
sm_prob_eff = p_max * smoothstep(R_ema)
p_max = 0.5 initially, then test 0.8
beta = 0.99 or 0.995
```

For fixed-lambda SLERP-SM from scratch:

```text
lambda_eff = lambda_target * smoothstep(R_ema)
sm_prob_eff = p_max * smoothstep(R_ema)
p_max = 0.5 initially, then test 0.8
beta = 0.99 or 0.995
```

Choose `R_start` and `R_full` empirically from a short vanilla MDLM warmup or a
pilot run. The schedule should begin when `R_ema` is clearly above random
chance, and should reach full strength only after the model has learned
nontrivial denoising.

## Final Position

The core adaptation is:

```text
reliability-conditioned feedback exposure for learned lambda
reliability-conditioned feedback exposure and strength for fixed lambda
```

This is more theoretically grounded than a purely step-based linear ramp because
it connects the schedule to the condition required for soft masking to help: the
model's feedback must contain information about the clean token.
