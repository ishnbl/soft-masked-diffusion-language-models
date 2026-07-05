# From-scratch soft-masked MDLM training

Trains `MDLM_SM` from a **freshly initialized** DiT backbone (random
weights) on the same architecture/data as the released `mdlm_owt`
checkpoint (`model=small`, `data=openwebtext-split`) — no
`training.finetune_path` is set. This is the from-scratch counterpart to
`../../run_verda.sh` / `../../run_verda_topk.sh`, which finetune from that
released checkpoint instead.

Two scripts, kept deliberately separate (same convention as the rest of the
repo — see `run_verda_topk.sh`'s header comment):

| Script | `transparency_alg` | Geometry |
|---|---|---|
| `run_scratch_slerp.sh` | `slerp_sm` | SLERP — great-circle blend in embedding space |
| `run_scratch_topk.sh` | `mixinputs_with_topk` | LERP — convex blend in vocab-probability space |

Both share an identical `--fixed-lambda` flag, so all four combinations are
one flag away:

```bash
# SLERP, learnt lambda
bash scripts/from_scratch/run_scratch_slerp.sh --num-gpus 1

# SLERP, fixed lambda = 0.5
bash scripts/from_scratch/run_scratch_slerp.sh --num-gpus 1 --fixed-lambda 0.5

# LERP/top-k, learnt lambda
bash scripts/from_scratch/run_scratch_topk.sh --num-gpus 1

# LERP/top-k, fixed lambda = 0.5
bash scripts/from_scratch/run_scratch_topk.sh --num-gpus 1 --fixed-lambda 0.5
```

`--fixed-lambda` must lie in `[0, 1]`; both scripts validate this and only
append `algo.tran_head.fixed_lambda=...` to the Hydra command when set,
otherwise the learned entropy-gated head runs as-is.

## Cold-start caveat (learnt lambda only)

`../../experiments/SLERP_FINETUNING.md` documents that from-scratch training
hits a cold-start trap: a random backbone emits near-uniform Pass-1 logits,
so the fed-back predictions are pure noise and the learnable `scale`
parameter tends to get driven toward 0, collapsing `lambda` to ~0 (i.e. both
`slerp_sm` and `mixinputs_with_topk` degenerate to plain MDLM). That's why
the finetune scripts (`10_slerp_vs_topk_finetune_v2.sh` etc.) start from the
released checkpoint instead — a confident backbone from step 0 keeps
`lambda` non-trivial.

These scripts use `init_scale=0.5, init_centre=-2.5` (the validated fix from
`scripts/07_slerp_vs_topk_fixed.sh`), which keeps `lambda` small while the
backbone is still near-uniform and lets it grow as the backbone's
predictions become more confident over training — it mitigates the collapse
but doesn't eliminate it. Watch `transparency/lambda_mean` and
`transparency/scale` in W&B; if `lambda_mean` flattens near 0, either:

- use `--fixed-lambda` (sidesteps the learned-scale collapse entirely), or
- raise `--sm-prob-warmup-steps` (default `2000`, see below) so the backbone
  gets longer to become confident before soft-masking ever engages, or
- let training run longer — the trap resolves once the backbone's own
  Pass-1 predictions become confident enough to push `scale` back up, or
- start from the finetune scripts (`../10_slerp_vs_topk_finetune_v2.sh`)
  instead, if you actually want a meaningful learnt-lambda A/B rather than a
  from-scratch training run.

## Other notes

- `optim.lr=3e-4` (config default for training from scratch — 10x higher
  than the `3e-5` used by the finetune scripts, which start from an
  already-trained backbone).
- `lr_scheduler.num_warmup_steps=2500` by default (the project config
  default for a fresh backbone), vs. `200` in the finetune scripts.
- `--sm-prob` (default `0.8`) exposes `optim.sm_prob` — the TARGET fraction of
  training steps that run the two-pass soft-mask forward (mask -> Pass 1 ->
  transparency head -> Pass 2), once ramped up (see below). The remaining
  steps run a single plain-MDLM forward, which keeps the backbone honest.
  Raise it toward `1.0` to weight training more heavily toward the
  soft-masked objective.
- `--sm-prob-warmup-steps` (default `2000`) makes `sm_prob` ramp **linearly**
  from `0` at `global_step=0` up to `--sm-prob` at this step, then hold
  constant — implemented in `algo.py: MDLM_SM._current_sm_prob()` /
  `optim.sm_prob_warmup_steps` (project-wide config default `0`, i.e. no
  ramp, so this only changes behavior where it's explicitly set — these two
  scripts). This applies identically whether lambda is fixed or learnt: it
  only decides whether the two-pass soft-mask forward runs *at all* this
  step, before the transparency head's fixed/learnt lambda logic ever runs.
  It gives from-scratch training a real warm-up period where the backbone
  trains as plain MDLM with zero soft-mask interference, which directly
  helps the cold-start trap below (the noisiest, least-confident phase of
  training never touches the transparency head at all). Set to `0` to
  disable the ramp and use `--sm-prob` from step 0, as before. The realized
  value is logged each step as `transparency/sm_prob` in W&B.
- `checkpointing.resume_from_ckpt=false` is forced so a stale
  `checkpoints/best.ckpt` sitting in the output dir from a previous run can
  never silently resume instead of starting from a fresh random init.
- No `--base-ckpt` flag — there is nothing to point it at; the backbone is
  always randomly initialized.
- Preflight checks (CUDA availability, GPU count, batch-size divisibility)
  and the DUO `setup.sh` auto-run are identical to `run_verda.sh`.
- Default `--max-steps 5000` is enough to see whether lambda activates and
  to smoke-test the setup, but from-scratch convergence to something
  resembling coherent language modeling needs far more steps (script
  `02_sm_pretraining_scratch_owt.sh` uses `1_000_000`) — bump `--max-steps`
  accordingly for a real training run.

## Running via SLURM

`slurm_scratch_train.sbatch` wraps both launchers the same way
`../../verda/slurm_train.sbatch` wraps `run_slerp.sh`/`run_topk.sh`, minus
`BASE_CKPT` and plus `FIXED_LAMBDA`/`SM_PROB`. It requires the conda env from
`../../verda/setup_cluster.sh` to already exist (run that once, on a node
with internet, before submitting).

Edit the `#SBATCH` lines marked `TODO` for your cluster (partition, account,
GPU count via `--gres=gpu:N`, walltime), then:

```bash
# SLERP, learnt lambda (defaults)
sbatch scripts/from_scratch/slurm_scratch_train.sbatch

# top-k/LERP, learnt lambda
sbatch --export=ALL,ALG=topk scripts/from_scratch/slurm_scratch_train.sbatch

# SLERP, fixed lambda = 0.5
sbatch --export=ALL,ALG=slerp,FIXED_LAMBDA=0.5 scripts/from_scratch/slurm_scratch_train.sbatch

# top-k/LERP, fixed lambda = 0.5, custom seed and sm_prob
sbatch --export=ALL,ALG=topk,FIXED_LAMBDA=0.5,SEED=2,SM_PROB=0.5 \
  scripts/from_scratch/slurm_scratch_train.sbatch
```

Other overridable vars (same `--export=ALL,VAR=...` mechanism): `SEED`,
`MAX_STEPS`, `SM_PROB_WARMUP_STEPS` (default `2000`), `MODEL`, `DATA`,
`GLOBAL_BATCH_SIZE`, `BATCH_SIZE`, `DATA_DIR`, `OUT_DIR`, `ENV_NAME`,
`CONDA_HOME`, `WANDB_MODE`.

Notes specific to the SLURM path:

- `NUM_GPUS` is **not** a knob — the job script derives it from
  `CUDA_VISIBLE_DEVICES` (which SLURM sets from `--gres=gpu:N`) at runtime
  and passes it through as `--num-gpus`. Change GPU count by editing
  `--gres` (and keep `BATCH_SIZE`/`GLOBAL_BATCH_SIZE` divisible by
  `NUM_GPUS * BATCH_SIZE`, same constraint as running the launcher directly).
- `DATA_DIR` and `OUT_DIR` default to `$HOME/sm_data` and `$HOME/sm_outputs`
  — put them on **shared, persistent** storage visible from the compute
  node, not local scratch that disappears with the job.
- `WANDB_MODE=offline` by default (compute nodes are frequently air-gapped);
  sync afterwards from a node with internet:
  `wandb sync $OUT_DIR/*/wandb/offline-run-*`.
- If you're submitting more than one of these jobs against a **shared, still
  -empty** `DATA_DIR`, read the "Data / tokenization caching" section below
  first — the first job to touch it does a slow, unsharded download +
  tokenization pass, and two cold jobs racing on the same empty cache can
  step on each other.

## Data / tokenization caching

`--data-dir` (default `/workspace/data`; the SLURM wrapper defaults it to
`$HOME/sm_data`) is used as `data.cache_dir=${DATA_DIR}/owt_cache` — that one
directory serves two purposes for `data=openwebtext-split`, both handled
internally by `dataloader.py`'s `get_dataset()` (fetched by `setup.sh` from
the DUO repo and patched by `patches/dataloader.patch`):

1. **HF `datasets` download/build cache** — the raw OpenWebText dataset is
   pulled via `datasets.load_dataset('openwebtext', ..., cache_dir=...)` and
   cached here in HF's own on-disk format.
2. **Final tokenized + chunked cache** — after GPT-2 tokenization and
   grouping into fixed-length blocks, the result is saved directly under
   `owt_cache/` via `datasets.save_to_disk()` as:
   - `openwebtext-train_train_bs<model.length>_wrapped.dat`
   - `openwebtext-valid_validation_bs<model.length>_wrapped.dat`

   For `model=small` (`length=1024`, the default here), that's
   `openwebtext-train_train_bs1024_wrapped.dat` and
   `openwebtext-valid_validation_bs1024_wrapped.dat`. Every run first checks
   whether that exact path already exists (`utils.fsspec_exists`) and, if
   so, loads it directly with `datasets.load_from_disk()` — **no download or
   re-tokenization**.

Consequences worth knowing before you run these scripts:

- **The cache key is `(dataset_name, split, block_size, wrap, eos_tag)` —
  not algorithm, lambda mode, or seed.** So the first of your four combos
  you run against a given `--data-dir` pays the full OpenWebText
  download + tokenization cost; the other three (and any finetune run via
  `../../run_verda.sh` with the same `--model small --data
  openwebtext-split`) reuse the same cached `.dat` file instantly. You do
  not need to pre-tokenize per-script or per-lambda-mode.
- **First-run cost is real**: OpenWebText is tens of GB raw, plus tokenizer
  overhead (`num_proc` parallel `.map()` passes for tokenizing and grouping)
  and the final arrow cache on top of that — budget disk space accordingly
  and expect the first run to spend significant wall-clock time on data prep
  before training starts (this is HF `datasets` work, not GPU work, and adds
  up on top of the normal training time from `--max-steps`).
- **Do not point two cold jobs at the same empty `--data-dir` at once** —
  `get_dataset()` has no cross-process lock around the "does `_path` exist"
  check + `save_to_disk()`, so two simultaneous first-time runs can both
  decide the cache is missing and duplicate the download/tokenize work (or
  worse, race writing the same `_path`). Warm the cache with one short run
  first (e.g. `--max-steps 1`) and let it finish, then submit the rest of
  the combos — they'll hit the fast path.
- To point at a dataset cache tokenized on another machine, just copy the
  whole `owt_cache/` directory (both the HF cache and the two `.dat`
  directories) to the same relative layout on the new machine and pass the
  matching `--data-dir`.
