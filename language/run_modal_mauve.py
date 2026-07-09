"""
Modal runner for MAUVE evaluation (sample_eval mode).

Mirrors `run_modal_eval.py`'s style (same image, volumes, secrets, app naming
convention, plain `@app.function` + `modal run`) but instead of perplexity
eval it runs the `mode=sample_eval` branch in main.py:
  1. Generate `num_sample_batches * loader.eval_batch_size` samples from a
     diffusion checkpoint using the configured sampler.
  2. Compute generative perplexity + sample entropy on the fly.
  3. Load / cache human references (real OWT sequences of identical length)
     — `load_human_references` writes a JSON cache to `data.cache_dir` that
     persists across runs on the `mdlm-data` volume, so re-tokenization only
     happens once.
  4. Run mauve.compute_mauve(p_text=refs, q_text=samples) and dump
     {gen_ppl, entropy, MAUVE, entropies, text_samples} to
     `sampling.generated_seqs_path` on the `mdlm-outputs` volume.

Caching behaviour (matches your other run_modal_*.py scripts):
  - Image layer cache: identical pip-install / run_commands spec to
    run_modal_eval.py so Modal reuses the built image — torch,
    transformers, flash-attn, mauve-text etc. are NOT re-installed on
    subsequent `modal run` invocations.
  - Modal Volumes:
      * mdlm-data  — holds the tokenized OWT .dat files (under
                     /vol/data/owt_cache) and the human_refs_*.json cache.
                     Subsequent runs skip the 30-min OWT tokenization and
                     the 5000-ref reference generation.
      * mdlm-checkpoints — base mdlm_owt.ckpt lives here.
      * mdlm-outputs    — finetuned checkpoints live here; MAUVE result
                          JSONs are written here too.
  - HF / mauve-text feature-extractor downloads land in /root/.cache; this
    is per-container (re-downloaded on cold start) but it's a one-shot,
    parallelized download inside the existing image layer, not a re-install.

Setup (one-time, local):
  pip install modal
  modal setup
  modal secret create wandb-secret WANDB_API_KEY=<your-key>

Upload the MDLM checkpoint (base or soft-masked finetune):
  modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt
  modal volume put mdlm-outputs    /local/path/<finetune>.ckpt <finetune>.ckpt

OR pull a soft-masking checkpoint directly from HuggingFace into the volume
on first run (subsequent runs find it on the volume and skip the download):
  modal run run_modal_mauve.py \
      --checkpoint slerp_seed3_lambda_learnt_step5000.ckpt \
      --hf-repo lavanyanigam/soft-masking-checkpoints \
      --algo mdlm_sm

Run MAUVE on the base MDLM (mdlm sampler, 5000 samples):
  cd language/
  modal run run_modal_mauve.py

Run on a soft-masked finetune (mdlm_sm algo, mdlm sampler):
  modal run run_modal_mauve.py \
      --checkpoint softmask_md5-100000.ckpt --algo mdlm_sm

Run a remdm-loop sampler (matches scripts/05_gen_ppl_owt_remdm_loop_sm.sh):
  modal run run_modal_mauve.py \
      --checkpoint softmask_md5-100000.ckpt \
      --algo mdlm_sm --sampler remdm-loop \
      --eta 0.02 --t-on 0.55 --t-off 0.05 --alpha-on 0.9

Quick smoke test (250 samples, eval_batch_size=1, 250 batches):
  modal run run_modal_mauve.py --num-sample-batches 250

Download the JSON of generated sequences + MAUVE score:
  modal volume get mdlm-outputs <remote-path> ./local_outputs/

HuggingFace checkpoint discovery (one-shot, the runner does this automatically
when `--hf-repo` is set and the file is missing from the volume):
  from huggingface_hub import snapshot_download
  snapshot_download(
      repo_id="lavanyanigam/soft-masking-checkpoints",
      allow_patterns=["slerp_seed3_lambda_learnt_step5000.ckpt"],
      local_dir=CKPT_DIR,
  )
"""

import os
import sys
import modal

# ── volumes (persistent across runs) ─────────────────────────────────────────
ckpt_volume = modal.Volume.from_name("mdlm-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("mdlm-data", create_if_missing=True)
out_volume = modal.Volume.from_name("mdlm-outputs", create_if_missing=True)

CKPT_DIR = "/vol/checkpoints"
DATA_DIR = "/vol/data"
OUT_DIR = "/vol/outputs"

# ── container image ───────────────────────────────────────────────────────────
# Identical spec to run_modal_eval.py so Modal reuses the cached image layers.
# mauve-text was already in the base pip_install list (eager-imported by
# main.py) but here we actively use it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "patch", "git-lfs")
    .env({
        # Enable hf_transfer's rust backend so the multi-GB ckpt download via
        # huggingface_hub.snapshot_download finishes quickly.
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .pip_install(
        "numpy<2",
        "datasets==2.15.0",
        "einops==0.7.0",
        "fsspec==2023.10.0",
        "h5py==3.10.0",
        "hydra-core==1.3.2",
        "lightning==2.2.1",
        "omegaconf==2.3.0",
        "packaging==23.2",
        "pandas==2.2.1",
        "rich==13.7.1",
        "scikit-learn==1.4.0",
        "transformers==4.38.2",
        "torchmetrics==1.6.1",
        "wandb",
        "timm",
        "huggingface-hub",
        "hf_transfer",
        "mauve-text",  # used by main.py in mode=sample_eval
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "triton==3.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "pip install --no-deps "
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    )
)

# ── local code copy ──────────────────────────────────────────────────────────
LANGUAGE_DIR = "/workspace/language"
image = image.add_local_dir(".", remote_path=LANGUAGE_DIR, copy=True)

app = modal.App("slerp-vs-topk-mauve")


# ── MAUVE eval function ──────────────────────────────────────────────────────


@app.function(
    image=image,
    gpu="L40S",
    timeout=6 * 3600,  # 6h — 5000 samples @ seq_len=1024 on L40S is ~3-4h
    volumes={
        CKPT_DIR: ckpt_volume,
        DATA_DIR: data_volume,
        OUT_DIR: out_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_mauve(
    checkpoint_filename: str = "mdlm_owt.ckpt",
    algo: str = "mdlm",  # mdlm | mdlm_sm
    sampler: str = "mdlm",  # mdlm | remdm-loop | remdm-cap
    transparency_alg: str = "slerp_sm",  # only used when algo=mdlm_sm
    fixed_lambda: float = -1.0,  # <0 keeps learned lambda; otherwise pinned
    slerp_n_iter: int = 3,
    num_sample_batches: int = 5000,
    eval_batch_size: int = 1,
    seq_len: int = 1024,
    sampling_steps: int = 128,
    p_nucleus: float = 0.9,
    eta: float = 0.02,  # remdm only
    t_on: float = 0.55,  # remdm-loop only
    t_off: float = 0.05,  # remdm-loop only
    alpha_on: float = 0.9,  # remdm-loop only
    perplexity_batch_size: int = 1,
    seed: int = 1,
    output_subdir: str = "mauve",
    wandb_offline: bool = True,
    # ── optional HuggingFace checkpoint source ──────────────────────────────
    # If `hf_repo` is set and the checkpoint isn't on the local Modal Volume
    # yet, the runner downloads it via `huggingface_hub.snapshot_download`
    # into the appropriate volume (`mdlm-checkpoints` for `mdlm_owt.ckpt`,
    # `mdlm-outputs` otherwise) and then proceeds normally. Subsequent runs
    # find the file on the volume and skip the download entirely.
    hf_repo: str = "",
    hf_revision: str = "main",
    hf_token: str = "",  # optional; pass `hf_*****` for gated repos
):
    import subprocess
    import shutil

    if algo not in ("mdlm", "mdlm_sm"):
        raise ValueError(f"algo must be 'mdlm' or 'mdlm_sm', got {algo!r}")
    if sampler not in ("mdlm", "remdm-loop", "remdm-cap"):
        raise ValueError(
            f"sampler must be 'mdlm', 'remdm-loop', or 'remdm-cap'; got {sampler!r}"
        )
    if fixed_lambda >= 0.0 and not 0.0 <= fixed_lambda <= 1.0:
        raise ValueError(
            f"fixed_lambda must lie in [0, 1] when set, got {fixed_lambda}"
        )

    # ── resolve checkpoint location (matches run_modal_eval.py convention) ──
    if checkpoint_filename == "mdlm_owt.ckpt":
        checkpoint_path = f"{CKPT_DIR}/{checkpoint_filename}"
    else:
        checkpoint_path = f"{OUT_DIR}/{checkpoint_filename}"

    # ── one-shot HuggingFace download into the appropriate Modal Volume ────
    # This runs ONLY if the file is missing locally AND --hf-repo was supplied.
    # The downloaded file lands on the persistent volume, so the next run
    # finds it and skips the download (no re-pull on subsequent `modal run`s).
    if not os.path.exists(checkpoint_path):
        if not hf_repo:
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}.\n"
                f"Either upload it manually:\n"
                f"  modal volume put mdlm-checkpoints /local/path/<ckpt> "
                f"{checkpoint_filename}\n"
                f"or:\n"
                f"  modal volume put mdlm-outputs /local/path/<ckpt> "
                f"{checkpoint_filename}\n"
                f"…or have the runner fetch it by passing --hf-repo "
                f"<user/repo> (e.g. --hf-repo "
                f"lavanyanigam/soft-masking-checkpoints)."
            )

        from huggingface_hub import snapshot_download

        # The base `mdlm_owt.ckpt` is not in the public soft-masking repo
        # this runner is wired against, so refuse to silently 404 on the
        # base name. Users wanting the public base ckpt should use the
        # upstream MDLM release on HF and pass --hf-repo accordingly.
        if checkpoint_filename == "mdlm_owt.ckpt" and \
                hf_repo == "lavanyanigam/soft-masking-checkpoints":
            raise FileNotFoundError(
                f"{checkpoint_filename!r} is not in {hf_repo}. "
                f"Upload it via `modal volume put mdlm-checkpoints ...` "
                f"or point --hf-repo at a repo that contains it."
            )

        download_volume = (
            ckpt_volume if checkpoint_filename == "mdlm_owt.ckpt"
            else out_volume
        )
        local_dir = (
            CKPT_DIR if checkpoint_filename == "mdlm_owt.ckpt" else OUT_DIR
        )
        os.makedirs(local_dir, exist_ok=True)

        # hf_transfer (set via the image env) speeds up multi-GB ckpt pulls
        # via the rust backend.
        print(f"[hf] Missing locally; downloading {checkpoint_filename} "
              f"from {hf_repo}@{hf_revision} into {local_dir} ...")
        downloaded = snapshot_download(
            repo_id=hf_repo,
            revision=hf_revision,
            allow_patterns=[checkpoint_filename],
            local_dir=local_dir,
            token=hf_token or None,
        )
        # snapshot_download with allow_patterns returns the local dir; the
        # file lands at <local_dir>/<checkpoint_filename> when the pattern
        # is a single file. Some HF layouts nest under blobs/, snapshots/,
        # so fall back to a recursive scan.
        downloaded_path = os.path.join(downloaded, checkpoint_filename)
        if not os.path.exists(downloaded_path):
            for root, _, files in os.walk(downloaded):
                if checkpoint_filename in files:
                    downloaded_path = os.path.join(root, checkpoint_filename)
                    break
            else:
                raise FileNotFoundError(
                    f"hf_hub snapshot_download did not produce "
                    f"{checkpoint_filename} under {downloaded}"
                )
        if downloaded_path != checkpoint_path:
            shutil.move(downloaded_path, checkpoint_path)
        download_volume.commit()
        print(f"[hf] Downloaded and committed to volume: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        # Belt-and-braces in case the download step above completed but the
        # move or commit didn't land where we expect.
        raise FileNotFoundError(
            f"Checkpoint still missing at {checkpoint_path} after HF fetch."
        )

    # ── writable workdir (mounted source is read-only) ───────────────────────
    work_dir = "/root/language"
    if not os.path.exists(work_dir):
        shutil.copytree(LANGUAGE_DIR, work_dir)
    os.chdir(work_dir)

    if not os.path.exists(f"{work_dir}/models/dit.py"):
        print("[setup] DUO files missing — running setup.sh ...")
        subprocess.run(["bash", "setup.sh"], check=True, cwd=work_dir)
    else:
        print("[setup] DUO files present — skipping setup.sh")

    # ── drop corrupt cached dataset dirs (same guard as run_modal.py) ────────
    # A previous crashed run can leave a .dat directory that exists but has
    # incomplete Arrow files. load_from_disk() then raises FileNotFoundError.
    # Detect and remove these before training.  Crucially, the .dat directories
    # that DO load_from_disk successfully are NOT touched — those are the
    # tokenization cache that lets subsequent runs skip the 30-min OWT pass.
    import datasets as hf_datasets

    owt_cache = f"{DATA_DIR}/owt_cache"
    for dat_name in [
        "openwebtext-split_train_bs1024_wrapped.dat",
        "openwebtext-split_valid_bs1024_wrapped.dat",
    ]:
        dat_path = os.path.join(owt_cache, dat_name)
        if os.path.exists(dat_path):
            try:
                hf_datasets.load_from_disk(dat_path)
                print(f"[cache] Valid: {dat_path}")
            except Exception as e:
                print(f"[cache] Corrupted ({e}) — removing: {dat_path}")
                shutil.rmtree(dat_path, ignore_errors=True)

    # ── output paths (Hydra run dir + JSON dump of generations+MAUVE) ────────
    tag = f"{algo}_{sampler}_{os.path.splitext(checkpoint_filename)[0]}_seed{seed}"
    out_dir = f"{OUT_DIR}/{output_subdir}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    generated_seqs_path = os.path.join(
        out_dir,
        f"{sampler}_T-{sampling_steps}"
        f"_topp-{p_nucleus}_eta-{eta}"
        f"_ton-{t_on}_toff-{t_off}_alphaon-{alpha_on}.json",
    )

    # ── build the hydra command (mirrors scripts/03..05_gen_ppl_owt_*.sh) ──
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "main",
        "mode=sample_eval",
        f"algo={algo}",
        "model=small",
        f"model.length={seq_len}",
        "data=openwebtext-split",
        f"data.cache_dir={DATA_DIR}/owt_cache",
        f"eval.checkpoint_path={checkpoint_path}",
        f"seed={seed}",
        "trainer.accelerator=cuda",
        "trainer.devices=1",
        # sample_eval only iterates `trainer.max_steps`; cap it at the number of
        # batches we want so lightning doesn't try to spin up a never-ending
        # training loop.
        f"trainer.max_steps={num_sample_batches}",
        "trainer.num_sanity_val_steps=0",
        # Inherit the project default (ddp). On a single visible device
        # Lightning's DDP just becomes single-process; the rank-0-only
        # generate + MAUVE path in main.py runs locally without spawning
        # extra workers. There's no `auto` strategy in this project's
        # configs/strategy dir (only ddp / fsdp).
        "strategy=ddp",
        f"loader.eval_batch_size={eval_batch_size}",
        f"loader.batch_size={eval_batch_size}",
        f"eval.perplexity_batch_size={perplexity_batch_size}",
        "loader.num_workers=4",
        # MAUVE lives in main.py, not in the trainer's validation_step — leave
        # limit_val_batches at its default (sample_eval path doesn't validate).
        f"sampling.steps={sampling_steps}",
        f"sampling.num_sample_batches={num_sample_batches}",
        f"sampling.p_nucleus={p_nucleus}",
        f"sampling.sampler={sampler}",
        f"sampling.eta={eta}",
        f"sampling.t_on={t_on}",
        f"sampling.t_off={t_off}",
        f"sampling.alpha_on={alpha_on}",
        f"sampling.generated_seqs_path={generated_seqs_path}",
        # sample_eval doesn't need EMA flips; keep the default.
        "eval.disable_ema=False",
        f"+wandb.offline={str(wandb_offline).lower()}",
        f"++hydra.run.dir={out_dir}",
        f"++checkpointing.save_dir={out_dir}",
    ]

    if algo == "mdlm_sm":
        cmd.append(f"algo.tran_head.transparency_alg={transparency_alg}")
        cmd.append(f"algo.tran_head.slerp_n_iter={slerp_n_iter}")
        if fixed_lambda >= 0.0:
            cmd.append(f"algo.tran_head.fixed_lambda={fixed_lambda}")
        # remdm-* samplers consume the soft-mask predictor; keep this as the
        # default for parity with the existing scripts.
        if sampler.startswith("remdm"):
            cmd.append("sampling.predictor=sm")
        if fixed_lambda >= 0.0:
            print(f"[mauve] Fixed lambda override: {fixed_lambda}")
        print(f"[mauve] Transparency algorithm: {transparency_alg}")
        print(f"[mauve] SLERP iterations: {slerp_n_iter}")

    print(f"[mauve] algo={algo} sampler={sampler} ckpt={checkpoint_filename}")
    print(f"[mauve] num_sample_batches={num_sample_batches} "
          f"eval_batch_size={eval_batch_size} seq_len={seq_len} "
          f"steps={sampling_steps} p_nucleus={p_nucleus}")
    if sampler.startswith("remdm"):
        print(
            f"[mauve] remdm params: eta={eta} t_on={t_on} t_off={t_off} "
            f"alpha_on={alpha_on}"
        )
    print(f"[mauve] Output dir: {out_dir}")
    print(f"[mauve] Generated seqs path: {generated_seqs_path}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
    finally:
        # data first (cached human_refs + .dat), then outputs (generations JSON).
        # Both volumes commit so the next `modal run` finds the freshly-cached
        # human_refs JSON + the MAUVE results JSON on the persistent filesystem.
        data_volume.commit()
        out_volume.commit()

    print(f"[mauve] Done. Results saved to {generated_seqs_path}")


# ── local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    checkpoint: str = "mdlm_owt.ckpt",
    algo: str = "mdlm",  # mdlm | mdlm_sm
    sampler: str = "mdlm",  # mdlm | remdm-loop | remdm-cap
    transparency_alg: str = "slerp_sm",  # only used when algo=mdlm_sm
    fixed_lambda: float = -1.0,
    slerp_n_iter: int = 3,
    num_sample_batches: int = 5000,
    eval_batch_size: int = 1,
    seq_len: int = 1024,
    sampling_steps: int = 128,
    p_nucleus: float = 0.9,
    eta: float = 0.02,
    t_on: float = 0.55,
    t_off: float = 0.05,
    alpha_on: float = 0.9,
    perplexity_batch_size: int = 1,
    seed: int = 1,
    output_subdir: str = "mauve",
    offline: bool = True,
    hf_repo: str = "",
    hf_revision: str = "main",
    hf_token: str = "",
):
    run_mauve.remote(
        checkpoint_filename=checkpoint,
        algo=algo,
        sampler=sampler,
        transparency_alg=transparency_alg,
        fixed_lambda=fixed_lambda,
        slerp_n_iter=slerp_n_iter,
        num_sample_batches=num_sample_batches,
        eval_batch_size=eval_batch_size,
        seq_len=seq_len,
        sampling_steps=sampling_steps,
        p_nucleus=p_nucleus,
        eta=eta,
        t_on=t_on,
        t_off=t_off,
        alpha_on=alpha_on,
        perplexity_batch_size=perplexity_batch_size,
        seed=seed,
        output_subdir=output_subdir,
        wandb_offline=offline,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        hf_token=hf_token,
    )
