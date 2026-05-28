"""
Modal runner for the SLERP vs top-k soft-masking A/B experiment.

Setup (one-time, local):
  pip install modal
  modal setup                              # browser auth
  modal secret create wandb-secret WANDB_API_KEY=<your-key>

Upload the base MDLM checkpoint:
  modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt

Run the A/B (sequential, one A100):
  cd language/
  modal run run_modal.py

Run in parallel (two A100s simultaneously):
  modal run run_modal.py --parallel

Run only one arm:
  modal run run_modal.py --alg topk
  modal run run_modal.py --alg slerp

Recovery variant (freeze backbone for first 1000 steps):
  modal run run_modal.py --freeze-until 1000

Download outputs:
  modal volume get mdlm-outputs <remote-path> <local-path>
"""

import os
import sys
import modal

# ── volumes (persistent across runs) ─────────────────────────────────────────
ckpt_volume = modal.Volume.from_name("mdlm-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("mdlm-data",        create_if_missing=True)
out_volume  = modal.Volume.from_name("mdlm-outputs",     create_if_missing=True)

CKPT_DIR = "/vol/checkpoints"
DATA_DIR = "/vol/data"
OUT_DIR  = "/vol/outputs"

# ── container image ───────────────────────────────────────────────────────────
# debian_slim + PyTorch cu124 wheels: torch bundles its own CUDA runtime so we
# don't need a heavy nvidia/cuda devel base.  flash_attn 2.7.4.post1 ships
# prebuilt PyPI wheels for torch 2.5.x / CUDA 12.4 / Python 3.12, so the
# ~15-minute compilation step is completely skipped.
#
# Version deltas vs requirements.txt (which targets torch 2.3.1):
#   torch 2.3.1  → 2.5.1   (API-stable for this use-case)
#   torchvision 0.18.1 → 0.20.1
#   torchaudio 2.3.1   → 2.5.1
#   triton 2.2.0       → 3.1.0  (required by torch 2.5.1)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "patch", "git-lfs")
    # Step 1: install ALL non-torch deps first.  Pin numpy<2 because the older
    # ecosystem here (transformers 4.38, datasets 2.15) is not numpy-2 ready,
    # and a transitive resolver may otherwise pull numpy 2.x.
    .pip_install(
        "numpy<2",
        "datasets==2.15.0",
        "einops==0.7.0",
        "fsspec==2023.10.0",     # match datasets 2.15 constraint
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
        "mauve-text",            # imported eagerly by main.py even when unused
    )
    # Step 2: install torch LAST with cu124 index — overrides any CPU torch
    # that lightning/transformers may have pulled in transitively.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "triton==3.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Step 3: flash_attn wheel — binds to the cu124 torch installed above.
    # --no-deps is critical: the wheel's metadata declares a loose torch dep
    # that pip otherwise uses to downgrade torch to 2.4.1 (breaks ABI).
    # Verified at https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .run_commands(
        "pip install --no-deps "
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    )
)

# ── local code copy ──────────────────────────────────────────────────────────
# Bake the entire language/ directory into the image at build time.
# DUO files (models/dit.py etc.) are downloaded by setup.sh at runtime if absent.
# `copy=True` makes the files part of the image layer (cached, fast warm starts).
LANGUAGE_DIR = "/workspace/language"
image = image.add_local_dir(".", remote_path=LANGUAGE_DIR, copy=True)

app = modal.App("slerp-vs-topk-finetune")


# ── training function ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=7 * 3600,        # 7 h — generous for 5k steps
    volumes={
        CKPT_DIR: ckpt_volume,
        DATA_DIR: data_volume,
        OUT_DIR:  out_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    transparency_alg: str,
    base_ckpt_filename: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    freeze_until: int = 0,
):
    import subprocess
    import shutil

    # ── preflight ────────────────────────────────────────────────────────────
    base_ckpt = f"{CKPT_DIR}/{base_ckpt_filename}"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at {base_ckpt}.\n"
            f"Upload it with:\n"
            f"  modal volume put mdlm-checkpoints /local/path/mdlm.ckpt {base_ckpt_filename}"
        )

    # The mounted source at LANGUAGE_DIR is read-only — setup.sh would fail
    # when curl tries to write models/dit.py.  Copy to a writable workdir.
    work_dir = "/root/language"
    if not os.path.exists(work_dir):
        shutil.copytree(LANGUAGE_DIR, work_dir)
    os.chdir(work_dir)

    # ── setup.sh: pull DUO files + apply patches (skipped if already present) ─
    if not os.path.exists(f"{work_dir}/models/dit.py"):
        print("[setup] DUO files missing — running setup.sh ...")
        subprocess.run(["bash", "setup.sh"], check=True, cwd=work_dir)
    else:
        print("[setup] DUO files present — skipping setup.sh")

    # ── build the hydra command ───────────────────────────────────────────────
    alg_tag = "topk" if transparency_alg == "mixinputs_with_topk" else "slerp"
    group    = f"slerp_vs_topk_v2_{model}_{data}_seed{seed}"
    run_name = f"{alg_tag}-ft-v2-{model}-{data}-seed{seed}"
    out_dir  = f"{OUT_DIR}/{group}/{alg_tag}"

    cmd = [
        sys.executable, "-u", "-m", "main",
        "algo=mdlm_sm",
        f"model={model}",
        f"data={data}",
        f"data.cache_dir={DATA_DIR}/owt_cache",
        f"seed={seed}",
        "loader.batch_size=32",
        "loader.eval_batch_size=32",
        f"trainer.max_steps={max_steps}",
        "trainer.val_check_interval=200",
        "trainer.log_every_n_steps=50",
        "optim.lr=3e-5",
        "optim.tran_head_lr=0.01",
        "optim.sm_prob=0.8",
        "lr_scheduler.num_warmup_steps=200",
        "sampling.predictor=sm",
        "strategy.find_unused_parameters=True",
        "eval.compute_generative_perplexity=False",
        "algo.tran_head.mixinputs_k=3",
        "algo.tran_head.init_scale=0.5",
        "algo.tran_head.init_centre=-2.5",
        f"training.finetune_path={base_ckpt}",
        "checkpointing.resume_from_ckpt=false",
        f"wandb.group={group}",
        f"algo.tran_head.transparency_alg={transparency_alg}",
        f"wandb.name={run_name}",
        f"++hydra.run.dir={out_dir}",
        f"++checkpointing.save_dir={out_dir}",
    ]

    if transparency_alg == "slerp_sm":
        cmd.append(f"algo.tran_head.slerp_n_iter={slerp_n_iter}")

    if freeze_until > 0:
        cmd += [
            "+callbacks/freeze_backbone=freeze_backbone",
            f"callbacks.freeze_backbone.freeze_until_step={freeze_until}",
        ]

    print(f"[train] Starting: {run_name}")
    print(f"[train] Output dir: {out_dir}")
    try:
        subprocess.run(cmd, check=True, cwd=work_dir)
    finally:
        # Persist outputs even on crash so partial logs/checkpoints survive
        out_volume.commit()
        data_volume.commit()
    print(f"[train] Done. Outputs saved to volume at {out_dir}")


# ── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    alg: str = "both",               # "topk" | "slerp" | "both"
    base_ckpt: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    freeze_until: int = 0,           # >0 enables FreezeBackboneCallback
    parallel: bool = False,          # True = two A100s simultaneously
):
    kwargs = dict(
        base_ckpt_filename=base_ckpt,
        seed=seed,
        max_steps=max_steps,
        model=model,
        data=data,
        slerp_n_iter=slerp_n_iter,
        freeze_until=freeze_until,
    )

    algs = []
    if alg in ("topk", "both"):
        algs.append("mixinputs_with_topk")
    if alg in ("slerp", "both"):
        algs.append("slerp_sm")
    if not algs:
        raise ValueError(f"--alg must be 'topk', 'slerp', or 'both'; got '{alg}'")

    if parallel and len(algs) > 1:
        # Spawn both on separate A100s; wait for both to finish
        print("[local] Launching both arms in parallel on separate A100s ...")
        handles = [train.spawn(a, **kwargs) for a in algs]
        for h in handles:
            h.get()
    else:
        for a in algs:
            train.remote(a, **kwargs)

    print("\n[local] All runs complete.")
    print(f"[local] Download outputs:  modal volume get mdlm-outputs {OUT_DIR}/<group> ./local_outputs/")
