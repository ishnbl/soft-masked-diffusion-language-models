"""
Multi-GPU Modal runner for the SLERP (`slerp_sm`) soft-masking path.

This is the multi-GPU counterpart to run_modal.py. It launches the `slerp_sm`
pathway under PyTorch-Lightning DDP across N GPUs *in a single Modal container*
(single node, N processes), which is exactly what exercises the DDP-safe
soft-mask gate added in algo.py (`MDLM_SM.nll`): step-seeded Bernoulli +
globally all-reduced time band, so every rank takes the same soft-mask branch.

It duplicates the image / volumes / path constants from run_modal.py (identical
spec so Modal reuses cached image layers), and adds the multi-GPU wiring
(N visible GPUs => device_count()==N==trainer.devices,
so the world size, the `device_count:` resolver, and the derived
accumulate_grad_batches all stay in lockstep).

Setup (one-time, local) — same as run_modal.py:
  pip install modal
  modal setup
  modal secret create wandb-secret WANDB_API_KEY=<your-key>
  modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt

GPU count/type are set via NUM_GPUS / GPU_TYPE env vars (they must be fixed at
import time; this Modal version has no per-call gpu override). Everything else is
a normal --flag.

Quick 2-GPU smoke test (20 steps, small global batch so it finishes fast):
  cd language/
  NUM_GPUS=2 GPU_TYPE=L40S modal run run_modal_multigpu.py \
      --max-steps 20 --global-batch-size 64 --batch-size 16

Full 4-GPU finetune run:
  NUM_GPUS=4 GPU_TYPE=A100 modal run run_modal_multigpu.py \
      --max-steps 5000 --global-batch-size 512 --batch-size 16

Pin lambda to a fixed value (skips the learned/entropy-gated lambda head, same as
run_modal.py's --fixed-lambda); must be in [0, 1]:
  NUM_GPUS=4 GPU_TYPE=A100 modal run run_modal_multigpu.py \
      --max-steps 5000 --global-batch-size 512 --batch-size 16 --fixed-lambda 0.5

Download outputs:
  modal volume get mdlm-outputs <remote-path> <local-path>
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
LANGUAGE_DIR = "/workspace/language"

# ── container image ───────────────────────────────────────────────────────────
# Identical spec to run_modal.py so Modal reuses the cached image layers.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "patch", "git-lfs")
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
        "mauve-text",
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
    .add_local_dir(".", remote_path=LANGUAGE_DIR, copy=True)
)

app = modal.App("slerp-multigpu")

# GPU allocation must be fixed at function-definition (import) time. Older Modal
# has no per-call gpu override (Function.with_options), so the count/type are
# read from env vars here instead of CLI flags. Set them before `modal run`:
#   NUM_GPUS=4 GPU_TYPE=A100 modal run run_modal_multigpu.py --max-steps 5000 ...
NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
GPU_TYPE = os.environ.get("GPU_TYPE", "L40S")  # L40S | A100 | A100-80GB | H100 ...
GPU_SPEC = f"{GPU_TYPE}:{NUM_GPUS}"


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=7 * 3600,
    volumes={CKPT_DIR: ckpt_volume, DATA_DIR: data_volume, OUT_DIR: out_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_mg(
    num_gpus: int,
    base_ckpt_filename: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    global_batch_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    mixinputs_k: int = 3,
    checkpoint_every_n_steps: int = 100,
    fixed_lambda: float = -1.0,
    log_every_n_steps: int = 3,
):
    import subprocess
    import shutil

    import torch

    # ── preflight: GPUs visible to this container ────────────────────────────
    visible = torch.cuda.device_count()
    if visible < num_gpus:
        raise RuntimeError(
            f"Requested num_gpus={num_gpus} but the container only sees "
            f"{visible} GPU(s). Pass a matching gpu spec, e.g. "
            f'gpu="A100:{num_gpus}".'
        )
    # All `visible` GPUs are used by DDP; keep the world size consistent with the
    # accumulate math below (and with the `device_count:` resolver + dataloader
    # assertion, which both key off torch.cuda.device_count()).
    if visible != num_gpus:
        raise RuntimeError(
            f"Container sees {visible} GPUs but num_gpus={num_gpus}; this runner "
            f"uses ALL visible GPUs. Set --num-gpus {visible} or request "
            f"exactly {num_gpus} GPUs."
        )

    base_ckpt = f"{CKPT_DIR}/{base_ckpt_filename}"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at {base_ckpt}.\n"
            f"Upload it with:\n"
            f"  modal volume put mdlm-checkpoints /local/path/mdlm.ckpt "
            f"{base_ckpt_filename}"
        )

    if global_batch_size % (num_gpus * batch_size) != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"num_gpus*batch_size ({num_gpus}*{batch_size}={num_gpus * batch_size}). "
            f"The dataloader asserts global == batch * device_count * accum."
        )
    if fixed_lambda >= 0.0 and not 0.0 <= fixed_lambda <= 1.0:
        raise ValueError(
            f"fixed_lambda must lie in [0, 1] when set, got {fixed_lambda}"
        )
    accumulate_grad_batches = global_batch_size // (num_gpus * batch_size)

    # The mounted source is read-only — copy to a writable workdir (setup.sh
    # writes models/dit.py etc.).
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
    import datasets as hf_datasets

    owt_cache = f"{DATA_DIR}/owt_cache"
    for dat_name in [
        f"{data.replace('-split', '-train')}_train_bs1024_wrapped.dat",
        f"{data.replace('-split', '-valid')}_valid_bs1024_wrapped.dat",
    ]:
        dat_path = os.path.join(owt_cache, dat_name)
        if os.path.exists(dat_path):
            try:
                hf_datasets.load_from_disk(dat_path)
                print(f"[cache] Valid: {dat_path}")
            except Exception as e:
                print(f"[cache] Corrupted ({e}) — removing: {dat_path}")
                shutil.rmtree(dat_path, ignore_errors=True)

    group = f"slerp_mg_{num_gpus}gpu_{model}_{data}_seed{seed}"
    run_name = f"slerp-mg-{num_gpus}gpu-{model}-{data}-seed{seed}"
    out_dir = f"{OUT_DIR}/{group}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "main",
        "algo=mdlm_sm",
        "algo.tran_head.transparency_alg=slerp_sm",
        f"algo.tran_head.slerp_n_iter={slerp_n_iter}",
        f"algo.tran_head.mixinputs_k={mixinputs_k}",
        # lambda-activation init (same as the finetune scripts).
        "algo.tran_head.init_scale=0.5",
        "algo.tran_head.init_centre=-4",
        f"model={model}",
        f"data={data}",
        f"data.cache_dir={DATA_DIR}/owt_cache",
        f"seed={seed}",
        f"loader.global_batch_size={global_batch_size}",
        f"loader.eval_global_batch_size={global_batch_size}",
        f"loader.batch_size={batch_size}",
        f"loader.eval_batch_size={batch_size}",
        f"loader.num_workers={num_workers}",
        # --- multi-GPU DDP ---
        "strategy=ddp",
        "strategy.find_unused_parameters=True",
        # Set explicitly: the ${device_count:} resolver can read 0 before CUDA
        # init. All `visible` GPUs are used, and visible==num_gpus (asserted above).
        f"trainer.devices={num_gpus}",
        "trainer.num_nodes=1",
        f"trainer.max_steps={max_steps}",
        "trainer.num_sanity_val_steps=0",
        "trainer.limit_val_batches=0",
        f"trainer.log_every_n_steps={log_every_n_steps}",
        "optim.lr=3e-5",
        "optim.tran_head_lr=0.01",
        "optim.sm_prob=0.8",
        "optim.sm_t_min=0.2",
        "optim.sm_t_max=0.8",
        "lr_scheduler.num_warmup_steps=200",
        "sampling.predictor=sm",
        "eval.compute_generative_perplexity=False",
        "eval.generate_samples=False",
        f"training.finetune_path={base_ckpt}",
        "checkpointing.resume_from_ckpt=false",
        f"callbacks.checkpoint_every_n_steps.every_n_train_steps={checkpoint_every_n_steps}",
        f"wandb.group={group}",
        f"wandb.name={run_name}",
        f"++hydra.run.dir={out_dir}",
        f"++checkpointing.save_dir={out_dir}",
    ]

    if fixed_lambda >= 0.0:
        cmd.append(f"algo.tran_head.fixed_lambda={fixed_lambda}")

    print(f"[train] Starting DDP run: {run_name}")
    print(
        f"[train] world size={num_gpus}  per-GPU batch={batch_size}  "
        f"accum={accumulate_grad_batches}  global={global_batch_size}"
    )
    if fixed_lambda >= 0.0:
        print(f"[train] Fixed lambda override: {fixed_lambda}")
    print(f"[train] Output dir: {out_dir}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
    finally:
        data_volume.commit()
        out_volume.commit()
    print(f"[train] Done. Outputs saved to volume at {out_dir}")


@app.local_entrypoint()
def main(
    base_ckpt: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    global_batch_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    mixinputs_k: int = 3,
    checkpoint_every_n_steps: int = 100,
    fixed_lambda: float = -1.0,
    log_every_n_steps: int = 3,
):
    # GPU count/type are fixed at import from NUM_GPUS / GPU_TYPE env vars
    # (see the module-level GPU_SPEC), because this Modal version can't override
    # the gpu allocation per call.
    print(
        f"[local] gpu={GPU_SPEC} (single node, {NUM_GPUS}-way DDP). "
        f"Set NUM_GPUS / GPU_TYPE env vars to change this."
    )
    train_mg.remote(
        num_gpus=NUM_GPUS,
        base_ckpt_filename=base_ckpt,
        seed=seed,
        max_steps=max_steps,
        model=model,
        data=data,
        slerp_n_iter=slerp_n_iter,
        global_batch_size=global_batch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mixinputs_k=mixinputs_k,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        fixed_lambda=fixed_lambda,
        log_every_n_steps=log_every_n_steps,
    )
    print("\n[local] Run complete.")
    print(
        f"[local] Download outputs:  modal volume get mdlm-outputs "
        f"{OUT_DIR}/slerp_mg_{NUM_GPUS}gpu_{model}_{data}_seed{seed} ./local_outputs/"
    )
