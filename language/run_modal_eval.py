import os
import sys
import modal

ckpt_volume = modal.Volume.from_name("mdlm-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("mdlm-data", create_if_missing=True)
out_volume = modal.Volume.from_name("mdlm-outputs", create_if_missing=True)

CKPT_DIR = "/vol/checkpoints"
DATA_DIR = "/vol/data"
OUT_DIR = "/vol/outputs"

# Use the same image definition as run_modal.py
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
)

LANGUAGE_DIR = "/workspace/language"
image = image.add_local_dir(".", remote_path=LANGUAGE_DIR, copy=True)

app = modal.App("slerp-vs-topk-eval")

# We dynamically check if wandb-secret exists to decide if we mount it
secrets = []
try:
    # If the user has a local secret list, or if we just want to run with it:
    secrets.append(modal.Secret.from_name("wandb-secret"))
except Exception:
    pass


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
    volumes={
        CKPT_DIR: ckpt_volume,
        DATA_DIR: data_volume,
        OUT_DIR: out_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_eval(
    checkpoint_filename: str = "mdlm_owt.ckpt",
    limit_val_batches: str = "1.0",
    fixed_lambda: float = -1.0,
    wandb_offline: bool = False,
):
    import subprocess
    import shutil

    # Decide volume based on file name
    if checkpoint_filename == "mdlm_owt.ckpt":
        checkpoint_path = f"{CKPT_DIR}/{checkpoint_filename}"
        algo_name = "mdlm"
    else:
        checkpoint_path = f"{OUT_DIR}/{checkpoint_filename}"
        algo_name = "mdlm_sm"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    work_dir = "/root/language"
    if not os.path.exists(work_dir):
        shutil.copytree(LANGUAGE_DIR, work_dir)
    os.chdir(work_dir)

    if not os.path.exists(f"{work_dir}/models/dit.py"):
        print("[setup] DUO files missing — running setup.sh ...")
        subprocess.run(["bash", "setup.sh"], check=True, cwd=work_dir)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "main",
        "mode=ppl_eval",
        f"eval.checkpoint_path={checkpoint_path}",
        f"algo={algo_name}",
        "model=small",
        "data=openwebtext-split",
        f"data.cache_dir={DATA_DIR}/owt_cache",
        "seed=3",
        "trainer.accelerator=cuda",
        "trainer.devices=1",
        f"trainer.limit_val_batches={limit_val_batches}",
        "trainer.num_sanity_val_steps=0",
        f"+wandb.offline={str(wandb_offline).lower()}",
        "loader.eval_global_batch_size=16",
        "loader.eval_batch_size=16",
        "loader.num_workers=4",
        "strategy.find_unused_parameters=True",
    ]

    if algo_name == "mdlm_sm":
        cmd.append("algo.tran_head.transparency_alg=slerp_sm")
        if fixed_lambda >= 0.0:
            cmd.append(f"algo.tran_head.fixed_lambda={fixed_lambda}")

    print(f"[eval] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
    finally:
        data_volume.commit()


@app.local_entrypoint()
def main(
    checkpoint: str = "mdlm_owt.ckpt",
    limit_val_batches: str = "1.0",
    fixed_lambda: float = -1.0,
    offline: bool = False,
):
    run_eval.remote(
        checkpoint_filename=checkpoint,
        limit_val_batches=limit_val_batches,
        fixed_lambda=fixed_lambda,
        wandb_offline=offline,
    )
