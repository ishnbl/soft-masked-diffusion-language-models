"""
Multi-GPU Modal runner for the SLERP (`slerp_sm`) soft-masking path.

This is the multi-GPU counterpart to run_modal.py. It launches the `slerp_sm`
pathway under PyTorch-Lightning DDP across N GPUs *in a single Modal container*
(single node, N processes), which is exactly what exercises the DDP-safe
soft-mask gate added in algo.py (`MDLM_SM.nll`): step-seeded Bernoulli +
globally all-reduced time band, so every rank takes the same soft-mask branch.

It reuses run_modal.py's image / volumes / DUO-setup / dataset-cache hardening,
and adds the multi-GPU wiring (N visible GPUs => device_count()==N==trainer.devices,
so the world size, the `device_count:` resolver, and the derived
accumulate_grad_batches all stay in lockstep).

Setup (one-time, local) — same as run_modal.py:
  pip install modal
  modal setup
  modal secret create wandb-secret WANDB_API_KEY=<your-key>
  modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt

Quick 2-GPU smoke test (20 steps, small global batch so it finishes fast):
  cd language/
  modal run run_modal_multigpu.py --num-gpus 2 --gpu-type L40S \
      --max-steps 20 --global-batch-size 64

Full 4-GPU finetune run:
  modal run run_modal_multigpu.py --num-gpus 4 --gpu-type A100 \
      --max-steps 5000 --global-batch-size 512 --batch-size 32

Download outputs:
  modal volume get mdlm-outputs <remote-path> <local-path>
"""

import os
import sys

import modal

# Reuse the proven image / volumes / path constants from the single-GPU runner.
from run_modal import (
    image,
    ckpt_volume,
    data_volume,
    out_volume,
    CKPT_DIR,
    DATA_DIR,
    OUT_DIR,
    LANGUAGE_DIR,
)

app = modal.App("slerp-multigpu")


@app.function(
    image=image,
    # Default to 2 L40S; override per-run with `.with_options(gpu=...)` below so
    # --num-gpus / --gpu-type from the CLI actually change the allocation.
    gpu="L40S:2",
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
    batch_size: int = 32,
    num_workers: int = 4,
    checkpoint_every_n_steps: int = 100,
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
        "algo.tran_head.mixinputs_k=3",
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

    print(f"[train] Starting DDP run: {run_name}")
    print(
        f"[train] world size={num_gpus}  per-GPU batch={batch_size}  "
        f"accum={accumulate_grad_batches}  global={global_batch_size}"
    )
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
    num_gpus: int = 2,
    gpu_type: str = "L40S",  # L40S | A100 | A100-80GB | H100 ...
    base_ckpt: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    global_batch_size: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    checkpoint_every_n_steps: int = 100,
    log_every_n_steps: int = 3,
):
    # Override the static decorator GPU spec at call time so --num-gpus/--gpu-type
    # actually change the allocation (e.g. "A100:4").
    gpu_spec = f"{gpu_type}:{num_gpus}"
    print(f"[local] Requesting gpu={gpu_spec} (single node, {num_gpus}-way DDP)")
    train = train_mg.with_options(gpu=gpu_spec)
    train.remote(
        num_gpus=num_gpus,
        base_ckpt_filename=base_ckpt,
        seed=seed,
        max_steps=max_steps,
        model=model,
        data=data,
        slerp_n_iter=slerp_n_iter,
        global_batch_size=global_batch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        log_every_n_steps=log_every_n_steps,
    )
    print("\n[local] Run complete.")
    print(
        f"[local] Download outputs:  modal volume get mdlm-outputs "
        f"{OUT_DIR}/slerp_mg_{num_gpus}gpu_{model}_{data}_seed{seed} ./local_outputs/"
    )
