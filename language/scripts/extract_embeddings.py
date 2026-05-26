"""
Extract dense token embeddings from a saved MDLM checkpoint.

Reads the backbone.vocab_embed.embedding parameter (shape V × D) from a
PyTorch-Lightning .ckpt file and writes a .npz with:

  embeddings        float32  (V, D)   raw embeddings as stored in the checkpoint
  embeddings_normed float32  (V, D)   L2-normalised (unit vectors on S^{D-1})
  tokens            str      (V,)     human-readable token strings from the tokenizer

Usage:
  python scripts/extract_embeddings.py -c /path/to/mdlm.ckpt
  python scripts/extract_embeddings.py -c /path/to/mdlm.ckpt -o embeddings.npz
  python scripts/extract_embeddings.py -c /path/to/mdlm.ckpt --tokenizer gpt2
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


EMBEDDING_KEY = "backbone.vocab_embed.embedding"


def parse_args():
    p = argparse.ArgumentParser(description="Extract token embeddings from an MDLM checkpoint")
    p.add_argument("-c", "--checkpoint", required=True,
                   help="Path to the .ckpt file (Lightning checkpoint)")
    p.add_argument("-o", "--output", default=None,
                   help="Output .npz path. Defaults to <checkpoint_dir>/token_embeddings.npz")
    p.add_argument("--tokenizer", default="gpt2",
                   help="HuggingFace tokenizer name/path for token strings (default: gpt2)")
    p.add_argument("--no-tokens", action="store_true",
                   help="Skip loading the tokenizer (embeddings + normed only)")
    return p.parse_args()


def load_embeddings(ckpt_path: str) -> torch.Tensor:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in ckpt:
        sys.exit(
            f"[error] Checkpoint does not have a 'state_dict' key. "
            f"Top-level keys: {list(ckpt.keys())}"
        )

    state_dict = ckpt["state_dict"]

    if EMBEDDING_KEY not in state_dict:
        candidates = [k for k in state_dict if "embed" in k.lower()]
        sys.exit(
            f"[error] Key '{EMBEDDING_KEY}' not found in state_dict.\n"
            f"Keys containing 'embed': {candidates}\n"
            f"If your checkpoint uses a different key, edit EMBEDDING_KEY at the top of this script."
        )

    emb = state_dict[EMBEDDING_KEY]   # (V, D)
    print(f"  Embedding shape : {tuple(emb.shape)}  (vocab_size={emb.shape[0]}, dim={emb.shape[1]})")
    print(f"  dtype           : {emb.dtype}")
    return emb


def load_token_strings(tokenizer_name: str, vocab_size: int):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[warning] transformers not installed — skipping token strings. "
              "Run `pip install transformers` to include them.")
        return None

    print(f"Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    if len(tok) != vocab_size:
        print(f"[warning] Tokenizer vocab size ({len(tok)}) != embedding vocab size ({vocab_size}). "
              f"Token strings may be misaligned. Proceeding anyway.")

    tokens = [tok.convert_ids_to_tokens(i) for i in range(vocab_size)]
    return np.array(tokens, dtype=object)


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[error] Checkpoint not found: {args.checkpoint}")

    emb = load_embeddings(args.checkpoint)
    V, D = emb.shape

    # Raw embeddings (float32)
    raw = emb.to(torch.float32).numpy()

    # L2-normalised embeddings
    normed = F.normalize(emb.to(torch.float32), dim=-1).numpy()

    # Build output path
    if args.output is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.output = os.path.join(ckpt_dir, "token_embeddings.npz")

    # Token strings
    save_dict = {
        "embeddings": raw,
        "embeddings_normed": normed,
    }

    if not args.no_tokens:
        tokens = load_token_strings(args.tokenizer, V)
        if tokens is not None:
            save_dict["tokens"] = tokens

    # Save
    print(f"Saving to: {args.output}")
    np.savez(args.output, **save_dict)

    # Summary
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone.")
    print(f"  File size  : {size_mb:.1f} MB")
    print(f"  Arrays     : {list(save_dict.keys())}")
    print(f"  embeddings : shape {raw.shape}, dtype {raw.dtype}")
    print(f"\nTo load in Python:")
    print(f"  data = np.load('{args.output}', allow_pickle=True)")
    print(f"  emb  = data['embeddings']        # ({V}, {D})  raw")
    print(f"  norm = data['embeddings_normed']  # ({V}, {D})  unit vectors")
    if "tokens" in save_dict:
        print(f"  toks = data['tokens']             # ({V},)     token strings")


if __name__ == "__main__":
    main()
