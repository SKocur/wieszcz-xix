"""Generate text from a trained checkpoint, using the KV-cache in GPT.generate.

    python src/sample.py --ckpt checkpoints/final.pt --prompt "Rankiem, gdy"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from train import GPT, get_device

TOKENIZER_DIR = Path("tokenizer")
EOT = "<|endoftext|>"


def load_model(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(ckpt["cfg"])
    # Strip the torch.compile prefix if the checkpoint was compiled.
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/final.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0, help="0 disables; nucleus (--top-p) is the default filter")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus cumulative-mass cutoff; 1.0 disables")
    p.add_argument("--seed", type=int, default=1337,
                   help="sampling seed; a quoted sample nobody can reproduce is an anecdote")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model = load_model(args.ckpt, device)
    tok = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt")
    )
    eot_id = tok.token_to_id(EOT)

    ids = tok.encode(args.prompt).ids or [eot_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, eot_id=eot_id
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
