"""High-precision held-out perplexity for a trained checkpoint.

The training loop's `estimate_loss` draws 20 random windows per eval, which on the 350M
run spanned 3.009-3.098 across the last dozen readings. This produces a quotable number
instead: deterministic non-overlapping windows evenly strided across the split, per-window
losses kept so the report carries a 95% CI, and fp32 without autocast.

Cross-entropy is in nats/token with its exponential. Every window contributes exactly
`block_size` predicted tokens, so the mean over windows is the token-level mean.

    python src/eval_val.py --ckpt checkpoints/wieszcz_350m_final.pt \
        --data data/clean/val_tail.bin --windows 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train import GPT, get_device

METRICS_DIR = Path("metrics")


def load_model(ckpt_path: str, device: str, dtype: torch.dtype):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = GPT(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device=device, dtype=dtype).eval(), cfg, int(ckpt.get("step", -1)), n_params


def file_fingerprint(path: Path) -> dict:
    """Size plus a hash of the head and tail, which catches a swapped or truncated file
    without rehashing 100+ MB on every eval."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    return {"path": str(path), "bytes": size, "head_tail_sha256": h.hexdigest()}


@torch.no_grad()
def evaluate(model, data: np.memmap, block: int, n_windows: int, batch_size: int,
             device: str, dtype: torch.dtype) -> tuple[np.ndarray, dict]:
    """Mean cross-entropy over `n_windows` non-overlapping windows spread across `data`.

    Stride `block`, so no token is scored twice. Windows are evenly spaced rather than
    taken from the front, because the corpus is ordered by source -- and spacing them by
    an integer stride is not enough to achieve that. `max_windows // n` truncates, so the
    last window starts at `(n - 1) * stride`, which for 8192 windows over this validation
    split is window 49,146 of 55,042: the final 10.7% of the stream is never scored, and
    the Wolne Lektury block begins at 98.9%. A measurement labelled "full" was reading the
    Internet Archive prefix alone.

    Anchoring the last window to the end instead covers the range, and reduces to every
    window when `n` reaches `max_windows`.
    """
    max_windows = (len(data) - 1) // block
    n = min(n_windows, max_windows)
    starts = np.linspace(0, max_windows - 1, n, dtype=np.int64) * block

    losses = np.empty(n, dtype=np.float64)
    t0 = time.time()
    for i in range(0, n, batch_size):
        chunk = starts[i : i + batch_size]
        xs = np.stack([data[s : s + block].astype(np.int64) for s in chunk])
        ys = np.stack([data[s + 1 : s + 1 + block].astype(np.int64) for s in chunk])
        x = torch.from_numpy(xs).to(device)
        y = torch.from_numpy(ys).to(device)
        logits, _ = model(x)
        # reduction="none": the CI needs per-window losses, not the batch mean.
        per_tok = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none"
        ).view(len(chunk), block)
        # .cpu() before .double(): MPS has no float64.
        losses[i : i + len(chunk)] = per_tok.mean(dim=1).cpu().double().numpy()

        done = i + len(chunk)
        if done % (batch_size * 20) == 0 or done == n:
            rate = done / (time.time() - t0)
            print(f"  {done}/{n} windows | running CE {losses[:done].mean():.4f} "
                  f"| {rate:.2f} win/s | ETA {(n - done) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    meta = {
        "windows_evaluated": int(n),
        "windows_available": int(max_windows),
        "first_window_token": int(starts[0]),
        "last_window_token": int(starts[-1]),
        # What fraction of the split the sampled windows actually span. Recorded because a
        # value below 1.0 is the difference between "full" and "the front of it".
        "span_fraction": round(float(starts[-1] + block) / (max_windows * block), 6),
        "tokens_scored": int(n * block),
        "wall_s": round(time.time() - t0, 1),
    }
    return losses, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/clean/val_tail.bin",
                   help="uint16 token file holding the held-out split")
    p.add_argument("--windows", type=int, default=4096,
                   help="number of non-overlapping windows to score (0 = all)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="JSON report path (default: metrics/eval_<ckpt>.json)")
    args = p.parse_args()

    device = args.device or get_device()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    model, cfg, step, n_params = load_model(args.ckpt, device, dtype)
    data_path = Path(args.data)
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    block = cfg["block_size"]

    print(f"ckpt {args.ckpt} | {n_params/1e6:.1f}M params | step {step}")
    print(f"data {data_path} | {len(data):,} tokens | block {block} | {device} {args.dtype}")

    n_windows = args.windows or (len(data) - 1) // block
    losses, meta = evaluate(model, data, block, n_windows, args.batch_size, device, dtype)

    ce = float(losses.mean())
    sem = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    lo, hi = ce - 1.96 * sem, ce + 1.96 * sem

    report = {
        "ckpt": args.ckpt, "step": step, "params": n_params,
        "device": device, "dtype": args.dtype, "batch_size": args.batch_size,
        "data": file_fingerprint(data_path), "data_tokens": int(len(data)),
        "block_size": block,
        "cross_entropy_nats": round(ce, 5),
        "sem": round(sem, 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "perplexity": round(math.exp(ce), 4),
        "perplexity_ci95": [round(math.exp(lo), 4), round(math.exp(hi), 4)],
        "window_loss_std": round(float(losses.std(ddof=1)), 5),
        **meta,
    }

    out = Path(args.out) if args.out else METRICS_DIR / f"eval_{Path(args.ckpt).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out.with_suffix(".windows.npy"), losses)  # for paired tests between models

    print(f"\nval CE = {ce:.4f} nats  (95% CI {lo:.4f}-{hi:.4f}, ±{1.96*sem:.4f})")
    print(f"val PPL = {math.exp(ce):.3f}  (95% CI {math.exp(lo):.3f}-{math.exp(hi):.3f})")
    print(f"scored {meta['tokens_scored']:,} tokens in {meta['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()
