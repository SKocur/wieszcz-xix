"""Score the modern comparators on the held-out windows the ladder is scored on.

The era contrast of `temporal_probe.py` compares models on hand-written carriers. This
gives the plain baseline beside it: bits per byte of Bielik and papuGaPT2 on the text of
the dense protocol's own windows, pooled and per source, against the three rungs on the
same windows.

A window is the one `src/eval_val.evaluate` scores: `block + 1` tokens of the held-out
file, the first as context and the rest as targets. Its bytes are rebuilt from the
byte-level vocabulary, a document terminator becoming a blank line, and handed to the
comparator's own tokenizer. As for our models, the first token is context only and every
later token is scored, so each model is charged for the bytes after its first token.
Where a window is longer than the comparator's context, the remainder is scored in
further passes that keep the context full, so no token is scored twice and none is left
out. Bits per byte is total bits over total bytes, the paper's conversion, and intervals
are a bootstrap over windows; the difference from each rung resamples the same windows
for both models.

The rungs are not re-run. Their per-window losses come from the report of
`eval_by_source.py`, which uses the same window rule.

    .venv/bin/python3 scripts/comparator_heldout.py \\
        --hf-model ~/research/models/bielik-1.5b-v3 --name bielik-1.5b-v3 \\
        --out metrics/comparator_heldout_bielik_2026-10-06.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
LN2 = math.log(2)
EOT = "<|endoftext|>"
EOT_BYTES = b"\n\n"
RUNGS = ("47M", "107M", "349M")


def disable_triton_bmm_override() -> bool:
    """Keep `bmm` on its reference kernel, as `temporal_probe.py` does."""
    try:
        from torch._native.registry import deregister_op_overrides
    except ImportError:
        return False
    deregister_op_overrides(disable_op_symbols="bmm")
    return True


def token_bytes(vocab_path: Path) -> tuple[list[bytes], int]:
    """Bytes of every token, by reversing the GPT-2 byte-to-unicode mapping."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    table = [b""] * (max(vocab.values()) + 1)
    for tok, i in vocab.items():
        table[i] = EOT_BYTES if tok == EOT else bytes(u2b[ch] for ch in tok)
    return table, vocab[EOT]


def window_starts(n_tokens: int, block: int, cap: int) -> np.ndarray:
    max_windows = (n_tokens - 1) // block
    return np.linspace(0, max_windows - 1, min(cap, max_windows), dtype=np.int64) * block


@torch.no_grad()
def score_text(model, ids: list[int], block: int, device: str) -> float:
    """Summed NLL in nats of ids[1:], each token predicted from at most `block - 1` before it."""
    total = 0.0
    pos = 1
    while pos < len(ids):
        end = min(len(ids), block) if pos == 1 else min(len(ids), pos + block // 2)
        start = max(0, end - block)
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        lp = torch.log_softmax(model(x).logits[0].float(), dim=-1)
        tgt = torch.tensor(ids[pos:end], dtype=torch.long, device=device)
        total += float(-lp[pos - start - 1:end - start - 1].gather(1, tgt.unsqueeze(1)).sum())
        pos = end
    return total


def ratio_ci(bits: np.ndarray, nbytes: np.ndarray, idx: np.ndarray) -> list[float]:
    draws = bits[idx].sum(axis=1) / nbytes[idx].sum(axis=1)
    return [round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default="data/val_2026-08-03.bin")
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--ladder", default="metrics/eval_by_source_2026-08-17_fullspan.json")
    ap.add_argument("--windows", type=int, default=8192)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--limit", type=int, default=0,
                    help="score an evenly spaced subset of the windows (smoke test)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    triton_off = disable_triton_bmm_override()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    path = str(Path(args.hf_model).expanduser())
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    kw = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
    model = model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    context = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
    print(f"{args.name}: {n_params/1e6:.1f}M params, context {context}, {args.device}",
          flush=True)

    table, eot = token_bytes(REPO / "tokenizer/vocab.json")
    byte_len = np.array([len(b) for b in table], dtype=np.int64)
    data = np.asarray(np.memmap(REPO / args.val, dtype=np.uint16, mode="r"))
    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    n_ia = sum(1 for i in split["val_ids"] if i.startswith("ia_"))
    cut = int(np.flatnonzero(data == eot)[n_ia - 1]) + 1
    subsets = {"full": data, "ia": data[:cut], "wl": data[cut:]}
    ladder_path = REPO / args.ladder
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    if ladder["meta"]["source_boundary_token"] != cut:
        raise SystemExit("source boundary differs from the ladder report's")

    rng = np.random.default_rng(args.seed)
    results: dict = {}
    arrays: dict = {}
    for name, arr in subsets.items():
        starts = window_starts(len(arr), args.block, args.windows)
        rung_nats = np.load(ladder_path.with_suffix(f".{name}.windows.npy"))
        if rung_nats.shape != (len(RUNGS), len(starts)):
            raise SystemExit(f"{name}: ladder windows {rung_nats.shape}, ours {len(starts)}")
        rung_bytes = np.array([byte_len[arr[s + 1:s + 1 + args.block]].sum() for s in starts])
        if args.limit and args.limit < len(starts):
            keep = np.linspace(0, len(starts) - 1, args.limit, dtype=np.int64)
            starts, rung_nats, rung_bytes = starts[keep], rung_nats[:, keep], rung_bytes[keep]

        bits = np.empty(len(starts))
        nbytes = np.empty(len(starts), dtype=np.int64)
        ntok = np.empty(len(starts), dtype=np.int64)
        extra_passes = 0
        for k, s in enumerate(starts):
            raw = b"".join(table[t] for t in arr[s:s + 1 + args.block])
            text = raw.decode("utf-8", errors="ignore")
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            first_end = enc["offset_mapping"][0][1]
            bits[k] = score_text(model, ids, context, args.device) / LN2
            nbytes[k] = len(text[first_end:].encode("utf-8"))
            ntok[k] = len(ids) - 1
            extra_passes += len(ids) > context
            if (k + 1) % 500 == 0 or k + 1 == len(starts):
                rate = (k + 1) / (time.time() - t0)
                print(f"  {name} {k+1}/{len(starts)} | bpb {bits[:k+1].sum()/nbytes[:k+1].sum():.4f}"
                      f" | {rate:.1f} win/s", flush=True)

        idx = rng.integers(0, len(starts), size=(args.bootstrap, len(starts)))
        own = bits.sum() / nbytes.sum()
        res = {
            "windows": int(len(starts)), "scored_bytes": int(nbytes.sum()),
            "scored_tokens": int(ntok.sum()),
            "bytes_per_token": round(float(nbytes.sum() / ntok.sum()), 4),
            "windows_longer_than_context": int(extra_passes),
            "bits_per_byte": round(float(own), 4),
            "ci95": ratio_ci(bits, nbytes, idx),
            "rungs": {},
        }
        for r, label in enumerate(RUNGS):
            rbits = rung_nats[r] * args.block / LN2
            level = rbits.sum() / rung_bytes.sum()
            diff = (bits[idx].sum(axis=1) / nbytes[idx].sum(axis=1)
                    - rbits[idx].sum(axis=1) / rung_bytes[idx].sum(axis=1))
            res["rungs"][label] = {
                "bits_per_byte": round(float(level), 4),
                "ci95": ratio_ci(rbits, rung_bytes, idx),
                "comparator_minus_rung": round(float(own - level), 4),
                "comparator_minus_rung_ci95": [round(float(x), 4)
                                               for x in np.percentile(diff, [2.5, 97.5])],
            }
        results[name] = res
        arrays[f"{name}_bits"] = bits
        arrays[f"{name}_bytes"] = nbytes
        arrays[f"{name}_rung_bytes"] = rung_bytes
        print(json.dumps({name: res}, indent=1), flush=True)

    report = {
        "meta": {
            "script": "scripts/comparator_heldout.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(), "device": args.device, "dtype": args.dtype,
            "model": args.name, "hf_model": args.hf_model, "params": n_params,
            "context_tokens": context, "attn_implementation": "eager",
            "triton_bmm_override_disabled": triton_off,
            "val": args.val, "split": args.split, "ladder": args.ladder,
            "block": args.block, "windows_cap": args.windows, "limit": args.limit,
            "source_boundary_token": cut, "eot_rendered_as": EOT_BYTES.decode(),
            "bootstrap_draws": args.bootstrap, "seed": args.seed,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "by_subset": results,
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(out.with_suffix(".windows.npz"), **arrays)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
