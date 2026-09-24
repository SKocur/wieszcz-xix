"""Reference entropies for the frozen corpus: where the loss axis begins and ends.

A validation loss means little without reference lines. The true entropy rate of
19th-century Polish print cannot be measured directly. It can be bracketed from above by
achievable models, cheapest first: the unigram and conditional bigram entropies of
the training stream (what 1- and 2-gram models reach), the validation cross-entropy
under train-fitted distributions (the held-out version of the same), and a general
compressor on the validation text. Each line is exact and reproducible; together
they turn the ladder's losses into positions on a scale rather than bare numbers.

The compressor runs per source too: the validation floor differs between OCR'd
scans and clean transcriptions, and the gap, in nats per token, is the price of
OCR noise, measured rather than asserted.

    .venv/bin/python3 scripts/entropy_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
VOCAB = 8000
LN2 = math.log(2)


def stream_counts(path: Path, chunk: int = 1 << 27):
    """Unigram and bigram counts in one pass over a uint16 stream."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    uni = np.zeros(VOCAB, dtype=np.int64)
    bi = np.zeros(VOCAB * VOCAB, dtype=np.int64)
    prev_last = None
    for s in range(0, arr.size, chunk):
        block = np.asarray(arr[s:s + chunk]).astype(np.int64)
        uni += np.bincount(block, minlength=VOCAB)
        pairs = block[:-1] * VOCAB + block[1:]
        bi += np.bincount(pairs, minlength=VOCAB * VOCAB)
        if prev_last is not None:
            bi[prev_last * VOCAB + block[0]] += 1
        prev_last = int(block[-1])
    return uni, bi


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(REPO / "data/tokens_frozen_6.69B.bin"))
    ap.add_argument("--val", default=str(REPO / "data/val_2026-08-03.bin"))
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--out", default="metrics/entropy_baseline_2026-08-03.json")
    args = ap.parse_args()

    t0 = time.time()
    print("counting train n-grams...", flush=True)
    uni, bi = stream_counts(Path(args.train))
    n = uni.sum()
    p1 = uni / n
    h1 = entropy(p1)

    bi2 = bi.reshape(VOCAB, VOCAB)
    row = bi2.sum(axis=1, keepdims=True)
    h2 = 0.0
    nz = row[:, 0] > 0
    pcond = bi2[nz] / row[nz]
    w = (row[nz, 0] / row.sum())
    for i in range(pcond.shape[0]):
        h2 += w[i] * entropy(pcond[i])
    print(f"  H1 {h1:.4f} nats, H2 {h2:.4f} nats ({time.time()-t0:.0f}s)", flush=True)

    val = np.asarray(np.memmap(args.val, dtype=np.uint16, mode="r")).astype(np.int64)
    val_ce_uni = float(-np.log(np.maximum(p1[val], 1e-12)).mean())

    # compressor floor on the validation text, per source
    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    comp = {}
    for src in ("ia", "wl"):
        docs = [d for d in split["val_ids"] if d.startswith(src + "_")]
        text = "".join(
            (Path(args.clean) / f"{d}.txt").read_text(encoding="utf-8",
                                                      errors="replace")
            for d in docs).encode("utf-8")
        packed = lzma.compress(text, preset=9)
        bpb = 8 * len(packed) / len(text)
        comp[src] = {"documents": len(docs), "bytes": len(text),
                     "xz_bytes": len(packed), "bits_per_byte": round(bpb, 4)}
        print(f"  xz {src}: {bpb:.3f} bits/byte", flush=True)

    val_bytes = sum(c["bytes"] for c in comp.values())
    val_tokens = len(val)
    bytes_per_token = val_bytes / val_tokens
    xz_all_bits = sum(8 * c["xz_bytes"] for c in comp.values())
    xz_nats_per_token = xz_all_bits / val_tokens * LN2

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    report = {
        "meta": {
            "script": "scripts/entropy_baseline.py",
            "train": args.train, "val": args.val,
            "vocab": VOCAB,
            "script_sha256": script_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "train_tokens": int(n),
        "unigram_entropy_nats": round(h1, 4),
        "bigram_conditional_entropy_nats": round(float(h2), 4),
        "val_tokens": val_tokens,
        "val_ce_under_train_unigram_nats": round(val_ce_uni, 4),
        "val_bytes_per_token": round(bytes_per_token, 4),
        "xz_val": comp,
        "xz_val_nats_per_token": round(xz_nats_per_token, 4),
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    print(f"\nunigram H (train)        : {h1:.4f} nats/token")
    print(f"bigram cond. H (train)   : {h2:.4f} nats/token")
    print(f"val CE under unigram     : {val_ce_uni:.4f} nats/token")
    print(f"xz on val text           : {xz_nats_per_token:.4f} nats/token "
          f"({8*sum(c['xz_bytes'] for c in comp.values())/val_bytes:.3f} bits/byte)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
