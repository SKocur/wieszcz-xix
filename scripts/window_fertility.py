"""Bytes per token over the windows the dense protocol actually scores.

The paper converts nats per token to bits per byte with the held-out split's exact
ratio, but the protocol scores 8,192 windows of it, not every token. The strictly right
divisor is the bytes of the scored targets over their count. This measures it, with the
window rule of `src/eval_val.evaluate`, so the paper can state how far the two divisors
are apart instead of leaving a reader to wonder.

Token byte lengths come from reversing the GPT-2 byte-to-unicode mapping over
`tokenizer/vocab.json`, so the count is of UTF-8 bytes, the unit the conversion uses.

    python scripts/window_fertility.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
VAL = REPO / "data/val_2026-08-03.bin"
VOCAB = REPO / "tokenizer/vocab.json"
EVAL = REPO / "metrics/eval_by_source_2026-08-17_fullspan.json"
OUT = REPO / "metrics/window_fertility_2026-08-17.json"
BLOCK = 1024
WINDOWS = 8192


def unicode_to_bytes() -> dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    u2b = unicode_to_bytes()
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    byte_len = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    for tok, i in vocab.items():
        if all(ch in u2b for ch in tok):
            byte_len[i] = len(tok)

    data = np.memmap(VAL, dtype=np.uint16, mode="r")
    max_windows = (len(data) - 1) // BLOCK
    starts = np.linspace(0, max_windows - 1, min(WINDOWS, max_windows),
                         dtype=np.int64) * BLOCK
    target_bytes = sum(int(byte_len[data[s + 1:s + 1 + BLOCK]].sum()) for s in starts)
    target_tokens = len(starts) * BLOCK
    windows = target_bytes / target_tokens
    split = int(byte_len[data].sum()) / len(data)

    ev = json.loads(EVAL.read_text())
    bpb = {}
    for rung, res in ev["by_subset"]["full"].items():
        ce = res["cross_entropy_nats"]
        bpb[rung] = {"cross_entropy_nats": ce,
                     "bpb_windows": ce / (math.log(2) * windows)}

    out = {"meta": {"script": "scripts/window_fertility.py", "val": VAL.name,
                    "tokenizer": VOCAB.name, "eval": EVAL.name, "block": BLOCK,
                    "windows": int(len(starts))},
           "scored_windows": {"bytes": target_bytes, "tokens": target_tokens,
                              "bytes_per_token": windows},
           "split_decoded": {"bytes_per_token": split},
           "bpb": bpb}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"scored windows {windows:.4f} bytes/token, whole split {split:.4f} -> {OUT.name}")


if __name__ == "__main__":
    main()
