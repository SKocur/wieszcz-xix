"""Recover per-source statistics from the frozen token file.

The frozen `tokens.bin` carries no filenames, but it does not need to. Documents were
written in filename order and the two sources use disjoint prefixes (`ia_`, `wl_`), so the
Internet-Archive documents occupy one contiguous block followed by the Wolne Lektury
documents. One boundary therefore splits the whole corpus, and counting document
terminators to that boundary gives the token split exactly rather than by estimate.

`--decode-tail` writes the Wolne Lektury block back out as text, which is what the OCR
audit needs: the clean half of the corpus is the detectors' false-positive floor and the
only part of the frozen build available locally in readable form.

    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --decode-sample 40 --out output/ocr_sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# The current frozen stream. The committed `metrics/ocr_audit_frozen.json` and the
# 1.63% the paper quotes "for history" were produced against the previous build's
# `data/clean/tokens_frozen_5.40B.bin`; re-running now measures what exists.
TOKENS = REPO / "data/tokens_frozen_6.69B.bin"
EOT = 0
SPLIT = REPO / "metrics/doc_split_2026-08-03.json"
CHUNK = 1 << 26            # 64M tokens per pass, ~128 MB resident


def eot_positions(path: Path) -> np.ndarray:
    """Absolute index of every document terminator, scanned in chunks so a 10.8 GB file
    never has to be resident."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    out = []
    for start in range(0, arr.size, CHUNK):
        block = np.asarray(arr[start:start + CHUNK])
        hits = np.flatnonzero(block == EOT)
        if hits.size:
            out.append(hits + start)
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def ia_document_count() -> int:
    """How many Internet-Archive documents the stream opens with, from the split file.

    This was a hardcoded constant carried over from an earlier build, which is exactly the
    way a stale number survives a rebuild. Reading it from the split that produced the
    stream means the two cannot disagree.
    """
    ids = json.loads(SPLIT.read_text())["train_ids"]
    return sum(1 for x in ids if x.startswith("ia_"))


def main() -> None:
    IA_DOCS = ia_document_count()
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default=str(TOKENS))
    p.add_argument("--decode-sample", type=int, default=0,
                   help="write this many documents from each source as text")
    p.add_argument("--out", default="output/corpus_sample")
    p.add_argument("--json", default="metrics/corpus_source_split.json")
    args = p.parse_args()

    path = Path(args.tokens)
    total = path.stat().st_size // 2
    print(f"{path.name}: {total:,} tokens")

    pos = eot_positions(path)
    print(f"document terminators: {len(pos):,}")

    # The boundary is the end of the last Internet-Archive document.
    if len(pos) < IA_DOCS:
        raise SystemExit(f"only {len(pos):,} terminators; expected at least {IA_DOCS:,}")
    boundary = int(pos[IA_DOCS - 1]) + 1

    ia_tokens = boundary
    wl_tokens = total - boundary
    report = {
        "tokens_total": int(total),
        "documents_total": int(len(pos)),
        "boundary_token_index": boundary,
        "internet_archive": {"documents": IA_DOCS, "tokens": int(ia_tokens),
                             "share": round(ia_tokens / total, 6)},
        "wolne_lektury": {"documents": int(len(pos) - IA_DOCS), "tokens": int(wl_tokens),
                          "share": round(wl_tokens / total, 6)},
    }
    for k in ("internet_archive", "wolne_lektury"):
        d = report[k]
        print(f"{k:18s} {d['documents']:>8,} docs  {d['tokens']:>15,} tokens  "
              f"{d['share']*100:5.2f}%  {d['tokens']/d['documents']:>8,.0f} tok/doc")

    out_json = REPO / args.json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")

    if args.decode_sample:
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                    str(REPO / "tokenizer/merges.txt"))
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        outdir = REPO / args.out
        (outdir / "ia").mkdir(parents=True, exist_ok=True)
        (outdir / "wl").mkdir(parents=True, exist_ok=True)

        # Evenly spaced rather than the first N: the first documents of each block share a
        # filename prefix and would be one uploader's scans.
        for label, lo, hi, sub in (("ia", 0, IA_DOCS, "ia"),
                                   ("wl", IA_DOCS, len(pos), "wl")):
            picks = np.linspace(lo + 1, hi - 1, args.decode_sample).astype(int)
            for i in picks:
                s = int(pos[i - 1]) + 1
                e = int(pos[i])
                text = tok.decode(np.asarray(arr[s:e]).tolist())
                (outdir / sub / f"{label}_{i:06d}.txt").write_text(text, encoding="utf-8")
            print(f"wrote {args.decode_sample} {label} documents to {outdir/sub}")


if __name__ == "__main__":
    main()
