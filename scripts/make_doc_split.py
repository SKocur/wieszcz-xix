"""Hold out whole documents for validation, before tokenization.

The first build's validation window was a contiguous tail slice of the token stream —
documents could straddle the boundary, and the tail was not a random sample of
sources. This split is drawn the way the paper's limitations section prescribes:
whole documents, chosen by a seeded shuffle, stratified by source (Internet Archive /
Wolne Lektury) so the 0.75% WL share survives sampling, sized at ~1% of documents per
source. Documents on the exclusion list never enter the draw.

The report carries the complete train and validation id lists — the split is data,
not a procedure to re-run, and the tokenizer step consumes these lists verbatim.

    python scripts/make_doc_split.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = 1337


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--exclusions", default="metrics/exclusions_2026-08-03.json")
    ap.add_argument("--sidecar",
                    default="metrics/corpus_report_2026-08-03_raw.per_file.csv.gz",
                    help="per-file byte counts from the corpus report")
    ap.add_argument("--out", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()

    excluded = set(json.loads((REPO / args.exclusions)
                              .read_text(encoding="utf-8"))["ids"])
    nbytes: dict[str, int] = {}
    for row in csv.DictReader(gzip.open(REPO / args.sidecar, "rt")):
        nbytes[row["file"].removesuffix(".txt")] = int(row["bytes"])

    files = sorted(f.stem for f in Path(args.clean).glob("*.txt"))
    kept = [f for f in files if f not in excluded]

    rng = random.Random(SEED)
    val: list[str] = []
    per_source = {}
    for src in ("ia", "wl"):
        pool = [f for f in kept if f.split("_", 1)[0] == src]
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * args.val_frac))
        val.extend(pool[:n_val])
        per_source[src] = {
            "documents": len(pool), "val_documents": n_val,
            "bytes": sum(nbytes.get(f, 0) for f in pool),
            "val_bytes": sum(nbytes.get(f, 0) for f in pool[:n_val]),
        }
    val_set = set(val)
    train = [f for f in kept if f not in val_set]

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None

    report = {
        "meta": {
            "script": "scripts/make_doc_split.py",
            "seed": SEED, "val_frac": args.val_frac,
            "corpus_dir": str(args.clean),
            "exclusions": args.exclusions,
            "git_commit": git,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
        },
        "documents_total": len(files),
        "documents_excluded": len(files) - len(kept),
        "documents_kept": len(kept),
        "per_source": per_source,
        "train_documents": len(train),
        "val_documents": len(val),
        "val_bytes_share": round(
            sum(nbytes.get(f, 0) for f in val)
            / max(1, sum(nbytes.get(f, 0) for f in kept)), 5),
        "val_ids": sorted(val),
        "train_ids": train,
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    print(f"kept {len(kept):,} of {len(files):,} documents "
          f"({len(files) - len(kept):,} excluded)")
    for src, d in per_source.items():
        print(f"  {src:3s} {d['documents']:>8,} docs -> val {d['val_documents']:>6,} "
              f"({d['val_bytes']/1e6:.1f} MB of {d['bytes']/1e9:.2f} GB)")
    print(f"validation bytes share: {report['val_bytes_share']*100:.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
