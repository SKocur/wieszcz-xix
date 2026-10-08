"""Count post-reform and period spellings in every document of the cleaned build.

The orthography share of the temporal audit (words in -cja/-sja/-zja against period
-cya/-sya/-zya) was only kept for the documents that audit flagged, which is a biased
subset. This counts both forms in every ledger document with the audit's own patterns,
so the share can be aggregated by any catalogue field: place of publication, decade,
document type.

    .venv/bin/python3 scripts/orto_per_document.py --out metrics/orto_per_document_2026-10-06
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from anachronism_audit import ORTO_MODERN, ORTO_PERIOD  # noqa: E402

CLEAN: Path | None = None


def _init(clean: str) -> None:
    global CLEAN
    CLEAN = Path(clean)


def count(doc_id: str) -> tuple[str, int, int]:
    text = (CLEAN / f"{doc_id}.txt").read_text(encoding="utf-8", errors="replace")
    return doc_id, len(ORTO_MODERN.findall(text)), len(ORTO_PERIOD.findall(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/orto_per_document",
                    help="output stem; writes <stem>.csv.gz and <stem>.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    docs: list[str] = []
    for path in sorted(REPO.glob(args.ledger)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            docs += [r["document_id"] for r in csv.DictReader(f)]
    docs.sort()

    stem = REPO / args.out
    out_csv = stem.with_name(stem.name + ".csv.gz")
    t0 = time.time()
    modern = period = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(str(REPO / args.clean),)) as pool, \
            gzip.open(out_csv, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "orto_modern", "orto_period"])
        for i, (doc_id, m, p) in enumerate(pool.imap(count, docs, chunksize=200), 1):
            w.writerow([doc_id, m, p])
            modern += m
            period += p
            if i % 20000 == 0:
                print(f"{i:,}/{len(docs):,}  {time.time()-t0:4.0f}s", flush=True)

    summary = {
        "meta": {
            "script": "scripts/orto_per_document.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "ledger": args.ledger,
            "patterns": {"orto_modern": ORTO_MODERN.pattern, "orto_period": ORTO_PERIOD.pattern},
        },
        "documents": len(docs),
        "orto_modern": modern,
        "orto_period": period,
        "share_modern": round(modern / (modern + period), 4),
        "rows": out_csv.name,
        "rows_sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest(),
    }
    stem.with_name(stem.name + ".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(docs):,} documents, post-reform share {summary['share_modern']}, "
          f"{time.time()-t0:.0f}s; wrote {out_csv.name}")


if __name__ == "__main__":
    main()
