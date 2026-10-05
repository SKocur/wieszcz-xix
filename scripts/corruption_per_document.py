"""Per-document OCR corruption over the released corpus.

Table 4 of the paper measures the four character-level detectors of `src/analyze_ocr.py`
on 2,000 documents per source. The corpus ships without that number per document, so a
reader who wants to set a quality threshold has nothing to set it on, and the paper's
own remedy (filter by line, keep the document) stays a recommendation. This scores every
document of the split with the same detectors and writes one row per document, plus a
report with the population rates, the per-source distribution, and what each threshold on
the rate would cost in documents and bytes.

Rates are percentages of whitespace words, as in the paper. The clean half (Wolne Lektury)
is the detectors' false-positive floor; corruption in the OCR half is the difference.
Wolne Lektury documents are scored after the same colophon pass tokenization applies, so
every row describes the released text and its byte count matches the provenance ledger.

    python scripts/corruption_per_document.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from analyze_ocr import reasons, strip_edges  # noqa: E402
from clean_ocr import is_anachronism  # noqa: E402

SPLIT = REPO / "metrics/doc_split_2026-08-03.json"
CLEAN = REPO / "data/clean"
DATE = "2026-10-05"
OUT_CSV = REPO / f"metrics/corruption_per_document_{DATE}.csv.gz"
OUT_JSON = REPO / f"metrics/corruption_per_document_{DATE}.json"

REASONS = ("symbol", "digit_mix", "midcaps", "no_vowel")
THRESHOLDS_PCT = (1, 2, 3, 5, 10)
MIN_WORDS = 50  # a rate on fewer words is noise, the same floor analyze_ocr.py uses


def score(doc_id: str) -> tuple:
    path = CLEAN / f"{doc_id}.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    if doc_id.startswith("wl_"):
        text = "\n".join(ln for ln in text.split("\n") if not is_anachronism(ln))
    words = suspicious = 0
    by_reason = Counter()
    for raw in text.split():
        tok = strip_edges(raw)
        if not tok:
            continue
        words += 1
        r = reasons(tok)
        if r:
            suspicious += 1
            for x in r:
                by_reason[x] += 1
    return (doc_id, len(text.encode("utf-8")), words, suspicious,
            *(by_reason[x] for x in REASONS))


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def main() -> None:
    split = json.loads(SPLIT.read_text())
    ids = sorted(split["train_ids"] + split["val_ids"])
    missing = [i for i in ids if not (CLEAN / f"{i}.txt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} split documents are not in {CLEAN}; "
                         f"first: {missing[0]}")

    with Pool(os.cpu_count()) as pool:
        rows = list(pool.imap(score, ids, chunksize=256))

    with gzip.open(OUT_CSV, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["document_id", "source", "bytes", "words", "suspicious", *REASONS,
                    "suspicious_pct"])
        for doc_id, nbytes, words, susp, *reason_counts in rows:
            pct = 100 * susp / words if words else 0.0
            w.writerow([doc_id, doc_id.split("_", 1)[0], nbytes, words, susp,
                        *reason_counts, f"{pct:.4f}"])

    by_source: dict[str, dict] = {}
    for src in ("ia", "wl"):
        mine = [r for r in rows if r[0].startswith(src + "_")]
        words = sum(r[2] for r in mine)
        susp = sum(r[3] for r in mine)
        nbytes = sum(r[1] for r in mine)
        rates = sorted(100 * r[3] / r[2] for r in mine if r[2] >= MIN_WORDS)
        thresholds = {}
        for t in THRESHOLDS_PCT:
            kept = [r for r in mine if r[2] < MIN_WORDS or 100 * r[3] / r[2] <= t]
            thresholds[str(t)] = {
                "documents_kept": len(kept),
                "documents_kept_share": round(len(kept) / len(mine), 4),
                "bytes_kept_share": round(sum(r[1] for r in kept) / nbytes, 4),
            }
        by_source[src] = {
            "documents": len(mine),
            "bytes": nbytes,
            "words": words,
            "suspicious": susp,
            "rate_pct": round(100 * susp / words, 4),
            "by_reason_pct": {x: round(100 * sum(r[4 + k] for r in mine) / words, 4)
                              for k, x in enumerate(REASONS)},
            "per_document_pct": {
                "documents_scored": len(rates),
                "median": round(quantile(rates, 0.5), 3),
                "p90": round(quantile(rates, 0.9), 3),
                "p99": round(quantile(rates, 0.99), 3),
                "max": round(rates[-1], 3) if rates else 0.0,
            },
            "kept_at_threshold_pct": thresholds,
        }
    floor = by_source["wl"]["rate_pct"]
    report = {
        "meta": {"script": "scripts/corruption_per_document.py", "split": SPLIT.name,
                 "corpus": str(CLEAN.relative_to(REPO)), "date": DATE,
                 "detectors": "src/analyze_ocr.py", "min_words_for_rate": MIN_WORDS,
                 "rows": OUT_CSV.name},
        "documents": len(rows),
        "by_source": by_source,
        "false_positive_floor_pct": floor,
        "ia_corruption_above_floor_pct": round(by_source["ia"]["rate_pct"] - floor, 4),
    }
    OUT_JSON.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for src, s in by_source.items():
        d = s["per_document_pct"]
        print(f"{src}: {s['documents']:,} docs, rate {s['rate_pct']:.2f}%, "
              f"median {d['median']:.2f}, p90 {d['p90']:.2f}, max {d['max']:.2f}")
    print(f"floor {floor:.2f}%, IA above floor {report['ia_corruption_above_floor_pct']:.2f}%")
    print(f"-> {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
