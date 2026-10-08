"""Measure what fetch-time cleaning removed from the Internet Archive text.

The freeze holds text already cleaned by `prepare_data.clean_ocr`, so the freeze alone
cannot say how much that pass took out. The crawl kept the text as fetched under
`data/raw/` for most documents. For every ledger document that still has its raw file,
this replays `clean_ocr` one step at a time and attributes every removed byte to one of
four causes: the scanning-service boilerplate cut, the alphabetic-share line test, the
watermark, URL and page-number line test, and whitespace handling (line-ending and
blank-line normalisation, rejoining of wrapped lines, collapsing of runs of spaces). The
four sum to the difference between the raw text and the replayed result, and the
replayed result is compared with the cleaned file on disk.

The alphabetic test is selective by construction, so two things are counted on the lines
it drops against all lines: digit characters, and four-digit years after 1918, which the
temporal audit later looks for in the cleaned text.

    .venv/bin/python3 scripts/cleaning_effect.py --out metrics/cleaning_effect_2026-10-06.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from prepare_data import (LINE_MIN_ALPHA, LINE_MIN_WORDS, _is_junk_line,  # noqa: E402
                          alpha_ratio, strip_google_boilerplate,
                          strip_low_quality_lines, unwrap_lines)

FIELDS = ("raw_bytes", "boilerplate_bytes", "low_alpha_bytes", "junk_bytes",
          "whitespace_bytes", "replay_bytes", "clean_bytes", "replay_equals_clean",
          "lines", "low_alpha_lines", "digits", "low_alpha_digits",
          "years_post1918", "low_alpha_years_post1918")
DIGIT = re.compile(r"\d")
YEAR_POST1918 = re.compile(r"(?<!\d)(?:1919|19[2-9]\d|20[0-2]\d)(?!\d)")
RAW: Path | None = None
CLEAN: Path | None = None


def _init(raw: str, clean: str) -> None:
    global RAW, CLEAN
    RAW, CLEAN = Path(raw), Path(clean)


def size(text: str) -> int:
    return len(text.encode("utf-8"))


def measure(doc_id: str) -> tuple[int, ...] | None:
    raw_path = RAW / f"{doc_id}.txt"
    if not raw_path.exists():
        return None
    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    t1 = re.sub(r"\r\n?", "\n", raw)
    t1b = strip_google_boilerplate(t1)
    t2 = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", t1b))
    lines = t2.split("\n")
    low = [l for l in lines if not _is_junk_line(l)
           and len(l.split()) >= LINE_MIN_WORDS and alpha_ratio(l) < LINE_MIN_ALPHA]
    t3 = strip_low_quality_lines(t2)
    t4 = unwrap_lines(t3)
    t5 = "\n".join(l for l in t4.split("\n") if not _is_junk_line(l))
    t6 = re.sub(r"[ \t]{2,}", " ", t5).strip() + "\n"

    low_bytes = sum(size(l) + 1 for l in low)
    junk_bytes = (size(t2) - size(t3) - low_bytes) + (size(t4) - size(t5))
    whitespace = ((size(raw) - size(t1)) + (size(t1b) - size(t2))
                  + (size(t3) - size(t4)) + (size(t5) - size(t6)))
    clean_bytes = (CLEAN / f"{doc_id}.txt").stat().st_size
    count = lambda pattern, ls: sum(len(pattern.findall(l)) for l in ls)  # noqa: E731
    return (size(raw), size(t1) - size(t1b), low_bytes, junk_bytes, whitespace,
            size(t6), clean_bytes, int(size(t6) == clean_bytes),
            len(lines), len(low), count(DIGIT, lines), count(DIGIT, low),
            count(YEAR_POST1918, lines), count(YEAR_POST1918, low))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/cleaning_effect.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    docs: list[str] = []
    for path in sorted(REPO.glob(args.ledger)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            docs += [r["document_id"] for r in csv.DictReader(f)
                     if r["source"] == "internet_archive"]
    docs.sort()

    totals = dict.fromkeys(FIELDS, 0)
    measured = 0
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(str(REPO / args.raw), str(REPO / args.clean))) as pool:
        for i, row in enumerate(pool.imap_unordered(measure, docs, chunksize=200), 1):
            if row is not None:
                measured += 1
                for k, v in zip(FIELDS, row):
                    totals[k] += v
            if i % 20000 == 0:
                print(f"{i:,}/{len(docs):,}  {time.time()-t0:4.0f}s", flush=True)

    raw = totals["raw_bytes"]
    pct = lambda a, b: round(100 * a / b, 3)  # noqa: E731
    removed = raw - totals["replay_bytes"]
    parts = ("boilerplate_bytes", "low_alpha_bytes", "junk_bytes", "whitespace_bytes")
    if sum(totals[k] for k in parts) != removed:
        raise SystemExit("the four causes do not sum to the bytes removed")
    out = {
        "meta": {
            "script": "scripts/cleaning_effect.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "ledger": args.ledger,
            "line_min_words": LINE_MIN_WORDS, "line_min_alpha": LINE_MIN_ALPHA,
            "year_pattern": YEAR_POST1918.pattern,
        },
        "internet_archive_documents": len(docs),
        "documents_with_raw": measured,
        "share_documents_with_raw": pct(measured, len(docs)),
        "totals": totals,
        "removed_share_bytes": pct(removed, raw),
        "removed_share_bytes_on_disk": pct(raw - totals["clean_bytes"], raw),
        "replay_equals_clean_share_documents": pct(totals["replay_equals_clean"], measured),
        "low_alpha_share_bytes": pct(totals["low_alpha_bytes"], raw),
        "junk_share_bytes": pct(totals["junk_bytes"], raw),
        "boilerplate_share_bytes": pct(totals["boilerplate_bytes"], raw),
        "whitespace_share_bytes": pct(totals["whitespace_bytes"], raw),
        "low_alpha_share_lines": pct(totals["low_alpha_lines"], totals["lines"]),
        "low_alpha_share_digits": pct(totals["low_alpha_digits"], totals["digits"]),
        "low_alpha_share_years_post1918": pct(totals["low_alpha_years_post1918"],
                                              totals["years_post1918"]),
    }
    path = REPO / args.out
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))
    print(f"{time.time()-t0:.0f}s; wrote {path}")


if __name__ == "__main__":
    main()
