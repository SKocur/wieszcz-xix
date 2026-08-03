"""Measure a frozen corpus build, before and after cleaning, with one classifier.

The paper's "raw -> cleaned" table is only honest if both columns come from the same
instruments, so this script is run twice — once on the frozen build as crawled (`--label
raw`) and once on the final cleaned build (`--label final`) — and every number it reports
is produced by detectors the pipeline already uses: the anachronism battery from
`scripts/anachronism_audit.py`, the OCR-suspicion classifier from `src/analyze_ocr.py`,
and the alpha-ratio definition of the cleaning gates in `src/clean_ocr.py`.

The report is deliberately overfull (counts per uploader, log-binned length histograms,
decade histograms, marker contexts): the paper will use a fraction of it, but the
fraction is chosen later, and a frozen build cannot be re-measured after it is replaced.
Aggregates go to `metrics/corpus_report_<freeze-date>_<label>.json`; per-file counters go
to a `.per_file.csv.gz` sidecar next to it, because 298k rows belong in a table, not in
a JSON anyone has to open.

Token counts here are estimates from the byte/token calibration measured on the 5.40B
build (3.38 B/tok); exact
counts exist only after tokenization and are recorded by that step, not this one.

    python scripts/corpus_report.py --label raw
    python scripts/corpus_report.py --label final --clean data/clean
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import platform
import re
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_ocr import reasons, strip_edges  # noqa: E402
from anachronism_audit import MODERN, PERIOD_CONTROL, YEAR, YEAR_CTX  # noqa: E402

BYTES_PER_TOKEN = 3.38          # measured on the 5.40B build with the shipped tokenizer
CONTEXT_CAP = 4                 # example snippets kept per modern marker

# Character classes are counted by regex substitution because the C scan is what makes
# a 23 GB pass tractable; alpha/(non-space) below is the same quantity clean_ocr's
# alpha_ratio() computes per line.
ALPHA = re.compile(r"[^\W\d_]")
DIGIT = re.compile(r"\d")
WS = re.compile(r"\s")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
DIACRITIC = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")

TLDS = {"pl", "gov", "org", "com", "net", "edu", "info", "de", "cz", "uk", "ru"}


def uploader(fname: str) -> str:
    """Bucket by digital-library uploader, recovered from the IA identifier.

    Library uploads carry their dLibra domain as a dotted prefix
    (`jbc.bj.uj.edu.pl.NDIGCZAS017303-...`); Google Books scans end in `goog`; the rest
    of Internet Archive has no usable prefix and is grouped as `ia-other`.
    """
    if fname.startswith("wl_"):
        return "wolne-lektury"
    ident = fname[3:].removesuffix(".txt")
    labels = ident.split(".")
    for i, lab in enumerate(labels):
        if lab.lower() in TLDS:
            # extend over compound suffixes (edu.pl, org.pl) rather than cutting at
            # the first TLD-looking label
            while i + 1 < len(labels) and labels[i + 1].lower() in TLDS:
                i += 1
            # leading accession numbers ("101727264.nlm.nih.gov", "47920560R.nlm...")
            # are per-item, not per-library — a real domain label carries no digits
            domain = [l for l in labels[: i + 1] if not any(c.isdigit() for c in l)]
            return ".".join(domain) if domain else "ia-other"
    if ident.endswith("goog"):
        return "google-books"
    return "ia-other"


def scan_files(paths: list[str]) -> dict:
    """One worker's pass over its share of the corpus: char classes, length counters,
    the anachronism battery, and per-file rows for the sidecar."""
    agg = {
        "docs": 0, "bytes": 0, "chars": 0, "words": 0,
        "alpha": 0, "digit": 0, "ws": 0, "cyrillic": 0, "diacritic": 0,
        "per_source": {},           # src -> [docs, bytes, chars, words]
        "per_uploader": Counter(),  # uploader -> bytes
        "uploader_docs": Counter(),
        "len_bytes": [],            # per-doc byte lengths, for percentiles
        "alpha_ratios": [],         # per-doc alpha/(non-space), the gate quantity
        "marker_docs": Counter(), "marker_occ": Counter(),
        "contexts": {t: [] for t in MODERN},
        "ctx_year_docs": 0, "ctx_year_decades": Counter(),
        "bare_year_docs": 0, "bare_year_decades": Counter(),
        "rows": [],
    }
    for p in paths:
        path = Path(p)
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        low = text.lower()
        n_bytes, n_chars = len(raw), len(text)
        n_words = len(text.split())
        n_alpha = n_chars - len(ALPHA.sub("", text))
        n_digit = n_chars - len(DIGIT.sub("", text))
        n_ws = n_chars - len(WS.sub("", text))
        n_cyr = n_chars - len(CYRILLIC.sub("", text))
        n_dia = n_chars - len(DIACRITIC.sub("", text))

        src = path.name.split("_", 1)[0]
        up = uploader(path.name)
        agg["docs"] += 1
        agg["bytes"] += n_bytes
        agg["chars"] += n_chars
        agg["words"] += n_words
        agg["alpha"] += n_alpha
        agg["digit"] += n_digit
        agg["ws"] += n_ws
        agg["cyrillic"] += n_cyr
        agg["diacritic"] += n_dia
        s = agg["per_source"].setdefault(src, [0, 0, 0, 0])
        s[0] += 1; s[1] += n_bytes; s[2] += n_chars; s[3] += n_words
        agg["per_uploader"][up] += n_bytes
        agg["uploader_docs"][up] += 1
        agg["len_bytes"].append(n_bytes)
        nonspace = n_chars - n_ws
        agg["alpha_ratios"].append(n_alpha / nonspace if nonspace else 0.0)

        modern_occ = 0
        for t in MODERN + PERIOD_CONTROL:
            n = low.count(t)
            if not n:
                continue
            agg["marker_docs"][t] += 1
            agg["marker_occ"][t] += n
            if t in MODERN:
                modern_occ += n
                if len(agg["contexts"][t]) < CONTEXT_CAP:
                    pos = low.find(t)
                    lo = max(0, pos - 60)
                    agg["contexts"][t].append(
                        {"doc": path.name,
                         "ctx": text[lo:pos + len(t) + 60].replace("\n", " ")})
        ctx_years = [a or b for a, b in YEAR_CTX.findall(low)]
        if ctx_years:
            agg["ctx_year_docs"] += 1
            for y in ctx_years:
                agg["ctx_year_decades"][int(y) // 10 * 10] += 1
        bare_years = YEAR.findall(low)
        if bare_years:
            agg["bare_year_docs"] += 1
            for y in bare_years:
                agg["bare_year_decades"][int(y) // 10 * 10] += 1

        agg["rows"].append((path.name, src, up, n_bytes, n_chars, n_words,
                            n_alpha, n_dia, n_cyr, n_digit,
                            modern_occ, len(ctx_years), len(bare_years)))
    return agg


def merge(total: dict, part: dict) -> None:
    for k in ("docs", "bytes", "chars", "words", "alpha", "digit", "ws",
              "cyrillic", "diacritic", "ctx_year_docs", "bare_year_docs"):
        total[k] += part[k]
    for src, v in part["per_source"].items():
        t = total["per_source"].setdefault(src, [0, 0, 0, 0])
        for i in range(4):
            t[i] += v[i]
    for k in ("per_uploader", "uploader_docs", "marker_docs", "marker_occ",
              "ctx_year_decades", "bare_year_decades"):
        total[k].update(part[k])
    for t, snips in part["contexts"].items():
        room = CONTEXT_CAP - len(total["contexts"][t])
        if room > 0:
            total["contexts"][t].extend(snips[:room])
    total["len_bytes"].extend(part["len_bytes"])
    total["alpha_ratios"].extend(part["alpha_ratios"])
    total["rows"].extend(part["rows"])


def distribution(values: list) -> dict:
    """Mean, percentiles p1-p99, and a log2-binned histogram."""
    vs = sorted(values)
    n = len(vs)
    if not n:
        return {}
    pct = {f"p{p}": vs[min(n - 1, int(n * p / 100))]
           for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    hist = Counter(v.bit_length() if isinstance(v, int) else int(v * 100)
                   for v in vs)
    if isinstance(vs[0], int):
        histogram = {f"[{2**(b-1)},{2**b})": hist[b] for b in sorted(hist)}
    else:
        histogram = {f"{b/100:.2f}": hist[b] for b in sorted(hist)}
    return {"n": n, "mean": sum(vs) / n, "min": vs[0], "max": vs[-1],
            **pct, "histogram": histogram}


def ocr_sample(files: list[Path], docs_per_source: int) -> dict:
    """audit_frozen_ocr.py's protocol on the text files: evenly spaced picks through
    each source block, so no uploader's run of scans dominates the sample."""
    out = {"documents_sampled_per_source": docs_per_source}
    by_src: dict[str, list[Path]] = {}
    for f in files:
        by_src.setdefault(f.name.split("_", 1)[0], []).append(f)
    for src, fs in sorted(by_src.items()):
        n = min(docs_per_source, len(fs))
        picks = [fs[int(i * (len(fs) - 1) / max(1, n - 1))] for i in range(n)]
        total = suspicious = 0
        by_reason = Counter()
        per_doc = []
        for f in picks:
            text = f.read_text(encoding="utf-8", errors="replace")
            dn = db = 0
            for tok in text.split():
                t = strip_edges(tok)
                if not t:
                    continue
                dn += 1
                r = reasons(t)
                if r:
                    db += 1
                    for reason in r:
                        by_reason[reason] += 1
            if dn:
                per_doc.append(db / dn)
                total += dn
                suspicious += db
        per_doc.sort()
        out[src] = {
            "documents": n, "words": total, "suspicious": suspicious,
            "rate": suspicious / total if total else 0.0,
            "by_reason": {k: v / total for k, v in by_reason.most_common()},
            "median_doc_rate": per_doc[len(per_doc) // 2] if per_doc else 0.0,
            "p90_doc_rate": per_doc[int(len(per_doc) * 0.9)] if per_doc else 0.0,
            "max_doc_rate": per_doc[-1] if per_doc else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"),
                    help="directory of corpus .txt files")
    ap.add_argument("--label", choices=("raw", "final"), required=True,
                    help="raw = frozen build before cleaning, final = after")
    ap.add_argument("--freeze-date", default="2026-08-03")
    ap.add_argument("--ocr-docs", type=int, default=400,
                    help="documents sampled per source for the OCR audit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: scan only the first N files")
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(Path(args.clean).glob("*.txt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .txt files under {args.clean}")
    print(f"{len(files):,} files under {args.clean}, {args.workers} workers")

    total = scan_files([])
    chunk = 2000
    chunks = [[str(f) for f in files[i:i + chunk]] for i in range(0, len(files), chunk)]
    with mp.Pool(args.workers) as pool:
        done = 0
        for part in pool.imap_unordered(scan_files, chunks):
            merge(total, part)
            done += 1
            if done % 20 == 0 or done == len(chunks):
                print(f"  scanned {total['docs']:,}/{len(files):,} docs "
                      f"({time.time() - t0:.0f}s)")

    print(f"OCR-suspicion sample: {args.ocr_docs} docs/source")
    ocr = ocr_sample(files, args.ocr_docs)

    est_tokens = total["bytes"] / BYTES_PER_TOKEN
    per_source = {
        src: {"documents": v[0], "bytes": v[1], "chars": v[2], "words": v[3],
              "est_tokens": round(v[1] / BYTES_PER_TOKEN),
              "share_bytes": round(v[1] / total["bytes"], 6)}
        for src, v in sorted(total["per_source"].items())}
    per_uploader = {
        up: {"documents": total["uploader_docs"][up], "bytes": b,
             "est_tokens": round(b / BYTES_PER_TOKEN),
             "share_bytes": round(b / total["bytes"], 6)}
        for up, b in total["per_uploader"].most_common()}

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    report = {
        "meta": {
            "script": "scripts/corpus_report.py",
            "label": args.label,
            "freeze_date": args.freeze_date,
            "corpus_dir": str(args.clean),
            "script_sha256": script_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "args": vars(args),
            "bytes_per_token_calibration": BYTES_PER_TOKEN,
            "elapsed_seconds": None,   # filled below
        },
        "counts": {
            "documents": total["docs"], "bytes": total["bytes"],
            "chars": total["chars"], "words": total["words"],
            "est_tokens": round(est_tokens),
            "per_source": per_source,
            "per_uploader": per_uploader,
        },
        "doc_length_bytes": distribution(total["len_bytes"]),
        "chars": {
            "alpha_share": total["alpha"] / total["chars"],
            "digit_share": total["digit"] / total["chars"],
            "whitespace_share": total["ws"] / total["chars"],
            "diacritic_share": total["diacritic"] / total["chars"],
            "cyrillic_share": total["cyrillic"] / total["chars"],
            "other_share": (total["chars"] - total["alpha"] - total["digit"]
                            - total["ws"]) / total["chars"],
            "per_doc_alpha_ratio": distribution(total["alpha_ratios"]),
        },
        "anachronisms": {
            "modern_markers": {t: {"docs": total["marker_docs"][t],
                                   "occurrences": total["marker_occ"][t]}
                               for t in MODERN},
            "period_control": {t: {"docs": total["marker_docs"][t],
                                   "occurrences": total["marker_occ"][t]}
                               for t in PERIOD_CONTROL},
            "docs_with_ctx_years": total["ctx_year_docs"],
            "ctx_year_decade_histogram":
                {str(d): total["ctx_year_decades"][d]
                 for d in sorted(total["ctx_year_decades"])},
            "docs_with_bare_years": total["bare_year_docs"],
            "bare_year_decade_histogram":
                {str(d): total["bare_year_decades"][d]
                 for d in sorted(total["bare_year_decades"])},
            "contexts": total["contexts"],
        },
        "ocr_suspicion": ocr,
        "duplication": {
            "status": "pending",
            "note": "measured by the dedup step for this build; "
                    "the final pass should reference its report",
        },
    }
    report["meta"]["elapsed_seconds"] = round(time.time() - t0, 1)

    stem = f"corpus_report_{args.freeze_date}_{args.label}"
    out = REPO / "metrics" / f"{stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    sidecar = REPO / "metrics" / f"{stem}.per_file.csv.gz"
    with gzip.open(sidecar, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "uploader", "bytes", "chars", "words",
                    "alpha", "diacritic", "cyrillic", "digit",
                    "modern_marker_occ", "ctx_years", "bare_years"])
        for row in sorted(total["rows"]):
            w.writerow(row)

    print(f"\n{total['docs']:,} documents, {total['bytes']/1e9:.2f} GB, "
          f"~{est_tokens/1e9:.2f}B tokens (est. at {BYTES_PER_TOKEN} B/tok)")
    for src, d in per_source.items():
        print(f"  {src:4s} {d['documents']:>8,} docs  {d['bytes']/1e9:>7.2f} GB  "
              f"{d['share_bytes']*100:5.2f}%")
    print(f"uploaders: {len(per_uploader)}; top: "
          + ", ".join(f"{u} {d['share_bytes']*100:.1f}%"
                      for u, d in list(per_uploader.items())[:5]))
    print(f"modern-marker hits: {sum(total['marker_occ'][t] for t in MODERN):,} "
          f"in {sum(total['marker_docs'][t] for t in MODERN):,} doc-hits; "
          f"ctx-year docs: {total['ctx_year_docs']:,}")
    print(f"wrote {out}")
    print(f"wrote {sidecar} ({total['docs']:,} rows)")


if __name__ == "__main__":
    main()
