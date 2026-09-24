"""Hold out the final 1% of the corpus by filename order and report what lands there.

Document ids carry a source prefix (`ia_`, `wl_`), so sorting groups by source and a
positional tail is a sample of whichever source sorts last rather than of the corpus.
This measures how far that goes on the frozen build.

Both sides of the document-level split are recombined first: the split was drawn to fix
exactly this, so measuring on it would return the repair, not the defect.

Sizes are bytes. The ledger records bytes per document and tokens only in aggregate, so
byte shares stand in for token shares and the two differ by each source's
characters-per-token ratio.

A document counts as held out when its start falls past the boundary. The real idiom cuts
mid-document, so the one straddling the boundary is reported on its own rather than
folded into either side.

    .venv/bin/python3 scripts/positional_split_demo.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGERS = (REPO / "ledger/provenance_ledger_2026-08-03_train.csv.gz",
           REPO / "ledger/provenance_ledger_2026-08-03_val.csv.gz")

# The window the idiom holds out, as a fraction of the stream.
HELD_OUT = 0.01


def load_corpus() -> list[tuple[str, str, int]]:
    """Both ledger halves concatenated: every document in the frozen build."""
    docs: list[tuple[str, str, int]] = []
    for path in LEDGERS:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                docs.append((row["document_id"], row["source"], int(row["bytes"])))
    return docs


def positional_tail(docs: list[tuple[str, str, int]], fraction: float) -> dict:
    """Sort by document id, concatenate, and report the composition of the final `fraction`."""
    ordered = sorted(docs)
    total = sum(size for _, _, size in ordered)
    boundary = total * (1.0 - fraction)

    by_source_held: dict[str, dict[str, int]] = {}
    by_source_total: dict[str, dict[str, int]] = {}
    severed = None
    cursor = 0

    for doc_id, source, size in ordered:
        tally = by_source_total.setdefault(source, {"documents": 0, "bytes": 0})
        tally["documents"] += 1
        tally["bytes"] += size

        if cursor + size > boundary:
            if cursor < boundary and severed is None:
                severed = {"document_id": doc_id, "source": source, "bytes": size,
                           "bytes_before_boundary": int(boundary - cursor)}
            held = by_source_held.setdefault(source, {"documents": 0, "bytes": 0})
            held["documents"] += 1
            held["bytes"] += size
        cursor += size

    held_bytes = sum(v["bytes"] for v in by_source_held.values())
    return {
        "corpus": {"documents": len(ordered), "bytes": total,
                   "by_source": by_source_total},
        "held_out_window": {"fraction_requested": fraction,
                            "documents": sum(v["documents"] for v in by_source_held.values()),
                            "bytes": held_bytes,
                            "by_source": by_source_held},
        "severed_document": severed,
        "coverage": {
            source: {
                "share_of_source_held_out": by_source_held.get(source, {}).get("bytes", 0)
                / by_source_total[source]["bytes"],
                "share_of_window": by_source_held.get(source, {}).get("bytes", 0) / held_bytes,
            }
            for source in by_source_total
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fraction", type=float, default=HELD_OUT)
    ap.add_argument("--out", default=str(REPO / "metrics/positional_split_demo.json"))
    args = ap.parse_args()

    report = positional_tail(load_corpus(), args.fraction)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    corpus, window = report["corpus"], report["held_out_window"]
    print(f"corpus: {corpus['documents']:,} documents, {corpus['bytes']:,} bytes")
    for source, tally in sorted(corpus["by_source"].items()):
        print(f"   {source:20} {tally['documents']:>8,} docs  {tally['bytes']:>15,} bytes")
    print(f"\nlast {args.fraction:.0%} by filename order: "
          f"{window['documents']:,} documents, {window['bytes']:,} bytes")
    for source, tally in sorted(window["by_source"].items()):
        cov = report["coverage"][source]
        print(f"   {source:20} {tally['documents']:>8,} docs  {tally['bytes']:>15,} bytes"
              f"   {cov['share_of_window']:6.1%} of window"
              f"   {cov['share_of_source_held_out']:6.1%} of that source")
    if report["severed_document"]:
        sev = report["severed_document"]
        print(f"\nsevered at the boundary: {sev['document_id']} ({sev['source']}), "
              f"{sev['bytes_before_boundary']:,} of {sev['bytes']:,} bytes on the training side")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
