"""Turn the anachronism audit and the collection sweep into the release exclusion list.

Three criteria, each with a different justification:

1. **Content markers** — any document the audit flagged on a Polish modern-vocabulary
   stem (internet, smartfon, komputer, telewiz, radiow, radjo, telefon komórk) or on
   "digitized by" boilerplate is excluded, false positives and all: over-exclusion costs
   a few period documents out of 220k, under-exclusion ships post-1918 text. The URL
   markers (`www.`, `http`) deliberately do NOT exclude — 87% of their hits are one
   library's watermark on legitimate period scans (RCIN), and they are reported as
   digitisation noise instead.
2. **Contextual years** — documents with at least `--ctx-threshold` matches of Polish
   date phrasing naming a year past 1919 ("r. 1936", "w roku 1925", "1936 r."). Unlike
   bare four-digit strings (numeric noise, flat decade histogram out to the 2090s), a
   date phrase is how genuinely post-1918 text talks about its own time.
3. **Provenance** — every document the IA collection sweep classified as
   `ia_modern_scan` (Internet Archive lending-library scans: where in-copyright modern
   editions live, and where the audit's one confirmed 1972 book came from). A modern
   book that avoids modern vocabulary defeats criteria 1-2; its collection membership
   does not lie. `--exclude-community` extends this to `opensource` community uploads,
   whose rights are simply unasserted.

    python scripts/make_exclusion_list.py --ctx-threshold 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONTENT = ["internet", "smartfon", "komputer", "telewiz", "radiow", "radjo",
           "telefon komórk", "digitized by"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="metrics/anachronism_audit.json")
    ap.add_argument("--sweep", default="metrics/ia_collection_sweep.json")
    ap.add_argument("--out", default="metrics/post1918_exclusions.json")
    ap.add_argument("--ctx-threshold", type=int, default=3,
                    help="exclude documents with at least this many contextual "
                         "post-1919 date phrases")
    ap.add_argument("--exclude-community", action="store_true",
                    help="also exclude 'opensource' community uploads")
    args = ap.parse_args()

    audit = json.loads((REPO / args.audit).read_text(encoding="utf-8"))
    sweep = json.loads((REPO / args.sweep).read_text(encoding="utf-8"))

    by_content: set[str] = set()
    by_ctx: set[str] = set()
    for doc_id, hits in audit["flags"].items():
        if any(t in hits for t in CONTENT):
            by_content.add(doc_id)
        elif hits.get("ctx_years", 0) >= args.ctx_threshold:
            by_ctx.add(doc_id)

    classes = {"ia_modern_scan"} | ({"community_upload"} if args.exclude_community
                                    else set())
    by_prov = {d for d, info in sweep["docs"].items() if info["class"] in classes}

    ids = sorted(by_content | by_ctx | by_prov)
    out = REPO / args.out
    out.write_text(json.dumps({
        "rule": {
            "content_markers": CONTENT,
            "ctx_year_threshold": args.ctx_threshold,
            "provenance_classes": sorted(classes),
            "inputs": [args.audit, args.sweep],
        },
        "by_content": len(by_content),
        "by_ctx_years": len(by_ctx),
        "by_provenance": len(by_prov),
        "ids": ids,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"content markers : {len(by_content):,}")
    print(f"ctx years >= {args.ctx_threshold}  : {len(by_ctx):,}")
    print(f"provenance      : {len(by_prov):,}")
    print(f"total excluded  : {len(ids):,} -> {out}")


if __name__ == "__main__":
    main()
