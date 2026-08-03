"""Turn the audits into the exclusion list: markers, dates, provenance, duplicates.

Four arms, and a corroboration matrix learned by reading the corpus (2026-08-03):

1. **Strong markers** — vocabulary and boilerplate that hand-verified sweeps showed
   cannot occur in period text (hitler, gestapo, nkwd, kołchoz, faszyzm, międzywojenny,
   "druga wojna światowa", mikrofilm, smartfon, "telefon komórkowy", "digitized by").
   Any hit excludes, false positives and all: over-exclusion costs a few documents out
   of 298k, under-exclusion ships post-1918 text. Edition apparatus (copyright, ISBN
   with digits, "wszelkie prawa zastrzeżone", domena publiczna) excludes the same way
   but exempts Wolne Lektury files, whose own colophon is part of an accepted source.
2. **Noisy markers** (internet, komputer, telewiz, www., http) — real modernity *and*
   frequent OCR garble ("Internet acja" is a 1917 interpelacja, "telewizytę" a 1903
   tę wizytę, and most www./http hits are one library's watermark on period scans).
   These exclude only when the post-reform orthography share corroborates (>=0.5):
   garbled period print scores near 0, genuinely modern text near 1.
3. **Contextual years** — Polish date phrasing naming a post-1919 year. Excludes at
   >=2 phrases when orthography corroborates (>=0.8), at >=3 with any non-period
   orthography (>=0.2). At >=3 with period orthography (<0.2) the document goes to the
   review list instead: that profile is OCR misreading old dates as 19xx (an
   Orgelbrand encyclopedia carries 87 such phrases and is legitimate).
4. **Provenance and duplicates** — every `ia_modern_scan` lending-library document
   (where in-copyright modern editions live; collection membership does not lie), and
   every document the dedup pass marked dropped (the longest member of its cluster
   stays).

Ids in the output are bare document ids (no .txt), matching the provenance ledger.

    python scripts/make_exclusion_list.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STRONG = ["smartfon", "telefon komórk", "digitized by",
          "hitler", "gestapo", "nkwd", "kołchoz", "kolchoz",
          "faszyzm", "faszyst", "międzywojenn", "miedzywojenn", "mikrofilm",
          "druga_wojna_światowa"]
APPARATUS = ["copyright", "wszelkie prawa zastrze", "isbn_number", "domena_publiczna"]
NOISY = ["internet", "komputer", "telewiz", "www.", "http"]


def orto_share(hits: dict) -> float:
    om, op = hits.get("orto_modern", 0), hits.get("orto_period", 0)
    return om / (om + op) if om + op else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="metrics/anachronism_audit_2026-08-03.json")
    ap.add_argument("--sweep", default="metrics/ia_collection_sweep_2026-08-03.json")
    ap.add_argument("--dedup", default="metrics/dedup_2026-08-03.json")
    ap.add_argument("--out", default="metrics/exclusions_2026-08-03.json")
    args = ap.parse_args()

    audit = json.loads((REPO / args.audit).read_text(encoding="utf-8"))
    sweep = json.loads((REPO / args.sweep).read_text(encoding="utf-8"))
    dedup = json.loads((REPO / args.dedup).read_text(encoding="utf-8"))

    by_strong: set[str] = set()
    by_apparatus: set[str] = set()
    by_noisy: set[str] = set()
    by_ctx: set[str] = set()
    review: dict[str, dict] = {}

    for doc_id, hits in audit["flags"].items():
        doc = doc_id.removesuffix(".txt")
        share = orto_share(hits)
        ctx = hits.get("ctx_years", 0)
        if any(t in hits for t in STRONG):
            by_strong.add(doc)
        if not doc.startswith("wl_") and any(t in hits for t in APPARATUS):
            by_apparatus.add(doc)
        if share >= 0.5 and any(t in hits for t in NOISY):
            by_noisy.add(doc)
        if (ctx >= 2 and share >= 0.8) or (ctx >= 3 and share >= 0.2):
            by_ctx.add(doc)
        elif ctx >= 3:
            review[doc] = {"ctx_years": ctx, "orto_share": round(share, 3),
                           "reason": "date phrases with period orthography — "
                                     "OCR-misread old dates?"}

    by_prov = {d for d, info in sweep["docs"].items()
               if info["class"] == "ia_modern_scan"}
    by_dup = {d.removesuffix(".txt") for d in dedup["dropped"]}

    ids = sorted(by_strong | by_apparatus | by_noisy | by_ctx | by_prov | by_dup)
    review = {d: r for d, r in review.items() if d not in set(ids)}

    out = REPO / args.out
    out.write_text(json.dumps({
        "rule": {
            "strong_markers": STRONG,
            "apparatus_markers": APPARATUS + ["(wl_ exempt)"],
            "noisy_markers": NOISY + ["(require orto_share >= 0.5)"],
            "ctx_years": "excl. >=2 with orto>=0.8, or >=3 with orto>=0.2; "
                         ">=3 with orto<0.2 -> review",
            "provenance_classes": ["ia_modern_scan"],
            "duplicates": "dedup dropped list (longest member of each cluster kept)",
            "inputs": [args.audit, args.sweep, args.dedup],
        },
        "by_strong_markers": len(by_strong),
        "by_apparatus": len(by_apparatus),
        "by_noisy_corroborated": len(by_noisy),
        "by_ctx_years": len(by_ctx),
        "by_provenance": len(by_prov),
        "by_duplicates": len(by_dup),
        "excluded_total": len(ids),
        "review_count": len(review),
        "review": review,
        "ids": ids,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"strong markers        : {len(by_strong):,}")
    print(f"apparatus (no wl_)    : {len(by_apparatus):,}")
    print(f"noisy + orto >= 0.5   : {len(by_noisy):,}")
    print(f"ctx years corroborated: {len(by_ctx):,}")
    print(f"provenance            : {len(by_prov):,}")
    print(f"duplicates            : {len(by_dup):,}")
    print(f"excluded total        : {len(ids):,}")
    print(f"review (not excluded) : {len(review):,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
