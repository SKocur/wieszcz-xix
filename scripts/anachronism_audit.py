"""Scan the frozen corpus for post-1918 content markers.

The 1918 bound is enforced by source *metadata*, and metadata lies: a 2026-08-02 spot
check found a 1972 Wańkowicz book carrying date=1900 in its Internet Archive record,
modern-edition apparatus attached to a public-domain novel, and "Digitized by the
Internet Archive in 2025" boilerplate surviving the cleaner. This audits the *content*:
every document is checked for words that did not exist before 1918, for modern web and
digitisation boilerplate, and for standalone year strings 1920-2099.

Reads the parquet shards built by `build_hf_dataset.py` rather than decoding the token
stream: that build is verified byte-for-byte against the provenance ledger, so the two
are the same text, and the parquet scan runs in minutes instead of hours.

Period-legitimate near-anachronisms (telefon, automobil, aeroplan, samochód) are counted
as a control battery but never flag a document — their presence in quantity is what a
genuine pre-1918 corpus looks like.

Year strings are reported two ways and left unflagged, because the first measurement
showed bare four-digit matches to be numeric noise (their decade histogram is flat out to
the 2090s — prices and catalogue numbers, not dates). The contextual pattern requires
Polish date phrasing around the number ("r. 1936", "w roku 1925", "1936 r."), which bare
noise cannot satisfy; its per-document counts are what an exclusion rule should consult.

    python scripts/anachronism_audit.py
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# substring match on lowercased text, so Polish inflection is covered by the stem
MODERN = ["internet", "smartfon", "komputer", "telewiz", "radiow", "radjo",
          "telefon komórk", "digitized by", "www.", "http"]
PERIOD_CONTROL = ["telefon", "automobil", "aeroplan", "samochod", "samochód"]
YEAR = re.compile(r"(?<!\d)(?:19[2-9]\d|20\d\d)(?!\d)")
YEAR_CTX = re.compile(r"(?:\b(?:r\.|roku|rok|w\s+r\.)\s{1,2}(19[2-9]\d|20\d\d)(?!\d)"
                      r"|(?<!\d)(19[2-9]\d|20\d\d)\s{1,2}r\.)")


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "dataset/wieszcz-xix-corpus/data"),
                    help="directory of train-*.parquet shards")
    ap.add_argument("--out", default="metrics/anachronism_audit.json")
    ap.add_argument("--contexts", type=int, default=8,
                    help="example contexts kept per modern term")
    args = ap.parse_args()

    terms = MODERN + PERIOD_CONTROL
    docs_hit = {t: 0 for t in terms}
    occurrences = {t: 0 for t in terms}
    contexts: dict[str, list] = {t: [] for t in MODERN}
    flagged: dict[str, dict] = {}
    total_docs = total_chars = 0

    shards = sorted(glob.glob(str(Path(args.data) / "train-*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.data}")

    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512,
                                                       columns=["id", "text"]):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for doc_id, text in zip(ids, texts):
                total_docs += 1
                total_chars += len(text)
                low = text.lower()
                hits = {}
                for t in terms:
                    n = low.count(t)
                    if not n:
                        continue
                    docs_hit[t] += 1
                    occurrences[t] += n
                    if t in MODERN:
                        hits[t] = n
                        if len(contexts[t]) < args.contexts:
                            pos = low.find(t)
                            s = max(0, pos - 60)
                            snippet = text[s:pos + len(t) + 60].replace("\n", " ")
                            contexts[t].append({"doc": doc_id, "ctx": snippet})
                years = YEAR.findall(low)
                if years:
                    hits["year_strings"] = len(years)
                    hits["year_examples"] = sorted(set(years))[:8]
                ctx = [a or b for a, b in YEAR_CTX.findall(low)]
                if ctx:
                    hits["ctx_years"] = len(ctx)
                    hits["ctx_year_examples"] = sorted(set(ctx))[:8]
                if hits:
                    flagged[doc_id] = hits
        print(f"scanned {Path(path).name}  docs={total_docs:,}  flagged={len(flagged):,}")

    modern_docs = {d for d, h in flagged.items() if any(t in h for t in MODERN)}
    year_counts = [h["year_strings"] for h in flagged.values() if "year_strings" in h]
    year_hist = {f">={k}": sum(1 for n in year_counts if n >= k) for k in (1, 2, 3, 5, 10)}
    ctx_counts = [h["ctx_years"] for h in flagged.values() if "ctx_years" in h]
    ctx_hist = {f">={k}": sum(1 for n in ctx_counts if n >= k) for k in (1, 2, 3, 5, 10)}

    report = {
        "total_docs": total_docs,
        "total_chars": total_chars,
        "docs_hit": docs_hit,
        "occurrences": occurrences,
        "docs_with_modern_terms": len(modern_docs),
        "docs_with_year_strings": year_hist,
        "docs_with_ctx_years": ctx_hist,
        "contexts": contexts,
        "flags": flagged,
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{total_docs:,} documents, {total_chars/1e9:.2f}G chars")
    print(f"modern-term documents : {len(modern_docs):,}")
    print(f"year-string documents : {year_hist}")
    print(f"ctx-year documents    : {ctx_hist}")
    for t in MODERN:
        if docs_hit[t]:
            print(f"  {t:16s} {docs_hit[t]:>7,} docs  {occurrences[t]:>8,} occ")
    print("period control battery:")
    for t in PERIOD_CONTROL:
        print(f"  {t:16s} {docs_hit[t]:>7,} docs  {occurrences[t]:>8,} occ")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
