"""Scan the frozen corpus for post-1918 content markers.

The 1918 bound is enforced by source *metadata*, and metadata lies: a 2026-08-02 spot
check found a 1972 Wańkowicz book carrying date=1900 in its Internet Archive record,
modern-edition apparatus attached to a public-domain novel, and "Digitized by the
Internet Archive in 2025" boilerplate surviving the cleaner. This audits the *content*:
every document is checked for words that did not exist before 1918, for modern web and
digitisation boilerplate, and for standalone year strings 1920-2099.

The battery is curated against the corpus, not against intuition. A 2026-08-03 sweep
with hand-read contexts rejected the plausible-sounding candidates: *stalin* (the 1914
press reviews a brochure by "K. Stalin"), *czołg* ("Rozwój i znaczenie czołgów",
Warszawa 1918), *lotnisko* (a 1911 flight report), *samolot* (Żuławski 1911; "Siedem
samolotów włoskich nad Wiedniem", 1918), *bolszewik/sowiet* (current affairs of
1917-18), *milicja obywatelska* (attested 1886 and the 1915 Warsaw militia),
*rzeczpospolita ludowa* in any wording (the 1918 press reports daily on the
Ukrainian People's Republic, and "polska rzeczpospolita ludowa" itself is the
language of 1905-07 revolutionary agitation and of Daszyński's November-1918
government — 26 years before the PRL), and *pzpr/nazis/zsrr/rwpg* (dominated by OCR
capital-soup and word-fragment noise). The period-legitimate ones now sit in the
control battery. The same sweep confirmed low-noise markers the battery had been
missing — interwar/PRL vocabulary (hitler, gestapo, nkwd, kołchoz, faszyzm,
międzywojenny) and edition apparatus (copyright, wszelkie prawa zastrzeżone,
ISBN followed by digits, domena publiczna, mikrofilm) — apparatus that
line-level cleaning removes from *lines* but which marks the whole *document* as a
modern edition. *radjo/radiow* moved from markers to controls: wireless telegraphy is
period ("Stacja radjotelegraficzna", 1915) and broadcast-era hits ride along with the
other markers.

Two document-level features are recorded on flagged documents for the exclusion rule
to corroborate with — they never flag or exclude on their own. The post-reform
orthography share — words in -cja/-sja/-zja against period -cya/-sya/-zya —
separates modern editions from Galician print (a 1983 theatre programme and a 1972
Wańkowicz score 0.88-1.00, Galician papers 0.00-0.02), but the full-corpus run
showed Kryński's -ja spelling is the *norm* in 1905-18 Congress-Poland press (72k
documents ≥0.8 modern share), so it can only corroborate, never convict. Its real
value is directional: a high share backs up a marker or ctx-year hit, and a *period*
share protects against OCR misreading mediaeval dates as 19xx (an Orgelbrand volume
shows "w r. 1989" for 1289 — 87 false ctx-years on a legitimate document). Filename
years ≥1919 are recorded the same way: for IA-native identifiers they are
publication years (tygodnikillustro1923unse), for bc.radom-style numeric ids they
are catalogue numbers, so they feed the provenance sweep instead of excluding
directly.

Reads either the parquet shards built by `build_hf_dataset.py` (release audit) or a
directory of cleaned `.txt` files (corpus audit before tokenization).

Period-legitimate near-anachronisms are counted as a control battery but never flag a
document — their presence in quantity is what a genuine pre-1918 corpus looks like.

Year strings are reported two ways and left unflagged, because the first measurement
showed bare four-digit matches to be numeric noise (their decade histogram is flat out to
the 2090s — prices and catalogue numbers, not dates). The contextual pattern requires
Polish date phrasing around the number ("r. 1936", "w roku 1925", "1936 r."), which bare
noise cannot satisfy; its per-document counts are what an exclusion rule should consult.

    python scripts/anachronism_audit.py --txt-dir data/clean --out metrics/anachronism_audit_2026-08-03.json
    python scripts/anachronism_audit.py            # parquet shards of the HF build
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import re
import socket
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# substring match on lowercased text, so Polish inflection is covered by the stem
MODERN = ["internet", "smartfon", "komputer", "telewiz", "telefon komórk",
          "digitized by", "www.", "http",
          "hitler", "gestapo", "nkwd", "kołchoz", "kolchoz", "faszyzm", "faszyst",
          "międzywojenn", "miedzywojenn",
          "copyright", "wszelkie prawa zastrze", "mikrofilm"]
PERIOD_CONTROL = ["telefon", "automobil", "aeroplan", "samochod", "samochód",
                  "samolot", "lotnisk", "czołg", "bolszewi", "sowiet",
                  "radjo", "radiow"]
# phrase markers that need more than a substring
MODERN_RE = {
    "isbn_number": re.compile(r"\bisbn[\s.:]{0,3}\d[\d\w'’–-]{4,}", re.I),
    "druga_wojna_światowa":
        re.compile(r"(?:drugiej|drugą|druga|\bii)\s+wojn\w*\s+światow", re.I),
    "domena_publiczna": re.compile(r"domen\w*\s+publiczn", re.I),
}
YEAR = re.compile(r"(?<!\d)(?:19[2-9]\d|20\d\d)(?!\d)")
YEAR_CTX = re.compile(r"(?:\b(?:r\.|roku|rok|w\s+r\.)\s{1,2}(19[2-9]\d|20\d\d)(?!\d)"
                      r"|(?<!\d)(19[2-9]\d|20\d\d)\s{1,2}r\.)")

# 1918-reform orthography: -cja/-sja/-zja words vs period -cya/-sya/-zya
ORTO_MODERN = re.compile(r"\b\w+[csz]j[aeięąo]\w*\b", re.I)
ORTO_PERIOD = re.compile(r"\b\w+[csz]y[aeięąo]\w*\b", re.I)
FNAME_YEAR = re.compile(r"(?<!\d)(1[5-9]\d\d|20[0-2]\d)(?!\d)")

CONTEXT_CAP = 8


def new_agg() -> dict:
    return {"total_docs": 0, "total_chars": 0,
            "docs_hit": Counter(), "occurrences": Counter(),
            "contexts": {t: [] for t in MODERN + list(MODERN_RE)},
            "flagged": {}}


def scan_document(agg: dict, doc_id: str, text: str) -> None:
    agg["total_docs"] += 1
    agg["total_chars"] += len(text)
    low = text.lower()
    hits = {}

    for t in MODERN + PERIOD_CONTROL:
        n = low.count(t)
        if not n:
            continue
        agg["docs_hit"][t] += 1
        agg["occurrences"][t] += n
        if t in MODERN:
            hits[t] = n
            if len(agg["contexts"][t]) < CONTEXT_CAP:
                pos = low.find(t)
                s = max(0, pos - 60)
                agg["contexts"][t].append(
                    {"doc": doc_id,
                     "ctx": text[s:pos + len(t) + 60].replace("\n", " ")})
    for name, rx in MODERN_RE.items():
        ms = rx.findall(low)
        if not ms:
            continue
        agg["docs_hit"][name] += 1
        agg["occurrences"][name] += len(ms)
        hits[name] = len(ms)
        if len(agg["contexts"][name]) < CONTEXT_CAP:
            m = rx.search(low)
            s = max(0, m.start() - 60)
            agg["contexts"][name].append(
                {"doc": doc_id, "ctx": text[s:m.end() + 60].replace("\n", " ")})

    years = YEAR.findall(low)
    if years:
        hits["year_strings"] = len(years)
        hits["year_examples"] = sorted(set(years))[:8]
    ctx = [a or b for a, b in YEAR_CTX.findall(low)]
    if ctx:
        hits["ctx_years"] = len(ctx)
        hits["ctx_year_examples"] = sorted(set(ctx))[:8]

    if hits:
        # corroborating features only, never flags of their own: the full-corpus run
        # showed Kryński's -ja spelling is the NORM in 1905-18 Congress-Poland press
        # (72k documents ≥0.8 modern share), so orthography cannot convict alone
        hits["orto_modern"] = len(ORTO_MODERN.findall(text))
        hits["orto_period"] = len(ORTO_PERIOD.findall(text))
        fy = [int(y) for y in FNAME_YEAR.findall(doc_id) if int(y) >= 1919]
        if fy:
            hits["fname_years"] = fy
        agg["flagged"][doc_id] = hits


def merge(total: dict, part: dict) -> None:
    total["total_docs"] += part["total_docs"]
    total["total_chars"] += part["total_chars"]
    total["docs_hit"].update(part["docs_hit"])
    total["occurrences"].update(part["occurrences"])
    for t, snips in part["contexts"].items():
        room = CONTEXT_CAP - len(total["contexts"][t])
        if room > 0:
            total["contexts"][t].extend(snips[:room])
    total["flagged"].update(part["flagged"])


def scan_txt_files(paths: list[str]) -> dict:
    agg = new_agg()
    for p in paths:
        path = Path(p)
        text = path.read_text(encoding="utf-8", errors="replace")
        scan_document(agg, path.name, text)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "dataset/wieszcz-xix-corpus/data"),
                    help="directory of train-*.parquet shards")
    ap.add_argument("--txt-dir", default=None,
                    help="audit a directory of cleaned .txt files instead of parquet")
    ap.add_argument("--out", default="metrics/anachronism_audit.json")
    ap.add_argument("--workers", type=int, default=8,
                    help="txt-dir mode only; parquet decoding is already sequential-fast")
    args = ap.parse_args()

    t0 = time.time()
    total = new_agg()

    if args.txt_dir:
        files = sorted(Path(args.txt_dir).glob("*.txt"))
        if not files:
            raise SystemExit(f"no .txt files under {args.txt_dir}")
        print(f"{len(files):,} files under {args.txt_dir}, {args.workers} workers")
        chunk = 2000
        chunks = [[str(f) for f in files[i:i + chunk]]
                  for i in range(0, len(files), chunk)]
        with mp.Pool(args.workers) as pool:
            done = 0
            for part in pool.imap_unordered(scan_txt_files, chunks):
                merge(total, part)
                done += 1
                if done % 20 == 0 or done == len(chunks):
                    print(f"  scanned {total['total_docs']:,}/{len(files):,} docs "
                          f"flagged={len(total['flagged']):,} ({time.time()-t0:.0f}s)")
    else:
        import pyarrow.parquet as pq
        shards = sorted(glob.glob(str(Path(args.data) / "train-*.parquet")))
        if not shards:
            raise SystemExit(f"no parquet shards under {args.data}")
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512,
                                                           columns=["id", "text"]):
                for doc_id, text in zip(batch.column("id").to_pylist(),
                                        batch.column("text").to_pylist()):
                    scan_document(total, doc_id, text)
            print(f"scanned {Path(path).name}  docs={total['total_docs']:,}  "
                  f"flagged={len(total['flagged']):,}")

    flagged = total["flagged"]
    marker_names = MODERN + list(MODERN_RE)
    modern_docs = {d for d, h in flagged.items() if any(t in h for t in marker_names)}
    year_counts = [h["year_strings"] for h in flagged.values() if "year_strings" in h]
    year_hist = {f">={k}": sum(1 for n in year_counts if n >= k) for k in (1, 2, 3, 5, 10)}
    ctx_counts = [h["ctx_years"] for h in flagged.values() if "ctx_years" in h]
    ctx_hist = {f">={k}": sum(1 for n in ctx_counts if n >= k) for k in (1, 2, 3, 5, 10)}
    ratios = [(h["orto_modern"], h["orto_period"]) for h in flagged.values()]
    orto_hist = {f">={t}": sum(1 for m, p in ratios
                               if m >= 20 and m / (m + p or 1) >= t)
                 for t in (0.5, 0.8, 0.9)}
    fname_docs = sum(1 for h in flagged.values() if "fname_years" in h)

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None

    report = {
        "meta": {
            "script": "scripts/anachronism_audit.py",
            "mode": "txt-dir" if args.txt_dir else "parquet",
            "source": args.txt_dir or args.data,
            "git_commit": git,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "battery": {"modern": MODERN, "modern_re": list(MODERN_RE),
                    "period_control": PERIOD_CONTROL},
        "total_docs": total["total_docs"],
        "total_chars": total["total_chars"],
        "docs_hit": dict(total["docs_hit"]),
        "occurrences": dict(total["occurrences"]),
        "docs_with_modern_terms": len(modern_docs),
        "docs_with_year_strings": year_hist,
        "docs_with_ctx_years": ctx_hist,
        "flagged_docs_orto_modern_share": orto_hist,
        "flagged_docs_with_fname_years": fname_docs,
        "contexts": total["contexts"],
        "flags": flagged,
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{total['total_docs']:,} documents, {total['total_chars']/1e9:.2f}G chars")
    print(f"modern-term documents : {len(modern_docs):,}")
    print(f"year-string documents : {year_hist}")
    print(f"ctx-year documents    : {ctx_hist}")
    print(f"orto modern (m>=20)   : {orto_hist}")
    print(f"flagged with fname yr : {fname_docs:,}")
    for t in marker_names:
        if total["docs_hit"][t]:
            print(f"  {t:22s} {total['docs_hit'][t]:>7,} docs  "
                  f"{total['occurrences'][t]:>8,} occ")
    print("period control battery:")
    for t in PERIOD_CONTROL:
        print(f"  {t:22s} {total['docs_hit'][t]:>7,} docs  "
              f"{total['occurrences'][t]:>8,} occ")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
