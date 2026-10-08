"""Read the Wolne Lektury record of every transcribed document for who wrote and translated it.

The exclusion rule dates Wolne Lektury documents by the library's epoch, and the epoch
describes the work. Two kinds of document can sit under a pre-1918 epoch and still hold
later text: a translation made after 1918, and a work its author wrote after 1918 while
the library files the author under an earlier epoch. This fetches the metadata header of
each document's source XML (creator, translator, language, the edition transcribed, the
rights note, and the year the text entered the public domain) and tabulates both kinds.
A document whose rights note is a free licence instead of a public-domain statement is a
translation by a living translator, so it is both recent text and text that is not in
the public domain.

The year of entry into the public domain is the last death among author and translator
plus 71, so it says who was still alive after 1918. It does not date the text. The
report therefore lists candidates, with their bytes, and a hand-kept list of documents
whose original is known to postdate 1918.

    python scripts/wl_translation_sweep.py --out metrics/wl_translations_2026-10-06.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
XML = "https://wolnelektury.pl/media/book/xml/{slug}.xml"
HEADER_BYTES = 12000
PD_TERM = 71
PUBLIC_DOMAIN_NOTE = re.compile(r"domena publiczna|\bzm\.", re.IGNORECASE)
FIELD = re.compile(r"<dc:([\w.]+)[^>]*>([^<]*)</dc:[\w.]+>")
# Originals first published after 1918, checked by title against the publication record:
# the volumes of Proust's cycle after the first appeared between 1919 and 1927.
KNOWN_POST1918_ORIGINAL = re.compile(
    r"^wl_(?:proust-utracona|sodoma-i-gomora-|strona-guermantes-|uwieziona-"
    r"|w-cieniu-zakwitajacych-dziewczat-|czas-odnaleziony)")


def fetch(slug: str) -> dict:
    req = Request(XML.format(slug=slug), headers={
        "User-Agent": "wieszcz-xix translation sweep", "Range": f"bytes=0-{HEADER_BYTES}"})
    for attempt in range(4):
        try:
            with urlopen(req, timeout=40) as r:
                head = r.read(HEADER_BYTES + 1).decode("utf-8", errors="replace")
            break
        except HTTPError as e:
            if e.code == 404:
                return {"slug": slug, "error": "404"}
            time.sleep(2 * (attempt + 1))
        except (URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    else:
        return {"slug": slug, "error": "unreachable"}
    fields: dict[str, list[str]] = {}
    for name, value in FIELD.findall(head):
        if value.strip():
            fields.setdefault(name, []).append(" ".join(value.split()))
    pd = fields.get("date.pd", [""])[0]
    return {
        "slug": slug,
        "creators": fields.get("creator", []),
        "translators": fields.get("contributor.translator", []),
        "epochs": fields.get("subject.period", []),
        "language": fields.get("language", [""])[0],
        "source": fields.get("source", [""])[0],
        "rights": fields.get("rights", [""])[0],
        "public_domain_from": int(pd) if pd.isdigit() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/wl_translations.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    size: dict[str, int] = {}
    corpus_bytes = 0
    for path in sorted(REPO.glob(args.ledger)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                corpus_bytes += int(r["bytes"])
                if r["source"] == "wolne_lektury":
                    size[r["document_id"]] = int(r["bytes"])
    docs = sorted(size)
    print(f"{len(docs):,} Wolne Lektury documents in the ledger", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda d: {"document_id": d, "bytes": size[d],
                                           **fetch(d[3:])}, docs))

    wl_bytes = sum(size.values())
    last_death_after_1918 = 1918 + PD_TERM
    classes: dict[str, list[dict]] = {k: [] for k in (
        "unreadable", "translation", "translation_translator_alive_after_1918",
        "original_author_alive_after_1918", "known_post1918_original",
        "free_licence", "certain_post1918", "possibly_post1918", "not_polish")}
    translators: Counter = Counter()
    for rec in records:
        if rec.get("error"):
            classes["unreadable"].append(rec)
            continue
        late = (rec["public_domain_from"] or 0) > last_death_after_1918
        if rec["translators"]:
            classes["translation"].append(rec)
            for t in rec["translators"]:
                translators[t] += 1
            if late:
                classes["translation_translator_alive_after_1918"].append(rec)
        elif late:
            classes["original_author_alive_after_1918"].append(rec)
        known = bool(KNOWN_POST1918_ORIGINAL.match(rec["document_id"]))
        licensed = not PUBLIC_DOMAIN_NOTE.search(rec["rights"])
        if known:
            classes["known_post1918_original"].append(rec)
        if licensed:
            classes["free_licence"].append(rec)
        if known or licensed:
            classes["certain_post1918"].append(rec)
        if known or licensed or late:
            classes["possibly_post1918"].append(rec)
        if rec["language"] != "pol":
            classes["not_polish"].append(rec)

    pct = lambda a, b: round(100 * a / b, 4)  # noqa: E731

    def tally(rows: list[dict]) -> dict:
        b = sum(r["bytes"] for r in rows)
        return {"documents": len(rows), "bytes": b,
                "share_wl_bytes": pct(b, wl_bytes), "share_corpus_bytes": pct(b, corpus_bytes)}

    out = {
        "meta": {
            "script": "scripts/wl_translation_sweep.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(), "source": XML, "ledger": args.ledger,
            "public_domain_term_years": PD_TERM,
            "known_post1918_original_pattern": KNOWN_POST1918_ORIGINAL.pattern,
        },
        "wl_documents": len(docs), "wl_bytes": wl_bytes, "corpus_bytes": corpus_bytes,
        "summary": {k: tally(v) for k, v in classes.items()},
        "translators": dict(translators.most_common()),
        "certain_post1918": [r["document_id"] for r in classes["certain_post1918"]],
        "free_licence": {r["document_id"]: r["rights"] for r in classes["free_licence"]},
        "not_polish": {r["document_id"]: r["language"] for r in classes["not_polish"]},
        "documents": records,
    }
    path = REPO / args.out
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(dict(translators.most_common(25)), ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
