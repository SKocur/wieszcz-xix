"""Write one metadata row per corpus document, keyed to the provenance ledger.

The released text rows carry an identifier and nothing a reader can slice by. This joins
the ledger with the catalogue records pulled by `ia_catalogue_metadata.py` and writes,
for every document of the cleaned build: the split it is in, the digitising provider
(from the identifier), and for Internet Archive documents the catalogue year, place of
publication, the state that held the place under the borders of 1914, the catalogue
type and the title.

Two columns record the rights position per document. `rights_basis` says what the
public-domain status rests on: the holding library's own statement in the catalogue
record's rights field; a weaker assertion by the scanning institution (the archive's
not-in-copyright status, which follows the scanner's jurisdiction, or a Public Domain
Mark link); Wolne Lektury's public-domain note; or nothing. A Wolne Lektury translation
published under a free licence is marked as such, since it is not in the public domain.
`creator_died_in_term` marks the documents `creator_death_years.py` found, whose
catalogue creator died within the last 70 years. For Wolne Lektury documents the
language, author, translator and rights note of the library's record are carried over
from `wl_translation_sweep.py`, which is the attribution a free licence asks for, with
its two findings: whether the text is certainly later than 1918, and whether its author
or translator lived past it.
Every catalogue field is the cataloguing library's claim.

    python scripts/document_metadata.py --out ledger/document_metadata_2026-10-06.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from corpus_composition import place_of, rights_of, state_of, type_of, year_of
from corpus_report import uploader

REPO = Path(__file__).resolve().parent.parent
COLUMNS = ("document_id", "source", "source_identifier", "bytes", "split", "provider",
           "year", "place", "state_1914", "type", "title", "language", "rights_basis",
           "creator_died_in_term", "wl_certain_post1918", "wl_possibly_post1918",
           "wl_author", "wl_translator", "wl_rights_note")
SCANNER_ASSERTION = re.compile(r"not.in.copyright|public domain|unaware of any", re.IGNORECASE)


def rights_basis(cat: dict) -> str:
    stated = rights_of(cat.get("rights"))
    if stated == "public_domain_statement":
        return "library_public_domain_statement"
    if ("publicdomain" in str(cat.get("licenseurl") or "")
            or SCANNER_ASSERTION.search(str(cat.get("possible-copyright-status") or ""))):
        return "scanning_institution_assertion"
    return "other_statement" if stated == "other_statement" or cat.get("licenseurl") else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ledger/document_metadata.csv.gz")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_{split}.csv.gz")
    ap.add_argument("--catalogue", default="metrics/ia_catalogue_metadata_2026-10-06.jsonl.gz")
    ap.add_argument("--death-years", default="metrics/creator_death_years_2026-10-06.json")
    ap.add_argument("--wl", default="metrics/wl_translations_2026-10-06.json")
    args = ap.parse_args()

    catalogue: dict[str, dict] = {}
    with gzip.open(REPO / args.catalogue, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            catalogue[row["document_id"]] = row
    deaths = json.loads((REPO / args.death_years).read_text(encoding="utf-8"))
    in_term = {d["document_id"] for d in deaths["documents_creator_died_in_term"]}

    wl = json.loads((REPO / args.wl).read_text(encoding="utf-8"))
    wl_record = {d["document_id"]: d for d in wl["documents"]}
    wl_certain = set(wl["certain_post1918"])
    wl_possible = {d["document_id"] for d in wl["documents"]
                   if d["document_id"] in wl_certain
                   or (d["public_domain_from"] or 0) > 1918 + wl["meta"]["public_domain_term_years"]}

    docs: Counter = Counter()
    nbytes: Counter = Counter()
    rows = 0
    out = REPO / args.out
    with gzip.open(out, "wt", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=COLUMNS)
        w.writeheader()
        for split in ("train", "val"):
            with gzip.open(REPO / args.ledger.format(split=split), "rt", encoding="utf-8") as f:
                for led in csv.DictReader(f):
                    doc = led["document_id"]
                    rec = {**led, "split": split, "provider": uploader(doc),
                           "creator_died_in_term": int(doc in in_term),
                           "wl_certain_post1918": int(doc in wl_certain),
                           "wl_possibly_post1918": int(doc in wl_possible),
                           "wl_author": "", "wl_translator": "", "wl_rights_note": ""}
                    if led["source"] == "internet_archive":
                        cat = catalogue.get(doc, {})
                        place = place_of(cat.get("coverage"))
                        title = cat.get("title")
                        rec.update(
                            year=year_of(cat) or "", place=place or "",
                            state_1914=state_of(place), type=type_of(cat.get("source")),
                            title=" ".join(str(title).split()) if title else "",
                            language="pol",
                            rights_basis=rights_basis(cat))
                    else:
                        rec.update(year="", place="", state_1914="", type="", title="",
                                   language=wl_record[doc]["language"],
                                   wl_author="; ".join(wl_record[doc]["creators"]),
                                   wl_translator="; ".join(wl_record[doc]["translators"]),
                                   wl_rights_note=wl_record[doc]["rights"],
                                   rights_basis="free_licence" if doc in wl["free_licence"]
                                   else "wolne_lektury_public_domain_note")
                    w.writerow(rec)
                    rows += 1
                    for key in (rec["rights_basis"],
                                "creator_died_in_term" if rec["creator_died_in_term"] else None):
                        if key:
                            docs[key] += 1
                            nbytes[key] += int(led["bytes"])
                    nbytes["all"] += int(led["bytes"])

    pct = lambda a, b: round(100 * a / b, 3)  # noqa: E731
    summary = {
        "meta": {
            "script": "scripts/document_metadata.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(), "catalogue": args.catalogue,
            "death_years": args.death_years, "wl": args.wl, "table": args.out,
            "table_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "columns": list(COLUMNS),
        },
        "documents": rows,
        "bytes": nbytes["all"],
        "by_rights_basis": {k: {"documents": docs[k], "bytes": nbytes[k],
                                "share_documents": pct(docs[k], rows),
                                "share_bytes": pct(nbytes[k], nbytes["all"])}
                            for k in sorted(docs)},
    }
    path = out.with_name(out.name.replace(".csv.gz", ".json"))
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=1))
    print(f"wrote {out} and {path}")


if __name__ == "__main__":
    main()
