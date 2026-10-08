"""Describe the Internet Archive documents by catalogue type, decade and place.

Reads the provenance ledger (sizes), the catalogue records pulled by
`ia_catalogue_metadata.py` and the spelling counts of `orto_per_document.py`, and
tabulates the Internet Archive documents of the cleaned build:

- by type, from the catalogue's `source` field (periodical issue, book or pamphlet,
  ephemera, unspecified);
- by decade, from the catalogue year, with the share of documents whose identifier
  carries a year that agrees with it;
- by state of publication under the borders of 1914, from the catalogue's `coverage`
  field through the place table below;
- by the holding library's rights statement;
- the -cja share (-cja against -cya spellings) by state, by period, by decade for the
  largest places of publication, and by periodical title, which is what the paper's
  account of its orthography measure rests on.

Every field is the cataloguing library's claim. Places outside the table are counted as
unclassified and reported, so the table's reach is visible.

    python scripts/corpus_composition.py --out metrics/corpus_composition_2026-10-06.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

AUSTRIA = """Kraków Lwów Wiedeń Tarnów Nowy_Sącz Przemyśl Sambor Rzeszów Nowy_Targ Zakopane
Podgórze Stanisławów Krynica-Zdrój Miejsce_Piastowe Kołomyja Bruckenthal Chrzanów Sanok
Cieszyn Wadowice Gorlice Drohobycz Stryj Jarosław Tarnopol Tarnobrzeg Bochnia Jasło Krosno
Wieliczka Biała Żywiec Brody Złoczów"""
RUSSIA = """Warszawa Warsaw Wilno Sankt_Petersburg Petersburg Kijów Dąbrowa_Górnicza
Sosnowiec Łódź Kalisz Płock Lublin Kielce Włocławek Radom Piotrków_Trybunalski
Nowo-Radomsk Opoczno Zamość Miechów Opatów Olkusz Wierzbnik Częstochowa Włoszczowa
Krasnystaw Suwałki Łomża Siedlce Mińsk Grodno Żytomierz Odessa Moskwa Ryga"""
GERMANY = """Poznań Leszno Grodzisk Bochum Katowice Bytom Niemieckie_Piekary_Piekary_Śląskie
Bytom-Rozbark Piekary_Śląskie Bydgoszcz Berlin Lipsk Olsztyn Drezno Gniezno Kościan
Pelplin Wrocław Sopot Toruń Gdańsk Grudziądz Chełmno Inowrocław Opole Racibórz"""
ELSEWHERE = """Paryż Paris Londyn Bruksela Turyn Kurytyba Bendlikon Genewa Poitiers Rzym
Zurych"""
STATE_OF = {}
for _state, _names in (("austria_hungary", AUSTRIA), ("russian_empire", RUSSIA),
                       ("german_empire", GERMANY), ("elsewhere", ELSEWHERE)):
    for _n in _names.split():
        STATE_OF[_n.replace("_", " ")] = _state

COUNTRY_CODE = re.compile(r"[a-z]{2}")
US_PLACE = re.compile(r", (?:[A-Z]{2}|Ill)\.?$")
ID_YEAR = re.compile(r"(?<!\d)(1[89]\d\d)(?!\d)")
LIBRARY_PREFIX = re.compile(r"^(?:[a-z0-9-]+\.)+pl\.(.+)$")
TITLE_END = re.compile(r"\.\s|\s:\s|\s\d{4}|, nr|\[")
# A place-decade cell is reported as a point only above this many documents, and a
# periodical title enters the per-title distribution only above this many counted forms.
MIN_CELL_DOCUMENTS = 50
MIN_TITLE_FORMS = 1000
PLOTTED_PLACES = 3
NAMED_TITLES = (("Lwów", "Gazeta Lwowska"), ("Lwów", "Dziennik Polski"),
                ("Warszawa", "Kurjer Warszawski"), ("Kraków", "Czas"))


def place_of(coverage) -> str | None:
    """First place name in a `coverage` value, without brackets and country codes."""
    if not coverage:
        return None
    for part in re.split(r"\s*;\s*", str(coverage)):
        s = re.sub(r"\(.*?\)", "", part)
        s = re.sub(r"[\[\]?]", "", s).strip(" .,")
        if s and not COUNTRY_CODE.fullmatch(s):
            return None if s == "miejsce nieznane" else s
    return None


def title_of(title) -> str:
    """A periodical's title without its issue: the catalogue writes `Czas. 1888, nr 100`."""
    s = str(title or "").strip().lstrip("[").strip()
    return TITLE_END.split(s, maxsplit=1)[0].strip(" ].")


def state_of(place: str | None) -> str:
    if place is None:
        return "no_place"
    if US_PLACE.search(place):
        return "elsewhere"
    return STATE_OF.get(place, "unclassified")


def type_of(source) -> str:
    values = source if isinstance(source, list) else [source]
    s = " ".join(str(v) for v in values if v).lower()
    if "czasopism" in s or "gazet" in s or "journal" in s or "jednodniówka" in s:
        return "periodical"
    if any(k in s for k in ("książk", "book", "broszura", "encyklopedia", "odbitka",
                            "nadbitka", "starodruk")):
        return "book"
    if any(k in s for k in ("nekrolog", "afisz", "druk ulotny", "odezwa", "dżs")):
        return "ephemera"
    return "unspecified"


def rights_of(rights) -> str:
    s = str(rights or "").lower()
    if "domena publiczna" in s or "public domain" in s:
        return "public_domain_statement"
    return "other_statement" if s else "none"


def year_of(row: dict) -> int | None:
    for v in (row.get("year"), row.get("date")):
        m = ID_YEAR.search(str(v)) if v else None
        if m:
            return int(m.group(1))
    return None


def identifier_year(identifier: str) -> int | None:
    m = LIBRARY_PREFIX.match(identifier)
    years = set(ID_YEAR.findall(m.group(1))) if m else set()
    return int(years.pop()) if len(years) == 1 else None


class Tally:
    def __init__(self) -> None:
        self.docs: Counter = Counter()
        self.bytes: Counter = Counter()

    def add(self, key, size: int) -> None:
        self.docs[key] += 1
        self.bytes[key] += size

    def table(self, n_docs: int, n_bytes: int) -> dict:
        return {str(k): {"documents": self.docs[k], "bytes": self.bytes[k],
                         "share_documents": round(100 * self.docs[k] / n_docs, 2),
                         "share_bytes": round(100 * self.bytes[k] / n_bytes, 2)}
                for k in sorted(self.docs, key=str)}


def share(modern: int, period: int) -> dict:
    total = modern + period
    return {"orto_modern": modern, "orto_period": period,
            "share_modern": round(modern / total, 4) if total else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/corpus_composition.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--catalogue", default="metrics/ia_catalogue_metadata_2026-10-06.jsonl.gz")
    ap.add_argument("--orto", default="metrics/orto_per_document_2026-10-06.csv.gz")
    args = ap.parse_args()

    size: dict[str, int] = {}
    for path in sorted(REPO.glob(args.ledger)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source"] == "internet_archive":
                    size[r["document_id"]] = int(r["bytes"])
    with gzip.open(REPO / args.orto, "rt", encoding="utf-8") as f:
        orto = {r["document_id"]: (int(r["orto_modern"]), int(r["orto_period"]))
                for r in csv.DictReader(f)}

    types, decades, states, rights, places = Tally(), Tally(), Tally(), Tally(), Tally()
    orto_state: dict = defaultdict(lambda: [0, 0])
    orto_state_period: dict = defaultdict(lambda: [0, 0])
    orto_place_period: dict = defaultdict(lambda: [0, 0])
    orto_place_decade: dict = defaultdict(lambda: [0, 0, 0])
    orto_title: dict = defaultdict(lambda: [0, 0])
    cell_titles: dict = defaultdict(Counter)
    comparable = agree = 0
    seen = 0
    with gzip.open(REPO / args.catalogue, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc = row["document_id"]
            if doc not in size:
                continue
            seen += 1
            b = size[doc]
            year = year_of(row)
            place = place_of(row.get("coverage"))
            state = state_of(place)
            kind = type_of(row.get("source"))
            types.add(kind, b)
            decades.add(year // 10 * 10 if year else "undated", b)
            states.add(state, b)
            places.add(place or "(none)", b)
            rights.add(rights_of(row.get("rights")), b)
            id_year = identifier_year(row["identifier"])
            if id_year and year:
                comparable += 1
                agree += id_year == year
            m, p = orto.get(doc, (0, 0))
            period = "undated" if not year else ("from_1905" if year >= 1905 else "before_1905")
            for bucket, key in ((orto_state, state), (orto_state_period, (state, period)),
                                (orto_place_period, (place, period))):
                bucket[key][0] += m
                bucket[key][1] += p
            title = title_of(row.get("title")) if kind == "periodical" else None
            if title:
                orto_title[(place, title)][0] += m
                orto_title[(place, title)][1] += p
            if year:
                cell = orto_place_decade[(place, year // 10 * 10)]
                cell[0] += m
                cell[1] += p
                cell[2] += 1
                cell_titles[(place, year // 10 * 10)][title or "(no title)"] += m + p

    n_docs, n_bytes = len(size), sum(size.values())
    if seen != n_docs:
        raise SystemExit(f"catalogue covers {seen:,} of {n_docs:,} ledger documents")
    state_table = states.table(n_docs, n_bytes)
    placed = n_docs - states.docs["no_place"]
    top = [p for p, _ in places.docs.most_common(6) if p != "(none)"][:5]

    def cell(place: str, decade: int, m: int, p: int, n: int) -> dict:
        name, forms = cell_titles[(place, decade)].most_common(1)[0]
        return {**share(m, p), "documents": n, "top_title": name,
                "top_title_share_forms": round(100 * forms / (m + p), 1) if m + p else None}

    place_decade = {
        pl: {str(d): cell(pl, d, m, p, n)
             for (q, d), (m, p, n) in sorted(orto_place_decade.items(),
                                            key=lambda kv: kv[0][1]) if q == pl}
        for pl in top}
    plotted = [c for pl in top[:PLOTTED_PLACES] for c in place_decade[pl].values()
               if c["documents"] >= MIN_CELL_DOCUMENTS]
    tops = sorted(c["top_title_share_forms"] for c in plotted)

    titles = {k: v for k, v in orto_title.items() if sum(v) >= MIN_TITLE_FORMS}
    title_forms = sum(sum(v) for v in titles.values())
    low = sum(sum(v) for v in titles.values() if v[0] / sum(v) <= 0.1)
    high = sum(sum(v) for v in titles.values() if v[0] / sum(v) >= 0.9)
    all_modern = sum(v[0] for v in orto_state.values())
    out = {
        "meta": {
            "script": "scripts/corpus_composition.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "ledger": args.ledger, "catalogue": args.catalogue, "orto": args.orto,
            "borders": "state of publication under the borders of 1914",
        },
        "internet_archive": {"documents": n_docs, "bytes": n_bytes},
        "by_type": types.table(n_docs, n_bytes),
        "by_decade": decades.table(n_docs, n_bytes),
        "year_check": {"comparable": comparable, "agree": agree,
                       "agreement": round(100 * agree / comparable, 2)},
        "by_state": state_table,
        "place_table_reach": round(
            100 * (placed - states.docs["unclassified"]) / placed, 2),
        "top_places": {p: {"documents": places.docs[p],
                           "share_documents": round(100 * places.docs[p] / n_docs, 2),
                           "share_bytes": round(100 * places.bytes[p] / n_bytes, 2)}
                       for p in top},
        "by_rights": rights.table(n_docs, n_bytes),
        "orto": {
            "all": share(*map(sum, zip(*orto_state.values()))),
            "by_state": {k: share(*v) for k, v in sorted(orto_state.items())},
            "by_state_period": {f"{s}.{p}": share(*v)
                                for (s, p), v in sorted(orto_state_period.items())},
            "by_place_period": {f"{pl}.{p}": share(*orto_place_period[(pl, p)])
                                for pl in top for p in ("before_1905", "from_1905")},
            "modern_mass_by_state": {k: round(100 * v[0] / all_modern, 1)
                                     for k, v in sorted(orto_state.items())},
            "by_place_decade": place_decade,
            "plotted_cells": {
                "places": top[:PLOTTED_PLACES], "min_documents": MIN_CELL_DOCUMENTS,
                "cells": len(plotted),
                "fewest_documents": min(c["documents"] for c in plotted),
                "fewest_forms": min(c["orto_modern"] + c["orto_period"] for c in plotted),
                "median_top_title_share_forms": tops[len(tops) // 2],
            },
            "by_title": {
                "min_forms": MIN_TITLE_FORMS, "titles": len(titles),
                "share_forms_in_titles_at_most_0.1": round(100 * low / title_forms, 1),
                "share_forms_in_titles_at_least_0.9": round(100 * high / title_forms, 1),
                "named": {f"{pl} / {name}": share(*orto_title[(pl, name)])
                          for pl, name in NAMED_TITLES},
                "largest": [{"place": pl, "title": name, **share(*v)}
                            for (pl, name), v in sorted(titles.items(),
                                                        key=lambda kv: -sum(kv[1]))[:20]],
            },
        },
    }
    path = REPO / args.out
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("by_type", "by_decade", "by_state", "by_rights"):
        print(name)
        for k, v in out[name].items():
            print(f"  {k:26s} {v['documents']:8,} {v['share_documents']:6.2f}%  "
                  f"bytes {v['share_bytes']:6.2f}%")
    print("year check:", out["year_check"], "| place table reach:", out["place_table_reach"])
    print("orto:", json.dumps(out["orto"], ensure_ascii=False, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
