"""Classify every Wolne Lektury document in the corpus by its catalogue epoch.

Content signals do not work on Wolne Lektury files in either direction: WL modernizes
orthography (so the post-reform share always reads modern), its colophon carries an
ISBN and a licence URL (so apparatus markers always fire), and its texts carry no OCR
garble (so nothing needs the noise defences). What WL does have is a curated
catalogue: every book carries a literary epoch. This asks the API for the whole
catalogue in one request and classifies our wl_ files: `Dwudziestolecie
międzywojenne` and `Współczesność` are post-1918 by definition.

The known blind spot, recorded rather than solved: the epoch describes the *work*,
not the edition, so an interwar translation of an older work (Boy's Stendhal) carries
the original's epoch. The exclusion step therefore also carries over the
hand-verified WL list from the first release audit, which caught those by content.

    python scripts/wl_epoch_sweep.py
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
API = "https://wolnelektury.pl/api/books/"
POST1918 = {"Dwudziestolecie międzywojenne", "Współczesność"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--out", default="metrics/wl_epochs_2026-08-03.json")
    args = ap.parse_args()

    slugs = sorted(f.stem[3:] for f in Path(args.clean).glob("wl_*.txt"))
    print(f"{len(slugs):,} wl_ documents in the corpus")

    req = Request(API, headers={"User-Agent": "wieszcz-xix epoch sweep"})
    with urlopen(req, timeout=60) as r:
        catalogue = json.load(r)
    by_slug = {b["slug"]: b.get("epoch") or "" for b in catalogue}
    print(f"catalogue: {len(by_slug):,} books")

    epochs: dict[str, str] = {}
    missing: list[str] = []
    for slug in slugs:
        if slug in by_slug:
            epochs[slug] = by_slug[slug]
        else:
            missing.append(slug)

    # books removed from the catalogue since the crawl: ask for them one by one
    still_missing: list[str] = []
    for slug in missing:
        try:
            with urlopen(Request(f"{API}{slug}/",
                                 headers={"User-Agent": "wieszcz-xix epoch sweep"}),
                         timeout=30) as r:
                d = json.load(r)
            epochs[slug] = ", ".join(e["name"] for e in d.get("epochs", []))
        except Exception:  # noqa: BLE001 — gone from the API entirely
            still_missing.append(slug)

    post = sorted(s for s, e in epochs.items()
                  if any(p in e for p in POST1918))
    counts: dict[str, int] = {}
    for e in epochs.values():
        counts[e or "(none)"] = counts.get(e or "(none)", 0) + 1

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None

    out = REPO / args.out
    out.write_text(json.dumps({
        "meta": {
            "script": "scripts/wl_epoch_sweep.py",
            "api": API,
            "post1918_epochs": sorted(POST1918),
            "git_commit": git,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
        },
        "documents": len(slugs),
        "epoch_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "post1918_documents": len(post),
        "post1918_ids": [f"wl_{s}" for s in post],
        "missing_from_api": [f"wl_{s}" for s in still_missing],
        "epochs": {f"wl_{s}": e for s, e in sorted(epochs.items())},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("epochs:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"post-1918 by epoch: {len(post)}")
    for s in post[:20]:
        print("  wl_" + s)
    if still_missing:
        print(f"missing from API: {len(still_missing)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
