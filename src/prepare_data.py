"""Download and clean the 19th-century Polish public-domain corpus.

Sources: Wolne Lektury (clean transcriptions), curated Wikiźródła pages, and Internet
Archive OCR. Pre-1936 orthography is preserved.

Re-running is incremental: anything already in data/clean/ is skipped.
APIs: https://wolnelektury.pl/api/ and https://pl.wikisource.org/w/api.php.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import socket

import requests
import urllib3.util.connection
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# WSL2 has no IPv6 route, so hosts with an AAAA record (wolnelektury.pl) fail with
# "Network is unreachable". Every archive here is reachable over IPv4.
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

WL_API = "https://wolnelektury.pl/api"
WS_API = "https://pl.wikisource.org/w/api.php"
IA_SCRAPE = "https://archive.org/services/search/v1/scrape"
# "Polish" matches ~1.2k items, "pol" ~342k.
IA_QUERY = (
    "(language:(pol) OR language:(Polish)) "
    "AND year:[1800 TO 1918] AND mediatype:texts"
)
IA_SORT = "downloads desc"
IA_MAX_ITEMS = 5000
IA_WORKERS = 2

# Gate thresholds, calibrated on measured files. See docs/data-preparation.md.
IA_MIN_ALPHA = 0.60
IA_MIN_DIACRITIC = 0.04
IA_MIN_STOPWORD = 0.03
IA_MAX_CYRILLIC = 0.005
LINE_MIN_ALPHA = 0.55
LINE_MIN_WORDS = 4

# Distinctly Polish, avoiding cross-Slavic "i"/"w"/"na".
POLISH_STOPWORDS = {
    "się", "że", "nie", "jest", "który", "była", "było", "oraz", "przez", "jego",
    "ich", "tylko", "już", "dla", "tego", "żeby", "aby", "więc", "aż", "lub",
}

EPOCHS = ["romantyzm", "pozytywizm", "modernizm"]

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
REJECTED_LOG = Path("data/rejected.jsonl")
IA_CURSOR_FILE = Path("data/ia_cursor.json")
WIKISOURCE_TITLES = Path("data/wikisource_titles.txt")
REQUEST_PAUSE_S = 0.05


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    return session


SESSION = make_session()


def gate_fingerprint() -> str:
    """Hash of every gate setting, stamped on each rejection so loosening a threshold
    re-opens the items it excluded."""
    payload = json.dumps(
        {
            "alpha": IA_MIN_ALPHA,
            "diacritic": IA_MIN_DIACRITIC,
            "stopword": IA_MIN_STOPWORD,
            "cyrillic": IA_MAX_CYRILLIC,
            "line_alpha": LINE_MIN_ALPHA,
            "line_words": LINE_MIN_WORDS,
            "clean_pass": "junk_v1",       # bump when the cleaning passes change
            "stopwords": sorted(POLISH_STOPWORDS),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


GATE_ID = gate_fingerprint()
_REJECTED_LOCK = threading.Lock()


def load_rejected() -> set[str]:
    """Identifiers already judged unusable under the current gates."""
    if not REJECTED_LOG.exists():
        return set()
    seen = set()
    for line in REJECTED_LOG.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn line from a killed run
        if entry.get("gate") == GATE_ID:
            seen.add(entry["id"])
    return seen


def record_rejected(text_id: str, reason: str, **metrics: float) -> None:
    """Append one rejection so future runs skip the download.

    Content rejections only: a 503 says nothing about the text.
    """
    entry = {"id": text_id, "reason": reason, "gate": GATE_ID,
             **{k: round(v, 4) for k, v in metrics.items()}}
    with _REJECTED_LOCK:
        with REJECTED_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def already_have(text_id: str) -> bool:
    return (CLEAN_DIR / f"{text_id}.txt").exists()


def save(text_id: str, raw: str, cleaned: str) -> None:
    (RAW_DIR / f"{text_id}.txt").write_text(raw, encoding="utf-8")
    (CLEAN_DIR / f"{text_id}.txt").write_text(cleaned, encoding="utf-8")


def normalize(text: str, strip_footer: bool = False) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    if strip_footer:  # Wolne Lektury appends a licence block after a dashed line
        text = re.split(r"\n-{3,}\n", text)[0]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def fetch_wolne_lektury() -> int:
    added = 0
    for epoch in EPOCHS:
        books = SESSION.get(f"{WL_API}/epochs/{epoch}/books/", timeout=30).json()
        for book in tqdm(books, desc=f"WL/{epoch}"):
            text_id = f"wl_{book['slug']}"
            if already_have(text_id):
                continue
            try:
                detail = SESSION.get(book["href"], timeout=30).json()
                url = detail.get("txt")
                if not url:
                    continue
                raw = SESSION.get(url, timeout=60).text
            except requests.RequestException as exc:
                print(f"  skip {book['slug']}: {exc}")
                continue
            save(text_id, raw, normalize(raw, strip_footer=True))
            added += 1
            time.sleep(REQUEST_PAUSE_S)
    return added


def fetch_wikisource() -> int:
    """Download curated Wikiźródła pages listed in data/wikisource_titles.txt.

    One page title per line. Long works are split across subpages, which have to be
    listed individually.
    """
    if not WIKISOURCE_TITLES.exists():
        print(f"  (skip Wikisource: create {WIKISOURCE_TITLES}, one page title per line)")
        return 0
    titles = [t.strip() for t in WIKISOURCE_TITLES.read_text("utf-8").splitlines() if t.strip()]
    added = 0
    for title in tqdm(titles, desc="WS"):
        text_id = "ws_" + re.sub(r"[^\w-]+", "_", title)
        if already_have(text_id):
            continue
        try:
            resp = SESSION.get(
                WS_API,
                params={
                    "action": "query", "format": "json", "prop": "extracts",
                    "explaintext": 1, "redirects": 1, "titles": title,
                },
                timeout=30,
            ).json()
            extract = next(iter(resp["query"]["pages"].values())).get("extract", "")
        except (requests.RequestException, KeyError) as exc:
            print(f"  skip {title}: {exc}")
            continue
        if not extract.strip():
            print(f"  empty extract for '{title}' (may need subpages)")
            continue
        save(text_id, extract, normalize(extract))
        added += 1
        time.sleep(REQUEST_PAUSE_S)
    return added


def strip_google_boilerplate(text: str) -> str:
    """Cut through the last 'Google' mention in the first 300 lines. A 19th-century book
    never mentions Google."""
    lines = text.split("\n")
    cut = 0
    for i, line in enumerate(lines[:300]):
        if "google" in line.lower():
            cut = i + 1
    return "\n".join(lines[cut:])


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph; keep blank-line paragraph breaks.

    Internet Archive text only: Wolne Lektury line breaks carry verse.
    """
    paragraphs = []
    for para in text.split("\n\n"):
        buf = ""
        for line in (l.strip() for l in para.split("\n")):
            if not line:
                continue
            if buf.endswith("-"):
                buf = buf[:-1] + line  # word continues on the next line
            elif buf:
                buf += " " + line
            else:
                buf = line
        if buf:
            paragraphs.append(buf)
    return "\n\n".join(paragraphs)


def clean_ocr(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = strip_google_boilerplate(text)
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace breaks paragraph splits
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = strip_low_quality_lines(text)  # must run while lines are still physical
    text = unwrap_lines(text)
    # A watermark split across two lines rejoins here.
    text = "\n".join(l for l in text.split("\n") if not _is_junk_line(l))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip() + "\n"


def alpha_ratio(text: str) -> float:
    """Share of whitespace tokens that are clean alphabetic words.

    Period prose scores ~0.75-0.80, abbreviation dictionaries ~0.30.
    """
    words = text.split()
    if not words:
        return 0.0
    good = sum(1 for w in words if re.fullmatch(r"[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ-]+", w))
    return good / len(words)


def _is_junk_line(line: str) -> bool:
    """Watermarks, archive.org URLs and page-number-only lines. These read as clean words,
    so the alpha-density filter never flags them."""
    s = line.strip()
    if not s:
        return False
    low = " ".join(s.lower().split())  # OCR leaves runs of spaces
    if "digitized by" in low or "archive.org" in low or low == "google":
        return True
    if any(c.isalpha() for c in s):
        return False          # not a bare page/figure line
    return any(c.isdigit() for c in s)


def strip_low_quality_lines(text: str) -> str:
    """Drop non-prose physical lines, before unwrapping: OCR furniture, then
    formatting-dense long lines. Lines under LINE_MIN_WORDS are never judged."""
    kept = [line for line in text.split("\n")
            if not _is_junk_line(line)
            and (len(line.split()) < LINE_MIN_WORDS or alpha_ratio(line) >= LINE_MIN_ALPHA)]
    return "\n".join(kept)


def polish_metrics(text: str) -> dict[str, float]:
    """The three language signals, measured once so callers can log them."""
    lower = text.lower()
    if not lower:
        return {"cyrillic": 0.0, "diacritic": 0.0, "stopword": 0.0}
    toks = re.findall(r"[a-ząćęłńóśźż]+", lower)
    return {
        "cyrillic": len(re.findall(r"[Ѐ-ӿ]", lower)) / len(lower),
        "diacritic": len(re.findall(r"[ąćęłńóśźż]", lower)) / len(lower),
        "stopword": sum(1 for w in toks if w in POLISH_STOPWORDS) / len(toks) if toks else 0.0,
    }


def ia_reject_reason(text: str) -> tuple[str, dict[str, float]] | None:
    """Why this text fails the IA gates, or None if it passes.

    IA's language tag admits Italian and English books, 19th-century editions of
    17th-century chancery records, and Russian-era publications carrying Cyrillic.
    Both Polish signals must clear: accepting either let the chancery material through
    on diacritics alone.
    """
    m = polish_metrics(text)
    m["alpha"] = alpha_ratio(text)
    if not text.strip():
        return "empty", m
    if m["alpha"] < IA_MIN_ALPHA:
        return "alpha", m
    if m["cyrillic"] > IA_MAX_CYRILLIC:
        return "cyrillic", m
    if m["diacritic"] < IA_MIN_DIACRITIC:
        return "diacritic", m
    if m["stopword"] < IA_MIN_STOPWORD:
        return "stopword", m
    return None


def is_polish(text: str) -> bool:
    m = polish_metrics(text)
    return (m["cyrillic"] <= IA_MAX_CYRILLIC
            and m["diacritic"] >= IA_MIN_DIACRITIC
            and m["stopword"] >= IA_MIN_STOPWORD)


def ia_accept(text: str) -> bool:
    return ia_reject_reason(text) is None


def _query_id() -> str:
    """A cursor encodes a position in one ordering, so changing query or sort
    invalidates it."""
    return hashlib.sha1(f"{IA_QUERY}|{IA_SORT}".encode()).hexdigest()[:12]


def load_cursor() -> str | None:
    if not IA_CURSOR_FILE.exists():
        return None
    try:
        state = json.loads(IA_CURSOR_FILE.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    if state.get("query") != _query_id():
        return None
    return state.get("cursor")


def save_cursor(cursor: str | None) -> None:
    """cursor is None once the catalogue is exhausted."""
    IA_CURSOR_FILE.write_text(
        json.dumps({"query": _query_id(), "cursor": cursor}, ensure_ascii=False),
        encoding="utf-8",
    )


def scrape_identifiers(cursor: str | None) -> tuple[list[str], str | None]:
    """One page of the IA Scrape API: (identifiers, next_cursor).

    Sending `count` alongside `cursor` makes the API ignore the cursor and re-return the
    first page forever, so a cursor request takes the default page size of 5000.
    """
    params = {"q": IA_QUERY, "fields": "identifier", "sorts": IA_SORT}
    if cursor:
        params["cursor"] = cursor
    resp = SESSION.get(IA_SCRAPE, params=params, timeout=90).json()
    ids = [d["identifier"] for d in resp.get("items", [])]
    return ids, resp.get("cursor")


def ia_text_url(identifier: str) -> tuple[str | None, str]:
    """Locate the item's DjVuTXT file via its metadata.

    Guessing `{id}_djvu.txt` 404s on digitised-library imports, which keep the original
    filename. Returns (url, "ok"), (None, "no_ocr") for image-only scans, or
    (None, "miss") on a transient metadata failure.
    """
    try:
        meta = SESSION.get(f"https://archive.org/metadata/{identifier}", timeout=60).json()
    except (requests.RequestException, ValueError):
        return None, "miss"
    for f in meta.get("files", []):
        if f.get("format") == "DjVuTXT":
            name = urllib.parse.quote(f["name"])
            return f"https://archive.org/download/{identifier}/{name}", "ok"
    return None, "no_ocr"


def fetch_one_ia(identifier: str) -> str:
    """Download, clean, gate and save one IA item. Returns an outcome tag:

    kept    passed the gates and saved
    gated   below the norm, logged to the rejection ledger
    no_ocr  image-only scan
    miss    transient failure, not recorded so a re-run retries it

    Runs in a worker thread and touches only its own files.
    """
    url, why = ia_text_url(identifier)
    if url is None:
        return why
    try:
        r = SESSION.get(url, timeout=90)  # follows the 302 to a data node
    except requests.RequestException:
        return "miss"
    if r.status_code != 200 or not r.text.strip():
        return "miss"
    cleaned = clean_ocr(r.text)
    verdict = ia_reject_reason(cleaned)
    if verdict is not None:
        reason, metrics = verdict
        record_rejected(f"ia_{identifier}", reason, **metrics)
        return "gated"
    save(f"ia_{identifier}", r.text, cleaned)
    time.sleep(REQUEST_PAUSE_S)
    return "kept"


def fetch_internet_archive(max_items: int = IA_MAX_ITEMS, workers: int = IA_WORKERS) -> int:
    """Walk the Scrape API cursor, collecting up to `max_items` unseen identifiers, then
    download them over `workers` connections.

    The cursor is persisted only after the downloads finish, so a crash never advances
    past items it failed to fetch.
    """
    rejected = load_rejected()
    # safe_cursor is the last position whose pending items are all accounted for.
    safe_cursor = load_cursor()
    cursor = safe_cursor
    pending: list[str] = []
    exhausted = False

    # Pages are a fixed 5000, so one can overflow the budget. Take what fits and leave
    # safe_cursor on the previous page; the re-run re-walks it and already_have() skips.
    while len(pending) < max_items:
        ids, cursor = scrape_identifiers(cursor)
        page_pending = [i for i in ids
                        if not already_have(f"ia_{i}") and f"ia_{i}" not in rejected]
        room = max_items - len(pending)
        if len(page_pending) <= room:
            pending.extend(page_pending)
            safe_cursor = cursor
            if cursor is None:
                exhausted = True
                break
        else:
            pending.extend(page_pending[:room])   # partial page, safe_cursor stays put
            break

    if rejected:
        print(f"  IA: skipping {len(rejected)} items already judged unusable "
              f"(data/rejected.jsonl, gates {GATE_ID})")
    if not pending:
        save_cursor(safe_cursor)
        print("  IA: " + ("reached the end of the catalogue for this query"
                          if exhausted else "nothing new on the walked pages"))
        return 0

    tally: Counter = Counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch_one_ia, i) for i in pending]
        for n, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="IA"), 1):
            tally[future.result()] += 1
            if n % 250 == 0:
                print(f"  IA [{n}/{len(pending)}] kept={tally['kept']} "
                      f"gated={tally['gated']} no_ocr={tally['no_ocr']} miss={tally['miss']}")

    save_cursor(safe_cursor)
    print(f"  IA outcome: kept={tally['kept']} | gated (below norm)={tally['gated']} | "
          f"no_ocr (image-only)={tally['no_ocr']} | miss (transient)={tally['miss']}")
    if exhausted:
        print("  IA: reached the end of the catalogue for this query.")
    return tally["kept"]


def prune_existing() -> int:
    """Re-apply the current IA gates to files already in clean/ and drop the failures.

    Without this the corpus is a mix of whatever rules were in force when each file
    was fetched.
    """
    removed = 0
    for path in sorted(CLEAN_DIR.glob("ia_*.txt")):
        verdict = ia_reject_reason(path.read_text("utf-8"))
        if verdict is None:
            continue
        reason, metrics = verdict
        record_rejected(path.stem, reason, **metrics)
        path.unlink()
        (RAW_DIR / path.name).unlink(missing_ok=True)
        removed += 1
    return removed


def reclean_from_raw() -> int:
    """Re-generate every clean/ file from raw/, offline. Use after changing normalize()."""
    count = pruned = 0
    for raw_path in sorted(RAW_DIR.glob("*.txt")):
        text_id = raw_path.stem
        raw = raw_path.read_text("utf-8")
        if text_id.startswith("ia_"):
            cleaned = clean_ocr(raw)
            verdict = ia_reject_reason(cleaned)
            if verdict is not None:
                reason, metrics = verdict
                record_rejected(text_id, reason, **metrics)
                (CLEAN_DIR / f"{text_id}.txt").unlink(missing_ok=True)
                raw_path.unlink()
                pruned += 1
                continue
        else:
            cleaned = normalize(raw, strip_footer=text_id.startswith("wl_"))
        (CLEAN_DIR / f"{text_id}.txt").write_text(cleaned, encoding="utf-8")
        count += 1
    if pruned:
        print(f"Pruned {pruned} low-quality IA files.")
    return count


def summarise_rejected() -> None:
    """Breakdown of the rejection ledger by reason."""
    if not REJECTED_LOG.exists():
        print("No rejections recorded yet.")
        return
    entries = [json.loads(l) for l in REJECTED_LOG.read_text("utf-8").splitlines() if l.strip()]
    current = [e for e in entries if e.get("gate") == GATE_ID]
    stale = len(entries) - len(current)
    print(f"{len(current)} rejections under the current gates ({GATE_ID})"
          + (f", {stale} from older gate settings (ignored)" if stale else ""))
    by_reason: dict[str, int] = {}
    for e in current:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<10} {n:>5}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclean", action="store_true",
                        help="re-clean existing raw/ files offline, then exit")
    parser.add_argument("--prune", action="store_true",
                        help="re-apply current IA gates to clean/ and drop failures, then exit")
    parser.add_argument("--rejected", action="store_true",
                        help="summarise the rejection ledger, then exit")
    parser.add_argument("--ia-workers", type=int, default=IA_WORKERS,
                        help=f"parallel IA downloads (default {IA_WORKERS})")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if args.rejected:
        summarise_rejected()
        return

    if args.prune:
        n = prune_existing()
        print(f"Pruned {n} files that fail the current gates. "
              f"Corpus now {len(list(CLEAN_DIR.glob('*.txt')))} texts.")
        return

    if args.reclean:
        n = reclean_from_raw()
        print(f"Re-cleaned {n} files from raw/. Delete data/clean/tokens.bin before training.")
        return

    def run(name: str, fetch) -> int:
        """One archive being unreachable must not kill the crawl."""
        try:
            return fetch()
        except Exception as exc:
            print(f"  ! {name} unavailable, skipping: {type(exc).__name__}: {exc}")
            return 0

    new_wl = run("Wolne Lektury", fetch_wolne_lektury)
    new_ws = run("Wikiźródła", fetch_wikisource)
    new_ia = run("Internet Archive",
                 lambda: fetch_internet_archive(workers=args.ia_workers))

    total = len(list(CLEAN_DIR.glob("*.txt")))
    print(f"\nAdded {new_wl} (Wolne Lektury) + {new_ws} (Wikiźródła) + {new_ia} "
          f"(Internet Archive). Corpus now {total} texts.")
    print("Re-running only fetches what is missing.")
    print("After changing the corpus, delete data/clean/tokens.bin so training re-tokenizes.")
    print("Record new sources in data/PROVENANCE.md.")


if __name__ == "__main__":
    main()
