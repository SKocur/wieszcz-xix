"""OCR / scan-layout cleaner for the scanned-print corpus (newspapers, gazettes, books).

Cleans the scan layer, never the language. Letter-level OCR errors ("wSżwaycaryi") are
kept: the LM tolerates them and fixing them is low-precision. Removed instead:

  1. normalize        NFC, drop control chars, unify newlines
  2. split_columns    rebuild reading order when OCR merged two columns with a `|`
  3. drop lines       page furniture, gibberish, short masthead/boilerplate
  4. dehyphenate      join a word split across a line break ("prze-\\ncież" -> "przecież")
  5. reflow           collapse per-line wrapping into paragraphs

Documents still too garbled afterwards, mostly unmarked two-column merges, go to
rejected/. `route()` returns the verdict, `clean_document()` just the cleaned text.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- metrics

def alpha_ratio(s: str) -> float:
    ns = sum(1 for c in s if not c.isspace())
    return sum(1 for c in s if c.isalpha()) / ns if ns else 0.0


def _onechar_frac(words: list[str]) -> float:
    return sum(1 for w in words if len(w) == 1) / len(words) if words else 0.0


def noise_score(t: str) -> float:
    """Penalises low letter-content and token fragmentation. Clean prose lands near
    0.2-0.3, garbled scans well above 0.6."""
    ns = sum(1 for c in t if not c.isspace()) or 1
    alpha = sum(1 for c in t if c.isalpha()) / ns
    return (1 - alpha) * 3 + _onechar_frac(t.split()) * 2


# ------------------------------------------------------------------- line predicates

_CAPS_TOK = re.compile(r"^[A-ZĄĆĘŁŃÓŚŹŻ]{1,4}$")
_PAGE_NUM = re.compile(r"^[\W_]*\d{1,4}[\W_]*$")
_RUN_HDR = re.compile(r"^(Rok|Nr|No|Str|Tom|Cz[eę][sś][cć])\b.*\d", re.IGNORECASE)
_BOILER = re.compile(
    r"prenumerat|og[łl]oszeni|redakcj|redakcy|administracj|administracy|wychodzi w|"
    r"r[eę]kopis|nie zwraca|dodatek (tygodniow|do)|ekspedycj|expedycy|adres redak|"
    r"cena (numeru|egzempl|pojedynczego)|za wiersz|halerz|kopiejek|prenum\.|"
    # almanac furniture: masthead, calendar, sunrise/sunset, saint-of-day
    r"kalendarzyk|okna dziennika|wsch[óo]d s[łl]o[ńn]|zach[óo]d s[łl]o[ńn]|"
    r"wsch[óo]d ksi[eę][żz]|godzin[ieę]? +\d+ +minut|^dzi[śs]\b",
    re.IGNORECASE,
)


# Digitisation boilerplate. Formulaic markers only, since these cannot occur in 19th-c
# print. Not here: modern-year numbers (mostly page numbers and prices) and the © glyph,
# which is 77% OCR noise mid-sentence and would delete most of the corpus.
_ANACHRON = re.compile(
    r"https?://|www\.[a-z]|\bISBN\b|creative commons|\bCC[ -]?BY|wolne lektury|wolnelektur|"
    r"archive\.org|internet archive|domen\w* +publiczn\w*|public domain|wszelkie prawa zastrze|"
    r"copyright|praw\w* +autorski\w*|digiti[sz]ed by|zdigitalizowa|biblioteka cyfrowa|fundacja nowoczesna",
    re.IGNORECASE,
)


def is_anachronism(s: str) -> bool:
    return bool(_ANACHRON.search(s))


def _caps_gibberish_frac(words: list[str]) -> float:
    return sum(1 for w in words if _CAPS_TOK.match(w)) / len(words) if words else 0.0


def is_page_furniture(s: str) -> bool:
    s = s.strip()
    if _PAGE_NUM.match(s):
        return True
    return len(s) <= 14 and bool(_RUN_HDR.match(s))


def is_garbage_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    words = s.split()
    ar = alpha_ratio(s)
    if len(s) <= 6 and ar < 0.5:                                   # "=.", "%'", "l\\"
        return True
    if len(words) >= 3 and _caps_gibberish_frac(words) > 0.5:      # "NH BEC BPE Ha U JMR TW"
        return True
    if len(words) >= 4 and _onechar_frac(words) > 0.45:            # "A SR ckw l ki nn UR"
        return True
    if len(s) >= 8 and ar < 0.45:                                  # symbol soup
        return True
    return False


def is_boilerplate(s: str) -> bool:
    # Short lines only: a long sentence that happens to say "prenumerata" is real content.
    return len(s.strip()) < 65 and bool(_BOILER.search(s))


# ------------------------------------------------------------------------- transforms

def normalize(t: str) -> str:
    t = unicodedata.normalize("NFC", t).replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("©", " ")  # 77% OCR noise, 23% modern glyph
    return "".join(c for c in t if c in "\n\t" or unicodedata.category(c)[0] != "C")


def split_columns(text: str) -> str:
    """When OCR fuses two print columns with a `|`, reading order is every left half
    top-to-bottom, then every right half. A rare internal `|` is stray noise instead."""
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    internal = [ln for ln in lines if "|" in ln.strip()[1:-1]]
    if not nonblank or len(internal) < max(5, 0.15 * len(nonblank)):
        return text.replace("|", " ")
    left, right = [], []
    for ln in lines:
        if "|" in ln:
            a, _, b = ln.partition("|")
            left.append(a)
            right.append(b)
        else:
            left.append(ln)
    return "\n".join(left) + "\n\n" + "\n".join(right)


def dehyphenate(t: str) -> str:
    return re.sub(r"([^\W\d_])[-¬]\n[ \t]*([^\W\d_])", r"\1\2", t)


def reflow(t: str) -> str:
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
    t = re.sub(r"\n{2,}", "\x00", t)          # protect paragraph breaks
    t = t.replace("\n", " ").replace("\x00", "\n\n")
    return re.sub(r"[ \t]{2,}", " ", t).strip()


# ----------------------------------------------------------------------------- pipeline

def clean_document(text: str) -> str:
    t = split_columns(normalize(text))
    kept: list[str] = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if is_page_furniture(s) or is_garbage_line(s) or is_boilerplate(s) or is_anachronism(s):
            continue
        kept.append(ln)
    return reflow(dehyphenate("\n".join(kept)))


def route(text: str, max_score: float = 0.55, min_len: int = 200):
    """(verdict, reason, cleaned), where verdict is 'clean' or 'reject'."""
    cleaned = clean_document(text)
    if len(cleaned) < min_len:
        return "reject", "za krótki po czyszczeniu", cleaned
    sc = noise_score(cleaned)
    if sc > max_score:
        return "reject", f"nieodratowalny szum (score={sc:.2f})", cleaned
    return "clean", f"score={sc:.2f}", cleaned
