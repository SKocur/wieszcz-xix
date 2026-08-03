"""Quantify OCR corruption in the cleaned corpus.

Measures how much of the Internet-Archive half is corrupted and how it is distributed:
evenly spread corruption needs per-word correction, concentrated corruption is fixed by
dropping whole documents.

Classifies each whitespace token with character-level heuristics, then reports the
suspicious-word rate by source and reason, the per-document histogram, and how much text
each document-drop threshold would cost.

Read-only.
"""

import sys
from collections import Counter
from pathlib import Path

CLEAN_DIR = Path("data/clean")

VOWELS = set("aeiouyąęó")
PL_LOWER = set("abcdefghijklmnopqrstuvwxyząćęłńóśźż")
PL_UPPER = set(c.upper() for c in PL_LOWER)
PL_ALPHA = PL_LOWER | PL_UPPER

# Never inside a Polish word.
NOISE_SYMBOLS = set("^~|\\«»°*§¶`_=+<>{}[]@#$%&")

EDGE_PUNCT = ".,;:!?\"'()[]{}—–-…«»„”“‚’·•"


def strip_edges(tok: str) -> str:
    return tok.strip(EDGE_PUNCT)


def _midcaps(tok: str) -> bool:
    # "SoKoła", "ledaKcji". 19th-c prose has no CamelCase, so interior uppercase is a
    # case-flip. Caller guarantees tok is neither all-lower nor all-upper.
    seen_lower = False
    for c in tok:
        if c.islower():
            seen_lower = True
        elif c.isupper() and seen_lower:
            return True
    return False


def reasons(tok: str):
    """OCR-suspicion reasons a token trips, empty if clean.

    Fast path first: plain alphabetic words cannot trip digit_mix or symbol, and
    isalpha()/islower() are C-level, so the character loops run only on the minority.
    """
    if tok.isalpha():
        r = set()
        low = tok if tok.islower() else tok.lower()
        if len(low) >= 4 and VOWELS.isdisjoint(low):
            r.add("no_vowel")
        if not tok.islower() and not tok.isupper() and _midcaps(tok):
            r.add("midcaps")
        return r

    r = set()
    if not NOISE_SYMBOLS.isdisjoint(tok):
        r.add("symbol")
    for i, c in enumerate(tok):
        if c.isdigit():
            if (i > 0 and tok[i - 1].isalpha()) or (i + 1 < len(tok) and tok[i + 1].isalpha()):
                r.add("digit_mix")
                break
    if not tok.islower() and not tok.isupper() and _midcaps(tok):
        r.add("midcaps")
    return r


def main():
    files = sorted(CLEAN_DIR.glob("*.txt"))
    if not files:
        print("no .txt files under", CLEAN_DIR)
        sys.exit(1)

    reason_counts = Counter()
    src_words = Counter()      # total words per source
    src_suspicious = Counter()  # suspicious words per source

    # per-file: (source, total_words, suspicious_words)
    per_file = []

    for n, f in enumerate(files):
        src = f.name.split("_", 1)[0]  # "ia" or "wl"
        text = f.read_text(encoding="utf-8", errors="replace")
        total = 0
        susp = 0
        for raw in text.split():
            tok = strip_edges(raw)
            if not tok:
                continue
            total += 1
            rs = reasons(tok)
            if rs:
                susp += 1
                for x in rs:
                    reason_counts[x] += 1
        src_words[src] += total
        src_suspicious[src] += susp
        per_file.append((f.name, src, total, susp))
        if (n + 1) % 500 == 0:
            print(f"  ...{n + 1}/{len(files)} files", file=sys.stderr)

    grand_total = sum(src_words.values())
    grand_susp = sum(src_suspicious.values())

    print("=" * 66)
    print("OVERALL")
    print("=" * 66)
    print(f"documents:        {len(files)}")
    print(f"total words:      {grand_total:,}")
    print(f"suspicious words: {grand_susp:,}  ({100 * grand_susp / grand_total:.2f}%)")
    print()
    print("By source:")
    for src in sorted(src_words):
        w = src_words[src]
        s = src_suspicious[src]
        print(f"  {src:<4} words {w:>12,}   suspicious {s:>10,}  ({100 * s / w:.2f}%)")
    print()
    print("By reason (a word can trip several):")
    for reason, c in reason_counts.most_common():
        print(f"  {reason:<12} {c:>10,}  ({100 * c / grand_total:.2f}% of all words)")

    # ---- per-document distribution (ia only; wl is the clean baseline) --------
    ia = [(name, t, s) for (name, src, t, s) in per_file if src == "ia" and t >= 50]
    print()
    print("=" * 66)
    print(f"PER-DOCUMENT CORRUPTION, Internet Archive only ({len(ia)} docs >=50 words)")
    print("=" * 66)
    buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 101)]
    hist = Counter()
    for name, t, s in ia:
        pct = 100 * s / t
        for lo, hi in buckets:
            if lo <= pct < hi:
                hist[(lo, hi)] += 1
                break
    for lo, hi in buckets:
        c = hist[(lo, hi)]
        bar = "#" * round(50 * c / max(1, len(ia)))
        print(f"  {lo:>3}-{hi - 1 if hi <= 100 else 100:<3}% corrupt: {c:>5} docs  {bar}")

    # ---- decision table: drop docs above threshold, how much text lost? -------
    print()
    print("=" * 66)
    print("IF WE DROP ia DOCS ABOVE A CORRUPTION THRESHOLD")
    print("=" * 66)
    ia_total_words = sum(t for _, t, _ in ia)
    print(f"(ia text pool: {ia_total_words:,} words in {len(ia)} docs)")
    print(f"{'thresh':>7} {'docs dropped':>13} {'words dropped':>15} {'% text lost':>12}")
    for thr in (50, 40, 30, 25, 20, 15, 10, 8, 5):
        dropped_docs = [(t, s) for _, t, s in ia if 100 * s / t >= thr]
        dw = sum(t for t, _ in dropped_docs)
        print(f"{thr:>6}% {len(dropped_docs):>13} {dw:>15,} {100 * dw / ia_total_words:>11.2f}%")

    # ---- show the worst and a median-ish document ----------------------------
    ia_sorted = sorted(ia, key=lambda x: x[2] / x[1])
    print()
    print("cleanest 3 ia docs:")
    for name, t, s in ia_sorted[:3]:
        print(f"  {100 * s / t:5.1f}%  {name}  ({t:,} words)")
    print("dirtiest 3 ia docs:")
    for name, t, s in ia_sorted[-3:]:
        print(f"  {100 * s / t:5.1f}%  {name}  ({t:,} words)")


if __name__ == "__main__":
    main()
