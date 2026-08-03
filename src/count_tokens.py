"""Token-count of the training corpus, without a full re-tokenisation.

Words are counted with a `cat | wc -w` pipeline, and the token/word ratio is measured on a
spread-out sample with the real tokenizer. `--exact` tokenises every file instead, which
takes ~15 min on the whole corpus.

    python src/count_tokens.py            # fast estimate (seconds)
    python src/count_tokens.py --exact    # exact, slow

Run from the repo root; reads data/clean/ and the tokenizer in tokenizer/.
"""

import argparse
import subprocess
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

CLEAN = Path("data/clean")
TOK = Path("tokenizer")


def wc_words(prefix: str) -> int:
    """Total words across data/clean/{prefix}_*.txt, via a streaming shell pipeline."""
    cmd = f"find {CLEAN} -maxdepth 1 -name '{prefix}_*.txt' -print0 | xargs -0 cat | wc -w"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def sample_ratio(tok: ByteLevelBPETokenizer, files: list[Path], n: int = 60) -> tuple[float, int]:
    step = max(1, len(files) // n)
    sample = files[::step][:n]
    words = toks = 0
    for f in sample:
        txt = f.read_text("utf-8", errors="replace")
        words += len(txt.split())
        toks += len(tok.encode(txt).ids)
    return toks / max(1, words), len(sample)


def exact_tokens(tok: ByteLevelBPETokenizer, files: list[Path]) -> int:
    return sum(len(tok.encode(f.read_text("utf-8", errors="replace")).ids) for f in files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", action="store_true", help="tokenise every file (accurate, slow)")
    args = ap.parse_args()

    tok = ByteLevelBPETokenizer(str(TOK / "vocab.json"), str(TOK / "merges.txt"))
    ia = sorted(CLEAN.glob("ia_*.txt"))
    wl = sorted(CLEAN.glob("wl_*.txt"))

    if args.exact:
        t_ia, t_wl = exact_tokens(tok, ia), exact_tokens(tok, wl)
        print(f"ia    {len(ia):>5} files  {t_ia:>14,} tokens")
        print(f"wl    {len(wl):>5} files  {t_wl:>14,} tokens")
        print(f"TOTAL {len(ia) + len(wl):>5} files  {t_ia + t_wl:>14,} tokens  (exact)")
        return

    w_ia, w_wl = wc_words("ia"), wc_words("wl")
    ratio, ns = sample_ratio(tok, ia + wl)
    total = int((w_ia + w_wl) * ratio)
    print(f"ia    {len(ia):>5} files  {w_ia:>13,} words")
    print(f"wl    {len(wl):>5} files  {w_wl:>13,} words")
    print(f"tok/word {ratio:.3f}  (sampled {ns} files)")
    print(f"~ {total:,} tokens  ({total / 1e6:.0f}M)   [estimate; --exact for precise]")


if __name__ == "__main__":
    main()
