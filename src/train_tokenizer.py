"""Train a dedicated BPE tokenizer on the historical corpus.

A modern tokenizer fragments pre-1936 spelling badly.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

CLEAN_DIR = Path("data/clean")
OUT_DIR = Path("tokenizer")
VOCAB_SIZE = 8_000
SPECIAL_TOKENS = ["<|endoftext|>"]
# The trainer holds the word-frequency table in RAM: on the full 2.6 GB corpus it hit
# 9.8 GB RSS and was OOM-killed. An 8k merge table converges on a fraction of the text.
MAX_TRAIN_BYTES = int(os.environ.get("TOKENIZER_MAX_BYTES", 400_000_000))
SAMPLE_SEED = 0


def sample_corpus(paths: list[Path], budget: int) -> list[Path]:
    """Pick files up to `budget` bytes, keeping each source's share of the corpus.

    Uniform sampling would let the Internet Archive (93% of the bytes) swamp Wolne
    Lektury, and OCR text tokenizes at 2.52 tokens/word against 1.90 for clean
    transcriptions.
    """
    total = sum(p.stat().st_size for p in paths)
    if total <= budget:
        return paths

    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(p.name.split("_")[0], []).append(p)

    chosen: list[Path] = []
    rng = random.Random(SAMPLE_SEED)
    for prefix, group in sorted(groups.items()):
        group_bytes = sum(p.stat().st_size for p in group)
        share = budget * group_bytes / total
        rng.shuffle(group)
        taken = 0
        for p in group:
            if taken >= share:
                break
            chosen.append(p)
            taken += p.stat().st_size
        print(f"  {prefix}: {len(group)} files ({group_bytes/1e9:.2f} GB) "
              f"-> sampled {len([c for c in chosen if c.name.startswith(prefix)])} "
              f"files ({taken/1e6:.0f} MB)")
    return chosen


def main() -> None:
    paths = [p for p in CLEAN_DIR.glob("*.txt") if not p.name.startswith(".")]
    if not paths:
        raise SystemExit("No corpus found. Run src/prepare_data.py first.")

    print(f"Corpus: {len(paths)} files, "
          f"{sum(p.stat().st_size for p in paths)/1e9:.2f} GB")
    files = [str(p) for p in sample_corpus(paths, MAX_TRAIN_BYTES)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(OUT_DIR))
    print(f"Trained BPE ({VOCAB_SIZE} tokens) on {len(files)} of {len(paths)} files "
          f"-> {OUT_DIR}/")


if __name__ == "__main__":
    main()
