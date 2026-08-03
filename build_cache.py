"""Build data/clean/tokens.bin, or report it if already built. Tokenization itself lives
in train.build_token_cache. Run from the repo root:

    rm -f data/clean/tokens.bin   # only when re-tokenizing after a corpus change
    python build_cache.py
"""

import sys
sys.path.insert(0, "src")

from train import build_token_cache

data = build_token_cache()
print(f"TOKENS_TOTAL={len(data):,}")
