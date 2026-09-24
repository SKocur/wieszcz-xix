"""Per-source token counts for both frozen streams, recovered from the token files.

Table 1 gives the corpus split by source in tokens, and that number cannot be read off
the tokenization report, which records the train/held-out split and not the source one.
It is recoverable without re-tokenizing: documents are written in code-point-sorted
filename order and the two sources use disjoint prefixes (`ia_`, `wl_`), so each source
occupies one contiguous block. Counting document terminators to the boundary gives the
split exactly rather than by apportioning the total.

The per-source document counts come from the published split file rather than a constant,
so this cannot drift from the build it describes.

    python scripts/corpus_source_tokens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corpus_stats import eot_positions

REPO = Path(__file__).resolve().parent.parent
SPLIT = REPO / "metrics/doc_split_2026-08-03.json"
TOKENIZE = REPO / "metrics/tokenize_2026-08-03.json"
OUT = REPO / "metrics/corpus_source_tokens_2026-08-03.json"


def main() -> None:
    split = json.loads(SPLIT.read_text())
    tok = json.loads(TOKENIZE.read_text())
    out = {"meta": {"script": "scripts/corpus_source_tokens.py",
                    "split": SPLIT.name, "tokenization": TOKENIZE.name},
           "streams": {}}
    for side, ids_key in (("train", "train_ids"), ("val", "val_ids")):
        ids = sorted(split[ids_key])
        n_ia = sum(1 for x in ids if x.startswith("ia_"))
        path = REPO / "data" / tok[side]["file"]
        eot = eot_positions(path)
        if eot.size != len(ids):
            raise SystemExit(f"{side}: {eot.size} terminators against {len(ids)} identifiers")
        ia_tokens = int(eot[n_ia - 1]) + 1
        total = int(np.memmap(path, dtype=np.uint16, mode="r").size)
        out["streams"][side] = {
            "file": tok[side]["file"], "documents": len(ids), "tokens": total,
            "ia_documents": n_ia, "wl_documents": len(ids) - n_ia,
            "ia_tokens": ia_tokens, "wl_tokens": total - ia_tokens}
    s = out["streams"]
    c = {k: s["train"][k] + s["val"][k]
         for k in ("documents", "tokens", "ia_documents", "wl_documents",
                   "ia_tokens", "wl_tokens")}
    c["wl_share_pct"] = round(100 * c["wl_tokens"] / c["tokens"], 2)
    out["corpus"] = c
    OUT.write_text(json.dumps(out, indent=1))
    print(f"ia {c['ia_tokens']:,} | wl {c['wl_tokens']:,} | {c['wl_share_pct']}% -> {OUT.name}")


if __name__ == "__main__":
    main()
