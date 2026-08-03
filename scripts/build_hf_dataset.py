"""Materialise the frozen corpus as a Hugging Face dataset directory.

The frozen token file plus the tokenizer reconstruct every document byte-for-byte:
byte-level BPE round-trips losslessly, and the provenance ledger records each document's
original size. That makes this build self-verifying — every decoded document's UTF-8
length is checked against the ledger, so a truncated bin, a wrong tokenizer or a
misaligned ledger fails loudly instead of shipping a subtly wrong corpus.

Output is the loader-script-free layout `load_dataset` reads directly: parquet shards
under data/, one `train` split, columns id / source / source_identifier / text. The
dataset card (README.md with YAML) is written by hand, not here — numbers to quote are
printed at the end.

`--exclude` drops the documents listed in an exclusion file (the post-1918 content
audit's output) from the *release* without touching the frozen build: excluded documents
are still decoded and byte-verified against the ledger, so the self-check covers the
whole stream, they just are not written out. The skipped ids, tokens and bytes are
reported so the dataset card can state exactly what the release omits.

    python scripts/build_hf_dataset.py --out ../dataset/wieszcz-xix-corpus \
        --exclude metrics/post1918_exclusions.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer

REPO = Path(__file__).resolve().parent.parent
TOKENS = REPO / "data/clean/tokens_frozen_5.40B.bin"
LEDGER = REPO / "metrics/provenance_ledger.csv.gz"
EOT = 0
CHUNK = 1 << 26                 # tokens per read; ~128 MB of uint16 at a time
SHARD_TEXT_BYTES = 500 << 20    # flush a shard when its raw text reaches ~500 MB
DECODE_BATCH = 256

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("source_identifier", pa.string()),
    ("text", pa.large_string()),
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "dataset/wieszcz-xix-corpus"))
    ap.add_argument("--exclude", help="JSON with an 'ids' list (or a bare list) of "
                                      "document ids to omit from the release")
    args = ap.parse_args()

    excluded_ids: set[str] = set()
    if args.exclude:
        import json
        spec = json.loads(Path(args.exclude).read_text(encoding="utf-8"))
        excluded_ids = set(spec["ids"] if isinstance(spec, dict) else spec)
        print(f"excluding {len(excluded_ids):,} documents per {args.exclude}")

    out = Path(args.out) / "data"
    out.mkdir(parents=True, exist_ok=True)

    ledger = list(csv.DictReader(gzip.open(LEDGER, "rt")))
    tok = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                str(REPO / "tokenizer/merges.txt"))
    arr = np.memmap(TOKENS, dtype=np.uint16, mode="r")

    t0 = time.time()
    doc_idx = shard_idx = 0
    total_bytes = 0
    carry: list[int] = []        # tokens of a document split across chunk borders
    pending: list[list[int]] = []
    rows = {k: [] for k in ("id", "source", "source_identifier", "text")}
    rows_bytes = 0
    shard_paths: list[Path] = []

    skipped = {"docs": 0, "tokens": 0, "bytes": 0}

    def flush_docs() -> None:
        nonlocal doc_idx, total_bytes, rows_bytes
        for toks, text in zip(pending, tok.decode_batch(pending)):
            led = ledger[doc_idx]
            b = len(text.encode("utf-8"))
            if b != int(led["bytes"]):
                raise SystemExit(f"doc {doc_idx} ({led['document_id']}): decoded {b} B, "
                                 f"ledger says {led['bytes']} B — bin/ledger mismatch")
            doc_idx += 1
            if led["document_id"] in excluded_ids:
                skipped["docs"] += 1
                skipped["tokens"] += len(toks)
                skipped["bytes"] += b
                continue
            rows["id"].append(led["document_id"])
            rows["source"].append(led["source"])
            rows["source_identifier"].append(led["source_identifier"])
            rows["text"].append(text)
            rows_bytes += b
            total_bytes += b
        pending.clear()

    def flush_shard() -> None:
        nonlocal rows_bytes, shard_idx
        path = out / f"train-part-{shard_idx:05d}.parquet"
        pq.write_table(pa.table(rows, schema=SCHEMA), path, compression="zstd")
        shard_paths.append(path)
        for v in rows.values():
            v.clear()
        rows_bytes = 0
        shard_idx += 1

    for start in range(0, arr.size, CHUNK):
        block = np.asarray(arr[start:start + CHUNK])
        prev = 0
        for e in np.flatnonzero(block == EOT):
            pending.append(carry + block[prev:e].tolist() if carry else block[prev:e].tolist())
            carry = []
            prev = e + 1
            if len(pending) >= DECODE_BATCH:
                flush_docs()
                if rows_bytes >= SHARD_TEXT_BYTES:
                    flush_shard()
        carry += block[prev:].tolist()
        el = time.time() - t0
        print(f"{doc_idx:>7,}/{len(ledger):,} docs  {total_bytes/1e9:6.2f} GB  "
              f"shard {shard_idx}  {el:6.0f}s", flush=True)

    if carry:
        pending.append(carry)
    if pending:
        flush_docs()
    if rows["id"]:
        flush_shard()

    if doc_idx != len(ledger):
        raise SystemExit(f"decoded {doc_idx} docs, ledger has {len(ledger)}")

    # HF's sharded-file convention encodes the total count in every name.
    n = len(shard_paths)
    for i, p in enumerate(shard_paths):
        p.rename(out / f"train-{i:05d}-of-{n:05d}.parquet")

    pq_bytes = sum(f.stat().st_size for f in out.glob("*.parquet"))
    released = doc_idx - skipped["docs"]
    print(f"\ndone: {released:,} documents released, {total_bytes/1e9:.2f} GB text, "
          f"{n} shards, {pq_bytes/1e9:.2f} GB parquet, {(time.time()-t0)/60:.1f} min")
    if excluded_ids:
        missing = excluded_ids - {l["document_id"] for l in ledger}
        print(f"excluded from release: {skipped['docs']:,} documents, "
              f"{skipped['tokens']:,} content tokens, {skipped['bytes']/1e6:.1f} MB text")
        if missing:
            print(f"WARNING: {len(missing)} exclusion ids not found in ledger")
    print(f"every document verified against the ledger byte length ({doc_idx:,} decoded)")


if __name__ == "__main__":
    main()
