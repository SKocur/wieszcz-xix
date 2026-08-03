"""Measure duplication in the frozen token corpus.

No cross-source deduplication was done. A work appearing on both sides of the
train/validation boundary turns validation into a recall test, so the rate matters.

Reads `tokens.bin` rather than the source `.txt` files, so it characterises the corpus
the models were trained on rather than a directory that has kept growing.

Two passes: SHA-1 over each document's token bytes for exact duplicates, then MinHash
over token n-gram shingles with LSH banding for editions that differ in front matter,
OCR noise or line breaks. Writes a JSON report and changes nothing on disk.

    python src/measure_duplication.py --tokens data/clean/tokens.bin --out metrics/duplication.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EOT = 0                       # appended after every document by build_token_cache
MASK = (1 << 61) - 1          # Mersenne prime, keeps the permutation arithmetic in int64
BASE = np.uint64(1_000_003)


def document_bounds(path: Path, chunk_tokens: int = 1 << 26) -> np.ndarray:
    """Start offset of every document, found by scanning for the EOT separator.

    Chunked: a single boolean mask over 5.4 billion tokens costs 5.4 GB of RAM for a
    result of a few hundred thousand integers."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ends = []
    for start in range(0, len(data), chunk_tokens):
        block = np.asarray(data[start : start + chunk_tokens])
        ends.append(np.flatnonzero(block == EOT).astype(np.int64) + start)
    ends = np.concatenate(ends)
    starts = np.empty(len(ends), dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    return np.stack([starts, ends])            # ends exclude the EOT itself


def shingle_hashes(tokens: np.ndarray, width: int) -> np.ndarray:
    """Hash every `width`-token window to one uint64. MinHash is over the shingle set,
    so repeats must collapse before signing."""
    if len(tokens) < width:
        return np.unique(tokens.astype(np.uint64))
    t = tokens.astype(np.uint64)
    h = np.zeros(len(t) - width + 1, dtype=np.uint64)
    for j in range(width):
        h = h * BASE + t[j : len(t) - width + 1 + j]
    return np.unique(h)


def signature(hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature: the minimum of each random permutation over the shingle set."""
    # The vectorised form allocates a perms x shingles outer product, gigabytes for a
    # long document. Looping keeps peak memory at one row.
    sig = np.empty(len(a), dtype=np.uint64)
    for k in range(len(a)):
        sig[k] = np.min((a[k] * hashes + b[k]) & np.uint64(MASK))
    return sig


def process_range(args) -> list:
    path, bounds, lo, hi, width, a, b = args
    data = np.memmap(path, dtype=np.uint16, mode="r")
    out = []
    for i in range(lo, hi):
        s, e = bounds[0, i], bounds[1, i]
        tokens = np.asarray(data[s:e])
        if len(tokens) == 0:
            out.append((i, 0, b"", None))
            continue
        digest = hashlib.sha1(tokens.tobytes()).digest()
        out.append((i, int(len(tokens)), digest, signature(shingle_hashes(tokens, width), a, b)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="data/clean/tokens.bin")
    p.add_argument("--out", default="metrics/duplication.json")
    p.add_argument("--perms", type=int, default=64, help="MinHash permutations (8 bands x 8 rows)")
    p.add_argument("--shingle", type=int, default=5, help="tokens per shingle")
    p.add_argument("--threshold", type=float, default=0.8, help="Jaccard for 'near duplicate'")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only the first N documents (for a smoke test)")
    p.add_argument("--dump-dir", default=None,
                   help="also write the per-document raw data and cluster membership here")
    args = p.parse_args()

    path = Path(args.tokens)
    t0 = time.time()
    bounds = document_bounds(path)
    n_docs = bounds.shape[1] if not args.limit else min(args.limit, bounds.shape[1])
    total_tokens = int(bounds[1, :n_docs].sum() - bounds[0, :n_docs].sum())
    print(f"{n_docs:,} documents, {total_tokens:,} tokens, boundaries in {time.time()-t0:.1f}s",
          flush=True)

    rng = np.random.default_rng(1337)
    a = rng.integers(1, MASK, size=args.perms, dtype=np.uint64)
    b = rng.integers(0, MASK, size=args.perms, dtype=np.uint64)

    step = max(1, n_docs // (args.workers * 8))
    tasks = [(str(path), bounds, lo, min(lo + step, n_docs), args.shingle, a, b)
             for lo in range(0, n_docs, step)]

    lengths = np.zeros(n_docs, dtype=np.int64)
    digests, sigs = [None] * n_docs, [None] * n_docs
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(process_range, tasks):
            for i, ln, dg, sg in chunk:
                lengths[i], digests[i], sigs[i] = ln, dg, sg
            done += len(chunk)
            print(f"\r  {done:,}/{n_docs:,} docs | {time.time()-t0:.0f}s", end="", flush=True)
    print()

    # ---- pass 1: exact duplicates -------------------------------------------------
    by_digest = defaultdict(list)
    for i, dg in enumerate(digests):
        if lengths[i]:
            by_digest[dg].append(i)
    exact_groups = [g for g in by_digest.values() if len(g) > 1]
    exact_redundant = sum(int(lengths[g[1:]].sum()) for g in exact_groups)

    # ---- pass 2: near duplicates via LSH banding ----------------------------------
    # 8 bands of 8 rows puts the detection threshold near 0.77 Jaccard, under the 0.8 that
    # counts as a duplicate, so banding over-generates and the comparison below decides.
    rows = 8
    bands = args.perms // rows
    buckets = defaultdict(list)
    for i, sg in enumerate(sigs):
        if sg is None:
            continue
        for bnd in range(bands):
            buckets[(bnd, sg[bnd*rows:(bnd+1)*rows].tobytes())].append(i)

    parent = list(range(n_docs))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    pairs_checked = 0
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:   # a giant bucket is boilerplate
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if find(i) == find(j):
                    continue
                pairs_checked += 1
                if np.count_nonzero(sigs[i] == sigs[j]) / args.perms >= args.threshold:
                    union(i, j)

    clusters = defaultdict(list)
    for i in range(n_docs):
        if lengths[i]:
            clusters[find(i)].append(i)
    near_groups = [g for g in clusters.values() if len(g) > 1]
    # Keeping the longest copy, so the rest is what deduplication would remove.
    near_redundant = 0
    for g in near_groups:
        keep = max(g, key=lambda i: lengths[i])
        near_redundant += int(sum(lengths[i] for i in g if i != keep))

    report = {
        "tokens_file": str(path), "documents": int(n_docs), "tokens": total_tokens,
        "perms": args.perms, "shingle_tokens": args.shingle, "jaccard_threshold": args.threshold,
        "seed": 1337, "lsh_bands": bands, "lsh_rows": rows,
        "exact_duplicate_groups": len(exact_groups),
        "exact_redundant_documents": sum(len(g) - 1 for g in exact_groups),
        "exact_redundant_tokens": exact_redundant,
        "exact_redundant_pct": round(100 * exact_redundant / total_tokens, 3),
        "near_duplicate_clusters": len(near_groups),
        "near_redundant_documents": sum(len(g) - 1 for g in near_groups),
        "near_redundant_tokens": near_redundant,
        "near_redundant_pct": round(100 * near_redundant / total_tokens, 3),
        "candidate_pairs_checked": pairs_checked,
        "largest_clusters": sorted((len(g) for g in near_groups), reverse=True)[:10],
        "wall_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dump_dir:
        # Signatures let the threshold move without another pass over 5.4B tokens;
        # offsets let a flagged document be decoded out of tokens.bin and read.
        d = Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "per_document.npz",
            start=bounds[0, :n_docs], end=bounds[1, :n_docs], length=lengths,
            sha1=np.frombuffer(b"".join(dg if dg else b"\0"*20 for dg in digests),
                               dtype=np.uint8).reshape(n_docs, 20),
            signature=np.stack([s if s is not None else np.zeros(args.perms, np.uint64)
                                for s in sigs]),
        )
        groups = []
        for g in exact_groups:
            groups.append({"kind": "exact", "members": [int(i) for i in g],
                           "lengths": [int(lengths[i]) for i in g],
                           "starts": [int(bounds[0, i]) for i in g]})
        for g in near_groups:
            keep = max(g, key=lambda i: lengths[i])
            groups.append({"kind": "near", "members": [int(i) for i in g],
                           "lengths": [int(lengths[i]) for i in g],
                           "starts": [int(bounds[0, i]) for i in g], "keep": int(keep)})
        (d / "clusters.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
        print(f"dumped per-document arrays and {len(groups)} clusters to {d}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
