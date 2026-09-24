"""Deduplicate a frozen corpus directory: exact and near duplicates, with verdicts.

The measurement algorithm is `src/measure_duplication.py`'s, lifted from the token
stream onto the cleaned `.txt` files so deduplication can run *before* tokenization:
SHA-1 over the whitespace-normalized word sequence for exact duplicates (the `.txt`
analogue of an identical token stream), then MinHash over 5-word shingles, 64
permutations, seed 1337, LSH banding at 8 bands x 8 rows (detection threshold near
0.77 Jaccard, under the 0.8 that counts as a duplicate, so banding over-generates and
the signature comparison decides). Word shingles stand where the paper's first
measurement had token shingles; the parameters are otherwise unchanged and recorded
in the report.

Unlike the measurement script this one renders verdicts: in every cluster the longest
member (by bytes) is kept and the rest are marked dropped. Nothing is deleted here:
the report's `dropped` list feeds the exclusion step, which is where removal is
decided and applied.

    .venv/bin/python3 scripts/dedup_corpus.py --clean data/clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import socket
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MASK = (1 << 61) - 1          # Mersenne prime, keeps the permutation arithmetic in int64
BASE = np.uint64(1_000_003)
PERMS = 64
SHINGLE = 5
ROWS = 8
SEED = 1337

_rng = np.random.default_rng(SEED)
PERM_A = _rng.integers(1, MASK, size=PERMS, dtype=np.uint64)
PERM_B = _rng.integers(0, MASK, size=PERMS, dtype=np.uint64)


def shingle_hashes(words: np.ndarray, width: int) -> np.ndarray:
    """Hash every `width`-word window to one uint64. MinHash is over the shingle set,
    so repeats must collapse before signing."""
    if len(words) < width:
        return np.unique(words)
    h = np.zeros(len(words) - width + 1, dtype=np.uint64)
    for j in range(width):
        h = h * BASE + words[j: len(words) - width + 1 + j]
    return np.unique(h)


def signature(hashes: np.ndarray) -> np.ndarray:
    """MinHash signature: the minimum of each random permutation over the shingle set.
    Looping keeps peak memory at one row instead of a perms x shingles outer product."""
    sig = np.empty(PERMS, dtype=np.uint64)
    for k in range(PERMS):
        sig[k] = np.min((PERM_A[k] * hashes + PERM_B[k]) & np.uint64(MASK))
    return sig


def scan_files(paths: list[str]) -> list:
    out = []
    for p in paths:
        path = Path(p)
        raw = path.read_bytes()
        words = raw.decode("utf-8", errors="replace").split()
        if not words:
            out.append((path.name, len(raw), 0, b"", None))
            continue
        joined = "\x00".join(words).encode("utf-8", errors="replace")
        digest = hashlib.sha1(joined).digest()
        # stable across processes, unlike Python's salted hash(); 32 bits per word is
        # already double the entropy of the token ids the first measurement shingled
        wh = np.fromiter((zlib.crc32(w.encode("utf-8", "replace")) for w in words),
                         dtype=np.uint64, count=len(words))
        out.append((path.name, len(raw), len(words), digest,
                    signature(shingle_hashes(wh, SHINGLE))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--out", default="metrics/dedup_2026-08-03.json")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="signature-agreement Jaccard for 'near duplicate'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N documents")
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(Path(args.clean).glob("*.txt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .txt files under {args.clean}")
    print(f"{len(files):,} files under {args.clean}, {args.workers} workers", flush=True)

    names: list[str] = []
    nbytes: list[int] = []
    nwords: list[int] = []
    digests: list[bytes] = []
    sigs: list = []
    chunk = 1000
    chunks = [[str(f) for f in files[i:i + chunk]] for i in range(0, len(files), chunk)]
    with mp.Pool(args.workers) as pool:
        done = 0
        for part in pool.imap(scan_files, chunks):
            for nm, nb, nw, dg, sg in part:
                names.append(nm); nbytes.append(nb); nwords.append(nw)
                digests.append(dg); sigs.append(sg)
            done += 1
            if done % 20 == 0 or done == len(chunks):
                print(f"  hashed {len(names):,}/{len(files):,} docs "
                      f"({time.time()-t0:.0f}s)", flush=True)

    n_docs = len(names)
    total_bytes = sum(nbytes)

    # ---- pass 1: exact duplicates (identical word sequence) -----------------------
    by_digest = defaultdict(list)
    for i, dg in enumerate(digests):
        if nwords[i]:
            by_digest[dg].append(i)
    exact_groups = [g for g in by_digest.values() if len(g) > 1]

    # ---- pass 2: near duplicates via LSH banding ----------------------------------
    bands = PERMS // ROWS
    buckets = defaultdict(list)
    for i, sg in enumerate(sigs):
        if sg is None:
            continue
        for bnd in range(bands):
            buckets[(bnd, sg[bnd*ROWS:(bnd+1)*ROWS].tobytes())].append(i)

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

    # exact duplicates are near duplicates too; union them first so clusters merge
    for g in exact_groups:
        for i in g[1:]:
            union(g[0], i)

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
                if np.count_nonzero(sigs[i] == sigs[j]) / PERMS >= args.threshold:
                    union(i, j)

    clusters = defaultdict(list)
    for i in range(n_docs):
        if nwords[i]:
            clusters[find(i)].append(i)
    groups = [g for g in clusters.values() if len(g) > 1]

    # ---- verdicts: keep the longest member of each cluster ------------------------
    exact_digest_sets = [set(g) for g in exact_groups]
    out_clusters = []
    dropped: list[str] = []
    dropped_bytes = 0
    for g in sorted(groups, key=len, reverse=True):
        keep = max(g, key=lambda i: nbytes[i])
        kind = "exact" if any(set(g) <= s for s in exact_digest_sets) else "near"
        members = []
        for i in sorted(g, key=lambda i: -nbytes[i]):
            agree = (float(np.count_nonzero(sigs[i] == sigs[keep]) / PERMS)
                     if i != keep and sigs[i] is not None and sigs[keep] is not None
                     else 1.0)
            members.append({"file": names[i], "bytes": nbytes[i], "words": nwords[i],
                            "sig_jaccard_vs_keep": round(agree, 3)})
            if i != keep:
                dropped.append(names[i])
                dropped_bytes += nbytes[i]
        out_clusters.append({"kind": kind, "keep": names[keep], "members": members})

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    report = {
        "meta": {
            "script": "scripts/dedup_corpus.py",
            "corpus_dir": str(args.clean),
            "script_sha256": script_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": round(time.time() - t0, 1),
            "perms": PERMS, "shingle_words": SHINGLE, "lsh_bands": bands,
            "lsh_rows": ROWS, "jaccard_threshold": args.threshold, "seed": SEED,
        },
        "documents": n_docs,
        "bytes": total_bytes,
        "exact_duplicate_groups": len(exact_groups),
        "exact_redundant_documents": sum(len(g) - 1 for g in exact_groups),
        "clusters_total": len(groups),
        "dropped_documents": len(dropped),
        "dropped_bytes": dropped_bytes,
        "dropped_pct_bytes": round(100 * dropped_bytes / total_bytes, 3),
        "candidate_pairs_checked": pairs_checked,
        "largest_clusters": sorted((len(g) for g in groups), reverse=True)[:10],
        "clusters": out_clusters,
        "dropped": sorted(dropped),
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    print(f"\n{n_docs:,} documents, {total_bytes/1e9:.2f} GB")
    print(f"exact groups: {len(exact_groups)} "
          f"({sum(len(g)-1 for g in exact_groups)} redundant docs)")
    print(f"clusters (incl. exact): {len(groups)}; dropped {len(dropped):,} docs, "
          f"{dropped_bytes/1e6:.1f} MB ({report['dropped_pct_bytes']}% of bytes)")
    print(f"candidate pairs checked: {pairs_checked:,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
