"""Measure text leakage across the train/validation split.

Deduplication ran before the split was drawn, so no cluster it found can straddle
the boundary, since one member of each cluster survives at all. What that guarantees
nothing about is overlap *below* the 0.8 duplicate threshold: another edition of a
validation document sitting in train, or a validation poem contained inside a
training-side collected volume. Jaccard cannot see containment (a small document
inside a large one shares almost nothing by union), so this measures both:

1. **Containment**: for every validation document, the fraction of its 5-word
   shingles that occur anywhere in the training side. Common phrases give every
   document a nonzero floor; what matters is the tail near 1.0.
2. **Low-threshold LSH**: bands of 4 rows over the MinHash signatures, sensitive
   from roughly Jaccard 0.5, listing the best training match per flagged
   validation document.

Shingles are literal, so orthographic variants of the same work still evade both
measures, the same blind spot the dedup pass records.

    .venv/bin/python3 scripts/check_split_leakage.py
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
SHINGLE = 5
PERMS = 64
ROWS = 4                       # 16 bands of 4: detection threshold near 0.5 Jaccard
MASK = (1 << 61) - 1
BASE = np.uint64(1_000_003)

_rng = np.random.default_rng(1337)
PERM_A = _rng.integers(1, MASK, size=PERMS, dtype=np.uint64)
PERM_B = _rng.integers(0, MASK, size=PERMS, dtype=np.uint64)

_clean: Path | None = None
_val_shingles: np.ndarray | None = None


def shingles_of(path: Path) -> np.ndarray:
    words = path.read_text(encoding="utf-8", errors="replace").split()
    if not words:
        return np.empty(0, dtype=np.uint64)
    wh = np.fromiter((zlib.crc32(w.encode("utf-8", "replace")) for w in words),
                     dtype=np.uint64, count=len(words))
    if len(wh) < SHINGLE:
        return np.unique(wh)
    h = np.zeros(len(wh) - SHINGLE + 1, dtype=np.uint64)
    for j in range(SHINGLE):
        h = h * BASE + wh[j: len(wh) - SHINGLE + 1 + j]
    return np.unique(h)


def signature(hashes: np.ndarray) -> np.ndarray | None:
    if not len(hashes):
        return None
    sig = np.empty(PERMS, dtype=np.uint64)
    for k in range(PERMS):
        sig[k] = np.min((PERM_A[k] * hashes + PERM_B[k]) & np.uint64(MASK))
    return sig


def _init(clean: str, val_shingle_file: str | None) -> None:
    global _clean, _val_shingles
    _clean = Path(clean)
    if val_shingle_file:
        _val_shingles = np.load(val_shingle_file, mmap_mode="r")


def sig_chunk(docs: list[str]) -> list:
    return [(d, signature(shingles_of(_clean / f"{d}.txt"))) for d in docs]


def hit_chunk(docs: list[str]) -> np.ndarray:
    """Positions in the sorted validation-shingle array that occur in these
    training documents."""
    hits: list[np.ndarray] = []
    for d in docs:
        sh = shingles_of(_clean / f"{d}.txt")
        if not len(sh):
            continue
        pos = np.searchsorted(_val_shingles, sh)
        valid = pos < len(_val_shingles)
        pos, sh = pos[valid], sh[valid]
        hits.append(pos[np.take(_val_shingles, pos) == sh])
    return np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--out", default="metrics/split_leakage_2026-08-03.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    train, val = sorted(split["train_ids"]), sorted(split["val_ids"])
    print(f"train {len(train):,}, val {len(val):,}", flush=True)

    # ---- validation shingle table: (sorted shingle, owner doc) ----------------
    val_sh: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    clean = Path(args.clean)
    for i, d in enumerate(val):
        sh = shingles_of(clean / f"{d}.txt")
        val_sh.append(sh)
        owners.append(np.full(len(sh), i, dtype=np.int32))
    flat = np.concatenate(val_sh)
    owner = np.concatenate(owners)
    order = np.argsort(flat, kind="stable")
    flat, owner = flat[order], owner[order]
    sizes = np.array([len(s) for s in val_sh], dtype=np.int64)
    tmp = REPO / "data/val_shingles.tmp.npy"
    np.save(tmp, flat)
    print(f"validation shingles: {len(flat):,} ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass over train: which validation shingles occur there --------------
    seen = np.zeros(len(flat), dtype=bool)
    chunk = 500
    chunks = [train[i:i + chunk] for i in range(0, len(train), chunk)]
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.clean, str(tmp))) as pool:
        for done, pos in enumerate(pool.imap_unordered(hit_chunk, chunks), 1):
            if len(pos):
                seen[pos] = True
            if done % 50 == 0 or done == len(chunks):
                print(f"  train scan {done}/{len(chunks)} chunks, "
                      f"matched {int(seen.sum()):,} shingles "
                      f"({time.time()-t0:.0f}s)", flush=True)

    matched_per_doc = np.bincount(owner[seen], minlength=len(val))
    containment = matched_per_doc / np.maximum(sizes, 1)
    order = np.argsort(-containment)
    hist = {f">={t}": int((containment >= t).sum())
            for t in (0.3, 0.5, 0.7, 0.9, 0.95)}

    # ---- low-threshold LSH on signatures -------------------------------------
    print("signatures for LSH...", flush=True)
    sigs: dict[str, np.ndarray | None] = {}
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.clean, None)) as pool:
        all_docs = train + val
        for part in pool.imap_unordered(
                sig_chunk, [all_docs[i:i + chunk]
                            for i in range(0, len(all_docs), chunk)]):
            sigs.update(part)
    val_set = set(val)
    buckets = defaultdict(list)
    for d, sg in sigs.items():
        if sg is None:
            continue
        for b in range(PERMS // ROWS):
            buckets[(b, sg[b*ROWS:(b+1)*ROWS].tobytes())].append(d)
    best: dict[str, tuple[str, float]] = {}
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            continue
        vs = [d for d in members if d in val_set]
        ts = [d for d in members if d not in val_set]
        for v in vs:
            for tdoc in ts:
                agree = float(np.count_nonzero(sigs[v] == sigs[tdoc]) / PERMS)
                if agree > best.get(v, ("", 0.0))[1]:
                    best[v] = (tdoc, agree)
    pairs = sorted(((v, t, a) for v, (t, a) in best.items() if a >= 0.4),
                   key=lambda x: -x[2])

    tmp.unlink()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    report = {
        "meta": {
            "script": "scripts/check_split_leakage.py",
            "split": args.split, "shingle_words": SHINGLE,
            "lsh_rows": ROWS, "perms": PERMS,
            "script_sha256": script_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "val_documents": len(val),
        "containment_mean": round(float(containment.mean()), 4),
        "containment_median": round(float(np.median(containment)), 4),
        "containment_p90": round(float(np.quantile(containment, 0.9)), 4),
        "containment_max": round(float(containment.max()), 4),
        "containment_hist": hist,
        "top_containment": [
            {"val": val[i], "containment": round(float(containment[i]), 3),
             "shingles": int(sizes[i])}
            for i in order[:20]],
        "containment_by_doc": {val[i]: round(float(containment[i]), 4)
                               for i in range(len(val))},
        "lsh_pairs_agree_ge_0.4": [
            {"val": v, "train": t, "sig_agreement": round(a, 3)}
            for v, t, a in pairs[:20]],
        "lsh_pairs_count": len(pairs),
    }
    out = REPO / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"\ncontainment: median {report['containment_median']}, "
          f"p90 {report['containment_p90']}, max {report['containment_max']}")
    print("hist:", hist)
    print(f"LSH pairs >=0.4 agreement: {len(pairs)}")
    for v, t, a in pairs[:8]:
        print(f"  {a:.2f}  {v}  ~  {t}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
