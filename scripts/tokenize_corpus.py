"""Tokenize the cleaned corpus into training and validation streams.

Consumes the document split verbatim (train and validation id lists, which already
embed the exclusion list), tokenizes with the shipped tokenizer, and writes one
uint16 stream per side with an <|endoftext|> terminator (id 0) after every document.
Documents are written in sorted-id order: the Internet Archive block followed by the
Wolne Lektury block, as in the first build.

Wolne Lektury files get one extra pass the scanned sources already had at crawl
time: lines matching the cleaner's digitisation-boilerplate battery are dropped,
which removes the WL colophon (ISBN, licence, foundation URL), after the exclusion
list, the only modern text left in the corpus.

The training stream's filename carries the measured token count
(`tokens_frozen_<N>B.bin`), and the report records SHA-256 of both streams, the
per-source token counts, and the two-epoch step count at the training batch of
262,144 tokens. Reproducible from the corpus directory, the split report and the
tokenizer alone.

    .venv/bin/python3 scripts/tokenize_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from clean_ocr import is_anachronism  # noqa: E402

EOT = 0
BATCH_TOKENS = 262_144
EPOCHS = 2

_tok = None
_clean_dir: Path | None = None


def _init(tokenizer_dir: str, clean_dir: str) -> None:
    global _tok, _clean_dir
    from tokenizers import ByteLevelBPETokenizer
    _tok = ByteLevelBPETokenizer(f"{tokenizer_dir}/vocab.json",
                                 f"{tokenizer_dir}/merges.txt")
    _clean_dir = Path(clean_dir)


def encode_chunk(doc_ids: list[str]) -> tuple[bytes, int, int, int]:
    """Tokenize a run of documents; returns the packed uint16 stream, its token
    count, and how many colophon lines and bytes the wl_ pass dropped."""
    out: list[np.ndarray] = []
    total = 0
    lines_dropped = bytes_dropped = 0
    for doc in doc_ids:
        text = (_clean_dir / f"{doc}.txt").read_text(encoding="utf-8",
                                                     errors="replace")
        if doc.startswith("wl_"):
            kept_lines = []
            for ln in text.split("\n"):
                if is_anachronism(ln):
                    lines_dropped += 1
                    bytes_dropped += len(ln.encode("utf-8")) + 1
                else:
                    kept_lines.append(ln)
            text = "\n".join(kept_lines)
        ids = _tok.encode(text).ids
        ids.append(EOT)
        out.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
    return b"".join(a.tobytes() for a in out), total, lines_dropped, bytes_dropped


def write_stream(docs: list[str], path: Path, pool: mp.pool.Pool,
                 chunk: int = 200) -> dict:
    t0 = time.time()
    chunks = [docs[i:i + chunk] for i in range(0, len(docs), chunk)]
    tokens = 0
    lines_dropped = bytes_dropped = 0
    sha = hashlib.sha256()
    with path.open("wb") as fh:
        done = 0
        for blob, n, ld, bd in pool.imap(encode_chunk, chunks):
            fh.write(blob)
            sha.update(blob)
            tokens += n
            lines_dropped += ld
            bytes_dropped += bd
            done += 1
            if done % 50 == 0 or done == len(chunks):
                print(f"  {path.name}: {min(done*chunk, len(docs)):,}/{len(docs):,} "
                      f"docs, {tokens/1e9:.2f}B tokens ({time.time()-t0:.0f}s)",
                      flush=True)
    return {"file": path.name, "documents": len(docs), "tokens": tokens,
            "bytes": path.stat().st_size, "sha256": sha.hexdigest(),
            "wl_colophon_lines_dropped": lines_dropped,
            "wl_colophon_bytes_dropped": bytes_dropped,
            "elapsed_seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(REPO / "data/clean"))
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer"))
    ap.add_argument("--out-dir", default=str(REPO / "data"))
    ap.add_argument("--freeze-date", default="2026-08-03")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N train documents")
    args = ap.parse_args()

    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    train = sorted(split["train_ids"])
    val = sorted(split["val_ids"])
    if args.limit:
        train, val = train[: args.limit], val[: max(1, args.limit // 100)]
    print(f"train {len(train):,} docs, val {len(val):,} docs; "
          f"{args.workers} workers", flush=True)

    out_dir = Path(args.out_dir)
    train_tmp = out_dir / "tokens_frozen.tmp.bin"
    val_path = out_dir / f"val_{args.freeze_date}.bin"

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.tokenizer, args.clean)) as pool:
        train_info = write_stream(train, train_tmp, pool)
        val_info = write_stream(val, val_path, pool)

    n = f"{train_info['tokens']/1e9:.2f}B"
    train_path = out_dir / f"tokens_frozen_{n}.bin"
    train_tmp.rename(train_path)
    train_info["file"] = train_path.name

    steps = EPOCHS * train_info["tokens"] / BATCH_TOKENS
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    report = {
        "meta": {
            "script": "scripts/tokenize_corpus.py",
            "freeze_date": args.freeze_date,
            "corpus_dir": args.clean,
            "split": args.split,
            "tokenizer": args.tokenizer,
            "eot_id": EOT,
            "script_sha256": script_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
        },
        "train": train_info,
        "val": val_info,
        "tokens_total": train_info["tokens"] + val_info["tokens"],
        "batch_tokens": BATCH_TOKENS,
        "epochs": EPOCHS,
        "train_steps_exact": round(steps, 2),
        "train_steps": -(-train_info["tokens"] * EPOCHS // BATCH_TOKENS),
    }
    out = REPO / f"metrics/tokenize_{args.freeze_date}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    colophon = (train_info["wl_colophon_lines_dropped"]
                + val_info["wl_colophon_lines_dropped"])
    print(f"\ntrain: {train_info['tokens']:,} tokens -> {train_path.name}")
    print(f"val  : {val_info['tokens']:,} tokens -> {val_path.name}")
    print(f"wl colophon: {colophon:,} lines dropped")
    print(f"steps for {EPOCHS} epochs @ {BATCH_TOKENS:,} tok: {report['train_steps']:,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
