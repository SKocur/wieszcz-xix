"""What the 8,000-token vocabulary costs, measured on the corpus it was trained for.

The paper argues for a small vocabulary in one sentence and gives no number, which is the
shape of claim a reviewer is right to refuse. Two numbers settle it, and neither needs a
GPU.

*Fertility.* Bytes per token on held-out text, for our vocabulary and for the comparators'.
This is the quantity that decides everything downstream: how much text a 1,024-token window
holds, how many FLOPs a byte of corpus costs, and the conversion between the nats per token
the ladder reports and the bits per byte any other model can be compared in. A vocabulary
is too small when it spends materially more tokens on the same text than the alternatives;
whether ours does is measurable rather than arguable.

*Embedding share.* A tied embedding is `vocab x d_model` parameters that a larger vocabulary
grows linearly, while the rest of the model stays fixed. At the small end of a ladder this
is the whole argument, and it reverses as the model grows, which is worth reporting
honestly, because it means the choice is right for 47M for a reason that is weak by 349M.

    python scripts/tokenizer_fertility.py --hf ~/models/bielik-1.5b-v3 ~/models/papugapt2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tokenizers import ByteLevelBPETokenizer

from train_tokenizer import MAX_TRAIN_BYTES, SPECIAL_TOKENS, sample_corpus

LN2 = math.log(2)
RUNGS = (("47M", 576, 47_105_088), ("107M", 768, 106_859_264), ("349M", 1152, 349_069_440))
ALTERNATIVES = (8_000, 16_000, 32_000, 50_000, 128_000)


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def sample_documents(clean: Path, split_path: Path, n_docs: int, doc_bytes: int,
                     seed: int) -> list[dict]:
    """Held-out documents, kept whole up to a byte cap and tagged by source.

    Per source as well as pooled: the two halves of this corpus differ in ways a tokenizer
    feels, scanned press carries OCR noise that no vocabulary has merges for, while the
    transcriptions are clean, so a pooled figure alone would hide whichever half is
    doing the work.
    """
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = list(split["val_ids"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    out = []
    for did in ids:
        if len(out) >= n_docs:
            break
        f = clean / f"{did}.txt"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")[:doc_bytes]
        if len(text.strip()) < 500:
            continue
        out.append({"id": did, "source": "wl" if did.startswith("wl_") else "ia",
                    "text": text})
    return out


def fertility(encode, docs: list[dict]) -> dict:
    """Bytes per token, pooled and per source. Bytes are UTF-8, which is what BPB counts."""
    agg: dict[str, list[int]] = {}
    for d in docs:
        n_bytes = len(d["text"].encode("utf-8"))
        n_tokens = len(encode(d["text"]))
        for key in ("all", d["source"]):
            b, t = agg.setdefault(key, [0, 0])
            agg[key] = [b + n_bytes, t + n_tokens]
    return {k: {"bytes": b, "tokens": t, "bytes_per_token": round(b / t, 4),
                "tokens_per_1k_bytes": round(1000 * t / b, 2)}
            for k, (b, t) in sorted(agg.items())}


def embedding_shares() -> dict:
    """Tied-embedding parameters as a share of the budget, at each rung and vocabulary."""
    out = {}
    for name, d_model, params in RUNGS:
        base = params - 8_000 * d_model
        out[name] = {
            "d_model": d_model,
            "params_at_8k": params,
            "non_embedding_params": base,
            "by_vocab": {
                str(v): {
                    "embedding_params": v * d_model,
                    "total_params": base + v * d_model,
                    "embedding_share": round((v * d_model) / (base + v * d_model), 4),
                } for v in ALTERNATIVES
            },
        }
    return out


def counterfactual_vocabularies(sizes: list[int], clean: Path, docs: list[dict],
                                work: Path) -> dict:
    """Train the vocabularies we did not choose, on the sample we did, and measure them.

    This is the counterfactual the choice of 8,000 needs and the only one available without
    retraining a rung: what a larger merge table would actually buy on this corpus, rather
    than what it buys on the web text the comparators were built for. Everything but
    `vocab_size` is held to the released tokenizer's procedure, same corpus sample, same
    seed, same minimum frequency, so the difference between two rows is the vocabulary
    and nothing else.
    """
    paths = [q for q in clean.glob("*.txt") if not q.name.startswith(".")]
    files = [str(q) for q in sample_corpus(paths, MAX_TRAIN_BYTES)]
    work.mkdir(parents=True, exist_ok=True)
    out = {}
    for size in sizes:
        print(f"  training {size} on {len(files)} files ...", flush=True)
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=size, min_frequency=2,
                  special_tokens=SPECIAL_TOKENS)
        d = work / f"v{size}"
        d.mkdir(exist_ok=True)
        tok.save_model(str(d))
        f = fertility(lambda t: tok.encode(t).ids, docs)
        out[str(size)] = {"vocab_size": tok.get_vocab_size(), "fertility": f}
        print(f"    {size}: {f['all']['bytes_per_token']} B/tok", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", nargs="*", default=[],
                    help="comparator tokenizers, by path or hub id")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--doc-bytes", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sweep", type=int, nargs="*", default=[],
                    help="vocabulary sizes to train and measure as counterfactuals")
    ap.add_argument("--sweep-dir", default="output/tokenizer_sweep")
    ap.add_argument("--out", default="metrics/tokenizer_fertility.json")
    args = ap.parse_args()

    docs = sample_documents(REPO / args.clean, REPO / args.split, args.docs,
                            args.doc_bytes, args.seed)
    n_wl = sum(1 for d in docs if d["source"] == "wl")
    print(f"{len(docs)} held-out documents ({n_wl} transcriptions, {len(docs)-n_wl} scans), "
          f"{sum(len(d['text'].encode('utf-8'))for d in docs)/1e6:.1f} MB", flush=True)

    ours = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                 str(REPO / "tokenizer/merges.txt"))
    report = {
        "meta": {
            "script": "scripts/tokenizer_fertility.py",
            "script_sha256": script_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "documents": len(docs), "doc_bytes_cap": args.doc_bytes, "seed": args.seed,
        },
        "tokenizers": {},
        "embedding_shares": embedding_shares(),
    }

    report["tokenizers"]["wieszcz-8k"] = {
        "vocab_size": ours.get_vocab_size(),
        "fertility": fertility(lambda t: ours.encode(t).ids, docs),
    }
    print(f"  wieszcz-8k: {report['tokenizers']['wieszcz-8k']['fertility']['all']}",
          flush=True)

    for path in args.hf:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        name = Path(path).name
        report["tokenizers"][name] = {
            "vocab_size": tok.vocab_size,
            "fertility": fertility(
                lambda t: tok(t, add_special_tokens=False)["input_ids"], docs),
        }
        print(f"  {name}: {report['tokenizers'][name]['fertility']['all']}", flush=True)

    if args.sweep:
        print("counterfactual vocabularies", flush=True)
        report["counterfactual_vocabularies"] = counterfactual_vocabularies(
            args.sweep, REPO / args.clean, docs, REPO / args.sweep_dir)

    ours_bpt = report["tokenizers"]["wieszcz-8k"]["fertility"]["all"]["bytes_per_token"]
    report["relative_to_ours"] = {
        name: round(v["fertility"]["all"]["bytes_per_token"] / ours_bpt, 4)
        for name, v in report["tokenizers"].items()
    }
    # The conversion the ladder needs to be quotable against models with other vocabularies.
    report["nats_per_token_to_bits_per_byte"] = round(1.0 / (ours_bpt * LN2), 6)

    out = REPO / args.out
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nbytes/token relative to ours: {report['relative_to_ours']}")
    print(f"1 nat/token = {report['nats_per_token_to_bits_per_byte']:.4f} bits/byte")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
