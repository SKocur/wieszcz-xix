"""Package a training checkpoint into a self-contained model directory.

A checkpoint alone is not a model: it carries no tokenizer, and the tokenizer it needs is
not recoverable from it. Loading a checkpoint against the wrong vocabulary produces fluent
nonsense rather than an error, so the two travel together here, both hashed.

Optimizer state is dropped. It is over half the file and nothing outside a resume can use
it, so the exported weights are roughly half the size of the checkpoint.

    python scripts/export_model.py \
        --ckpt checkpoints/wieszcz_350m_2026-07-25/final.pt \
        --name wieszcz-349m-2026-07-25 \
        --eval metrics/wieszcz_350m_2026-07-25_eval.json \
        --log  metrics/wieszcz_350m_2026-07-25.out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from train import GPT

REPO = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = REPO / "tokenizer"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_from_log(log: Path | None) -> tuple[int | None, int | None]:
    """(corpus tokens, val tokens) from a training log's first Corpus line."""
    if not log or not log.exists():
        return None, None
    for line in log.read_text(errors="replace").splitlines()[:20]:
        m = re.search(r"Corpus:\s*([\d,]+) tokens \(([\d,]+) val\)", line)
        if m:
            return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    return None, None


DESCRIPTIONS = [
    (["model.pt"], "weights and config, no optimizer state"),
    (["vocab.json", "merges.txt"], "the tokenizer this model requires"),
    (["config.json"], "the configuration the run actually used, read out of the checkpoint"),
    (["manifest.json"], "the run's provenance: seed, corpus size, tokenizer hashes"),
    (["eval.json"], "held-out evaluation report"),
    (["eval.windows.npy"], "per-window losses, for a paired test against another model"),
    (["train.log"], "the run's step and validation history"),
]


def file_table(payload: list[str]) -> list[str]:
    """One row per file actually in the bundle, so the card cannot drift from SHA256SUMS."""
    rows = []
    for files, what in DESCRIPTIONS:
        present = [f for f in files if f in payload]
        if present:
            rows.append(f"| {', '.join(f'`{f}`' for f in present)} | {what} |")
    rows.append("| `SHA256SUMS` | `shasum -a 256 -c SHA256SUMS` |")
    return rows


def card(name: str, cfg: dict, step: int, n_params: int, tok_sha: dict,
         corpus: int | None, val: int | None, ev: dict | None, payload: list[str]) -> str:
    tokens_seen = cfg["batch_size"] * cfg.get("grad_accum", 1) * cfg["block_size"] * cfg["max_steps"]
    lines = [
        f"# {name}",
        "",
        f"{n_params/1e6:.1f}M parameter decoder-only transformer, trained from scratch on "
        "Polish text published before 1918.",
        "",
        "## Files",
        "",
        "| file | what |",
        "|---|---|",
        *file_table(payload),
    ]
    lines += [
        "",
        "## Run",
        "",
        f"- {cfg['n_layer']} layers, d_model {cfg['n_embd']}, "
        f"{cfg['n_head']} heads ({cfg['n_kv_head']} KV), context {cfg['block_size']}",
        f"- {cfg['max_steps']:,} steps at "
        f"{cfg['batch_size'] * cfg.get('grad_accum', 1) * cfg['block_size']:,} tokens/step "
        f"= {tokens_seen/1e9:.2f}B tokens seen",
        f"- schedule {cfg.get('schedule', 'wsd')}, decay over the final "
        f"{cfg.get('decay_frac', 0.1):.0%}, seed {cfg.get('seed', 1337)}",
    ]
    if step >= 0:
        lines.append(f"- final step in this checkpoint: {step:,}")
    else:
        lines.append("- the checkpoint records no step; it predates that field")
    if corpus:
        # Epochs and unique tokens count the training split, not the whole corpus: train.py
        # holds the split out before the first step, so the model never sees it. Dividing by
        # the full corpus reports 2.00 for a run that train.py's own epoch counter, and the
        # epoch checkpoints it wrote, put at 2.02.
        train_tokens = corpus - (val or 0)
        lines += [
            f"- corpus {corpus:,} tokens, {val:,} held out, {train_tokens:,} trained on",
            f"- {tokens_seen/train_tokens:.2f} epochs over the training split",
            # Unique and seen differ by the epoch count, and the two are routinely confused in
            # multi-epoch runs, so state both rather than one number called "tokens/parameter".
            f"- {train_tokens/n_params:.0f} unique tokens per parameter, "
            f"{tokens_seen/n_params:.0f} seen per parameter",
        ]
    if ev:
        ci = ev.get("ci95", [None, None])
        lines += [
            "",
            "## Held-out evaluation",
            "",
            f"Cross-entropy {ev['cross_entropy_nats']} nats, perplexity {ev['perplexity']}, "
            f"95% CI [{ci[0]}, {ci[1]}].",
            f"Scored on {ev['tokens_scored']:,} tokens in {ev['windows_evaluated']:,} "
            f"non-overlapping windows, {ev['dtype']}, deterministic. See `src/eval_val.py`.",
        ]
    lines += [
        "",
        "## Loading",
        "",
        "```python",
        "import sys; sys.path.insert(0, 'src')",
        "from sample import load_model",
        "from tokenizers import ByteLevelBPETokenizer",
        "",
        f"model = load_model('{name}/model.pt', 'cpu')",
        f"tok = ByteLevelBPETokenizer('{name}/vocab.json', '{name}/merges.txt')",
        "```",
        "",
        "The tokenizer in this directory is the one the model was trained with "
        f"(vocab.json sha256 {tok_sha['vocab.json'][:16]}). A different vocabulary of the "
        "same size will load without error and generate nonsense.",
        "",
        "## Intended use and limitations",
        "",
        "Research and digital humanities. This is a base model: it continues text rather "
        "than answering questions, and it reproduces the worldview of its sources, "
        "including antisemitic, nationalist, colonial and misogynist discourse that was "
        "ordinary in nineteenth-century Polish print. That is a property of the artifact, "
        "not a defect of data cleaning. See `docs/release-and-ethics.md`.",
        "",
        "Not suitable for human-facing deployment without a filtering layer and explicit "
        "framing, and not an authority on any subject.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True, help="directory name, e.g. wieszcz-349m-2026-07-25")
    p.add_argument("--out", default="models")
    p.add_argument("--eval", help="eval report from src/eval_val.py")
    p.add_argument("--log", help="training log, for the corpus size")
    p.add_argument("--manifest", help="run manifest, to check the tokenizer against")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # The checkpoint records nothing about the vocabulary it was trained against, so this is
    # the only place the pairing can be checked. Without a manifest it is an assumption.
    want = json.loads(Path(args.manifest).read_text()).get("tokenizer") if args.manifest else None
    if want:
        have = {f: sha256(TOKENIZER_DIR / f) for f in want}
        bad = [f for f, h in want.items() if h and have[f] != h]
        if bad:
            raise SystemExit(
                f"tokenizer/ does not match the one this run used: {', '.join(bad)} differ.\n"
                f"Exporting would pair the weights with a vocabulary they were not trained on,\n"
                f"which loads without error and generates nonsense.")
        print(f"tokenizer verified against {args.manifest}")
    elif args.manifest:
        print(f"warning: {args.manifest} records no tokenizer; pairing is unverified")
    else:
        print("warning: no --manifest given; the tokenizer pairing is unverified")

    dest = Path(args.out) / args.name
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} exists; pass --force to overwrite")
    dest.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, step = ckpt["cfg"], int(ckpt.get("step", -1))
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    model = GPT(cfg)
    model.load_state_dict(state)          # fails here rather than at somebody else's inference
    n_params = sum(q.numel() for q in model.parameters())

    torch.save({"model": state, "cfg": cfg, "step": step}, dest / "model.pt")

    for f in ("vocab.json", "merges.txt"):
        shutil.copy2(TOKENIZER_DIR / f, dest / f)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    ev = json.loads(Path(args.eval).read_text()) if args.eval else None
    if ev:
        (dest / "eval.json").write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
        # Per-window losses travel with the report: they are what lets somebody else redo the
        # paired comparison against another model, which two confidence intervals cannot.
        windows = Path(args.eval).with_suffix(".windows.npy")
        if windows.exists():
            shutil.copy2(windows, dest / "eval.windows.npy")

    if args.manifest:
        shutil.copy2(args.manifest, dest / "manifest.json")

    log = Path(args.log) if args.log else None
    corpus, val = corpus_from_log(log)
    if log and log.exists():
        shutil.copy2(log, dest / "train.log")

    # An explicit payload list, not a directory listing: on a re-export the directory also
    # holds the previous README.md and SHA256SUMS, and hashing the checksum file into itself,
    # or a README written after it, produces a file that fails its own check.
    payload = ["model.pt", "vocab.json", "merges.txt", "config.json"]
    payload += [f for f in ("manifest.json", "eval.json", "eval.windows.npy", "train.log")
                if (dest / f).exists()]
    sums = {f: sha256(dest / f) for f in payload}
    (dest / "SHA256SUMS").write_text(
        "".join(f"{h}  {f}\n" for f, h in sums.items()), encoding="utf-8")

    (dest / "README.md").write_text(
        card(args.name, cfg, step, n_params, sums, corpus, val, ev, payload),
        encoding="utf-8")

    src_mb = Path(args.ckpt).stat().st_size / 1e6
    out_mb = (dest / "model.pt").stat().st_size / 1e6
    print(f"{dest}: {n_params/1e6:.1f}M params, step {step:,}")
    print(f"  weights {out_mb:,.0f} MB (checkpoint was {src_mb:,.0f} MB, optimizer state dropped)")
    print(f"  files: {', '.join(sums)}")


if __name__ == "__main__":
    main()
