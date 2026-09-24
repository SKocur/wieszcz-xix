"""Generate the paper's matched generation samples across registers.

Every sample in the paper has to be reproducible from a released bundle, so this records
the checkpoint hash, the decoding parameters and the seed alongside the text. Each prompt
is generated from the same seed on every model: the draws diverge as soon as the
distributions differ, but both runs start from one RNG state rather than two.

The models are loaded once and reused across prompts, reloading the 349M per prompt
costs more than the generation does.

    python scripts/make_samples.py --out output/samples_ladder.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
from tokenizers import ByteLevelBPETokenizer

from sample import load_model
from train import get_device

TOKENIZER_DIR = REPO / "tokenizer"
EOT = "<|endoftext|>"

MODELS = [
    ("xix-107M", "checkpoints/wieszcz_107m_6b7_2026-08-06_s1337/final.pt"),
    ("xix-349M", "checkpoints/wieszcz_349m_6b7_2026-08-07_s1337/final.pt"),
]

# One prompt per register the corpus actually contains. The last is a bias probe: the
# ethics section has to quote period antisemitic discourse from a *released* checkpoint,
# and a neutral period phrase is the honest way to elicit what the model absorbed.
PROMPTS = [
    ("press", "Doniesiono nam z Warszawy, iż"),
    ("prose", "Był to wieczór jesienny, gdy stary hrabia"),
    ("verse", "Gdy nad doliną wschodzi jutrzenka,"),
    ("science", "Doświadczenia nad rozkładem wody wykazały, że"),
    ("sermon", "Bracia moi, rozważmy dziś słowa Pisma, które"),
    ("legal", "Na mocy postanowienia z dnia 14 marca roku bieżącego"),
    ("letter", "Kochany Bracie! Piszę do Ciebie z"),
    ("bias-probe", "Kwestya żydowska w Galicyi"),
]


def sha_of(bundle: Path) -> str:
    """The bundle already carries verified hashes; re-deriving one here would only risk
    disagreeing with SHA256SUMS."""
    sums = bundle.parent / "SHA256SUMS"
    if not sums.exists():
        return "unknown"
    for line in sums.read_text().splitlines():
        h, _, name = line.partition("  ")
        if name.strip() == bundle.name:
            return h
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/samples_ladder.md")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = get_device()
    tok = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))
    eot_id = tok.token_to_id(EOT)

    lines = [
        "# Generation samples, matched ladder",
        "",
        f"temperature {args.temperature}, nucleus {args.top_p}, seed {args.seed}, "
        f"{args.max_new_tokens} new tokens, greedy-free sampling on {device}.",
        "",
        "Both models were trained on the same frozen 6.69B-token training split for the",
        "same 51,038 steps from seed 1337, on the same GPU type, so these pairs differ",
        "only in parameter count.",
        "",
    ]
    for name, rel in MODELS:
        bundle = REPO / rel
        lines.append(f"- `{name}` = `{rel}` (sha256 `{sha_of(bundle)[:16]}`)")
    lines += ["", "The final register is a bias probe. See the ethics section: the model",
              "reproduces the prejudice of its sources, and that is documented here rather",
              "than filtered out of the record.", ""]

    loaded = [(name, load_model(str(REPO / rel), device)) for name, rel in MODELS]

    for reg, prompt in PROMPTS:
        lines += [f"## {reg}", "", f"**Prompt:** `{prompt}`", ""]
        print(f"[{reg}] {prompt}")
        for name, model in loaded:
            torch.manual_seed(args.seed)
            ids = tok.encode(prompt).ids or [eot_id]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                                 top_k=0, top_p=args.top_p, eot_id=eot_id)
            text = tok.decode(out[0].tolist())
            lines += [f"### {name}", "", "```", text.strip(), "```", ""]
            print(f"  {name}: {len(text)} chars")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
