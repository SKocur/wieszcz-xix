"""Build Hugging Face model repositories from the released bundles.

Two distributions of the same weights, shaped by what each host can hold. Zenodo takes a
flat list of files, so the nine paper checkpoints go there as one zip per bundle: nine
unpacked bundles would be ninety files with nine of them called `model.pt`. A Hugging
Face model repository is a git tree whose readers expect loose files, so it takes the
three released models unpacked, one repository per rung.

Weights ship as safetensors here. A `.pt` archive is a pickle, and a public download that
executes code while loading is a hazard a released model does not need. `save_model`
resolves the tied embedding and output head, and `load_model` restores the tie.

Each repository carries `model.py` copied verbatim from `src/`, so a reader can rebuild
the network without cloning this project. Copying rather than rewriting is deliberate: a
second hand-maintained definition of the architecture would drift from the one that
trained the weights, and the failure would be silent.

Beside it goes `modeling_wieszcz.py`, the thin `transformers` wrapper, so the repositories
answer to `AutoModelForCausalLM.from_pretrained` as well as to `GPT`. Two details make one
checkpoint serve both entry points. The stored tensor of the tied pair is `tok_emb.weight`
rather than `head.weight`, because `from_pretrained` rebuilds the head from the input
embedding and would otherwise leave it on the meta device; `GPT` reads the same file
unchanged, since its head and embedding are one tensor from construction. And the file
carries `format: pt` metadata, without which `transformers` refuses to open it.

Both entry points are loaded at the end of a build and their logits compared, and the
packaged tokenizer is re-encoded against the one that tokenized the corpus.

    python scripts/build_hf_models.py                 # all three rungs
    python scripts/build_hf_models.py --rung 47m      # one
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_model, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from model import GPT  # noqa: E402

BUNDLES = REPO.parent / "models"
CORPUS_REPO = "SKocur/polish-pre1918-corpus"
DOI_PAPER = "10.5281/zenodo.21745210"
DOI_CORPUS = "10.5281/zenodo.22099302"
DOI_WEIGHTS = "10.5281/zenodo.22099358"

RUNGS = ["47m", "107m", "349m"]

# The tokenizer terminates every document with this token and numbers it 0, so it is the
# only sentinel the model ever saw and it serves for all four roles the library asks for.
EOT = "<|endoftext|>"
EOT_ID = 0

# Segmentation the packaged tokenizer has to reproduce exactly: pre-reform diacritics,
# period quotation marks, a bare year, and a chapter break, which is where a byte-level
# vocabulary trained on this corpus differs most from a modern Polish one.
TOKENIZER_PROBES = (
    "Doniesiono nam z Warszawy, iż",
    "Jaśnie Wielmożny Pan Podkomorzy rzekł: „Mości Sędzio!”",
    "gęś, źdźbło, żółć, ćma, łąka",
    "1863",
    "\n\nRozdział IV.\n",
)


def write_tokenizer(src: Path, out: Path, block_size: int) -> None:
    """Write `tokenizer.json` for `AutoTokenizer`, built from the shipped vocab and merges.

    Serializing from those two files rather than from a saved tokenizer object keeps one
    definition of the segmentation; a second one would be free to drift from the corpus.
    """
    from tokenizers import ByteLevelBPETokenizer

    ByteLevelBPETokenizer(str(src / "vocab.json"),
                          str(src / "merges.txt")).save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2TokenizerFast",
        "bos_token": EOT, "eos_token": EOT, "unk_token": EOT, "pad_token": EOT,
        "model_max_length": block_size,
        "clean_up_tokenization_spaces": False,
    }, indent=1) + "\n", encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": EOT, "eos_token": EOT, "unk_token": EOT, "pad_token": EOT,
    }, indent=1) + "\n", encoding="utf-8")


def verify(rung: str, out: Path, cfg: dict) -> None:
    """Load the built repository both ways and require identical logits.

    The wrapper adds no arithmetic, so any difference at all is a packaging fault:
    a tensor that never loaded, a broken tie, a config field read wrong.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import ByteLevelBPETokenizer

    native = GPT(json.loads((out / "config.json").read_text()))
    load_model(native, str(out / "model.safetensors"))
    native.eval()

    wrapped = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True).eval()
    stranded = [n for n, p in wrapped.named_parameters() if p.device.type == "meta"]
    if stranded:
        raise SystemExit(
            f"{rung}: {len(stranded)} parameters were never loaded, starting at {stranded[0]}")

    ids = torch.randint(0, cfg["vocab_size"], (1, 64),
                        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        bare, _ = native(ids)
        through_wrapper = wrapped(input_ids=ids).logits
    if bare.shape != (1, 64, cfg["vocab_size"]):
        raise SystemExit(f"{rung}: reloaded model produced {tuple(bare.shape)}")
    if not torch.equal(bare, through_wrapper):
        gap = (bare - through_wrapper).abs().max().item()
        raise SystemExit(f"{rung}: the wrapper disagrees with GPT by up to {gap:.3e}")

    trained = ByteLevelBPETokenizer(str(out / "vocab.json"), str(out / "merges.txt"))
    packaged = AutoTokenizer.from_pretrained(str(out))
    for probe in TOKENIZER_PROBES:
        if trained.encode(probe).ids != packaged(probe)["input_ids"]:
            raise SystemExit(f"{rung}: the packaged tokenizer re-segments {probe!r}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def card(rung: str, cfg: dict, ev: dict, man: dict) -> str:
    params_m = man["n_params_millions"]
    heads = f'{cfg["n_head"]} query heads over {cfg["n_kv_head"]} key/value heads'
    return f"""---
license: apache-2.0
language:
- pl
library_name: transformers
pipeline_tag: text-generation
tags:
- historical
- 19th-century
- polish
- time-capsule
- from-scratch
datasets:
- {CORPUS_REPO}
---

# wieszcz-xix-{rung}

A {params_m:.0f}M-parameter decoder-only language model trained **from scratch**, and only
on Polish text published between 1800 and 1918. It writes in the language of that period,
in pre-reform orthography, and knows nothing of the world after its cutoff. It is a base
model: it continues text and does not answer questions.

This is the {rung.upper()} rung of a three-model ladder ({", ".join(RUNGS).upper()}) trained on one
frozen corpus with the same steps, batch and seed, so the rungs differ only in parameter
count. The smaller rungs write the same period Polish but lose track of what they are
talking about within a few sentences, so this repository carries the rung worth using;
the whole ladder, including the pre-decay and epoch-one checkpoints the paper's epochs
table reads, is archived at [doi:{DOI_WEIGHTS}](https://doi.org/{DOI_WEIGHTS}). The corpus
is [{CORPUS_REPO}](https://huggingface.co/datasets/{CORPUS_REPO}).

> ⚠️ **Read [Intended use and content warning](#intended-use-and-content-warning) before
> loading this model.** It reproduces the prejudices of nineteenth-century print.

## What it scores

Dense evaluation on the held-out split of its own corpus, 8,192 windows of 1,024 tokens
in fp32, covering the whole split:

| | |
|---|---|
| Cross-entropy | {ev["cross_entropy_nats"]:.3f} nats/token (95% CI {ev["ci95"][0]:.3f} to {ev["ci95"][1]:.3f}) |
| Perplexity | {ev["perplexity"]:.2f} |
| Parameters | {ev["params"]:,} |
| Training | {man["train_tokens"]:,} tokens, {ev["step"] + 1:,} steps, seed {man["seed"]} |

Scored under the same protocol, the smaller rungs reach 2.905 (107M) and 3.101 (47M)
nats, which is what the paper's ladder measures.

## Architecture

Llama and Gemma-2 lineage: RoPE, RMSNorm applied before and after each sublayer, QK-Norm,
SwiGLU, grouped-query attention, weight tying. Trained with Muon on the 2D hidden weights
and AdamW on embeddings and norm gains.

| | |
|---|---|
| Layers | {cfg["n_layer"]} |
| Model dimension | {cfg["n_embd"]} |
| Attention | {heads} |
| Context | {cfg["block_size"]} tokens |
| Vocabulary | {cfg["vocab_size"]}, byte-level BPE trained on this corpus |

The tokenizer is part of the model. It was trained on nineteenth-century Polish, and a
modern Polish tokenizer shreds pre-reform spelling, so it ships here in both forms:
`tokenizer.json` for `AutoTokenizer`, and the `vocab.json` and `merges.txt` it was built
from. All three are hashed in `SHA256SUMS`.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "SKocur/wieszcz-xix-{rung}"
tok = AutoTokenizer.from_pretrained(name)
net = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

ids = tok("Doniesiono nam z Warszawy, iż", return_tensors="pt")
out = net.generate(**ids, max_new_tokens=200, do_sample=True, temperature=0.9, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

`trust_remote_code` is needed because this architecture is not one the library ships. The
code it runs is `modeling_wieszcz.py` in this repository, a short wrapper that adds the
library's interface around `model.py` and no arithmetic of its own. Both files are meant
to be read before they are run.

Sampling through the wrapper recomputes the whole prefix at every step, because the KV
cache in `model.py` predates the library's cache protocol and does not implement it. For
long or repeated generation, load the network directly and use its own cached sampler:

```python
import json, sys, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from tokenizers import ByteLevelBPETokenizer

path = snapshot_download("SKocur/wieszcz-xix-{rung}")
sys.path.insert(0, path)
from model import GPT

net = GPT(json.load(open(f"{{path}}/config.json")))
load_model(net, f"{{path}}/model.safetensors")
net.eval()

tok = ByteLevelBPETokenizer(f"{{path}}/vocab.json", f"{{path}}/merges.txt")
ids = torch.tensor([tok.encode("Doniesiono nam z Warszawy, iż").ids])
out = net.generate(ids, max_new_tokens=200, temperature=0.9, top_p=0.9)
print(tok.decode(out[0].tolist()))
```

That path needs only `torch`, `safetensors` and `tokenizers`, and it reads the same
`model.safetensors` through the file that trained the weights. The build checks that the
two paths return identical logits.

## Intended use and content warning

Research and digital humanities: period language, register and discourse, historical NLP,
scaling and data-constraint experiments.

**This model reproduces antisemitic, nationalist, colonial and misogynist discourse that
was ordinary in nineteenth-century Polish print.** The prejudice comes from the sources,
and the training preserved it, because a model of a period that had been cleaned of its
prejudices would model a century that never existed. The accompanying paper documents
unprompted prejudiced output from these checkpoints and reports a first prevalence
estimate.

Do not put this model in front of end users, interactively or through generated text
presented as information, without a filtering layer, visible framing that identifies the
output as a historical-language simulation, and your own legal review. It is not an
authority on any subject. Full conditions are in `TERMS-OF-USE.md`.

## Limitations

The context is {cfg["block_size"]} tokens, which bounds both coherence and any interactive use.
The corpus is OCR of period scans, so the model learned some of that damage and
occasionally writes it back. Generation stays in period register by construction, which
means a modern prompt does not produce a modern answer.

## Citation

```bibtex
@misc{{kocur2026wieszcz,
  title  = {{A pre-1918 Polish corpus and a ladder of time-capsule language models}},
  author = {{Kocur, Szymon}},
  year   = {{2026}},
  doi    = {{{DOI_PAPER}}},
}}
```

Corpus: [doi:{DOI_CORPUS}](https://doi.org/{DOI_CORPUS}).
Weights archive with all nine checkpoints: [doi:{DOI_WEIGHTS}](https://doi.org/{DOI_WEIGHTS}).
"""


def build(rung: str, out_root: Path) -> None:
    src = BUNDLES / f"wieszcz-{rung}-final"
    if not src.is_dir():
        raise SystemExit(f"no bundle at {src}; run scripts/export_release.sh first")

    ckpt = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    cfg, ev, man = ckpt["cfg"], json.loads((src / "eval.json").read_text()), \
        json.loads((src / "manifest.json").read_text())

    # The bundle's own evaluation is what the card quotes, so a bundle carrying another
    # rung's report would publish the wrong number under the right name.
    if ev["params"] != sum(p.numel() for p in GPT(cfg).parameters()):
        raise SystemExit(f"{rung}: eval.json reports {ev['params']} parameters, config builds a different model")

    net = GPT(cfg)
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()

    out = out_root / f"wieszcz-xix-{rung}"
    out.mkdir(parents=True, exist_ok=True)

    # Of the tied pair, keep the embedding and drop the head. `GPT` ties the two at
    # construction, so it reads the file back either way, while `from_pretrained` derives
    # the head from the embedding and strands it on the meta device if only the head is
    # stored. The metadata key is what makes `transformers` willing to open the file.
    state = net.state_dict()
    if state["tok_emb.weight"].data_ptr() != state["head.weight"].data_ptr():
        raise SystemExit(f"{rung}: embedding and head are not tied, so the export is ambiguous")
    state.pop("head.weight")
    save_file({k: v.contiguous() for k, v in state.items()},
              str(out / "model.safetensors"), metadata={"format": "pt"})

    (out / "config.json").write_text(json.dumps({
        **{k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head",
                               "n_kv_head", "n_embd", "multiple_of", "rope_theta")},
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "model_type": "wieszcz",
        "architectures": ["WieszczForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_wieszcz.WieszczConfig",
            "AutoModelForCausalLM": "modeling_wieszcz.WieszczForCausalLM",
        },
        "bos_token_id": EOT_ID,
        "eos_token_id": EOT_ID,
        "pad_token_id": EOT_ID,
        "use_cache": False,
        "torch_dtype": "float32",
    }, indent=1) + "\n", encoding="utf-8")

    shutil.copy2(REPO / "src/model.py", out / "model.py")
    shutil.copy2(REPO / "src/modeling_wieszcz.py", out / "modeling_wieszcz.py")
    shutil.copy2(BUNDLES / "TERMS-OF-USE.md", out / "TERMS-OF-USE.md")
    shutil.copy2(BUNDLES / "LICENSE", out / "LICENSE")
    for name in ("vocab.json", "merges.txt"):
        shutil.copy2(src / name, out / name)
    write_tokenizer(src, out, cfg["block_size"])
    (out / "README.md").write_text(card(rung, cfg, ev, man), encoding="utf-8")

    # Importing model.py from a built repository leaves __pycache__ behind, and anything
    # listed here is a file the repository publishes.
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(out / n)}  {n}\n" for n in names), encoding="utf-8")

    # Loading is the claim the card makes; a build that cannot be loaded back is a
    # repository that fails for its first reader instead of here.
    verify(rung, out, cfg)

    size = sum(p.stat().st_size for p in out.iterdir()) / (1 << 20)
    print(f"{out.relative_to(out_root.parent)}: {len(names) + 1} files, {size:.0f} MB, "
          f"CE {ev['cross_entropy_nats']:.3f}, both entry points verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO.parent / "hf-models"))
    ap.add_argument("--rung", choices=RUNGS, action="append")
    args = ap.parse_args()
    for rung in args.rung or RUNGS:
        build(rung, Path(args.out))


if __name__ == "__main__":
    main()
