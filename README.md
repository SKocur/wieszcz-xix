# wieszcz-xix

A small language model trained **from scratch** on **19th-century Polish public-domain
text only** — a "time capsule" model that speaks in the authentic voice, spelling, and
worldview of the era, and knows nothing of the world after its cutoff.

Inspired by [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM) and its
*Selective Temporal Training* idea, ported to Polish — where pre-1936 orthography and
the Romantic/Positivist worldview make the effect audible from the first sentence.

## The core rule

**Nothing after the cutoff.** The training corpus contains only text written (and
spelled) before the 1936 Polish orthographic reform. No modern Polish, no modernized
editions, no post-reform spelling. That constraint is not a limitation — it *is* the
project. Any leak of modern data defeats the purpose.

## Scope

- **Period:** 19th century (Romanticism, Positivism, early Young Poland), pre-1936 orthography.
- **Language:** Polish, original spelling preserved (no normalization).
- **Model:** decoder-only Transformer, small (~20–60M params) — enough to hear the period
  voice. This is a base model (text continuation), not an instruction-tuned chatbot.
- **Architecture:** RoPE · RMSNorm (pre+post, Gemma-2 sandwich) · QK-Norm · SwiGLU ·
  Grouped-Query Attention · FlashAttention (via SDPA) · weight tying, scaled residual init.
- **Optimizer:** Muon (Newton-Schulz orthogonalized momentum) on hidden weights + AdamW on
  embeddings/norms — currently the fastest known recipe at nanoGPT scale.
  Excluded on purpose (would hurt at this scale): MoE, sliding-window attention, soft-capping.

## Pipeline

1. **Collect** — public-domain texts, starting with clean transcribed sources
   (Wolne Lektury, Wikiźródła). See `src/prepare_data.py`.
2. **Clean** — deduplicate, strip editorial apparatus, keep original orthography.
3. **Tokenize** — train a dedicated BPE tokenizer on the historical corpus
   (a modern tokenizer would shred pre-reform spelling). See `src/train_tokenizer.py`.
4. **Train** — next-token pretraining, nanoGPT-style. See `src/train.py`.
5. **Sample** — eyeball generations for period coherence.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py       # download + clean corpus -> data/clean/
python src/train_tokenizer.py    # train BPE -> tokenizer/
python src/train.py --config configs/wieszcz_20m.json
```

## Docs

- [`docs/data-preparation.md`](docs/data-preparation.md) — why the corpus filters are what
  they are (sources chosen and rejected, quality/language gates, calibration data).
- [`docs/training.md`](docs/training.md) — environment setup, CUDA/WSL gotchas
  (`nvcc` + `torch.compile`, `/mnt/c` vs ext4), hardware paths, right-sizing, what a
  healthy loss curve looks like.
- [`docs/run-log.md`](docs/run-log.md) — the first campaign end to end: laptop validation,
  the 3080 run with loss numbers, the KV-cache masking bug that produced garbage, and what
  the working model actually learned.

## Hardware

A single 24 GB consumer GPU (3090/4090) is plenty for a 20–60M model; the corpus, not
compute, is the bottleneck. Expect to be data-limited and to train for more epochs.

## Data licensing

The underlying 19th-century works are public domain (author died >70 years ago). The
real constraints are **platform terms of use** (e.g. Polona API limits) and the EU
**sui generis database right** when redistributing a harvested collection — not the
copyright of the texts themselves. Every source is logged in
[`data/PROVENANCE.md`](data/PROVENANCE.md) with its rights status.

> Not legal advice. For publication or commercial use, consult a specialist.

## Acknowledgements

- [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM) — Selective Temporal Training.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — training scaffold.
- [Wolne Lektury](https://wolnelektury.pl/), [Wikiźródła](https://pl.wikisource.org/) — the corpus.
