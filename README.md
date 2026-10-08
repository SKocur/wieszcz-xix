# wieszcz-xix

A small language model trained **from scratch** on **Polish text published between 1800 and
1918**: a "time capsule" model that speaks in the authentic voice, spelling, and
worldview of the era, and knows nothing of the world after its cutoff.

Inspired by [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM) and its
*Selective Temporal Training* idea, ported to Polish, where pre-1936 orthography and
the Romantic/Positivist worldview make the effect audible from the first sentence.

## The release

| | |
|---|---|
| Corpus | [SKocur/polish-pre1918-corpus](https://huggingface.co/datasets/SKocur/polish-pre1918-corpus), 294,369 documents, 6,745,956,943 tokens |
| Model | [SKocur/wieszcz-xix-349m](https://huggingface.co/SKocur/wieszcz-xix-349m), safetensors, loads with `AutoModelForCausalLM` |
| Corpus record | [doi:10.5281/zenodo.22099302](https://doi.org/10.5281/zenodo.22099302), the provenance ledger, exclusions and rights review |
| Weights record | [doi:10.5281/zenodo.22099358](https://doi.org/10.5281/zenodo.22099358), all nine checkpoints of the ladder |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "SKocur/wieszcz-xix-349m"
tok = AutoTokenizer.from_pretrained(name)
net = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
```

## The core rule

**Nothing after the cutoff.** The corpus is bounded by **publication date: 1800–1918**,
enforced at the crawl and then audited by content and provenance before tokenisation;
3,733 documents held by the crawl were excluded from the frozen build. A measured residue
remains in the transcribed Wolne Lektury part (0.7% of the corpus, in modernised spelling):
90 documents there are certainly later text, 0.04% of the corpus bytes. That constraint *is* the project, and any leak of modern data
defeats the purpose.

Two dates get confused here, so both are stated: **1918** is the bound on the corpus, and
**1936** is the orthographic reform. Pre-1918 print is written in pre-reform spelling as a
consequence. The reform is why the register is audible; the 1918 bound is what the corpus
actually enforces.

## Scope

- **Period:** publication 1800–1918 (Romanticism, Positivism, early Young Poland), in
  pre-1936 orthography.
- **Language:** Polish, original spelling preserved in the scanned text (no normalization);
  the Wolne Lektury transcriptions are in modernised spelling.
- **Model:** decoder-only Transformer. A matched ladder at 47M, 107M and 349M parameters,
  trained on the same frozen corpus for the same steps from the same seed, so the rungs
  differ only in size. These are base models (text continuation), not instruction-tuned
  chatbots.
- **Architecture:** RoPE · RMSNorm (pre+post, Gemma-2 sandwich) · QK-Norm · SwiGLU ·
  Grouped-Query Attention · FlashAttention (via SDPA) · weight tying, scaled residual init.
- **Optimizer:** Muon (Newton-Schulz orthogonalized momentum) on hidden weights + AdamW on
  embeddings/norms, currently the fastest known recipe at nanoGPT scale.
  Left out because they would hurt at this scale: MoE, sliding-window attention, soft-capping.

## Pipeline

1. **Collect**: public-domain texts: clean transcriptions from Wolne Lektury, then the
   Internet Archive at scale. See `src/prepare_data.py`.
2. **Clean**: deduplicate, strip editorial apparatus, keep original orthography.
3. **Tokenize**: train a dedicated BPE tokenizer on the historical corpus
   (a modern tokenizer would shred pre-reform spelling). See `src/train_tokenizer.py`.
4. **Train**: next-token pretraining, nanoGPT-style. See `src/train.py`.
5. **Sample**: eyeball generations for period coherence.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py            # download + clean corpus -> data/clean/
python src/train_tokenizer.py         # train BPE -> tokenizer/
python scripts/make_doc_split.py      # hold out whole documents -> metrics/doc_split_*.json
python scripts/tokenize_corpus.py     # -> data/tokens_*.bin and data/val_*.bin

python src/train.py --config configs/wieszcz_47m_6b7.json \
    --data data/tokens_frozen_6.69B.bin --val-data data/val_2026-08-03.bin
```

Pass `--data` and `--val-data`. Without them `train.py` tokenizes a sorted glob and holds
out the positional final 1%, which is the split defect the paper's section on
validation-split source bias is about:
filenames carry a source prefix, so the tail of the stream is one source rather than a
sample of the corpus.

## Docs

- [`docs/data-preparation.md`](docs/data-preparation.md): why the corpus filters are what
  they are (sources chosen and rejected, quality/language gates, calibration data).
- [`docs/training.md`](docs/training.md): the environments the ladder was trained and
  evaluated on, right-sizing a run to the corpus, VRAM arithmetic, context length, and
  what a healthy loss curve looks like.
- [`docs/run-log.md`](docs/run-log.md): the first campaign end to end: laptop validation,
  the 3080 run with loss numbers, the KV-cache masking bug that produced garbage, and what
  the working model actually learned.

## Hardware

All three rungs were trained on a single rented RTX PRO 4500 Blackwell (32 GB). The 47M
rung fits a 10 GB card; evaluation and generation for the whole ladder run on a 3080. The
bottleneck is data rather than compute, so expect to be data-limited and to make more than
one pass over the text.

## Content warning

A model trained only on 19th-century text is faithful to the period in every respect,
including its prejudices. **The models reproduce antisemitic, nationalist and colonial
content of the era, and so does the corpus.** The prejudice comes from the material, and
the training preserved it, because a time-capsule model cleaned of it would model a
century that never existed.

Two places in this repository contain such text verbatim, because measuring the problem
required reading it:

- `metrics/bias_sheet_*.json`: the adjudication sheet. Generations flagged for naming a
  national, ethnic or religious group, quoted in full, with hand-assigned labels. The
  charged-term list in `scripts/screen_bias.py` is period slur vocabulary.
- `output/samples_ladder_*.md` and the generation samples in the paper's appendix.

Anyone reusing the models should read the ethical considerations section of the
accompanying paper before deployment. The short version: a documented release to
researchers is a different thing from an interactive system that will say these things to
a member of the public unprompted, and the second is not recommended without a
filtering layer and visible framing.

## Data licensing

The texts were published between 1800 and 1918, and the catalogue records of 93% of the
Internet Archive documents carry the holding library's public-domain statement. Author
death dates were not checked document by document, so a work whose author died after 1955
may still be protected where the term is life plus 70 years. The Public Domain Mark is
applied only where a library states public domain; documents without a statement carry no
rights mark. 79 Wolne Lektury translations are still in copyright and keep the licence that
library publishes them under (CC BY-SA 3.0 or the Free Art Licence 1.3).
`ledger/document_metadata_2026-10-06.csv.gz` gives the rights basis of every document and
names author, translator and licence for the Wolne Lektury rows. Redistributing a harvested
collection is also bound by the EU **sui generis database right** over the compilation and
by each platform's terms of use. The released corpus carries CC0 1.0 over the compilation and database layer only. Sources and the per-document ledger are described in
[`data/PROVENANCE.md`](data/PROVENANCE.md); the per-source rights review is archived with
the corpus record on Zenodo.

> Not legal advice. For publication or commercial use, consult a specialist.

## Citation

```bibtex
@misc{kocur2026wieszcz,
  title  = {Wieszcz-XIX: A 3.1-Billion-Word Corpus of Pre-1918 Polish
            and Temporally Bounded Language Models Trained From Scratch},
  author = {Kocur, Szymon},
  year   = {2026},
  doi    = {10.5281/zenodo.21745210},
}
```

Code in this repository is Apache 2.0, see [`LICENSE`](LICENSE). The corpus is CC0 1.0
over its compilation layer, and the model weights are Apache 2.0 with the additional
conditions in their `TERMS-OF-USE.md`.

## Acknowledgements

- [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM): Selective Temporal Training.
- [nanoGPT](https://github.com/karpathy/nanoGPT): training scaffold.
- [Wolne Lektury](https://wolnelektury.pl/), [Internet Archive](https://archive.org/): the corpus.
