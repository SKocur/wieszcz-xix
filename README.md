# wieszcz-xix

A small language model trained **from scratch** on **Polish text catalogued as published
between 1800 and 1918**: a "time capsule" model that writes in the language, spelling and
worldview of the era, trained on text audited for anything printed after 1918.

Inspired by [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM) and its
*Selective Temporal Training* idea, ported to the Polish print of Romanticism and
Positivism.

## The release

| | |
|---|---|
| Corpus | [SKocur/polish-pre1918-corpus](https://huggingface.co/datasets/SKocur/polish-pre1918-corpus), 294,369 documents, about 3.1 billion words (6,745,956,943 tokens) |
| Model | [SKocur/wieszcz-xix-349m](https://huggingface.co/SKocur/wieszcz-xix-349m), safetensors, loads with `AutoModelForCausalLM` |
| Corpus record | [doi:10.5281/zenodo.22099302](https://doi.org/10.5281/zenodo.22099302), the provenance ledger, exclusions and rights review |
| Weights record | [doi:10.5281/zenodo.22099358](https://doi.org/10.5281/zenodo.22099358), all nine checkpoints of the ladder |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "SKocur/wieszcz-xix-349m"
tok = AutoTokenizer.from_pretrained(name)
net = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
```

## The temporal bound

The corpus is bounded by **publication date, 1800 to 1918 with 1918 included**, enforced at
the crawl and then audited by content and provenance before tokenisation; 3,733 documents
held by the crawl were excluded from the frozen build. A known residue remains in the
transcribed Wolne Lektury part (0.7% of the corpus, in modernised spelling): 90 documents
there are certainly later text, 0.04% of the corpus bytes, and counting every translation
or work by someone who lived past 1918 gives an upper bound of 1,116 documents, 0.38% of
the bytes. Post-1918 text that avoids modern vocabulary and dated self-reference can pass
the audit.

Spelling is a separate matter. The 1918 rules of the Academy of Learning made *-cja* the
standard form over *-cya*, and the 1936 reform set the present norm. Both spellings were in
print long before 1918, so spelling does not date a text; the 1918 bound is what the corpus
enforces.

## Scope

- **Period:** catalogued publication 1800–1918 (Romanticism, Positivism, early Young
  Poland).
- **Language:** Polish, original spelling preserved in the scanned text (no normalization);
  the Wolne Lektury transcriptions are in modernised spelling.
- **Model:** decoder-only Transformer. A matched ladder at 47M, 107M and 349M parameters,
  trained on the same frozen corpus for the same steps from the same seed, so the rungs
  differ only in size. These are base models (text continuation), not instruction-tuned
  chatbots.
- **Architecture:** RoPE · RMSNorm (pre+post, Gemma-2 sandwich) · QK-Norm · SwiGLU ·
  Grouped-Query Attention · FlashAttention (via SDPA) · weight tying, scaled residual init.
- **Optimizer:** Muon (Newton-Schulz orthogonalized momentum) on hidden weights + AdamW on
  embeddings/norms. MoE, sliding-window attention and soft-capping are not used.

## Pipeline

1. **Collect**: clean transcriptions from Wolne Lektury, then OCR text from the Internet
   Archive at scale. See `src/prepare_data.py`.
2. **Clean**: filter OCR and layout noise, remove near-identical duplicates, strip
   editorial apparatus, keep each source's spelling (original in the scanned text,
   modernised in Wolne Lektury).
3. **Tokenize**: train a dedicated byte-level BPE tokenizer on the corpus. See
   `src/train_tokenizer.py`.
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

## Hardware

All three rungs were trained on a single rented RTX PRO 4500 Blackwell (32 GB). The 47M
rung fits a 10 GB card; evaluation and generation for the whole ladder run on a 3080. The
bottleneck is data rather than compute, so expect to be data-limited and to make more than
one pass over the text.

## Content warning

A model trained only on period text reproduces the period's prejudices along with its
language. **The corpus contains antisemitic, nationalist, colonial and misogynist
discourse, and the models reproduce period prejudice, including antisemitic statements.**
The prejudice comes from the material; the cleaning did not introduce it and was not
designed to remove it.

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

The Internet Archive texts are catalogued as published from 1800 to 1918, and the catalogue
records of 93% of those documents carry the holding library's public-domain statement.
Author death dates were not checked document by document, so a work whose author died
after 1955 may still be protected where the term is life plus 70 years. The catalogue
names a creator who died in 1956 or later for 915 documents, 849 of them periodical issues
whose named creator is an editor. The Public Domain Mark is
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
  eprint = {2610.10592},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2610.10592},
}
```

Code in this repository is Apache 2.0, see [`LICENSE`](LICENSE). The corpus is CC0 1.0
over its compilation layer, and the model weights are Apache 2.0 with the additional
conditions in their `TERMS-OF-USE.md`.

## Acknowledgements

- [TimeCapsuleLLM](https://github.com/haykgrigo/TimeCapsuleLLM): Selective Temporal Training.
- [nanoGPT](https://github.com/karpathy/nanoGPT): training scaffold.
- [Wolne Lektury](https://wolnelektury.pl/), [Internet Archive](https://archive.org/): the corpus.
