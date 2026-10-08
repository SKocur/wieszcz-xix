---
license: cc0-1.0
language:
- pl
pretty_name: Polish pre-1918 corpus (wieszcz-xix)
size_categories:
- 100K<n<1M
task_categories:
- text-generation
tags:
- historical
- 19th-century
- polish
- ocr
- public-domain
- time-capsule
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Polish pre-1918 corpus

**294,369 documents, 22.77 GB of Polish text published between 1800 and 1918**, assembled
by a fully code-driven pipeline for training temporally bounded ("time-capsule") language
models from scratch. Every filter is a documented function, each cleaning step writes a
machine-readable report with the hash of the script that produced it, and the corpus's
defects are quantified in those reports.

This is the corpus behind the `wieszcz-xix` model ladder (47M / 107M / 349M parameters).

> ⚠️ **Read [Content warning](#content-warning) and [Rights](#rights) before use.** This is
> nineteenth-century print. It contains antisemitic, nationalist and colonial discourse
> that was ordinary in its time.

## What is in it

| | |
|---|---|
| Documents | 294,369 |
| Text | 22.77 GB (UTF-8) |
| Tokens | 6,745,956,943 under the corpus's own 8k byte-level BPE |
| On disk | 10.93 GB in 43 shards as zstd-compressed parquet |
| Period | publication dates 1800–1918 |
| Language | Polish, **original orthography preserved** in the Internet Archive text (no normalisation to post-1936 spelling); the Wolne Lektury transcriptions are in modernised spelling |
| Sources | Internet Archive 291,655 documents (22.60 GB) · Wolne Lektury 2,714 (0.17 GB) |

Fields, one row per document:

| field | |
|---|---|
| `id` | stable document identifier, joins to the provenance ledger |
| `source` | `internet_archive` or `wolne_lektury` |
| `source_identifier` | the identifier at that source, so any document can be traced back |
| `text` | the cleaned full text |

Catalogue metadata for the Internet Archive documents (title, date, place of publication,
type, creator and rights statement, as retrieved on 2026-10-06) is in the code repository
at `metrics/ia_catalogue_metadata_2026-10-06.jsonl.gz`, keyed by the same identifier. `ledger/document_metadata_2026-10-06.csv.gz` in this
repository joins it to every document as one table
(split, provider, year, place, type, title, language, rights basis).

There is a single `train` split. The document-level train/validation split used by the
paper is **not** applied here; it is published separately as identifier lists, so a user
can reproduce it or ignore it. Applying it here would hand every reader one particular
experiment's split as if it were a property of the corpus.

## This release is exactly what the models were trained on

The corpus published here is the frozen build, document for document: **294,369
documents**, the same stream the `wieszcz-xix` ladder was trained on. Nothing was removed
after training and nothing is held back.

This holds because the temporal audit ran **before** tokenisation, so the documents it
excluded never entered the training stream. `exclusions/exclusions_2026-08-03.json` lists
the **3,733 documents** removed at that point, 2.47% of the pre-exclusion crawl by bytes,
under a rule that trades precision for safety. The list ships with the data, so the
decision can be audited.

The exclusion battery is calibrated against period-legitimate controls, because the naive
version of this filter is worse than useless. *Czołg* (1918), *lotnisko* (1911),
*aeroplan*, *automobil* and *samolot* are period words; so is *komputer*, which in
nineteenth-century Polish means a register or roll, from Latin *computus*. A rule that
excluded on those would have thrown out legitimate press and left the actual
twentieth-century material in place.

## How it was built

1. **Crawl**: Polish text published 1800–1918 from the Internet Archive and Wolne Lektury,
   queried by publication-date range and frozen with a manifest.
2. **Filter**: OCR-noise and layout heuristics, each a documented function with a report.
3. **Deduplicate**: near-duplicate removal at the document level.
4. **Temporal audit**: a content battery (anachronistic vocabulary, dated self-reference)
   calibrated on its own false alarms against period-legitimate controls, plus a provenance
   check. 3,733 documents excluded before tokenisation.
5. **Freeze**: a per-document provenance ledger recording identifier, source and byte
   length, hashed.

Every step's report is in the code repository, and each carries the SHA-256 of the script
that wrote it.

### The ledger

`ledger/provenance_ledger_2026-08-03_{train,val}.csv.gz` records every document in the
frozen corpus: `document_id`, `source`, `source_identifier`, `bytes`. If the full text
ever has to come down, the ledger is what makes the corpus reconstructible from the
sources. It is also what this release's self-check verifies against: each document is
decoded from the frozen token stream and its UTF-8 length compared to the ledger, so a
truncated shard or a misaligned stream fails the build instead of shipping unnoticed.

Sizes in the ledger are what the *tokeniser* saw. For Wolne Lektury that differs from the
file on disk by the digitisation colophon, which tokenisation drops.

## OCR quality

OCR quality was estimated on twenty passages of about 400 characters, drawn with a fixed
seed from twelve Internet Archive documents and corrected by hand from context, without
the page images. Period orthography is treated as correct: *blizko*, *Jenerał*, *terrytorjum* and *niczem* are right
for the period, and counting them as errors would overstate the noise.

**Eleven passages could be corrected: CER 0.68%, WER 4.71%.**
**Nine, 45% of the sample, could not be corrected at all.**

The second number matters more. In those nine passages the corruption *removes* text
instead of garbling it: columns interleaved line by line, a fragment carried in from a
neighbouring article, a clause ending mid-word. Context does not determine what the page
said.

So 0.68% is the error rate **conditional on the text being legible as running prose**, and
that condition fails on nine of the twenty passages. The corpus's problem is occasional
total loss of structure more than frequent small misreadings. Read the CER figure as a
description of the legible fraction rather than of the whole corpus.

The sample is twenty passages read by one annotator.

## Content warning

A corpus of nineteenth-century print is faithful to the period in every respect, including
its prejudices. **This corpus contains antisemitic, nationalist, colonial and misogynist
discourse**, at times as the explicit subject of a text and at times as unremarked
background. It is present in the press material especially.

The prejudice comes from the material itself; the cleaning did not introduce it and was
not designed to remove it. A period corpus scrubbed of it would describe a century that
never existed and would be useless for the historical and linguistic work this resource
exists for.

Models trained on it reproduce that content. If you train on this corpus, that is a
property of what you build, and it is yours to disclose.

## Rights

The texts were published between 1800 and 1918. The catalogue records of 93% of the
Internet Archive documents (84% of their bytes) carry the holding library's public-domain
statement. Author death dates were not checked document by document, so a work whose
author died after 1955 may still be protected where the term is life plus 70 years.
The **Public Domain Mark** applies where the holding library states public domain, and to
the Wolne Lektury documents that library records as public domain. The other Internet
Archive documents (7% of them, 16% of the bytes) carry no rights mark. 79 Wolne Lektury
translations are still in copyright and keep the licence that library publishes them under,
CC BY-SA 3.0 or the Free Art Licence 1.3. The per-document table in this repository
(`ledger/document_metadata_2026-10-06.csv.gz`) gives each document's `rights_basis`
(`library_public_domain_statement`, `scanning_institution_assertion`,
`wolne_lektury_public_domain_note`, `free_licence` or `none`) and names author, translator and licence for the Wolne Lektury rows. **CC0** is asserted over the
compilation layer (selection, cleaning, arrangement), which is the part that could attract
the EU *sui generis* database right.

Redistributing a harvested collection is also bound by platform terms of use and by that
database right. The per-source rights review is archived with the provenance record
(doi:10.5281/zenodo.22099302).

**Takedown:** if you hold rights in a work you believe is included in error, write to
`legal@szymonkocur.com` or open an issue in the code repository, and it will be removed.
The full text lives here and not in the archival record for exactly this reason: a Zenodo
version is effectively immutable, so the archival deposit (doi:10.5281/zenodo.22099302)
holds only facts (identifiers, checksums, reviews) while the texts sit where a verified
claim can actually remove one. The temporal audit exists because catalogue metadata is
wrong often enough to matter; on the predecessor build it let an in-copyright
twentieth-century work through. We assume the current rule has a residue too, and would
rather hear about it than claim there is none.

*Not legal advice. For publication or commercial use, consult a specialist.*

## Known limitations

- **The temporal bound is audited and carries a residual risk.** The audit removes 2.47% of the freeze by
  bytes, and after cleaning every strong
  marker counts zero, but post-1918 text that avoids modern vocabulary and dated
  self-reference evades it.
- **Coverage is inherited.** Whatever the digitising institutions under-represent (by
  region, publisher or genre) this corpus inherits in the same proportion. Four fifths of the text comes from the
  Jagiellonian Digital Library, and 62% of the Internet Archive documents were published
  in Austria-Hungary under the borders of 1914.
- **Polona's press holdings are absent**, because its OCR endpoint is authentication-gated.
  94.2% of the Internet Archive documents are periodical issues, and they are the ones the
  remaining libraries digitised.
- **The Internet Archive dominates**, at 99% of documents. Wolne Lektury contributes clean
  transcriptions but a small fraction of the text.
- **The Wolne Lektury part holds later text.** It is dated by the library's epoch for the
  work. 696 of its 2,714 documents are translations, and 90 documents (0.04% of the corpus
  bytes) are certainly later than 1918: 79 translations still in copyright, which the
  library publishes under CC BY-SA 3.0 or the Free Art Licence 1.3, and eleven volumes of
  Proust in Boy-Żeleński's translation. 141 documents are in other languages. The
  per-document table in this repository marks each of them.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("SKocur/polish-pre1918-corpus", split="train", streaming=True)
print(next(iter(ds))["text"][:500])
```

At 22 GB, `streaming=True` is usually what you want.

## Citation

```bibtex
@dataset{kocur2026corpus,
  title     = {Polish pre-1918 corpus (wieszcz-xix)},
  author    = {Kocur, Szymon},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22099302},
}
```

The accompanying paper carries doi:10.5281/zenodo.21745210.

## See also

- Models trained on this corpus: the `wieszcz-xix` ladder at 47M, 107M and 349M parameters,
  each released at three training points, archived at
  [doi:10.5281/zenodo.22099358](https://doi.org/10.5281/zenodo.22099358) (Apache 2.0).
