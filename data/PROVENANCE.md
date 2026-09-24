# Corpus provenance and licensing

The rule the corpus is built to: **only public-domain Polish text published 1800 to 1918,
in its original orthography.** Publication date is enforced at the crawl and then audited
by content before tokenisation, which removed 3,733 documents from the frozen build.

## Sources actually in the frozen build

| Source | Access | Rights status | What it contributes |
|---|---|---|---|
| Internet Archive | `archive.org` item API, language tag `pol` | Per-item public domain | 291,655 documents of OCR in original spelling, the bulk of the corpus and all of its register breadth |
| Wolne Lektury | `wolnelektury.pl/api` | Public domain, CC | 2,714 transcribed documents, no OCR, modernised spelling |

Nothing else is in it. Polona holds the richest period press, and it was rejected because
its OCR endpoint answers 401 and cannot be scripted into a reproducible pipeline. The
Federacja Bibliotek Cyfrowych libraries were reached through the Internet Archive rather
than individually. `docs/data-preparation.md` records the full list of sources considered
and why each was taken or dropped.

## Where the per-document record lives

This file states the rule. The evidence is a ledger with one row per document, giving its
identifier, its source, that source's own resolvable identifier, and its byte size:

- `ledger/provenance_ledger_2026-08-03_train.csv.gz`, 291,605 rows
- `ledger/provenance_ledger_2026-08-03_val.csv.gz`, 2,764 rows
- `exclusions/exclusions_2026-08-03.json`, the documents the temporal audit removed, each
  with the rule that selected it

The same three files are published in the archival record at
[doi:10.5281/zenodo.22099302](https://doi.org/10.5281/zenodo.22099302), alongside
`RIGHTS-REVIEW.md`, which is the per-source rights review for this build, and the Wolne
Lektury per-slug licence sweep behind it.

## Rights checklist, per source

- Author died more than 70 years ago, so the work is public domain in PL and the EU.
- Original orthography preserved, with no modernised or critical-edition text.
- The source platform's terms of use are respected: rate limits, attribution, and the
  per-item rights label.
- The collection is assembled from individual items rather than bulk-copied, which is what
  the EU sui generis database right turns on.

The underlying texts are public domain. The constraint that actually binds when
redistributing is the database right over the compilation, which is why the released
corpus carries CC0 1.0 over the compilation and database layer while leaving the texts
themselves in the public domain where they already are.

> Not legal advice. For commercial use, consult a specialist.

Rights, safety or takedown concerns: legal@szymonkocur.com
