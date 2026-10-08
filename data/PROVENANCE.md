# Corpus provenance and licensing

The rule the corpus is built to: **Polish text published 1800 to 1918, in its original
orthography where the source keeps it.** Publication date is enforced at the crawl and then audited
by content before tokenisation, which removed 3,733 documents from the frozen build.

## Sources actually in the frozen build

| Source | Access | Rights status | What it contributes |
|---|---|---|---|
| Internet Archive | `archive.org` item API, language tag `pol` | Library public-domain statement on 93% of documents, none on the rest | 291,655 documents of OCR in original spelling, the bulk of the corpus and all of its register breadth |
| Wolne Lektury | `wolnelektury.pl/api` | Public domain, except 79 translations under CC BY-SA 3.0 or the Free Art Licence 1.3 | 2,714 transcribed documents, no OCR, modernised spelling |

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

- Published between 1800 and 1918, with the holding library's public-domain statement
  where it gives one. Author death dates were not checked document by document.
- Original orthography preserved, with no modernised or critical-edition text.
- The source platform's terms of use are respected: rate limits, attribution, and the
  per-item rights label.
- The collection is assembled from individual items rather than bulk-copied, which is what
  the EU sui generis database right turns on.

The checklist is the rule the crawl was built to. What each document's status rests on is
recorded per document in `ledger/document_metadata_2026-10-06.csv.gz`: a holding library's
public-domain statement, a weaker assertion by the scanning institution, Wolne Lektury's
public-domain note, a free licence, or nothing.
The Public Domain Mark is applied only where a library states public domain or Wolne
Lektury notes it. The released corpus carries
CC0 1.0 over the compilation and database layer.

> Not legal advice. For commercial use, consult a specialist.

Rights, safety or takedown concerns: legal@szymonkocur.com
