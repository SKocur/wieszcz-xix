# Data preparation — design rationale

Why the corpus pipeline (`src/prepare_data.py`) makes the choices it does. Every
decision here is **in code**, not applied by hand, so the corpus is reproducible from a
clean checkout: `prepare_data.py` → `prepare_data.py --reclean` → `train_tokenizer.py`.
No manual "delete the bad file" steps — if a rule is worth applying, it is a function.

## Goal

Train a small model **from scratch on 19th-century Polish only**, so it speaks in the
language and worldview of the period. Two things follow:

1. **No modern-language leakage.** Editorial footers, library boilerplate, and OCR of
   non-period front matter are modern Polish and must be stripped — they would teach the
   model 21st-century phrasing.
2. **Register breadth over pure size.** A corpus of only literary classics biases the
   model toward poetic style and gives it little factual "knowledge of the world." We
   deliberately mix literary text with press-adjacent non-fiction, history, and science.

## Sources: chosen and rejected

| Source | Role | Why / why not |
|--------|------|---------------|
| **Wolne Lektury** | clean literary core | Transcribed (no OCR), public domain, one plain-text file per work. Spelling is *modernized*, but the language and worldview are period. |
| **Wikiźródła** | curated non-fiction | Clean transcriptions; pulled from a hand-listed title file for provenance control. |
| **Internet Archive** | breadth (history, science, memoirs) | Open, scriptable, full OCR text via `_djvu.txt`. Original pre-1936 spelling. This is the TimeCapsuleLLM approach. |
| Polona (National Library) | *rejected* | Richest source of period **press**, but the OCR endpoint is auth-gated (HTTP 401) — not scriptable into a reproducible pipeline. |
| HathiTrust | *rejected* | Has 19th-c Polish, but bulk full-text access is gated behind an agreement. |
| Clean novel corpora (ELTeC-pol, MetaPNC, 100_polish_novels) | *candidate* | Clean and easy, but literary-only and usually *modernized* spelling — they add fluency, not register breadth. Deferred; dedup against Wolne Lektury needed first. |

### The Internet Archive language tag — a 210× difference

IA tags Polish with the **ISO 639-2 code `pol`**, not the English word `Polish`. For
1800–1918 the two differ enormously:

| Query | Items |
|-------|-------|
| `language:(Polish)` | ~1,700 |
| `language:(pol)` | **~358,000** |
| `language:(pl)` | 0 (not used) |

The first corpus build used `language:(Polish)` and therefore saw roughly **0.5%** of what
is available. The query now ORs both tags. Sampling the `pol` results confirms they are
genuine period material across registers — a 1808 satirical comedy, 1890 funeral sermons,
an 1884 textbook on infectious diseases, gazetteers, ethnographic maps.

The practical consequence: **the corpus is no longer capacity-limited by availability**,
only by download time and disk. `IA_MAX_ITEMS` caps each pull; fetching is incremental, so
re-running goes deeper rather than starting over.

**The core trade-off:** clean transcriptions (Wolne Lektury, novel corpora) tend to be
*modernized spelling*; original-spelling text (Internet Archive, Polona) comes as *noisy
OCR*. Since the stated priority is period **language + worldview**, not orthographic
fidelity, modernized-but-clean text is acceptable and OCR adds the authentic-spelling
flavour on top. We do not over-optimize for spelling.

## The filtering pipeline

Filters apply **per source**. Wolne Lektury and Wikiźródła are trusted (curated Polish,
no OCR), so only Internet Archive — an open firehose with unreliable metadata — passes
through the quality and language gates. Order: fetch → clean → gate.

### 1. Footer / boilerplate stripping

- **Wolne Lektury** appends a license/source block after a dashed line. It is modern
  Polish (URLs, editors' names, "reprodukcja cyfrowa"). We split on `\n-{3,}\n`.
  *Lesson learned:* the files are CRLF, so newline normalization must run **before** the
  split — doing it after left the footer in place. For a short poem the footer was >50%
  of the file, so this bug mattered.
- **Internet Archive** Google-digitized scans open with a bilingual Google Books notice.
  We cut through the last "Google" mention in the header region — a 19th-century book
  never says "Google", so this is safe.

### 2. Quality gate — `alpha_ratio ≥ 0.60`

**Metric:** share of whitespace tokens that are clean alphabetic words
(`[A-Za-ząćęłńóśźż-]+`).

**Why this metric.** IA "texts" include abbreviation dictionaries, indexes, statistical
tables, and badly-OCR'd scans. These are not usable prose, but they are hard to spot by
title. A token that is a clean word is a good proxy for "running prose"; dictionaries and
garbled OCR are full of numbers, single letters, and punctuation-mangled tokens.

**Why 0.60.** Measured, not guessed:

| File | `alpha_ratio` |
|------|---------------|
| IA geographic-abbreviation dictionary (garbage) | 31% |
| IA history book (*Jadwiga i Jagiełło*) | 79% |
| IA Mickiewicz, *Dzieła* | 77% |
| Wolne Lektury (clean) | 76% |

Good prose clusters at 75–80%; non-prose sits near 30%. A 0.60 threshold separates them
with wide margin on both sides. It removed ~10% of the raw IA pull (dictionaries, science
journals that are mostly tables, one 0%-alphabetic scan).

**Why not something heavier** (a language model, perplexity filter): overkill for a hobby
corpus, adds a dependency and non-determinism, and a single ratio already separates the
classes cleanly.

### 3. Language gate — `is_polish`

**Problem the quality gate misses:** clean prose *in the wrong language*. IA's
`language:Polish` tag is unreliable — English, French, Latin, Slovenian, and Italian
books slipped through, and they score *high* on `alpha_ratio` (they are clean prose),
so the quality gate passes them.

**Metric:** Polish diacritic density **or** Polish stopword share.

**Why both, combined with OR.** Measured:

| File | stopword % | diacritic % |
|------|-----------|-------------|
| IA Italian (`iduellimortali`) | 0.0% | 0.55% |
| IA Polish files | 3.8–5.7% | 4.6–7.5% |
| Wolne Lektury | 5.7% | 7.5% |

Diacritics (ą ć ę ł ń ó ś ź ż) are the strongest signal — Italian, English, Latin, and
French essentially lack them. But heavy OCR can strip diacritics from genuinely Polish
text, so we back it with Polish **stopwords** ("się", "że", "nie", …). Either signal
clearing its threshold (diacritics ≥ 2% *or* stopwords ≥ 1.5%) accepts the file; a
non-Polish book fails both. The thresholds sit far below every Polish sample and far
above the Italian one. This removed 28 non-Polish files.

**Why not a langdetect library.** Same reasoning as above: a two-line character/stopword
heuristic is deterministic, dependency-free, and the calibration gap is enormous
(0.55% vs 4.6%+). A probabilistic detector would add a dependency and a coin-flip on
short or OCR-noisy inputs.

**Stopword list** deliberately excludes cross-Slavic ambiguous words ("i", "w", "na",
"do", "z") and keeps distinctly-Polish ones ("się", "że", "który", "żeby", …), so the
signal is Polish, not merely Slavic.

### 4. Deterministic ordering

The IA search sorts by `identifier asc`, so a fresh run fetches the *same* top-N set
every time. Without an explicit sort, relevance order can drift and break
reproducibility.

## Second pass — what the bigger crawl exposed

Widening the query surfaced material the first-pass gates were too loose for.

### Old-Polish chancery records and Cyrillic

19th-century *editions* of 17th-century court records (`Akta grodzkie i ziemskie`,
`Akty…`) sailed through: they are Polish, they are prose, and they were published in our
window. But the language inside is **Old Polish plus Latin** — `msca`, `xięztwie`, `y` for
`i`, `Sąd reddendarum rationum` — two centuries earlier than the target era. Some volumes,
published under Russian administration, carry substantial **Cyrillic**.

They passed because the language gate accepted *either* signal. Measured:

| Material | stopwords | diacritics |
|----------|-----------|------------|
| Good 19th-c prose | 3.8–5.7% | 4.6–7.5% |
| Chancery records / Russian-era editions | 1.9–2.6% | 3.3–3.4% |

Fixes: require **both** signals (`and`, not `or`), raise the bars to 3% / 4%, and reject
text with more than 0.5% Cyrillic characters. Verified against known-good and known-bad
files: all three suspect volumes rejected, all three good ones kept; 48 of 363 existing
files dropped.

### Alphabetical sorting is not a sample

`sort=identifier asc` was chosen for reproducibility, but identifiers correlate with
titles, so the first N items arrived as a run of `akta…` / `akty…` record collections —
a systematically skewed slice of the catalogue. Switching to **`downloads desc`** is
equally deterministic while spreading across the collection and favouring well-digitized
mainstream works. The material changed immediately: *Chimera*, *Biblioteka Warszawska*,
Academy of Sciences proceedings, Orgelbrand's encyclopedia, the natural-science weekly
*Kosmos* — stopword rates of 5.8–7.5%, the top of the range.

### Reference works stay — a deliberate non-tightening

Measuring both gates across all 2171 accepted IA files put the stopword distribution at
p50 = 0.065, p5 = 0.041, floor 0.030 (the gate). The bottom of that tail is **not** garbage,
it is *catalogue-shaped* material — few function words because it is entries, not sentences:

| stopwords | alpha | size | work |
|-----------|-------|------|------|
| 0.031 | 0.707 | 2.0 MB | Niesiecki's armorial |
| 0.033 | 0.641 | 2.0 MB | Orgelbrand's encyclopedia |
| 0.033 | 0.624 | 2.7 MB | dictionary of foreign words |
| 0.039 | 0.618 | 1.8 MB | *Pamiętnik fizyograficzny* (plant-locality index) |

Files under 0.05 stopwords are **373 files / 391 MB — 20% of the IA corpus**. Raising the
threshold to 0.05 would remove all of it, and to 0.035 would remove the armorials and the
dictionary.

**Decision: leave the gate at 0.03.** The goal is a model with *knowledge of the period's
world*, and an encyclopedia is the densest source of exactly that — a stopword filter cannot
tell "list of noble families" from "list of facts about the world," and losing the latter
costs more than keeping the former. A model that has read Orgelbrand knows what a 19th-century
Pole knew; one trained only on novels knows how they talked. We want both. The 0.60 alpha gate
still catches genuinely unusable scans, which is the failure this pipeline actually needs to
prevent.

### Retroactive filtering

Tightening a threshold must apply to what is already on disk, or the corpus becomes a
sediment of whichever rules were in force when each file arrived. `--prune` re-runs the
current gates over `clean/` and deletes failures. It works off `clean/` rather than
`raw/`, so it still applies when raw was never kept locally.

## The rejection ledger

`already_have()` only knows about files on disk, so a text that *failed* the gates left no
trace and was downloaded again on every resume. After the second campaign that was roughly
700 wasted downloads per restart — about ten minutes each time.

`data/rejected.jsonl` records one line per content rejection:

```json
{"id": "ia_akta1234", "reason": "stopword", "gate": "83b6fe89eb8d", "alpha": 0.71, "stopword": 0.021}
```

Three decisions worth keeping:

**Only content rejections are recorded.** A 503 or a timeout says nothing about the text; a
blacklist on transient failure would permanently lose a good book because archive.org
hiccuped once. Network failures stay retryable, gate failures do not.

**Every entry carries a gate fingerprint** — a hash of the four thresholds *and* the stopword
set. The ledger is only true while the gates are unchanged: a file rejected at
`stopword >= 0.03` might pass at 0.02. Entries whose fingerprint no longer matches are
ignored, so loosening a threshold silently re-opens exactly the items it had excluded. This
is the same reproducibility rule as `--prune`, in the other direction: tightening must reach
backwards into the corpus, loosening must reach backwards into the ledger.

**Reasons are stored, not just identifiers.** A gate is judged by what it actually throws
away, and `--rejected` prints the breakdown:

```
$ python src/prepare_data.py --rejected
814 rejections under the current gates (83b6fe89eb8d)
  stopword     402
  alpha        289
  diacritic    101
  cyrillic      22
```

If one reason dominates unexpectedly, that threshold is miscalibrated.

Writes go through a lock and append a single line, so six worker threads cannot interleave,
and a torn line from a killed run is skipped rather than crashing the next crawl.

## Crawl reliability

Three failures, each worth its fix:

**IPv6 with no route.** `wolnelektury.pl` died with `Errno 101 Network is unreachable`
while `archive.org` worked and `ping` looked fine. Cause: WSL2 has no IPv6 route,
`wolnelektury.pl` publishes an AAAA record, `archive.org` does not, and `ping` defaults to
IPv4. Python resolved AAAA first and hit a dead route — a **deterministic** failure that
retries could never fix. `urllib3.util.connection.allowed_gai_family` is pinned to
`AF_INET`.

**No retries.** A multi-hour crawl meets transient DNS failures and 429/5xx from the
archives. All requests now go through a `Session` with exponential backoff.

**One dead source killed the run.** `fetch_wolne_lektury` raising took the Internet
Archive fetch down with it, even though Wolne Lektury was already complete. Each source is
now isolated: a failure logs a warning and the crawl continues.

**Parallelism.** Downloads are latency-bound, not CPU-bound — six worker threads took the
rate from 4.04 s/item to 1.69 s/item.

## Known limitations (documented, not hidden)

- **Press is thin.** Daily newspapers — the best source of everyday register and world
  events — live mostly on the auth-gated Polona, so the breadth here is books, not press.
- **OCR noise remains** in accepted IA files (period typefaces mangle letters). Measured at
  **≈2.8% of words**, line-local rather than document-local — see the OCR-corruption audit
  below. The gates remove the *worst files*, not the noise; that is the deliberate
  breadth-vs-cleanliness trade-off, and the fix is line/character-level, not word-deletion.
- **No cross-source dedup yet.** A public-domain work (e.g. Mickiewicz) can appear in both
  Wolne Lektury (clean) and Internet Archive (OCR). Left in for now — the two versions
  differ in spelling — but a dedup-by-author/title step is the next candidate.

## OCR corruption — measured, not guessed

`src/analyze_ocr.py` quantifies how much of the accepted corpus is OCR-mangled and, more
importantly, *how it is distributed* — because that decides the fix. It classifies every
whitespace token with cheap character heuristics (noise symbol inside a word; digit welded
to a letter; interior uppercase after a lowercase, e.g. `SoKoła`; a ≥4-letter word with no
vowel) and reports rates by source, per-document histograms, and a drop-threshold table.

**Result (full corpus, 333.9M words):**

| Source | words | suspicious | rate |
|--------|------:|-----------:|-----:|
| ia (OCR) | 309.5M | 9.42M | **3.05%** |
| wl (clean) | 24.4M | 63.7k | 0.26% |

The 0.26% on the clean Wolne-Lektury half is the detectors' false-positive floor, so true
OCR corruption is **≈2.8%**. By type: symbol 2.02%, midcaps 0.73%, digit-in-word 0.28%,
no-vowel 0.05%.

**The corruption is line-local, not document-local.** The per-document histogram is not
bimodal: most IA docs sit at 1–5%, a tail of ~100 sit at 10–25%, and the *dirtiest single
document is only 27%*. There is no "60% garbage" file, because within a badly-scanned book
the junk concentrates in mastheads, price tables, running heads and page numbers while the
body prose dilutes it. Consequently **document-level dropping is a bad lever** — the
threshold table shows that to remove meaningful garbage you must discard mostly-good text:

| drop docs above | docs dropped | text lost |
|-----------------|-------------:|----------:|
| 25% | 1 | 0.01% |
| 15% | 20 | 0.60% |
| 10% | 100 | 3.30% |
| 5% | 514 | **17.6%** |

**The vocabulary is already clean.** A decode-aware audit of `vocab.json` (byte-level BPE
encodes Polish diacritics as visible Latin-1 — `ł`→`ÅĤ` — so a naïve scan reports 22.9%
false garbage; tokens must be byte-decoded first) finds **<1%** real garbage in the 8k
vocab: zero digit-welded tokens, and only a handful of Google-Books watermark scraps
(`VjOOQIC`). `min_frequency=2` already excludes hapax OCR junk, so a string like
`bezgraniczn3rm` is **not a token** — when it appears in generation it is the model
reassembling learned byte fragments, not a polluted vocabulary.

**Fix strategy that follows from the shape of the data** (ranked by value, chosen to respect
"never delete a word — it breaks syntax"):

1. **Line-level filtering** — drop lines whose garbage density exceeds a threshold (mastheads,
   tables, running heads). Targets the real structure; keeps prose sentences whole.
2. **Mechanical character repair, in place** — strip embedded noise symbols (2.02%) and
   lowercase interior case-flips (0.73%). Fixes the word without moving it; ~2.7% of the 2.8%.
   Deeper letter substitutions (`rn→m`, `l→i`) survive and need a dictionary pass.
3. **`min_frequency` is *not* the lever** — the vocab is already clean (measured), and raising
   the threshold cannot distinguish a rare archaic word from rare garbage, so it would evict
   real period vocabulary for negligible gain. The damage lives in the token *stream* (learned
   patterns), which only source-cleaning touches.

## Tunable parameters

All near the top of `prepare_data.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `IA_MAX_ITEMS` | 5000 | cap per IA pull (re-run fetches the next batch) |
| `IA_MIN_ALPHA` | 0.60 | quality gate |
| `IA_MIN_DIACRITIC` | 0.04 | language gate (diacritic density) |
| `IA_MIN_STOPWORD` | 0.03 | language gate (stopword share) |
| `IA_MAX_CYRILLIC` | 0.005 | reject Russian-era editions |
| `IA_WORKERS` | 6 | parallel downloads |
| `EPOCHS` | romantyzm, pozytywizm, modernizm | Wolne Lektury 19th-c epochs |

## Reproducing the corpus

```bash
python src/prepare_data.py             # fetch WL + Wikiźródła + IA (gated at fetch time)
python src/prepare_data.py --reclean   # re-apply all cleaning/gates offline from raw/
python src/prepare_data.py --prune     # re-apply gates to clean/, drop failures
python src/prepare_data.py --rejected  # what the gates threw away, and why
rm -f data/clean/tokens.bin            # force re-tokenization after any corpus change
python src/train_tokenizer.py
```

Re-running the crawl is cheap: files in `clean/` are skipped by `already_have()`, and
identifiers in `rejected.jsonl` are skipped without a download.

Corpus after the second campaign (crawl still running at the time of writing):
**5035 files** (2992 Wolne Lektury + 2171 Internet Archive, minus rejects), **2.1 GB**.
The first build was 3222 files / 285 MB — the `pol` language tag is the whole difference.
