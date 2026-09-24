# What each script produces, and which claim it supports

The paper reports no number that was typed by hand. Every quantity is a macro that
`paper_numbers.py` generates from a file under `metrics/`, and every one of those files is
the output of a script here. This table is the map from a claim in the paper back to the
code that produced it.

Run order matters only within a stage. Across stages it is strictly top to bottom, because
each stage consumes the frozen output of the last.

## 1. Build the corpus

| script | produces | supports |
|---|---|---|
| `src/prepare_data.py` | the raw crawl under `data/raw/` | § Corpus construction, *Sources and rights* |
| `ia_collection_sweep.py` | `metrics/ia_collection_sweep_*.json`, collection membership per document | *Filtering*, and the lending-library exclusions |
| `wl_epoch_sweep.py` | `metrics/wl_epochs_*.json`, every Wolne Lektury document by catalogue epoch | *Filtering*, the epoch-based arm of the exclusion rule |
| `src/clean_ocr.py`, `clean_corpus.py` | `data/clean/` | *Filtering* |
| `dedup_corpus.py` | `metrics/dedup_*.json`, exact and near-duplicate verdicts | *Deduplication* |
| `anachronism_audit.py` | `metrics/anachronism_audit_*.json`, post-1918 content markers | *The temporal bound* |
| `make_exclusion_list.py` | `exclusions/exclusions_*.json`, the 3,733 removed documents with the rule that selected each | *The temporal bound* |
| `corpus_freeze.py` | `metrics/crawl_universe_*.sha256.gz`, the manifest every later copy is verified against | *The frozen crawl* |
| `build_provenance_ledger.py` | `ledger/provenance_ledger_*.csv.gz`, one row per document | *Sources and rights*, and the archival record |

## 2. Measure it

| script | produces | supports |
|---|---|---|
| `corpus_report.py` | `metrics/corpus_report_*` for `raw` and `final`, one classifier run twice | *Corpus statistics*, *Who digitised it* |
| `corpus_stats.py` | per-source token counts recovered from the frozen token file | *Corpus statistics* |
| `src/analyze_ocr.py`, `audit_frozen_ocr.py` | `metrics/ocr_audit_frozen.json`, corruption rates by type | § OCR corruption, measured |
| `ocr_cer_sample.py` | `metrics/ocr_cer_*.json`, character and word error rates against a hand-keyed sample | *Character and word error rates* |
| `src/measure_duplication.py` | duplication over the token stream rather than the text | *Deduplication* |
| `entropy_baseline.py` | `metrics/entropy_baseline_*.json`, unigram and bigram entropy | § Evaluation, the ends of the loss axis |

## 3. Split and tokenize

| script | produces | supports |
|---|---|---|
| `make_doc_split.py` | `metrics/doc_split_*.json`, whole documents held out before tokenization | *A document-level split* |
| `check_split_leakage.py` | `metrics/split_leakage_*.json`, containment and low-threshold LSH | *A document-level split, tested for leakage* |
| `positional_split_demo.py` | `metrics/positional_split_demo.json`, what the superseded idiom would hold out of the released corpus | *Validation-split source bias, and its repair* |
| `src/train_tokenizer.py` | `tokenizer/` | § Tokenizer |
| `tokenize_corpus.py` | `metrics/tokenize_*.json` and the token bins | § Tokenizer |
| `tokenizer_fertility.py` | `metrics/tokenizer_fertility_*.json`, what the 8,000-token vocabulary costs | § Tokenizer |
| `src/count_tokens.py` | a token count without re-tokenising | § Tokenizer |

## 4. Train

| script | produces | supports |
|---|---|---|
| `src/model.py` | the network. The same file ships with the released weights | *Architecture* |
| `src/modeling_wieszcz.py` | the `transformers` wrapper that ships beside it, so the released repository answers to `AutoModelForCausalLM` | § Availability |
| `src/train.py` | checkpoints, plus `metrics/<variant>_<date>_{train,val}.csv` | *Training the ladder* |
| `src/metrics_logger.py` | that CSV convention | *Training the ladder* |
| `runpod_prep.sh`, `runpod_train.sh` | the rented-pod stages the three rungs ran in | *Training the ladder* |
| `volume_s3.py`, `pull_checkpoints.py` | verified retrieval of checkpoints from the network volume | reproducibility appendix |

## 5. Evaluate

| script | produces | supports |
|---|---|---|
| `src/eval_val.py` | `metrics/eval_*.json` and `*.windows.npy`, per-window losses | *A dense held-out protocol*, *What each rung buys* |
| `eval_by_source.py` | `metrics/eval_by_source_*.json`, the split as a whole and each source alone | *Validation-split source bias* |
| `temporal_probe.py` | `metrics/temporal_probe_*.json` against papuGaPT2 and Bielik | *Is the model temporally bounded?* |
| `src/sample.py`, `make_samples.py` | `output/samples_ladder_*.md`, matched registers across rungs | appendix, *Generation samples* |
| `plot_ladder_curves.py` | `figures/loss_ladder` | *Training the ladder* |
| `plot_350m_curve.py` | `figures/loss_350m` | *Validation-split source bias*, the defect seen from inside a run |

## 6. Measure the prejudice

Read `../docs/release-and-ethics.md` before running these. The sheets contain the text
verbatim.

| script | produces | supports |
|---|---|---|
| `build_bias_prompts.py` | `metrics/bias_prompts_*.json`, a frozen prompt set drawn from held-out text | § Ethical considerations |
| `bias_prevalence.py` | the generation sample the estimate rests on | § Ethical considerations |
| `screen_bias.py` | `metrics/bias_screen_*.json` and `bias_sheet_*.json`, the lexicon screen and the sheet | § Ethical considerations, and *What the prejudice screen reaches* |
| `bias_guidelines.py` | the criterion itself. **Do not edit**: its SHA-256 is recorded in every adjudication report, and its single revision between judge runs is a control the paper reports | § Ethical considerations |
| `review_sheet.py`, `label_studio_io.py` | the human annotation path, terminal and Label Studio | § Ethical considerations |
| `llm_adjudicate.py` | `output/adj*.json`, five models scoring the same sheet, and `metrics/bias_adjudicator_agreement_*.json` | § Ethical considerations, the between-judge spread |

## 7. Release

| script | produces |
|---|---|
| `export_model.py`, `export_release.sh` | the nine self-contained bundles on the weights record |
| `build_hf_dataset.py`, `assemble_dataset_release.sh` | the corpus as published on Hugging Face |
| `build_hf_models.py` | the model repository, both entry points verified against each other before it prints success |
| `paper_numbers.py` | `numbers.tex`. `--check` reports any number typed into the paper by hand that a macro already holds |

## The one that guards the rest

`paper_numbers.py --check` is the only thing standing between a mistyped number and the
published PDF, because the LaTeX source is not released and a reader cannot rebuild the
document to find a transcription error. It has caught one already: an exponent interval
that appeared four times in the paper and in no measurement file at all.
