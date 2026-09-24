# metrics/

The top level of this directory spans two frozen corpus builds. Files dated 2026-08-03
or later, and every run tagged `_6b7`, belong to the build the paper reports:

    6,745,956,943 tokens | 294,369 documents | frozen 2026-08-03
    document-level split: 6,689,593,236 train / 56,363,707 held-out
    (`tokenize_2026-08-03.json` records the counts, `doc_split_2026-08-03.json` the split)

Everything dated July 2026 belongs to its predecessor, 5,402,429,161 tokens, 219,957
documents, frozen 2026-07-24, kept as provenance for the runs that trained on it
(`wieszcz_350m_2026-07-25`, `wieszcz_47m_5b`, `wieszcz_107m_5b`). **Loss values are not
comparable across builds**, and between these two the split protocol changed as well: the
old build held out the positional last 1% of the token stream, the current one holds out
whole documents drawn at random. A model with a lower number against a different split is
not a better model. Everything under `legacy/` is older still.

The corpus grew in five stages, none of which is recorded in the filenames:

| tokens | where it appears | note |
|---|---|---|
| 88,497,698 | `docs/run-log.md`, `docs/training.md` | the 20M/47M/65M experiments |
| 838,072,196 | `docs/run-log.md` | intermediate build, 6320 files |
| 2,441,095,312 | `metrics/legacy/wieszcz_ar_100m_*` | what `legacy/wieszcz_ar_100m_*` trained on |
| 5,402,429,161 | `wieszcz_350m_2026-07-25.out` | frozen 2026-07-24; the v1 350M and ladder |
| 6,745,956,943 | `tokenize_2026-08-03.json` | frozen 2026-08-03; everything in the paper |

This is why the paper's ladder retrains 47M and 107M on the frozen corpus instead of
reusing numbers already sitting beside them: a row whose corpus differs from its
neighbours' confounds scale with data volume.

## Reading the top level

Every file for a run shares the run's prefix, `<variant>_<date>`, which is the name
`src/metrics_logger.py` generates. Dense evaluations are written with an explicit
`--out metrics/<run>_eval.json` so they join the same prefix rather than the checkpoint's;
evaluations of intermediate checkpoints add a suffix (`_eval_epoch1`, `_eval_predecay`) so
they sort beside the run they belong to. The paper's ladder runs are
`wieszcz_{47m,107m,349m}_6b7_*` (manifests and CSVs) with stdout in
`train_wieszcz_*_6b7.out`.

Two families of files supersede each other in place:

- `eval_{47m,107m,349m}_{epoch1,predecay}_2026-08-17.json` and
  `eval_by_source_2026-08-17.json` came from a harness that anchored windows with an
  integer stride: 89% of the split covered, zero Wolne Lektury windows. The repaired
  readings are `eval_*_2026-08-20.json` and `eval_by_source_2026-08-17_fullspan.json`.
  Current reports carry `span_fraction`, and `scripts/paper_numbers.py` refuses any below
  0.999; a report with a `window_stride_tokens` key and no `span_fraction` is pre-repair.
- `bias_sheet_349m_2026-08-17.json` is the first adjudication sheet, superseded by
  `bias_sheet_349m_2026-08-19.json` (strata mixed, rechecks spread) after the external
  review of the screen.

## The frozen token files

The current build, in `data/` at the repo root:

    data/tokens_frozen_6.69B.bin   13,379,186,472 bytes = 6,689,593,236 uint16 tokens (train)
    data/val_2026-08-03.bin           112,727,414 bytes =    56,363,707 uint16 tokens (held-out)

Each `_6b7` run's `<run>_manifest.json` records the seed and the fingerprint of the token
file it trained against, so a silent data change cannot masquerade as the same run.

The predecessor's stream was deleted locally in the 2026-08-21 cleanup; its hash is its
record:

    data/clean/tokens_frozen_5.40B.bin  (deleted)
    10,804,858,322 bytes = 5,402,429,161 uint16 tokens
    sha256 88deac8f55db3848e625a5c02bfdb234f9ea18528542edd95bc6b7b58fecf33c

That file could not be regenerated even while it existed: the frozen stream ordered
documents with the shell's `sort` under a locale collation, `build_token_cache()` orders
them with Python's `sorted()`, and with a positional split the two orderings do not hold
out the same documents. The hash is the only way to tell whether a surviving copy (the
retired RunPod volume may hold one) is the stream the v1 runs trained on.

## legacy/

Runs on the 2.44B-token build, kept only as provenance. The paper no longer places any of
these numbers in a table: a row whose corpus differs from its neighbours' confounds scale
with data volume, so the ladder is retrained on the frozen corpus instead. The one figure
still quoted is the 107M's 3.079, and it is quoted in order to be withdrawn
(the paper repo, `wieszcz-xix-paper/main.tex`, section 6.3), these CSVs are where it
came from.

The two `wieszcz_ar_100m` pairs are consecutive segments of one run, not duplicates: the
undated pair covers steps 0-18420 under the old logger, the dated pair covers 16520-32360
after a resume.

The matched diffusion run from the same build, and the diffusion code itself, moved to
the `ar-vs-diffusion-on-wieszcz-xix` repo, where that comparison is being pursued.

Charts are not kept here. They are derived from these CSVs and can be redrawn.
