# metrics/

Everything at the top level of this directory belongs to one corpus build:

    5,402,429,161 tokens | 219,957 documents | frozen 2026-07-24

Everything under `legacy/` belongs to an earlier, smaller build. **Loss values are not
comparable across that boundary.** The held-out split is the last 1% of `tokens.bin`, so a
different corpus means a different validation set drawn from different documents, and a
model with a lower number there is not a better model.

The corpus grew in four stages, none of which is recorded in the filenames:

| tokens | where it appears | note |
|---|---|---|
| 88,497,698 | `docs/run-log.md`, `docs/training.md` | the 20M/47M/65M experiments |
| 838,072,196 | `docs/run-log.md` | intermediate build, 6320 files |
| 2,441,095,312 | `docs/ar-run-notes.md` | what `legacy/wieszcz_ar_100m_*` trained on |
| 5,402,429,161 | `wieszcz_350m_2026-07-25.out` | frozen; the 349M and the ladder |

This is why the paper's ladder retrains 47M and 107M on the frozen corpus instead of
reusing the 100M numbers already sitting in `legacy/`.

## Top level

| file | what it is |
|---|---|
| `wieszcz_350m_2026-07-25_train.csv` / `_val.csv` | the 349M run, recovered from the stdout log |
| `wieszcz_350m_2026-07-25.out` | that stdout log; the CSVs derive from it |
| `wieszcz_350m_2026-07-25_eval.json` | dense held-out eval: CE 3.03824, PPL 20.868, 8192 windows |
| `wieszcz_350m_2026-07-25_eval.windows.npy` | per-window losses, kept so runs can be compared pairwise |
| `wieszcz_47m_5b_2026-07-27_s1337.out` | the 47M run's stdout log |
| `wieszcz_47m_5b_2026-07-27_s1337_train.csv` / `_val.csv` | written by the run itself, not recovered |
| `wieszcz_47m_5b_2026-07-27_s1337_manifest.json` | seed, corpus size, tokenizer hashes |
| `wieszcz_47m_5b_2026-07-27_s1337_eval.json` | dense eval of `final.pt`: CE 3.42914, PPL 30.850 |
| `..._eval_epoch1.json`, `..._eval_predecay.json` | the same protocol on the epoch-1 and pre-decay checkpoints |
| `duplication.json` | MinHash/LSH scan: 0.141% exact, 0.434% near-duplicate |
| `tokenize_2026-07-24_stream.log` | tokenization of the frozen corpus; source of the 219,957 count |

Every file for a run shares the run's prefix, `<variant>_<date>`, which is the name
`src/metrics_logger.py` generates. Dense evaluations are written with an explicit
`--out metrics/<run>_eval.json` so they join the same prefix rather than the checkpoint's;
evaluations of intermediate checkpoints add a suffix (`_eval_epoch1`, `_eval_predecay`) so
they sort beside the run they belong to.

## The frozen token file

    data/clean/tokens_frozen_5.40B.bin
    10,804,858,322 bytes = 5,402,429,161 uint16 tokens
    sha256 88deac8f55db3848e625a5c02bfdb234f9ea18528542edd95bc6b7b58fecf33c

**This file cannot be regenerated.** Section 6.5 of the paper records why: the frozen stream
ordered documents with the shell's `sort` under a locale collation, while
`build_token_cache()` orders them with Python's `sorted()`, and because the split is
positional the two orderings do not hold out the same documents. Re-running the pipeline
yields a different corpus, not the same one.

The hash above is therefore the only way to tell whether a copy is the corpus these models
were trained on. It was pulled off the RunPod network volume on 2026-07-28 and verified
against the remote hash while that volume was still the only copy. `data/clean/tokens.bin`
is a different, much older build (88M tokens) and is not this file.

Runs started after 2026-07-25 also write `<run>_manifest.json`, which records the seed and a
`tokens.bin` fingerprint. The 349M predates it, so its provenance is the `.out` header.

## legacy/

Runs on the 2.44B-token build, kept only as provenance. The paper no longer places any of
these numbers in a table: a row whose corpus differs from its neighbours' confounds scale
with data volume, so the ladder is retrained on the frozen corpus instead. The one figure
still quoted is the 107M's 3.079, and it is quoted in order to be withdrawn
(the paper repo, `wieszcz-xix-paper/main.tex`, section 6.3) --- these CSVs are where it
came from.

The two `wieszcz_ar_100m` pairs are consecutive segments of one run, not duplicates: the
undated pair covers steps 0-18420 under the old logger, the dated pair covers 16520-32360
after a resume.

The matched diffusion run from the same build, and the diffusion code itself, moved to
the `ar-vs-diffusion-on-wieszcz-xix` repo, where that comparison is being pursued.

Charts are not kept here. They are derived from these CSVs and can be redrawn.
