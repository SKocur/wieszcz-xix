# Training — setup, hardware paths, and gotchas

Companion to [`data-preparation.md`](data-preparation.md). Everything below is about
turning the prepared corpus into a trained model.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### WSL: keep the project on ext4, not `/mnt/c`

Put the repo in the Linux home (`~/wieszcz-xix`), **not** under `/mnt/c/...`. On the
Windows mount:

- `python3 -m venv` fails (`ensurepip` non-zero exit, `Operation not permitted` on
  `activate.fish`) because the mount does not support the symlinks/permissions venv needs;
- file I/O is much slower, which matters when tokenizing thousands of files.

### `torch.compile` needs a Linux `nvcc` (the WSL trap)

TorchInductor shells out to `nvcc`. WSL inherits the Windows `PATH`, so `nvcc` often
resolves to a Windows binary Linux cannot execute:

```
PermissionError: [Errno 13] Permission denied: 'nvcc'
torch._inductor.exc.InductorError: ...
```

Two ways out — pick one:

| Option | Command | Effect |
|--------|---------|--------|
| **Install the toolkit** | `sudo apt install -y nvidia-cuda-toolkit` (~2 GB) | real Linux `nvcc`, `torch.compile` works |
| **Disable compile** | set `"compile": false` in the config | skips the compiler; costs some speed only |

The failure is deliberately left loud: suppressing it (e.g. via
`torch._dynamo.config.suppress_errors`) would silently drop you into eager mode while you
believe the model is compiled. Better to crash with the real error and pick one of the two
fixes above. bf16 and FlashAttention are unaffected either way — `torch.compile` is a
speed optimization, not a requirement.

Do **not** install an NVIDIA *driver* inside WSL; the Windows driver provides the GPU.
Verify with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.

## Hardware paths

`get_device()` picks `cuda → mps → cpu` and the run prints what it got:

| Device | Precision / kernels | Use |
|--------|--------------------|-----|
| **CUDA** (RTX 3080) | bf16 autocast + FlashAttention (SDPA) + optional `torch.compile` | the real runs |
| **MPS** (M1) | fp32, no FlashAttention kernel, no compile | prototyping and pipeline validation only |
| CPU | fp32 | last resort |

Muon's Newton-Schulz iteration runs in bf16 on CUDA and fp32 elsewhere, since MPS/CPU
lack fast bf16.

Expect roughly a 5–15× speed difference between the 3080 and an M1 Pro — develop on the
laptop, train on the GPU.

## Configs

| Config | Params | Purpose |
|--------|--------|---------|
| `wieszcz_debug.json` | ~1.4M | smoke test — 200 steps, verifies the whole chain |
| `wieszcz_m1.json` | ~5M | longer laptop stress test — 2000 steps |
| `wieszcz_20m.json` | ~16M | the real run — 5000 steps |

```bash
python src/train.py --config configs/wieszcz_20m.json
```

## Right-sizing to the corpus

The corpus is ~88.5M tokens, which drove two deliberate choices:

- **Vocab 8k, not 16k.** At `n_embd=512` a 16k vocab puts 8.2M params in the (tied)
  embedding table — ~40% of a 20M model. 8k halves that and hands the budget to the
  transformer blocks. The cost is a higher tokens/word ratio (byte-level BPE spends
  tokens on Polish diacritics), which is the accepted trade of matching the SOTA
  tokenizer family at small scale.
- **5000 steps, not 20000.** At `batch 32 × grad_accum 4 × block 512` = 65,536 tokens per
  step, 5000 steps ≈ 3.7 epochs. The original 20000 would have been ~15 epochs on a small
  corpus — a direct route to memorization. Raise this only while watching validation loss.

## What healthy training looks like

The header confirms the hardware path:

```
Corpus: 88,497,698 tokens (884,976 val) | device: cuda | bf16 + FlashAttention
Model: ~16M params | GQA 8q/2kv | Muon+AdamW
```

Then loss should fall from ~9.0 (random init over an 8k vocab) and **validation should
track training**. A 2000-step run of the 5M config on an M1 produced:

| Step | train | val |
|------|-------|-----|
| 0 | 9.03 | 9.03 |
| 800 | 5.55 | 5.47 |
| 2000 | 5.05 | 5.02 |

Validation sitting at or just below training loss means it is learning, not memorizing —
expected, since 2000 steps covered only ~9% of one epoch. **Validation rising while
training falls is the signal to stop.**

## Sizing the second model — memory is the constraint, not parameters

With the corpus heading for ~1.6B tokens, Chinchilla puts the optimum near **65M
parameters** (`configs/wieszcz_65m.json`: 14 layers, `n_embd` 640, GQA 10q/2kv,
`block_size` 1024). Choosing the batch is where estimates go wrong.

**Activations dominate, not weights.** 65M parameters in bf16 is 130 MB; the memory goes
to activations, which scale with `batch × sequence × width × depth`. Anchored on a
measurement — the 16M model (8 layers, `n_embd` 512) at `batch 32 × block 512` used ~6 GB:

| Configuration | vs. the 16M anchor | Estimated |
|---------------|--------------------|-----------|
| batch 32 × block 512 | ×2.2 | ~13 GB ❌ |
| batch 16 × block 512 | ×1.1 | ~6.5 GB ✅ |
| batch 16 × block 1024 | ×2.2 | ~13 GB ❌ |
| batch 8 × block 1024 | ×1.1 | ~6.5 GB ✅ |

An earlier suggestion of `batch 64 × grad_accum 2` was right for the 16M model and badly
wrong for this one. Keep the effective batch at 128 and shift the split toward
accumulation: **`batch 8 × grad_accum 16`**. Note that FlashAttention is what makes this
tractable at all — without it the `B × n_head × S × S` attention matrix alone would be
over a gigabyte per layer at these settings.

Since these are estimates, `train.py` prints measured VRAM at step 10 so the batch can be
raised on evidence rather than arithmetic.

**Dropout is 0.0 here**, against 0.1 for the first model. Dropout regularizes against
repetition; at roughly one pass over 1.6B unique tokens there is nothing to regularize
against, and it would only slow learning.

## Context length

`block_size` is the training context and simultaneously a hard inference ceiling — RoPE
tables are precomputed for exactly that many positions, so a trained model cannot simply
be given a longer window by editing the config.

At 2.4 tokens per word, 512 tokens is a long paragraph and 1024 is roughly a page. That is
far below the 262k of a current flagship, and deliberately so: attention compute grows
quadratically with length, and a 65M model cannot exploit long-range structure anyway —
its coherence breaks down after a couple of sentences, not a couple of pages. The standard
escape hatch remains open: pretrain short, then extend with RoPE scaling (`rope_theta`)
in a separate phase, which is how long-context models were built before flagship budgets
made long pretraining affordable.

## Document boundaries

The token cache inserts `<|endoftext|>` between documents. Without it the stream runs the
last sentence of one book straight into the first of the next, and the model learns
transitions that do not exist — Mickiewicz into a medical textbook.

## Running long jobs on the Windows box

**Logging out of Windows destroys the WSL VM**, and everything inside it, including tmux
sessions. tmux protects against a dropped terminal or SSH connection, not against the
whole Linux instance being torn down — and starting the job over SSH does not help either,
because the VM is tied to the user session regardless of what launched the process.

Lock the screen (`Win+L`) instead of logging out. Long crawls also write to
`data/crawl.log`, so progress is readable without attaching:

```bash
tail -f ~/wieszcz-xix/data/crawl.log
tmux attach -t data
```

## Token cache

The first run encodes the whole corpus to `data/clean/tokens.bin` (a few minutes, with a
progress bar) and memory-maps it afterwards. **Delete it after any corpus change**, or
training silently reuses the stale tokenization:

```bash
rm -f data/clean/tokens.bin
```

Files starting with `.` are skipped when globbing the corpus — macOS `tar` emits
AppleDouble `._*` companions that match `*.txt` but are binary, and they used to crash
tokenization with a `UnicodeDecodeError`.

## Sampling

```bash
python src/sample.py --ckpt models/wieszcz-349m-2026-07-25/model.pt --prompt "Rankiem, gdy"
```

Generation uses a per-layer KV-cache (`GPT.generate`) — the prompt is prefilled once, then
each new token only computes its own K/V and attends to the cached past. The cache is an
inference-only optimization; training processes the whole sequence in parallel and has no
use for it.

## The matched-data ladder (`*_5b` configs)

`wieszcz_47m_5b.json` and `wieszcz_107m_5b.json` exist to make the model ladder mean
something. The original 47M and 100M runs each trained on whatever corpus build existed
that week, so parameter count and data volume grew together and no comparison between them
isolates the effect of scale. These configs retrain both on the *same* frozen 5.40B-token
build the 349M used.

Everything that could confound the comparison is held identical across all three:

| | 47M | 107M | 349M |
|---|---|---|---|
| effective batch | 256 seq | 256 seq | 256 seq |
| tokens/step | 262,144 | 262,144 | 262,144 |
| `max_steps` | 41,225 | 41,225 | 41,225 |
| epochs over the corpus | 2.000 | 2.000 | 2.000 |
| tokens/parameter | 229 | 101 | 31 |
| `lr` / `muon_lr` | 3e-4 / 0.015 | 3e-4 / 0.015 | 3e-4 / 0.015 |
| warmup, schedule | 3000, WSD | 3000, WSD | 3000, WSD |

`max_steps` is identical because it is a property of the *data*, not the model: two epochs
of 5,402,429,161 tokens at 262,144 tokens per step. Note that the earlier 47M and 100M
configs used `muon_lr` 0.02; these use the 349M's 0.015 instead. Hyper-parameter parity
across the ladder matters more than each rung's own best setting — an unmatched optimizer
is the first thing a reviewer will attack, and it would make the comparison unusable.

**Micro-batch is the only thing tuned per model, and only for VRAM.** The product
`batch_size * grad_accum` must stay 256: that invariant preserves both the epoch arithmetic
above and the learning-rate setting. Sizing extrapolates from the one measurement we have
(349M at micro-batch 16 peaked at 25.2 GB), splitting it into a static part that scales with
parameters and an activation part that scales with `n_layer * n_embd * batch`:

| | `n_layer * n_embd` | micro-batch | est. peak VRAM |
|---|---|---|---|
| 47M | 6,912 | 32 | ~10 GB |
| 107M | 12,288 | 16 | ~10 GB |
| 349M | 27,648 | 16 | 25.2 GB (measured) |

Both leave headroom on a 24 GB card deliberately. The failure mode is not a clean
out-of-memory at startup but an OOM minutes in, during `torch.compile`'s *backward*
compilation, when inductor materialises fp32 intermediates — which is what happened to the
349M at micro-batch 16 on a 24 GB 4090. Estimates are extrapolations; treat the first
`VRAM:` line in the run's output as the real number and raise the micro-batch afterwards if
there is room.

**Picking the GPU.** VRAM is not what makes these runs cheap or expensive — throughput per
dollar is. A card at 60% of the hourly price and 40% of the throughput costs more per run,
not less. Availability in EU-RO-1 (forced by the volume's region) is narrow, so check
`--gpus` for what is actually schedulable, then measure: the metrics CSV records elapsed
time from the first steps, so tokens/s per dollar is known within minutes and the run can be
moved before it has cost anything. The 349M's 60.7k tok/s at $0.99/hr on the RTX 5090 is the
reference point to beat.
