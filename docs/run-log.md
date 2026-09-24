# Run log: first model, end to end

A detailed record of the first training campaign: laptop validation → GPU training →
a decoding bug that produced garbage → working inference. Written down because most of
the useful findings were failures, and they are easy to forget once something works.

Date: 2026-07-20. Corpus at the time: 3222 files, 88.5M tokens.

---

## 1. Pipeline validation on an M1 Pro (MPS)

The laptop cannot train the real model in reasonable time, so it was used to prove the
chain works: data → tokenizer → training loop → validation → checkpoint.

### Smoke test: `wieszcz_debug.json`, 1.4M params, 200 steps

```
Corpus: 88,497,698 tokens (884,976 val) | device: mps | fp32 (no bf16/FA kernel)
Model: 1.4M params | GQA 4q/2kv | Muon+AdamW
step   0 | loss 9.0049      val 9.0153
step 190 | loss 7.1194      val (step 150) 7.2436
```

Loss fell, validation tracked it, no NaNs. That was the only question being asked.

### Stress test: `wieszcz_m1.json`, 5.0M params, 2000 steps, block 256

```
step    0 | loss 9.0289     val 9.0312
step  800 | loss 5.5515     val 5.4677
step 2000 | loss 5.05       val 5.0159
```

Validation sat *at or slightly below* training loss for the whole run, no overfitting,
as expected: 2000 × 16 × 256 = 8.2M tokens ≈ **9% of one epoch**.

**Observation: low hardware utilization is normal here.** CPU sat around 20% and the
machine felt idle. On Apple silicon the compute runs on the GPU via MPS, so the CPU only
feeds data; and a model this small is *latency-bound*, not throughput-bound, per-step
kernel-launch and synchronization overhead rivals the actual math. Neither CPU nor GPU
saturates, and that is fine. The number that matters is the loss curve, not the load
meter.

---

## 2. Right-sizing before the real run

Two config decisions driven by the corpus size (88.5M tokens):

**Vocabulary 16k → 8k.** At `n_embd=512`, a 16k vocab puts 8.2M parameters into the
(weight-tied) embedding table, about 40% of a 20M model, spent on a lookup table rather
than on transformer blocks. Halving it freed ~4M parameters for depth and width. The cost
is a worse tokens/word ratio, since byte-level BPE encodes Polish diacritics as byte
pairs. Measured after training the tokenizer: **2.405 tokens per word**.

**Steps 20000 → 5000.** At `batch 32 × grad_accum 4 × block 512` = 65,536 tokens per step,
20000 steps would have been ~15 epochs over a small corpus, a direct route to
memorization. 5000 steps ≈ **3.7 epochs**.

**A measurement lesson:** an early token count extrapolated from a 40-file sample gave
~105M tokens. The full tokenization gave **88.5M**. Small samples of a heterogeneous
corpus skew, the sample happened to over-represent heavily-tokenized OCR files.

---

## 3. Moving to the RTX 3080 (WSL): four environment failures

None of these were bugs in the project. All are worth writing down because each cost real
time.

**`venv` cannot be created on `/mnt/c`.** The project was first extracted onto the Windows
filesystem. `python3 -m venv` failed with `ensurepip` returning non-zero and
`[Errno 1] Operation not permitted: .../activate.fish`: the DrvFs mount does not support
the symlinks and permissions venv needs. Fix: keep the project on WSL's ext4 (`~/`). This
also matters for speed, since tokenizing thousands of files is I/O heavy.

**macOS `tar` poisoned the corpus.** The transfer archive was built on macOS, which emits
AppleDouble `._*` companion files. These match the `*.txt` glob but are **binary**, so
tokenization died with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3`. Fixed
twice: deleted locally (`find . -name '._*' -delete`) and in code, `train.py` and
`train_tokenizer.py` now skip dot-files when globbing the corpus.

**`torch.compile` could not find a usable `nvcc`.** TorchInductor shells out to `nvcc`;
WSL inherits the Windows `PATH`, so it resolved to a Windows binary Linux cannot execute:
`PermissionError: [Errno 13] Permission denied: 'nvcc'`. Two valid fixes: install
`nvidia-cuda-toolkit` in WSL (chosen), or set `"compile": false`.

**A workaround that was correctly rejected.** Setting
`torch._dynamo.config.suppress_errors = True` would have made the run continue by falling
back to eager. It was added, then removed: it swallows *every* compile error, so you would
believe the model is compiled while it silently runs slower. A loud failure plus
documentation beats a quiet degradation.

---

## 4. The real run: `wieszcz_20m.json`, ~16M params

`device: cuda | bf16 + FlashAttention`, 5000 steps.

| Step | train | val | notes |
|------|-------|-----|-------|
| 1000 | 4.0012 |, | ~0.75 epoch; perplexity ≈ 55 |
| 2500 | 3.6158 | 3.6174 | gap 0.002, indistinguishable |
| 4000 | 3.4427 | 3.5280 | gap 0.085; LR multiplier 0.189 |
| 4750 |, | 3.5130 | validation flattening |
| final | **3.395** | **3.513** | perplexity ≈ 30 / 33.5 |

**Reading the curve.** Validation kept falling the whole way, so the model never
overfitted outright. But the train/val gap opened from 0.002 to 0.118 **between roughly
2 and 3 epochs**, and validation flattened in the last 750 steps (3.528 → 3.513). Together
that says the model extracted nearly everything this corpus has to offer at this size.
**The bottleneck is data, not capacity or step count**: the conclusion that matters for
the next run. (True *at this corpus size*: 88.5M tokens ≈ 5.5 unique tok/param, well under
Chinchilla. The postscript below shows the corpus was really ~10× larger; once it grew to
838M the 47M run flipped to capacity-bound, see §4. Data-vs-capacity is a function of the
ratio, not a fixed verdict.)

**VRAM: ~6 GB of 10 GB.** Weights are trivial (16M params ≈ 32 MB in bf16); the memory
goes to activations at `batch 32 × block 512`, and `nvidia-smi` reports *reserved* memory,
which PyTorch's caching allocator holds above actual use. The ~4 GB of headroom is better
spent on a longer `block_size` (more context per sample) or a bigger micro-batch with less
gradient accumulation (same effective batch, fewer accumulation loops, faster wall clock)
than on a bigger model, see the data-bound conclusion above.

---

## 5. The KV-cache bug: good model, broken decoder

First sampling attempt from `final.pt`:

```
Rankiem, gdy sięosuuerÓ.azÓee zyyeuue.eÓU eue.eeazeeÓWuanyJALyarzzyyichzyuzy…
```

Pure character soup: impossible for a model at perplexity 30. So the fault had to be in
the inference path, not the weights.

**Root cause.** `F.scaled_dot_product_attention(..., is_causal=True)` aligns the causal
mask to the **upper left** when the query and key lengths differ (documented behaviour).
During cached decoding the query length is 1 while the key length is N, so the mask was a
1×N row with only position 0 unmasked: **every generated token attended solely to the
first token of the prompt.**

Training never hit this because a training forward always has equal query and key lengths,
where upper-left alignment *is* the correct causal mask. The bug lived exclusively on the
generation path, which is precisely why a small model with a healthy loss curve produced
garbage.

**Fix**: use the causal flag only when the lengths match; during incremental decode every
cached key is by construction in the past, so no mask is needed at all:

```python
causal = q.size(2) == k.size(2)
out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=...)
```

**Transferable lesson:** garbage output from a model whose training loss looks fine points
at the decoder, not the training. Suspect masking, positional offsets, and cache handling
first, the three things that only run at inference time.

---

## 6. Working inference: what the model actually learned

### Novelistic register: prompt `"Rankiem, gdy"`

> Rankiem, gdy wtem, tuż, tak zwariował, że się zdało, iż jest na pół oślepiona i że jej
> się patrzy.
>
>, Dobrze się stanie, panie bracie, rzekł kapitan Amaury, kiedyśmy się zaczęli oglądać.

Period markers throughout: `iż`, `rzekł`, `owa`, the agglutinated `kiedyśmy`, vocabulary
like *kareta, panicz, pastwisko*. Polish literary dialogue punctuation (em-dash openers,
`— …, rzekł`) was learned unprompted. French character names (*Amaury*, *Amy*) come from
the translated 19th-century novels that make up much of Wolne Lektury.

### Memoir register: prompt `"Nauka o "`

> …u którego **nie masz** już miejsca w domu, a potem w domu, gdzie tyle pięknych rzeczy
> kosztowałem… i dla tej wielkiej szkody osierocono, i w ten dzień **mię** naznaczono…
> **gdyśmy** się z nim obsiedli…

Markedly *more archaic* than the novelistic sample: `nie masz` meaning "there is none",
the older accusative `mię`, agglutinated `gdyśmy` / `gdzieśmy`, spelling like `Kaźmirza`.
These forms come from the Internet Archive texts, which preserve original pre-1936
orthography, Wolne Lektury is largely modernized and could not have produced them.

### Reference register: prompt `"W roku 1863"`

> Kapitan Kotkowski (Henryk), słynny w **Galicyi** w Polsce, syn kasztelana wileńskiego,
> **ur.** 1807 w Ja-nuszu…

A biographical-dictionary entry: abbreviations, dates, offices, `Galicyi` in the old
locative. This is encyclopedic reference prose, a register that exists in the corpus only
because of Internet Archive.

**Verdict on the corpus design.** Adding Internet Archive was the right call and is
visible in the output: three prompts produced three distinct registers, two of them
non-literary. A Wolne Lektury–only corpus would have yielded novels and verse and nothing
else, and the model would have had no factual, expository voice at all.

**What the model does *not* have** is meaning. The prose is locally grammatical and
stylistically convincing but semantically adrift ("To mój brat, co to człowiek nie jest,
to mój brat!"). At 16M parameters on complex historical prose that is the expected
outcome: form and register are learnable at this scale, a world model is not.

Also observed: on list-like material the model degenerates into repetition
(`na karcie 3-ej, na r. 1869, na rok 1868…`). Index and catalogue text is intrinsically
repetitive, and small models fall into loops on it.

---

## 7. Finding that drives the next iteration: OCR page geometry

Both Internet Archive samples show short lines and the artifact `Ja-\nnuszu`. The corpus
preserved the hard line breaks of scanned pages, so **the model learned column width
instead of sentences**, and the original de-hyphenation regex (`-\n(\w)`) missed cases.

Fixed in `prepare_data.py` with `unwrap_lines()`: paragraphs are joined into single lines,
blank lines are kept as paragraph separators, and a trailing hyphen joins without a space
(`Ja-` + `nuszu` → `Januszu`). Applied to `ia_` files only, Wolne Lektury line breaks are
meaningful verse and must not be unwrapped.

Re-apply offline, without re-downloading anything:

```bash
python src/prepare_data.py --reclean
rm -f data/clean/tokens.bin
```

## Postscript: the data ceiling was an illusion

The conclusion above ("the bottleneck is data") held only because of a bad query. Internet
Archive tags Polish as `pol`, not `Polish`; switching the query raised the available pool
from ~1,700 items to **~352,000** for 1800–1918. The first corpus was built from roughly
half a percent of what is reachable.

So the binding constraint is now download time and disk, not availability, and the
sizing advice inverts: with a corpus an order of magnitude larger, a model well above 16M
parameters becomes the right target rather than an overfitting risk.

## Open items for the next run

- Cross-source dedup, the same public-domain work can appear in both Wolne Lektury
  (clean, modernized) and Internet Archive (OCR, original spelling). This matters more now
  that the Internet Archive pull is much larger.
- Spend the VRAM headroom on `block_size` 512 → 1024 rather than on parameters.
- `batch 64 × grad_accum 2` instead of `32 × 4`: identical gradients, fewer accumulation
  loops, better GPU saturation.

---
---

# Run log: second model (47M), end to end

Date: 2026-07-21. The corpus grew ~10× after the `pol`-tag fix (see postscript above);
this run trains a model sized to match it. Second campaign is recorded in the same
detail as the first, because, again, most of the useful findings were failures, and
this time several of them were *mine*, in the diagnosis rather than the code.

Corpus: **6320 files, 838,072,196 tokens** (8.38M held out for validation, the 1% split).

## 1. Config: sized to the corpus, not the hardware

`configs/wieszcz_47m.json`: 12 layers, `n_embd` 576, GQA 9q/3kv, `block_size` 1024,
tied embeddings, `vocab_size` 8000. **47.1M parameters.**

- **Batch 8 × grad_accum 16 × block 1024 = 131,072 tokens/step.** Effective batch 128.
- **18,709 steps = 2.93 epochs = 52 tokens/param.** Chinchilla for this corpus is ~43M
  params / 18 tok-per-param; we deliberately overtrained a slightly larger model, the
  standard practice for models meant to be *used* rather than to win a fixed-compute
  benchmark. SmolLM3 sits at ~3,667 tok/param, so 52 is still very conservative.
- **WSD schedule** (`"schedule": "wsd"`, `decay_frac` 0.1): flat at peak LR through step
  16,839, then a linear decay to `min_lr` over the last 1,870 steps. Chosen over cosine
  specifically so the epoch count could be decided *after* measuring the real token count,
  and so the run could be extended without a restart.
- LR 3e-4 (AdamW), Muon 0.02, warmup 200, grad clip 1.0.

**Right-sizing the batch, measured, not estimated.** VRAM at step 10:

```
VRAM: 3.6 GB used / 3.8 GB reserved     (torch.cuda.max_memory_allocated)
4998 MiB of 10240 MiB                    (nvidia-smi, whole process)
```

The `docs/training.md` sizing table predicted **~6.5 GB and was ~2× too high.** Root
cause of the error: the table's anchor ("16M model used ~6 GB") came from **`nvidia-smi`**,
which counts the whole process, CUDA context, driver buffers, allocator reserve, while
the prediction was compared against **`max_memory_allocated()`**, which counts only
tensors. This run shows the gap directly: 5.0 GB (nvidia-smi) vs 3.6 GB (torch) = ~1.4 GB
of overhead that was double-counted into the forecast. **Lesson: never mix the two VRAM
numbers.** The practical consequence was benign here but would matter at 65M: the table
would have argued for a smaller batch or model than the card can actually hold.

**GPU was compute-bound, not memory-bound.** At 109,227 tokens/s the model FLOPs
utilization works out to **~67% MFU** (of the 3080's ~59.5 TFLOPS bf16-with-fp32-accumulate
peak), a good figure; large training runs often sit at 35-50%. So the 50% of unused VRAM
was not headroom worth spending: a bigger micro-batch buys little when the cores are
already the bottleneck. Half the memory pool empty ≠ half the card idle.

**`vocab_size` alignment, a benchmark rule that did not transfer.** The Smol playbook
recommends rounding vocab to a multiple of 128 for tensor-core tiling; nanoGPT famously
got ~25% from 50257 → 50304. Benchmarked on *this* card and shape (8×1024, C=576):
8000 → 8064 gave **2%, and only on the output matmul**, a small fraction of the step.
8000 already divides by 64 and by 125, so it tiles fine; 50257 was prime and huge relative
to its model. Kept 8000. **Lesson: a tuning rule from someone else's config is a
hypothesis, measurable in ten seconds, not a law.**

## 2. The environment saga: one real failure, four misdiagnoses

The tokenizer and training would not stay alive on the WSL box for most of an evening.
Exactly **one** cause was a resource limit; the rest I misdiagnosed, and the misdiagnoses
share a single root, a broken measurement.

**The one real resource failure: tokenizer OOM.** `train_tokenizer.py` holds the whole
word-frequency table in RAM. On the 2.6 GB corpus (13.2M distinct words) it climbed to
7.5 GB RSS and was OOM-killed, because WSL2 defaults to **8 GB or half of host RAM,
whichever is smaller**, and this box has 16 GB (confirmed with `wmic memorychip`, a
single 16 GB DDR5 stick in one channel; my earlier "16 GB" was a lucky guess from the
7 GB WSL figure, not a reading). Fix: `.wslconfig` with `memory=12GB` + `swap=8GB`.

**Then four "the process disappeared" failures that were NOT memory:**

1. **VM-restart race.** Launching seconds after `wsl --shutdown` started a job inside a VM
   that was already tearing down; a fresh VM booted moments later without the job. Hyper-V
   Worker log confirmed only one VM start, at a timestamp *after* the launch.
2. **User-session teardown.** With `Linger=no`, closing the last session collapses
   `user@1000.service` and everything in its cgroup, including tmux. The long crawl earlier
   had survived for hours only because a logged-in graphical session held the manager up.
3. **`vmIdleTimeout`.** With no user logged into Windows, WSL2 tears the distro down ~60s
   after the last `wsl.exe` client exits, and one-shot SSH commands are exactly that.
4. **A "full-corpus tokenizer OOM" that was not an OOM.** `oom_w_logu=0` with 10.5 GB free.

**Root cause of the false alarms: a broken instrument.** The monitor detected the process
with `ps -C python` / `pgrep -x python`, which require an **exact** name match. The venv
binary is `python3` (`.venv/bin/python → python3 → /usr/bin/python3`), so these commands
returned empty **on a live, healthy training run** and the monitor reported "no process."
Every subsequent hypothesis, lingering, `vmIdleTimeout`, memory, was built on that
false negative. Compounding it: `dmesg` is **per-boot** and was cleared on each VM
restart, so "no OOM in dmesg" was read off a kernel log that no longer existed.

**The signal that was there the whole time:** the same command run in the **foreground**
over SSH *worked* (the tokenizer finished in 3 minutes, exit 0), while **detached in
tmux** it died. That contrast pointed at session/distro lifetime from the start, not at
resources. I chased four resource theories before trusting it.

**What actually fixed it, and stuck:**
- `.wslconfig memory=12GB`: for the one genuine OOM.
- **`sudo loginctl enable-linger $USER`** (run by the operator), decouples processes from
  session lifetime.
- **User logged into Windows, screen locked with `Win+L` (never *logged out*)**, the
  scheduled task runs `Interactive only` and needs a session.
- **Launch via a Windows Scheduled Task**, not one-shot SSH. The task holds a `wsl.exe`
  client open for the job's whole lifetime, so the distro never collapses under it. The
  training process's parent chain is `python ← bash ← Relay(wsl.exe) ← systemd`, fully
  detached from any SSH connection: verified by cycling SSH sessions while it ran.

An orphaned `python3` from an earlier attempt (RSS 9.8 GB) was also found and killed. Two
tokenizers resident at once would never fit; it is possible the full corpus was never the
real constraint at all.

**Meta-lesson, the most transferable thing here:** *before treating a negative reading as
a failure, prove the instrument returns a positive on a known-good case.* Four hypotheses
were built on `ps` output that could not have said "alive" even when the process was.

## 3. Training dynamics: WSD, grad-norm logging, and data repetition

Two `train.py` changes earned their keep this run:

**Gradient-norm logging.** `clip_grad_norm_` already returns the pre-clip norm; we now log
it instead of discarding it. It answered the learning-rate question in 100 steps: the norm
fell 29.6 → 1.8 → 0.95 through warmup and **settled at ~0.10-0.12 for the entire run**,
far below the clip threshold of 1.0. Clipping never fired after step 150. That says 3e-4
is **conservative**: the model would tolerate a higher LR, likely nearer the 5e-4 the
playbook cites. A direct, cheap input to the 65M run, visible only because the metric was
logged.

**Resume support** (added mid-run, so this run's own checkpoints are weights-only). New
`save_ckpt` stores model + Muon + AdamW state + step; `--resume` reloads all three and
restarts the step counter. Optimizer state matters: Muon momentum and AdamW moments take
hundreds of steps to warm up, so resuming weights-only costs a visible loss bump, the
code detects the weights-only case and warns rather than pretending. Saved from the
**uncompiled** module, since `torch.compile` prefixes keys with `_orig_mod.`. Verified with
a full save→reload cycle plus a check that extending `max_steps` returns a WSD schedule to
its flat phase rather than a cosine discontinuity.

**Validation loss: and an empirical read on data repetition:**

| Step | Epoch | val loss | note |
|------|-------|----------|------|
| 500 | 0.08 | 4.159 | |
| 2000 | 0.31 | 3.581 | already below the 16M model's *final* 3.513 |
| 6000 | 0.94 | 3.360 | end of epoch 1 |
| 8000 | 1.25 | 3.355 | into epoch 2, still falling |
| 10000 | 1.56 | 3.321 | |
| 12500 | 1.96 | 3.283 | end of epoch 2 |
| 14000 | 2.19 | 3.233 | |
| 16000 | 2.50 | 3.262 | |
| 18000 | 2.82 | 3.232 | |
| 18500 | 2.89 | **3.212** | after WSD decay |

**Repetition helped, with diminishing returns.** Val loss kept falling *past the epoch-1
boundary* while train loss also fell, no memorization turn, an empirical confirmation of
Muennighoff et al. (repeats up to ~4 epochs are nearly as good as fresh tokens). But the
gains shrank each pass: epoch 1 did most of the work (4.16 → 3.36), epoch 3 only trimmed
~0.04. **A diagnosis I got wrong mid-run:** at step ~12,500 I read eight noisy values
(~3.29-3.30) and declared val loss had "hit a capacity ceiling." It had not: it was a
local plateau; the third epoch kept improving slowly to 3.212. Same error as the broken
instrument: a confident conclusion from too small a sample. The honest reading is
*sharply diminishing returns*, not a wall, which argues against a 4th epoch but not
against a bigger model.

**One inference not to overstate:** diminishing returns *across epochs* show that
**re-showing the same 838M tokens** is nearly exhausted as a lever. They do **not** show
that data is a non-issue, fresh, unseen tokens are a different axis, and this run never
tested it (no held-back corpus was added at fixed model size). The clean data-vs-capacity
signal in this campaign is the **16M → 47M ladder on identical data** (3.513 → 3.212): more
parameters, same tokens, lower loss ⇒ the smaller model was capacity-bound. That is the
result to lean on; the epoch curve only speaks to repetition.

**Speed.** 1.33 s/step sustained, 838M-token corpus tokenized to `tokens.bin` in ~16 min
(one-time), 18,709 steps in ~6.9 h (22:22 → 05:18). No spikes, no NaNs, grad_norm flat
throughout.

## 4. What the 47M model learned: the leap to meaning

Final val loss **3.212 vs 3.513** for the 16M model. The generated text shows this is a
qualitative jump, not just a lower number.

### Expository register: prompt `"Naród polski"`

> Naród polski, wyniszczając kraj przez to, że go nie było, wcielił się do organizmu
> narodu. […] Zdawało się, że naród, który nie…

Coherent clauses carrying an argument across several sentences, in the register of
19th-century political essay. The 16M model had authentic *form* but no semantics; this
one sustains a line of thought.

### Register is selected by the prompt

- `"W roku 1863"` → dry learned-society report: `Komitet centralny Towarzystwa
  geograficznego w Warszawie`, dates, page ranges. Encyclopedic/reference voice.
- `"Miłość jest"` → exalted reflective prose: *niewzruszoną… przemijającą… ufajmy Bogu…
  miłosierdzie*.
- `"Rankiem, gdy"` → first-person narrative with correct 19th-c dialogue punctuation.

Period orthography throughout: `teorye`, `w r. 1863-ym`, `gdym wyjechał`, em-dash dialogue.

### The 47M ceiling is visible, and healthy to see

- **Loops on long generation:** `i ja, i mój krewny, i ja, i mój krewny`; `nowe teorye i
  nowe teorye`; `centralny centralny centralny`. The classic small-model failure, it
  loses the thread after a few sentences.
- **OCR artifacts appear in generated text** (`bezgraniczn3rm`, `widzimj^`, `w^szystko`),
  though the vocabulary is not polluted. An earlier version of this log claimed the
  tokenizer "absorbed OCR noise as real tokens"; a direct measurement of `vocab.json`
  (`src/analyze_ocr.py`, and a token-decode audit) disproved it. `min_frequency=2` already
  excludes hapax OCR garbage, so `bezgraniczn3rm` is **not a token at all**: it is the model
  reassembling byte/subword fragments it learned from the ~3% noisy token stream. Real
  garbage in the 8k vocab is a handful of tokens (Google-Books watermark scraps like
  `VjOOQIC`), well under 1%. See the OCR-corruption audit in `docs/data-preparation.md`.

**Verdict.** The model crossed from "sounds like the era" to "says something from the
era." The looping and short coherence horizon are the classic small-model failure and point
at a capacity limit, consistent with the 16M → 47M gain on fixed data.

**Why capacity, not data, is the binding constraint *at this size*: stated precisely:**
- The 16M → 47M ladder on identical data is a clean capacity experiment: same tokens, more
  parameters, loss fell 3.513 → 3.212. The 16M was capacity-bound.
- 47M is already near Chinchilla-optimal on *unique* data: 838M unique tokens / 47M params
  ≈ **17.8 unique tok/param** vs Chinchilla's ~20. So the 47M is essentially fed, not
  starved, which is why more parameters, not more tokens, is the next lever *here*.
- **The caveat that keeps this honest:** the complementary experiment, hold 47M fixed, add
  fresh unique tokens beyond 838M: was never run. So "capacity is the *only* wall" is not
  proven; "capacity is the binding constraint at 47M, and re-showing existing data is spent"
  is what the evidence supports.
- **This flips at 65M.** Chinchilla-optimal for 65M is ~1.3B unique tokens; we have 838M, so
  65M would be mildly *under*-fed. The IA reservoir (~37B tokens reachable) can cover the
  gap, so the 65M run should plan to **enlarge the corpus**, not just repeat the current one,
  to avoid dropping below Chinchilla.

## Open items after the second run

- **65M is the next target**, with LR raised toward 5e-4 (grad-norm evidence) and the
  measured 3.6 GB VRAM base showing ample room. Re-tune batch upward only if MFU, not
  memory, says there is headroom. **Enlarge the corpus first:** 65M wants ~1.3B unique
  tokens for Chinchilla-optimum and we have 838M, so crawl more of the ~37B-token IA
  reservoir before the run rather than leaning harder on repetition.
- **OCR cleanup**: `bezgraniczn3rm`-class artifacts suggest a tighter post-OCR filter, or
  retraining the tokenizer with a higher `min_frequency` to keep garbage merges out of the
  vocabulary.
- **Cross-source dedup** still open from the first run.
- The 65M run should be **resumable from the start** (this run's checkpoints are
  weights-only because resume landed mid-run).
