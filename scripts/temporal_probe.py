"""Measure whether the models are temporally bounded, instead of asserting it.

The corpus is audited for post-1918 leakage from both sides, but that licenses a claim
about the *training data*. The claim the title makes is about the *models*: that a model
trained only on pre-1918 text has no posterior knowledge to suppress. Nothing in the
project has tested it. Auditing the corpus to three decimal places and then trusting
construction for the model is the same evidentiary asymmetry the corpus audit exists to
refuse.

Four measurements, all inference-only, wieszcz against a modern Polish LM:

1. *Anachronism rate in generations.* The battery from `anachronism_audit.py` is reused
   verbatim --- the instrument that decided what leaves the corpus decides what leaves the
   model, so the two numbers are commensurable. Its control battery of period-legitimate
   near-anachronisms (telefon, automobil, aeroplan) carries over as the false-positive
   floor: a genuine period model *should* hit those and not the modern ones.

2. *Post-reform orthography share.* Already calibrated on the corpus: confirmed modern
   editions score 0.88-1.00 on -cja against period -cya, period print 0.00-0.02. Applied
   to generations it is a continuous period-fidelity metric. The comparator is given two
   chances at it, since a base model has no single "prompt" to speak of: a declarative
   frame that describes the register, and held-out period passages to continue, which is
   the strongest conditioning available short of training on them.

3. *Era contrast in bits per byte.* The measurement that survives the size difference.
   Both models score a set of post-1918 terms and a set of period terms in identical
   hand-written carriers, and what is compared is not a level but an interaction: each
   model's (modern - period) gap, differenced across models. A crossover --- wieszcz
   surprised by modern vocabulary while the modern model is surprised by period
   vocabulary --- cannot be explained by one model simply being smaller or worse, which a
   raw perplexity comparison could not rule out. Bits per byte rather than per token
   because the tokenizers differ (8k byte-level BPE against Qwen2.5's ~150k inherited by
   Bielik); bytes are the only unit both agree on.

4. *Counterfactual futures.* Prompts whose natural completion is a post-1918 fact, scored
   on the gold modern continuation and also generated freely. This is the "silences" claim
   made checkable: what a model says when the honest answer lies past its cutoff.

The modern side is a *base* model on purpose. Our models only continue text, so an
instruction-tuned comparator would measure alignment artefacts rather than what a modern
Polish LM does with period prose. Bielik-1.5B-v3 is itself initialised from Qwen2.5-1.5B
and continued on 292B Polish tokens, which is worth stating wherever this baseline is
quoted: even the modern Polish line is adaptation, not from-scratch training.

    python scripts/temporal_probe.py \
        --wieszcz checkpoints/wieszcz_349m_6b7_2026-08-07_s1337/final.pt \
        --hf-model ~/models/bielik-1.5b-v3 \
        --out metrics/temporal_probe_2026-08-17.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch
from tokenizers import ByteLevelBPETokenizer

from anachronism_audit import MODERN, MODERN_RE, ORTO_MODERN, ORTO_PERIOD, PERIOD_CONTROL
from sample import load_model
from train import get_device

LN2 = math.log(2)
EOT = "<|endoftext|>"


def _disable_triton_bmm_override() -> bool:
    """Route aten::bmm back to its reference kernel.

    PyTorch overrides the outer-product case of `bmm` --- which is how rotary embeddings
    build their frequency table --- with a Triton kernel compiled on first use. A host
    without a C toolchain gets an exception rather than a fallback. Removing the override
    selects the unfused reference implementation, which costs nothing at this workload size
    and makes the scores more exact rather than less; applying it unconditionally also
    keeps both models on the same kernel regardless of the host.
    """
    try:
        from torch._native.registry import deregister_op_overrides
    except ImportError:
        return False
    deregister_op_overrides(disable_op_symbols="bmm")
    return True


TRITON_BMM_OVERRIDE_DISABLED = _disable_triton_bmm_override()

# Neutral period-register openings; no group, nation, faith or politics named, so what
# comes out is the model's default behaviour rather than a topic the prompt supplied.
# Shared with bias_prevalence.py by intent: the same generations answer both questions.
PROMPTS = [
    "Doniesiono nam z Warszawy, iż",
    "Wczoraj po południu na rynku",
    "Był to wieczór jesienny, gdy",
    "Sprawozdanie z posiedzenia rady miejskiej",
    "W ostatnich dniach pogoda",
    "Kochany Bracie! Piszę do Ciebie z",
    "Nowo otwarty zakład przy ulicy",
    "Pociąg pośpieszny odchodzi ze stacyi",
]

# A base model has no system prompt and no instruction-following to invoke, so the two
# conditionings below are what it can actually be given. This one is declarative rather
# than imperative -- a document header the model continues as text, not an order it obeys
# -- and it is the weaker of the two: it describes the register instead of exhibiting it.
IMITATION_PREAMBLE = (
    "Poniżej znajduje się fragment polskiej gazety z roku 1905, napisany ówczesną "
    "polszczyzną, w ortografii sprzed reformy 1918 roku.\n\n"
)

# The strong form: show the register instead of naming it. Exemplars are held-out period
# passages placed before the prompt, which is how a base model is natively asked for a
# style, and the most the comparator can be given without training it. Reporting only the
# declarative arm would leave the obvious objection open -- that the comparator was told
# about pre-reform spelling rather than shown any.
FEWSHOT_CHARS = 1400
# Exemplars must themselves carry the spelling: a passage at the corpus average would ask
# the comparator to imitate the very thing being measured, and one with no -cja/-cya
# evidence at all demonstrates nothing about orthography either way.
FEWSHOT_MAX_ORTO_MODERN = 0.10
FEWSHOT_MIN_ORTO_PERIOD = 3

# Carriers end without a space and targets begin with one, so the boundary falls where
# byte-level BPE puts it anyway and the scored span is exactly the target. Case agreement
# is imperfect in a few frames; it is identical for both models, so it cancels in the
# interaction and only inflates both levels equally.
MODERN_TERMS = [
    ("internet", "Wiadomość rozeszła się błyskawicznie przez"),
    ("komputer", "Na biurku stał nowy"),
    ("telewizja", "Program nadaje wieczorami"),
    ("telefon komórkowy", "Z kieszeni wyjął"),
    ("Hitler", "Kanclerzem Rzeszy został wówczas"),
    ("gestapo", "Aresztowaniami kierowało"),
    ("NKWD", "Śledztwo prowadziło"),
    ("kołchoz", "Ziemię gromadzką zamieniono w"),
    ("faszyzm", "Nowym prądem politycznym stał się"),
    ("druga wojna światowa", "W roku tym wybuchła"),
    ("smartfon", "Na stoliku leżał"),
    ("mikrofilm", "Zbiory przeniesiono na"),
    ("copyright", "Na odwrocie karty tytułowej widniał"),
    ("okupacja hitlerowska", "Nastała wówczas"),
]

# Period terms the audit either confirmed as epoch-legitimate or left untouched. Three of
# them (telefon, automobil, aeroplan) are the audit's own control battery, which keeps
# this set anchored to the corpus measurement rather than to intuition.
PERIOD_TERMS = [
    ("telegraf", "Wiadomość nadeszła wczoraj przez"),
    ("dorożka", "Pod bramą kamienicy czekała"),
    ("gubernia", "Utworzono wówczas nową"),
    ("rubel", "Walutą obiegową był wówczas"),
    ("kopiejka", "Najmniejszą monetą była"),
    ("jenerał", "Na czele oddziału stał stary"),
    ("automobil", "Przed gmachem zatrzymał się"),
    ("aeroplan", "Nad miastem przeleciał"),
    ("telefon", "W kantorze zainstalowano nowy"),
    ("cesarstwo", "Rozciągało się nad nimi potężne"),
    ("gimnazyum", "Chłopca oddano do"),
    ("folwark", "Za wsią leżał rozległy"),
    ("karczma", "Przy trakcie stała stara"),
    ("guwernantka", "Dzieci uczyła francuska"),
]

# Openings whose natural completion is a fact from after the cutoff. None of them touches
# the traps the corpus audit found (stalin as a 1914 surname, czołg 1918, "rzeczpospolita
# ludowa" as 1905-07 agitation) --- every gold here is unambiguously post-1918.
FUTURES = [
    ("W roku 1939 wybuchła", " druga wojna światowa"),
    ("Pierwszym prezydentem odrodzonej Rzeczypospolitej został", " Gabriel Narutowicz"),
    ("Człowiek po raz pierwszy stanął na Księżycu w roku", " 1969"),
    ("Po drugiej wojnie światowej Polska znalazła się w strefie wpływów",
     " Związku Radzieckiego"),
    ("Wiadomości czyta się dziś najczęściej przez", " internet"),
    ("Mur berliński runął w roku", " 1989"),
    ("Polska wstąpiła do Unii Europejskiej w roku", " 2004"),
    ("Najcięższe walki powstania warszawskiego toczyły się w roku", " 1944"),
]


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def battery_scan(text: str) -> dict:
    """Marker counts under the corpus audit's battery, plus the orthography share.

    Returns raw counts rather than a verdict: the report keeps everything and the paper
    decides what to quote.
    """
    low = text.lower()
    modern = {t: low.count(t) for t in MODERN if low.count(t)}
    for name, rx in MODERN_RE.items():
        n = len(rx.findall(low))
        if n:
            modern[name] = n
    control = {t: low.count(t) for t in PERIOD_CONTROL if low.count(t)}
    om = len(ORTO_MODERN.findall(text))
    op = len(ORTO_PERIOD.findall(text))
    return {
        "modern_hits": modern,
        "modern_total": sum(modern.values()),
        "control_hits": control,
        "control_total": sum(control.values()),
        "orto_modern": om,
        "orto_period": op,
        "orto_modern_share": round(om / (om + op), 4) if (om + op) else None,
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
    }


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson interval, because the counts here are small and the normal approximation
    puts the lower bound below zero exactly where it matters."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 5), round(min(1.0, centre + half), 5)]


class WieszczBackend:
    """The project's own decoder, loaded exactly as the sampling and eval scripts load it."""

    def __init__(self, ckpt: str, device: str, dtype: torch.dtype):
        self.name = ckpt
        self.device = device
        self.dtype = dtype
        self.tok = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                         str(REPO / "tokenizer/merges.txt"))
        self.eot = self.tok.token_to_id(EOT)
        self.model = load_model(str(REPO / ckpt) if not Path(ckpt).is_absolute() else ckpt,
                                device).to(dtype=dtype)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.block = self.model.cfg["block_size"]

    def encode_with_offsets(self, text: str):
        enc = self.tok.encode(text)
        return enc.ids, enc.offsets

    @torch.no_grad()
    def score_span(self, carrier: str, target: str) -> dict:
        ids, offsets = self.encode_with_offsets(carrier + target)
        return _score_common(self.model, ids, offsets, len(carrier), target,
                             self.device, self.block, hf=False)

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int, temperature: float, top_p: float,
                 seed: int) -> str:
        torch.manual_seed(seed)
        ids = self.tok.encode(prompt).ids or [self.eot]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new, temperature=temperature, top_k=0,
                                  top_p=top_p, eot_id=self.eot)
        full = self.tok.decode(out[0].tolist())
        return full[len(prompt):] if full.startswith(prompt) else full


class HFBackend:
    """A transformers causal LM, used in full precision by default.

    Quantised weights were the obvious convenience here and are the wrong instrument:
    quantisation error concentrates in the low-probability tail, which is exactly where
    period orthography lives, and the whole point of measurement 3 is that tail.
    """

    def __init__(self, path: str, device: str, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = path
        self.device = device
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        # Eager attention on purpose: the optimised paths JIT-compile Triton kernels, which
        # needs a C toolchain the measurement host does not have, and eager is exact rather
        # than merely equivalent-up-to-fused-numerics. The workloads here are far too small
        # for the speed difference to matter.
        kw = {"low_cpu_mem_usage": True, "attn_implementation": "eager"}
        # `torch_dtype` was renamed to `dtype`; accept whichever this transformers takes.
        try:
            m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
        self.model = m.to(device).eval()
        self.attn_implementation = getattr(m.config, "_attn_implementation", "eager")
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.block = getattr(self.model.config, "max_position_embeddings", 4096)

    def encode_with_offsets(self, text: str):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    @torch.no_grad()
    def score_span(self, carrier: str, target: str) -> dict:
        ids, offsets = self.encode_with_offsets(carrier + target)
        return _score_common(self.model, ids, offsets, len(carrier), target,
                             self.device, self.block, hf=True)

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int, temperature: float, top_p: float,
                 seed: int) -> str:
        torch.manual_seed(seed)
        enc = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                  temperature=temperature, top_p=top_p,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)


def _score_common(model, ids, offsets, split_char: int, target: str, device: str,
                  block: int, hf: bool) -> dict:
    """Summed NLL of the target span under teacher forcing, in nats.

    The span is located by character offsets rather than by tokenizing the two halves
    separately, so a merge that crosses the boundary is detected instead of silently
    shifting the span. `straddle` records that case for audit; with carriers ending
    before the space it should never fire.
    """
    first = next((i for i, (s, e) in enumerate(offsets) if e > split_char), None)
    if first is None or first == 0:
        return {"error": "target span not locatable", "n_tokens": 0}
    straddle = offsets[first][0] < split_char
    n = min(len(ids), block)
    if first >= n:
        return {"error": "target span past the context window", "n_tokens": 0}
    ids_t = torch.tensor([ids[:n]], dtype=torch.long, device=device)
    out = model(ids_t)
    logits = out.logits if hf else out[0]
    # Predicting token i uses the logits at i-1, so the span shifts left by one.
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    tgt = torch.tensor(ids[first:n], dtype=torch.long, device=device)
    nll = -lp[first - 1:n - 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    total = float(nll.sum())
    tbytes = len(target.encode("utf-8"))
    return {
        "n_tokens": int(tgt.numel()),
        "target_bytes": tbytes,
        "nll_nats": round(total, 5),
        "nats_per_token": round(total / max(int(tgt.numel()), 1), 5),
        "bits_per_byte": round(total / LN2 / tbytes, 5),
        "straddle": bool(straddle),
    }


def mean_ci(xs: list[float]) -> dict:
    """Mean with a 95% interval; a single value has no spread to report."""
    if not xs:
        return {"n": 0}
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return {"n": 1, "mean": round(m, 5)}
    sem = statistics.stdev(xs) / math.sqrt(len(xs))
    return {"n": len(xs), "mean": round(m, 5), "sem": round(sem, 5),
            "ci95": [round(m - 1.96 * sem, 5), round(m + 1.96 * sem, 5)]}


VOWELS = set("aąeęioóuyAĄEĘIOÓUY")
# Long s read as f. The substitution marks eighteenth- and early-nineteenth-century
# typography, which is a century away from the frame the declarative arm describes, and
# native Polish almost never puts f before these consonants.
LONG_S = re.compile(r"(?i)f[tpckwzn]")


def reads_as_prose(piece: str) -> bool:
    """Reject scanner debris, so the demonstration arm shows prose rather than noise.

    Pre-reform orthography in this corpus lives almost entirely in the scanned press --- of
    the twenty clean transcriptions on the held-out side, none carries it, because literary
    transcription modernises spelling --- so exemplars must come from OCR'd text and
    therefore have to be screened rather than assumed clean. Column rules, running heads and
    stray marks survive cleaning as short vowel-less tokens; a passage carrying many of them
    would ask the comparator to imitate the scanner instead of the century.
    """
    words = [w for w in piece.split() if any(c.isalpha() for c in w)]
    if len(words) < 30:
        return False
    novowel = sum(1 for w in words if not (set(w) & VOWELS))
    long_s = sum(1 for w in words if LONG_S.search(w))
    letters = sum(1 for c in piece if c.isalpha() or c.isspace())
    return (letters / len(piece) >= 0.90
            and sum(len(w) for w in words) / len(words) >= 4.0
            and novowel / len(words) <= 0.08
            and long_s / len(words) <= 0.03)


def build_fewshot_passages(clean_dir: Path, split_path: Path, budget_chars: int,
                           seed: int, max_orto_modern: float,
                           min_orto_period: int) -> list[dict]:
    """Held-out passages that demonstrably carry pre-reform spelling.

    Selection is on the exemplars' own orthography rather than on chance. The corpus
    averages 0.33 post-reform, so a passage drawn at random would be asking the comparator
    to imitate the very thing the metric measures; a passage carrying no -cja/-cya evidence
    at all would demonstrate nothing about spelling in either direction. Documents come
    from the validation side, so no model has trained on them, and passages are cut at
    paragraph and sentence boundaries so the exemplar reads as prose rather than as a
    fragment starting mid-word.
    """
    import random

    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = list(split["val_ids"])
    rng = random.Random(seed)
    piece_chars = max(200, budget_chars // 3)
    passages: list[dict] = []
    total = 0
    tries = 0
    while total < budget_chars and tries < 4000:
        tries += 1
        doc = clean_dir / f"{rng.choice(ids)}.txt"
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        if len(text) < piece_chars + 200:
            continue
        start = rng.randrange(0, len(text) - piece_chars - 100)
        nl = text.find("\n", start)
        start = nl + 1 if 0 <= nl - start < 200 else start
        piece = text[start:start + piece_chars]
        stop = max(piece.rfind(". "), piece.rfind("? "), piece.rfind("! "))
        if stop > piece_chars // 2:
            piece = piece[:stop + 1]
        piece = piece.strip()
        if not piece:
            continue
        if not reads_as_prose(piece):
            continue
        scan = battery_scan(piece)
        if scan["orto_period"] < min_orto_period:
            continue
        if scan["orto_modern_share"] is None or scan["orto_modern_share"] > max_orto_modern:
            continue
        passages.append({"doc": doc.name, "text": piece, "scan": scan})
        total += len(piece)
    return passages


def fit_preamble(backend, passages: list[dict], prompts: list[str],
                 max_new: int) -> tuple[str, dict]:
    """Size the exemplars to the context this backend actually has.

    The comparators differ by an order of magnitude here -- 1,024 positions for a GPT-2
    against 8,192 for a Llama -- so one preamble length cannot serve both, and a preamble
    that overruns is silently truncated by the tokenizer at whichever end it prefers.
    Exemplars are dropped from the front, keeping the one adjacent to the prompt because
    that is the one conditioning most strongly, and what each model received is recorded
    rather than assumed.
    """
    longest_prompt = max(len(backend.encode_with_offsets(p)[0]) for p in prompts)
    budget = backend.block - max_new - longest_prompt - 8
    join = lambda ps: "\n\n".join(x["text"] for x in ps) + "\n\n"

    kept = list(passages)
    while len(kept) > 1 and len(backend.encode_with_offsets(join(kept))[0]) > budget:
        kept.pop(0)
    text = join(kept) if kept else ""
    ids = backend.encode_with_offsets(text)[0] if text else []
    while len(ids) > budget and len(text) > 32:
        cut = max(16, int(len(text) * (1 - budget / len(ids))))
        text = text[cut:]
        space = text.find(" ")
        if 0 <= space < 40:
            text = text[space + 1:]
        ids = backend.encode_with_offsets(text)[0]

    return text, {
        "passages_kept": len(kept),
        "passages_dropped": len(passages) - len(kept),
        "tokens": len(ids),
        "budget_tokens": budget,
        "chars": len(text),
        "scan": battery_scan(text) if text else None,
    }


def run_generations(backend, prompts: list[str], preamble: str, seeds: int,
                    max_new: int, temperature: float, top_p: float, label: str) -> dict:
    items = []
    for si in range(seeds):
        seed = 1337 + si
        for pi, prompt in enumerate(prompts):
            text = backend.generate(preamble + prompt, max_new, temperature, top_p, seed)
            items.append({"id": f"s{seed}_p{pi}", "seed": seed, "prompt": prompt,
                          "continuation": text, "scan": battery_scan(text)})
        print(f"  [{label}] seed {seed} done ({len(items)} generations)", flush=True)

    hit = sum(1 for it in items if it["scan"]["modern_total"])
    ctrl = sum(1 for it in items if it["scan"]["control_total"])
    tot_bytes = sum(it["scan"]["bytes"] for it in items)
    shares = [it["scan"]["orto_modern_share"] for it in items
              if it["scan"]["orto_modern_share"] is not None]
    return {
        "generations": len(items),
        "preamble": preamble,
        "total_bytes": tot_bytes,
        "docs_with_modern_marker": hit,
        "modern_marker_doc_rate": round(hit / len(items), 5) if items else None,
        "modern_marker_doc_rate_ci95": wilson(hit, len(items)),
        "modern_markers_per_100k_bytes":
            round(1e5 * sum(it["scan"]["modern_total"] for it in items) / tot_bytes, 3)
            if tot_bytes else None,
        "docs_with_control_marker": ctrl,
        "control_marker_doc_rate": round(ctrl / len(items), 5) if items else None,
        "orto_modern_share": mean_ci(shares),
        "items": items,
    }


def run_corpus_reference(clean_dir: Path, split_path: Path, n_chunks: int,
                         chunk_bytes: int, seed: int) -> dict:
    """The same battery over real held-out text, at the same volume as the generations.

    Without this arm the generation numbers float free. "The model scores 0.27 on
    post-reform orthography" is only a statement about period fidelity once we know what
    the corpus itself scores under the identical metric --- and the corpus is not at 0.02,
    because Kryński's reformed spelling is the norm of 1905-18 Congress-Poland press. The
    control battery needs the same treatment: period-legitimate near-anachronisms are too
    rare in 16k tokens of generation to establish a false-positive floor, so the floor has
    to be measured on the same volume of genuine text.

    Held-out documents rather than training ones, so the reference is the text the models
    are scored against and shares no material with what they memorised.
    """
    import random

    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = list(split["val_ids"])
    rng = random.Random(seed)
    chunks = []
    tries = 0
    while len(chunks) < n_chunks and tries < n_chunks * 20:
        tries += 1
        doc = clean_dir / f"{rng.choice(ids)}.txt"
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        raw = text.encode("utf-8")
        if len(raw) <= chunk_bytes:
            piece = text
        else:
            start = rng.randrange(0, len(raw) - chunk_bytes)
            piece = raw[start:start + chunk_bytes].decode("utf-8", errors="ignore")
        if piece.strip():
            chunks.append({"doc": doc.name, "text": piece, "scan": battery_scan(piece)})

    hit = sum(1 for c in chunks if c["scan"]["modern_total"])
    ctrl = sum(1 for c in chunks if c["scan"]["control_total"])
    tot = sum(c["scan"]["bytes"] for c in chunks)
    shares = [c["scan"]["orto_modern_share"] for c in chunks
              if c["scan"]["orto_modern_share"] is not None]
    return {
        "chunks": len(chunks),
        "chunk_bytes_target": chunk_bytes,
        "total_bytes": tot,
        "docs_with_modern_marker": hit,
        "modern_marker_doc_rate": round(hit / len(chunks), 5) if chunks else None,
        "modern_marker_doc_rate_ci95": wilson(hit, len(chunks)),
        "modern_markers_per_100k_bytes":
            round(1e5 * sum(c["scan"]["modern_total"] for c in chunks) / tot, 3)
            if tot else None,
        "docs_with_control_marker": ctrl,
        "control_marker_doc_rate": round(ctrl / len(chunks), 5) if chunks else None,
        "orto_modern_share": mean_ci(shares),
        "items": [{"doc": c["doc"], "scan": c["scan"]} for c in chunks],
    }


def run_contrast(backend) -> dict:
    """Bits per byte on both term sets, keeping every per-term reading for paired tests."""
    out = {}
    for era, terms in (("modern", MODERN_TERMS), ("period", PERIOD_TERMS)):
        rows = []
        for term, carrier in terms:
            r = backend.score_span(carrier, " " + term)
            r.update({"term": term, "carrier": carrier})
            rows.append(r)
        bpb = [r["bits_per_byte"] for r in rows if "bits_per_byte" in r]
        out[era] = {"terms": rows, "summary": mean_ci(bpb)}
    gap = out["modern"]["summary"].get("mean")
    base = out["period"]["summary"].get("mean")
    out["modern_minus_period_bpb"] = round(gap - base, 5) if None not in (gap, base) else None
    return out


def run_futures(backend, max_new: int, temperature: float, top_p: float,
                seed: int) -> dict:
    rows = []
    for prompt, gold in FUTURES:
        scored = backend.score_span(prompt, gold)
        scored.update({"prompt": prompt, "gold": gold,
                       "generated": backend.generate(prompt, max_new, temperature,
                                                     top_p, seed)})
        scored["generated_scan"] = battery_scan(scored["generated"])
        rows.append(scored)
    bpb = [r["bits_per_byte"] for r in rows if "bits_per_byte" in r]
    return {"items": rows, "summary": mean_ci(bpb)}


def interaction(w: dict, h: dict) -> dict:
    """The crossover, computed per term so the two models are paired before differencing.

    Each term contributes one difference (wieszcz - modern LM) in bits per byte; the
    statistic is the mean of those differences on modern terms minus the mean on period
    terms. Pairing first is what makes the interval honest: the two sets contain different
    words, but each word is scored by both models.
    """
    per_era = {}
    for era in ("modern", "period"):
        wm = {r["term"]: r.get("bits_per_byte") for r in w[era]["terms"]}
        hm = {r["term"]: r.get("bits_per_byte") for r in h[era]["terms"]}
        diffs = [{"term": t, "wieszcz_bpb": wm[t], "modern_lm_bpb": hm[t],
                  "delta_bpb": round(wm[t] - hm[t], 5)}
                 for t in wm if wm.get(t) is not None and hm.get(t) is not None]
        per_era[era] = {"terms": diffs,
                        "summary": mean_ci([d["delta_bpb"] for d in diffs])}
    dm = per_era["modern"]["summary"]
    dp = per_era["period"]["summary"]
    result = {"per_era": per_era}
    if "mean" in dm and "mean" in dp:
        diff = dm["mean"] - dp["mean"]
        sem = math.sqrt(dm.get("sem", 0) ** 2 + dp.get("sem", 0) ** 2)
        result["interaction_bpb"] = round(diff, 5)
        result["interaction_sem"] = round(sem, 5)
        result["interaction_ci95"] = [round(diff - 1.96 * sem, 5),
                                      round(diff + 1.96 * sem, 5)]
        result["reading"] = (
            "positive = wieszcz pays more for modern vocabulary than the modern LM does, "
            "relative to how the two price period vocabulary")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wieszcz",
                    default="checkpoints/wieszcz_349m_6b7_2026-08-07_s1337/final.pt")
    ap.add_argument("--hf-model", default=None,
                    help="path or hub id of the modern Polish base LM (omit to skip it)")
    ap.add_argument("--hf-dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="fp32 by default: the contrast lives in the low-probability tail")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--clean", default="data/clean",
                    help="cleaned .txt corpus, for the held-out reference arm")
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--no-fewshot", dest="fewshot", action="store_false",
                    help="skip the demonstration arm and report only the declarative one")
    ap.add_argument("--fewshot-chars", type=int, default=FEWSHOT_CHARS,
                    help="exemplar budget before per-backend context fitting")
    ap.add_argument("--fewshot-max-orto", type=float, default=FEWSHOT_MAX_ORTO_MODERN,
                    help="reject exemplars whose own post-reform share exceeds this")
    ap.add_argument("--out", default="metrics/temporal_probe.json")
    args = ap.parse_args()

    t0 = time.time()
    device = args.device or get_device()
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

    print(f"device {device}")
    print(f"loading wieszcz: {args.wieszcz}", flush=True)
    wieszcz = WieszczBackend(args.wieszcz, device, torch.float32)
    print(f"  {wieszcz.n_params/1e6:.1f}M params, block {wieszcz.block}")

    report = {
        "meta": {
            "script": "scripts/temporal_probe.py",
            "script_sha256": script_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "device": device,
            "wieszcz_ckpt": args.wieszcz,
            "wieszcz_params": wieszcz.n_params,
            "hf_model": args.hf_model,
            "hf_dtype": args.hf_dtype if args.hf_model else None,
            "seeds": args.seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed_base": 1337,
            "triton_bmm_override_disabled": TRITON_BMM_OVERRIDE_DISABLED,
        },
        "battery": {"modern": MODERN, "modern_re": list(MODERN_RE),
                    "period_control": PERIOD_CONTROL},
        "term_sets": {"modern": MODERN_TERMS, "period": PERIOD_TERMS},
        "futures": [{"prompt": p, "gold": g} for p, g in FUTURES],
    }

    print("\n[1/4] generations under neutral period prompts", flush=True)
    report["generations_wieszcz"] = run_generations(
        wieszcz, PROMPTS, "", args.seeds, args.max_new_tokens, args.temperature,
        args.top_p, "wieszcz")

    if args.fewshot:
        print("\n[1b/4] few-shot arm: exemplars of period prose", flush=True)
        passages = build_fewshot_passages(
            REPO / args.clean, REPO / args.split, args.fewshot_chars, args.seed,
            args.fewshot_max_orto, FEWSHOT_MIN_ORTO_PERIOD)
        report["fewshot"] = {
            "requested_chars": args.fewshot_chars,
            "max_orto_modern_share": args.fewshot_max_orto,
            "min_orto_period": FEWSHOT_MIN_ORTO_PERIOD,
            "passages": passages,
        }
        shares = [x["scan"]["orto_modern_share"] for x in passages]
        print(f"  {len(passages)} passages, "
              f"{sum(len(x['text']) for x in passages)} chars, "
              f"exemplar post-reform shares {shares}", flush=True)
        pre, fit = fit_preamble(wieszcz, passages, PROMPTS, args.max_new_tokens)
        report["fewshot"]["fit_wieszcz"] = fit
        print(f"  wieszcz: {fit['tokens']}/{fit['budget_tokens']} tokens, "
              f"{fit['passages_kept']} kept", flush=True)
        report["generations_wieszcz_fewshot"] = run_generations(
            wieszcz, PROMPTS, pre, args.seeds, args.max_new_tokens, args.temperature,
            args.top_p, "wieszcz-fewshot")

    print("\n[2/4] held-out corpus reference at matched volume", flush=True)
    gen = report["generations_wieszcz"]
    report["corpus_reference"] = run_corpus_reference(
        REPO / args.clean, REPO / args.split, gen["generations"],
        max(1, gen["total_bytes"] // max(gen["generations"], 1)), args.seed)

    print("\n[3/4] era contrast in bits per byte", flush=True)
    report["contrast_wieszcz"] = run_contrast(wieszcz)

    print("\n[4/4] counterfactual futures", flush=True)
    report["futures_wieszcz"] = run_futures(wieszcz, 60, args.temperature, args.top_p,
                                            args.seed)

    if args.hf_model:
        del wieszcz.model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"\nloading modern LM: {args.hf_model} ({args.hf_dtype})", flush=True)
        hf = HFBackend(args.hf_model, device, dtypes[args.hf_dtype])
        print(f"  {hf.n_params/1e6:.1f}M params")

        for mode, preamble in (("plain", ""), ("imitate", IMITATION_PREAMBLE)):
            print(f"\n[1/4] modern LM generations ({mode})", flush=True)
            report[f"generations_modern_lm_{mode}"] = run_generations(
                hf, PROMPTS, preamble, args.seeds, args.max_new_tokens,
                args.temperature, args.top_p, f"modern-{mode}")

        if args.fewshot:
            pre, fit = fit_preamble(hf, report["fewshot"]["passages"], PROMPTS,
                                    args.max_new_tokens)
            report["fewshot"]["fit_modern_lm"] = fit
            print(f"\n[1b/4] modern LM generations (few-shot: {fit['tokens']}"
                  f"/{fit['budget_tokens']} tokens, {fit['passages_kept']} kept)",
                  flush=True)
            report["generations_modern_lm_fewshot"] = run_generations(
                hf, PROMPTS, pre, args.seeds, args.max_new_tokens, args.temperature,
                args.top_p, "modern-fewshot")

        print("\n[3/4] modern LM era contrast", flush=True)
        report["contrast_modern_lm"] = run_contrast(hf)

        print("\n[4/4] modern LM futures", flush=True)
        report["futures_modern_lm"] = run_futures(hf, 60, args.temperature, args.top_p,
                                                  args.seed)

        report["era_interaction"] = interaction(report["contrast_wieszcz"],
                                                report["contrast_modern_lm"])
        report["meta"]["hf_params"] = hf.n_params
        report["meta"]["hf_attn_implementation"] = hf.attn_implementation

    report["meta"]["elapsed_seconds"] = round(time.time() - t0, 1)
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    g = report["generations_wieszcz"]
    print(f"\nwieszcz modern-marker docs : {g['docs_with_modern_marker']}"
          f"/{g['generations']}  ci95 {g['modern_marker_doc_rate_ci95']}")
    print(f"wieszcz control-marker docs: {g['docs_with_control_marker']}"
          f"/{g['generations']}  (period words: a floor, not a leak)")
    print(f"wieszcz orto modern share  : {g['orto_modern_share']}")
    if "generations_wieszcz_fewshot" in report:
        fs = report["generations_wieszcz_fewshot"]
        print(f"wieszcz few-shot markers   : {fs['docs_with_modern_marker']}"
              f"/{fs['generations']}  orto {fs['orto_modern_share'].get('mean')}")
    cr = report["corpus_reference"]
    print(f"corpus ref modern-marker   : {cr['docs_with_modern_marker']}/{cr['chunks']}"
          f"  control {cr['docs_with_control_marker']}/{cr['chunks']}"
          f"  orto {cr['orto_modern_share'].get('mean')}")
    print(f"wieszcz modern-period bpb  : "
          f"{report['contrast_wieszcz']['modern_minus_period_bpb']}")
    if args.hf_model:
        for mode in ("plain", "imitate", "fewshot"):
            m = report.get(f"generations_modern_lm_{mode}")
            if not m:
                continue
            print(f"modern LM ({mode:8s}) markers: {m['docs_with_modern_marker']}"
                  f"/{m['generations']}  orto {m['orto_modern_share'].get('mean')}")
        print(f"modern LM modern-period bpb: "
              f"{report['contrast_modern_lm']['modern_minus_period_bpb']}")
        ei = report["era_interaction"]
        print(f"\nERA INTERACTION            : {ei.get('interaction_bpb')} bits/byte "
              f"ci95 {ei.get('interaction_ci95')}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
