"""Pretraining for a small decoder-only Transformer.

Architecture (Llama/Gemma-2 lineage):
  - RoPE, RMSNorm with pre- and post-norm around each sublayer, QK-Norm
  - SwiGLU, Grouped-Query Attention
  - FlashAttention via F.scaled_dot_product_attention
  - Scaled residual init, weight tying

Optimizer: Muon on 2D hidden weights, AdamW on embeddings and norm gains.

Not used at this scale: MoE (data-starved), sliding-window attention (any useful window
would exceed the context), logit soft-capping (disables the FlashAttention kernel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from metrics_logger import METRICS_DIR, MetricsLogger, run_name

CLEAN_DIR = Path("data/clean")
TOKENS_CACHE = Path("data/clean/tokens.bin")
TOKENIZER_DIR = Path("tokenizer")
CKPT_DIR = Path("checkpoints")


# ----------------------------- model components ------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    b, kvh, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, kvh, n_rep, t, hd).reshape(b, kvh * n_rep, t, hd)


class KVCache:
    """Per-layer key/value cache for autoregressive inference (not training)."""

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.n_kv_head = cfg["n_kv_head"]
        self.head_dim = cfg["n_embd"] // cfg["n_head"]
        self.dropout = cfg["dropout"]
        self.q_proj = nn.Linear(cfg["n_embd"], self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg["n_embd"], self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg["n_embd"], self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg["n_embd"], bias=False)
        self.q_norm = RMSNorm(self.head_dim)  # QK-Norm
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, cache: "KVCache | None" = None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(k, v)
        k = repeat_kv(k, self.n_head // self.n_kv_head)
        v = repeat_kv(v, self.n_head // self.n_kv_head)

        # is_causal aligns the mask upper-left when q and k lengths differ, so a single
        # cached-decode query would see only position 0. Decode needs no mask anyway.
        causal = q.size(2) == k.size(2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = int(8 / 3 * cfg["n_embd"])
        mult = cfg["multiple_of"]
        hidden = mult * ((hidden + mult - 1) // mult)
        self.w_gate = nn.Linear(cfg["n_embd"], hidden, bias=False)
        self.w_up = nn.Linear(cfg["n_embd"], hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg["n_embd"], bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Gemma-2 sandwich: pre-norm AND post-norm around each sublayer."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["n_embd"]
        self.pre_attn, self.post_attn = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(cfg)
        self.pre_mlp, self.post_mlp = RMSNorm(d), RMSNorm(d)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.post_attn(self.attn(self.pre_attn(x), cos, sin, cache))
        x = x + self.post_mlp(self.mlp(self.pre_mlp(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layer"]))
        self.norm_f = RMSNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        cos, sin = precompute_rope(cfg["n_embd"] // cfg["n_head"], cfg["block_size"], cfg["rope_theta"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Scale residual output projections by 1/sqrt(2*n_layer) (GPT-2/Llama init).
        scale = (2 * cfg["n_layer"]) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w_down.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        t = idx.size(1)
        cos = self.rope_cos[start_pos : start_pos + t].to(idx.device)
        sin = self.rope_sin[start_pos : start_pos + t].to(idx.device)
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.9, top_k=50, top_p=1.0, eot_id=None):
        """Yield generated token ids one at a time, KV-cached.

        top_k caps the candidate count, top_p the cumulative mass; they combine. A
        decayed model has a sharp distribution and loops under top_k alone.
        """
        self.eval()
        caches = [KVCache() for _ in self.blocks]
        logits, _ = self(idx, caches=caches, start_pos=0)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            if pos >= self.cfg["block_size"]:
                break  # RoPE table only covers block_size positions
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Shift the cut one right, so a peaked step still keeps its top token.
                drop = cum_probs > top_p
                drop[..., 1:] = drop[..., :-1].clone()
                drop[..., 0] = False
                logits = logits.masked_fill(drop.scatter(-1, sorted_idx, drop), float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tok_id = int(next_id.item())
            yield tok_id
            if eot_id is not None and tok_id == eot_id:
                break
            logits, _ = self(next_id, caches=caches, start_pos=pos)
            pos += 1

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=50, top_p=1.0, eot_id=None):
        """Batch wrapper over stream(). Returns prompt+continuation as one tensor."""
        new = list(self.stream(idx, max_new_tokens, temperature, top_k, top_p, eot_id))
        if new:
            idx = torch.cat([idx, torch.tensor([new], dtype=torch.long, device=idx.device)], dim=1)
        return idx


# ----------------------------- Muon optimizer --------------------------------

def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Orthogonalize a 2D matrix via a quintic Newton-Schulz iteration.

    Runs in bf16 on CUDA; falls back to fp32 elsewhere (MPS/CPU lack fast bf16).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    compute_dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32
    X = G.to(compute_dtype)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum orthogonalized by Newton-Schulz. For 2D hidden weights only."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])  # Nesterov
                g = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)


# ----------------------------- data ------------------------------------------

def build_token_cache() -> np.ndarray:
    if TOKENS_CACHE.exists():
        return np.memmap(TOKENS_CACHE, dtype=np.uint16, mode="r")
    tok = ByteLevelBPETokenizer(str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))
    eot = tok.token_to_id("<|endoftext|>")
    corpus = sorted(p for p in CLEAN_DIR.glob("*.txt") if not p.name.startswith("."))
    # Written incrementally: np.concatenate over every chunk peaks at ~2x the corpus.
    tmp = TOKENS_CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for path in tqdm(corpus, desc="tokenizing corpus"):
            ids = tok.encode(path.read_text(encoding="utf-8")).ids
            if eot is not None:
                ids.append(eot)
            np.asarray(ids, dtype=np.uint16).tofile(f)
    tmp.rename(TOKENS_CACHE)
    return np.memmap(TOKENS_CACHE, dtype=np.uint16, mode="r")


def file_sha256(path: Path) -> str | None:
    """Hash for the run manifest. A model paired with the wrong vocabulary of the same
    size loads without error and generates nonsense, so the pairing has to be recorded
    while it is still known."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_batch(data: np.ndarray, cfg: dict, device: str):
    ix = torch.randint(len(data) - cfg["block_size"] - 1, (cfg["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i : i + cfg["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg["block_size"]].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_mult(step: int, cfg: dict) -> float:
    """Learning-rate multiplier in [min_lr/lr, 1], with linear warmup.

    WSD (the default) holds the peak rate flat and decays over the final `decay_frac`.
    Cosine decays from the first step after warmup, which ties its shape to `max_steps`
    and makes a run impossible to extend.
    """
    warmup, total = cfg["warmup"], cfg["max_steps"]
    floor = cfg["min_lr"] / cfg["lr"]
    if step < warmup:
        return (step + 1) / warmup

    if cfg.get("schedule", "wsd") == "cosine":
        ratio = (step - warmup) / max(1, total - warmup)
        return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * ratio))

    decay_steps = max(1, int(total * cfg.get("decay_frac", 0.1)))
    decay_start = total - decay_steps
    if step < decay_start:
        return 1.0
    return floor + (1 - floor) * (1 - (step - decay_start) / decay_steps)


# ----------------------------- train -----------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, data, cfg, device, iters=20):
    """Fixed, evenly spaced windows: every eval scores the same text, so the val curve
    is comparable across steps and runs, and evaluation consumes no training RNG."""
    model.eval()
    block, bs = cfg["block_size"], cfg["batch_size"]
    starts = np.linspace(0, len(data) - block - 1, iters * bs).astype(np.int64)
    losses = []
    for it in range(iters):
        ix = starts[it * bs:(it + 1) * bs]
        x = torch.stack([torch.from_numpy(data[i: i + block].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block].astype(np.int64)) for i in ix]).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_optimizers(model: GPT, cfg: dict):
    """Muon on 2D hidden weights; AdamW on embeddings and norm gains."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok_emb" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=cfg["muon_lr"], momentum=0.95)
    adamw = torch.optim.AdamW(adamw_params, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)
    return muon, adamw


def prune_ckpts(keep_last: int | None, keep_every: int | None) -> None:
    """Keep the most recent `keep_last` checkpoints, plus every `keep_every`-th step."""
    if not keep_last:
        return
    saved = []
    for p in CKPT_DIR.glob("step*.pt"):
        try:
            saved.append((int(p.stem.removeprefix("step")), p))
        except ValueError:
            continue
    saved.sort()
    for s, p in saved[:-keep_last]:
        if keep_every and s % keep_every == 0:
            continue
        p.unlink(missing_ok=True)


def save_ckpt(path: Path, raw_model, muon, adamw, step: int, cfg: dict) -> None:
    """Write a resumable checkpoint: weights, both optimizer states, and the step.

    Optimizer state matters because Muon's momentum takes hundreds of steps to warm
    back up; the step matters because the LR schedule is a function of it.

    Saved from the uncompiled module, since torch.compile prefixes every key with
    `_orig_mod.`. Written to `.tmp` and renamed, so a kill mid-write leaves the previous
    checkpoint intact and the partial never matches the `step*.pt` glob.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": raw_model.state_dict(),
            "muon": muon.state_dict(),
            "adamw": adamw.state_dict(),
            "step": step,
            "cfg": cfg,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)},
        },
        tmp,
    )
    tmp.replace(path)
    print(f"           ckpt {path.name} @ step {step} "
          f"sha256 {file_sha256(path)[:16]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="checkpoint to continue from")
    parser.add_argument("--seed", type=int, default=None, help="override cfg['seed']")
    parser.add_argument("--data", help="training token stream (uint16 .bin)")
    parser.add_argument("--val-data", help="validation token stream (uint16 .bin); "
                                           "with --data, nothing is sliced off the "
                                           "training stream")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override cfg['max_steps'] (the decay-branch resume "
                             "runs a shorter schedule than the main pass)")
    parser.add_argument("--run-suffix", default="",
                        help="appended to the run name; a decay branch gets its own "
                             "checkpoint dir and CSV lineage instead of appending "
                             "to the main run's")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    device = get_device()
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    torch.manual_seed(seed)
    if args.data:
        train_data = np.memmap(args.data, dtype=np.uint16, mode="r")
        if not args.val_data:
            raise SystemExit("--data requires --val-data: the held-out stream "
                             "is a separate file, not a slice")
        val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")
        data = train_data
    else:
        data = build_token_cache()
        n_val = max(cfg["block_size"] + 1, int(len(data) * 0.01))
        train_data, val_data = data[:-n_val], data[-n_val:]
    precision = "bf16 + FlashAttention" if device == "cuda" else "fp32 (no bf16/FA kernel)"
    print(f"Corpus: {len(train_data):,} train tokens ({len(val_data):,} val) | "
          f"device: {device} | {precision}")

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | GQA {cfg['n_head']}q/{cfg['n_kv_head']}kv | Muon+AdamW")

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)

    raw_model = model  # uncompiled, for checkpointing
    if cfg.get("compile") and device == "cuda":
        model = torch.compile(model)

    muon, adamw = build_optimizers(model, cfg)
    grad_accum = cfg.get("grad_accum", 1)

    start_step = 0
    if ckpt is not None:
        start_step = ckpt.get("step", 0) + 1
        if "muon" in ckpt and "adamw" in ckpt:
            muon.load_state_dict(ckpt["muon"])
            adamw.load_state_dict(ckpt["adamw"])
            extra = "with optimizer state"
            if "rng" in ckpt:
                torch.set_rng_state(ckpt["rng"]["torch"].cpu())
                if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [s.cpu() for s in ckpt["rng"]["cuda"]])
                extra += " and RNG state"
        else:
            extra = "WEIGHTS ONLY - optimizer momentum restarts, expect a brief loss bump"
        print(f"Resuming from {args.resume} at step {start_step:,} ({extra})")
        if start_step >= cfg["max_steps"]:
            raise SystemExit(
                f"Checkpoint is at step {start_step:,} but max_steps is "
                f"{cfg['max_steps']:,}. Raise max_steps to extend the run."
            )

    global CKPT_DIR
    run = run_name(args.config, bool(args.resume), seed) + args.run_suffix
    CKPT_DIR = CKPT_DIR / run          # per-run, so concurrent runs never clobber each other
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tokens_per_step = cfg["batch_size"] * grad_accum * cfg["block_size"]
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    ckpt_every_epoch = bool(cfg.get("ckpt_every_epoch"))
    train_log = MetricsLogger(METRICS_DIR / f"{run}_train.csv",
                              ["step", "tokens", "elapsed_s", "unix_ts", "loss",
                               "grad_norm", "lr_mult", "tok_s"])
    val_log = MetricsLogger(METRICS_DIR / f"{run}_val.csv",
                            ["step", "tokens", "elapsed_s", "unix_ts", "val_loss"])
    # Provenance that cannot be reconstructed afterwards, including data fingerprints.
    data_path = Path(args.data) if args.data else TOKENS_CACHE
    (METRICS_DIR / f"{run}_manifest.json").write_text(json.dumps({
        "run": run, "seed": seed, "config": args.config, "config_values": cfg,
        "n_params_millions": round(n_params, 3),
        "train_tokens": int(len(train_data)), "val_tokens": int(len(val_data)),
        "data": {"train": str(data_path), "train_sha256": file_sha256(data_path),
                 "val": args.val_data,
                 "val_sha256": file_sha256(Path(args.val_data)) if args.val_data else None},
        "tokenizer": {f: file_sha256(TOKENIZER_DIR / f) for f in ("vocab.json", "merges.txt")},
        "train_py_sha256": file_sha256(Path(__file__)),
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": device,
                "device_name": (torch.cuda.get_device_name(0)
                                if device == "cuda" else platform.processor())},
        "resume": args.resume, "started_unix": time.time(),
    }, indent=2))
    t0 = time.time()
    last_log_step, last_log_t = start_step, t0
    # Branch point for another epoch: a post-decay checkpoint is annealed into a sharp
    # minimum and resumes badly. Fixed name, so rotation never reclaims it.
    decay_start = None
    if cfg.get("schedule", "wsd") != "cosine":
        decay_steps = max(1, int(cfg["max_steps"] * cfg.get("decay_frac", 0.1)))
        decay_start = cfg["max_steps"] - decay_steps
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        mult = lr_mult(step, cfg)
        for opt, base in ((muon, cfg["muon_lr"]), (adamw, cfg["lr"])):
            for group in opt.param_groups:
                group["lr"] = base * mult

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, cfg, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_val += loss.item() / grad_accum
        # Logged pre-clip: a norm riding at the threshold means every step is truncated.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()

        if step % cfg["log_every"] == 0:
            now = time.time()
            tok_s = (step - last_log_step) * tokens_per_step / max(1e-9, now - last_log_t)
            last_log_step, last_log_t = step, now
            print(f"step {step:>6} | loss {loss_val:.4f} | grad_norm {grad_norm:.3f}"
                  f" | lr x{mult:.3f} | {tok_s/1e3:,.0f}k tok/s")
            train_log.log(step=step, tokens=step * tokens_per_step, elapsed_s=f"{now - t0:.1f}",
                          unix_ts=f"{now:.1f}", loss=f"{loss_val:.4f}",
                          grad_norm=f"{grad_norm:.3f}", lr_mult=f"{mult:.4f}",
                          tok_s=f"{tok_s:.0f}")
        if step == start_step + 10 and device == "cuda":   # once memory has stabilized
            print(f"           VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB used"
                  f" / {torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved")
        if step % cfg.get("eval_every", 250) == 0:
            vl = estimate_loss(model, val_data, cfg, device, cfg.get("eval_iters", 20))
            print(f"           val loss {vl:.4f}")
            val_log.log(step=step, tokens=step * tokens_per_step, elapsed_s=f"{time.time() - t0:.1f}",
                        unix_ts=f"{time.time():.1f}", val_loss=f"{vl:.4f}")
        if decay_start is not None and step == decay_start:
            save_ckpt(CKPT_DIR / "predecay.pt", raw_model, muon, adamw, step, cfg)
        # The one-epoch branch point: the decay_start a run with a one-epoch budget
        # would have. Fixed name for the same reason as predecay.pt.
        if cfg.get("branch_ckpt_step") is not None and step == cfg["branch_ckpt_step"]:
            save_ckpt(CKPT_DIR / "stable_branch.pt", raw_model, muon, adamw, step, cfg)
        if step > 0 and step % cfg["ckpt_every"] == 0:
            save_ckpt(CKPT_DIR / f"step{step}.pt", raw_model, muon, adamw, step, cfg)
            prune_ckpts(cfg.get("keep_last"), cfg.get("keep_every"))
        # epoch*.pt, so prune_ckpts (which globs step*.pt) never reclaims them.
        if ckpt_every_epoch and step > 0 and step % steps_per_epoch == 0:
            save_ckpt(CKPT_DIR / f"epoch{step // steps_per_epoch}.pt", raw_model, muon, adamw, step, cfg)

    save_ckpt(CKPT_DIR / "final.pt", raw_model, muon, adamw, cfg["max_steps"] - 1, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
