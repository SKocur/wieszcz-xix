"""Decoder-only transformer: RoPE, RMSNorm before and after each sublayer, QK-Norm,
SwiGLU, grouped-query attention, weight tying.

A checkpoint holds tensors and a config dict, never a class, so this file is what turns
saved weights back into a model. It imports nothing beyond torch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


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

        One sequence per call: sampling reads a single token per step, so `idx` is
        shape (1, t) and a wider batch raises.

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
        """Collect stream() into a single tensor of prompt followed by continuation."""
        new = list(self.stream(idx, max_new_tokens, temperature, top_k, top_p, eot_id))
        if new:
            idx = torch.cat([idx, torch.tensor([new], dtype=torch.long, device=idx.device)], dim=1)
        return idx
