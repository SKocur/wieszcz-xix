"""Score the ladder on the held-out split as a whole and on each source separately.

The paper's sharpest claim is that a source-biased validation split flattens the measured
scaling exponent. Its present evidence compares two exponents from two *different* corpus
builds, so the split is confounded with deduplication, decontamination and 1.3B extra
tokens; the attribution to the split alone is not earned.

This measures the mechanism with everything else held fixed. The held-out split already
contains both sources, so the same three checkpoints are scored on the Internet Archive
subset (the dominant training source, in-distribution) and on the Wolne Lektury subset
(0.71% of training tokens, clean transcription against OCR, effectively near-OOD), and an
exponent is fitted to each. Same models, same corpus, same protocol -- only the evaluation
source differs. If the exponent flattens on the minority source, the mechanism is shown
rather than inferred.

The two source blocks are contiguous in the token file: documents are written in
code-point-sorted filename order and the prefixes are disjoint, so `ia_` precedes `wl_`
and counting document terminators locates the boundary exactly.

Three parameters through three points fit without residual, so the exponent carries no
residual-based uncertainty. The interval here is a bootstrap over the *windows* instead:
resample each rung's per-window losses, refit, repeat. That propagates the only randomness
the measurement actually has.

    .venv/bin/python3 scripts/eval_by_source.py --windows 8192
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from eval_val import evaluate, load_model
from train import get_device

def eot_id() -> int:
    """Read the terminator's id from the tokenizer rather than assuming it.

    Guessing it as the last vocabulary slot is wrong here -- it is id 0 -- and a wrong
    terminator would silently put the source boundary in the middle of a document.
    """
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                str(REPO / "tokenizer/merges.txt"))
    return tok.token_to_id("<|endoftext|>")


LADDER = [
    ("47M", "checkpoints/wieszcz_47m_6b7_2026-08-05_s1337/final.pt"),
    ("107M", "checkpoints/wieszcz_107m_6b7_2026-08-06_s1337/final.pt"),
    ("349M", "checkpoints/wieszcz_349m_6b7_2026-08-07_s1337/final.pt"),
]


def source_boundary(data: np.ndarray, eot_id: int, n_ia_docs: int) -> int:
    """Token index where the Wolne Lektury block starts.

    Every document is followed by one terminator, so the Internet Archive block ends just
    past its last one. Verified against the split's own document counts by the caller.
    """
    eots = np.flatnonzero(data == eot_id)
    if len(eots) < n_ia_docs:
        raise SystemExit(f"found {len(eots)} terminators, expected at least {n_ia_docs}")
    return int(eots[n_ia_docs - 1]) + 1


def fit_alpha(ns: list[float], ls: list[float]) -> dict:
    """Solve L = E + A*N^-alpha exactly through three points.

    Three unknowns and three points means the fit is determined, not estimated: the
    exponent is whichever value makes (N^-alpha, L) collinear. Bisection on that
    collinearity residual is more robust here than a least-squares optimiser, which has no
    residual to descend.
    """
    (n1, n2, n3), (l1, l2, l3) = ns, ls

    def resid(a: float) -> float:
        x1, x2, x3 = n1 ** -a, n2 ** -a, n3 ** -a
        return (l2 - l1) * (x3 - x2) - (l3 - l2) * (x2 - x1)

    lo, hi = 1e-3, 3.0
    flo, fhi = resid(lo), resid(hi)
    if flo * fhi > 0:
        return {"alpha": None, "note": "no sign change; points not power-law separable"}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if flo * resid(mid) <= 0:
            hi = mid
        else:
            lo, flo = mid, resid(mid)
    a = 0.5 * (lo + hi)
    x1, x2 = n1 ** -a, n2 ** -a
    A = (l2 - l1) / (x2 - x1)
    E = l1 - A * x1
    return {"alpha": round(a, 5), "A": round(A, 5), "E_nats": round(E, 5)}


def bootstrap_alpha(ns: list[float], per_window: list[np.ndarray], draws: int,
                    seed: int) -> dict:
    """Resample windows within each rung and refit, for an interval on the exponent."""
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(draws):
        ls = [float(rng.choice(w, size=len(w), replace=True).mean()) for w in per_window]
        f = fit_alpha(ns, ls)
        if f.get("alpha") is not None:
            alphas.append(f["alpha"])
    if not alphas:
        return {"draws_valid": 0}
    a = np.array(alphas)
    return {"draws_valid": len(alphas),
            "alpha_median": round(float(np.median(a)), 5),
            "alpha_ci95": [round(float(np.percentile(a, 2.5)), 5),
                           round(float(np.percentile(a, 97.5)), 5)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/val_2026-08-03.bin")
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--windows", type=int, default=8192,
                    help="cap per subset; the smaller source uses all it has")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="metrics/eval_by_source_2026-08-17.json")
    args = ap.parse_args()

    t0 = time.time()
    device = args.device or get_device()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16}[args.dtype]

    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    val_ids = split["val_ids"]
    n_ia = sum(1 for i in val_ids if i.startswith("ia_"))
    n_wl = sum(1 for i in val_ids if i.startswith("wl_"))

    eot = eot_id()
    data = np.memmap(REPO / args.val, dtype=np.uint16, mode="r")
    total_eot = int(np.count_nonzero(np.asarray(data) == eot))
    cut = source_boundary(np.asarray(data), eot, n_ia)
    subsets = {"full": np.asarray(data),
               "ia": np.asarray(data)[:cut],
               "wl": np.asarray(data)[cut:]}

    print(f"val {len(data):,} tokens | terminators {total_eot:,} | split says "
          f"{len(val_ids):,} docs ({n_ia:,} ia + {n_wl} wl)")
    print(f"boundary at token {cut:,}  ->  ia {cut:,} | wl {len(data)-cut:,}")
    if total_eot != len(val_ids):
        print(f"  WARNING: terminator count != document count", flush=True)

    results = {name: {} for name in subsets}
    per_window = {name: [] for name in subsets}
    params = []

    for label, rel in LADDER:
        model, cfg, step, n_params = load_model(str(REPO / rel), device, dtype)
        params.append(float(n_params))
        block = cfg["block_size"]
        print(f"\n{label}: {n_params/1e6:.1f}M params, step {step}", flush=True)
        for name, arr in subsets.items():
            avail = (len(arr) - 1) // block
            n = min(args.windows, avail) if args.windows else avail
            losses, meta = evaluate(model, arr, block, n, args.batch_size, device, dtype)
            ce = float(losses.mean())
            sem = float(losses.std(ddof=1) / math.sqrt(len(losses)))
            results[name][label] = {
                "cross_entropy_nats": round(ce, 5), "sem": round(sem, 5),
                "ci95": [round(ce - 1.96 * sem, 5), round(ce + 1.96 * sem, 5)],
                "perplexity": round(math.exp(ce), 4),
                "params": n_params, "step": step,
                "windows_available": int(avail), **meta,
            }
            per_window[name].append(losses)
            print(f"  {name:>4}: CE {ce:.4f} +-{1.96*sem:.4f} "
                  f"({meta['windows_evaluated']} of {avail} windows)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    fits = {}
    for name in subsets:
        ls = [results[name][lab]["cross_entropy_nats"] for lab, _ in LADDER]
        fits[name] = fit_alpha(params, ls)
        fits[name].update(bootstrap_alpha(params, per_window[name], args.bootstrap,
                                          args.seed))
        fits[name]["losses"] = ls

    report = {
        "meta": {
            "script": "scripts/eval_by_source.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(), "device": device, "dtype": args.dtype,
            "val": args.val, "split": args.split,
            "val_tokens": int(len(data)), "val_documents": len(val_ids),
            "val_documents_ia": n_ia, "val_documents_wl": n_wl,
            "eot_id": eot, "terminators_found": total_eot,
            "source_boundary_token": cut,
            "windows_cap": args.windows, "bootstrap_draws": args.bootstrap,
            "seed": args.seed,
            "elapsed_seconds": None,
        },
        "ladder": [{"label": lab, "ckpt": rel} for lab, rel in LADDER],
        "by_subset": results,
        "alpha_fits": fits,
    }
    report["meta"]["elapsed_seconds"] = round(time.time() - t0, 1)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    for name in subsets:
        np.save(out.with_suffix(f".{name}.windows.npy"),
                np.stack([w[:min(len(x) for x in per_window[name])]
                          for w in per_window[name]]))

    print("\n" + "=" * 60)
    for name in ("full", "ia", "wl"):
        f = fits[name]
        print(f"{name:>4}: CE {[round(x,4) for x in f['losses']]}  "
              f"alpha {f.get('alpha')}  ci95 {f.get('alpha_ci95')}  E {f.get('E_nats')}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
