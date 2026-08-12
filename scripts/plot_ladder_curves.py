"""Draw the paper figure for the matched ladder trained on the 2026-08-03 build.

All three runs log to per-run CSVs (the `<variant>_<date>_train/val.csv` convention),
so unlike the 350M predecessor there is nothing to recover from stdout. Two panels:
the full curves against the corpus entropy anchors, and the final third, where the
in-distribution train/held-out gap is visible at all.

    python scripts/plot_ladder_curves.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = Path("metrics")
# The paper lives in its own repo, checked out alongside this one.
FIGURES = Path("../wieszcz-xix-paper/figures")

RUNS = [
    ("47M", "wieszcz_47m_6b7_2026-08-05_s1337", "#0072B2"),
    ("107M", "wieszcz_107m_6b7_2026-08-06_s1337", "#E69F00"),
    ("349M", "wieszcz_349m_6b7_2026-08-07_s1337", "#009E73"),
]
TOKENS_PER_STEP = 262_144          # effective batch 256 seq x 1024 tokens
MAX_STEPS = 51_038                 # exactly 2 epochs of the 6.69B-token train split
DECAY_FRAC = 0.1                   # WSD: LR decays over the final tenth

UNIGRAM_NATS = 7.3927              # metrics/entropy_baseline_2026-08-03.json
BIGRAM_NATS = 5.3190

C_GRID = "#d8d8d4"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"


def read_run(run: str):
    with open(METRICS / f"{run}_train.csv", newline="") as fh:
        train = [(int(r["step"]), float(r["loss"])) for r in csv.DictReader(fh)]
    with open(METRICS / f"{run}_val.csv", newline="") as fh:
        val = [(int(r["step"]), float(r["val_loss"])) for r in csv.DictReader(fh)]
    return train, val


def style(ax) -> None:
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8, length=3)


def plot(out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "text.color": C_INK, "axes.labelcolor": C_INK})

    B = 1e9
    epoch = MAX_STEPS / 2 * TOKENS_PER_STEP / B
    decay = MAX_STEPS * (1 - DECAY_FRAC) * TOKENS_PER_STEP / B
    end = MAX_STEPS * TOKENS_PER_STEP / B

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    for ax in (ax1, ax2):
        style(ax)
        ax.axvspan(decay, end, color=C_GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axvline(epoch, color=C_MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlabel("tokens processed (billions)")

    for anchor, name in ((UNIGRAM_NATS, "unigram"), (BIGRAM_NATS, "bigram")):
        ax1.axhline(anchor, color=C_MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
        ax1.text(end, anchor + 0.06, f"{name} {anchor:.2f} ", color=C_MUTED,
                 fontsize=7, ha="right", va="bottom")

    lo = 0.62 * end
    handles = []
    for label, run, color in RUNS:
        train, val = read_run(run)
        tx = [s * TOKENS_PER_STEP / B for s, _ in train]
        ty = [l for _, l in train]
        vx = [s * TOKENS_PER_STEP / B for s, _ in val]
        vy = [l for _, l in val]

        ax1.plot(tx, ty, color=color, linewidth=0.5, alpha=0.45, zorder=2)
        (line,) = ax1.plot(vx, vy, color=color, linewidth=1.3, zorder=3, label=label)
        handles.append(line)

        ax2.plot([x for x in tx if x >= lo], [y for x, y in zip(tx, ty) if x >= lo],
                 color=color, linewidth=0.5, alpha=0.45, zorder=2)
        ax2.plot([x for x in vx if x >= lo], [y for x, y in zip(vx, vy) if x >= lo],
                 color=color, linewidth=1.3, zorder=3)
        ax2.text(vx[-1] + 0.06, vy[-1], f"{label}  {vy[-1]:.3f}", color=color,
                 fontsize=7.5, ha="left", va="center")

    ax1.set_ylabel("cross-entropy (nats/token)")
    ax1.set_ylim(2.4, 9.5)
    ax1.text(epoch, 9.2, " 1 epoch", color=C_MUTED, fontsize=7.5, va="top")
    ax1.text((decay + end) / 2, 4.0, "LR\ndecay", color=C_MUTED, fontsize=7,
             va="bottom", ha="center", linespacing=1.1)
    ax1.legend(handles=handles, frameon=False, fontsize=8, labelcolor=C_MUTED,
               loc="upper right", handlelength=1.4, borderaxespad=0.2)

    ax2.set_xlim(lo, end * 1.075)
    ax2.set_ylim(2.55, 3.30)
    # Thin line = loss on training batches (second-epoch, i.e. repeated data);
    # thick line = the held-out probe. The caption carries the gap numbers.
    ax2.text(lo + 0.12, 2.575, "thin: train batches   thick: held-out",
             color=C_MUTED, fontsize=7, va="bottom")

    fig.tight_layout(pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(FIGURES / "loss_ladder.pdf"))
    args = p.parse_args()
    plot(Path(args.out))


if __name__ == "__main__":
    main()
