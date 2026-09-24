"""Recover the 350M run's loss curves from its stdout log and draw its figure.

SUPERSEDED by `plot_ladder_curves.py`, which draws the ladder the paper actually
reports. This one targets `wieszcz_350m_2026-07-25`, a run on the previous 5.40B
build whose checkpoints are no longer kept locally; it is retained because the log
parsing is the only record of a run that predates the per-run metrics logger.

The 350M run predates the per-run metrics logger, so its only surviving record is
`train.out` on the network volume. Parsed back into the CSV shape the logger writes.

Two panels: the full curve, and the final epoch, where the 0.05-nat band that matters is
invisible on a y-axis spanning the 9.2-nat start.

    python scripts/plot_350m_curve.py train.out
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = Path("metrics")
# The paper lives in its own repo, checked out alongside this one.
FIGURES = Path("../wieszcz-xix-paper/figures")

RUN = "wieszcz_350m_2026-07-25"
TOKENS_PER_STEP = 262_144          # effective batch 256 seq x 1024 tokens
MAX_STEPS = 41_225                 # exactly 2 epochs of the 5.40B-token corpus
DECAY_FRAC = 0.1                   # WSD: LR decays over the final tenth

# Protanopia-safe pair, worst adjacent dE 24.7.
C_TRAIN = "#2a78d6"
C_VAL = "#eb6834"
C_GRID = "#d8d8d4"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"

STEP_RE = re.compile(r"step\s+(\d+) \| loss ([\d.]+) \| grad_norm ([\d.]+) \| lr x([\d.]+)")
VAL_RE = re.compile(r"val loss ([\d.]+)")


def parse(path: Path):
    train, val, current = [], [], None
    for line in path.read_text(errors="ignore").splitlines():
        if m := STEP_RE.match(line.strip()):
            current = int(m.group(1))
            train.append((current, float(m.group(2)), float(m.group(3)), float(m.group(4))))
        elif (m := VAL_RE.search(line)) and current is not None:
            # The val line carries no step of its own; it follows the step it belongs to.
            val.append((current, float(m.group(1))))
    return train, val


def write_csvs(train, val) -> None:
    METRICS.mkdir(exist_ok=True)
    with open(METRICS / f"{RUN}_train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "tokens", "elapsed_s", "loss", "grad_norm", "lr_mult"])
        for step, loss, gn, lr in train:
            w.writerow([step, step * TOKENS_PER_STEP, "", f"{loss:.4f}", f"{gn:.3f}", f"{lr:.4f}"])
    with open(METRICS / f"{RUN}_val.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "tokens", "elapsed_s", "val_loss"])
        for step, vl in val:
            w.writerow([step, step * TOKENS_PER_STEP, "", f"{vl:.4f}"])


def style(ax) -> None:
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8, length=3)


def plot(train, val, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "text.color": C_INK, "axes.labelcolor": C_INK})

    B = 1e9
    tx = [s * TOKENS_PER_STEP / B for s, *_ in train]
    ty = [loss for _, loss, _, _ in train]
    vx = [s * TOKENS_PER_STEP / B for s, _ in val]
    vy = [v for _, v in val]

    epoch = MAX_STEPS / 2 * TOKENS_PER_STEP / B
    decay = MAX_STEPS * (1 - DECAY_FRAC) * TOKENS_PER_STEP / B

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    for ax in (ax1, ax2):
        style(ax)
        ax.axvspan(decay, tx[-1], color=C_GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axvline(epoch, color=C_MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlabel("tokens processed (billions)")

    ax1.plot(tx, ty, color=C_TRAIN, linewidth=0.7, alpha=0.9, label="train", zorder=2)
    ax1.plot(vx, vy, color=C_VAL, linewidth=1.4, label="held-out", zorder=3)
    ax1.set_ylabel("cross-entropy (nats/token)")
    ax1.set_ylim(2.5, 9.5)
    ax1.legend(frameon=False, fontsize=8, labelcolor=C_MUTED, loc="upper right")
    ax1.text(epoch, 9.2, " 1 epoch", color=C_MUTED, fontsize=7.5, va="top")
    ax1.text(decay, 2.7, " LR decay ", color=C_MUTED, fontsize=7.5, va="bottom")

    # Right panel: the final third, where the LR decay could be doing all the work.
    lo = 0.62 * tx[-1]
    ax2.plot([x for x in tx if x >= lo], [y for x, y in zip(tx, ty) if x >= lo],
             color=C_TRAIN, linewidth=0.7, alpha=0.9, zorder=2)
    ax2.plot([x for x in vx if x >= lo], [y for x, y in zip(vx, vy) if x >= lo],
             color=C_VAL, linewidth=1.4, marker="o", markersize=2.4,
             markeredgecolor="white", markeredgewidth=0.3, zorder=3)
    ax2.set_xlim(lo, tx[-1] * 1.005)
    ax2.set_ylim(2.55, 3.35)

    best = min(val, key=lambda v: v[1])
    bx = best[0] * TOKENS_PER_STEP / B
    ax2.annotate(f"best {best[1]:.3f}", xy=(bx, best[1]), xytext=(bx - 1.9, 3.24),
                 color=C_MUTED, fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", color=C_MUTED, linewidth=0.6))
    # Direct labels, so the series are not identified by colour alone. Both go above
    # their line; below the train curve there is no room before the axis.
    ax2.text(tx[-1], 2.90, "train", color=C_TRAIN, fontsize=8, ha="right", va="bottom")
    ax2.text(tx[-1], vy[-1] + 0.05, "held-out", color=C_VAL, fontsize=8, ha="right", va="bottom")
    ax2.annotate("", xy=(tx[-1] * 0.945, 2.72), xytext=(tx[-1] * 0.945, 3.09),
                 arrowprops=dict(arrowstyle="<->", color=C_MUTED, linewidth=0.6))
    ax2.text(tx[-1] * 0.94, 2.90, "cross-source gap\n$\\approx$0.35 nats", color=C_MUTED,
             fontsize=7, ha="right", va="center")

    fig.tight_layout(pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", help="path to the run's train.out")
    p.add_argument("--out", default=str(FIGURES / "loss_350m.pdf"))
    args = p.parse_args()

    train, val = parse(Path(args.log))
    print(f"parsed {len(train)} train points, {len(val)} val points")
    write_csvs(train, val)
    plot(train, val, Path(args.out))


if __name__ == "__main__":
    main()
