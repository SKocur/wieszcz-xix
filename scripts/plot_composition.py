"""Draw the paper figure for the corpus composition over time.

Two panels from the composition report of `corpus_composition.py`: the share of
Internet Archive documents and of bytes in each decade of the catalogue year, and the
-cja share (-cja against -cya spellings) per decade in print from the three largest
places of publication. A place contributes a point only for decades above the report's
document threshold.

    python scripts/plot_composition.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path("metrics/corpus_composition_2026-10-06.json")
# The paper lives in its own repo, checked out alongside this one.
FIGURES = Path("../wieszcz-xix-paper/figures")

PLACES = [
    ("Warszawa", "Warsaw", "#0072B2"),
    ("Kraków", "Kraków", "#E69F00"),
    ("Lwów", "Lwów", "#009E73"),
]

C_GRID = "#d8d8d4"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_DOCS = "#b9b8b3"
C_BYTES = "#52514e"


def style(ax) -> None:
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8, length=3)


def plot(report: Path, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "text.color": C_INK, "axes.labelcolor": C_INK})
    rep = json.loads(report.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (ax1, ax2):
        style(ax)
        ax.set_xlabel("decade of publication")
        ax.set_xticks(range(1800, 1911, 20))

    decades = sorted(int(d) for d in rep["by_decade"])
    for offset, key, label, color in ((-1.9, "share_documents", "documents", C_DOCS),
                                      (1.9, "share_bytes", "bytes", C_BYTES)):
        ax1.bar([d + 5 + offset for d in decades],
                [rep["by_decade"][str(d)][key] for d in decades],
                width=3.4, color=color, linewidth=0, zorder=2, label=label)
    ax1.set_xlim(1797, 1923)
    ax1.set_ylabel("share of the total (%)")
    ax1.legend(frameon=False, fontsize=8, labelcolor=C_MUTED, loc="upper left",
               handlelength=1.0, borderaxespad=0.2)

    handles = []
    for key, label, color in PLACES:
        rows = rep["orto"]["by_place_decade"][key]
        pts = [(int(d) + 5, r["share_modern"]) for d, r in rows.items()
               if r["documents"] >= rep["orto"]["plotted_cells"]["min_documents"]]
        xs, ys = zip(*pts)
        (line,) = ax2.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=3.6,
                           solid_capstyle="round", solid_joinstyle="round", zorder=3,
                           label=label)
        handles.append(line)
        ax2.text(xs[-1] + 4, ys[-1], label, color=C_INK, fontsize=7.5,
                 ha="left", va="center")
    ax2.set_xlim(1797, 1938)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("-cja share")
    ax2.legend(handles=handles, frameon=False, fontsize=8, labelcolor=C_MUTED,
               loc="upper left", handlelength=1.4, borderaxespad=0.2)

    fig.tight_layout(pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} (+ .png)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=str(REPORT))
    p.add_argument("--out", default=str(FIGURES / "composition.pdf"))
    args = p.parse_args()
    plot(Path(args.report), Path(args.out))


if __name__ == "__main__":
    main()
