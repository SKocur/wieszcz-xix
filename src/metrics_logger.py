"""Tiny dependency-free CSV logger for training runs.

Writes one row per call, line-buffered so a killed process still leaves a complete,
parseable file up to the last flush. Append mode + header-only-when-new means a --resume
run extends the same CSV instead of clobbering it.
"""

from __future__ import annotations

import datetime
from pathlib import Path

METRICS_DIR = Path("metrics")


def run_name(config_path: str | Path, resume: bool = False, seed: int | None = None) -> str:
    """Metrics prefix `<variant>_<date>[_s<seed>]` (e.g. wieszcz_ar_100m_2026-07-23_s0), so every
    run's CSVs are named by model variant, start date and seed. The seed suffix keeps multi-seed
    runs of the SAME config in separate lineages (no clobbering). On --resume, reuse the newest
    existing dated prefix for this variant+seed, keeping one CSV lineage per run."""
    stem = Path(config_path).stem
    suffix = f"_s{seed}" if seed is not None else ""
    if resume:
        existing = sorted(METRICS_DIR.glob(f"{stem}_20*{suffix}_train.csv"))
        if existing:
            return existing[-1].name[: -len("_train.csv")]
    return f"{stem}_{datetime.date.today().isoformat()}{suffix}"


class MetricsLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self.f = open(self.path, "a", buffering=1)
        if fresh:
            self.f.write(",".join(fieldnames) + "\n")

    def log(self, **row) -> None:
        self.f.write(",".join(str(row.get(k, "")) for k in self.fieldnames) + "\n")

    def close(self) -> None:
        self.f.close()
