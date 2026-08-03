"""Run the OCR/scan cleaner over a corpus directory, streaming file-by-file.

Clean documents go to <out>/clean/. Documents too garbled to recover, mostly unmarked
two-column merges, are logged to <out>/rejected.tsv with a reason.

    python scripts/clean_corpus.py --in data/clean --out data/cleaned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
from clean_ocr import route  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-score", type=float, default=0.55)
    args = ap.parse_args()

    src = Path(args.inp)
    out = Path(args.out)
    clean_dir = out / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    rej_log = out / "rejected.tsv"

    n = kept = rejected = 0
    chars_in = chars_out = 0
    with open(rej_log, "w", encoding="utf-8") as rej:
        rej.write("file\treason\n")
        for p in src.glob("*.txt"):
            if p.name.startswith("."):
                continue
            n += 1
            raw = p.read_text(encoding="utf-8", errors="replace")
            chars_in += len(raw)
            verdict, reason, cleaned = route(raw, max_score=args.max_score)
            if verdict == "clean":
                (clean_dir / p.name).write_text(cleaned, encoding="utf-8")
                kept += 1
                chars_out += len(cleaned)
            else:
                rej.write(f"{p.name}\t{reason}\n")
                rejected += 1
            if n % 5000 == 0:
                print(f"  ...{n:,} plikow ({kept:,} clean / {rejected:,} reject)", flush=True)

    print(f"\n=== gotowe: {n:,} plikow ===")
    print(f"clean:  {kept:,} ({kept/n*100:.1f}%)  -> {clean_dir}")
    print(f"reject: {rejected:,} ({rejected/n*100:.1f}%) -> {rej_log}")
    if chars_in:
        print(f"znaki: {chars_in:,} -> {chars_out:,} ({chars_out/chars_in*100:.0f}% zachowane)")


if __name__ == "__main__":
    main()
