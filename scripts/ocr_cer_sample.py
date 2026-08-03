"""Build and score a character-error-rate sample for the Internet Archive half.

The corruption audit in Section 5 measures detector rates, which say how much text *looks*
wrong. They cannot say how wrong it is, because nothing in the pipeline compares the text
against what the page said. This builds that comparison: fixed-seed passages, a
hand-corrected reference for each, then CER and WER.

One limitation is structural and is reported rather than hidden. We do not hold the page
images, so a reference is a reconstruction from context, not a transcription of the
original. Where the text is corrupt enough that context does not determine the reading, no
reference can honestly be written; those passages are marked `null` and counted separately,
and that unreconstructable rate is itself a measurement.

    python scripts/ocr_cer_sample.py build --n 20
    python scripts/ocr_cer_sample.py score
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO / "output/ocr_sample/ia"
OUT = REPO / "metrics/ocr_cer_sample.json"
SEED = 1337
PASSAGE_CHARS = 400


def levenshtein(a: str, b: str) -> int:
    """Two rows rather than a full matrix: passages are short, but the scorer runs over
    every pair and there is no reason to hold n*m integers."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise(s: str) -> str:
    """Collapse whitespace and NFC-normalise. Line breaks in the raw text are page layout,
    not content, and scoring them as errors would measure the scan's column structure."""
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s)).strip()


def cmd_build(args) -> None:
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"no decoded documents in {SAMPLE_DIR}; run corpus_stats.py first")
    rng = random.Random(SEED)
    items = []
    for i in range(args.n):
        f = files[i % len(files)]
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) < PASSAGE_CHARS * 3:
            continue
        start = rng.randrange(len(text) // 10, len(text) - PASSAGE_CHARS - 1)
        # Start and end on a space so no passage begins or ends mid-word.
        start = text.find(" ", start) + 1
        end = text.find(" ", start + PASSAGE_CHARS)
        if end == -1:
            end = start + PASSAGE_CHARS
        items.append({"id": f"p{i:02d}", "doc": f.name,
                      "raw": normalise(text[start:end]), "ref": None,
                      "unreconstructable": False, "note": ""})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "passage_chars": PASSAGE_CHARS,
                               "items": items}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {len(items)} passages to {OUT}")


def cmd_score(args) -> None:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    items = data["items"]
    done = [it for it in items if it["ref"] and not it["unreconstructable"]]
    bad = [it for it in items if it["unreconstructable"]]
    if not done:
        raise SystemExit("no corrected references yet")

    c_err = c_tot = w_err = w_tot = 0
    per = []
    for it in done:
        raw, ref = normalise(it["raw"]), normalise(it["ref"])
        ce = levenshtein(raw, ref)
        we = levenshtein(raw.split(), ref.split())
        c_err += ce; c_tot += len(ref)
        w_err += we; w_tot += len(ref.split())
        per.append({"id": it["id"], "cer": ce / len(ref), "wer": we / len(ref.split())})

    report = {
        "passages_scored": len(done),
        "passages_unreconstructable": len(bad),
        "unreconstructable_rate": round(len(bad) / len(items), 4),
        "characters": c_tot, "character_errors": c_err, "cer": round(c_err / c_tot, 5),
        "words": w_tot, "word_errors": w_err, "wer": round(w_err / w_tot, 5),
        "per_passage": per,
        "protocol": "Reference reconstructed from context by one annotator; no page images "
                    "were available. Whitespace collapsed before scoring, so line breaks "
                    "from page layout are not counted as errors.",
    }
    out = OUT.with_name("ocr_cer_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scored {len(done)} passages, {len(bad)} unreconstructable")
    print(f"  CER {report['cer']:.4f}  ({c_err:,} edits over {c_tot:,} characters)")
    print(f"  WER {report['wer']:.4f}  ({w_err:,} edits over {w_tot:,} words)")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=20)
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("score"); s.set_defaults(func=cmd_score)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
