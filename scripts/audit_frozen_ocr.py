"""Re-run the corruption audit against the frozen build.

Table 4 was measured on the 333.9M-word build that existed before the corpus was frozen.
The pipeline is unchanged, so the rates should carry over, but "should" is not a
measurement. This decodes a sample of the frozen token file and applies the same detectors
from `src/analyze_ocr.py`, so the two numbers are produced by one classifier.

Sampling rather than the whole corpus: decoding 5.4B tokens takes hours and buys a third
decimal place on a rate whose first decimal place is what the argument turns on. Documents
are drawn at even intervals through each source block, so no uploader's run of scans
dominates.

    python scripts/audit_frozen_ocr.py --docs 400
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from analyze_ocr import reasons, strip_edges  # noqa: E402

TOKENS = REPO / "data/clean/tokens_frozen_5.40B.bin"
IA_DOCS = 216_965
EOT = 0


def audit(texts) -> dict:
    total = suspicious = 0
    by_reason = Counter()
    per_doc = []
    for text in texts:
        n = bad = 0
        for tok in text.split():
            t = strip_edges(tok)
            if not t:
                continue
            n += 1
            r = reasons(t)
            if r:
                bad += 1
                for reason in r:
                    by_reason[reason] += 1
        if n:
            per_doc.append(bad / n)
            total += n
            suspicious += bad
    return {"words": total, "suspicious": suspicious,
            "rate": suspicious / total if total else 0.0,
            "by_reason": {k: v / total for k, v in by_reason.most_common()},
            "per_doc_rates": per_doc}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=400, help="documents sampled per source")
    p.add_argument("--out", default="metrics/ocr_audit_frozen.json")
    args = p.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(str(REPO / "tokenizer/vocab.json"),
                                str(REPO / "tokenizer/merges.txt"))
    arr = np.memmap(TOKENS, dtype=np.uint16, mode="r")

    print("locating document boundaries...")
    pos = []
    CH = 1 << 26
    for s in range(0, arr.size, CH):
        blk = np.asarray(arr[s:s + CH])
        h = np.flatnonzero(blk == EOT)
        if h.size:
            pos.append(h + s)
    pos = np.concatenate(pos)
    print(f"  {len(pos):,} documents")

    report = {"documents_sampled_per_source": args.docs}
    for label, lo, hi in (("internet_archive", 0, IA_DOCS),
                          ("wolne_lektury", IA_DOCS, len(pos))):
        picks = np.linspace(lo + 1, hi - 1, min(args.docs, hi - lo - 1)).astype(int)
        texts = []
        for i in picks:
            s, e = int(pos[i - 1]) + 1, int(pos[i])
            texts.append(tok.decode(np.asarray(arr[s:e]).tolist()))
        r = audit(texts)
        rates = sorted(r.pop("per_doc_rates"))
        r["median_doc_rate"] = rates[len(rates) // 2] if rates else 0.0
        r["p90_doc_rate"] = rates[int(len(rates) * 0.9)] if rates else 0.0
        r["max_doc_rate"] = rates[-1] if rates else 0.0
        report[label] = r
        print(f"{label:18s} {r['words']:>12,} words  suspicious {r['rate']*100:5.2f}%  "
              f"median doc {r['median_doc_rate']*100:5.2f}%  p90 {r['p90_doc_rate']*100:5.2f}%")
        for k, v in list(r["by_reason"].items())[:5]:
            print(f"     {k:24s} {v*100:5.2f}%")

    ia, wl = report["internet_archive"]["rate"], report["wolne_lektury"]["rate"]
    report["true_corruption_estimate"] = round(ia - wl, 5)
    print(f"\nfalse-positive floor (Wolne Lektury): {wl*100:.2f}%")
    print(f"true corruption estimate (IA - WL)  : {(ia-wl)*100:.2f}%")

    out = REPO / args.out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
