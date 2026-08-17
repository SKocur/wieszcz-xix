"""Label an adjudication sheet from the terminal, one generation at a time.

Three hundred passages is an hour of reading if the tool stays out of the way, and an
afternoon if the reader has to hunt for the group name, keep a tally, or edit JSON by hand.
So: the matched term is highlighted, one keystroke labels, and every answer is written to
disk before the next passage appears --- a crash or a Ctrl-C costs nothing and the session
resumes where it stopped.

The sheet's hash covers file and id, not labels, so filling it in does not invalidate the
freeze that `screen_bias.py score` checks.

Reading order is the sheet's own: flagged before audit, charged terms first. That order is
a convenience for the reader, not a sampling decision --- every item in a stratum is read,
so the order it is read in cannot bias the count.

    .venv/bin/python3 scripts/review_sheet.py metrics/bias_sheet_2026-08-17.json
    .venv/bin/python3 scripts/review_sheet.py metrics/bias_sheet_2026-08-17.json \\
        --stratum flagged --file bias_holdout_349m_2026-08-17.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from build_bias_prompts import GROUP
from screen_bias import CHARGED

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
HIT, WARN = "\033[43;30m", "\033[41;37m"

KEYS = {"p": "prejudiced", "n": "not", "u": "unclear"}


def paint(text: str) -> str:
    """Group names on yellow, charged terms on red, so the eye lands on the decision."""
    spans: list[tuple[int, int, str]] = []
    for m in GROUP.finditer(text):
        spans.append((m.start(), m.end(), HIT))
    for m in CHARGED.finditer(text):
        spans.append((m.start(), m.end(), WARN))
    if not spans:
        return text
    spans.sort()
    out, pos = [], 0
    for s, e, colour in spans:
        if s < pos:
            continue
        out.append(text[pos:s])
        out.append(f"{colour}{text[s:e]}{RESET}")
        pos = e
    out.append(text[pos:])
    return "".join(out)


def wrap(text: str, width: int = 92) -> str:
    """Wrap on whitespace without breaking the escape sequences paint() inserted."""
    lines, line, n = [], [], 0
    for word in text.split(" "):
        visible = len(re.sub(r"\033\[[0-9;]*m", "", word))
        if n + visible > width and line:
            lines.append(" ".join(line))
            line, n = [], 0
        line.append(word)
        n += visible + 1
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--stratum", choices=["flagged", "audit"], default=None)
    ap.add_argument("--file", default=None, help="restrict to one generation file")
    ap.add_argument("--relabel", action="store_true", help="revisit rows already labelled")
    args = ap.parse_args()

    path = REPO / args.sheet
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["rows"]

    queue = [i for i, r in enumerate(rows)
             if (args.relabel or r["label"] is None)
             and (args.stratum is None or r["stratum"] == args.stratum)
             and (args.file is None or r["file"] == args.file)]
    if not queue:
        print("nothing left to label with those filters")
        return

    done = sum(1 for r in rows if r["label"] is not None)
    print(f"{len(queue)} to read | {done} of {len(rows)} already labelled")
    print(f"{BOLD}p{RESET} prejudiced   {BOLD}n{RESET} not   {BOLD}u{RESET} unclear   "
          f"{BOLD}b{RESET} back   {BOLD}q{RESET} save and quit\n")

    pos = 0
    while 0 <= pos < len(queue):
        i = queue[pos]
        r = rows[i]
        tally = {v: sum(1 for x in rows if x["label"] == v) for v in KEYS.values()}
        print("=" * 96)
        print(f"[{pos+1}/{len(queue)}] {r['file']} :: {r['id']}  "
              f"{DIM}{r['stratum']}  group={r['group_term']}  "
              f"charged={','.join(r['charged']) or '-'}{RESET}")
        print(f"{DIM}running: {tally['prejudiced']} prejudiced / "
              f"{tally['not']} not / {tally['unclear']} unclear{RESET}\n")
        print(f"{DIM}prompt:{RESET} {wrap(r['prompt'])}\n")
        print(wrap(paint(r["text"])))
        if r["label"]:
            print(f"\n{DIM}current label: {r['label']}{RESET}")

        try:
            key = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ninterrupted; labels so far are already saved")
            return

        if key == "q":
            break
        if key == "b":
            pos = max(0, pos - 1)
            continue
        if key not in KEYS:
            print(f"{DIM}use p / n / u / b / q{RESET}")
            continue

        rows[i]["label"] = KEYS[key]
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        pos += 1

    done = sum(1 for r in rows if r["label"] is not None)
    tally = {v: sum(1 for x in rows if x["label"] == v) for v in KEYS.values()}
    print(f"\nlabelled {done} of {len(rows)}: {tally['prejudiced']} prejudiced, "
          f"{tally['not']} not, {tally['unclear']} unclear")
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
