"""Generate the paper's numeric macros from the measurement files, and catch hand-typing.

The LaTeX source is not published, so a reader cannot rebuild the document and find a
transcription error the way they could find a code bug. The only way a wrong number gets
caught from outside is if someone recomputes it from the released metrics and notices the
disagreement. That makes the guard against mistyping entirely our problem, and it has
already failed once: the exponent's interval $[0.192, 0.204]$ appears four times in the
paper and in no file at all --- the committed report carries the pre-fix $[0.060, 0.351]$,
and recomputing from the saved per-window losses gives $[0.193, 0.205]$.

So the numbers are emitted from the files rather than copied out of them. `numbers.tex`
holds one macro per quantity; main.tex cites the macro; re-measuring the ladder means
regenerating and rebuilding rather than a search-and-replace across the prose.

`--check` is the half that matters after the macros exist. It reads main.tex and reports
any generated value that also appears as a bare literal, because a literal is a number that
will not follow the next re-measurement.

    python scripts/paper_numbers.py
    python scripts/paper_numbers.py --check
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO.parent / "wieszcz-xix-paper"

# Control sequence names may contain letters only, so the rungs are named by
# their place in the ladder rather than by parameter count.
LADDER = (("47M", "small"), ("107M", "mid"), ("349M", "large"))

# The orthography arm. The paper calls it the measurement it would keep if it could keep
# only one, which makes it the last place a number should be retyped. Each rung's own
# report carries the wieszcz arms; the comparator arms are named by the model they scored,
# since the same rung is compared against both.
PROBE_RUNGS = {
    "mid": "metrics/temporal_probe_107m_papugapt2_2026-08-18.json",
    "large": "metrics/temporal_probe_349m_bielik_2026-08-18.json",
}
PROBE_COMPARATORS = {
    "bielik": "metrics/temporal_probe_349m_bielik_2026-08-18.json",
    "papuga": "metrics/temporal_probe_349m_papugapt2_2026-08-18.json",
}


def dig(obj, path: str):
    """Walk a dotted path, tolerating list indices."""
    cur = obj
    for part in path.split("."):
        cur = cur[int(part)] if part.lstrip("-").isdigit() else cur[part]
    return cur


def tex_int(v) -> str:
    return f"{int(v):,}".replace(",", "{,}")


def tex_f(v, places: int) -> str:
    return f"{float(v):.{places}f}"


def tex_ci(v, places: int = 3) -> str:
    lo, hi = v
    return f"[{float(lo):.{places}f}, {float(hi):.{places}f}]"


def probe_macros(out: dict[str, str]) -> None:
    """Orthography shares from the temporal probes, plus what the exemplars themselves were.

    The comparison across reports is only meaningful if every model saw the same exemplars,
    so that is checked here rather than assumed: the passages are drawn under a fixed seed
    and identical selection thresholds, and a report whose preamble differs would be
    comparing two different demonstrations.
    """
    share = lambda block: tex_f(block["orto_modern_share"]["mean"], 3)
    preambles = {}

    for tag, path in PROBE_RUNGS.items():
        ev = _load_probe(path)
        out[f"orto{tag}"] = share(ev["generations_wieszcz"])
        out[f"orto{tag}fewshot"] = share(ev["generations_wieszcz_fewshot"])
        preambles[path] = ev["generations_wieszcz_fewshot"]["preamble"]

    for tag, path in PROBE_COMPARATORS.items():
        ev = _load_probe(path)
        out[f"orto{tag}plain"] = share(ev["generations_modern_lm_plain"])
        out[f"orto{tag}frame"] = share(ev["generations_modern_lm_imitate"])
        out[f"orto{tag}fewshot"] = share(ev["generations_modern_lm_fewshot"])
        preambles[path] = ev["generations_modern_lm_fewshot"]["preamble"]

    if len(set(preambles.values())) != 1:
        raise SystemExit("probe reports disagree on the exemplars, so their orthography "
                         f"shares are not comparable: {sorted(preambles)}")

    ev = _load_probe(PROBE_COMPARATORS["bielik"])
    out["ortocorpus"] = share(ev["corpus_reference"])
    fs = ev["fewshot"]
    out["fewshotpassages"] = tex_int(len(fs["passages"]))
    out["fewshotchars"] = tex_int(fs["fit_wieszcz"]["chars"])
    out["fewshotshare"] = tex_f(fs["fit_wieszcz"]["scan"]["orto_modern_share"], 3)


def _load_probe(rel: str) -> dict:
    path = REPO / rel
    if not path.exists():
        raise SystemExit(f"{rel} not found. Re-run scripts/temporal_probe.py, or point "
                         f"PROBE_RUNGS/PROBE_COMPARATORS at the reports you mean.")
    return json.loads(path.read_text())


def collect(eval_file: str) -> dict[str, str]:
    """Every macro, with the file it came from stated in one place."""
    out: dict[str, str] = {}

    tok = json.loads((REPO / "metrics/tokenize_2026-08-03.json").read_text())
    split = json.loads((REPO / "metrics/doc_split_2026-08-03.json").read_text())
    n_train, n_val = len(split["train_ids"]), len(split["val_ids"])

    out["traintokens"] = tex_int(dig(tok, "train.tokens"))
    out["valtokens"] = tex_int(dig(tok, "val.tokens"))
    out["corpustokens"] = tex_int(dig(tok, "tokens_total"))
    out["batchtokens"] = tex_int(dig(tok, "batch_tokens"))
    out["corpusdocs"] = tex_int(n_train + n_val)
    out["traindocs"] = tex_int(n_train)
    out["valdocs"] = tex_int(n_val)

    if dig(tok, "train.tokens") + dig(tok, "val.tokens") != dig(tok, "tokens_total"):
        raise SystemExit("tokenise report disagrees with itself on the total")

    ev_path = REPO / eval_file
    if not ev_path.exists():
        raise SystemExit(f"{eval_file} not found. Re-measure the ladder first, or pass "
                         f"--eval with the report you mean; producing the corpus macros "
                         f"alone would leave the results silently missing.")
    ev = json.loads(ev_path.read_text())
    for subset, tag in (("full", "full"), ("ia", "ia"), ("wl", "wl")):
        if subset not in ev.get("alpha_fits", {}):
            continue
        fit = ev["alpha_fits"][subset]
        if fit.get("alpha") is not None:
            out[f"alpha{tag}"] = tex_f(fit["alpha"], 3)
        if fit.get("alpha_ci95"):
            out[f"alpha{tag}ci"] = tex_ci(fit["alpha_ci95"])
        if fit.get("E_nats") is not None:
            out[f"irreducible{tag}"] = tex_f(fit["E_nats"], 3)
    span = None
    for rung, tag in LADDER:
        cell = dig(ev, f"by_subset.full.{rung}")
        out[f"ce{tag}"] = tex_f(cell["cross_entropy_nats"], 4)
        out[f"ceci{tag}"] = tex_ci(cell["ci95"], 4)
        out[f"params{tag}"] = tex_int(cell["params"])
        span = cell.get("span_fraction", span)
    # Per source, and the penalty on the clean transcription: the table that states these
    # is the last place a number should be retyped, since it carries three columns of them.
    for rung, tag in LADDER:
        for subset in ("ia", "wl"):
            if subset in ev["by_subset"]:
                cell = dig(ev, f"by_subset.{subset}.{rung}")
                out[f"ce{subset}{tag}"] = tex_f(cell["cross_entropy_nats"], 3)
                out[f"half{subset}{tag}"] = tex_f(
                    (cell["ci95"][1] - cell["ci95"][0]) / 2, 3)
                out[f"ppl{subset}{tag}"] = tex_f(cell["perplexity"], 1)
        if "ia" in ev["by_subset"] and "wl" in ev["by_subset"]:
            gap = (dig(ev, f"by_subset.wl.{rung}.cross_entropy_nats")
                   - dig(ev, f"by_subset.ia.{rung}.cross_entropy_nats"))
            out[f"cegap{tag}"] = tex_f(gap, 3)
    out["wlwindows"] = tex_int(dig(ev, "by_subset.wl.47M.windows_evaluated"))

    losses = [float(dig(ev, f"by_subset.full.{r}.cross_entropy_nats")) for r, _ in LADDER]
    # Quoted to three places in the prose, which is the precision the +-0.001
    # half-widths support.
    out["gainsmalltomid"] = tex_f(losses[0] - losses[1], 3)
    out["gainmidtolarge"] = tex_f(losses[1] - losses[2], 3)
    if span is not None:
        out["evalspan"] = tex_f(span, 4)
    out["evalwindows"] = tex_int(ev["meta"]["windows_cap"])

    probe_macros(out)

    bad = [n for n in out if not n.isalpha()]
    if bad:
        raise SystemExit(f"macro names must be letters only, LaTeX cannot define {bad}")
    return out


def cmd_write(args) -> None:
    macros = collect(args.eval)
    lines = [
        "% Generated by scripts/paper_numbers.py from the files in wieszcz-xix/metrics.",
        "% Do not edit: run the script again after re-measuring.",
        f"% eval source: {args.eval}",
        "",
    ]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    out = PAPER / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in sorted(macros):
        print(f"  \\{name:<18} {macros[name]}")
    print(f"\n{len(macros)} macros -> {out}")


def cmd_check(args) -> None:
    macros = collect(args.eval)
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    body = re.sub(r"(?m)^\s*%.*$", "", tex)

    hardcoded, unused = [], []
    for name, value in sorted(macros.items()):
        bare = value.replace("{,}", ",")
        used = f"\\{name}" in body
        # A value typed out where a macro exists will not follow the next measurement.
        # Short values are exempt: a bare "4" matches a section number or a table cell in
        # any paper, and a check that fires on those is one a reader learns to ignore.
        literal = len(bare) >= 4 and (bare in body or (value != bare and value in body))
        if literal:
            hardcoded.append((name, bare))
        elif not used:
            unused.append(name)

    if hardcoded:
        print("wpisane dosłownie, choć istnieje makro:")
        for name, value in hardcoded:
            print(f"  {value:<22} -> \\{name}")
    else:
        print("brak wartości wpisanych dosłownie")
    if unused:
        print(f"\nmakra zdefiniowane, lecz nieużyte ({len(unused)}): "
              f"{', '.join(unused)}")
    raise SystemExit(1 if hardcoded else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="metrics/eval_by_source_2026-08-17_fullspan.json",
                    help="the ladder evaluation report the results come from")
    ap.add_argument("--out", default="numbers.tex")
    ap.add_argument("--check", action="store_true",
                    help="report values typed into main.tex instead of cited as macros")
    args = ap.parse_args()
    (cmd_check if args.check else cmd_write)(args)


if __name__ == "__main__":
    main()
