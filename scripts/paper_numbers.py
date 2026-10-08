"""Generate the paper's numeric macros from the measurement files, and catch hand-typing.

The LaTeX source is not published, so a reader cannot rebuild the document and find a
transcription error the way they could find a code bug. The only way a wrong number gets
caught from outside is if someone recomputes it from the released metrics and notices the
disagreement. That makes the guard against mistyping entirely our problem, and it has
already failed once: the exponent's interval $[0.192, 0.204]$ appears four times in the
paper and in no file at all, the committed report carries the pre-fix $[0.060, 0.351]$,
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
import math
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
BIAS_SHEET = "metrics/bias_sheet_349m_2026-08-19.json"
BIAS_SCREEN_MID = "metrics/bias_screen_107m_2026-08-19.json"
BIAS_RATES = "metrics/bias_rates_349m_2026-08-19.json"
BIAS_PROMPTS = "metrics/bias_prompts_2026-08-17.json"
BIAS_JUDGES = "metrics/bias_adjudicator_agreement_run2.json"
FERTILITY = "metrics/tokenizer_fertility_2026-08-18.json"
LEDGER_VAL = "metrics/provenance_ledger_2026-08-03_val.csv.gz"
CORRUPTION = "metrics/corruption_per_document_2026-10-05.json"
WINDOW_FERTILITY = "metrics/window_fertility_2026-08-17.json"
COMPOSITION = "metrics/corpus_composition_2026-10-06.json"
CLEANING = "metrics/cleaning_effect_2026-10-06.json"
WL_TRANSLATIONS = "metrics/wl_translations_2026-10-06.json"
DOCUMENT_METADATA = "ledger/document_metadata_2026-10-06.json"
DEATH_YEARS = "metrics/creator_death_years_2026-10-06.json"
COMPARATOR_HELDOUT = {"bielik": "metrics/comparator_heldout_bielik_2026-10-06.json",
                      "papuga": "metrics/comparator_heldout_papugapt2_2026-10-06.json"}
PROBE_LOGS = {
    "small": "metrics/wieszcz_47m_6b7_2026-08-05_s1337_val.csv",
    "mid": "metrics/wieszcz_107m_6b7_2026-08-06_s1337_val.csv",
    "large": "metrics/wieszcz_349m_6b7_2026-08-07_s1337_val.csv",
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


SMALL_NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine")


def tex_word(n: int) -> str:
    """Counts below ten are spelled out in running text, as APA style asks."""
    if not 0 <= n < 10:
        raise SystemExit(f"{n} is not a count the prose spells out")
    return SMALL_NUMBERS[n]


def tex_ci(v, places: int = 3) -> str:
    lo, hi = v
    return f"[{float(lo):.{places}f}, {float(hi):.{places}f}]"


def tokenizer_macros(out: dict[str, str], ev: dict, tok: dict) -> None:
    """The vocabulary trade, and the conversion that makes the ladder vocabulary-free.

    Bits per byte is derived from the exact held-out totals rather than from the sampled
    fertility figure: the evaluation now spans the whole split, so the right divisor is
    every byte of it over every token of it, and the sample would introduce an error into
    a conversion that has no need of one.
    """
    import csv
    import gzip

    fert = json.loads((REPO / FERTILITY).read_text())
    bpt = lambda block: block["fertility"]["all"]["bytes_per_token"]

    out["fertdocs"] = tex_int(fert["meta"]["documents"])
    out["fertcapkb"] = tex_int(fert["meta"]["doc_bytes_cap"] // 1000)
    out["fertilityours"] = tex_f(bpt(fert["tokenizers"]["wieszcz-8k"]), 3)
    out["fertilitybielik"] = tex_f(bpt(fert["tokenizers"]["bielik-1.5b-v3"]), 3)
    out["fertilitypapuga"] = tex_f(bpt(fert["tokenizers"]["papugapt2"]), 3)
    named = {"4000": "four", "16000": "sixteen", "32000": "thirtytwo"}
    for size, word in named.items():
        cf = fert["counterfactual_vocabularies"][size]
        out[f"fertility{word}"] = tex_f(bpt(cf), 3)

    shares = fert["embedding_shares"]
    for rung, tag in (("47M", "small"), ("349M", "large")):
        for size, word in (("8000", "eight"), ("32000", "thirtytwo")):
            share = shares[rung]["by_vocab"][size]["embedding_share"]
            out[f"embed{tag}{word}"] = tex_f(100 * share, 1)

    with gzip.open(REPO / LEDGER_VAL, "rt", encoding="utf-8") as f:
        val_bytes = sum(int(row["bytes"]) for row in csv.DictReader(f))
    exact = val_bytes / dig(tok, "val.tokens")
    out["valbytes"] = tex_int(val_bytes)
    out["fertilityexact"] = tex_f(exact, 3)
    factor = 1.0 / (exact * math.log(2))
    out["natstobpb"] = tex_f(factor, 4)
    for rung, tag in LADDER:
        out[f"bpb{tag}"] = tex_f(
            dig(ev, f"by_subset.full.{rung}.cross_entropy_nats") * factor, 3)

    # The protocol scores windows of the split, not all of it; the paper states how far
    # their own ratio sits from the split's and what that does to the converted figures.
    win = json.loads((REPO / WINDOW_FERTILITY).read_text())
    out["fertilitywindows"] = tex_f(win["scored_windows"]["bytes_per_token"], 3)
    shift = max(abs(dig(ev, f"by_subset.full.{rung}.cross_entropy_nats") * factor
                    - win["bpb"][rung]["bpb_windows"]) for rung, _ in LADDER)
    out["bpbshift"] = tex_f(math.ceil(shift * 1e4) / 1e4, 4)

    # What the corpus would have been under the vocabulary we did not pick, which is the
    # other half of the trade and the only form in which a token count is comparable.
    corpus_bytes = bpt(fert["tokenizers"]["wieszcz-8k"]) * dig(tok, "tokens_total")
    out["corpustokensthirtytwo"] = tex_f(
        corpus_bytes / bpt(fert["counterfactual_vocabularies"]["32000"]) / 1e9, 2)
    out["corpustokensbillions"] = tex_f(dig(tok, "tokens_total") / 1e9, 2)


def bias_macros(out: dict[str, str]) -> None:
    """The prevalence screen's shape, which the ethics section describes and must not retype.

    The counts are checked against each other rather than read one by one: a screen whose
    flagged and unflagged arms do not sum to the generations it saw, or a sheet that is not
    the flagged set plus its audit sample, would be describing a different measurement than
    the one the files record.
    """
    sheet = json.loads((REPO / BIAS_SHEET).read_text())
    (bf,) = sheet["by_file"].values()
    prompts = json.loads((REPO / BIAS_PROMPTS).read_text())
    items = prompts.get("prompts") or prompts["items"]
    neutral = sum(1 for x in items if x.get("neutral"))

    if bf["flagged"] + bf["unflagged"] != bf["generations"]:
        raise SystemExit("bias screen: flagged + unflagged disagrees with generations")
    if bf["flagged"] + bf["audit_drawn"] != len(sheet["rows"]):
        raise SystemExit("bias sheet is not the flagged set plus its audit sample")
    if neutral != bf["generations"]:
        raise SystemExit("neutral prompt count disagrees with the generations produced")

    out["biasprompts"] = tex_int(neutral)
    out["biastriggered"] = tex_int(len(items) - neutral)
    out["biasflagged"] = tex_int(bf["flagged"])
    out["biaslexicon"] = tex_int(bf["flagged_by_lexicon"])
    out["biasclosure"] = tex_int(bf["flagged_by_closure"])
    out["biasadjdeclined"] = tex_int(bf["adjective_declined"])

    # The middle rung is screened but not adjudicated, and the screen is free: it says
    # whether the tendency to name a group at all moves with size, which is the part of
    # the question that does not need a reader.
    (bm,) = json.loads((REPO / BIAS_SCREEN_MID).read_text())["by_file"].values()
    if bm["generations"] != bf["generations"]:
        raise SystemExit("the two rungs were screened over different generation counts")
    out["biasmentionrate"] = tex_f(100 * bf["mention_rate"], 2)
    out["biasmentionratemid"] = tex_f(100 * bm["mention_rate"], 2)
    for name, block in (("biasmentionrateci", bf), ("biasmentionratemidci", bm)):
        rate, n = block["mention_rate"], block["generations"]
        half = 1.96 * math.sqrt(rate * (1 - rate) / n)
        out[name] = tex_ci([100 * (rate - half), 100 * (rate + half)], 1)

    # The adjudication as far as it has been read. Emitted from the scored file rather than
    # from the sheet, so the paper cannot describe a state the rate was not computed in:
    # the read counts are the denominators of every interval beside them.
    (br,) = json.loads((REPO / BIAS_RATES).read_text())["by_file"].values()
    if br["mentions"] != bf["flagged"]:
        raise SystemExit("the scored file and the sheet disagree on the flagged count")
    out["biasread"] = tex_int(br["flagged_read"])
    out["biasprejudiced"] = tex_int(br["flagged_prejudiced"])
    out["biasamong"] = tex_f(100 * br["rate_among_mentions"], 1)
    out["biasamongci"] = tex_ci([100 * x for x in br["rate_among_mentions_ci95"]], 1)
    out["biasauditread"] = tex_int(br["audit_read"])
    out["biasauditprejudiced"] = tex_int(br["audit_prejudiced"])
    out["biasunclear"] = tex_int(br["unclear"])
    out["biasunclearrate"] = tex_f(100 * br["unclear_rate"], 1)

    # How much of the frozen prompt draw the closure would have rejected, had it existed
    # when the draw was made. Reported rather than repaired, because the generations are
    # spent and a redraw would change what the neutral arm means.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from screen_bias import CLOSURE
    out["biasclosureprompts"] = tex_int(
        sum(1 for x in items if x.get("neutral") and CLOSURE.search(x["prompt"])))
    out["biasunflagged"] = tex_int(bf["unflagged"])
    out["biasaudit"] = tex_int(bf["audit_drawn"])
    out["biassheet"] = tex_int(len(sheet["rows"]))
    n_judges = len(json.loads((REPO / BIAS_JUDGES).read_text()))
    out["biasjudges"] = tex_int(n_judges)
    out["biasjudgesword"] = tex_word(n_judges)
    out["biasjudgesWord"] = tex_word(n_judges).capitalize()
    judge_macros(out)


def judge_macros(out: dict[str, str]) -> None:
    """What the model judges disagree about, and what they agree to follow.

    Two runs of the same five judges over the same sheet. Between them the written
    criterion was amended once (git 09f1ad4) to stop truncation counting as *unclear*,
    and four of the five saw the amendment while one did not, so the pair is a spread
    measurement with a control rather than a test-retest. Both arms are read here because
    the paper's claim rests on the contrast between them: the judges follow the criterion
    on the dimension it addressed and still disagree six-fold on the prejudice rate.

    The per-judge totals are checked to be the same set of items, since a spread computed
    over different denominators would not be a spread.
    """
    runs = {r: json.loads((REPO / f"metrics/bias_adjudicator_agreement_{r}.json").read_text())
            for r in ("prelim", "run2")}

    def counts(entry: dict) -> tuple[int, int, int]:
        vc = entry["verdict_counts"]
        return sum(vc.values()), vc.get("prejudiced", 0), vc.get("unclear", 0)

    totals = {m: counts(e)[0] for m, e in runs["run2"].items()}
    if len(set(totals.values())) != 1:
        raise SystemExit(f"judges scored different numbers of items: {totals}")
    out["judgeitems"] = tex_int(next(iter(totals.values())))

    rates = {m: 100 * counts(e)[1] / counts(e)[0] for m, e in runs["run2"].items()}
    lo, hi = min(rates.values()), max(rates.values())
    out["judgeratelo"] = tex_f(lo, 1)
    out["judgeratehi"] = tex_f(hi, 1)
    out["judgeratespread"] = tex_f(hi / lo, 1)

    # The criterion amendment cut *unclear* for every judge that saw it, and the one
    # judge held at the old criterion moved the other way. That asymmetry is the control.
    crit = {m: json.loads((REPO / runs[r][m]["file"]).read_text())["guidelines_sha256"][:8]
            for r in ("prelim", "run2") for m in runs[r]}
    moved = [m for m in runs["run2"]
             if json.loads((REPO / runs["prelim"][m]["file"]).read_text())["guidelines_sha256"][:8]
             != json.loads((REPO / runs["run2"][m]["file"]).read_text())["guidelines_sha256"][:8]]
    held = [m for m in runs["run2"] if m not in moved]
    if len(held) != 1:
        raise SystemExit(f"expected exactly one judge at an unchanged criterion, got {held}")
    out["judgeamended"] = tex_int(len(moved))
    out["judgeamendedword"] = tex_word(len(moved))
    out["judgeunclearbefore"] = tex_int(sum(counts(runs["prelim"][m])[2] for m in moved))
    out["judgeunclearafter"] = tex_int(sum(counts(runs["run2"][m])[2] for m in moved))
    out["judgecontrolbefore"] = tex_int(counts(runs["prelim"][held[0]])[2])
    out["judgecontrolafter"] = tex_int(counts(runs["run2"][held[0]])[2])

    # The largest move in any judge's prejudice rate across the amendment, which is what
    # bounds how much of the spread the criterion could be responsible for.
    shifts = {m: abs(100 * counts(runs["run2"][m])[1] / counts(runs["run2"][m])[0]
                     - 100 * counts(runs["prelim"][m])[1] / counts(runs["prelim"][m])[0])
              for m in runs["run2"]}
    out["judgeratedrift"] = tex_f(max(shifts.values()), 1)

    # The human overlap, which is why no agreement coefficient is quoted: the positive
    # class is three items.
    cm = next(iter(runs["run2"].values()))["confusion_human|model"]
    human: dict[str, int] = {}
    for key, n in cm.items():
        human[key.split("|")[0]] = human.get(key.split("|")[0], 0) + n
    out["judgeoverlap"] = tex_int(sum(human.values()))
    out["judgeoverlappos"] = tex_int(human.get("prejudiced", 0))
    kappas = [e["cohen_kappa"] for e in runs["run2"].values()]
    out["judgekappalo"] = tex_f(min(kappas), 2)
    out["judgekappahi"] = tex_f(max(kappas), 2)


def positional_split_macros(out: dict[str, str]) -> None:
    """Source composition of the positional 1% tail, computed from the ledger.

    Byte shares: the ledger records bytes per document and tokens only in aggregate.
    """
    demo = json.loads((REPO / "metrics/positional_split_demo.json").read_text())
    window, coverage = demo["held_out_window"], demo["coverage"]

    for source, key in (("internet_archive", "iadocs"), ("wolne_lektury", "wldocs")):
        out[key] = tex_int(demo["corpus"]["by_source"][source]["documents"])

    out["posplitwindocs"] = tex_int(window["documents"])
    out["posplitwlheld"] = tex_int(window["by_source"]["wolne_lektury"]["documents"])
    out["posplitwlshare"] = tex_f(100 * coverage["wolne_lektury"]["share_of_window"], 1)
    out["posplitwlcovered"] = tex_f(100 * coverage["wolne_lektury"]["share_of_source_held_out"], 0)
    out["posplitiaheld"] = tex_int(window["by_source"]["internet_archive"]["documents"])
    out["posplitiacovered"] = tex_f(100 * coverage["internet_archive"]["share_of_source_held_out"], 1)


def epochs_macros(out: dict[str, str], ev: dict) -> None:
    """The epochs table: what the second pass buys, at every rung, from one instrument.

    The finals come from the ladder report and the epoch/pre-decay readings from their own
    files, and the two are only combinable because all were produced by the repaired
    harness, the table carried pre-repair readings once, covering 89% of the split and
    no Wolne Lektury window, and the mismatch surfaced as a final column disagreeing with
    every other section by fifteen thousandths of a nat. Each file's span is checked here
    so a pre-repair report cannot slip back in.
    """
    for rung, tag in LADDER:
        final = dig(ev, f"by_subset.full.{rung}.cross_entropy_nats")
        ces = {"final": final}
        for ck in ("epoch1", "predecay"):
            rep = json.loads(
                (REPO / f"metrics/eval_{rung.lower()}_{ck}_2026-08-20.json").read_text())
            if rep.get("span_fraction", 0) < 0.999:
                raise SystemExit(f"eval_{rung.lower()}_{ck} spans "
                                 f"{rep.get('span_fraction')} of the split; "
                                 f"that is the pre-repair harness")
            ces[ck] = rep["cross_entropy_nats"]
        out[f"ce{tag}epochone"] = tex_f(ces["epoch1"], 3)
        out[f"ce{tag}predecay"] = tex_f(ces["predecay"], 3)
        out[f"gain{tag}secondepoch"] = tex_f(ces["epoch1"] - ces["predecay"], 3)
        out[f"gain{tag}decay"] = tex_f(ces["predecay"] - ces["final"], 3)
        out[f"gain{tag}both"] = tex_f(ces["epoch1"] - ces["final"], 3)


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

    # A share lies in [0, 1]; the probe's normal-approximation interval can step outside.
    interval = lambda block: tex_ci(
        [min(1.0, max(0.0, v)) for v in block["orto_modern_share"]["ci95"]], 2)
    for tag, path in PROBE_RUNGS.items():
        ev = _load_probe(path)
        out[f"orto{tag}ci"] = interval(ev["generations_wieszcz"])
        out[f"orto{tag}fewshotci"] = interval(ev["generations_wieszcz_fewshot"])
    for tag, path in PROBE_COMPARATORS.items():
        out[f"orto{tag}fewshotci"] = interval(
            _load_probe(path)["generations_modern_lm_fewshot"])

    ev = _load_probe(PROBE_COMPARATORS["bielik"])
    out["ortocorpus"] = share(ev["corpus_reference"])
    out["ortocorpusci"] = interval(ev["corpus_reference"])
    fs = ev["fewshot"]
    out["fewshotpassages"] = tex_int(len(fs["passages"]))
    out["fewshotchars"] = tex_int(fs["fit_wieszcz"]["chars"])
    out["fewshotshare"] = tex_f(fs["fit_wieszcz"]["scan"]["orto_modern_share"], 3)


def exclusion_macros(out: dict[str, str]) -> None:
    """The excluded set's byte size and its share of the freeze.

    Both were literals in the prose with nothing behind them. The byte total comes from
    `scripts/exclusion_bytes.py`.
    """
    exc = json.loads((REPO / "metrics/exclusion_bytes_2026-08-03.json").read_text())
    out["exclbytes"] = tex_int(exc["bytes"])
    freeze_bytes = 23.35e9
    out["exclshare"] = tex_f(100 * exc["bytes"] / freeze_bytes, 2)


def source_token_macros(out: dict[str, str]) -> None:
    """The corpus table's token column, split by source.

    The tokenization report records the train and held-out streams, not the two sources,
    so Table 1's per-source tokens were literals with nothing behind them. They are
    recovered by `scripts/corpus_source_tokens.py`, which locates the `ia_`/`wl_` boundary
    by counting document terminators, and are checked here against the corpus total.
    """
    c = json.loads((REPO / "metrics/corpus_source_tokens_2026-08-03.json").read_text())["corpus"]
    if c["ia_tokens"] + c["wl_tokens"] != c["tokens"]:
        raise SystemExit("the per-source token counts do not sum to the corpus total")
    out["iatokens"] = tex_int(c["ia_tokens"])
    out["wltokens"] = tex_int(c["wl_tokens"])
    out["wlsharetokens"] = tex_f(c["wl_share_pct"], 2)


def unfiltered_era_macros(out: dict[str, str]) -> None:
    """The model's own era gap, re-measured on modern terms the corpus filter never matched.

    Every term in the probe's default modern set is a string the exclusion rule deleted
    documents on, so a model trained on the filtered corpus cannot have met them and the
    modern half partly reads the rule back. This arm scores fourteen post-1918 terms that
    were never filter strings against the same period set in the same carriers.

    These are the 349M's own modern-minus-period gaps, not interactions: no comparator was
    run on the unfiltered terms. The intervals emitted here belong to the modern *levels*
    they are printed beside, since the report stores no interval on the difference.
    """
    filt = _load_probe(PROBE_COMPARATORS["bielik"])["contrast_wieszcz"]
    unfi = _load_probe("metrics/temporal_probe_349m_unfiltered.json")["contrast_wieszcz"]
    ident = lambda rows: [(r["term"], r["carrier"]) for r in rows]
    if ident(filt["period"]["terms"]) != ident(unfi["period"]["terms"]):
        raise SystemExit("the two era arms used different period terms, so their gaps "
                         "are not comparable")
    shared = set(ident(filt["modern"]["terms"])) & set(ident(unfi["modern"]["terms"]))
    if shared:
        raise SystemExit(f"the two modern sets overlap, so the arms are not independent: {shared}")
    # The arms ran on different machines. The period half is the control: identical terms,
    # identical carriers, so any drift would show here first.
    pf, pu = filt["period"]["summary"]["mean"], unfi["period"]["summary"]["mean"]
    if abs(pf - pu) > 5e-4:
        raise SystemExit(f"the period half did not reproduce across the two runs "
                         f"({pf:.6f} vs {pu:.6f}), so the arms are not comparable")
    out["eragapfiltered"] = tex_f(filt["modern_minus_period_bpb"], 3)
    out["eragapunfiltered"] = tex_f(unfi["modern_minus_period_bpb"], 3)
    out["eramodernfiltered"] = tex_f(filt["modern"]["summary"]["mean"], 3)
    out["eramodernunfiltered"] = tex_f(unfi["modern"]["summary"]["mean"], 3)
    out["eramodernfilteredci"] = tex_ci(filt["modern"]["summary"]["ci95"], 2)
    out["eramodernunfilteredci"] = tex_ci(unfi["modern"]["summary"]["ci95"], 2)
    out["eraperiodboth"] = tex_f(pf, 3)
    out["eraunfilteredterms"] = tex_int(len(unfi["modern"]["terms"]))


def probe_exponent_macros(out: dict[str, str], ev: dict) -> None:
    """The exponent the twenty-window in-training probe reads on the same checkpoints.

    Fitted exactly through the final probe row of each rung's training log, with the
    parameter counts the dense report carries, so the instrument comparison the paper
    draws rests on released files. The three logs must end at the same step, or the
    fit would compare checkpoints from different points of the schedule.
    """
    import csv
    from eval_by_source import fit_alpha

    ns, ls, steps = [], [], set()
    for rung, tag in LADDER:
        rows = list(csv.DictReader((REPO / PROBE_LOGS[tag]).open()))
        last = rows[-1]
        steps.add(int(last["step"]))
        ns.append(float(dig(ev, f"by_subset.full.{rung}.params")))
        ls.append(float(last["val_loss"]))
    if len(steps) != 1:
        raise SystemExit(f"the training logs end at different steps: {sorted(steps)}")
    fit = fit_alpha(ns, ls)
    if fit.get("alpha") is None:
        raise SystemExit("the in-training probe losses are not power-law separable")
    out["alphaprobe"] = tex_f(fit["alpha"], 3)


def corruption_population_macros(out: dict[str, str]) -> None:
    """Table 4: the four detectors over every released document, and the cost of two
    document thresholds.

    The shares in the report are of Internet Archive bytes kept, so the macro states what
    a threshold removes. Per-document percentiles are over documents long enough to carry
    a rate, which the report counts.
    """
    rep = json.loads((REPO / CORRUPTION).read_text())
    ia, wl = rep["by_source"]["ia"], rep["by_source"]["wl"]
    out["corrpopia"] = tex_f(ia["rate_pct"], 2)
    out["corrpopwl"] = tex_f(wl["rate_pct"], 2)
    out["corrpoptrue"] = tex_f(rep["ia_corruption_above_floor_pct"], 2)
    out["corrpopiawords"] = tex_f(ia["words"] / 1e9, 2) + "B"
    out["corrpopiasusp"] = tex_f(ia["suspicious"] / 1e6, 1) + "M"
    out["corrpopwlwords"] = tex_f(wl["words"] / 1e6, 1) + "M"
    out["corpuswordsbillions"] = tex_f((ia["words"] + wl["words"]) / 1e9, 1)
    out["corrpopwlsusp"] = tex_f(wl["suspicious"] / 1e3, 1) + "k"
    for reason, name in (("symbol", "symbol"), ("midcaps", "midcaps"),
                         ("digit_mix", "digit"), ("no_vowel", "novowel")):
        out[f"corrpop{name}"] = tex_f(ia["by_reason_pct"][reason], 2)
    out["corrpopscored"] = tex_int(ia["per_document_pct"]["documents_scored"])
    out["corrpopminwords"] = tex_int(rep["meta"]["min_words_for_rate"])
    out["corrpopmedian"] = tex_f(ia["per_document_pct"]["median"], 2)
    out["corrpopninety"] = tex_f(ia["per_document_pct"]["p90"], 2)
    out["corrpopworst"] = tex_f(ia["per_document_pct"]["max"], 2)
    out["corrpopwlmedian"] = tex_f(wl["per_document_pct"]["median"], 2)
    out["corrpopwlworst"] = tex_f(wl["per_document_pct"]["max"], 2)
    if wl["per_document_pct"]["max"] <= ia["per_document_pct"]["max"]:
        raise SystemExit("the clean source's worst document no longer exceeds the OCR'd "
                         "source's, so the paper's control argument is false")
    if ia["per_document_pct"]["p99"] <= 5:
        raise SystemExit("the 99th percentile sits at or below 5%, so the paper's claim "
                         "that a 5% threshold removes the whole tail beyond it is false")
    for thr, name in (("5", "corrdropfive"), ("3", "corrdropthree")):
        kept = ia["kept_at_threshold_pct"][thr]["bytes_kept_share"]
        out[name] = tex_f(100 * (1 - kept), 1)


def cleaning_macros(out: dict[str, str]) -> None:
    """What fetch-time cleaning removed, on the documents whose as-fetched text was kept."""
    rep = json.loads((REPO / CLEANING).read_text())
    out["cleanrawshare"] = tex_f(rep["share_documents_with_raw"], 1)
    out["cleanremoved"] = tex_f(rep["removed_share_bytes"], 1)
    out["cleanlowalphalines"] = tex_f(rep["low_alpha_share_lines"], 1)
    out["cleanlowalphabytes"] = tex_f(rep["low_alpha_share_bytes"], 1)
    out["cleanjunkbytes"] = tex_f(rep["junk_share_bytes"], 1)
    out["cleanwhitespacebytes"] = tex_f(rep["whitespace_share_bytes"], 1)
    out["cleanreplaymatch"] = tex_f(rep["replay_equals_clean_share_documents"], 1)
    out["cleandigits"] = tex_f(rep["low_alpha_share_digits"], 0)
    out["cleanyears"] = tex_f(rep["low_alpha_share_years_post1918"], 0)


def wl_macros(out: dict[str, str]) -> None:
    """What the Wolne Lektury records say about translations and later text."""
    rep = json.loads((REPO / WL_TRANSLATIONS).read_text())
    summary = rep["summary"]
    for name, key in (("wltranslations", "translation"),
                      ("wltranslatorlate", "translation_translator_alive_after_1918"),
                      ("wlauthorlate", "original_author_alive_after_1918"),
                      ("wlfreelicence", "free_licence"),
                      ("wlcertain", "certain_post1918"),
                      ("wlpossible", "possibly_post1918"),
                      ("wlnotpolish", "not_polish")):
        out[name] = tex_int(summary[key]["documents"])
        out[f"{name}share"] = tex_f(summary[key]["share_corpus_bytes"], 2)
    out["wlcertainproust"] = tex_int(summary["certain_post1918"]["documents"]
                                     - summary["free_licence"]["documents"])
    out["wlpossiblewlshare"] = tex_f(summary["possibly_post1918"]["share_wl_bytes"], 0)


def rights_macros(out: dict[str, str]) -> None:
    """Rights basis of the Internet Archive documents beyond the library statement."""
    basis = json.loads((REPO / DOCUMENT_METADATA).read_text())["by_rights_basis"]
    ia = ("library_public_domain_statement", "scanning_institution_assertion",
          "other_statement", "none")
    ia_bytes = sum(basis[k]["bytes"] for k in ia)
    for name, key in (("rightsscanner", "scanning_institution_assertion"),
                      ("rightsnone", "none")):
        out[f"{name}docs"] = tex_int(basis[key]["documents"])
        out[f"{name}bytes"] = tex_f(100 * basis[key]["bytes"] / ia_bytes, 1)
    deaths = json.loads((REPO / DEATH_YEARS).read_text())
    out["rightslatecreatordocs"] = tex_int(deaths["counts"]["creator_died_in_term"])
    out["rightslatecreatorperiodical"] = tex_int(
        deaths["by_type"]["periodical"]["creator_died_in_term"])
    out["rightslatecreatoryear"] = str(deaths["meta"]["first_protected_death_year"])


def comparator_heldout_macros(out: dict[str, str]) -> None:
    """Bits per byte of the comparators and the rungs on the dense protocol's windows."""
    reports = {tag: json.loads((REPO / path).read_text())["by_subset"]
               for tag, path in COMPARATOR_HELDOUT.items()}
    interval = lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]"  # noqa: E731
    for subset in ("full", "ia", "wl"):
        for tag, rep in reports.items():
            out[f"cmpbpb{tag}{subset}"] = tex_f(rep[subset]["bits_per_byte"], 3)
        for rung, tag in LADDER:
            out[f"cmpbpb{tag}{subset}"] = tex_f(
                reports["bielik"][subset]["rungs"][rung]["bits_per_byte"], 3)
    out["cmpwindowswl"] = tex_int(reports["bielik"]["wl"]["windows"])
    import numpy as np
    windows = np.load(REPO / COMPARATOR_HELDOUT["bielik"].replace(".json", ".windows.npz"))
    block = json.loads((REPO / COMPARATOR_HELDOUT["bielik"]).read_text())["meta"]["block"]
    for subset in ("ia", "wl"):
        scored = windows[f"{subset}_rung_bytes"]
        out[f"fertilitywindows{subset}"] = tex_f(float(scored.sum()) / (len(scored) * block), 2)
    for name, tag, rung in (("cmpdiffmatched", "papuga", "107M"),
                            ("cmpdiffbielik", "bielik", "349M"),
                            ("cmpdiffbieliksmall", "bielik", "47M")):
        row = reports[tag]["full"]["rungs"][rung]
        out[name] = tex_f(row["comparator_minus_rung"], 3)
        out[f"{name}ci"] = interval(row["comparator_minus_rung_ci95"])
    for tag in reports:
        row = reports[tag]["wl"]["rungs"]["349M"]
        out[f"cmpdiff{tag}wl"] = tex_f(row["comparator_minus_rung"], 3)
        out[f"cmpdiff{tag}wlci"] = interval(row["comparator_minus_rung_ci95"])


def crossover_macros(out: dict[str, str]) -> None:
    """The levels behind the era contrast, and the paired difference on period terms.

    The interaction says our gap is larger than a comparator's. That a model is better on
    one set and worse on the other needs the levels: per term, our score minus the
    comparator's, averaged over the period set, with the interval the probe uses for the
    interaction (mean plus or minus 1.96 standard errors over the terms).
    """
    pairs = {"bielik": PROBE_COMPARATORS["bielik"], "papuga": PROBE_COMPARATORS["papuga"],
             "matched": "metrics/temporal_probe_107m_papugapt2_2026-08-18.json"}
    for tag, path in pairs.items():
        era = _load_probe(path)["era_interaction"]["per_era"]
        for which in ("period", "modern"):
            terms = era[which]["terms"]
            n = len(terms)
            deltas = [t["delta_bpb"] for t in terms]
            mean = sum(deltas) / n
            sem = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1) / n)
            out[f"era{which}{tag}"] = tex_f(sum(t["modern_lm_bpb"] for t in terms) / n, 3)
            out[f"era{which}delta{tag}"] = tex_f(mean, 2)
            out[f"era{which}delta{tag}ci"] = tex_ci([mean - 1.96 * sem, mean + 1.96 * sem], 2)
            if which == "period" and mean + 1.96 * sem >= 0:
                raise SystemExit(f"the period-term difference against {tag} does not "
                                 "exclude zero, so the crossover claim fails for it")


def composition_macros(out: dict[str, str]) -> None:
    """The composition table: catalogue type and state of publication of the Internet
    Archive documents, the libraries' rights statements, and the post-reform spelling
    share by place. Shares are per cent of documents and of bytes. The per-decade series
    are drawn by plot_composition.py from the same report."""
    rep = json.loads((REPO / COMPOSITION).read_text())

    def pair(name: str, row: dict) -> None:
        out[name + "docs"] = tex_f(row["share_documents"], 1)
        out[name + "bytes"] = tex_f(row["share_bytes"], 1)

    for key, name in (("periodical", "periodical"), ("book", "book"),
                      ("ephemera", "ephemera"), ("unspecified", "unspec")):
        pair("comptype" + name, rep["by_type"][key])
    if set(rep["by_decade"]) != {str(d) for d in range(1800, 1911, 10)}:
        raise SystemExit("composition report has a decade outside 1800-1918 or undated rows")
    for key, name in (("austria_hungary", "austria"), ("russian_empire", "russia"),
                      ("german_empire", "germany"), ("elsewhere", "elsewhere"),
                      ("no_place", "noplace"), ("unclassified", "unclassified")):
        pair("compstate" + name, rep["by_state"][key])
    out["compyearagree"] = tex_f(rep["year_check"]["agreement"], 1)
    out["compyearcomparable"] = tex_int(rep["year_check"]["comparable"])
    pair("comprightspd", rep["by_rights"]["public_domain_statement"])

    orto = rep["orto"]
    out["ortocorpusall"] = tex_f(orto["all"]["share_modern"], 3)
    for key, name in (("austria_hungary", "austria"), ("russian_empire", "russia"),
                      ("german_empire", "germany")):
        out["ortostate" + name] = tex_f(orto["by_state"][key]["share_modern"], 3)
    out["ortomassaustria"] = tex_f(orto["modern_mass_by_state"]["austria_hungary"], 0)
    out["ortomassrussia"] = tex_f(orto["modern_mass_by_state"]["russian_empire"], 0)
    cells = orto["plotted_cells"]
    out["ortocellmindocs"] = tex_int(cells["fewest_documents"])
    out["ortocelltopmedian"] = tex_f(cells["median_top_title_share_forms"], 0)
    titles = orto["by_title"]
    out["ortotitlecount"] = tex_int(titles["titles"])
    out["ortotitleminforms"] = tex_int(titles["min_forms"])
    out["ortotitlelow"] = tex_f(titles["share_forms_in_titles_at_most_0.1"], 0)
    out["ortotitlehigh"] = tex_f(titles["share_forms_in_titles_at_least_0.9"], 0)
    for key, name in (("Lwów / Gazeta Lwowska", "gazetalwowska"),
                      ("Lwów / Dziennik Polski", "dziennikpolski"),
                      ("Warszawa / Kurjer Warszawski", "kurjerwarszawski"),
                      ("Kraków / Czas", "czas")):
        out["ortotitle" + name] = tex_f(titles["named"][key]["share_modern"], 2)


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

    positional_split_macros(out)
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
    probe_exponent_macros(out, ev)
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
    unfiltered_era_macros(out)
    source_token_macros(out)
    exclusion_macros(out)
    corruption_population_macros(out)
    composition_macros(out)
    crossover_macros(out)
    cleaning_macros(out)
    wl_macros(out)
    rights_macros(out)
    comparator_heldout_macros(out)
    tokenizer_macros(out, ev, tok)
    bias_macros(out)
    epochs_macros(out, ev)

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
        # The match is bounded by digits and separators on both sides, so a quoted figure
        # such as 700.000 does not read as the value 0.000.
        def typed(v: str) -> bool:
            return re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d.,])", body) is not None

        literal = len(bare) >= 4 and (typed(bare) or (value != bare and typed(value)))
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
