# Releasing the model: what actually protects you

Notes gathered 2026-07-27 while deciding how to publish weights that reproduce
nineteenth-century prejudice by design. Everything below about other projects was read
from their repositories and model cards, not recalled.

**This is not legal advice.** It is a record of what comparable projects do and where our
situation differs from theirs. Before any public, interactive deployment under Polish
jurisdiction, ask a technology lawyer.

## The structural point

Protection does not come from peer review. Neither Bielik nor PLLuM is peer-reviewed —
both are arXiv technical reports — and both are nonetheless covered. What covers them is
three things, none of which has anything to do with publication venue:

1. the **licence**, including its warranty disclaimer,
2. an explicit **limitations and bias statement on the model card**,
3. a stated **restriction on how the model may be deployed**.

The preprint is a technical report. The protective language lives in the model card. Get
that ordering right: a beautiful ethics section in the paper and a bare model card is
backwards.

## What the comparable projects actually do

### Bielik 11B v2 (SpeakLeash) — the directly transferable precedent

Polish, open weights, same jurisdiction, community-run rather than state-funded.

- Licence: **"Apache 2.0 and Terms of Use"** — not plain Apache; additional terms attached.
- Explicit *Limitations and Biases* section acknowledging the model can produce "lewd,
  false, biased or otherwise offensive outputs" despite data-cleaning efforts.
- The load-bearing sentence: **"not intended for deployment without fine-tuning. It should
  not be used for human-facing interactions without further guardrails and user consent."**
- No entity accepts responsibility for generated output.

Note where that line falls: weights may be released; putting them in front of a user
without guardrails may not. They arrived independently at the same boundary we did.

### PLLuM (six-institution state consortium) — the upper bound

Not copyable by one person, but it shows what the ceiling looks like: a Responsible AI
framework with strict data governance; a hybrid output-correction and safety-filtering
module; rule-based controls plus ML classifiers for moderation; prompt-level moderation and
reprompting for sensitive content; anonymisation modules aligned to GDPR and the AI Act;
explicit design for EU AI Act compliance with transparency and auditability; and safety
covered in a multilayered evaluation framework including LLM-as-a-judge.

### TimeCapsuleLLM — the closest project, and no help here

MIT licence on repo and weights. Publishes a genuinely good **quantified bias report**
(pronoun, geographic, temporal bias, with a separate v2 report) — worth copying as method.
But there is **no content warning** on the repository or on the 1.2B model card, and no
Bias/Risks/Limitations section in the Hugging Face sense. The framing is positive
throughout ("unfiltered historical mirror", "reducing modern bias").

So on the specific question of how to handle offensive period output, the nearest prior art
has not solved it — it has not raised it. Its author is a US student (Muhlenberg College);
US law gives latitude that Polish law does not.

## Why our statement cannot be a copy of theirs

Bielik and PLLuM both say some version of *"despite our cleaning efforts, offensive content
may slip through."* We cannot say that, and should not try.

For them offensive output is a **leak past a filter** — a defect. For us it is **fidelity to
the distribution** — the property the artifact exists to have. Filtering nineteenth-century
prejudice out would destroy exactly what makes the model useful for studying the period.

Our statement therefore has to be different in kind, not a weaker version of theirs: not
"despite our efforts" but "deliberately, for these reasons, and here is why filtering would
invalidate the result." That is harder to write and stronger when written well. A draft:

> This model was trained exclusively on Polish texts published before 1918 and reproduces
> the worldview of that period, including antisemitic, nationalist, colonial and misogynist
> discourse that was ordinary in its sources. This is not a defect of data cleaning. A
> temporally bounded model is only useful for studying a period if it represents that
> period faithfully, and suppressing its prejudices would substitute our judgement for the
> historical record the model is meant to expose. The model is released for research and
> digital-humanities use. It is not suitable for human-facing deployment without a
> filtering layer and explicit framing, and it is not an authority on any subject.

The claim "we reproduce the period faithfully, deliberately" is only credible with numbers
behind it. That is what makes the quantified bias audit due diligence rather than an
appendix — and, run against a modern Polish model on the same probes, a result in its own
right.

## Jurisdiction

- **US vs Poland is a real difference.** Publishing a model capable of offensive output is
  not in itself unlawful in the US. Polish criminal law has **art. 256 and 257 KK**
  (incitement to hatred on national, ethnic, racial or religious grounds; insulting a group
  of the population), with no US equivalent. Whether an interactive model generating such a
  sentence on demand meets the statutory elements is untested — untested is not the same as
  safe.
- **EU AI Act**: free and open-source general-purpose models get partial exemptions, but the
  exemption is conditional and transparency obligations (training-data summary, copyright
  policy) remain. The systemic-risk threshold (10^25 FLOPs) is orders of magnitude above
  anything here — the 349M run is nowhere near it.
- **GDPR**: essentially not engaged. Pre-1918 texts, no living data subjects.

## Checklist before releasing anything

- [ ] Licence chosen, with terms of use attached if the licence alone is too permissive
      (Bielik's "Apache 2.0 and Terms of Use" is the pattern).
- [ ] Model card carries an explicit content and ethics statement — the *deliberate*
      version above, not the *despite-our-efforts* version.
- [ ] Intended use stated: research and digital humanities.
- [ ] Deployment restriction stated: no human-facing use without guardrails.
- [ ] Quantified bias audit published alongside, with its method and term list.
- [ ] Repository gated on Hugging Face so acceptance is recorded.
- [ ] Any public demo carries visible framing and a filtering layer.
- [ ] Any instruction-tuned variant treated as a separate artifact with its own review.
- [ ] Lawyer consulted before public interactive deployment.

## Sources

- Bielik 11B v2: <https://huggingface.co/speakleash/Bielik-11B-v2>, arXiv:2505.02410
- PLLuM: arXiv:2511.03823
- TimeCapsuleLLM: <https://github.com/haykgrigo3/TimeCapsuleLLM>,
  <https://huggingface.co/haykgrigorian/TimeCapsuleLLM-London-1800-1875-v2-1.2B>
