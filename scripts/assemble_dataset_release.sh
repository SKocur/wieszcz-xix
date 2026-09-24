#!/usr/bin/env bash
# Put the non-parquet half of the dataset release in place, so the directory can be
# uploaded verbatim.
#
# `build_hf_dataset.py` writes data/. Everything else a reader needs, the card, the
# licence, the provenance ledger and the exclusion rule, lives in this repository and is
# copied in from here rather than assembled by hand at upload time. Hand-copying is how the
# published card and the code that produced it drift apart, and the card is the one file
# nobody re-derives.
#
# The card is the source of truth for its own layout: the paths it promises a reader
# (data/, ledger/, exclusions/) are checked to exist before anything is declared ready.
#
#     bash scripts/assemble_dataset_release.sh ../dataset/polish-pre1918-corpus
set -euo pipefail

# Resolve DEST against the caller's cwd BEFORE moving to the repo root: a relative
# argument resolved after the cd points into the repository itself, and the installs
# below would overwrite the repo's README.md and LICENSE. It happened.
DEST=${1:?usage: assemble_dataset_release.sh <release-dir>}
DEST=$(cd "$DEST" 2>/dev/null && pwd) || { echo "no such directory: $1" >&2; exit 1; }
cd "$(dirname "$0")/.."
[[ $DEST == "$(pwd)" ]] && { echo "refusing: DEST is the repository root" >&2; exit 1; }

CARD=release/dataset-card.md
M=metrics

# Refuse to publish a card with a hole in it. Token count and parquet size are only known
# once the build has run, and a placeholder that reaches Hugging Face is a card that says
# {{TOKENS}} to every reader.
if grep -q '{{[A-Z]*}}' "$CARD"; then
  echo "$CARD still has unfilled placeholders:" >&2
  grep -n '{{[A-Z]*}}' "$CARD" >&2
  echo "run the build first, then fill them from what it printed" >&2
  exit 1
fi

install -m 644 "$CARD" "$DEST/README.md"
install -m 644 release/dataset-LICENSE "$DEST/LICENSE"

mkdir -p "$DEST/ledger" "$DEST/exclusions"
install -m 644 "$M"/provenance_ledger_2026-08-03_train.csv.gz \
               "$M"/provenance_ledger_2026-08-03_val.csv.gz \
               "$M"/provenance_ledger_2026-08-03.json "$DEST/ledger/"
install -m 644 "$M"/exclusions_2026-08-03.json "$DEST/exclusions/"

# Every path the card names has to exist, or the card is describing a release that is not
# the one on disk.
fail=0
for p in data ledger exclusions README.md LICENSE; do
  [[ -e $DEST/$p ]] || { echo "card promises $p, which is missing" >&2; fail=1; }
done
shards=$(find "$DEST/data" -name 'train-*.parquet' 2>/dev/null | wc -l | tr -d ' ')
[[ $shards -gt 0 ]] || { echo "no parquet shards in $DEST/data, run build_hf_dataset.py" >&2; fail=1; }
[[ $fail -eq 0 ]] || exit 1

echo "$DEST is ready to upload:"
echo "  $shards parquet shards, $(du -sh "$DEST/data" | cut -f1)"
echo "  $(du -sh "$DEST" | cut -f1) total"
