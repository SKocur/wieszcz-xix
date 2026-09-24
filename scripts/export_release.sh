#!/usr/bin/env bash
# Export the checkpoints the paper cites into self-contained model directories.
#
# Nine of them: final, pre-decay and epoch-1 for each rung. Those are the checkpoints the
# results rest on, the epochs table in the paper reads all three, and everything else
# a run wrote is trajectory whose loss curve already ships as telemetry.
#
# This is a file rather than a loop typed at a prompt because the mapping from rung to run
# directory to eval key is three parallel facts, and getting them out of step is silent:
# an export carrying another rung's eval report loads, validates and reads plausibly.
#
#     bash scripts/export_release.sh            # all nine into models/
#     bash scripts/export_release.sh 47m        # one rung
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python3
LADDER="metrics/eval_by_source_2026-08-17_fullspan.json"

declare -A RUN=(
  [47m]=wieszcz_47m_6b7_2026-08-05_s1337
  [107m]=wieszcz_107m_6b7_2026-08-06_s1337
  [349m]=wieszcz_349m_6b7_2026-08-07_s1337
)
declare -A RUNG=([47m]=47M [107m]=107M [349m]=349M)

tags=("$@")
[[ ${#tags[@]} -eq 0 ]] && tags=(47m 107m 349m)

for tag in "${tags[@]}"; do
  run=${RUN[$tag]:?unknown rung $tag}
  for ckpt in final predecay epoch1; do
    # The final checkpoint is the one the ladder report scored; the other two have their own
    # single-model reports, which already carry the flat shape the card wants.
    if [[ $ckpt == final ]]; then
      eval_args=(--eval "$LADDER" --eval-rung "${RUNG[$tag]}")
    else
      # 2026-08-20: the 08-17 reports came from the pre-repair harness (89% span, no
      # Wolne Lektury window); these are the full-span rescores.
      eval_args=(--eval "metrics/eval_${tag}_${ckpt}_2026-08-20.json")
    fi
    $PY scripts/export_model.py \
      --ckpt "checkpoints/$run/$ckpt.pt" \
      --name "wieszcz-$tag-$ckpt" \
      --out models \
      "${eval_args[@]}" \
      --log "metrics/train_wieszcz_${tag}_6b7.out" \
      --manifest "metrics/${run}_manifest.json" \
      --force
    df -h . | awk 'NR==2 {print "   disk: " $4 " free"}'
  done
done
