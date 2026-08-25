#!/usr/bin/env bash
set -u

# CPU-only watcher for legacy queue workers that were started before the
# checkpoint-window hook was added. It never touches a running replay; once a
# final JSON exists, it derives the late-window evidence from the preserved
# checkpoint and marks that case ready for the audit.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
BASE="$ROOT/results/property/declared_persistent_4096/expanded_controls"
CHECK="/data1/tzh/cache/bias_long_expanded"
ids=(
  multishape-backward-cell-0057 multishape-backward-cell-0103
  multishape-backward-cell-0153 multishape-backward-cell-0190
  multishape-backward-cell-0191 multishape-backward-cell-0450
  multishape-backward-cell-0501 multishape-backward-cell-0508
  multishape-backward-cell-0543 multishape-backward-cell-0654
  multishape-backward-cell-0745 multishape-backward-cell-0747
)

while :; do
  remaining=0
  for id in "${ids[@]}"; do
    output="$BASE/${id}_4096.json"
    sidecar="$BASE/${id}_4096_windows.json"
    checkpoint="$CHECK/${id}.pt"
    [[ -s "$sidecar" ]] && continue
    [[ -s "$BASE/retry_failures/${id}.json" ]] && continue
    if [[ -s "$output" && -s "$checkpoint" ]]; then
      "$PY" "$ROOT/scripts/analyze_long_checkpoint_windows.py" \
        --checkpoint "$checkpoint" --output "$sidecar" \
        >"$BASE/${id}_4096_windows.log" 2>&1 || true
    else
      remaining=$((remaining + 1))
    fi
  done
  [[ "$remaining" -eq 0 ]] && exit 0
  sleep 60
done
