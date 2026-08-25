#!/usr/bin/env bash
set -u

# Run the frozen Gemma/Llama target set after existing long jobs release GPU 3.
# Pilot first; only a pilot risk starts a 4096-step consequence. Failures stay
# unresolved and are never converted to negative labels.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
PY_GEMMA="/data1/tzh/envs/pt_nightly_transformers5/bin/python"
MANIFEST="$ROOT/results/property/declared_persistent_4096/operator_scan_target_manifest.json"
GPU="${1:-3}"
SHARD="${2:-0}"
TOTAL_SHARDS="${3:-1}"
MIN_FREE_MB="${4:-12000}"
LOG="$ROOT/results/property/declared_persistent_4096/operator_scan_targets_queue_gpu${GPU}_shard${SHARD}of${TOTAL_SHARDS}.log"

if (( TOTAL_SHARDS < 1 || SHARD < 0 || SHARD >= TOTAL_SHARDS )); then
  echo "usage: $0 GPU [SHARD TOTAL_SHARDS]" >&2
  exit 2
fi

wait_for_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v g="$GPU" '$1+0==g {gsub(/ /,"",$2); print $2}')
    if [[ -n "$free" && "$free" -ge "$MIN_FREE_MB" ]]; then return 0; fi
    echo "[$(date -Is)] waiting for GPU${GPU}, free_memory=${free:-unknown} MiB" >>"$LOG"
    sleep 120
  done
}

run_one() {
  local row_json="$1"
  local case_id arch model input consequence carrier region endpoint symbol target_kind external_census external_exact retain_full screen_p screen_bias
  case_id=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["case_id"])')
  arch=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["architecture"])')
  local runner_py="$PY"
  local endpoint_rebind_flag=""
  if [[ "$arch" == "gemma4" ]]; then runner_py="$PY_GEMMA"; fi
  if [[ "$arch" == "gemma4" ]]; then
    endpoint_rebind_flag="--allow-single-output-rebind --allow-semantic-norm-rebind"
  fi
  model=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model_path"])')
  input=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["input_bank"])')
  consequence=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["consequence_bank"])')
  carrier=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["carrier"])')
  region=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("target_region", ""))')
  endpoint=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("target_endpoint", ""))')
  symbol=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("target_symbol", ""))')
  target_kind=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("target_kind", "TRITON"))')
  external_census=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("external_census", ""))')
  external_exact=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("external_exact_id", ""))')
  retain_full=$(echo "$row_json" | "$PY" -c 'import json,sys; print("1" if json.load(sys.stdin).get("retain_full_backward") else "0")')
  screen_p=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("screen_p_value", 1.0))')
  screen_bias=$("$PY" -c 'import sys; print("1" if float(sys.argv[1]) <= 0.05 else "0")' "$screen_p")
  local target_args=(--target-kind "$target_kind")
  if [[ "$target_kind" == "EXTERN" ]]; then
    target_args+=(--external-census "$ROOT/$external_census" --external-exact-id "$external_exact")
    if [[ "$retain_full" == "1" ]]; then target_args+=(--retain-full-backward); fi
  else
    target_args+=(--target-region "$region" --target-symbol "$symbol" --target-endpoint "$endpoint")
  fi
  local base="$ROOT/results/property/declared_persistent_4096/operator_scan_targets"
  local pilot="$base/${case_id}_pilot" long="$base/${case_id}"
  if [[ -f "$long/consequence.json" ]]; then echo "[$(date -Is)] skip complete $case_id" >>"$LOG"; return 0; fi
  # A manually launched long replay may already own the canonical output
  # directory.  Wait for it instead of treating the directory as a compiler
  # failure or starting a duplicate.  If a dead attempt remains, preserve it
  # under an explicit failed-attempt suffix and retry the canonical path.
  if [[ -d "$long" && ! -f "$long/consequence.json" ]]; then
    while pgrep -f "run_gemma4_v3_validation.py.*--output-dir $long" >/dev/null 2>&1; do
      echo "[$(date -Is)] waiting for active long replay $case_id" >>"$LOG"
      sleep 120
    done
    if [[ -f "$long/consequence.json" ]]; then
      echo "[$(date -Is)] active long replay completed $case_id" >>"$LOG"
      return 0
    fi
    local failed_suffix=0 failed_path="${long}_failed_attempt0"
    while [[ -e "$failed_path" ]]; do
      failed_suffix=$((failed_suffix + 1)); failed_path="${long}_failed_attempt${failed_suffix}"
    done
    mv "$long" "$failed_path"
    echo "[$(date -Is)] preserved incomplete attempt $case_id at $failed_path" >>"$LOG"
  fi
  # Reuse any completed pilot from a manual retry instead of creating a
  # duplicate process.  A failed/partial directory is left as evidence and a
  # fresh suffix is used below.
  local completed_pilot=""
  for candidate in "$base/${case_id}"_pilot*; do
    if [[ -f "$candidate/prediction.json" && -f "$candidate/short_screen.json" ]] && \
       "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); statuses=[str(x.get("status","")) for x in s.get("cases",[])]; old_zero=bool(statuses) and all(x=="UNRESOLVED_ZERO_ENERGY" for x in statuses); unresolved_repair=str(p.get("status","")).startswith("UNRESOLVED_ZERO_GRADIENT"); raise SystemExit(0 if p.get("evidence",{}).get("states",0)>=16 and s.get("input",{}).get("steps",0)>=16 and not old_zero and not unresolved_repair else 1)' \
         "$candidate/prediction.json" "$candidate/short_screen.json"; then
      completed_pilot="$candidate"
    fi
  done
  if [[ -n "$completed_pilot" ]]; then
    pilot="$completed_pilot"
  else
    local suffix=0
    while [[ -e "$pilot" ]]; do suffix=$((suffix + 1)); pilot="$base/${case_id}_pilot_retry${suffix}"; done
  fi

  # A previous attempt may have created a partial rebind directory and then
  # failed before writing prediction.json.  The runner intentionally refuses
  # to overwrite output directories, so always choose a fresh suffix instead
  # of turning a recoverable retry into a duplicate-directory failure.
  fresh_output_dir() {
    local requested="$1" candidate="$1" suffix=0
    while [[ -e "$candidate" && ! -f "$candidate/prediction.json" ]]; do
      suffix=$((suffix + 1)); candidate="${requested}_retry${suffix}"
    done
    printf '%s\n' "$candidate"
  }
  # A zero-energy Llama pilot can mean that the historical target is real but
  # the chosen final-norm carrier is downstream of the changed value (or the
  # fresh compiler rebound the callsite to a different semantic region).  Try
  # one declared downstream carrier before leaving the row unresolved.  This
  # is still a target-binding diagnostic, never a negative result.
  if [[ "$arch" == "generic" && "$target_kind" == "TRITON" && -f "$pilot/prediction.json" && -f "$pilot/short_screen.json" ]]; then
    if "$PY" - "$pilot/short_screen.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); rows=d.get("cases",[])
raise SystemExit(0 if rows and all(x.get("status") == "UNRESOLVED_ZERO_ENERGY" for x in rows) else 1)
PY
    then
      rebind="$base/${case_id}_pilot_rebind_embed"
      rebind=$(fresh_output_dir "$rebind")
      if [[ ! -f "$rebind/prediction.json" ]]; then
        wait_for_gpu
        echo "[$(date -Is)] rebind pilot $case_id carrier=model.embed_tokens.weight" >>"$LOG"
        if ! CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$runner_py" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
            --input-bank "$input" --consequence-bank "$consequence" --output-dir "$rebind" --steps 16 \
            --consequence-steps 16 "${target_args[@]}" \
            --carrier model.embed_tokens.weight --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 1000 \
            >>"$LOG" 2>&1; then
          echo "[$(date -Is)] rebind unresolved $case_id" >>"$LOG"
        fi
      fi
      if [[ -f "$rebind/prediction.json" && -f "$rebind/short_screen.json" ]]; then
        pilot="$rebind"
      fi
    fi
  fi
  if [[ -f "$pilot/prediction.json" ]]; then
    local decision
    decision=$("$PY" -c 'import json,sys,os; p=sys.argv[1]; frozen_bias=sys.argv[2]=="1"; prediction=json.load(open(p+"/prediction.json")); pred=str(prediction.get("source_prediction","")); short=json.load(open(p+"/short_screen.json")) if os.path.exists(p+"/short_screen.json") else {}; not_applicable=pred.startswith("NOT_APPLICABLE"); print("STOP" if not_applicable else ("LONG" if frozen_bias or pred=="SOURCE_PERSISTENCE_RISK" or any(x.get("status")=="RISK_CANDIDATE" for x in short.get("cases",[])) else "STOP"))' "$pilot" "$screen_bias")
    if [[ "$decision" != LONG ]]; then echo "[$(date -Is)] reuse pilot no-risk $case_id" >>"$LOG"; return 0; fi
  fi
  wait_for_gpu
  echo "[$(date -Is)] pilot $case_id region=$region endpoint=$endpoint" >>"$LOG"
  if ! CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$runner_py" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
      --input-bank "$input" --consequence-bank "$consequence" --output-dir "$pilot" --steps 16 \
      --consequence-steps 16 "${target_args[@]}" --carrier "$carrier" \
      --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 1000 \
      $endpoint_rebind_flag \
      $([[ "$arch" == "gemma4" ]] && echo --allow-graph-breaks) >>"$LOG" 2>&1; then
    # A small norm carrier can compile while the historical target is absent
    # from the fresh graph. Retry once with the tied embedding before calling
    # the row unresolved; this is a binding repair, never a negative verdict.
    if [[ "$arch" == "generic" && "$target_kind" == "TRITON" && "$carrier" != "model.embed_tokens.weight" ]]; then
      # Try a small early LayerNorm carrier before the very large tied
      # embedding.  This often keeps the target reachable without the
      # embedding's multi-gigabyte optimizer buffers.
      for fallback in model.layers.0.input_layernorm.weight model.embed_tokens.weight; do
        tag="${fallback//./_}"
        rebind="$base/${case_id}_pilot_rebind_${tag}"
        rebind=$(fresh_output_dir "$rebind")
        if [[ -f "$rebind/prediction.json" ]]; then pilot="$rebind"; break; fi
        wait_for_gpu
        echo "[$(date -Is)] rebind after target failure $case_id carrier=$fallback" >>"$LOG"
        if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$runner_py" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
            --input-bank "$input" --consequence-bank "$consequence" --output-dir "$rebind" --steps 16 \
            --consequence-steps 16 "${target_args[@]}" \
            --carrier "$fallback" --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 1000 \
            >>"$LOG" 2>&1; then
          pilot="$rebind"
          break
        fi
      done
    fi
    if [[ ! -f "$pilot/prediction.json" ]]; then
      mkdir -p "$base/unresolved"
      "$PY" -c 'import json,sys; print(json.dumps({"schema":"kernel-analyzer-target-replay-failure-v1","case_id":sys.argv[1],"status":"UNRESOLVED_PILOT_REPLAY_RESOURCE","log":sys.argv[2],"output_dir":sys.argv[3],"claim_boundary":"Target pilot failed or could not allocate resources; no negative label is assigned."},indent=2))' "$case_id" "$LOG" "$pilot" >"$base/unresolved/${case_id}_pilot.json"
      echo "[$(date -Is)] pilot unresolved $case_id" >>"$LOG"; return 0
    fi
  fi

  # A representable endpoint repair with zero gradient on the declared
  # carrier is a reachability failure, not a negative.  Retry a small,
  # predeclared set of downstream parameters and keep the first replay that
  # produces a real gradient contrast.  This also covers external GEMM/BMM
  # targets, for which the original manifest carrier is only a hypothesis.
  if [[ -f "$pilot/prediction.json" ]] && \
     "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if str(d.get("status",""))=="UNRESOLVED_ZERO_GRADIENT_AFTER_REPRESENTABLE_REPAIR" else 1)' "$pilot/prediction.json"; then
    local -a carrier_fallbacks=()
    if [[ "$arch" == "gemma4" ]]; then
      carrier_fallbacks=(
        model.language_model.layers.0.input_layernorm.weight
        model.language_model.per_layer_model_projection.weight
      )
    else
      carrier_fallbacks=(
        model.layers.0.input_layernorm.weight
        model.norm.weight
        model.embed_tokens.weight
      )
    fi
    for fallback in "${carrier_fallbacks[@]}"; do
      [[ "$fallback" == "$carrier" ]] && continue
      tag="${fallback//./_}"
      rebind="$base/${case_id}_pilot_rebind_${tag}"
      rebind=$(fresh_output_dir "$rebind")
      wait_for_gpu
      echo "[$(date -Is)] rebind after zero reachable gradient $case_id carrier=$fallback" >>"$LOG"
      if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
          OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$runner_py" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
          --input-bank "$input" --consequence-bank "$consequence" --output-dir "$rebind" --steps 16 \
          --consequence-steps 16 "${target_args[@]}" --carrier "$fallback" \
          --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 1000 \
          $endpoint_rebind_flag \
          $([[ "$arch" == "gemma4" ]] && echo --allow-graph-breaks) >>"$LOG" 2>&1; then
        if "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(1 if str(d.get("status",""))=="UNRESOLVED_ZERO_GRADIENT_AFTER_REPRESENTABLE_REPAIR" else 0)' "$rebind/prediction.json"; then
          pilot="$rebind"
          break
        fi
      fi
    done
  fi
  local decision
  decision=$("$PY" -c 'import json,sys,os; p=sys.argv[1]; frozen_bias=sys.argv[2]=="1"; prediction=json.load(open(p+"/prediction.json")); pred=str(prediction.get("source_prediction","")); short=json.load(open(p+"/short_screen.json")) if os.path.exists(p+"/short_screen.json") else {}; not_applicable=pred.startswith("NOT_APPLICABLE"); print("STOP" if not_applicable else ("LONG" if frozen_bias or pred=="SOURCE_PERSISTENCE_RISK" or any(x.get("status")=="RISK_CANDIDATE" for x in short.get("cases",[])) else "STOP"))' "$pilot" "$screen_bias")
  if [[ "$decision" != LONG ]]; then echo "[$(date -Is)] pilot no-risk $case_id" >>"$LOG"; return 0; fi
  # If the pilot had to rebind its carrier, the long replay must use that
  # exact same parameter location.  Falling back to the manifest carrier here
  # would silently change the experimental contrast between 16 and 4096 steps.
  local long_carrier="$carrier"
  if [[ -f "$pilot/consequence.json" ]]; then
    long_carrier=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("carrier") or sys.argv[2])' "$pilot/consequence.json" "$carrier")
  elif [[ -f "$pilot/prediction.json" ]]; then
    long_carrier=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("carrier") or sys.argv[2])' "$pilot/prediction.json" "$carrier")
  fi
  wait_for_gpu
  echo "[$(date -Is)] long $case_id carrier=$long_carrier" >>"$LOG"
  if ! CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$runner_py" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
      --input-bank "$input" --consequence-bank "$consequence" --output-dir "$long" --steps 16 \
      --consequence-steps 4096 "${target_args[@]}" --carrier "$long_carrier" \
      --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 2000 \
      $endpoint_rebind_flag \
      $([[ "$arch" == "gemma4" ]] && echo --allow-graph-breaks) >>"$LOG" 2>&1; then
    mkdir -p "$base/unresolved"
    "$PY" -c 'import json,sys; print(json.dumps({"schema":"kernel-analyzer-target-replay-failure-v1","case_id":sys.argv[1],"status":"UNRESOLVED_LONG_REPLAY_RESOURCE","log":sys.argv[2],"output_dir":sys.argv[3],"claim_boundary":"Target replay failed; no negative label is assigned."},indent=2))' "$case_id" "$LOG" "$long" >"$base/unresolved/${case_id}.json"
  fi
}

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] queue start" >>"$LOG"
while IFS= read -r row; do run_one "$row"; done < <(
  "$PY" -c '
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
shard, total = int(sys.argv[2]), int(sys.argv[3])
for index, row in enumerate(rows):
    if index % total == shard:
        print(json.dumps(row, sort_keys=True))
' "$MANIFEST" "$SHARD" "$TOTAL_SHARDS"
)
"$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$LOG" 2>&1 || true
echo "[$(date -Is)] queue complete gpu=$GPU shard=$SHARD/$TOTAL_SHARDS" >>"$LOG"
