#!/usr/bin/env bash
set -euo pipefail
ROOT="/pscratch/sd/k/kas7897/dCLIMAD_BA"
output_dir="$ROOT/outputs/outputs_AdamW_harmonic/jobs_LOCAspatioTempConv1d"
for exp_root in "$output_dir"/*-livneh ; do
  [[ -d "$exp_root" ]] || continue
  model="$(basename "$exp_root")"  # e.g., gfdl_esm4-gridmet
  outdir="$output_dir/$model"
  mkdir -p "$outdir"

  echo "[run] $model"
  python "$ROOT/run_model_selector.py" \
    --exp_root "$exp_root" \
    --out_csv  "$outdir/demo_select_livneh.csv" \
    --out_json "$outdir/demo_select_livneh.json" \
    --val_period "1965,1978"
done
