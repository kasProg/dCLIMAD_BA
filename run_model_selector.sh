#!/usr/bin/env bash
set -euo pipefail
ROOT="/pscratch/sd/k/kas7897/dCLIMAD_BA"
output_dir="$ROOT/outputs/AdamW_harmonic2/jobs_LOCAspatioTempLSTM"
for exp_root in "$output_dir"/*-gridmet ; do
  [[ -d "$exp_root" ]] || continue
  model="$(basename "$exp_root")"  # e.g., gfdl_esm4-gridmet
  outdir="$output_dir/$model"
  mkdir -p "$outdir"

  echo "[run] $model"
  python "$ROOT/run_model_selector.py" \
    --exp_root "$exp_root" \
    --out_csv  "$outdir/demo_select_gridmet.csv" \
    --out_json "$outdir/demo_select_gridmet.json" \
    --val_period "1995,2000"
done
