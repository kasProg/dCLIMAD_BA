#!/usr/bin/env bash
set -euo pipefail
ROOT="/pscratch/sd/k/kas7897/dCLIMAD_BA"
for exp_root in "$ROOT"/outputs_AdamW/jobs_monotone_LOCAspatioTempMLP_spatialLoss/*-livneh ; do
  [[ -d "$exp_root" ]] || continue
  model="$(basename "$exp_root")"  # e.g., gfdl_esm4-gridmet
  outdir="$ROOT/outputs_AdamW/jobs_monotone_LOCAspatioTempMLP_spatialLoss/$model"
  mkdir -p "$outdir"

  echo "[run] $model"
  python "$ROOT/demo_model_selector.py" \
    --exp_root "$exp_root" \
    --out_csv  "$outdir/demo_select_livneh.csv" \
    --out_json "$outdir/demo_select_livneh.json" \
    --val_period "1965,1978"
done
