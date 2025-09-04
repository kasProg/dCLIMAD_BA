# auto_select.py
import json, math, re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

############################
# 1) Configurable settings #
############################
# Your metrics keys (all are "lower is better" AFTER transform in _compute_J)
METRIC_WEIGHTS = {
    "tail_q95": 0.15,
    "tail_q99": 0.15,
    "tail_rx1": 0.10,
    "tail_rx5": 0.10,
    "wet_sdii": 0.10,
    "wet_cdd": 0.10,
    "wet_cwd": 0.10,
    "wet_r10": 0.10,
    "wet_r20": 0.10,
}

# Optionally weight across temporal aggregation scales
# SCALE_WEIGHTS = {"daily": 0.5, "monthly": 0.3, "seasonal": 0.2}

EMA_ALPHA = 0.3            # smoothing for J across epochs
MIN_EPOCHS = 10            # require at least this many epochs to evaluate a run
IMPROVEMENT_FLOOR = 0.02   # require >=2% improvement vs the run's early J, else mark as "stagnant"
NONINCR_TOL = 0.01         # allow tiny non-monotonicity (1%) in the tail of the curve

# Filenames expected inside each trial directory
VAL_LOG = "val_metrics.jsonl"    # one json per line: {"epoch": int, "loss": float, "metrics": {...}}
BASELINE_FILE = "baseline.json"  # raw-vs-obs metrics dict used for normalization (same keys as metrics)

#######################
# 2) Helper utilities #
#######################
def _ema(series: List[float], alpha: float) -> List[float]:
    out = []
    s = None
    for x in series:
        s = x if s is None else alpha * x + (1 - alpha) * s
        out.append(s)
    return out

def _safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return a / (b + eps)

def _normalize_metric(value: float, baseline: float, lower_better=True) -> float:
    """
    Map metric to [0,1] where 0 ~ perfect, 1 ~ as bad as raw baseline.
    If lower_better is False (e.g., PDF overlap), we invert appropriately.
    """
    if not lower_better:
        # convert to a "gap" first: 1 - overlap
        value = 1.0 - value
        baseline = 1.0 - baseline
    # Normalize relative to baseline (clip to [0,1.5] but we'll clamp later)
    norm = _safe_div(value, baseline if baseline > 0 else 1.0)
    return max(0.0, min(1.0, norm))

def _compute_J(metrics: Dict[str, float], baseline: Dict[str, float]) -> float:
    """
    Compute single scalar J from (possibly multi-scale) metrics.
    Expect keys like "daily/pdf_overlap", "monthly/rx1_err", etc.
    """
    accum = 0.0
    wsum = 0.0
    # for scale, sw in SCALE_WEIGHTS.items():
    scale_contrib = 0.0
    scale_wsum = 0.0

    # materialize a per-scale dict if present
    def get(key_suffix: str) -> Optional[float]:
        # prefer "<scale>/<key>", else plain "<key>"
        if f"{key_suffix}" in metrics:
            return metrics[f"{key_suffix}"]
        return metrics.get(key_suffix, None)



    # build normalized components
    parts = {
        "tail_rx1": _normalize_metric(get("Rx1day") or 0.0, baseline.get(f"Rx1day", 1.0)),
        "tail_rx5": _normalize_metric(get("Rx5day") or 0.0, baseline.get(f"Rx5day", 1.0)),
        "wet_sdii": _normalize_metric(get("SDII (Monthly)") or 0.0, baseline.get(f"SDII (Monthly)", 1.0)),
        "wet_cdd":  _normalize_metric(get("CDD (Yearly)") or 0.0,  baseline.get(f"CDD (Yearly)", 1.0)),
        "wet_cwd":  _normalize_metric(get("CWD (Yearly)") or 0.0,  baseline.get(f"CWD (Yearly)", 1.0)),
        "wet_r10":  _normalize_metric(get("R10mm") or 0.0,  baseline.get(f"R10mm", 1.0)),
        "wet_r20":  _normalize_metric(get("R20mm") or 0.0,  baseline.get(f"R20mm", 1.0)),
        "tail_q95": _normalize_metric(get("R95pTOT") or 0.0, baseline.get(f"R95pTOT", 1.0)),
        "tail_q99": _normalize_metric(get("R99pTOT") or 0.0, baseline.get(f"R99pTOT", 1.0)),

    }

    for k, v in parts.items():
        w = METRIC_WEIGHTS.get(k, 0.0)
        accum += w * v
        wsum += w

    # if scale_wsum > 0:
    #     accum += sw * (scale_contrib / scale_wsum)
    #     wsum += sw

    return accum / wsum if wsum > 0 else 1.0

@dataclass
class EpochEval:
    epoch: int
    loss: float
    J: float
    J_ema: float

@dataclass
class TrialResult:
    trial_dir: str
    best_epoch: int
    best_J: float
    best_J_ema: float
    final_loss: float
    improved: bool
    good_tail: bool
    config_summary: Dict[str, Any]  # model_type, layers, parameter_scale, etc.

########################
# 3) Core evaluation   #
########################
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def read_config_summary(trial_dir: Path) -> Dict[str, Any]:
    # Try a few common config filenames
    out = {"model_type": None, "layers": None, "hidden": None,
           "parameter_scale": None, "epochs": None, "trial": trial_dir.name}
    # Lightweight parse from path name (fallback)
    m = re.search(r"(MLP|CNN|LSTM)", trial_dir.name, re.I)
    if m: out["model_type"] = m.group(1).upper()
    for fname in ["config.json", "config.yaml", "train_config.yaml"]:
        p = trial_dir / fname
        if p.exists():
            try:
                if p.suffix == ".json":
                    cfg = json.loads(p.read_text())
                else:
                    # naive YAML loader to avoid pyyaml dependency
                    import yaml  # comment out if you truly can't have it
                    cfg = yaml.safe_load(p.read_text())
                out.update({k: cfg.get(k) for k in ["model_type","layers","hidden","parameter_scale","epochs"] if k in cfg})
            except Exception:
                pass
    return out

def evaluate_trial(trial_dir: Path) -> Optional[TrialResult]:
    val_path = trial_dir / VAL_LOG
    base_path = trial_dir / BASELINE_FILE
    if not val_path.exists() or not base_path.exists():
        return None

    logs = load_jsonl(val_path)
    if len(logs) < MIN_EPOCHS:
        return None

    baseline = json.loads(base_path.read_text())
    epochs, losses, Js = [], [], []

    for row in logs:
        ep = int(row.get("epoch", len(epochs)))
        met = row.get("metrics", {})
        loss = float(row.get("loss", math.nan))
        J = _compute_J(met, baseline)
        epochs.append(ep); losses.append(loss); Js.append(J)

    J_ema = _ema(Js, EMA_ALPHA)
    # choose best by raw J (not EMA), but keep EMA for stability diagnostics
    best_idx = int(min(range(len(Js)), key=lambda i: Js[i]))
    best_epoch = epochs[best_idx]
    best_J, best_J_ema = Js[best_idx], J_ema[best_idx]

    # diagnostics: did J meaningfully improve vs early training?
    early = Js[min(5, len(Js)-1)]
    improved = (early - best_J) / max(early, 1e-6) >= IMPROVEMENT_FLOOR

    # diagnostics: tail monotonic-ish check over last 20% epochs (EMA)
    tail_start = int(0.8 * len(J_ema))
    tail = J_ema[tail_start:]
    good_tail = True
    for i in range(1, len(tail)):
        if tail[i] - tail[i-1] > NONINCR_TOL * max(tail[i-1], 1e-6):
            good_tail = False; break

    cfg = read_config_summary(trial_dir)
    return TrialResult(
        trial_dir=str(trial_dir),
        best_epoch=best_epoch,
        best_J=best_J,
        best_J_ema=best_J_ema,
        final_loss=float(losses[-1]),
        improved=improved,
        good_tail=good_tail,
        config_summary=cfg
    )

########################
# 4) Batch orchestration
########################
def scan_and_rank(root: str) -> Dict[str, Any]:
    rootp = Path(root)
    results: List[TrialResult] = []
    for trial in sorted([p for p in rootp.glob("**/") if (p/VAL_LOG).exists()]):
        r = evaluate_trial(trial)
        if r is not None:
            results.append(r)

    # Filter out unstable runs first
    stable = [r for r in results if r.improved and r.good_tail]
    finalists = stable if stable else results  # if nothing stable, fall back

    # Rank by best_J, then tail-extremes proxy if you log it (we already folded into J)
    finalists.sort(key=lambda r: (r.best_J, r.best_J_ema, r.final_loss))

    # Summaries
    table = []
    for r in finalists:
        row = {
            "trial": Path(r.trial_dir).name,
            "best_epoch": r.best_epoch,
            "best_J": round(r.best_J, 6),
            "best_J_ema": round(r.best_J_ema, 6),
            "final_loss": round(r.final_loss, 6),
            "improved": r.improved,
            "good_tail": r.good_tail,
            **{f"cfg_{k}": v for k, v in r.config_summary.items()}
        }
        table.append(row)

    best = finalists[0] if finalists else None
    return {
        "best": asdict(best) if best else None,
        "ranked": table,
        "n_total": len(results),
        "n_stable": len(stable),
    }

if __name__ == "__main__":
    import argparse, csv, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True, help="Directory containing many trial subfolders")
    ap.add_argument("--out_csv", default="auto_select_results.csv")
    ap.add_argument("--out_json", default="auto_select_best.json")
    args = ap.parse_args()

    summary = scan_and_rank(args.exp_root)

    # CSV table for quick viewing
    rows = summary["ranked"]
    if rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    # JSON with winner & counts
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[auto_select] scanned={summary['n_total']} stable={summary['n_stable']}")
    if summary["best"]:
        print(f"[auto_select] BEST trial={Path(summary['best']['trial_dir']).name} epoch={summary['best']['best_epoch']} J={summary['best']['best_J']:.4f}")
    else:
        print("[auto_select] No valid trials found.")
