"""
build_kpi_probabilistes.py — BRIEF_kpi_probabilistes.md step 2: the full 5-
asset x 240-cell probabilistic KPI matrix, from the persisted sample clouds
(experiments/samples/), NOT a re-run of any model.

Concatenates per-row KPIs (CRPS empirical, coverage 50/80/95, sharpness,
Winkler, PIT, MASE) across all 5 assets, tags each row with its asset class
(index/bond/crypto -- same convention as pooled_analysis.py, reused so step 3
clusters correlated assets identically), and aggregates to a per-cell table
(model x asset x frequence x horizon_type x horizon_unit).

Output: experiments/kpi_probabilistes.json -- {"per_row": [...], "per_cell": [...]}.
per_row is also what experiments/pooled_analysis_prob.py (step 3) consumes.

Usage:
    python build_kpi_probabilistes.py
    python build_kpi_probabilistes.py --assets SPY BTC-USD   # subset, e.g. for a quick check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from compute_prob_kpi_pilot import load_all_samples, compute_row_kpis, add_mase  # noqa: E402
from honest_eval.metrics import mase as mase_fn                                  # noqa: E402
from prob_kpi_common import HORIZON_LABEL_ORDER                                  # noqa: E402

ASSETS = ["SPY", "BTC-USD", "ETH-USD", "ZN=F", "TLT"]
ASSET_CLASS = {"SPY": "index", "ZN=F": "bond", "TLT": "bond",
              "BTC-USD": "crypto", "ETH-USD": "crypto"}
MODEL_ORDER = ["ARIMA-GARCH", "SARIMA", "Prophet", "Naive", "LSTM", "TSDiff"]
OUT_PATH = ROOT / "experiments" / "kpi_probabilistes.json"


def build_per_row(assets: list) -> pd.DataFrame:
    frames = []
    for asset in assets:
        print(f"[{asset}] loading samples + computing row KPIs ...")
        index, samples = load_all_samples(asset)
        df = compute_row_kpis(index, samples)
        df = add_mase(df)   # per-asset join (Naive from the SAME asset only -- key includes 'asset')
        print(f"  {len(df)} rows ({index['method'].value_counts().to_dict()})")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["asset_class"] = out["asset"].map(ASSET_CLASS)
    return out


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "asset", "frequence", "horizon_type", "horizon_unit"]
    for keys, grp in df.groupby(group_cols):
        model, asset, frequence, horizon_type, horizon_unit = keys
        n = len(grp)
        entry = {
            "model": model, "asset": asset, "asset_class": ASSET_CLASS.get(asset),
            "frequence": frequence, "horizon_type": horizon_type, "horizon_unit": horizon_unit,
            "n_origins": n,
            "crps_mean": float(grp["crps"].mean()),
            "coverage_50": float(grp["cov50"].mean()),
            "coverage_80": float(grp["cov80"].mean()),
            "coverage_95": float(grp["cov95"].mean()),
            "sharpness_50": float(grp["sharp50"].mean()),
            "sharpness_80": float(grp["sharp80"].mean()),
            "sharpness_95": float(grp["sharp95"].mean()),
            "winkler_50": float(grp["winkler50"].mean()),
            "winkler_80": float(grp["winkler80"].mean()),
            "winkler_95": float(grp["winkler95"].mean()),
            "pit_mean": float(grp["pit"].mean()),
            "pit_std": float(grp["pit"].std()) if n > 1 else None,
        }
        matched = grp[grp["_has_naive"]] if "_has_naive" in grp else grp.iloc[0:0]
        entry["mase_n_matched"] = int(len(matched))
        if model == "Naive":
            entry["mase"] = 1.0
        elif len(matched) > 0:
            entry["mase"] = float(mase_fn(matched["y_true"], matched["sample_mean"], matched["naive_pred"]))
        else:
            entry["mase"] = None
        rows.append(entry)
    out = pd.DataFrame(rows)
    out["horizon_unit"] = pd.Categorical(out["horizon_unit"], categories=HORIZON_LABEL_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["asset", "horizon_unit", "model"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", nargs="+", default=ASSETS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    df = build_per_row(args.assets)
    print(f"\nTotal per-row: {len(df)}")
    per_cell = aggregate_per_cell(df)
    print(f"Total per-cell: {len(per_cell)} (of up to {6*5*8}=240 possible model x asset x regime-cell slots)")

    payload = {
        "assets": args.assets,
        "asset_class": ASSET_CLASS,
        "n_samples": int(df["n_samples"].iloc[0]) if "n_samples" in df else 500,
        "per_row": json.loads(df.drop(columns=["_has_naive"], errors="ignore")
                              .astype(object).where(pd.notnull(df), None)
                              .to_json(orient="records")),
        "per_cell": json.loads(per_cell.astype(object).where(pd.notnull(per_cell), None)
                               .to_json(orient="records")),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
