"""
compute_prob_kpi_pilot.py — load the persisted sample clouds (parametric +
native TSDiff) for one asset, compute CRPS/coverage(50/80/95)/PIT/sharpness/
Winkler/MASE per row, aggregate by (model, horizon_unit), print the pilot
table, and save the full per-row + aggregated KPIs to JSON.

MASE here follows honest_eval.metrics.mase's existing definition (MAE(model)
sample-mean point / MAE(Naive) on the SAME rows, matched by cutoff_date) --
not a separate in-sample scale, so it needs zero extra data fetch.

Usage:
    python compute_prob_kpi_pilot.py --asset SPY
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

from prob_kpi_common import row_kpis, HORIZON_LABEL_ORDER          # noqa: E402
from honest_eval.metrics import mase as mase_fn                    # noqa: E402

SAMPLES_DIR = ROOT / "experiments" / "samples"
MODEL_ORDER = ["ARIMA-GARCH", "SARIMA", "Prophet", "Naive", "LSTM", "TSDiff"]


def load_all_samples(asset: str) -> tuple:
    indices, sample_arrays = [], []
    for suffix in ("parametric", "tsdiff"):
        idx_path = SAMPLES_DIR / f"{asset}_{suffix}.index.parquet"
        smp_path = SAMPLES_DIR / f"{asset}_{suffix}.samples.npz"
        if not idx_path.exists():
            continue
        idx = pd.read_parquet(idx_path)
        smp = np.load(smp_path)["samples"]
        indices.append(idx)
        sample_arrays.append(smp)
    if not indices:
        raise SystemExit(f"no sample files found for asset={asset!r} in {SAMPLES_DIR}")
    index = pd.concat(indices, ignore_index=True)
    samples = np.concatenate(sample_arrays, axis=0)
    return index, samples


def compute_row_kpis(index: pd.DataFrame, samples: np.ndarray) -> pd.DataFrame:
    records = []
    for i, row in index.iterrows():
        k = row_kpis(samples[i], float(row["y_true"]))
        records.append({**row.to_dict(), **k})
    return pd.DataFrame(records)


def add_mase(df: pd.DataFrame) -> pd.DataFrame:
    """Per (frequence, horizon_type, horizon_unit, cutoff_date): join each
    model's sample_mean point forecast against Naive's on the SAME origin,
    then aggregate MAE(model)/MAE(naive) per (model, horizon_unit) group --
    honest_eval.metrics.mase's own definition, reused not reinvented."""
    key = ["asset", "frequence", "horizon_type", "horizon_unit", "cutoff_date", "target_date"]
    naive = df[df["model"] == "Naive"][key + ["sample_mean", "y_true"]].rename(
        columns={"sample_mean": "naive_pred"})
    merged = df.merge(naive, on=key, how="left", suffixes=("", "_naiveref"))
    merged["_has_naive"] = merged["naive_pred"].notna()
    return merged


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in df.groupby(["model", "horizon_unit"]):
        n = len(grp)
        entry = {
            "model": model, "horizon_unit": h, "n_origins": n,
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
            "pit_std": float(grp["pit"].std()),
        }
        matched = grp[grp["_has_naive"]] if "_has_naive" in grp else grp.iloc[0:0]
        entry["mase_n_matched"] = int(len(matched))
        if model == "Naive":
            entry["mase"] = 1.0
        elif len(matched) > 0:
            # MASE on whichever rows have a same-origin Naive match (brief guardrail:
            # N identical per model for the *sample cloud*, not a requirement that
            # every model share literally the same origin set -- TSDiff's sparse D+7/
            # regime-C coverage means it won't always match 1:1 with Naive's origins).
            entry["mase"] = float(mase_fn(matched["y_true"], matched["sample_mean"], matched["naive_pred"]))
            if len(matched) < n:
                entry["mase_note"] = f"only {len(matched)}/{n} origins had a matching Naive row"
        else:
            entry["mase"] = None
        rows.append(entry)
    out = pd.DataFrame(rows)
    out["horizon_unit"] = pd.Categorical(out["horizon_unit"], categories=HORIZON_LABEL_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["horizon_unit", "model"]).reset_index(drop=True)


def print_table(agg: pd.DataFrame) -> None:
    for h in HORIZON_LABEL_ORDER:
        sub = agg[agg["horizon_unit"] == h]
        if sub.empty:
            continue
        print(f"\n=== {h} ===")
        print(f"{'Model':<13}{'n':>5}{'CRPS':>10}{'Cov50':>8}{'Cov80':>8}{'Cov95':>8}"
              f"{'Sharp95':>10}{'Winkler95':>11}{'PIT-mean':>10}{'MASE':>8}")
        for _, r in sub.iterrows():
            mase_str = f"{r['mase']:.3f}" if r["mase"] is not None else "n/a"
            print(f"{r['model']:<13}{r['n_origins']:>5}{r['crps_mean']:>10.3f}"
                  f"{r['coverage_50']:>8.2f}{r['coverage_80']:>8.2f}{r['coverage_95']:>8.2f}"
                  f"{r['sharpness_95']:>10.3f}{r['winkler_95']:>11.3f}"
                  f"{r['pit_mean']:>10.3f}{mase_str:>8}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    index, samples = load_all_samples(args.asset)
    print(f"[{args.asset}] loaded {len(index)} rows x {samples.shape[1]} samples "
          f"({index['method'].value_counts().to_dict()})")

    df = compute_row_kpis(index, samples)
    df = add_mase(df)
    agg = aggregate(df)
    print_table(agg)

    out_path = Path(args.out) if args.out else (
        ROOT / "experiments" / f"kpi_probabilistes_pilot_{args.asset}.json")
    payload = {
        "asset": args.asset,
        "n_samples": int(index["n_samples"].iloc[0]),
        "aggregated": json.loads(agg.astype(object).where(pd.notnull(agg), None).to_json(orient="records")),
        "per_row": json.loads(df.drop(columns=["_has_naive"], errors="ignore")
                              .astype(object).where(pd.notnull(df), None)
                              .to_json(orient="records")),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
