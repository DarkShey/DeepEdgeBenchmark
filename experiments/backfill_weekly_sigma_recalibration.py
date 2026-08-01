"""
backfill_weekly_sigma_recalibration.py — apply the adopted weekly EWMA
calibration to the ALREADY-STORED weekly OOS rows in tracking.db.

Why: the weekly grid rows were backfilled BEFORE the calibration adoption
(weekly_multimodel calibrate_sigma="on", commit 85a59c1) -- they carry the raw,
known-miscalibrated bands (Prophet W+1/2/3 at 0% realized coverage on the last
30 resolutions, cf. the dashboard monitoring). Re-running the full weekly
backtest would take hours (LSTM refits per origin); it is also unnecessary:
given the same raw forecasts, the wired mechanism is a DETERMINISTIC transform
of the stored rows. This script applies exactly it:

  per (model in SIGMA_SCALE_MODELS, asset, horizon_unit) cell, ordered by
  cutoff_date: sigma'_t = sigma_t * sqrt(EWMA(z^2)), lambda=0.94, z measured
  on the RAW band, a z entering the state only once RESOLVED
  (target_date <= current cutoff_date -- the same date-causal rule as
  weekly_sigma_scale_validation.py, which measured this exact transform at
  Prophet 79-82 -> 93-94% coverage-95).

Idempotent by reconstruction: each row's raw band is recovered from the stored
band and its `sigma_scale_applied` (1.0 on never-touched rows), the whole scale
path is recomputed from raw z's, and the row is rewritten (bounds, factor,
in_interval re-evaluated where resolved). Running twice yields identical rows.

ARIMA-GARCH and TSDiff rows are never touched (out of calibration scope).
Live rows are untouched too (weekly live = TSDiff only today).

Usage:
    python experiments/backfill_weekly_sigma_recalibration.py            # dry-run
    python experiments/backfill_weekly_sigma_recalibration.py --apply    # write
"""

import argparse
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "validation" / "tracking.db"

Z975 = 1.959963984540054
EWMA_LAMBDA = 0.94
SIGMA_SCALE_MODELS = ("SARIMA", "Prophet", "Naive", "LSTM")


def recalibrate_cell(g: pd.DataFrame) -> pd.DataFrame:
    """g: one (model, asset, horizon_unit) cell sorted by cutoff_date. Returns
    the new (y_lower, y_upper, sigma_scale_applied, in_interval) columns."""
    applied = g["sigma_scale_applied"].fillna(1.0).values
    mu = g["y_pred"].values
    lo_raw = mu - (mu - g["y_lower"].values) / applied
    hi_raw = mu + (g["y_upper"].values - mu) / applied
    sig_raw = np.maximum((hi_raw - lo_raw) / (2.0 * Z975), 1e-12)
    z2 = ((g["y_true"].values - mu) / sig_raw) ** 2   # NaN si non résolu

    cutoffs = pd.to_datetime(g["cutoff_date"]).values
    targets = pd.to_datetime(g["target_date"]).values
    order = np.argsort(targets, kind="stable")
    s2, j = 1.0, 0
    scale = np.empty(len(g))
    for i in range(len(g)):
        while j < len(order) and targets[order[j]] <= cutoffs[i]:
            if np.isfinite(z2[order[j]]):
                s2 = EWMA_LAMBDA * s2 + (1 - EWMA_LAMBDA) * z2[order[j]]
            j += 1
        scale[i] = np.sqrt(s2)

    out = pd.DataFrame(index=g.index)
    out["y_lower"] = mu - (mu - lo_raw) * scale
    out["y_upper"] = mu + (hi_raw - mu) * scale
    out["sigma_scale_applied"] = scale
    y_true = g["y_true"].values
    out["in_interval"] = np.where(
        np.isfinite(y_true),
        ((y_true >= out["y_lower"].values) & (y_true <= out["y_upper"].values)).astype(float),
        np.nan)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true",
                   help="écrit réellement (défaut : dry-run, rapport seulement)")
    p.add_argument("--db", default=str(DB_PATH))
    args = p.parse_args()

    # migration paresseuse (ajoute sigma_scale_applied si la base ne l'a pas encore)
    import sys
    sys.path.insert(0, str(ROOT))
    from validation import tracking_db as td
    td.init_db(args.db)

    con = sqlite3.connect(args.db)
    df = pd.read_sql_query(
        """
        SELECT id, model, asset, horizon_unit, cutoff_date, target_date,
               y_pred, y_lower, y_upper, y_true, in_interval, sigma_scale_applied
        FROM predictions
        WHERE frequence='weekly' AND horizon_type='weekly'
              AND source='oos' AND daily_duplicate=0 AND model IN ({})
        ORDER BY cutoff_date
        """.format(",".join("?" * len(SIGMA_SCALE_MODELS))),
        con, params=list(SIGMA_SCALE_MODELS))

    if df.empty:
        print("aucune ligne weekly OOS à recalibrer"); con.close(); return

    updates = []
    stats = {}
    for (model, asset, hu), g in df.groupby(["model", "asset", "horizon_unit"]):
        g = g.sort_values("cutoff_date")
        new = recalibrate_cell(g)
        realized = g["y_true"].notna().values
        cov_before = float(np.nanmean(g["in_interval"].values[realized])) * 100 \
            if realized.any() else float("nan")
        cov_after = float(np.nanmean(new["in_interval"].values[realized])) * 100 \
            if realized.any() else float("nan")
        stats.setdefault((model, hu), []).append((cov_before, cov_after))
        for row_id, (_, r) in zip(g["id"].values, new.iterrows()):
            updates.append((float(r["y_lower"]), float(r["y_upper"]),
                            float(r["sigma_scale_applied"]),
                            None if pd.isna(r["in_interval"]) else int(r["in_interval"]),
                            int(row_id)))

    print(f"{len(updates)} lignes weekly OOS à recalibrer "
          f"({'APPLY' if args.apply else 'DRY-RUN'})")
    print(f"{'modèle':8} {'horizon':7} {'cov95 avant':>12} {'cov95 après':>12}")
    for (model, hu), pairs in sorted(stats.items()):
        b = np.nanmean([x[0] for x in pairs])
        a = np.nanmean([x[1] for x in pairs])
        print(f"{model:8} {hu:7} {b:11.1f}% {a:11.1f}%")

    if args.apply:
        backup = Path(args.db).with_suffix(".db.bak_recalib_" + date.today().isoformat())
        if not backup.exists():
            shutil.copy2(args.db, backup)
            print(f"backup -> {backup.name}")
        con.executemany(
            "UPDATE predictions SET y_lower=?, y_upper=?, sigma_scale_applied=?, "
            "in_interval=? WHERE id=?", updates)
        con.commit()
        print("UPDATE fait.")
    con.close()


if __name__ == "__main__":
    main()
