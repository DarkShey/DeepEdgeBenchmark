"""
weekly_sigma_scale_validation.py — post-hoc validation of the causal EWMA
sigma correction on the WEEKLY grid (W+1/2/3), the evidence behind
weekly_multimodel.py's calibrate_sigma="on" default (adopted 2026-07-31).

Motivation: the dashboard's rolling-coverage monitoring surfaced Prophet
W+1/2/3 at 0% realized coverage over the last 30 resolutions (BTC) -- the
weekly grid was the last path running entirely uncalibrated (the sigma_scale
hooks added at 2694243 were never fed).

Method: no model is re-run. The stored, resolved weekly OOS/live rows in
tracking.db (y_pred / y_lower / y_upper / y_true) are re-scored with the same
correction the walk-forward now applies: per (model, asset, W+k) cell, ordered
by cutoff_date, sigma'_t = sigma_t * sqrt(EWMA(z^2)) with lambda=0.94, where a
z^2 only enters the state once RESOLVED (target_date <= current cutoff_date --
the k-week resolution lag, zero leakage), z measured on the RAW stored band.
Coverage of the corrected band is then compared to the raw one, overall and on
the last-30 window (the monitoring window).

Usage:  python experiments/weekly_sigma_scale_validation.py
Output: experiments/weekly_sigma_scale_validation.json
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "validation" / "tracking.db"
OUT_PATH = ROOT / "experiments" / "weekly_sigma_scale_validation.json"

Z975 = 1.959963984540054
EWMA_LAMBDA = 0.94
MODELS = ("SARIMA", "Prophet", "Naive", "LSTM", "ARIMA-GARCH")


def causal_scale_path(g: pd.DataFrame, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """sqrt(EWMA(z^2)) per row, date-causal: z_j (raw band) is ingested before
    row i iff target_date_j <= cutoff_date_i."""
    mu = g["y_pred"].values
    act = g["y_true"].values
    sig = np.maximum((g["y_upper"].values - g["y_lower"].values) / (2 * Z975), 1e-12)
    z2 = ((act - mu) / sig) ** 2
    cutoffs = pd.to_datetime(g["cutoff_date"]).values
    targets = pd.to_datetime(g["target_date"]).values
    order = np.argsort(targets, kind="stable")
    s2, j = 1.0, 0
    out = np.empty(len(g))
    for i in range(len(g)):
        while j < len(order) and targets[order[j]] <= cutoffs[i]:
            s2 = lam * s2 + (1 - lam) * z2[order[j]]
            j += 1
        out[i] = np.sqrt(s2)
    return out


def main():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT model, asset, horizon_unit, cutoff_date, target_date,
               y_pred, y_lower, y_upper, y_true
        FROM predictions
        WHERE frequence='weekly' AND horizon_type='weekly' AND y_true IS NOT NULL
              AND daily_duplicate=0 AND source IN ('oos','live')
        ORDER BY cutoff_date
        """, con)
    con.close()
    df = df[df["model"].isin(MODELS)]

    out = {"config": {"ewma_lambda": EWMA_LAMBDA, "z_measured_on": "raw band",
                      "resolution": "date-causal (target_date <= cutoff_date)"},
           "cells": [], "summary": {}}
    acc = {}
    for (model, asset, hu), g in df.groupby(["model", "asset", "horizon_unit"]):
        g = g.sort_values("cutoff_date").reset_index(drop=True)
        scale = causal_scale_path(g)
        mu, act = g["y_pred"].values, g["y_true"].values
        lo_c = mu - (mu - g["y_lower"].values) * scale
        hi_c = mu + (g["y_upper"].values - mu) * scale
        inside_raw = (act >= g["y_lower"].values) & (act <= g["y_upper"].values)
        inside_c = (act >= lo_c) & (act <= hi_c)
        cell = {
            "model": model, "asset": asset, "horizon_unit": hu, "n": int(len(g)),
            "cov95_raw": round(float(inside_raw.mean()) * 100, 1),
            "cov95_ewma": round(float(inside_c.mean()) * 100, 1),
            "cov95_raw_last30": round(float(inside_raw[-30:].mean()) * 100, 1),
            "cov95_ewma_last30": round(float(inside_c[-30:].mean()) * 100, 1),
        }
        out["cells"].append(cell)
        acc.setdefault((model, hu), []).append(cell)
    for (model, hu), cells in sorted(acc.items()):
        out["summary"][f"{model}/{hu}"] = {
            k: round(float(np.mean([c[k] for c in cells])), 1)
            for k in ("cov95_raw", "cov95_ewma", "cov95_raw_last30", "cov95_ewma_last30")}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(f"DONE -> {OUT_PATH}")


if __name__ == "__main__":
    main()
