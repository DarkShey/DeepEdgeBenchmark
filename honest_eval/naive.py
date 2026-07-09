"""
naive.py — correct baselines (Point 0)
======================================
Point 0 of the brief: the dashboard's "Naive" baseline was NOT the last observed
price — it had ~3% noise injected (a uniform ±5% "drift by design", std 5%/√3 ≈
2.9%), inflating its RMSE ×1.5–×10 and invalidating every "vs Naive" column and
DM test.  The injection was eventually traced to ``models/naive_model.py`` once
the external ``Run`` pipeline was merged into this repository; that file is now
a strict persistence baseline and ``models/test_naive_model.py`` +
``verify_naive`` below keep it that way.

This module provides the single source of truth for baselines so nothing can
drift again, plus ``verify_naive`` implementing the acceptance criterion:

    for every asset, the naive prediction must coincide *exactly* with the
    previous close, and |RMSE_dashboard − RMSE_recomputed| / RMSE < 0.1%.

Baselines:
  * random-walk (level)        : ŷ_t = y_{t-1}                → Points 0,1,3
  * drift random-walk          : ŷ_{t+h} = y_t + h·mean(Δtrain)
  * EWMA / last realised vol    : σ̂²_{t+1}                    → Point 4 (vol)
  * always-up / majority-class  : direction baselines          → Point 4 (dir)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── level: random walk (D+1) ─────────────────────────────────────────────────

def naive_random_walk(train: pd.Series, test: pd.Series) -> dict:
    """The corrected D+1 naive: predict the previous observed close, no noise.

    Prediction interval from the train-set 1-step change sigma (Gaussian).
    """
    actual = np.asarray(test, float).ravel()
    prev = np.concatenate([[float(train.iloc[-1])], actual[:-1]])   # y_{t-1}
    sigma = float(np.std(np.diff(np.asarray(train, float))))
    return {
        "predictions": prev,
        "prev": prev,                 # last observed price at each origin
        "actual": actual,
        "lower": prev - 1.96 * sigma,
        "upper": prev + 1.96 * sigma,
        "sigma": sigma,
        "index": test.index,
    }


def naive_drift_path(history: np.ndarray, h: int):
    """Multi-step random walk with drift: ŷ_{t+k} = y_t + k·drift.

    Returns (mean_path[h], sigma_path[h]) with sigma growing like √k.
    Used as the multi-horizon benchmark in multistep.py.
    """
    history = np.asarray(history, float).ravel()
    y_t = history[-1]
    diffs = np.diff(history)
    drift = float(np.mean(diffs)) if len(diffs) else 0.0
    step_sigma = float(np.std(diffs)) if len(diffs) else 0.0
    ks = np.arange(1, h + 1)
    mean_path = y_t + drift * ks
    sigma_path = step_sigma * np.sqrt(ks)      # random-walk error scaling
    return mean_path, sigma_path


# ── volatility baselines (Point 4) ───────────────────────────────────────────

def naive_vol_last(realised_var: np.ndarray) -> np.ndarray:
    """Persistence vol forecast: σ̂²_{t+1} = RV_t (yesterday's realised var)."""
    rv = np.asarray(realised_var, float).ravel()
    return np.concatenate([[rv[0]], rv[:-1]])


def naive_vol_ewma(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA variance forecast, σ̂²_t = λσ̂²_{t-1} + (1−λ)r²_{t-1}.

    Returns the one-step-ahead variance forecast aligned to each t.
    """
    r = np.asarray(returns, float).ravel()
    var = np.empty(len(r))
    var[0] = r[0] ** 2
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var


# ── direction baselines (Point 4) ────────────────────────────────────────────

def naive_direction_alwaysup(n: int) -> np.ndarray:
    """The 'always predict up' baseline probability vector."""
    return np.ones(int(n))


def naive_direction_majority(train_updown: np.ndarray, n: int) -> np.ndarray:
    """Majority-class baseline: constant P(up) = empirical up-rate on train."""
    p = float(np.mean(np.asarray(train_updown, float) > 0))
    return np.full(int(n), p)


# ── acceptance verifier (Point 0) ────────────────────────────────────────────

def verify_naive(train: pd.Series, test: pd.Series,
                 dashboard_predictions=None, dashboard_rmse=None,
                 tol_frac: float = 1e-3) -> dict:
    """Check a naive baseline against the corrected random walk.

    Acceptance criterion (Point 0):
      * the naive prediction equals the previous close *exactly*;
      * |RMSE_dashboard − RMSE_recomputed| / RMSE_recomputed < tol_frac (0.1%).

    ``dashboard_predictions`` / ``dashboard_rmse`` are optional: pass what the
    external pipeline produced to audit it.  Returns a report dict with a
    boolean ``passed`` and the recomputed reference RMSE.
    """
    ref = naive_random_walk(train, test)
    actual = ref["actual"]
    ref_rmse = float(np.sqrt(np.mean((actual - ref["predictions"]) ** 2)))

    report = {"ref_rmse": round(ref_rmse, 6), "passed": True, "issues": []}

    if dashboard_predictions is not None:
        dp = np.asarray(dashboard_predictions, float).ravel()
        if dp.shape != ref["predictions"].shape:
            report["passed"] = False
            report["issues"].append(
                f"length mismatch: dashboard {dp.shape} vs ref {ref['predictions'].shape}")
        else:
            max_abs = float(np.max(np.abs(dp - ref["predictions"])))
            report["max_abs_pred_diff"] = round(max_abs, 8)
            # "exactly the previous close" — allow float round-off only
            if max_abs > 1e-6 * max(1.0, float(np.mean(np.abs(actual)))):
                report["passed"] = False
                report["issues"].append(
                    f"predictions differ from previous close (max |Δ|={max_abs:.4g}) "
                    f"— noise injection suspected")
            dash_rmse = float(np.sqrt(np.mean((actual - dp) ** 2)))
            report["dashboard_rmse"] = round(dash_rmse, 6)

    if dashboard_rmse is not None:
        rel = abs(dashboard_rmse - ref_rmse) / max(ref_rmse, 1e-12)
        report["rmse_rel_error"] = round(rel, 6)
        if rel >= tol_frac:
            report["passed"] = False
            report["issues"].append(
                f"RMSE relative error {rel*100:.3f}% ≥ {tol_frac*100:.3f}% "
                f"(dashboard {dashboard_rmse:.4g} vs ref {ref_rmse:.4g})")

    return report
