"""
prob_kpi_common.py — shared plumbing for BRIEF_kpi_probabilistes.md.

Reuses, never duplicates: experiments.crps_metrics.crps_empirical (the
project's existing empirical-CRPS estimator) and honest_eval.metrics.mase
(the project's existing MASE definition: MAE(model)/MAE(naive) on the *same*
evaluation set, not an in-sample scale).

Two families of sample generation, both N=500, both zero-lookahead:

1. Parametric (ARIMA-GARCH, SARIMA, Prophet, Naive, LSTM) — none of these
   models' raw sample clouds were ever persisted (tracking.db stores only
   point + 95% PI), and none has a reloadable checkpoint for every historical
   origin. But each constructs its 95% PI as an explicit mu +/- z*sigma band
   (see models/*.py) -- ARIMA-GARCH in log-return space (lognormal price),
   the other four directly in price space (Gaussian) -- so sigma is exactly
   recoverable from the already-stored (y_pred, y_lower, y_upper) and N
   samples can be drawn from that same distribution with zero retraining.
   This *is* "tirer N echantillons de leur distribution predictive
   (parametrique)" (brief Sec.3), not an approximation invented for this task.

2. Native (TSDiff) — see generate_samples_tsdiff.py. No checkpoint exists
   either (model_artifacts/pipeline.py never serializes one for TSDiff), so
   native resampling requires literally re-running the same train-once,
   walk-forward protocol that produced the current matrix (same frozen/live
   price data, same seed=42, same per-asset epoch already selected), just
   keeping the N=500 sample cloud instead of collapsing it to mean/quantiles.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

DEFAULT_DB_PATH = str(ROOT / "validation" / "tracking.db")

Z95 = float(stats.norm.ppf(0.975))   # 1.959963985 -- precise, not the hand-rolled 1.96

PARAMETRIC_MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "Naive", "LSTM")
NATIVE_MODELS = ("TSDiff",)
ALL_MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "Naive", "LSTM", "TSDiff")

HORIZON_LABEL_ORDER = ["D+1", "D+7", "W+1", "W+2", "W+3"]


# ── DB access ─────────────────────────────────────────────────────────────────

def load_matrix_rows(asset: str, models=None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """The 'current matrix' rows for one asset: source in (oos, live),
    daily_duplicate=0 (matches all_predictions / the dashboard), realised
    (y_true not null, otherwise there's nothing to score a sample cloud
    against). One row per (model, frequence, horizon_type, horizon_unit,
    cutoff_date) -- exactly the granularity the brief calls a cell x origin.
    """
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, horizon, frequence, horizon_type, horizon_unit,
                   cutoff_date, target_date, last_close, y_pred, y_lower, y_upper,
                   y_true, source
            FROM predictions
            WHERE asset = ? AND daily_duplicate = 0 AND source IN ('oos', 'live')
                  AND y_true IS NOT NULL
            """,
            con, params=(asset,),
        )
    finally:
        con.close()
    if models is not None:
        df = df[df["model"].isin(models)]
    df = df.sort_values(["model", "frequence", "horizon_type", "horizon_unit", "cutoff_date"])
    df = df.reset_index(drop=True)
    return df


def horizon_label(row) -> str:
    """D+1 / D+7 / W+1 / W+2 / W+3 -- the brief's 5-horizon axis, collapsing
    the frequence dimension (daily-native vs weekly-native origin of W+k)."""
    return row["horizon_unit"]


# ── parametric sampling (models 1-5) ──────────────────────────────────────────

def sample_parametric(model: str, y_pred: float, y_lower: float, y_upper: float,
                       last_close: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """N samples from the model's own predictive distribution, as implied by
    its stored (y_pred, y_lower, y_upper) 95% PI -- see models/*.py for each
    model's construction (checked directly, not assumed):
      - ARIMA-GARCH: pred = last*exp(mu), lower/upper = last*exp(mu -/+ z*sigma)
        -> lognormal in price space, sigma recovered in LOG space.
      - SARIMA / Prophet / Naive / LSTM: pred +/- z*sigma directly in price
        space -> Gaussian, sigma recovered in PRICE space.
    """
    if model == "ARIMA-GARCH":
        log_pred = np.log(y_pred / last_close)
        log_upper = np.log(y_upper / last_close)
        log_lower = np.log(y_lower / last_close)
        sigma_log = (log_upper - log_lower) / (2.0 * Z95)
        sigma_log = max(sigma_log, 1e-10)
        draws = rng.normal(loc=log_pred, scale=sigma_log, size=n_samples)
        return last_close * np.exp(draws)

    if model in ("SARIMA", "Prophet", "Naive", "LSTM"):
        sigma = (y_upper - y_lower) / (2.0 * Z95)
        sigma = max(sigma, 1e-10)
        return rng.normal(loc=y_pred, scale=sigma, size=n_samples)

    raise ValueError(f"sample_parametric: not a parametric model here: {model!r}")


# ── KPI functions on a sample cloud ───────────────────────────────────────────

def crps_from_samples(samples: np.ndarray, actual: float) -> float:
    from crps_metrics import crps_empirical
    return crps_empirical(samples, actual)


def coverage_flag(samples: np.ndarray, actual: float, level: float) -> bool:
    """Is `actual` inside the central `level` interval read off the sample
    quantiles (e.g. level=0.80 -> [10%, 90%] quantiles)."""
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return bool(lo <= actual <= hi)


def sharpness(samples: np.ndarray, level: float) -> float:
    """Interval width at `level` coverage -- finesse read at equal coverage."""
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(hi - lo)


def winkler_score(samples: np.ndarray, actual: float, level: float) -> float:
    """Winkler / interval score at `level` (Gneiting & Raftery 2007, eq. 43):
    width + penalty (2/alpha) for the observation falling outside the
    interval, on the miss side only. Lower is better."""
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    width = hi - lo
    if actual < lo:
        return float(width + (2.0 / alpha) * (lo - actual))
    if actual > hi:
        return float(width + (2.0 / alpha) * (actual - hi))
    return float(width)


def pit_value(samples: np.ndarray, actual: float) -> float:
    """Empirical PIT: F_hat(y_true) = fraction of the sample cloud <= y_true.
    A perfectly calibrated model has PIT ~ Uniform(0,1) pooled across rows."""
    return float(np.mean(samples <= actual))


def row_kpis(samples: np.ndarray, actual: float) -> dict:
    return {
        "crps": crps_from_samples(samples, actual),
        "pit": pit_value(samples, actual),
        "sample_mean": float(np.mean(samples)),
        **{f"cov{int(l*100)}": coverage_flag(samples, actual, l) for l in (0.5, 0.8, 0.95)},
        **{f"sharp{int(l*100)}": sharpness(samples, l) for l in (0.5, 0.8, 0.95)},
        **{f"winkler{int(l*100)}": winkler_score(samples, actual, l) for l in (0.5, 0.8, 0.95)},
    }
