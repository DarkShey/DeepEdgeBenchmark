"""
validation.py — robust validation schemes (Point 2)
===================================================
Walk-forward (rolling origin) stays the main scheme, but the brief asks us to
(a) test its robustness (expanding vs fixed-width rolling windows), (b) use
purged + embargoed blocked CV for any hyper-parameter tuning (never random
K-fold — Lopez de Prado), and (c) report metrics per sub-period / regime, not
just the global average.

Everything here yields *index arrays* into the price series so it is agnostic to
the model.  A generic ``evaluate_windows`` runs any ``fit_predict`` callable
over a chosen scheme and returns per-origin errors for downstream metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── walk-forward splitters ───────────────────────────────────────────────────

def walk_forward_splits(n, test_start, window=None, step=1, horizon=1):
    """Yield (train_idx, test_idx) for a rolling-origin walk-forward.

    ``window=None`` → expanding window (train = [0, origin]).
    ``window=k``    → fixed-width rolling window (train = [origin−k, origin]).
    ``step``        → origins stride (1 = daily / dense).
    ``horizon``     → forecast h steps; test_idx has length min(h, remaining).

    Origins run from ``test_start`` to ``n−horizon``.
    """
    for origin in range(test_start, n - horizon + 1, step):
        if window is None:
            train_idx = np.arange(0, origin)
        else:
            train_idx = np.arange(max(0, origin - window), origin)
        if len(train_idx) == 0:
            continue
        test_idx = np.arange(origin, min(origin + horizon, n))
        yield train_idx, test_idx


@dataclass
class WindowComparison:
    """Result of expanding-vs-rolling stability comparison (Point 2)."""
    label: str
    rmse: float
    mae: float
    n: int


def evaluate_windows(prices: np.ndarray, fit_predict, test_start,
                     window=None, step=1, horizon=1):
    """Run ``fit_predict(train_prices, h) -> pred_path`` over a walk-forward.

    Returns a dict with stacked per-origin arrays (prev, actual, pred, errors)
    for h == 1, or the full h-step paths otherwise.  Model-agnostic: ``fit_predict``
    can wrap ARIMA/SARIMA/etc. or a cheap analytic forecaster in tests.
    """
    prices = np.asarray(prices, float).ravel()
    n = len(prices)
    prevs, actuals, preds = [], [], []
    for train_idx, test_idx in walk_forward_splits(
            n, test_start, window=window, step=step, horizon=horizon):
        train_prices = prices[train_idx]
        h = len(test_idx)
        path = np.asarray(fit_predict(train_prices, h), float).ravel()[:h]
        prevs.append(prices[train_idx[-1]])
        actuals.append(prices[test_idx[-1]])          # h-step target
        preds.append(path[-1])
    prevs = np.asarray(prevs); actuals = np.asarray(actuals); preds = np.asarray(preds)
    err = actuals - preds
    return {
        "prev": prevs, "actual": actuals, "pred": preds, "error": err,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "n": len(err),
    }


def compare_windows(prices, fit_predict, test_start, windows=(None, 250, 500, 750),
                    step=1, horizon=1):
    """Expanding vs fixed rolling windows.  Large spread ⇒ unstable parameters.

    Returns a list of WindowComparison, one per window setting.
    """
    out = []
    for w in windows:
        res = evaluate_windows(prices, fit_predict, test_start,
                               window=w, step=step, horizon=horizon)
        label = "expanding" if w is None else f"rolling-{w}"
        out.append(WindowComparison(label, round(res["rmse"], 4),
                                    round(res["mae"], 4), res["n"]))
    return out


# ── purged + embargoed blocked CV (Lopez de Prado, Point 2) ──────────────────

def purged_kfold_splits(n, n_splits=5, embargo=0, purge=0):
    """Blocked K-fold with purge + embargo for time-series hyper-param tuning.

    Contiguous test blocks (no shuffling).  Around each test block we *purge*
    ``purge`` samples on each side and *embargo* ``embargo`` samples after it, so
    train labels whose information window overlaps the test block are dropped —
    preventing leakage from serial correlation / overlapping targets.

    Yields (train_idx, test_idx).  ``purge``/``embargo`` should be ≥ horizon−1
    for h-step targets.
    """
    n = int(n)
    fold_edges = np.linspace(0, n, n_splits + 1).astype(int)
    all_idx = np.arange(n)
    for i in range(n_splits):
        t0, t1 = fold_edges[i], fold_edges[i + 1]
        test_idx = np.arange(t0, t1)
        left = t0 - purge
        right = t1 + purge + embargo
        mask = (all_idx < left) | (all_idx >= right)
        train_idx = all_idx[mask]
        yield train_idx, test_idx


def cv_score(prices, fit_predict_score, n_splits=5, embargo=0, purge=0):
    """Average a scalar score over purged/embargoed folds (for tuning)."""
    prices = np.asarray(prices, float).ravel()
    scores = []
    for train_idx, test_idx in purged_kfold_splits(
            len(prices), n_splits, embargo=embargo, purge=purge):
        scores.append(float(fit_predict_score(prices[train_idx], prices[test_idx])))
    return float(np.mean(scores)), scores


# ── sub-period / regime reporting (Point 2) ──────────────────────────────────

def subperiod_report(index, actual, pred, by="quarter"):
    """Break errors down by sub-period so stability is visible, not just the mean.

    ``by`` ∈ {"quarter", "year", "month"} or an array of labels aligned to the
    errors.  Returns a DataFrame of per-bucket RMSE/MAE/n.
    """
    actual = np.asarray(actual, float).ravel()
    pred = np.asarray(pred, float).ravel()
    err = actual - pred
    if isinstance(by, str):
        idx = pd.DatetimeIndex(index)
        if by == "quarter":
            labels = idx.to_period("Q").astype(str)
        elif by == "year":
            labels = idx.year.astype(str)
        elif by == "month":
            labels = idx.to_period("M").astype(str)
        else:
            raise ValueError(f"unknown period {by!r}")
    else:
        labels = np.asarray(by).astype(str)
    df = pd.DataFrame({"bucket": labels, "err": err})
    g = df.groupby("bucket")["err"]
    out = pd.DataFrame({
        "rmse": g.apply(lambda e: float(np.sqrt(np.mean(e**2)))),
        "mae": g.apply(lambda e: float(np.mean(np.abs(e)))),
        "n": g.size(),
    })
    return out


def volatility_regimes(prices, window=20, n_regimes=3):
    """Label each point by rolling-volatility tercile (low/med/high regime).

    Returns a string array aligned to ``prices`` for use as ``by`` above.
    """
    p = pd.Series(np.asarray(prices, float).ravel())
    ret = np.log(p).diff()
    vol = ret.rolling(window).std()
    q = vol.rank(pct=True)
    names = np.array(["low", "med", "high"]) if n_regimes == 3 else \
        np.array([f"q{i}" for i in range(n_regimes)])
    bins = np.clip((q * n_regimes).fillna(0).astype(int).values, 0, n_regimes - 1)
    return names[bins]
