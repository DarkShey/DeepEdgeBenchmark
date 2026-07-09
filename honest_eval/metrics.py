"""
metrics.py — honest forecast metrics (Points 1, 3, 4)
=====================================================
The dashboard's core dishonesty was scoring *levels* under a re-anchoring
walk-forward: every model looked good because ``pred_t ≈ price_{t-1}``.  These
metrics score the *change* a model predicts (``Δpred_t = pred_t − price_{t-1}``)
against the realised change (``Δreal_t = price_t − price_{t-1}``), and measure
everything relative to the corrected naive (random-walk) baseline.

Reading rule (Point 1): if Theil's U ≈ 1 and the Diebold-Mariano test is not
significant, the model adds nothing over the naive — say so explicitly.

All functions are pure numpy/scipy; ``roc_auc`` uses scikit-learn.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_pair(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def changes(pred, prev, actual):
    """Return (Δpred, Δreal) = (pred − prev, actual − prev).

    ``prev`` is the last *observed* price at each origin (price_{t-1} for D+1).
    For a naive/random-walk forecast pred == prev so Δpred == 0.
    """
    pred   = np.asarray(pred,   float).ravel()
    prev   = np.asarray(prev,   float).ravel()
    actual = np.asarray(actual, float).ravel()
    return pred - prev, actual - prev


# ── point metrics on levels ──────────────────────────────────────────────────

def rmse(actual, pred):
    a, p = _clean_pair(actual, pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, pred):
    a, p = _clean_pair(actual, pred)
    return float(np.mean(np.abs(a - p)))


# ── skill vs naive (Point 1) ─────────────────────────────────────────────────

def theil_u(actual, pred, naive_pred):
    """Theil's U2 = RMSE(model) / RMSE(naive).

    U < 1 : model beats the random walk.  U ≈ 1 : no skill.  U > 1 : worse.
    """
    denom = rmse(actual, naive_pred)
    if denom == 0:
        return float("nan")
    return rmse(actual, pred) / denom


def mase(actual, pred, naive_pred):
    """Relative MAE = MAE(model) / MAE(naive) (the brief's MASE definition).

    Uses the *same evaluation set* naive rather than in-sample scaling so the
    number is directly comparable to Theil's U on that set.
    """
    denom = mae(actual, naive_pred)
    if denom == 0:
        return float("nan")
    return mae(actual, pred) / denom


def change_correlation(pred, prev, actual):
    """Pearson corr between predicted change and realised change.

    ≈ 0 means the model has no directional information about the *move*
    (the headline finding of the dashboard analysis).  Returns nan when the
    predicted change has no variance (e.g. the naive, Δpred ≡ 0).
    """
    dpred, dreal = changes(pred, prev, actual)
    m = np.isfinite(dpred) & np.isfinite(dreal)
    dpred, dreal = dpred[m], dreal[m]
    if len(dpred) < 3 or np.std(dpred) < 1e-12 or np.std(dreal) < 1e-12:
        return float("nan")
    return float(np.corrcoef(dpred, dreal)[0, 1])


# ── directional accuracy with binomial inference ─────────────────────────────

def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def directional_accuracy(pred, prev, actual):
    """Directional accuracy of the *predicted move* with a 95% binomial CI.

    Returns a dict: hit-rate, n, Wilson 95% CI, and the two-sided binomial
    p-value against the 50% coin-flip null.  Ties (Δ == 0) are dropped.
    """
    dpred, dreal = changes(pred, prev, actual)
    m = np.isfinite(dpred) & np.isfinite(dreal)
    sp, sr = np.sign(dpred[m]), np.sign(dreal[m])
    valid = (sp != 0) & (sr != 0)
    sp, sr = sp[valid], sr[valid]
    n = int(len(sp))
    if n == 0:
        return {"acc": float("nan"), "n": 0, "ci95": (float("nan"), float("nan")),
                "p_vs_coin": float("nan"), "hits": 0}
    hits = int(np.sum(sp == sr))
    lo, hi = _wilson_ci(hits, n)
    p = float(stats.binomtest(hits, n, 0.5, alternative="two-sided").pvalue)
    return {"acc": hits / n, "n": n, "ci95": (lo, hi), "p_vs_coin": p, "hits": hits}


# ── Diebold-Mariano with Newey-West / HAC variance (Points 1, 3) ─────────────

def diebold_mariano(errors_model, errors_bench, h=1, loss="squared", power=2):
    """Diebold-Mariano test with Newey-West (Bartlett) HAC variance.

    ``errors_*`` are forecast errors (actual − pred).  ``h`` is the forecast
    horizon; for overlapping multi-step forecasts the loss differential is
    autocorrelated up to lag h−1, so the HAC truncation is set to h−1 (at least
    ``T**(1/3)``).  Includes the Harvey-Leybourne-Newbold small-sample
    correction and reports a Student-t p-value (T−1 df).

    Convention: DM < 0 ⇒ the model has lower loss than the benchmark (better).
    Returns (DM_stat, p_value, lag_used).
    """
    ea = np.asarray(errors_model, float).ravel()
    eb = np.asarray(errors_bench, float).ravel()
    ea, eb = _clean_pair(ea, eb)
    T = len(ea)
    if T < 8:
        return (0.0, 1.0, 0)

    if loss == "squared":
        d = ea**2 - eb**2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    elif loss == "power":
        d = np.abs(ea)**power - np.abs(eb)**power
    else:
        raise ValueError(f"unknown loss {loss!r}")

    dbar = float(np.mean(d))
    lag = max(int(h) - 1, int(np.floor(T ** (1 / 3))))
    lag = min(lag, T - 1)

    # long-run variance (Newey-West, Bartlett kernel)
    gamma0 = float(np.mean((d - dbar) ** 2))
    var = gamma0
    for k in range(1, lag + 1):
        cov = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        var += 2.0 * (1.0 - k / (lag + 1)) * cov
    if var <= 0:
        return (0.0, 1.0, lag)

    dm = dbar / np.sqrt(var / T)
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    corr = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    dm *= corr
    p = 2.0 * stats.t.cdf(-abs(dm), df=T - 1)
    return (float(dm), float(p), int(lag))


# ── interval / probabilistic metrics (Points 3, 4) ───────────────────────────

def coverage(actual, lower, upper):
    a = np.asarray(actual, float).ravel()
    lo = np.asarray(lower, float).ravel()
    hi = np.asarray(upper, float).ravel()
    return float(np.mean((a >= lo) & (a <= hi)) * 100.0)


def avg_width(lower, upper):
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def winkler(actual, lower, upper, alpha=0.05):
    a = np.asarray(actual, float).ravel()
    lo = np.asarray(lower, float).ravel()
    hi = np.asarray(upper, float).ravel()
    width = hi - lo
    pen_lo = np.where(a < lo, (2 / alpha) * (lo - a), 0.0)
    pen_hi = np.where(a > hi, (2 / alpha) * (a - hi), 0.0)
    return float(np.mean(width + pen_lo + pen_hi))


def crps_gaussian(mu, sigma, actual):
    """Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma)."""
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), 1e-12)
    y = np.asarray(actual, float).ravel()
    z = (y - mu) / sigma
    return float(np.mean(
        sigma * (z * (2 * stats.norm.cdf(z) - 1)
                 + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))


def pit_values(mu, sigma, actual):
    """Probability Integral Transform F(y) under a Gaussian predictive law.

    A calibrated model gives PIT ~ Uniform(0,1).  Returned array feeds a
    KS-test / histogram in report.py.
    """
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), 1e-12)
    y = np.asarray(actual, float).ravel()
    return stats.norm.cdf((y - mu) / sigma)


def pit_uniformity(pit):
    """KS statistic and p-value of PIT vs Uniform(0,1) (calibration test)."""
    pit = np.asarray(pit, float).ravel()
    pit = pit[np.isfinite(pit)]
    if len(pit) < 5:
        return {"ks": float("nan"), "p": float("nan"), "n": len(pit)}
    ks, p = stats.kstest(pit, "uniform")
    return {"ks": float(ks), "p": float(p), "n": int(len(pit))}


# ── volatility target metrics (Point 4) ──────────────────────────────────────

def qlike(realised_var, pred_var):
    """QLIKE loss: mean( log σ̂² + RV / σ̂² ).  Lower is better.

    Robust to volatility-proxy noise; the standard loss for variance forecasts.
    """
    rv = np.asarray(realised_var, float).ravel()
    pv = np.maximum(np.asarray(pred_var, float).ravel(), 1e-18)
    m = np.isfinite(rv) & np.isfinite(pv)
    return float(np.mean(np.log(pv[m]) + rv[m] / pv[m]))


def mse_variance(realised_var, pred_var):
    rv = np.asarray(realised_var, float).ravel()
    pv = np.asarray(pred_var, float).ravel()
    m = np.isfinite(rv) & np.isfinite(pv)
    return float(np.mean((rv[m] - pv[m]) ** 2))


# ── direction target metrics (Point 4) ───────────────────────────────────────

def brier(prob_up, y_up):
    """Brier score: mean( (p − y)² ), y ∈ {0,1}.  Lower is better (0.25 = coin)."""
    p = np.asarray(prob_up, float).ravel()
    y = np.asarray(y_up, float).ravel()
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2))


def roc_auc(prob_up, y_up):
    """ROC-AUC of P(up) against the realised up/down label. 0.5 = no skill."""
    from sklearn.metrics import roc_auc_score
    p = np.asarray(prob_up, float).ravel()
    y = np.asarray(y_up, float).ravel()
    m = np.isfinite(p) & np.isfinite(y)
    y, p = y[m], p[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def binomial_vs_half(hits, n):
    """Two-sided binomial p-value of a hit-count against the 50% null."""
    if n == 0:
        return float("nan")
    return float(stats.binomtest(int(hits), int(n), 0.5, alternative="two-sided").pvalue)


# ── verdict helper (Point 1 reading rule) ────────────────────────────────────

def skill_verdict(theil, dm_p, alpha=0.05):
    """Turn (Theil U, DM p-value) into an explicit human verdict."""
    if not np.isfinite(theil):
        return "n/a"
    if dm_p < alpha and theil < 1:
        return "beats naive"
    if dm_p < alpha and theil > 1:
        return "worse than naive"
    return "no better than naive"
