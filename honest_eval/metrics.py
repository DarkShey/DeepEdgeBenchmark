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

def dm_hac_test(diff, h=1) -> dict:
    """Diebold-Mariano statistic on an ALREADY-COMPUTED loss differential series
    (extracted from diebold_mariano() so callers that pool/scale their own loss
    differential across assets before testing -- e.g. a MASE-scaled or CRPS-
    normalised differential, not raw dollar errors -- can reuse the exact same
    Newey-West/HLN machinery without going through diebold_mariano()'s "compute
    loss from two raw error series" step, cf. experiments/pooled_analysis.py).

    ``diff`` = loss(model_A) - loss(model_B) per (ordered, chronological)
    observation -- already whatever loss/scale the caller wants (squared error,
    absolute error, MASE-scaled error, ...). ``h`` is the forecast horizon;
    autocorrelation is truncated at lag h-1 (at least T**(1/3)), matching
    diebold_mariano()'s convention. Includes the Harvey-Leybourne-Newbold (1997)
    small-sample correction and a Student-t p-value (T-1 df).

    Convention: mean(diff) < 0 => model_A has lower loss (better). Returns a
    dict: dm_stat, p_value, lag, mean_diff, n."""
    d = np.asarray(diff, float).ravel()
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 8:
        return {"dm_stat": 0.0, "p_value": 1.0, "lag": 0, "mean_diff": float(np.mean(d)) if T else float("nan"), "n": T}

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
        return {"dm_stat": 0.0, "p_value": 1.0, "lag": lag, "mean_diff": dbar, "n": T}

    dm = dbar / np.sqrt(var / T)
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    corr = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    dm *= corr
    p = 2.0 * stats.t.cdf(-abs(dm), df=T - 1)
    return {"dm_stat": float(dm), "p_value": float(p), "lag": int(lag), "mean_diff": dbar, "n": T}


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

    out = dm_hac_test(d, h=h)
    return (out["dm_stat"], out["p_value"], out["lag"])


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


# ── distribution-aware CRPS / PIT (calibration follow-up, 2026-07) ───────────
# The historical convention throughout the repo (crps_gaussian callers,
# prob_kpi_common.sample_parametric) reconstructs every predictive law as a
# SYMMETRIC Gaussian from the stored 95% PI width. Since the sigma-calibration
# adoption (skew-t ARIMA-GARCH innovations, log-space Prophet), stored bounds
# can be asymmetric, and scoring them with a symmetric closed form is exactly
# the incoherence flagged in HANDOFF_dist_options_comparison.md. These helpers
# score the law the model actually used. crps_gaussian/pit_values above are
# kept unchanged for backward compatibility.

def crps_student_t(mu, scale, actual, dof):
    """Closed-form CRPS for a Student-t predictive law (Gneiting & Raftery
    2007, table 1). `scale` is the t scale parameter (NOT the std)."""
    from scipy.special import beta as beta_fn
    mu = np.asarray(mu, float).ravel()
    scale = np.maximum(np.asarray(scale, float).ravel(), 1e-12)
    y = np.asarray(actual, float).ravel()
    z = (y - mu) / scale
    Fz = stats.t.cdf(z, dof)
    fz = stats.t.pdf(z, dof)
    term = (z * (2 * Fz - 1) + 2 * fz * (dof + z ** 2) / (dof - 1)
            - (2 * np.sqrt(dof) / (dof - 1))
            * beta_fn(0.5, dof - 0.5) / beta_fn(0.5, dof / 2.0) ** 2)
    return float(np.mean(scale * term))


def crps_lognormal(mu_log, sigma_log, actual):
    """Closed-form CRPS for a lognormal predictive law: Y = exp(X),
    X ~ N(mu_log, sigma_log) (Baran & Lerch 2015, eq. 5). This is the exact
    score for ARIMA-GARCH (normal innovations) and log-space Prophet prices."""
    mu = np.asarray(mu_log, float).ravel()
    s = np.maximum(np.asarray(sigma_log, float).ravel(), 1e-12)
    y = np.maximum(np.asarray(actual, float).ravel(), 1e-300)
    w = (np.log(y) - mu) / s
    ex = np.exp(mu + s * s / 2.0)
    return float(np.mean(
        y * (2 * stats.norm.cdf(w) - 1)
        - 2 * ex * (stats.norm.cdf(w - s) + stats.norm.cdf(s / np.sqrt(2)) - 1)
    ))


def _crps_empirical_sorted(samples, y):
    """O(n log n) empirical CRPS (Gneiting & Raftery 2007, eq. 20) for one
    observation -- local, to keep honest_eval dependency-free."""
    x = np.sort(np.asarray(samples, float).ravel())
    n = len(x)
    mean_abs = float(np.mean(np.abs(x - y)))
    # E|X - X'| via the sorted-sample identity
    coef = 2.0 * np.arange(1, n + 1) - n - 1
    mean_pair = 2.0 * float(np.sum(coef * x)) / (n * n)
    return mean_abs - 0.5 * mean_pair


def crps_parametric(mu, sigma, actual, dist="normal", shape=None,
                    ppf_fn=None, n_samples=512, seed=0):
    """CRPS under the predictive law the model ACTUALLY used.

    mu/sigma: location and scale path (arrays or scalars); for dist='lognormal'
    they are the LOG-space mean and std. dist: 'normal' | 'student_t' (shape =
    dof) | 'ged' (shape = beta, unit-variance convention as in
    experiments/dist_options_common.fit_ged) | 'lognormal' | 'custom'
    (provide ppf_fn(p) -> unit-variance standardized quantiles, e.g. an arch
    skew-t ppf closure -- scored by inverse-transform sampling)."""
    if dist == "normal":
        return crps_gaussian(mu, sigma, actual)
    if dist == "lognormal":
        return crps_lognormal(mu, sigma, actual)
    if dist == "student_t":
        dof = float(shape)
        scale = np.asarray(sigma, float) * np.sqrt((dof - 2.0) / dof)
        return crps_student_t(mu, scale, actual, dof)
    if dist == "ged":
        beta = float(shape)
        uv_scale = 1.0 / np.sqrt(stats.gennorm(beta).var())
        ppf_fn = lambda p: stats.gennorm.ppf(p, beta) * uv_scale  # noqa: E731
    if ppf_fn is None:
        raise ValueError(f"crps_parametric: unsupported dist {dist!r} without ppf_fn")
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), 1e-12)
    y = np.asarray(actual, float).ravel()
    rng = np.random.default_rng(seed)
    scores = np.empty(len(y))
    for i in range(len(y)):
        draws = mu[i] + sigma[i] * ppf_fn(rng.random(n_samples))
        scores[i] = _crps_empirical_sorted(draws, y[i])
    return float(np.mean(scores))


def pit_parametric(mu, sigma, actual, dist="normal", shape=None, cdf_fn=None):
    """PIT under the actual predictive law (generalizes pit_values)."""
    mu = np.asarray(mu, float).ravel()
    sigma = np.maximum(np.asarray(sigma, float).ravel(), 1e-12)
    y = np.asarray(actual, float).ravel()
    z = (y - mu) / sigma
    if dist == "normal":
        return stats.norm.cdf(z)
    if dist == "lognormal":
        return stats.norm.cdf((np.log(np.maximum(y, 1e-300)) - mu) / sigma)
    if dist == "student_t":
        dof = float(shape)
        return stats.t.cdf(z / np.sqrt((dof - 2.0) / dof), dof)
    if dist == "ged":
        beta = float(shape)
        uv = 1.0 / np.sqrt(stats.gennorm(beta).var())
        return stats.gennorm.cdf(z / uv, beta)
    if cdf_fn is not None:
        return cdf_fn(z)
    raise ValueError(f"pit_parametric: unsupported dist {dist!r} without cdf_fn")
