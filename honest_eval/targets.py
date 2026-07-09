"""
targets.py — reformulated targets where signal may exist (Point 4)
==================================================================
The J+1 price *level* is essentially unpredictable (that is the whole point of
the brief).  This module re-frames the benchmark onto targets where a model can
plausibly beat chance, in priority order:

  1. Volatility  — the ARIMA-GARCH couple's real strength.  Forecast realised
     variance; score with QLIKE, MSE on variance, PIT calibration, Winkler.
     Baselines: persistence (RV_{t-1}) and RiskMetrics EWMA.
  2. Direction   — up/down as binary classification.  Score with AUC, Brier and
     a binomial test vs 50%; baseline: "always up" / majority class.
  3. Returns+exog — enrich with exogenous features (realised vol, volume,
     momentum, …) and test whether Theil's U < 1 becomes reachable.

Everything is offline-testable; the volatility GARCH path uses ``arch`` when
available and falls back to EWMA otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from . import naive as nv


# ── shared ───────────────────────────────────────────────────────────────────

def log_returns(prices) -> np.ndarray:
    p = np.asarray(prices, float).ravel()
    return np.diff(np.log(p))


def realized_variance(prices, span=1) -> np.ndarray:
    """Realised-variance proxy from squared log-returns.

    ``span=1`` → daily r²; ``span>1`` → rolling sum of r² over ``span`` days
    (a less noisy proxy).  Returned array is aligned to returns (len = n−1).
    """
    r = log_returns(prices)
    r2 = r ** 2
    if span <= 1:
        return r2
    return pd.Series(r2).rolling(span).sum().bfill().values


# ── 1. Volatility target ─────────────────────────────────────────────────────

def garch_variance_forecast(returns, test_start, refit_every=10):
    """One-step-ahead GARCH(1,1) variance forecast over [test_start, n).

    Refits every ``refit_every`` steps for speed; appends realised returns in
    between.  Falls back to EWMA if ``arch`` is unavailable or fails.
    """
    r = np.asarray(returns, float).ravel() * 100.0    # percent for numerical scale
    n = len(r)
    out = np.full(n, np.nan)
    try:
        from arch import arch_model
    except Exception:
        ew = nv.naive_vol_ewma(np.asarray(returns, float))
        out[test_start:] = ew[test_start:]
        return out

    res = None
    for t in range(test_start, n):
        if res is None or (t - test_start) % refit_every == 0:
            try:
                res = arch_model(r[:t], vol="Garch", p=1, q=1,
                                 dist="normal", rescale=False).fit(disp="off")
            except Exception:
                res = None
        if res is not None:
            try:
                fc = res.forecast(horizon=1, reindex=False)
                out[t] = fc.variance.values[-1, 0] / (100.0 ** 2)
                continue
            except Exception:
                pass
        # fallback: EWMA up to t
        out[t] = nv.naive_vol_ewma(np.asarray(returns, float)[:t + 1])[-1]
    return out


def evaluate_volatility(prices, test_ratio=0.15, refit_every=10):
    """Score GARCH / EWMA / persistence variance forecasts vs the persistence
    baseline.  Returns a dict of per-method metrics + PIT calibration.

    Verdict per method: "beats persistence" if QLIKE and MSE both improve.
    """
    prices = np.asarray(prices, float).ravel()
    r = log_returns(prices)
    rv = r ** 2                                   # realised-variance proxy
    n = len(r)
    test_start = int(n * (1 - test_ratio))
    y = rv[test_start:]
    ret_test = r[test_start:]

    persistence = nv.naive_vol_last(rv)[test_start:]
    ewma = nv.naive_vol_ewma(r)[test_start:]
    garch = garch_variance_forecast(r, test_start, refit_every=refit_every)[test_start:]

    base_qlike = metrics.qlike(y, persistence)
    base_mse = metrics.mse_variance(y, persistence)

    out = {}
    for name, pv in [("persistence", persistence), ("ewma", ewma), ("garch", garch)]:
        pv = np.asarray(pv, float)
        ql = metrics.qlike(y, pv)
        ms = metrics.mse_variance(y, pv)
        # PIT: returns ~ N(0, σ̂²) → standardised return CDF should be Uniform
        sigma = np.sqrt(np.maximum(pv, 1e-18))
        pit = metrics.pit_values(np.zeros_like(ret_test), sigma, ret_test)
        cal = metrics.pit_uniformity(pit)
        lo, hi = -1.96 * sigma, 1.96 * sigma
        wk = metrics.winkler(ret_test, lo, hi)
        cov = metrics.coverage(ret_test, lo, hi)
        beats = (name != "persistence" and ql < base_qlike and ms < base_mse)
        out[name] = {
            "qlike": round(ql, 6), "mse_var": round(ms, 10),
            "pit_ks": round(cal["ks"], 4), "pit_p": round(cal["p"], 4),
            "ret_coverage": round(cov, 2), "winkler": round(wk, 6),
            "beats_persistence": bool(beats) if name != "persistence" else None,
            "n": int(len(y)),
        }
    out["_baseline"] = {"qlike": round(base_qlike, 6), "mse_var": round(base_mse, 10)}
    return out


# ── 2. Direction target ──────────────────────────────────────────────────────

def direction_labels(prices) -> np.ndarray:
    """y_up ∈ {0,1}: 1 if the next log-return is positive."""
    return (log_returns(prices) > 0).astype(int)


def implied_prob_up(mean_path, sigma_path, last_price):
    """P(price goes up) from a Gaussian predictive law on the *level*.

    P(y > last) = 1 − Φ((last − μ)/σ).  Turns any point+PI forecast into a
    direction probability for AUC/Brier scoring.  ``last_price`` may be a scalar
    (single forecast) or an array (one previous close per origin).
    """
    from scipy.stats import norm
    mu = np.asarray(mean_path, float).ravel()
    sig = np.maximum(np.asarray(sigma_path, float).ravel(), 1e-12)
    last = np.asarray(last_price, float).ravel()
    if last.size == 1:
        last = np.full_like(mu, float(last))
    return 1.0 - norm.cdf((last - mu) / sig)


def evaluate_direction(y_up, prob_up, train_updown=None):
    """Score a P(up) vector vs 'always up' and majority-class baselines.

    Returns AUC, Brier, hit-rate + binomial p vs 50%, and the baseline Briers.
    Verdict: "beats coin" if the binomial test rejects 50% and AUC > 0.5.
    """
    y = np.asarray(y_up, int).ravel()
    p = np.asarray(prob_up, float).ravel()
    n = len(y)
    pred_up = (p >= 0.5).astype(int)
    hits = int(np.sum(pred_up == y))
    binom_p = metrics.binomial_vs_half(hits, n)
    auc = metrics.roc_auc(p, y)

    alwaysup = np.ones(n)
    maj_rate = float(np.mean(train_updown > 0)) if train_updown is not None else float(np.mean(y))
    majority = np.full(n, maj_rate)
    return {
        "n": n, "hit_rate": round(hits / n, 4) if n else float("nan"),
        "auc": round(auc, 4), "brier": round(metrics.brier(p, y), 4),
        "binom_p_vs_coin": round(binom_p, 4),
        "brier_alwaysup": round(metrics.brier(alwaysup, y), 4),
        "brier_majority": round(metrics.brier(majority, y), 4),
        "verdict": ("beats coin" if (np.isfinite(auc) and auc > 0.5 and binom_p < 0.05)
                    else "no better than coin"),
    }


# ── 3. Returns with exogenous features ───────────────────────────────────────

def build_features(prices, volume=None):
    """Causal feature matrix from the price (and optional volume) history.

    Features (all lagged, no look-ahead): lag-1/2/5 returns, 5/10-day momentum,
    realised vol (5/20), and (if given) volume z-score.  Target = next return.
    Returns (X, y, feature_names) with rows aligned so X[i] predicts y[i].
    """
    p = pd.Series(np.asarray(prices, float).ravel())
    r = np.log(p).diff()
    feats = pd.DataFrame(index=p.index)
    feats["ret_lag1"] = r.shift(1)
    feats["ret_lag2"] = r.shift(2)
    feats["ret_lag5"] = r.shift(5)
    feats["mom5"] = (p / p.shift(5) - 1).shift(1)
    feats["mom10"] = (p / p.shift(10) - 1).shift(1)
    feats["rvol5"] = r.rolling(5).std().shift(1)
    feats["rvol20"] = r.rolling(20).std().shift(1)
    if volume is not None:
        v = pd.Series(np.asarray(volume, float).ravel(), index=p.index)
        feats["vol_z"] = ((v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)).shift(1)
    target = r                                     # next-step return at row t
    df = feats.copy()
    df["_y"] = target
    df = df.dropna()
    y = df.pop("_y").values
    return df.values, y, list(df.columns)


def evaluate_returns_with_features(prices, volume=None, test_ratio=0.15,
                                   embargo=5, purge=5):
    """Ridge on exogenous features for next-return, scored vs the naive (U).

    Model selection uses purged/embargoed CV (validation.py) to avoid leakage;
    the reported Theil's U is on a final hold-out.  U < 1 ⇒ features help.
    """
    from sklearn.linear_model import Ridge
    from .validation import purged_kfold_splits

    X, y, names = build_features(prices, volume)
    n = len(y)
    split = int(n * (1 - test_ratio))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    # tune ridge alpha with purged K-fold on the training block
    best_alpha, best_cv = 1.0, np.inf
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
        errs = []
        for tr, va in purged_kfold_splits(len(ytr), n_splits=4, embargo=embargo, purge=purge):
            if len(tr) < 20 or len(va) == 0:
                continue
            m = Ridge(alpha=alpha).fit(Xtr[tr], ytr[tr])
            errs.append(float(np.mean((m.predict(Xtr[va]) - ytr[va]) ** 2)))
        cv = float(np.mean(errs)) if errs else np.inf
        if cv < best_cv:
            best_cv, best_alpha = cv, alpha

    model = Ridge(alpha=best_alpha).fit(Xtr, ytr)
    pred_ret = model.predict(Xte)
    # naive return forecast = 0 (random walk on the level)
    theil = metrics.theil_u(yte, pred_ret, np.zeros_like(yte))
    # directional skill on the sign of the predicted return
    da = metrics.directional_accuracy(pred_ret, np.zeros_like(pred_ret), yte)
    return {
        "alpha": best_alpha, "n_test": len(yte), "features": names,
        "theil_u_returns": round(theil, 4),
        "dir_acc": round(da["acc"], 4), "dir_ci95": da["ci95"], "dir_p": round(da["p_vs_coin"], 4),
        "verdict": ("features help (U<1)" if np.isfinite(theil) and theil < 1
                    else "no improvement over naive"),
    }
