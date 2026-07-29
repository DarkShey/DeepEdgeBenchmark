"""
dist_options_common.py — shared machinery for comparing PI-calibration options
across all 5 classical models (alternatives_distributions_pi.pdf, options 1/2/3).

Design choice (documented, not hidden): every model here already produces a point
forecast + a 95% PI from its own existing walk-forward backtest
(models.<model>.run_<model>(train, test), unmodified). Rather than re-deriving a
per-step sigma path from each model's internals (which would mean rewriting 4 of
the 5 walk-forward loops), sigma is backed out from the *existing* 95% PI width:

    sigma_hat = (upper_95 - lower_95) / (2 * Z95)

This is not a new trick invented for this script -- it is the SAME convention
already used elsewhere in this repo for the same reason (comparing/rescoring
models without re-deriving their internals): honest_eval/metrics.py's
crps_gaussian callers, model_artifacts/generate_distributions_dashboard.py:71,140,
and documentation/Loi_Gaussienne_vs_Empirique.pdf §1.2. Using it here keeps this
comparison honest about being an approximation for SARIMA/Prophet/LSTM/Naive
(their true sigma path, where one exists, may differ slightly step to step) while
costing zero extra model refits. ARIMA-GARCH is the one model where a step-exact
sigma is already available natively (models.arima_model.run_arima_garch(dist=...))
and is used in preference to the backed-out approximation for its own "native"
option-1 rows.

Option 1 (extended, uniform across models): fit Student-t / GED SHAPE ONLY (loc
fixed at 0) to standardized calibration-window residuals z = (actual-pred)/sigma_hat,
via scipy MLE, then discard the fit's own scale and re-derive a unit-variance
multiplier analytically (exactly mirroring arch's own internal convention, verified
against its installed source: StudentsT/GeneralizedError normalize to unit variance
before their ppf). This isolates a pure tail-shape effect: same mu, same sigma_hat,
only the quantile multiplier changes.

Option 2 (CQR / split conformal, Romano, Patterson & Candès 2019): calibrated on
the SAME calibration window, using the model's own (Gaussian-derived) interval at
each level as the base "quantile predictor" -- exactly the intended use case per
alternatives_distributions_pi.pdf §2.3 ("prend n'importe quel modèle existant tel
quel"). Guarantees marginal coverage on the eval window regardless of the true
residual distribution.

Both options reuse the identical calibration/eval split of the walk-forward TEST
window (no held-out data beyond what's already produced by run_<model>), so no
model is ever refit for this comparison beyond what it already does on its own.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from crps_metrics import crps_empirical  # noqa: E402

Z95 = float(stats.norm.ppf(0.975))
LEVELS = (0.50, 0.80, 0.95)


# ── Sigma backed out from an existing 95% PI (see module docstring) ──────────
def sigma_from_pi95(lower, upper):
    return np.maximum((np.asarray(upper, float) - np.asarray(lower, float)) / (2 * Z95), 1e-12)


# ── Alternate distribution fits (shape-only, unit-variance normalized) ───────
#
# NOTE: scipy.stats.t.fit / gennorm.fit fit LOC, SCALE *and* SHAPE jointly by
# unconstrained MLE. On near-Gaussian residuals the shape-vs-scale likelihood
# ridge is extremely flat (any dof above ~50 is indistinguishable from Normal in
# practice), and the unconstrained optimizer regularly wanders to dof ~1e12
# instead of stopping at a sane large value -- confirmed empirically on this
# project's own SARIMA/Prophet/Naive residuals (dof fit >1e12 on data that is,
# correctly, "close to Gaussian"). Harmless in the limit (t(dof->inf) IS Normal)
# but reports a meaningless number and stresses the ppf numerics. Fixed by
# fitting shape ONLY via a bounded 1-D MLE over the already unit-variance-
# normalized density (loc=0, scale analytically fixed so variance=1 at every
# candidate shape) -- both more robust and more honest about what's actually
# being estimated (shape, not scale, per this module's whole design).
_DOF_BOUNDS = (2.05, 200.0)
_BETA_BOUNDS = (0.3, 20.0)


def _bounded_shape_mle(z, nll_fn, bounds):
    res = optimize.minimize_scalar(nll_fn, bounds=bounds, method="bounded",
                                   options={"xatol": 1e-4})
    return float(res.x)


def fit_student_t(z: np.ndarray):
    """Bounded 1-D MLE of dof on standardized residuals z, density fixed to unit
    variance at every candidate dof (loc=0). Returns (dof, ppf_fn) where
    ppf_fn(p) is itself unit-variance-normalized, so
    PI = mu + sigma_hat * ppf_fn([alpha/2, 1-alpha/2])."""
    z = np.asarray(z, float)

    def nll(dof):
        uv_scale = np.sqrt((dof - 2) / dof)
        return -np.sum(stats.t.logpdf(z / uv_scale, dof) - np.log(uv_scale))

    dof = _bounded_shape_mle(z, nll, _DOF_BOUNDS)
    uv_scale = np.sqrt((dof - 2) / dof)

    def ppf_fn(p):
        return stats.t.ppf(p, dof) * uv_scale

    return dof, ppf_fn


def fit_ged(z: np.ndarray):
    """Bounded 1-D MLE of shape (beta) on standardized residuals z via scipy's
    gennorm (loc=0, unit variance at every candidate beta) -- gennorm(beta=2) is
    exactly Normal. Same unit-variance normalization convention as arch's
    GeneralizedError.ppf (verified against its installed source)."""
    z = np.asarray(z, float)

    def nll(beta):
        uv_scale = 1.0 / np.sqrt(stats.gennorm(beta).var())
        return -np.sum(stats.gennorm.logpdf(z / uv_scale, beta) - np.log(uv_scale))

    beta = _bounded_shape_mle(z, nll, _BETA_BOUNDS)
    uv_scale = 1.0 / np.sqrt(stats.gennorm(beta).var())

    def ppf_fn(p):
        return stats.gennorm.ppf(p, beta) * uv_scale

    return beta, ppf_fn


DIST_FITTERS = {
    "student_t": fit_student_t,
    "ged": fit_ged,
}


def quantile_bounds(mu, sigma_hat, ppf_fn, levels=LEVELS):
    """{level: (lower_arr, upper_arr)} from mu +/- sigma_hat * ppf_fn(pits)."""
    out = {}
    for level in levels:
        alpha = 1.0 - level
        q_lo, q_hi = ppf_fn(np.array([alpha / 2.0, 1.0 - alpha / 2.0]))
        out[level] = (mu + sigma_hat * q_lo, mu + sigma_hat * q_hi)
    return out


# ── Option 2: split-conformal (CQR) ───────────────────────────────────────────
def conformal_correction(y_cal, lo_cal, hi_cal, alpha) -> float:
    """Q such that [lo_eval - Q, hi_eval + Q] has guaranteed marginal coverage
    (1-alpha) on exchangeable data (Romano, Patterson & Candès 2019, eq. 4-5),
    using the finite-sample-corrected empirical quantile of the conformity score
    score_i = max(lo_cal_i - y_cal_i, y_cal_i - hi_cal_i)."""
    y_cal, lo_cal, hi_cal = map(lambda a: np.asarray(a, float), (y_cal, lo_cal, hi_cal))
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def conformalize_bounds(base_bounds_cal, y_cal, base_bounds_eval, levels=LEVELS):
    """base_bounds_cal/eval: {level: (lower_arr, upper_arr)}. Returns the
    conformalized {level: (lower_arr, upper_arr)} for the eval window."""
    out = {}
    for level in levels:
        alpha = 1.0 - level
        lo_cal, hi_cal = base_bounds_cal[level]
        Q = conformal_correction(y_cal, lo_cal, hi_cal, alpha)
        lo_eval, hi_eval = base_bounds_eval[level]
        out[level] = (lo_eval - Q, hi_eval + Q)
    return out


# ── KPIs ───────────────────────────────────────────────────────────────────
def coverage_width(actual, lo, hi):
    actual, lo, hi = map(lambda a: np.asarray(a, float), (actual, lo, hi))
    cov = float(np.mean((actual >= lo) & (actual <= hi))) * 100
    width = float(np.mean(hi - lo))
    return cov, width


def pinball_loss(y, q_pred, tau):
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def avg_pinball(actual, bounds, levels=LEVELS):
    """Mean pinball loss over the 2*len(levels) quantiles making up `bounds`
    ({level: (lower_arr, upper_arr)}) -- a proper-ish scoring rule usable for any
    interval predictor (parametric swap, CQR, or MDN quantiles alike), no sample
    cloud required."""
    losses = []
    for level in levels:
        alpha = 1.0 - level
        lo, hi = bounds[level]
        losses.append(pinball_loss(actual, lo, alpha / 2.0))
        losses.append(pinball_loss(actual, hi, 1.0 - alpha / 2.0))
    return float(np.mean(losses))


def crps_from_ppf(ppf_fn, mu, sigma_hat, actual, n_samples=300, seed=0):
    """Empirical CRPS via inverse-transform sampling of ppf_fn (same trick as
    models.arima_model.run_arima_garch(n_crps_samples=...)): draws n_samples
    uniforms per point, applies ppf_fn, scales by sigma_hat, scores against
    `actual` with crps_metrics.crps_empirical (Gneiting & Raftery 2007 eq. 20)."""
    rng = np.random.default_rng(seed)
    mu, sigma_hat, actual = map(lambda a: np.asarray(a, float), (mu, sigma_hat, actual))
    scores = np.empty(len(actual))
    for i in range(len(actual)):
        pits = rng.random(n_samples)
        samples = mu[i] + sigma_hat[i] * ppf_fn(pits)
        scores[i] = crps_empirical(samples, actual[i])
    return float(np.mean(scores))


def calibration_eval_split(n_test: int, calib_frac: float = 0.4):
    """Chronological split of a walk-forward test window into a calibration
    prefix (used to fit alt-distribution shapes / conformal scores -- never
    scored) and an eval suffix (the only window any KPI is computed on)."""
    n_calib = max(int(round(n_test * calib_frac)), 20)
    n_calib = min(n_calib, n_test - 20)
    return n_calib
