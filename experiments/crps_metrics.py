"""
crps_metrics.py — empirical CRPS on a sample cloud (weekly head-to-head protocol).

Unlike honest_eval.metrics.crps_gaussian (closed-form, assumes a Gaussian
predictive law from PI width) or archives/'s crps_gaussian_approx (same idea),
TSDiff produces genuine forecast *samples* — so its CRPS is computed directly
on that sample cloud, no distributional assumption.

Estimator (Gneiting & Raftery 2007, eq. 20 — the standard unbiased-in-the-limit
empirical CRPS for an ensemble):

    CRPS(F, y) ≈ mean_i |x_i - y| - 0.5 * mean_{i,j} |x_i - x_j|

See experiments/test_crps_metrics.py for correctness checks (degenerate
ensemble, convergence to the closed-form Gaussian CRPS, cross-check against
properscoring.crps_ensemble).

`crps_fair` (BRIEF_unification_protocole_duel.md §2.3) adds the Ferro,
Richardson & Sherlock (2014) finite-ensemble bias correction: `crps_empirical`
above averages |x_i - x_j| over the FULL m*m grid (including the m zero
diagonal terms i=j), which shrinks the spread term by a factor of (m-1)/m
relative to the m*(m-1) off-diagonal pairs that actually carry dispersion
information — since CRPS SUBTRACTS that spread term, this makes the plain
estimator biased HIGH (overestimates CRPS) at finite m, an O(1/m) bias that
vanishes as m grows but is NOT identical across models unless every model
draws the exact same m (brief §2.3: "m identique pour tous"). `crps_fair`
averages over the m*(m-1) off-diagonal pairs only, removing that bias at any
m (so `crps_fair <= crps_empirical` always) — recommended whenever a model's
CRPS might be compared against a closed-form/exact one (no sampling bias at
all) rather than another sampled cloud, per the brief.
"""

import numpy as np


def crps_empirical(samples, actual: float) -> float:
    """Empirical CRPS of a forecast sample cloud against a single realised value."""
    x = np.asarray(samples, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("crps_empirical: empty sample cloud.")
    term1 = np.mean(np.abs(x - actual))
    term2 = np.mean(np.abs(x[:, None] - x[None, :]))
    return float(term1 - 0.5 * term2)


def crps_fair(samples, actual: float) -> float:
    """Fair (finite-ensemble bias-corrected) empirical CRPS — Ferro, Richardson
    & Sherlock (2014). Identical to crps_empirical except the pairwise spread
    term is averaged over the m*(m-1) off-diagonal pairs instead of the full
    m*m grid: term2_fair = term2_plain * m / (m-1). For m=1 the spread term is
    zero either way (a single-point ensemble has no pairwise information), so
    crps_fair falls back to |x - y|, same as crps_empirical."""
    x = np.asarray(samples, dtype=float).ravel()
    m = x.size
    if m == 0:
        raise ValueError("crps_fair: empty sample cloud.")
    term1 = np.mean(np.abs(x - actual))
    if m == 1:
        return float(term1)
    term2_plain = np.mean(np.abs(x[:, None] - x[None, :]))
    term2_fair = term2_plain * m / (m - 1)     # off-diagonal-only average, >= term2_plain
    return float(term1 - 0.5 * term2_fair)     # so crps_fair <= crps_empirical
