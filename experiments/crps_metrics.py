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
