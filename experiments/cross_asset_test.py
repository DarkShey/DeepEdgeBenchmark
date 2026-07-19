"""
cross_asset_test.py — combine per-asset paired-test results into one panel-level
verdict (BRIEF weekly multi-asset extension: "test global inter-actifs").

Two pieces:
  - merge_correlated_diffs(): collapse two highly-correlated exposures (e.g. two
    Treasury instruments, ZN=F futures + TLT ETF) into ONE composite diff series
    before they enter the aggregate, so the panel isn't double-counting one
    underlying risk factor as two independent data points.
  - stouffer_combine(): Stouffer's Z-score meta-analysis, combining independent
    (post-merge) per-bucket paired_bootstrap_test results into a single combined
    p-value — standard, appropriate for combining a handful of small-sample
    tests, and more powerful than a sign test at k=4 (a 4-bucket sign test can
    never reach p<0.05: best case is 2*0.5**4=0.125 two-sided).
"""

import numpy as np
from scipy import stats


def merge_correlated_diffs(diff_a, diff_b):
    """Elementwise mean of two paired (same origin/horizon ordering) per-origin
    diff series — collapses two correlated exposures into one composite signal.
    Both inputs must already be aligned (same length, same origin/horizon order)."""
    diff_a = np.asarray(diff_a, dtype=float)
    diff_b = np.asarray(diff_b, dtype=float)
    if diff_a.shape != diff_b.shape:
        raise ValueError(f"merge_correlated_diffs: shape mismatch {diff_a.shape} vs {diff_b.shape}")
    return (diff_a + diff_b) / 2.0


def stouffer_combine(bucket_tests: dict) -> dict:
    """Combine bucket-level paired_bootstrap_test() results (dict: label ->
    {"mean_diff":..., "p_value":...}) via Stouffer's Z-method. Each bucket's
    two-sided p-value is converted to a signed z-score (sign = direction of
    mean_diff: negative mean_diff => favours the first term of the diff, e.g.
    TSDiff-W beating RandomWalk on CRPS); z's are summed and normalised by
    sqrt(k). ASSUMES buckets are independent — pre-merge correlated buckets
    with merge_correlated_diffs() before building `bucket_tests`."""
    labels = list(bucket_tests)
    k = len(labels)
    if k == 0:
        raise ValueError("stouffer_combine: no buckets given.")
    z_per_bucket = {}
    for label in labels:
        p = min(max(bucket_tests[label]["p_value"], 1e-12), 1 - 1e-12)
        sign = -1.0 if bucket_tests[label]["mean_diff"] < 0 else 1.0
        z_per_bucket[label] = sign * float(stats.norm.ppf(1 - p / 2))

    z_combined = float(np.sum(list(z_per_bucket.values())) / np.sqrt(k))
    p_combined = float(2 * (1 - stats.norm.cdf(abs(z_combined))))
    return {
        "buckets": labels, "k": k, "z_per_bucket": z_per_bucket,
        "z_combined": z_combined, "p_combined": p_combined,
        "favors_first_term": bool(z_combined < 0),
        "significant_at_05": bool(p_combined < 0.05),
    }
