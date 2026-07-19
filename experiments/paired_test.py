"""
paired_test.py — paired bootstrap significance test on a per-origin metric
difference (BRIEF_weekly_prediction_v2.md §6: "on ne conclut a une difference
que si elle est significative").

Resamples ORIGINS (the natural walk-forward unit) with replacement, so the
model-vs-model pairing at each origin is preserved in every resample — this is
what makes it a *paired* test rather than comparing two independent samples.

Caveat (documented, not fixed — brief §8d asks to read p-values with this in
mind): with weekly origins and W1-W3 targets, target windows overlap across
adjacent origins (origin k's W2 target ~= origin k+1's W1 target), so the
per-origin differences are autocorrelated, not iid. This bootstrap respects the
origin-level pairing but does NOT correct for that serial correlation — with
n_origins=30 walk-forward weekly origins, effective statistical power is lower
than the nominal n=30 suggests. Treat p-values as optimistic; only read strong,
robust effects as real.
"""

import numpy as np


def paired_block_bootstrap_test(diffs, block_length: int = 3, n_boot: int = 10000,
                                seed: int = 0) -> dict:
    """Moving-block bootstrap version of paired_bootstrap_test — MANDATORY
    (BRIEF_soir_D7_tests_apparies.md §2) whenever `diffs` are ordered by walk-forward
    origin and consecutive origins share overlapping target windows (our 30 weekly
    origins with W1-W3 targets: origin k's W2/W3 targets overlap origin k+1/k+2's
    W1/W2), which makes the per-origin differences autocorrelated, not iid — plain
    `paired_bootstrap_test` (element-wise resampling) would break that correlation
    and understate the true variance, giving over-optimistic p-values.

    `diffs` MUST be passed in chronological origin order (not shuffled) -- blocks
    are contiguous slices of this order. Resamples ceil(n/block_length) BLOCKS of
    `block_length` consecutive elements (with replacement, blocks may overlap each
    other in the original series), concatenates and truncates to length n, takes
    the mean -- one bootstrap replicate. block_length=3 by default: matches the
    W1-W3 overlap depth (a difference 3 origins apart no longer shares a target
    week), so blocks of 3 consecutive origins capture the correlation structure.

    Returns the same fields as paired_bootstrap_test, PLUS `effective_n` = n //
    block_length (the number of roughly-independent blocks -- report this
    alongside n so a reader doesn't mistake n=30 for the true statistical power,
    per brief: "~10-15", not 30)."""
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        raise ValueError("paired_block_bootstrap_test: empty diffs.")
    if block_length < 1 or block_length > n:
        raise ValueError(f"block_length={block_length} must be in [1, n={n}].")

    n_blocks_available = n - block_length + 1
    n_blocks_needed = -(-n // block_length)   # ceil(n / block_length)

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_blocks_available, size=(n_boot, n_blocks_needed))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        pieces = [diffs[s:s + block_length] for s in starts[b]]
        resampled = np.concatenate(pieces)[:n]
        boot_means[b] = resampled.mean()

    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = (float(q) for q in np.quantile(boot_means, [0.025, 0.975]))
    if mean_diff >= 0:
        p = 2.0 * float(np.mean(boot_means <= 0))
    else:
        p = 2.0 * float(np.mean(boot_means >= 0))
    p = min(p, 1.0)

    return {
        "n": int(n), "block_length": int(block_length),
        "effective_n": int(n // block_length),
        "mean_diff": mean_diff, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p_value": p, "significant_at_05": bool(p < 0.05),
    }


def paired_bootstrap_test(diffs, n_boot: int = 10000, seed: int = 0) -> dict:
    """diffs: per-origin (metric_model_a - metric_model_b), one value per origin.

    Returns mean_diff, a 95% percentile bootstrap CI of the mean, and a
    two-sided p-value (fraction of bootstrap means on the opposite side of zero
    from the observed mean, doubled, capped at 1.0 — the standard percentile-
    bootstrap significance test for "does this paired difference exclude zero").
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        raise ValueError("paired_bootstrap_test: empty diffs.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = (float(q) for q in np.quantile(boot_means, [0.025, 0.975]))
    if mean_diff >= 0:
        p = 2.0 * float(np.mean(boot_means <= 0))
    else:
        p = 2.0 * float(np.mean(boot_means >= 0))
    p = min(p, 1.0)

    return {
        "n": int(n), "mean_diff": mean_diff, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p_value": p, "significant_at_05": bool(p < 0.05),
    }
