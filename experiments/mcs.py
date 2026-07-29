"""
mcs.py — Model Confidence Set (Hansen, Lunde & Nason, 2011), plus an optional
SPA test (Hansen 2005) vs a designated benchmark.

BRIEF_unification_protocole_duel.md §2.5: "MCS neuf (absent du dépôt)" — the
duel's tests (dm_hac_test, block bootstrap, Clark-West, Holm — all reused via
duel_pairwise_tests.py) only ever answer pairwise questions ("is A different
from B"). Neither says which SUBSET of the 6 models is statistically
indistinguishable from the best one at a given (asset, horizon) — that is
what "jeu égal" actually means, and it is what pooled_analysis.py's own
refusal to rank models points at without providing. `grep -rn "MCS\\|SPA"`
over the repo before this file confirms neither exists anywhere — this is new
code, tested in test_mcs.py.

Algorithm (HLN 2011 §3, the elimination/T_max variant): given a loss matrix
`loss[t, i]` (lower = better, e.g. per-origin CRPS of model i), start from the
full model set M. At each step, for every model i in M compute its relative
loss against the CURRENT group's average, d_i,t = loss[t, i] - mean_j(loss[t,
j] for j in M), studentized by a bootstrap estimate of its variance:
t_i = mean_t(d_i,t) / sqrt(var_boot(mean_t(d_i,t))). The equivalence test
statistic T_max = max_i t_i is compared to its own bootstrap null distribution
(the same statistic recomputed on resampled data, recentered at the observed
dbar so the null "all models in M are equal" holds by construction in the
resamples); if the bootstrap p-value < alpha, H0,M is rejected and the model
with the largest t_i (worst relative to the group) is eliminated; otherwise
the current M is the MCS at level alpha. Elimination stops when a set is not
rejected, or a single model remains (trivially its own MCS).

Bootstrap: a moving-block bootstrap over the TIME axis (paired_test.py's own
convention — contiguous blocks, concatenated and truncated to length T,
block_length=3 by default, matching the W1-W3 target-window overlap depth used
everywhere else in this duel), with the SAME resampled time indices applied to
every model's column at once (preserves the cross-model correlation at each
origin, not just each model's own marginal variance) — drawn ONCE for the
whole elimination sequence, reused at every step regardless of |M|, which is
both cheaper and the standard MCS bootstrap convention (resample time, not
models).
"""

import numpy as np
import pandas as pd


def _block_bootstrap_time_indices(T: int, block_length: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, T) matrix of resampled time indices — contiguous blocks of
    `block_length`, drawn with replacement, concatenated and truncated to T,
    exactly paired_test.paired_block_bootstrap_test's scheme (kept identical
    for consistency across every bootstrap in this duel)."""
    if block_length < 1 or block_length > T:
        raise ValueError(f"block_length={block_length} must be in [1, T={T}].")
    n_blocks_available = T - block_length + 1
    n_blocks_needed = -(-T // block_length)   # ceil(T / block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_blocks_available, size=(n_boot, n_blocks_needed))
    idx = np.empty((n_boot, n_blocks_needed * block_length), dtype=int)
    for b in range(n_boot):
        idx[b] = np.concatenate([np.arange(s, s + block_length) for s in starts[b]])
    return idx[:, :T]


def model_confidence_set(loss, alpha: float = 0.05, block_length: int = 3,
                         n_boot: int = 5000, seed: int = 0) -> dict:
    """`loss`: DataFrame or 2D array [T, n_models], lower = better (e.g. CRPS
    per origin). Column names (or 0..n-1 if a bare array) identify models.

    Returns {"mcs": [surviving model names, in original column order],
    "alpha", "block_length", "n_boot", "steps": [{eliminated, t_stat,
    p_value, remaining_before}, ...]} — `steps` is the audit trail: one entry
    per elimination, in the order models were kicked out, so a reader can see
    exactly why any model is or isn't in the final MCS."""
    if isinstance(loss, pd.DataFrame):
        names = list(loss.columns)
        L = loss.values.astype(float)
    else:
        L = np.asarray(loss, dtype=float)
        names = list(range(L.shape[1]))
    T, n = L.shape
    if n < 1:
        raise ValueError("model_confidence_set: need at least one model.")
    if np.isnan(L).any():
        raise ValueError("model_confidence_set: loss matrix contains NaNs.")

    boot_time_idx = _block_bootstrap_time_indices(T, block_length, n_boot, seed)
    remaining = list(range(n))
    steps = []

    while len(remaining) > 1:
        Lsub = L[:, remaining]                                  # (T, n_sub)
        row_avg = Lsub.mean(axis=1)                             # (T,)
        d = Lsub - row_avg[:, None]                             # (T, n_sub)
        dbar = d.mean(axis=0)                                   # (n_sub,)

        Lsub_boot = Lsub[boot_time_idx]                         # (n_boot, T, n_sub)
        row_avg_boot = Lsub_boot.mean(axis=2, keepdims=True)    # (n_boot, T, 1)
        dbar_boot = (Lsub_boot - row_avg_boot).mean(axis=1)     # (n_boot, n_sub)

        var_i = np.maximum(dbar_boot.var(axis=0, ddof=1), 1e-12)
        t_i = dbar / np.sqrt(var_i)
        t_obs = float(np.max(t_i))

        t_i_boot = (dbar_boot - dbar[None, :]) / np.sqrt(var_i)[None, :]
        t_max_boot = t_i_boot.max(axis=1)
        p_value = float(np.mean(t_max_boot >= t_obs))

        if p_value >= alpha:
            break

        worst_local = int(np.argmax(t_i))
        worst_idx = remaining[worst_local]
        steps.append({
            "eliminated": names[worst_idx], "t_stat": float(t_i[worst_local]),
            "p_value": p_value, "remaining_before": [names[i] for i in remaining],
        })
        remaining.remove(worst_idx)

    return {
        "mcs": [names[i] for i in remaining], "alpha": alpha,
        "block_length": block_length, "n_boot": n_boot, "steps": steps,
    }


def spa_test(loss_benchmark, loss_models: dict, block_length: int = 3,
            n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Hansen (2005) Superior Predictive Ability test, "consistent" p-value
    variant. H0: no model in `loss_models` beats `loss_benchmark` (in
    expectation) — the recommended benchmark for this duel is GARCH(1,1)
    (brief §2.5), so d_k,t = loss_benchmark,t - loss_models[k],t; dbar_k > 0
    means model k has LOWER loss (beats the benchmark) on average.

    `loss_benchmark`: (T,) array. `loss_models`: {name: (T,) array}, each
    paired 1:1 with `loss_benchmark` (same T, same time order — same origins).

    Recentering rule (Hansen 2005 eq. 16-19, "consistent" p-value, the
    variant used by default in Hansen's own SPA implementation): under the
    bootstrap null, models whose observed dbar_k is not detectably positive
    (below a sqrt(2 log log T) threshold) are recentered to exactly zero
    rather than to their observed (possibly negative) dbar_k — this avoids
    the "upper" bound's over-conservatism without the "lower" bound's
    anti-conservatism; see module docstring for why the SAME block-bootstrap
    time-index scheme as model_confidence_set is reused here."""
    names = list(loss_models)
    if not names:
        raise ValueError("spa_test: need at least one model.")
    Lb = np.asarray(loss_benchmark, dtype=float)
    T = Lb.size
    cols = []
    for k in names:
        Lk = np.asarray(loss_models[k], dtype=float)
        if Lk.shape != Lb.shape:
            raise ValueError(f"spa_test: shape mismatch for model {k!r}: "
                             f"{Lk.shape} vs benchmark {Lb.shape}")
        cols.append(Lb - Lk)
    D = np.stack(cols, axis=1)                                  # (T, n)
    dbar = D.mean(axis=0)

    boot_time_idx = _block_bootstrap_time_indices(T, block_length, n_boot, seed)
    D_boot = D[boot_time_idx]                                   # (n_boot, T, n)
    dbar_boot = D_boot.mean(axis=1)                             # (n_boot, n)
    var_k = np.maximum(dbar_boot.var(axis=0, ddof=1), 1e-12)

    loglogT = np.log(np.log(max(T, 3)))
    threshold = np.sqrt(var_k / T * 2.0 * loglogT)
    # g_c(dbar_k): recenter fully to dbar_k (i.e. bootstrap term becomes
    # dbar_boot - dbar_k, a mean-zero resampling fluctuation) UNLESS dbar_k is
    # clearly negative (below -threshold), in which case g_c = 0 and the
    # bootstrap term is left as the raw dbar_boot (still centered on the
    # model's own, genuinely negative, sample mean) -- this is what keeps
    # clearly-inferior models from inflating the null max distribution while
    # still imposing H0 (mean exactly 0) on every model not clearly rejected.
    g_c = np.where(dbar >= -threshold, dbar, 0.0)

    z_obs = np.sqrt(T) * dbar / np.sqrt(var_k)
    t_spa_obs = max(0.0, float(np.max(z_obs)))

    z_boot = np.sqrt(T) * (dbar_boot - g_c[None, :]) / np.sqrt(var_k)[None, :]
    t_spa_boot = np.maximum(0.0, z_boot.max(axis=1))
    p_value = float(np.mean(t_spa_boot >= t_spa_obs))

    return {
        "t_spa": t_spa_obs, "p_value": p_value, "n": int(T),
        "block_length": block_length, "n_boot": n_boot,
        "reject_no_model_beats_benchmark": bool(p_value < alpha),
        "per_model_mean_gain_vs_benchmark": dict(zip(names, dbar.tolist())),
    }
