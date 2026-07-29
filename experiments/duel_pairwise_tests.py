"""
duel_pairwise_tests.py — pairwise significance tests for the diffusion-vs-
classiques duel (BRIEF_unification_protocole_duel.md §2.4, resolves ecart E3
de l'audit: les tests appliqués à la bonne comparaison).

Reuses, never reimplements:
  - honest_eval.metrics.dm_hac_test        (DM, HAC>=h-1, HLN correction)
  - paired_test.paired_block_bootstrap_test (via pooled_analysis.dual_test)
  - pooled_analysis.dual_test               (DM-HAC + bootstrap + concordance)
  - pooled_analysis.holm_correction
  - pooled_analysis.compute_asset_scales / class_series / ASSET_CLASS /
    CORRELATED_PAIRS / STANDALONE_CLASSES   (MASE scale + asset-class fusion)

New in this module (not present anywhere else in the repo): clark_west_test
(a nested-model test against the naive/random-walk benchmark), and the
grid-level orchestration — per-cell dual_test, per-pair Holm correction
across the 5-asset x 3-horizon grid, and the MASE-normalised, asset-class-
fused pooled verdict per pair.

Asset-label convention: functions that touch pooled_analysis's asset-class
machinery (`pooled_pair_verdict`) expect the FULL yfinance ticker in the
"asset" column ("SPY", "ZN=F", "TLT", "BTC-USD", "ETH-USD" — the keys of
pooled_analysis.ASSET_CLASS), not the short codes used elsewhere in
weekly_headtohead.ASSETS ("SPY", "ZN", "TLT", "BTC", "ETH") — callers building
the records DataFrame must use the ticker, not the short code, in that column.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from honest_eval.metrics import dm_hac_test                     # noqa: E402
from pooled_analysis import (                                    # noqa: E402
    dual_test, holm_correction, class_series,
    ASSET_CLASS, CORRELATED_PAIRS, STANDALONE_CLASSES, compute_asset_scales,
)


# ══════════════════════════════════════════════════════════════════════════
#  Clark-West vs the naive/random-walk benchmark
# ══════════════════════════════════════════════════════════════════════════

def clark_west_test(actual, pred_restricted, pred_candidate, h: int = 1, alpha: float = 0.05) -> dict:
    """Clark & West (2007) test of H0: the candidate model (which nests the
    restricted/naive model — naive is the special case "no incremental
    signal") has no incremental predictive content over it, tested via

        f_t = (y_t - yhat_restricted_t)^2
              - [ (y_t - yhat_candidate_t)^2 - (yhat_restricted_t - yhat_candidate_t)^2 ]

    E[f_t] > 0 (one-sided) is evidence the candidate beats the naive
    benchmark. The bracketed adjustment removes the extra noise the
    candidate's own parameter estimation injects into its MSPE under the
    null (candidate collapses to naive), which would otherwise bias a plain
    MSPE differential against the candidate even with zero true incremental
    skill (Clark & West 2007) — this is precisely why CW, not a plain DM
    test, is the right tool against a NESTED benchmark like the random walk.

    f_t is tested via dm_hac_test's own HAC(>=h-1)/HLN machinery (reused, not
    a plain t-test) for consistency with every other test in this duel being
    robust to the origins' overlapping target windows; dm_hac_test returns a
    TWO-sided p-value, converted here to Clark & West's one-sided p-value
    (mean_f > 0 is the only direction of interest — a candidate with
    significantly NEGATIVE mean_f is not "significantly worse" in the CW
    sense, just non-significant in the direction that matters).

    Applies to POINT forecasts (squared-error loss) — CW's adjustment term is
    specific to MSPE and does not generalise to CRPS. This is a SECONDARY,
    point-forecast check against the naive benchmark; the primary CRPS-based
    duel verdict is carried by pairwise_crps_tests()/dual_test() below."""
    y = np.asarray(actual, dtype=float)
    y0 = np.asarray(pred_restricted, dtype=float)
    y1 = np.asarray(pred_candidate, dtype=float)
    if not (y.shape == y0.shape == y1.shape):
        raise ValueError(f"shape mismatch: actual={y.shape}, restricted={y0.shape}, candidate={y1.shape}")
    f = (y - y0) ** 2 - ((y - y1) ** 2 - (y0 - y1) ** 2)
    out = dm_hac_test(f, h=h)
    dm = out["dm_stat"]
    p_one_sided = (out["p_value"] / 2.0) if dm > 0 else (1.0 - out["p_value"] / 2.0)
    return {
        "cw_stat": dm, "p_value": float(min(max(p_one_sided, 0.0), 1.0)),
        "lag": out["lag"], "mean_f": out["mean_diff"], "n": out["n"],
        "candidate_beats_naive": bool(dm > 0 and p_one_sided < alpha),
    }


# ══════════════════════════════════════════════════════════════════════════
#  Per-cell CRPS duel tests, grid-wide Holm correction
# ══════════════════════════════════════════════════════════════════════════

def pairwise_crps_tests(df: pd.DataFrame, pairs: list, h_by_horizon: dict) -> dict:
    """`df` columns: asset, horizon, model, origin (an integer, chronological
    walk-forward order — the natural pairing unit, cf. paired_test.py), crps.
    For each (asset, horizon) cell and each (A, B) in `pairs`: diff = crps_A -
    crps_B, paired on `origin`, scored by pooled_analysis.dual_test (DM-HAC +
    block bootstrap + concordance flag, reused as-is). Keyed
    "asset|horizon|A vs B"; `h_by_horizon` maps horizon label -> forecast
    horizon (in the unit the origins advance by, e.g. {"W1": 1, "W2": 2,
    "W3": 3}) for dm_hac_test's HAC truncation."""
    out = {}
    for (asset, horizon), g in df.groupby(["asset", "horizon"]):
        piv = g.pivot(index="origin", columns="model", values="crps").sort_index()
        h = h_by_horizon[horizon]
        for a, b in pairs:
            if a not in piv.columns or b not in piv.columns:
                continue
            diffs = (piv[a] - piv[b]).dropna().values
            out[f"{asset}|{horizon}|{a} vs {b}"] = dual_test(diffs, h=h)
    return out


def holm_correct_grid(cell_results: dict, assets: list, horizons: list, pair_label: str) -> dict:
    """Holm-corrects the block-bootstrap p-values of ONE pair-type (e.g.
    "TSDiff-W vs Naive") across the FULL 5-asset x 3-horizon grid (brief
    §2.4: "correction de multiplicité sur la grille complète 5 actifs x 3
    horizons x paires") — reuses pooled_analysis.holm_correction, does not
    reimplement Holm. Returns {cell_key: p_value_holm} for every grid cell
    that was actually tested for this pair (missing/insufficient-data cells
    are skipped, not silently counted as p=1 padding that would understate
    the correction)."""
    keys = [f"{a}|{hz}|{pair_label}" for a in assets for hz in horizons
            if f"{a}|{hz}|{pair_label}" in cell_results
            and cell_results[f"{a}|{hz}|{pair_label}"].get("status") == "tested"]
    pvals = [cell_results[k]["p_value_bootstrap"] for k in keys]
    adjusted = holm_correction(pvals) if pvals else []
    return dict(zip(keys, adjusted))


# ══════════════════════════════════════════════════════════════════════════
#  MASE-normalised, asset-class-fused pooled verdict (brief §2.4)
# ══════════════════════════════════════════════════════════════════════════

def pooled_pair_verdict(df: pd.DataFrame, pair: tuple, scales: dict, h: int) -> dict:
    """MASE-normalise (divide by scales[asset], reused from
    pooled_analysis.compute_asset_scales) the pair's per-origin CRPS
    differential, fuse the two correlated asset pairs (BTC-USD/ETH-USD,
    ZN=F/TLT) into one series each via pooled_analysis.class_series (reused,
    not reimplemented), concatenate the <=3 resulting independent asset-class
    series and run ONE dual_test — a single pooled verdict for this (pair,
    horizon) across all 5 assets, scale-free and not double-counting the two
    correlated pairs (brief §2.4: "normalisation MASE + fusion en classes
    d'actifs avant pooling").

    `df`'s "asset" column MUST hold the full ticker (module docstring)."""
    a, b = pair
    sub = df[df["model"].isin([a, b])]
    piv = sub.pivot_table(index=["asset", "origin"], columns="model", values="crps").reset_index()
    piv = piv.dropna(subset=[a, b])
    piv["diff_mase"] = (piv[a] - piv[b]) / piv["asset"].map(scales)
    classes = class_series(piv, "diff_mase", date_col="origin")
    diffs = np.concatenate([s.values for s in classes.values()]) if classes else np.array([])
    return dual_test(diffs, h=h)
