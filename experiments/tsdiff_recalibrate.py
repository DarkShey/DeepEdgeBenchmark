"""
tsdiff_recalibrate.py — per-level conformal recalibration of TSDiff's
persisted quantile forecasts (experiments/samples/*_tsdiff.*), to correct its
95% under-coverage (pooled 0.878, cf. experiments/pooled_analysis_prob.json)
without degrading its already-good 50% calibration (pooled 0.525). No
retraining: reuses the N=500 sample clouds already in experiments/samples/.

Method (split conformal, per level, per asset):
  1. Chronological calibration/test split PER ASSET: unique cutoff_dates
     sorted, first half -> calibration block, second half -> test block
     (calibration strictly precedes test -- no lookahead, same convention as
     pooled_analysis.py's compute_asset_scales / scale-end = first test
     origin).
  2. Per row: median_i = median(samples_i); at level L (alpha=1-L),
     lo_i, hi_i = empirical quantiles of samples_i at [alpha/2, 1-alpha/2].
  3. Nonconformity score (robust, no Gaussian assumption), side-aware so an
     asymmetric predictive cloud is not forced symmetric:
         s_i = (y_i - median_i) / (hi_i - median_i)   if y_i >= median_i
         s_i = (median_i - y_i) / (median_i - lo_i)    otherwise
     k_level = the standard split-conformal empirical quantile of
     {s_i : i in calibration} at level L (finite-sample-corrected index
     ceil((n+1)*L)/n, per Romano/Patterson/Candes-style split conformal).
  4. Recalibrated bound on the TEST block: new_bord = median + k_level *
     (bord - median), applied independently to each row's own lo/hi at that
     level (brief's exact formula) -- k is per (asset, level), NEVER pooled
     across assets (a global k would erase the asset-specific miscalibration
     this exercise is meant to fix).
  5. CRPS after recalibration needs a full distribution, not just 3 quantile
     pairs, so each test row's whole 500-sample cloud is warped with a
     piecewise-linear k(L) interpolated through the 3 calibrated anchors
     (L=0 -> k=1 at the median by construction; L=0.5/0.8/0.95 -> the fitted
     k; flat extrapolation past L=0.95, since nothing calibrates beyond it).
     This exactly reproduces the 3 calibrated quantile pairs and lets CRPS
     reuse experiments.crps_metrics.crps_empirical UNCHANGED on the warped
     cloud, instead of reinventing a quantile-based CRPS approximation.
  6. Significance: same guardrail as pooled_analysis.py's calibration
     question -- coverage gap (indicator - target level) on the test block,
     block-bootstrapped (experiments.paired_test.paired_block_bootstrap_test,
     block_length=3, same convention), reported before AND after, per asset
     per level. A tight CI around zero is a genuine result, not a failure.

Guardrails: k computed per (asset, level), never a global factor; calibration
block strictly precedes test block (verified, not assumed); flags any
(asset, level) whose calibration block is small enough that k is likely
noisy (n_calib < MIN_CALIB_FLAG).

Output: experiments/tsdiff_recalibre.json.

Usage:
    python tsdiff_recalibrate.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from crps_metrics import crps_empirical                      # noqa: E402
from paired_test import paired_block_bootstrap_test           # noqa: E402
from build_kpi_probabilistes import ASSETS, ASSET_CLASS        # noqa: E402

SAMPLES_DIR = ROOT / "experiments" / "samples"
OUT_PATH = ROOT / "experiments" / "tsdiff_recalibre.json"

LEVELS = (0.5, 0.8, 0.95)
BLOCK_LENGTH = 3          # same convention as pooled_analysis.py's BLOCK_LENGTH
MIN_CALIB_FLAG = 30       # below this, flag k as potentially noisy
CALIB_FRACTION = 0.5      # chronological split point, by unique cutoff_date


def load_tsdiff(asset: str) -> tuple:
    idx = pd.read_parquet(SAMPLES_DIR / f"{asset}_tsdiff.index.parquet")
    samples = np.load(SAMPLES_DIR / f"{asset}_tsdiff.samples.npz")["samples"]
    return idx.reset_index(drop=True), samples


def chronological_split(idx: pd.DataFrame, fraction: float = CALIB_FRACTION) -> tuple:
    """Row indices for (calibration, test), split by unique cutoff_date so the
    calibration block is STRICTLY before the test block (no shared dates, no
    lookahead) -- same no-leakage convention as the rest of the pipeline."""
    dates = np.sort(idx["cutoff_date"].unique())
    n_calib_dates = max(1, int(len(dates) * fraction))
    calib_dates = set(dates[:n_calib_dates])
    test_dates = set(dates[n_calib_dates:])
    calib_rows = idx.index[idx["cutoff_date"].isin(calib_dates)].to_numpy()
    test_rows = idx.index[idx["cutoff_date"].isin(test_dates)].to_numpy()
    return calib_rows, test_rows, (dates[0], dates[n_calib_dates - 1]), (dates[n_calib_dates], dates[-1])


def row_median_lo_hi(samples: np.ndarray, level: float) -> tuple:
    alpha = 1.0 - level
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=1)
    med = np.median(samples, axis=1)
    return med, lo, hi


def nonconformity_scores(y: np.ndarray, med: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Side-aware, scale-normalised nonconformity score -- robust (empirical
    quantile-based), no Gaussian assumption: distance from the median scaled
    by that side's own half-width, so an asymmetric predictive cloud keeps
    its own shape instead of being forced symmetric."""
    above = y >= med
    s = np.empty_like(y, dtype=float)
    s[above] = (y[above] - med[above]) / (hi[above] - med[above])
    s[~above] = (med[~above] - y[~above]) / (med[~above] - lo[~above])
    return s


def conformal_k(scores_calib: np.ndarray, level: float) -> float:
    """Standard split-conformal empirical quantile of the calibration scores,
    finite-sample corrected (ceil((n+1)*level)/n, capped at the max score) --
    this is what makes the resulting interval calibrated ON THE CALIBRATION
    BLOCK by construction."""
    n = len(scores_calib)
    s_sorted = np.sort(scores_calib)
    rank = int(np.ceil((n + 1) * level))
    rank = min(rank, n)
    return float(s_sorted[rank - 1])


def warp_cloud(samples_test: np.ndarray, med: np.ndarray, k_by_level: dict) -> np.ndarray:
    """Warp each test row's full 500-sample cloud with a piecewise-linear
    k(L) interpolated through the 3 calibrated (level, k) anchors (plus the
    trivial L=0 -> k=1 anchor at the median), so the warped cloud reproduces
    the 3 recalibrated quantile pairs exactly and CRPS can be computed on it
    with the existing crps_empirical estimator, unchanged."""
    n_rows, n_samples = samples_test.shape
    anchors_L = np.array([0.0] + list(LEVELS))
    anchors_k = np.array([1.0] + [k_by_level[level] for level in LEVELS])

    ranks = (np.arange(1, n_samples + 1)) / (n_samples + 1)   # plotting positions in (0,1)
    L_of_rank = np.abs(2.0 * ranks - 1.0)                      # two-sided level implied by each rank
    k_of_rank_within = np.interp(L_of_rank, anchors_L, anchors_k)   # flat extrap past L=0.95 (np.interp default)

    sorted_idx = np.argsort(samples_test, axis=1)
    sorted_samples = np.take_along_axis(samples_test, sorted_idx, axis=1)
    warped_sorted = med[:, None] + k_of_rank_within[None, :] * (sorted_samples - med[:, None])

    warped = np.empty_like(samples_test)
    np.put_along_axis(warped, sorted_idx, warped_sorted, axis=1)
    return warped


def coverage_gap_test(indicator: np.ndarray, level: float, dates: np.ndarray) -> dict:
    """Block-bootstrap CI on (indicator - level), same convention as
    pooled_analysis.py's question_calibration -- chronological order
    required, block_length=3."""
    order = np.argsort(dates, kind="stable")
    diffs = indicator[order].astype(float) - level
    if len(diffs) < 2:
        return {"status": "insufficient_data", "n": int(len(diffs))}
    bl = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
    return {
        "status": "tested", "n": bl["n"], "effective_n": bl["effective_n"],
        "coverage_gap_mean": bl["mean_diff"], "ci95_lo": bl["ci95_lo"], "ci95_hi": bl["ci95_hi"],
        "p_value_bootstrap": bl["p_value"], "significant_bootstrap": bl["significant_at_05"],
    }


def process_asset(asset: str) -> dict:
    idx, samples = load_tsdiff(asset)
    y = idx["y_true"].to_numpy(dtype=float)

    calib_rows, test_rows, calib_range, test_range = chronological_split(idx)
    n_calib, n_test = len(calib_rows), len(test_rows)

    result = {
        "n_calib_rows": int(n_calib), "n_test_rows": int(n_test),
        "calib_date_range": [str(calib_range[0]), str(calib_range[1])],
        "test_date_range": [str(test_range[0]), str(test_range[1])],
        "levels": {},
    }

    k_by_level = {}
    for level in LEVELS:
        med_all, lo_all, hi_all = row_median_lo_hi(samples, level)

        med_c, lo_c, hi_c = med_all[calib_rows], lo_all[calib_rows], hi_all[calib_rows]
        y_c = y[calib_rows]
        scores_c = nonconformity_scores(y_c, med_c, lo_c, hi_c)
        k = conformal_k(scores_c, level)
        k_by_level[level] = k

        med_t, lo_t, hi_t = med_all[test_rows], lo_all[test_rows], hi_all[test_rows]
        y_t = y[test_rows]
        new_lo_t = med_t + k * (lo_t - med_t)
        new_hi_t = med_t + k * (hi_t - med_t)

        cov_before = (y_t >= lo_t) & (y_t <= hi_t)
        cov_after = (y_t >= new_lo_t) & (y_t <= new_hi_t)
        test_dates = idx.loc[test_rows, "cutoff_date"].to_numpy()

        result["levels"][str(level)] = {
            "k": k,
            "n_calib_scores": int(n_calib),
            "calib_size_flag": "ok" if n_calib >= MIN_CALIB_FLAG else
                              f"n_calib={n_calib} < {MIN_CALIB_FLAG} -- k potentially noisy",
            "coverage_test_before": float(cov_before.mean()),
            "coverage_test_after": float(cov_after.mean()),
            "coverage_gap_before": coverage_gap_test(cov_before, level, test_dates),
            "coverage_gap_after": coverage_gap_test(cov_after, level, test_dates),
            "mean_width_before": float(np.mean(hi_t - lo_t)),
            "mean_width_after": float(np.mean(new_hi_t - new_lo_t)),
        }

    # CRPS before/after on the test block, full cloud warp (all 3 levels at once)
    samples_test = samples[test_rows]
    med_test = np.median(samples_test, axis=1)
    y_t = y[test_rows]
    crps_before = np.array([crps_empirical(samples_test[i], y_t[i]) for i in range(len(test_rows))])
    warped_test = warp_cloud(samples_test, med_test, k_by_level)
    crps_after = np.array([crps_empirical(warped_test[i], y_t[i]) for i in range(len(test_rows))])

    result["crps_test_before_mean"] = float(crps_before.mean())
    result["crps_test_after_mean"] = float(crps_after.mean())
    result["crps_degradation_pct"] = float(
        100.0 * (crps_after.mean() - crps_before.mean()) / crps_before.mean())

    return result


def build_pooled_summary(per_asset: dict) -> dict:
    """Row-weighted pooled summary across the 5 assets' test blocks, for
    direct comparison against the pooled 0.878/0.525 baseline in
    pooled_analysis_prob.json (that number pools ALL rows, not per-asset)."""
    summary = {}
    for level in LEVELS:
        lvl_key = str(level)
        n_tot = sum(a["n_test_rows"] for a in per_asset.values())
        cov_before = sum(a["levels"][lvl_key]["coverage_test_before"] * a["n_test_rows"]
                         for a in per_asset.values()) / n_tot
        cov_after = sum(a["levels"][lvl_key]["coverage_test_after"] * a["n_test_rows"]
                        for a in per_asset.values()) / n_tot
        summary[f"coverage_{int(level*100)}_before"] = cov_before
        summary[f"coverage_{int(level*100)}_after"] = cov_after

    n_tot = sum(a["n_test_rows"] for a in per_asset.values())
    summary["crps_before_mean"] = sum(a["crps_test_before_mean"] * a["n_test_rows"]
                                      for a in per_asset.values()) / n_tot
    summary["crps_after_mean"] = sum(a["crps_test_after_mean"] * a["n_test_rows"]
                                     for a in per_asset.values()) / n_tot
    summary["crps_degradation_pct"] = 100.0 * (
        summary["crps_after_mean"] - summary["crps_before_mean"]) / summary["crps_before_mean"]
    return summary


def print_table(per_asset: dict, pooled: dict) -> None:
    print(f"\n{'Asset':<10}{'Level':>7}{'k':>8}{'Cov before':>12}{'Cov after':>11}"
          f"{'Sig. after':>12}")
    for asset, a in per_asset.items():
        for level in LEVELS:
            lvl = a["levels"][str(level)]
            sig = "yes*" if lvl["coverage_gap_after"].get("significant_bootstrap") else "no"
            print(f"{asset:<10}{level:>7.2f}{lvl['k']:>8.3f}"
                  f"{lvl['coverage_test_before']:>12.3f}{lvl['coverage_test_after']:>11.3f}"
                  f"{sig:>12}")
        pct_str = f"({a['crps_degradation_pct']:+.1f}%)"
        print(f"{'':<10}{'CRPS':>7}{'':>8}{a['crps_test_before_mean']:>12.4f}"
              f"{a['crps_test_after_mean']:>11.4f}{pct_str:>12}")
    print(f"\n{'POOLED':<10}"
          + "".join(f"  cov{int(l*100)}: {pooled[f'coverage_{int(l*100)}_before']:.3f} -> "
                    f"{pooled[f'coverage_{int(l*100)}_after']:.3f}" for l in LEVELS))
    print(f"{'':<10}  CRPS: {pooled['crps_before_mean']:.4f} -> {pooled['crps_after_mean']:.4f} "
          f"({pooled['crps_degradation_pct']:+.1f}%)")
    print("* significant = coverage gap's 95% block-bootstrap CI excludes 0 (block_length=3).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=ASSETS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    per_asset = {}
    for asset in args.assets:
        print(f"[{asset}] loading TSDiff samples, splitting, calibrating ...")
        per_asset[asset] = process_asset(asset)

    pooled = build_pooled_summary(per_asset)
    print_table(per_asset, pooled)

    payload = {
        "config": {
            "method": "split conformal, per (asset, level), side-aware nonconformity score "
                      "|y-median|/half-width, finite-sample-corrected empirical quantile for k",
            "levels": list(LEVELS),
            "calib_fraction_by_unique_date": CALIB_FRACTION,
            "block_length": BLOCK_LENGTH,
            "min_calib_flag_threshold": MIN_CALIB_FLAG,
            "asset_classes": ASSET_CLASS,
            "note_pooling": "k fit and applied PER ASSET (never a global factor); all TSDiff "
                            "regimes/horizons (D+1, D+7, W+1-3, regimes A/B/C) pooled together "
                            "within an asset for the calibration score set, per the brief's "
                            "level x asset grouping.",
            "note_crps": "CRPS after is computed on a full 500-sample cloud per row, warped by "
                        "a piecewise-linear k(level) interpolated through the 3 calibrated "
                        "anchors (flat extrapolation past 95%) -- reuses crps_empirical "
                        "unchanged rather than approximating CRPS from 3 quantile pairs.",
            "baseline_reference": "pooled_analysis_prob.json TSDiff cov50=0.525, cov95=0.878 "
                                  "(pooled across ALL rows, not per-asset, not per-test-block-"
                                  "only -- see this file's pooled_summary for the apples-to-"
                                  "apples before/after on the SAME test block).",
        },
        "per_asset": per_asset,
        "pooled_summary": pooled,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
