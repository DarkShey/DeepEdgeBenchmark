"""
build_monthly_pairs.py -- monthly mirror of `matrice_paired_tests.
build_daily_weekly_pairs` / `comparison_3_daily_vs_weekly` (BRIEF_nsdiff_
mensuel_M1M2M3.md §4: "réutiliser comparison_3_daily_vs_weekly -- vérifier
qu'il est agnostique à l'unité d'horizon, sinon adapter a minima").

It is NOT agnostic: both functions hard-code `horizon_type == "weekly"` and
`frequence in ("daily", "weekly")` (matrice_paired_tests.py lines ~176/207).
Feeding it monthly rows (horizon_type='monthly', frequence in
'daily'/'monthly') would silently return empty results, not an error -- the
"adapt a minima" branch of the brief applies. Per this repo's own precedent
(`oos_nsdiff_daily_weekly.load_baseline_triplets_daily` mirrors
`load_baseline_triplets` rather than editing the shared, weekly-hardcoded
original) and the non-negotiable "aucun fichier source modifié hors ajouts",
this is a NEW module, not an edit to `matrice_paired_tests.py`.

Everything genuinely frequency-agnostic underneath IS reused, not
reimplemented: `paired_test.paired_block_bootstrap_test` (the bootstrap
itself) and `matrice_paired_tests.BLOCK_LENGTH`/`MIN_PAIRED_POINTS` (the
same block length / minimum-points convention as every other paired test in
this repo). Only the pairing/grouping glue -- the part that is inherently
weekly-shaped -- is mirrored.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from paired_test import paired_block_bootstrap_test          # noqa: E402
from matrice_paired_tests import BLOCK_LENGTH, MIN_PAIRED_POINTS  # noqa: E402


def build_daily_monthly_pairs(df: pd.DataFrame, horizon_units=None) -> pd.DataFrame:
    """Regime B (frequence=daily, daily-trained model evaluated at its native
    monthly target) vs regime C (frequence=monthly, monthly-native model),
    paired by (model, asset, horizon_unit, target_date) -- both sides share
    the exact same target_date/cutoff_date by construction (brief §1: "les
    deux visent exactement la même cible"), same as the weekly mirror."""
    monthly = df[df["horizon_type"] == "monthly"]
    if horizon_units is not None:
        monthly = monthly[monthly["horizon_unit"].isin(horizon_units)]
    daily_side = monthly[monthly["frequence"] == "daily"]
    monthly_side = monthly[monthly["frequence"] == "monthly"]

    frames = []
    for (model, asset, h), g_daily in daily_side.groupby(["model", "asset", "horizon_unit"]):
        g_monthly = monthly_side[(monthly_side["model"] == model) & (monthly_side["asset"] == asset)
                                 & (monthly_side["horizon_unit"] == h)]
        merged = g_daily.merge(g_monthly, on="target_date", suffixes=("_daily", "_monthly"))
        merged.insert(0, "model", model)
        merged.insert(1, "asset", asset)
        merged.insert(2, "horizon_unit", h)
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["model", "asset", "horizon_unit", "target_date"])
    return pd.concat(frames, ignore_index=True)


def comparison_3_daily_vs_monthly(df: pd.DataFrame) -> list:
    """Per (model, asset, horizon_unit in M+1/2/3): regime B (frequence=daily)
    vs regime C (frequence=monthly), paired by target_date -- same protocol
    inside a model, the cleanest comparison (no cross-model asymmetry)."""
    results = []
    pairs = build_daily_monthly_pairs(df)

    monthly = df[df["horizon_type"] == "monthly"]
    daily_keys = set(map(tuple, monthly[monthly["frequence"] == "daily"][["model", "asset", "horizon_unit"]]
                        .drop_duplicates().values))
    monthly_keys = set(map(tuple, monthly[monthly["frequence"] == "monthly"][["model", "asset", "horizon_unit"]]
                          .drop_duplicates().values))

    for model, asset, h in sorted(daily_keys & monthly_keys):
        if pairs.empty:
            merged = pairs
        else:
            merged = pairs[(pairs["model"] == model) & (pairs["asset"] == asset)
                          & (pairs["horizon_unit"] == h)].sort_values("target_date")
        if len(merged) < MIN_PAIRED_POINTS:
            results.append({"model": model, "asset": asset, "horizon_unit": h,
                            "status": "insufficient_data", "n": int(len(merged))})
            continue
        diffs = (merged["sq_error_daily"] - merged["sq_error_monthly"]).values   # >0 => monthly-native has lower sq error
        test = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
        if test["significant_at_05"]:
            verdict = "monthly_native_significantly_better" if test["mean_diff"] > 0 \
                else "daily_multistep_significantly_better"
        else:
            verdict = "indistinguishable"
        results.append({
            "model": model, "asset": asset, "horizon_unit": h, "status": "tested",
            "verdict": verdict, **test,
        })
    return results
