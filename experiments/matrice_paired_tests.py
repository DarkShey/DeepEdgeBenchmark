"""
matrice_paired_tests.py — paired, block-bootstrapped significance tests over the
full 240-cell matrix (BRIEF_soir_D7_tests_apparies.md §2), replacing the point-
estimate rankings with tested verdicts.

Block bootstrap is MANDATORY here (not the plain i.i.d. paired_bootstrap_test):
origins overlap (weekly W1-W3 targets from origins one week apart; D+1's daily
returns have their own short-range autocorrelation), so per-origin differences
are not independent draws -- paired_test.paired_block_bootstrap_test (block_length=3
throughout, matching the W1-W3 overlap depth) is used everywhere, and its
`effective_n` (n // block_length) is reported alongside the nominal n so nobody
mistakes n=30 for 30 independent data points.

Four comparisons, all INTRA-ASSET (RMSE/squared-error never compared across
assets of different price scale):

  1. Ranking per (asset, horizon_unit): each available (model, frequence)
     "competitor" vs the empirical RMSE leader, on squared-error differences.
     Verdict: "leader", "indistinguishable from leader", or "significantly worse".
  2. Calibration per (asset, horizon_unit, model, frequence): is empirical Cov95
     significantly different from the 0.95 target (in_interval - 0.95, tested
     against zero)?
  3. Daily vs weekly per model (regime B vs C), per (model, asset, horizon_unit
     in W+1/W+2/W+3): same protocol inside a model -> the cleanest comparison.
  4. D+7 vs W+1 per (model, asset), aligned on Friday origins (D+7 whose
     cutoff_date is a Friday, paired with the W+1/regime-C row sharing that same
     origin date) -- D+7 has only ~10 sparse rolling origins, so this comparison
     is often underpowered; reported as "insufficient_data" below a minimum count
     rather than tested on a handful of points.

CAVEAT carried into every comparison-1 result: inter-model rankings still carry
the training-protocol asymmetry (TSDiff trained once at T0, the other 5 models
refit at every origin with an expanding window) -- read with caution until the
protocol is unified (BRIEF_comparaison_rigoureuse.md §3, not done here).
Comparisons 3 and 4 are NOT affected (protocol constant within a single model).

Usage:
    python matrice_paired_tests.py
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from paired_test import paired_block_bootstrap_test

ROOT = Path(__file__).resolve().parent.parent
BLOCK_LENGTH = 3
MIN_PAIRED_POINTS = 5     # below this, report insufficient_data instead of testing
ASSET_CLASS = {"BTC-USD": "crypto", "ETH-USD": "crypto", "SPY": "index",
              "ZN=F": "bond", "TLT": "bond"}
HORIZON_UNITS = ("D+1", "D+7", "W+1", "W+2", "W+3")


def load_predictions(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT model, asset, frequence, horizon_type, horizon_unit,
               cutoff_date, target_date, last_close, y_pred, y_lower, y_upper, y_true
        FROM predictions
        WHERE source = 'oos'
    """, con)
    con.close()
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return df


def _competitor_label(model: str, frequence: str, horizon_unit: str) -> str:
    # regime A (D+1/D+7) has one variant per model -> no need to disambiguate by frequence
    if horizon_unit in ("D+1", "D+7"):
        return model
    return f"{model} ({frequence})"


def verify_d7_definition(df: pd.DataFrame) -> dict:
    """§5 de la vérification : aucune ligne D+7 avec cutoff != target - 7j calendaires,
    A L'EXCEPTION DOCUMENTEE des actifs crypto (BTC-USD/ETH-USD tradent 7j/7 -> 5 jours
    de bourse = 5 jours calendaires, pas 7 -- cf. échange sur l'item 5, pas une erreur)."""
    d7 = df[df["horizon_unit"] == "D+7"].copy()
    d7["delta_days"] = (pd.to_datetime(d7["target_date"]) - pd.to_datetime(d7["cutoff_date"])).dt.days
    non_crypto = d7[~d7["asset"].isin(["BTC-USD", "ETH-USD"])]
    crypto = d7[d7["asset"].isin(["BTC-USD", "ETH-USD"])]
    return {
        "n_total": int(len(d7)),
        "non_crypto_delta_distribution": non_crypto["delta_days"].value_counts().sort_index().to_dict(),
        "non_crypto_n_outside_5_10": int((~non_crypto["delta_days"].between(5, 10)).sum()),
        "crypto_delta_distribution": crypto["delta_days"].value_counts().sort_index().to_dict(),
        "crypto_n_outside_5": int((crypto["delta_days"] != 5).sum()),
    }


def comparison_1_ranking(df: pd.DataFrame) -> list:
    """Per (asset, horizon_unit): rank competitors by point-estimate RMSE, test
    each vs the empirical leader on squared-error differences (paired by
    target_date, block bootstrap)."""
    results = []
    for asset in sorted(df["asset"].unique()):
        for h in HORIZON_UNITS:
            cell = df[(df["asset"] == asset) & (df["horizon_unit"] == h)].copy()
            if cell.empty:
                continue
            cell["competitor"] = [
                _competitor_label(m, f, h) for m, f in zip(cell["model"], cell["frequence"])
            ]
            piv = cell.pivot_table(index="target_date", columns="competitor", values="sq_error")
            piv = piv.sort_index()   # chronological order, required for block bootstrap
            competitors = list(piv.columns)
            if len(competitors) < 2:
                continue
            rmse_by_competitor = {c: float(np.sqrt(piv[c].mean())) for c in competitors}
            leader = min(rmse_by_competitor, key=rmse_by_competitor.get)

            for comp in competitors:
                if comp == leader:
                    continue
                paired = piv[[leader, comp]].dropna()
                if len(paired) < MIN_PAIRED_POINTS:
                    results.append({
                        "asset": asset, "horizon_unit": h, "leader": leader, "competitor": comp,
                        "leader_rmse": rmse_by_competitor[leader], "competitor_rmse": rmse_by_competitor[comp],
                        "status": "insufficient_data", "n": int(len(paired)),
                    })
                    continue
                diffs = (paired[comp] - paired[leader]).values   # >0 => leader has lower sq error
                test = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
                verdict = "significantly_worse_than_leader" if test["significant_at_05"] and test["mean_diff"] > 0 \
                    else "indistinguishable_from_leader"
                results.append({
                    "asset": asset, "horizon_unit": h, "leader": leader, "competitor": comp,
                    "leader_rmse": rmse_by_competitor[leader], "competitor_rmse": rmse_by_competitor[comp],
                    "status": "tested", "verdict": verdict, **test,
                })
    return results


def comparison_2_calibration(df: pd.DataFrame) -> list:
    """Per (asset, horizon_unit, model, frequence): is Cov95 significantly
    different from 0.95 (in_interval - 0.95 tested against zero)?"""
    results = []
    group_cols = ["asset", "horizon_unit", "model", "frequence"]
    for keys, g in df.groupby(group_cols):
        asset, h, model, freq = keys
        g = g.sort_values("target_date")
        diffs = (g["in_interval"] - 0.95).values
        if len(diffs) < MIN_PAIRED_POINTS:
            results.append({"asset": asset, "horizon_unit": h, "model": model, "frequence": freq,
                            "cov95_observed": float(g["in_interval"].mean()),
                            "status": "insufficient_data", "n": int(len(diffs))})
            continue
        test = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
        verdict = "significantly_miscalibrated" if test["significant_at_05"] else "not_significantly_different_from_0.95"
        results.append({
            "asset": asset, "horizon_unit": h, "model": model, "frequence": freq,
            "cov95_observed": float(g["in_interval"].mean()), "status": "tested",
            "verdict": verdict, **test,
        })
    return results


def build_daily_weekly_pairs(df: pd.DataFrame, horizon_units=None) -> pd.DataFrame:
    """Regime B (frequence=daily, daily-trained model evaluated at its native
    weekly target) vs regime C (frequence=weekly, weekly-native model), paired
    by (model, asset, horizon_unit, target_date) -- both sides already share the
    exact same target_date and cutoff_date by construction (no origin-alignment
    approximation needed, unlike the D+7/W+1 Friday-restricted join): same
    protocol inside a model, the cleanest comparison (no cross-model asymmetry).
    Factored out of comparison_3_daily_vs_weekly so downstream consumers needing
    the per-origin merged rows share the exact same pairing -- see
    experiments/dashboard_d7_w1.py."""
    weekly = df[df["horizon_type"] == "weekly"]
    if horizon_units is not None:
        weekly = weekly[weekly["horizon_unit"].isin(horizon_units)]
    daily_side = weekly[weekly["frequence"] == "daily"]
    weekly_side = weekly[weekly["frequence"] == "weekly"]

    frames = []
    for (model, asset, h), g_daily in daily_side.groupby(["model", "asset", "horizon_unit"]):
        g_weekly = weekly_side[(weekly_side["model"] == model) & (weekly_side["asset"] == asset)
                               & (weekly_side["horizon_unit"] == h)]
        merged = g_daily.merge(g_weekly, on="target_date", suffixes=("_daily", "_weekly"))
        merged.insert(0, "model", model)
        merged.insert(1, "asset", asset)
        merged.insert(2, "horizon_unit", h)
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["model", "asset", "horizon_unit", "target_date"])
    return pd.concat(frames, ignore_index=True)


def comparison_3_daily_vs_weekly(df: pd.DataFrame) -> list:
    """Per (model, asset, horizon_unit in W+1/2/3): regime B (frequence=daily)
    vs regime C (frequence=weekly), paired by target_date -- same protocol
    inside a model, the cleanest comparison (no cross-model asymmetry)."""
    results = []
    pairs = build_daily_weekly_pairs(df)

    # enumerate the same (model, asset, horizon_unit) keys as before the
    # refactor: only combos present on BOTH sides (a combo missing one side
    # entirely was silently skipped by the original pivot_table check, not
    # reported as insufficient_data -- preserved here).
    weekly = df[df["horizon_type"] == "weekly"]
    daily_keys = set(map(tuple, weekly[weekly["frequence"] == "daily"][["model", "asset", "horizon_unit"]]
                        .drop_duplicates().values))
    weekly_keys = set(map(tuple, weekly[weekly["frequence"] == "weekly"][["model", "asset", "horizon_unit"]]
                         .drop_duplicates().values))

    for model, asset, h in sorted(daily_keys & weekly_keys):
        if pairs.empty:
            merged = pairs
        else:
            merged = pairs[(pairs["model"] == model) & (pairs["asset"] == asset)
                          & (pairs["horizon_unit"] == h)].sort_values("target_date")
        if len(merged) < MIN_PAIRED_POINTS:
            results.append({"model": model, "asset": asset, "horizon_unit": h,
                            "status": "insufficient_data", "n": int(len(merged))})
            continue
        diffs = (merged["sq_error_daily"] - merged["sq_error_weekly"]).values   # >0 => weekly-native has lower sq error
        test = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
        if test["significant_at_05"]:
            verdict = "weekly_native_significantly_better" if test["mean_diff"] > 0 \
                else "daily_multistep_significantly_better"
        else:
            verdict = "indistinguishable"
        results.append({
            "model": model, "asset": asset, "horizon_unit": h, "status": "tested",
            "verdict": verdict, **test,
        })
    return results


def build_d7_w1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Friday-aligned D+7 (regime A) / W+1 (regime C, weekly-native) join, one row
    per (model, asset, origin-Friday) -- factored out of comparison_4_d7_vs_w1 so
    downstream consumers needing the per-origin merged rows (not just the
    aggregated test verdict) share the exact same pairing instead of recomputing
    it (see experiments/dashboard_d7_w1.py, which imports this rather than
    reimplementing the join)."""
    d7 = df[df["horizon_unit"] == "D+7"].copy()
    d7["cutoff_dt"] = pd.to_datetime(d7["cutoff_date"])
    d7_fridays = d7[d7["cutoff_dt"].dt.weekday == 4]   # Monday=0 ... Friday=4

    w1 = df[(df["horizon_unit"] == "W+1") & (df["frequence"] == "weekly")].copy()

    frames = []
    for (model, asset), g_d7 in d7_fridays.groupby(["model", "asset"]):
        g_w1 = w1[(w1["model"] == model) & (w1["asset"] == asset)]
        merged = g_d7.merge(g_w1, on="cutoff_date", suffixes=("_d7", "_w1"))
        merged.insert(0, "model", model)
        merged.insert(1, "asset", asset)
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["model", "asset", "cutoff_date"])
    return pd.concat(frames, ignore_index=True)


def comparison_4_d7_vs_w1(df: pd.DataFrame) -> list:
    """Per (model, asset): D+7 (regime A) rows whose cutoff_date is a Friday,
    paired with the W+1 (regime C, weekly-native) row sharing that same origin
    Friday -- "pour 1 semaine, daily ou weekly natif ?", tested."""
    results = []
    pairs = build_d7_w1_pairs(df)

    # enumerate the same (model, asset) keys as before the refactor, so a Friday
    # D+7 group with zero matching W+1 rows still reports insufficient_data(n=0)
    # instead of silently vanishing from a 0-row pandas groupby.
    d7 = df[df["horizon_unit"] == "D+7"].copy()
    d7["cutoff_dt"] = pd.to_datetime(d7["cutoff_date"])
    all_keys = d7[d7["cutoff_dt"].dt.weekday == 4][["model", "asset"]].drop_duplicates()

    for _, key_row in all_keys.iterrows():
        model, asset = key_row["model"], key_row["asset"]
        if pairs.empty:
            merged = pairs
        else:
            merged = pairs[(pairs["model"] == model) & (pairs["asset"] == asset)].sort_values("cutoff_date")
        if len(merged) < MIN_PAIRED_POINTS:
            results.append({"model": model, "asset": asset, "status": "insufficient_data",
                            "n": int(len(merged))})
            continue
        diffs = (merged["sq_error_d7"] - merged["sq_error_w1"]).values   # >0 => W+1 better
        test = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, len(diffs)))
        if test["significant_at_05"]:
            verdict = "weekly_native_significantly_better" if test["mean_diff"] > 0 \
                else "daily_D+7_significantly_better"
        else:
            verdict = "indistinguishable"
        results.append({"model": model, "asset": asset, "status": "tested",
                        "verdict": verdict, **test})
    return results


def summarize_significant(comp1, comp2, comp3, comp4) -> dict:
    sig1 = [r for r in comp1 if r.get("verdict") == "significantly_worse_than_leader"]
    leaders = {(r["asset"], r["horizon_unit"]): r["leader"] for r in comp1}
    miscal = [r for r in comp2 if r.get("verdict") == "significantly_miscalibrated"]
    reg_diff = [r for r in comp3 if r.get("verdict") not in (None, "indistinguishable")]
    d7w1_diff = [r for r in comp4 if r.get("verdict") not in (None, "indistinguishable")]
    return {
        "n_leader_cells": len(leaders),
        "n_significantly_worse_than_leader": len(sig1),
        "n_significantly_miscalibrated": len(miscal),
        "n_regime_B_vs_C_significant": len(reg_diff),
        "regime_B_vs_C_significant_details": reg_diff,
        "n_D7_vs_W1_significant": len(d7w1_diff),
        "D7_vs_W1_significant_details": d7w1_diff,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "matrice_paired_tests.json"))
    args = p.parse_args()

    df = load_predictions(args.db_path)
    print(f"Loaded {len(df)} OOS predictions.")

    d7_check = verify_d7_definition(df)
    print(f"D+7 definition check: {json.dumps(d7_check, default=str)}")

    print("Comparison 1: ranking per (asset, horizon_unit) ...")
    comp1 = comparison_1_ranking(df)
    print(f"  {len(comp1)} competitor-vs-leader tests")

    print("Comparison 2: calibration per (asset, horizon_unit, model, frequence) ...")
    comp2 = comparison_2_calibration(df)
    print(f"  {len(comp2)} calibration tests")

    print("Comparison 3: daily (regime B) vs weekly (regime C) per model ...")
    comp3 = comparison_3_daily_vs_weekly(df)
    print(f"  {len(comp3)} B-vs-C tests")

    print("Comparison 4: D+7 vs W+1 (Friday-aligned) per model x asset ...")
    comp4 = comparison_4_d7_vs_w1(df)
    print(f"  {len(comp4)} D+7-vs-W+1 tests")

    summary = summarize_significant(comp1, comp2, comp3, comp4)

    payload = {
        "config": {
            "block_length": BLOCK_LENGTH, "min_paired_points": MIN_PAIRED_POINTS,
            "power_caveat": ("30 origines hebdomadaires chevauchantes (W1-W3) -> puissance "
                            "effective ~n/block_length (souvent ~10), pas n. Les p-values "
                            "restent probablement optimistes malgre le bootstrap par blocs "
                            "(la vraie structure de correlation peut depasser 3 origines). "
                            "Lire tout resultat 'significatif' avec cette reserve."),
            "inter_model_caveat": ("Comparaison 1 (classement inter-modeles) : asymetrie de "
                                  "protocole non neutralisee (TSDiff fige a T0, les 5 autres "
                                  "re-entraines a chaque origine) -- a interpreter avec prudence. "
                                  "Comparaisons 3 et 4 (intra-modele) n'en souffrent pas."),
        },
        "d7_definition_check": d7_check,
        "comparison_1_ranking_per_asset_horizon": comp1,
        "comparison_2_calibration": comp2,
        "comparison_3_daily_vs_weekly_per_model": comp3,
        "comparison_4_d7_vs_w1_friday_aligned": comp4,
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}")
    print(f"\nSummary: {json.dumps(summary, indent=2, default=str)}")


if __name__ == "__main__":
    main()
