"""
pooled_analysis_prob.py — BRIEF_kpi_probabilistes.md step 3: replay
pooled_analysis.py's exact statistical machinery (DM-HAC + block bootstrap
dual test, Holm-Bonferroni across the 6 models, asset-class clustering:
bonds=ZN=F+TLT averaged, crypto=BTC-USD+ETH-USD averaged, index=SPY standalone)
but on the NEW probabilistic KPIs from experiments/kpi_probabilistes.json
(empirical CRPS on the N=500 sample cloud, multi-level calibration) instead of
RMSE / the closed-form Gaussian-CRPS approximation pooled_analysis.py used
(it never had a sample cloud to work with -- this brief's whole point).

Same 3 questions as pooled_analysis.py, same guardrails (no verdict without a
test, effective-N always reported, Holm-corrected, protocol-asymmetry caveat
on inter-model ranking still applies and is NOT pooled here either):
  1. Frequency: regime B (daily->weekly) vs regime C (weekly-native), pooled
     per model, on CRPS_norm (now empirical, asset-scale-normalised).
  2. D+7 vs W+1 (Friday-aligned), pooled per model, on CRPS_norm.
  3. Calibration -- extended to THREE levels (50/80/95, brief Sec.2), not just
     95%: is a model's pooled coverage gap at each level significantly
     different from the nominal level.

Reuses pooled_analysis.py's class_series/pooled_diff_series/dual_test/
holm_correction UNCHANGED (imported, not reimplemented) -- same clustering,
same tests, only the input metric differs.

Usage:
    python pooled_analysis_prob.py
    python pooled_analysis_prob.py --kpi-json kpi_probabilistes.json
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

from pooled_analysis import (                                    # noqa: E402
    class_series, pooled_diff_series, dual_test, holm_correction, apply_holm,
    verdict_from_ci, compute_asset_scales, MODELS, ASSET_CLASS, MIN_POINTS,
)

KPI_JSON_DEFAULT = ROOT / "experiments" / "kpi_probabilistes.json"
OUT_PATH_DEFAULT = ROOT / "experiments" / "pooled_analysis_prob.json"


def load_kpi_df(kpi_json_path) -> pd.DataFrame:
    payload = json.loads(Path(kpi_json_path).read_text())
    df = pd.DataFrame(payload["per_row"])
    df["asset_class"] = df["asset"].map(ASSET_CLASS)
    return df


def add_crps_norm(df: pd.DataFrame, scales: dict) -> pd.DataFrame:
    """CRPS normalised by the same in-sample MASE-style asset scale
    pooled_analysis.py uses -- required before pooling across assets of very
    different price magnitudes (brief guardrail, [[feedback-statistical-rigor]])."""
    df = df.copy()
    df["scale"] = df["asset"].map(scales)
    df["crps_norm"] = df["crps"] / df["scale"]
    return df


# ── Question 1: frequency, regime B vs C, pooled per model ─────────────────

def question_frequency(df: pd.DataFrame) -> dict:
    weekly = df[df["horizon_type"] == "weekly"].copy()
    weekly["origin_key"] = weekly["cutoff_date"].astype(str) + "|" + weekly["horizon_unit"]
    results = {}
    for model in MODELS:
        m = weekly[weekly["model"] == model]
        b = m[m["frequence"] == "daily"]     # regime B: daily-trained, multi-step
        c = m[m["frequence"] == "weekly"]    # regime C: weekly-native
        diffs_crps = pooled_diff_series(b, c, "crps_norm", date_col="origin_key")
        test_crps = dual_test(diffs_crps, h=3)
        entry = {"crps_empirical_metric": test_crps}
        if test_crps.get("status") == "tested":
            entry["verdict"] = verdict_from_ci(
                test_crps["ci95_lo"], test_crps["ci95_hi"], test_crps["significant_bootstrap"],
                favors="regime B (daily->weekly)", against="regime C (weekly natif)")
        else:
            entry["verdict"] = "donnees insuffisantes"
        results[model] = entry
    return results


# ── Question 2: D+7 vs W+1, Friday-aligned, pooled per model ────────────────

def question_d7_vs_w1(df: pd.DataFrame) -> dict:
    d7 = df[df["horizon_unit"] == "D+7"].copy()
    d7["cutoff_dt"] = pd.to_datetime(d7["cutoff_date"])
    d7_fri = d7[d7["cutoff_dt"].dt.weekday == 4]
    w1 = df[(df["horizon_unit"] == "W+1") & (df["frequence"] == "weekly")]

    results = {}
    for model in MODELS:
        a = d7_fri[d7_fri["model"] == model]
        b = w1[w1["model"] == model]
        diffs_crps = pooled_diff_series(a, b, "crps_norm", date_col="cutoff_date")
        test_crps = dual_test(diffs_crps, h=1)
        entry = {"crps_empirical_metric": test_crps,
                 "power_note": "D+7 n'a que ~10 origines glissantes eparses par (modele,actif); "
                               "l'alignement vendredi en retient une fraction -- puissance tres "
                               "limitee, verdict a lire avec beaucoup de prudence."}
        if test_crps.get("status") == "tested":
            entry["verdict"] = verdict_from_ci(
                test_crps["ci95_lo"], test_crps["ci95_hi"], test_crps["significant_bootstrap"],
                favors="D+7 (daily)", against="W+1 (weekly natif)")
        else:
            entry["verdict"] = "donnees insuffisantes"
        results[model] = entry
    return results


# ── Question 3: calibration, pooled per model, THREE levels ────────────────

def question_calibration(df: pd.DataFrame, level: float) -> dict:
    """Same clustering/testing logic as pooled_analysis.py's question_calibration,
    generalised to any nominal `level` (0.5 / 0.8 / 0.95 -- brief Sec.2 multi-
    level calibration, not just 95%)."""
    cov_col = f"cov{int(level * 100)}"
    results = {}
    for model in MODELS:
        m = df[df["model"] == model].copy()
        m["diff_from_target"] = m[cov_col].astype(float) - level

        x = int(m[cov_col].sum())
        T = int(len(m))

        class_dfs = []
        for h in m["horizon_unit"].unique():
            for freq in m.loc[m["horizon_unit"] == h, "frequence"].unique():
                cell = m[(m["horizon_unit"] == h) & (m["frequence"] == freq)]
                for cls, series in class_series(cell, "diff_from_target", date_col="cutoff_date").items():
                    class_dfs.append(series)
        pooled = np.concatenate([s.values for s in class_dfs]) if class_dfs else np.array([])

        if len(pooled) < MIN_POINTS:
            results[model] = {"status": "insufficient_data", "n": int(len(pooled)),
                              "n_raw_pooled_naive": T,
                              "cov_observed_naive": (x / T if T else None)}
            continue

        from paired_test import paired_block_bootstrap_test
        from pooled_analysis import BLOCK_LENGTH
        bl = paired_block_bootstrap_test(pooled, block_length=min(BLOCK_LENGTH, len(pooled)))
        verdict = "significativement mal calibre" if bl["significant_at_05"] else \
            f"pas de miscalibration exploitable -- IC contenu dans [{bl['ci95_lo']:.4f}, {bl['ci95_hi']:.4f}]"
        results[model] = {
            "status": "tested", "level": level,
            "n_raw_pooled_naive": T, "cov_observed_naive": x / T if T else None,
            "n_cluster_aware": int(len(pooled)), "effective_n": bl["effective_n"],
            "coverage_gap_mean": bl["mean_diff"], "ci95_lo": bl["ci95_lo"], "ci95_hi": bl["ci95_hi"],
            "p_value_bootstrap": bl["p_value"], "significant_bootstrap": bl["significant_at_05"],
            "verdict": verdict,
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kpi-json", default=str(KPI_JSON_DEFAULT))
    p.add_argument("--scale-start", default="2015-01-01")
    p.add_argument("--scale-end", default="2025-12-05")
    p.add_argument("--out", default=str(OUT_PATH_DEFAULT))
    args = p.parse_args()

    print("Computing in-sample MASE-style asset scales (for CRPS normalisation) ...")
    scales = compute_asset_scales(args.scale_start, args.scale_end)
    print(f"  scales: {scales}")

    df = load_kpi_df(args.kpi_json)
    df = add_crps_norm(df, scales)
    print(f"Loaded {len(df)} probabilistic-KPI rows across {df['asset'].nunique()} assets.")

    print("Question 1: frequency (regime B vs C), pooled per model, empirical CRPS ...")
    q1 = question_frequency(df)
    apply_holm(q1, lambda e: e["crps_empirical_metric"].get("p_value_bootstrap")
              if e["crps_empirical_metric"].get("status") == "tested" else None)

    print("Question 2: D+7 vs W+1 (Friday-aligned), pooled per model, empirical CRPS ...")
    q2 = question_d7_vs_w1(df)
    apply_holm(q2, lambda e: e["crps_empirical_metric"].get("p_value_bootstrap")
              if e["crps_empirical_metric"].get("status") == "tested" else None)

    print("Question 3: calibration, pooled per model, levels 50/80/95 ...")
    q3 = {}
    for level in (0.5, 0.8, 0.95):
        q3_level = question_calibration(df, level)
        apply_holm(q3_level, lambda e: e.get("p_value_bootstrap") if e.get("status") == "tested" else None)
        q3[f"cov{int(level*100)}"] = q3_level

    payload = {
        "config": {
            "asset_scales_crps": scales, "scale_window": [args.scale_start, args.scale_end],
            "asset_classes": ASSET_CLASS,
            "multiple_comparison_correction": "Holm, across the 6 models, applied SEPARATELY per question "
                                              "(and per calibration level)",
            "metric": "empirical CRPS (Gneiting & Raftery 2007 eq.20) on the N=500 sample cloud "
                      "persisted in experiments/samples/ -- NOT the Gaussian approximation "
                      "pooled_analysis.py used (it had no sample cloud to work with).",
            "inter_model_ranking_caveat": ("Le classement inter-modeles n'est PAS poole ici (hors "
                                          "scope) : asymetrie de protocole TSDiff (train-once-forward, "
                                          "et ici de plus RE-entraine pour cette analyse faute de "
                                          "checkpoint persiste) vs les 5 autres modeles echantillonnes "
                                          "parametriquement depuis leur PI stockee sans reentrainement -- "
                                          "deux origines de nuage differentes, aucun test ne corrige ca."),
            "power_caveat": ("30 origines hebdomadaires chevauchantes -> puissance effective "
                            "~n/block_length, pas n. D+7 (~10 origines eparses) est encore "
                            "plus limite. Regime C (TSDiff weekly-native) est quasi-vide dans "
                            "la matrice actuelle (daily_duplicate=1 sur la quasi-totalite des "
                            "lignes, verifie directement) -- question 1 sera 'donnees "
                            "insuffisantes' pour TSDiff dans la plupart des cas."),
        },
        "question_1_frequency_B_vs_C": q1,
        "question_2_D7_vs_W1": q2,
        "question_3_calibration_by_level": q3,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}")

    print("\n=== Summary ===")
    for model in MODELS:
        f = q1[model]["verdict"]
        d = q2[model]["verdict"]
        print(f"{model:<14} freq(B vs C) [CRPS]: {f}")
        print(f"{'':<14} D7 vs W1     [CRPS]: {d}")
        for level in (0.5, 0.8, 0.95):
            c = q3[f"cov{int(level*100)}"][model].get("verdict", "donnees insuffisantes")
            print(f"{'':<14} calibration@{int(level*100)}%    : {c}")


if __name__ == "__main__":
    main()
