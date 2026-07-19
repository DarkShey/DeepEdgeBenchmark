"""
pooled_analysis.py — pooled, scale-free significance tests across the full
matrix (BRIEF_analyse_poolee.md). No new data/training: re-analyses the same
240 cells already in tracking.db, but POOLS evidence per model instead of
scattering it across 15+ separate per-cell tests, to turn "inconclusive on 30
origins" into a firm per-model verdict wherever the evidence supports it.

Three questions, all pooled BY MODEL:
  1. Frequency (regime B vs C) -- does training natively on weekly data change
     anything, pooled across all of a model's (asset x horizon) cells?
  2. D+7 vs W+1, Friday-aligned, same pooling (low power flagged explicitly).
  3. Calibration -- is a model's pooled coverage significantly different from
     the 0.95 target?

NOT pooled: the inter-model ranking (comparison 1 of matrice_paired_tests.py)
-- it still carries the training-protocol asymmetry (TSDiff frozen at T0, the
other 5 models refit at every origin) that no statistical test corrects. That
stays a separate, later brief (protocol unification + re-run).

── Scale-free loss differential (§2.1) ──────────────────────────────────────
RMSE in raw price units is NOT comparable across assets of wildly different
scale (BTC ~$60k vs ZN=F ~$110) -- pooling requires an adimensional loss:
  - Point: MASE-style scaling -- |y_pred - y_true| / scale_asset, where
    scale_asset = mean(|p_t - p_t-1|) over the IN-SAMPLE (pre-test, no
    lookahead) daily price history of that asset (Hyndman & Koehler 2006's
    classic MASE denominator: the naive 1-step in-sample MAE, independent of
    forecast horizon -- this is what "reste cohérent" calls for: one scale per
    asset, used everywhere, not recomputed per horizon or per model).
  - Distribution: CRPS normalised by the same scale_asset. IMPORTANT CAVEAT:
    this matrix only stores point + 95% PI (y_pred/y_lower/y_upper), never a
    raw sample cloud, for ANY of the 6 models (unlike the earlier TSDiff-only
    weekly work) -- so "CRPS" here is the closed-form GAUSSIAN approximation
    (honest_eval.metrics.crps_gaussian, mu=y_pred, sigma=(y_upper-y_lower)/(2*1.96)),
    not an empirical sample-based CRPS. Reported as a secondary, complementary
    differential alongside the primary MASE-point differential -- if both
    agree, that is itself informative (accuracy AND calibration point the same
    direction).

── Clustering by asset class (§2.3) ──────────────────────────────────────────
ZN=F/TLT (same underlying: US Treasury rates) and BTC-USD/ETH-USD (same
underlying: crypto beta) are correlated -- treating 5 assets as 5 independent
observations double-counts each pair. Both correlated pairs are averaged
(paired by matching cutoff/target date) into one "bonds" / one "crypto"
series BEFORE pooling, leaving 3 independent asset-class series (index=SPY,
bonds, crypto) that are then concatenated for the pooled test.

── Test + robustness (§2.2) ─────────────────────────────────────────────────
Primary: Diebold-Mariano with Newey-West/HAC variance (honest_eval.metrics.
dm_hac_test), lag = max horizon among the pooled cells (in weeks for the
weekly comparisons, conservative). Robustness: the existing block bootstrap
(experiments.paired_test.paired_block_bootstrap_test, block_length=3). The two
must agree in sign/significance; disagreement is flagged, not hidden.

Multiple comparisons: Holm correction across the 6 models within each question.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import tsdiff_model as td                                       # noqa: E402
from honest_eval.metrics import dm_hac_test, crps_gaussian        # noqa: E402
from paired_test import paired_block_bootstrap_test               # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS             # noqa: E402

MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive", "TSDiff")
ASSET_CLASS = {"SPY": "index", "ZN=F": "bond", "TLT": "bond",
              "BTC-USD": "crypto", "ETH-USD": "crypto"}
CORRELATED_PAIRS = {"bond": ("ZN=F", "TLT"), "crypto": ("BTC-USD", "ETH-USD")}
STANDALONE_CLASSES = {"index": "SPY"}
BLOCK_LENGTH = 3
MIN_POINTS = 8   # dm_hac_test's own floor; below this a class contributes nothing


def compute_asset_scales(start: str, end: str) -> dict:
    """In-sample MASE scale per asset: mean |1-day price change| over
    [start, end), end = first test-origin date (no lookahead into the test
    window). One number per asset, used for every horizon/model."""
    scales = {}
    for short, ticker in ASSET_TICKERS.items():
        prices = td.fetch_data(ticker, start, end)
        scales[ticker] = float(np.mean(np.abs(np.diff(prices.values))))
    return scales


def load_predictions(db_path: str, scales: dict) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT model, asset, frequence, horizon_type, horizon_unit,
               cutoff_date, target_date, y_pred, y_lower, y_upper, y_true
        FROM predictions
        WHERE source = 'oos'
    """, con)
    con.close()
    df["scale"] = df["asset"].map(scales)
    df["mase_error"] = np.abs(df["y_pred"] - df["y_true"]) / df["scale"]
    sigma_implied = (df["y_upper"] - df["y_lower"]) / (2 * 1.96)
    df["crps_norm"] = [
        crps_gaussian(mu, max(sigma, 1e-9), y) / scale
        for mu, sigma, y, scale in zip(df["y_pred"], sigma_implied, df["y_true"], df["scale"])
    ]
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    df["asset_class"] = df["asset"].map(ASSET_CLASS)
    return df


def class_series(cell_df: pd.DataFrame, value_col: str, date_col: str = "target_date") -> dict:
    """Collapse a (model, ...) slice's rows into 3 independent asset-class
    series: standalone assets used as-is; correlated pairs averaged after an
    inner join on `date_col` (only dates present for BOTH assets contribute --
    no fabricated alignment). Returns {class_name: pd.Series(index=date, value)}."""
    out = {}
    for cls, asset in STANDALONE_CLASSES.items():
        sub = cell_df[cell_df["asset"] == asset]
        if not sub.empty:
            out[cls] = sub.set_index(date_col)[value_col].sort_index()

    for cls, (a1, a2) in CORRELATED_PAIRS.items():
        s1 = cell_df[cell_df["asset"] == a1].set_index(date_col)[value_col]
        s2 = cell_df[cell_df["asset"] == a2].set_index(date_col)[value_col]
        joined = pd.concat([s1, s2], axis=1, join="inner")
        if not joined.empty:
            out[cls] = joined.mean(axis=1).sort_index()
    return out


def pooled_diff_series(df_a: pd.DataFrame, df_b: pd.DataFrame, value_col: str,
                       date_col: str = "target_date") -> np.ndarray:
    """Build ONE pooled, class-clustered differential series (df_a's value -
    df_b's value) for a model, concatenating the 3 asset-class series (each
    itself possibly spanning several horizons, already concatenated upstream
    by the caller via horizon_unit ordering) in a fixed, reproducible order."""
    diffs = []
    for cls in ("index", "bond", "crypto"):
        sa = class_series(df_a, value_col, date_col).get(cls)
        sb = class_series(df_b, value_col, date_col).get(cls)
        if sa is None or sb is None:
            continue
        joined = pd.concat([sa, sb], axis=1, join="inner")
        if joined.empty:
            continue
        diffs.append((joined.iloc[:, 0] - joined.iloc[:, 1]).values)
    return np.concatenate(diffs) if diffs else np.array([])


def dual_test(diffs: np.ndarray, h: int) -> dict:
    """Runs both DM-HAC and block bootstrap on the same differential and
    reports both plus a concordance flag (same sign, same significance call)."""
    n = len(diffs)
    if n < MIN_POINTS:
        return {"status": "insufficient_data", "n": int(n)}
    dm = dm_hac_test(diffs, h=h)
    bl = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, n))
    concordant = (dm["p_value"] < 0.05) == bl["significant_at_05"] and (
        np.sign(dm["mean_diff"]) == np.sign(bl["mean_diff"]) or abs(dm["mean_diff"]) < 1e-12
    )
    return {
        "status": "tested", "n": int(n), "effective_n": bl["effective_n"],
        "mean_diff": dm["mean_diff"], "ci95_lo": bl["ci95_lo"], "ci95_hi": bl["ci95_hi"],
        "dm_stat": dm["dm_stat"], "dm_lag": dm["lag"], "p_value_dm": dm["p_value"],
        "p_value_bootstrap": bl["p_value"],
        "significant_dm": bool(dm["p_value"] < 0.05),
        "significant_bootstrap": bl["significant_at_05"],
        "concordant": bool(concordant),
    }


def holm_correction(pvals: list) -> list:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    ORIGINAL order of `pvals` (standard monotone Holm adjustment)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def verdict_from_ci(ci_lo: float, ci_hi: float, significant: bool, favors: str, against: str) -> str:
    if not significant:
        return f"pas de difference exploitable -- IC contenu dans [{ci_lo:.4f}, {ci_hi:.4f}]"
    direction = favors if (ci_lo + ci_hi) / 2 < 0 else against
    return f"significatif ({direction} a un cout/erreur plus faible)"


# ── Question 1: frequency, regime B vs C, pooled per model ─────────────────

def question_frequency(df: pd.DataFrame) -> dict:
    weekly = df[df["horizon_type"] == "weekly"].copy()
    # GUARDRAIL: pooling W+1/W+2/W+3 together means plain target_date is NOT a safe
    # join key -- origin_k's W+2 target lands on the EXACT same calendar date as
    # origin_{k+1}'s W+1 target (origins are 1 week apart), which would silently
    # collide two unrelated (origin, horizon) predictions under a bare date key.
    # Composite key (cutoff_date, horizon_unit) is unambiguous within one model.
    weekly["origin_key"] = weekly["cutoff_date"].astype(str) + "|" + weekly["horizon_unit"]
    results = {}
    for model in MODELS:
        m = weekly[weekly["model"] == model]
        b = m[m["frequence"] == "daily"]     # regime B: daily-trained, multi-step
        c = m[m["frequence"] == "weekly"]    # regime C: weekly-native
        diffs_mase = pooled_diff_series(b, c, "mase_error", date_col="origin_key")
        diffs_crps = pooled_diff_series(b, c, "crps_norm", date_col="origin_key")
        test_mase = dual_test(diffs_mase, h=3)     # h=3: max horizon pooled (W+1..W+3, weeks)
        test_crps = dual_test(diffs_crps, h=3)
        entry = {"mase_point_metric": test_mase, "crps_distributional_metric": test_crps}
        if test_mase.get("status") == "tested":
            entry["verdict"] = verdict_from_ci(
                test_mase["ci95_lo"], test_mase["ci95_hi"], test_mase["significant_bootstrap"],
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
        # align on cutoff_date (the shared Friday origin), not target_date
        diffs_mase = pooled_diff_series(a, b, "mase_error", date_col="cutoff_date")
        diffs_crps = pooled_diff_series(a, b, "crps_norm", date_col="cutoff_date")
        test_mase = dual_test(diffs_mase, h=1)
        test_crps = dual_test(diffs_crps, h=1)
        entry = {"mase_point_metric": test_mase, "crps_distributional_metric": test_crps,
                 "power_note": "D+7 n'a que ~10 origines glissantes eparses par (modele,actif); "
                               "l'alignement vendredi en retient une fraction -- puissance tres "
                               "limitee, verdict a lire avec beaucoup de prudence."}
        if test_mase.get("status") == "tested":
            entry["verdict"] = verdict_from_ci(
                test_mase["ci95_lo"], test_mase["ci95_hi"], test_mase["significant_bootstrap"],
                favors="D+7 (daily)", against="W+1 (weekly natif)")
        else:
            entry["verdict"] = "donnees insuffisantes"
        results[model] = entry
    return results


# ── Question 3: calibration, pooled per model ───────────────────────────────

def kupiec_pof_test(x: int, T: int, p: float = 0.95) -> dict:
    """Classic Kupiec (1995) proportion-of-failures likelihood-ratio test:
    H0: true coverage = p. ASSUMES i.i.d. Bernoulli trials -- optimistic here
    (origins are correlated), reported for reference/interpretability
    alongside the block-bootstrap verdict (which does not assume independence
    and is the trustworthy number, cf. module docstring)."""
    if T == 0:
        return {"lr_stat": float("nan"), "p_value": float("nan"), "phat": float("nan")}
    phat = x / T
    if phat in (0.0, 1.0):
        # degenerate likelihood ratio (log(0)) -- fall back to a one-sided flag
        return {"lr_stat": float("inf"), "p_value": 0.0, "phat": phat}
    ll_null = x * np.log(p) + (T - x) * np.log(1 - p)
    ll_alt = x * np.log(phat) + (T - x) * np.log(1 - phat)
    lr = -2.0 * (ll_null - ll_alt)
    pval = float(1.0 - stats.chi2.cdf(lr, df=1))
    return {"lr_stat": float(lr), "p_value": pval, "phat": float(phat)}


def question_calibration(df: pd.DataFrame) -> dict:
    results = {}
    for model in MODELS:
        m = df[df["model"] == model].copy()
        m["diff_from_target"] = m["in_interval"] - 0.95

        # naive pooled Kupiec/binomial -- ALL rows treated as independent trials
        # (explicitly optimistic, cf. docstring)
        x = int(m["in_interval"].sum())
        T = int(len(m))
        kupiec = kupiec_pof_test(x, T, p=0.95)

        # cluster-aware, block-bootstrap verdict -- concatenate the 3 asset-class
        # series (bonds/crypto pre-averaged) across ALL horizons for this model.
        # Must split by (horizon_unit, frequence): weekly horizon_units carry BOTH
        # regime B (frequence='daily') and regime C (frequence='weekly') rows that
        # share the SAME target_date for a given asset -- pooling them into one
        # cell before indexing by date creates duplicate labels. cutoff_date is
        # unique within a single (model, asset, horizon_unit, frequence) group
        # (DB unique index), so it's used as the join key instead of target_date.
        class_dfs = []
        for h in m["horizon_unit"].unique():
            for freq in m.loc[m["horizon_unit"] == h, "frequence"].unique():
                cell = m[(m["horizon_unit"] == h) & (m["frequence"] == freq)]
                for cls, series in class_series(cell, "diff_from_target", date_col="cutoff_date").items():
                    class_dfs.append(series)
        pooled = np.concatenate([s.values for s in class_dfs]) if class_dfs else np.array([])

        if len(pooled) < MIN_POINTS:
            results[model] = {"status": "insufficient_data", "n": int(len(pooled)),
                              "kupiec_naive": kupiec}
            continue
        bl = paired_block_bootstrap_test(pooled, block_length=min(BLOCK_LENGTH, len(pooled)))
        verdict = "significativement mal calibre" if bl["significant_at_05"] else \
            f"pas de miscalibration exploitable -- IC contenu dans [{bl['ci95_lo']:.4f}, {bl['ci95_hi']:.4f}]"
        results[model] = {
            "status": "tested", "n_raw_pooled_naive": T, "cov95_observed_naive": x / T if T else None,
            "kupiec_naive": kupiec,
            "n_cluster_aware": int(len(pooled)), "effective_n": bl["effective_n"],
            "coverage_gap_mean": bl["mean_diff"], "ci95_lo": bl["ci95_lo"], "ci95_hi": bl["ci95_hi"],
            "p_value_bootstrap": bl["p_value"], "significant_bootstrap": bl["significant_at_05"],
            "verdict": verdict,
        }
    return results


def apply_holm(question_results: dict, pvalue_path) -> None:
    """In-place: adds `p_value_bootstrap_holm` next to each model's bootstrap
    p-value, Holm-corrected across the 6 models of this question."""
    models = list(question_results)
    pvals = []
    for model in models:
        entry = question_results[model]
        p = pvalue_path(entry)
        pvals.append(p if p is not None else 1.0)
    adjusted = holm_correction(pvals)
    for model, adj in zip(models, adjusted):
        question_results[model]["p_value_bootstrap_holm"] = adj
        question_results[model]["significant_after_holm"] = bool(adj < 0.05)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--scale-start", default="2015-01-01")
    p.add_argument("--scale-end", default="2025-12-05")   # first test origin -- no lookahead
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "pooled_analysis.json"))
    args = p.parse_args()

    print("Computing in-sample MASE scales per asset ...")
    scales = compute_asset_scales(args.scale_start, args.scale_end)
    print(f"  scales: {scales}")

    df = load_predictions(args.db_path, scales)
    print(f"Loaded {len(df)} OOS predictions.")

    print("Question 1: frequency (regime B vs C), pooled per model ...")
    q1 = question_frequency(df)
    apply_holm(q1, lambda e: e["mase_point_metric"].get("p_value_bootstrap") if e["mase_point_metric"].get("status") == "tested" else None)

    print("Question 2: D+7 vs W+1 (Friday-aligned), pooled per model ...")
    q2 = question_d7_vs_w1(df)
    apply_holm(q2, lambda e: e["mase_point_metric"].get("p_value_bootstrap") if e["mase_point_metric"].get("status") == "tested" else None)

    print("Question 3: calibration, pooled per model ...")
    q3 = question_calibration(df)
    apply_holm(q3, lambda e: e.get("p_value_bootstrap") if e.get("status") == "tested" else None)

    payload = {
        "config": {
            "asset_scales_mase": scales, "scale_window": [args.scale_start, args.scale_end],
            "block_length": BLOCK_LENGTH, "min_points": MIN_POINTS,
            "asset_classes": ASSET_CLASS, "correlated_pairs_merged": CORRELATED_PAIRS,
            "multiple_comparison_correction": "Holm, across the 6 models within each question",
            "crps_caveat": ("CRPS ici = approximation gaussienne (mu=y_pred, "
                           "sigma=(y_upper-y_lower)/(2*1.96)), pas un CRPS empirique sur "
                           "echantillons -- cette matrice ne stocke que point+PI95 pour "
                           "les 6 modeles, jamais un nuage d'echantillons."),
            "inter_model_ranking_caveat": ("Le classement inter-modeles n'est PAS poole ici "
                                          "(hors scope de ce brief) : il porte toujours "
                                          "l'asymetrie de protocole (TSDiff fige a T0 vs les 5 "
                                          "autres re-entraines a chaque origine), qu'aucun test "
                                          "statistique ne corrige. Traite par un re-run futur."),
            "power_caveat": ("30 origines hebdomadaires chevauchantes -> puissance effective "
                            "~n/block_length, pas n. D+7 (~10 origines eparses) est encore "
                            "plus limite, surtout apres filtrage vendredi (question 2)."),
        },
        "question_1_frequency_B_vs_C": q1,
        "question_2_D7_vs_W1": q2,
        "question_3_calibration": q3,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}")

    print("\n=== Summary ===")
    for model in MODELS:
        f = q1[model]["verdict"]
        d = q2[model]["verdict"]
        c = q3[model].get("verdict", "donnees insuffisantes")
        print(f"{model:<14} freq(B vs C): {f}")
        print(f"{'':<14} D7 vs W1    : {d}")
        print(f"{'':<14} calibration : {c}")


if __name__ == "__main__":
    main()
