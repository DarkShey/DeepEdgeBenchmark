"""
weekly_vs_daily_pooled.py — Test rigoureux régime B (daily->hebdo) vs régime C
(hebdo natif), par modèle, sur les prédictions weekly DÉJÀ en base
(validation/tracking.db, source='oos', 6 modèles x 5 actifs x W+1/2/3).

Motivation : experiments/pooled_analysis.py (question_frequency) fait exactement
ce test, mais (a) il exige le réseau Yahoo pour les échelles d'actifs, (b) il
importe les modèles lourds (torch/statsmodels). Ce script réutilise la MÊME
machinerie statistique validée du dépôt — honest_eval.metrics.dm_hac_test,
paired_test.paired_block_bootstrap_test, crps_gaussian, clustering par classe
d'actif, correction de Holm — mais :
  - dérive l'échelle d'actif (MASE-style) depuis les last_close quotidiens EN
    BASE (pas de réseau) ;
  - n'importe que honest_eval.metrics + paired_test (numpy/scipy), pas les modèles.

Deux tests par modèle, appariés par (origine, horizon), clusterisés par classe
d'actif (bond=ZN=F+TLT moyennés, crypto=BTC-USD+ETH-USD moyennés, index=SPY),
double test DM-HAC + bootstrap par blocs, Holm sur les 6 modèles :
  1. CRPS gaussien normalisé  (précision distributionnelle)
  2. Couverture 95% (in_interval)  (calibration)

Sortie : experiments/weekly_vs_daily_pooled.json

CRPS ici = forme fermée gaussienne récupérée depuis l'intervalle 95% stocké
(sigma=(u-l)/(2*1.96)). EXACT pour les 5 modèles paramétriques (leur IC EST
gaussien/log-normal) ; APPROXIMATION pour TSDiff (nuage non gaussien) — pour un
CRPS empirique il faudrait régénérer les nuages d'échantillons.
"""
import sys, sqlite3, json
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from honest_eval.metrics import dm_hac_test, crps_gaussian          # numpy/scipy only
from paired_test import paired_block_bootstrap_test

MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive", "TSDiff")
ASSET_CLASS = {"SPY": "index", "ZN=F": "bond", "TLT": "bond",
               "BTC-USD": "crypto", "ETH-USD": "crypto"}
CORRELATED_PAIRS = {"bond": ("ZN=F", "TLT"), "crypto": ("BTC-USD", "ETH-USD")}
STANDALONE_CLASSES = {"index": "SPY"}
BLOCK_LENGTH = 3
MIN_POINTS = 8


def class_series(cell_df, value_col, date_col):
    out = {}
    for cls, asset in STANDALONE_CLASSES.items():
        sub = cell_df[cell_df["asset"] == asset]
        if not sub.empty:
            out[cls] = sub.set_index(date_col)[value_col].sort_index()
    for cls, (a1, a2) in CORRELATED_PAIRS.items():
        s1 = cell_df[cell_df["asset"] == a1].set_index(date_col)[value_col]
        s2 = cell_df[cell_df["asset"] == a2].set_index(date_col)[value_col]
        j = pd.concat([s1, s2], axis=1, join="inner")
        if not j.empty:
            out[cls] = j.mean(axis=1).sort_index()
    return out


def pooled_diff_series(df_a, df_b, value_col, date_col):
    diffs = []
    for cls in ("index", "bond", "crypto"):
        sa = class_series(df_a, value_col, date_col).get(cls)
        sb = class_series(df_b, value_col, date_col).get(cls)
        if sa is None or sb is None:
            continue
        j = pd.concat([sa, sb], axis=1, join="inner")
        if j.empty:
            continue
        diffs.append((j.iloc[:, 0] - j.iloc[:, 1]).values)
    return np.concatenate(diffs) if diffs else np.array([])


def dual_test(diffs, h):
    n = len(diffs)
    if n < MIN_POINTS:
        return {"status": "insufficient_data", "n": int(n)}
    dm = dm_hac_test(diffs, h=h)
    bl = paired_block_bootstrap_test(diffs, block_length=min(BLOCK_LENGTH, n))
    return {"status": "tested", "n": int(n), "effective_n": bl["effective_n"],
            "mean_diff": dm["mean_diff"], "ci95_lo": bl["ci95_lo"], "ci95_hi": bl["ci95_hi"],
            "p_value_dm": dm["p_value"], "p_value_bootstrap": bl["p_value"],
            "significant_dm": bool(dm["p_value"] < 0.05),
            "significant_bootstrap": bl["significant_at_05"]}


def holm(pvals):
    pvals = np.asarray(pvals, float); m = len(pvals); order = np.argsort(pvals)
    adj = np.empty(m); rm = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]; rm = max(rm, a); adj[idx] = min(rm, 1.0)
    return adj.tolist()


def derive_scales(con):
    daily = pd.read_sql_query(
        "SELECT DISTINCT asset,cutoff_date,last_close FROM predictions WHERE frequence='daily'", con)
    scales = {}
    for a, g in daily.groupby("asset"):
        s = g.dropna().sort_values("cutoff_date")["last_close"].values
        scales[a] = float(np.mean(np.abs(np.diff(s)))) if len(s) > 5 else 1.0
    return scales


def main():
    db = str(ROOT / "validation" / "tracking.db")
    con = sqlite3.connect(db)
    scales = derive_scales(con)
    df = pd.read_sql_query(
        "SELECT model,asset,frequence,horizon_type,horizon_unit,cutoff_date,target_date,"
        "y_pred,y_lower,y_upper,y_true FROM predictions WHERE source='oos'", con)
    con.close()

    df["scale"] = df["asset"].map(scales)
    sig = (df["y_upper"] - df["y_lower"]) / (2 * 1.96)
    df["crps_norm"] = [crps_gaussian(mu, max(s, 1e-9), y) / sc
                       for mu, s, y, sc in zip(df.y_pred, sig, df.y_true, df.scale)]
    df["mase_error"] = np.abs(df.y_pred - df.y_true) / df.scale
    df["in_interval"] = ((df.y_true >= df.y_lower) & (df.y_true <= df.y_upper)).astype(float)

    weekly = df[df.horizon_type == "weekly"].copy()
    weekly["origin_key"] = weekly.cutoff_date.astype(str) + "|" + weekly.horizon_unit

    out = {"config": {"db": db, "asset_scales_from_db": scales, "n_weekly_rows": int(len(weekly)),
                      "block_length": BLOCK_LENGTH, "holm_across": list(MODELS)},
           "crps": {}, "coverage95": {}, "coverage_observed": {}}

    # test 1: CRPS ; test 2: coverage (diff C-B so >0 => weekly better)
    crps_p, cov_p = [], []
    for model in MODELS:
        m = weekly[weekly.model == model]
        b = m[m.frequence == "daily"]; c = m[m.frequence == "weekly"]
        tc = dual_test(pooled_diff_series(b, c, "crps_norm", "origin_key"), 3)   # B - C
        out["crps"][model] = tc
        crps_p.append(tc["p_value_bootstrap"] if tc.get("status") == "tested" else 1.0)
        tcov = dual_test(pooled_diff_series(c, b, "in_interval", "origin_key"), 3)  # C - B
        out["coverage95"][model] = tcov
        cov_p.append(tcov["p_value_bootstrap"] if tcov.get("status") == "tested" else 1.0)
        out["coverage_observed"][model] = {
            "B_daily_to_weekly": float(b["in_interval"].mean()),
            "C_weekly_native": float(c["in_interval"].mean())}
    for model, ph in zip(MODELS, holm(crps_p)):
        if out["crps"][model].get("status") == "tested": out["crps"][model]["p_holm"] = ph
    for model, ph in zip(MODELS, holm(cov_p)):
        if out["coverage95"][model].get("status") == "tested": out["coverage95"][model]["p_holm"] = ph

    outpath = ROOT / "experiments" / "weekly_vs_daily_pooled.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"saved -> {outpath}")
    for model in MODELS:
        tc = out["crps"][model]; tcov = out["coverage95"][model]
        co = out["coverage_observed"][model]
        print(f"\n{model}")
        if tc.get("status") == "tested":
            print(f"  CRPS  diff(B-C)={tc['mean_diff']:+.4f} IC[{tc['ci95_lo']:+.4f},{tc['ci95_hi']:+.4f}]"
                  f" p_boot={tc['p_value_bootstrap']:.4f} p_Holm={tc.get('p_holm',1):.4f}")
        print(f"  Cov95 B={co['B_daily_to_weekly']:.3f} C={co['C_weekly_native']:.3f}"
              f"  diff(C-B)={tcov['mean_diff']:+.4f} p_Holm={tcov.get('p_holm',1):.4f}"
              f" {'SIGNIF' if tcov.get('significant_bootstrap') else 'ns'}")


if __name__ == "__main__":
    main()
