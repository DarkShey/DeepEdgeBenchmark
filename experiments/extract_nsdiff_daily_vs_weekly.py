"""
extract_nsdiff_daily_vs_weekly.py — extraction pour NOTE_compare_daily_vs_weekly_nsdiff.md.

Ne relance AUCUN modèle et ne réimplémente AUCUN test : réutilise tel quel
build_enriched_pairs / build_pooled_series / run_pooled_test de
experiments/dashboard_d7_w1.py (qui lui-même réutilise paired_test.py et
matrice_paired_tests.py), avec le même seed poolé (42) et le même cache prix
local déjà présent (experiments/.price_cache_d7_w1/) -- aucun accès réseau.

Objectif : le JSON du dashboard (`aggregate`) poole les 7 modèles ENSEMBLE par
origine avant le bootstrap (cf. build_pooled_series / run_pooled_test) -- ce
n'est PAS un agrégat par modèle. Il n'y a donc pas d'agrégat "NsDiff seul" par
classe/global déjà présent dans le JSON. Ce script relance le même pooling,
restreint aux lignes model=='NsDiff' des mêmes paires (mêmes origines, même
DB, même cache prix), pour produire cet agrégat manquant -- et, en prime,
recoupe les métriques par cellule NsDiff avec celles déjà publiées dans
dashboard_d7_w1_data.json (doivent matcher bit-à-bit : mêmes pairs en entrée).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
for _p in (EXPERIMENTS_DIR, MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matrice_paired_tests as mpt  # noqa: E402
import dashboard_d7_w1 as dash      # noqa: E402

DB_PATH = ROOT / "validation" / "tracking.db"
DASHBOARD_JSON = EXPERIMENTS_DIR / "dashboard_d7_w1_data.json"
SEED_POOLED = 42  # identique à seed_pooled du JSON dashboard


def main():
    with open(DASHBOARD_JSON) as f:
        dashboard = json.load(f)
    assert dashboard["seed_pooled"] == SEED_POOLED, "seed_pooled du JSON a changé -- ajuster SEED_POOLED"

    df = mpt.load_predictions(str(DB_PATH))
    assets = sorted(df["asset"].unique())
    max_target = df["target_date"].max()
    price_cache = dash.load_price_history_cache(assets, max_target, refresh=False)
    pairs = dash.build_enriched_pairs(df, price_cache)

    # --- recoupement cellule par cellule vs le JSON déjà publié ---
    ns_cells_json = {c["asset"]: c for c in dashboard["cells"] if c["model"] == "NsDiff"}
    ns_pairs = pairs[pairs["model"] == "NsDiff"].copy()
    mismatches = []
    for asset, g in ns_pairs.groupby("asset"):
        if asset not in ns_cells_json:
            continue
        rmse_daily = (g["sq_error_daily"].mean()) ** 0.5
        rmse_weekly = (g["sq_error_weekly"].mean()) ** 0.5
        ref = ns_cells_json[asset]
        if abs(rmse_daily - ref["rmse_daily"]) > 1e-6 or abs(rmse_weekly - ref["rmse_weekly"]) > 1e-6:
            mismatches.append(asset)
    if mismatches:
        print(f"ATTENTION recoupement RMSE cellule KO pour: {mismatches}", file=sys.stderr)
    else:
        print(f"Recoupement OK : RMSE daily/weekly recalculés == JSON dashboard pour les {len(ns_cells_json)} "
              "cellules NsDiff (mêmes pairs, mêmes origines).")

    # --- agrégat NsDiff seul (pas dans le JSON, qui poole les 7 modèles) ---
    pooled_ns = dash.build_pooled_series(ns_pairs)
    aggregate_nsdiff = {"global": dash.run_pooled_test(pooled_ns, None, SEED_POOLED)}
    for cls in ("crypto", "index", "bond"):
        aggregate_nsdiff[cls] = dash.run_pooled_test(pooled_ns, cls, SEED_POOLED)

    out = {
        "note": "aggregate NsDiff-only (pooling restreint a model=='NsDiff'), "
                "recompute via dashboard_d7_w1.build_pooled_series/run_pooled_test (reuse, pas de reimplementation).",
        "seed_pooled": SEED_POOLED,
        "block_length": dashboard["block_length"],
        "cells_nsdiff_from_dashboard_json": ns_cells_json,
        "aggregate_nsdiff": aggregate_nsdiff,
    }
    out_path = EXPERIMENTS_DIR / "nsdiff_daily_vs_weekly_extract.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"-> {out_path}")

    for cls, label in [("global", "Global"), ("crypto", "Crypto"), ("index", "Actions"), ("bond", "Obligations")]:
        agg = aggregate_nsdiff[cls]
        if agg["status"] != "tested":
            print(f"  {label}: insuffisant (n_origines={agg.get('n_origins')})")
            continue
        sq = agg["skill_sqerror"]
        wk = agg["skill_winkler"]
        print(f"  {label}: n_origines={agg['n_origins']} (effective_n~{sq['effective_n']}) | "
              f"skill RMSE: {sq['verdict']} (p={sq['p_value']:.4f}) | "
              f"skill Winkler: {wk['verdict']} (p={wk['p_value']:.4f})")

    # --- bonus §4 du brief : W+2/W+3, si les lignes oos existent (elles existent,
    # cf. NOTE_nsdiff_dashboard_daily_oos.md). Réutilise comparison_3_daily_vs_weekly
    # (test par cellule déjà utilisé par le dashboard, seed interne 0) tel quel,
    # + winkler_score (dashboard) pour les KPI descriptifs -- pas de RW/skill-score
    # ici (pas nécessaire pour une table par cellule brute).
    print("\n--- Bonus W+2/W+3 (NsDiff, si lignes oos disponibles) ---")
    cell_tests = [r for r in mpt.comparison_3_daily_vs_weekly(df)
                  if r["model"] == "NsDiff" and r["horizon_unit"] in ("W+2", "W+3")]
    pairs_w23 = mpt.build_daily_weekly_pairs(df, horizon_units=["W+2", "W+3"])
    pairs_w23 = pairs_w23[pairs_w23["model"] == "NsDiff"].copy()
    extended = []
    for r in sorted(cell_tests, key=lambda x: (x["horizon_unit"], x["asset"])):
        g = pairs_w23[(pairs_w23["asset"] == r["asset"]) & (pairs_w23["horizon_unit"] == r["horizon_unit"])]
        wd = dash.winkler_score(g["y_true_daily"], g["y_lower_daily"], g["y_upper_daily"])
        ww = dash.winkler_score(g["y_true_weekly"], g["y_lower_weekly"], g["y_upper_weekly"])
        row = {
            "horizon_unit": r["horizon_unit"], "asset": r["asset"], "status": r["status"],
            "verdict": r.get("verdict"), "p_value": r.get("p_value"), "n": r.get("n"),
            "effective_n": r.get("effective_n"),
            "rmse_daily": float((g["sq_error_daily"].mean()) ** 0.5),
            "rmse_weekly": float((g["sq_error_weekly"].mean()) ** 0.5),
            "cov95_daily": float(g["in_interval_daily"].mean()),
            "cov95_weekly": float(g["in_interval_weekly"].mean()),
            "pi_width_daily": float((g["y_upper_daily"] - g["y_lower_daily"]).mean()),
            "pi_width_weekly": float((g["y_upper_weekly"] - g["y_lower_weekly"]).mean()),
            "winkler_daily": float(wd.mean()), "winkler_weekly": float(ww.mean()),
        }
        extended.append(row)
        print(f"  {row['horizon_unit']} {row['asset']}: verdict={row['verdict']} p={row['p_value']} "
              f"n={row['n']} eff_n={row['effective_n']} | RMSE {row['rmse_daily']:.4g}->{row['rmse_weekly']:.4g} "
              f"| Cov95 {row['cov95_daily']:.3f}->{row['cov95_weekly']:.3f} "
              f"| Winkler {row['winkler_daily']:.4g}->{row['winkler_weekly']:.4g}")
    out["cells_nsdiff_w2_w3"] = extended
    out_path.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
