"""
analyze_nsdiff_monthly.py -- extraction for NOTE_compare_daily_vs_monthly_nsdiff.md
(BRIEF_nsdiff_mensuel_M1M2M3.md §7 deliverable 3). Mirrors
`extract_nsdiff_daily_vs_weekly.py`'s role, adapted because there is no
`dashboard_d7_w1_data.json` for the monthly side (brief §5: the dashboard is
weekly-only, the monthly side is a standalone note, not a dashboard entry).

Relaunches NO model and reimplements NO statistical test:
  - `paired_block_bootstrap_test` (via `build_monthly_pairs.
    comparison_3_daily_vs_monthly`) for the per-cell RMSE test.
  - `dashboard_d7_w1.winkler_score` / `direction_correct` for the descriptive
    KPIs (Winkler, direction).
  - `dashboard_d7_w1.historical_h_day_returns` / `rw_pi_bounds` for the RW
    baseline (frequency-agnostic, parameterised in CALENDAR days -- works
    unmodified for monthly horizons, brief §2).
  - `dashboard_d7_w1.build_pooled_series` / `run_pooled_test` /
    `build_aggregate` for the pooled skill-score by class -- these three are
    ALREADY fully generic (operate on a `pairs` DataFrame with columns
    [model, asset, asset_class, cutoff_date, skill_diff_sqerror,
    skill_diff_winkler] -- nothing weekly-specific inside them), so they are
    called AS-IS on a monthly-built `pairs` frame, not mirrored.

Only the pairing glue that IS weekly-shaped (`mpt.build_daily_weekly_pairs`,
`dashboard_d7_w1.build_enriched_pairs`, both hard-code horizon_type=='weekly'
and a single fixed HORIZON_UNIT) is mirrored below as
`build_enriched_pairs_monthly` (on top of `build_monthly_pairs.
build_daily_monthly_pairs`, itself mirroring `mpt.build_daily_weekly_pairs`
-- see that module's own docstring).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
for _p in (EXPERIMENTS_DIR, MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matrice_paired_tests as mpt          # noqa: E402
import build_monthly_pairs as bmp           # noqa: E402
import dashboard_d7_w1 as dash              # noqa: E402

DB_PATH = ROOT / "validation" / "tracking.db"
SEED_POOLED = 42
MONTHLY_HORIZON_UNITS = ("M+1", "M+2", "M+3")
MULTISEED_JSON = EXPERIMENTS_DIR / "nsdiff_monthly_multiseed.json"
OUT_PATH = EXPERIMENTS_DIR / "nsdiff_daily_vs_monthly_extract.json"


def load_monthly_predictions(db_path: str) -> pd.DataFrame:
    """`mpt.load_predictions` reused as-is (source='oos', no horizon_type
    filter in the SQL -- filtered here in pandas) then restricted to
    horizon_type='monthly' AND model='NsDiff' (brief scope: NsDiff only,
    §4)."""
    df = mpt.load_predictions(str(db_path))
    df = df[(df["horizon_type"] == "monthly") & (df["model"] == "NsDiff")].reset_index(drop=True)
    return df


def build_enriched_pairs_monthly(df: pd.DataFrame, price_cache: dict,
                                 horizon_units=None) -> pd.DataFrame:
    """Monthly mirror of `dashboard_d7_w1.build_enriched_pairs`. Generalised
    (vs. the weekly original, which only ever handles ONE fixed horizon_unit)
    to key the per-origin RW computation on (asset, cutoff_date, h) instead
    of just (asset, cutoff_date): M+1/M+2/M+3 share a cutoff_date but target
    different h (calendar-day distances), so collapsing on (asset,
    cutoff_date) alone would silently keep only one horizon's RW bounds for
    all three when horizon_units spans more than one value."""
    pairs = bmp.build_daily_monthly_pairs(df, horizon_units=horizon_units)
    if pairs.empty:
        raise SystemExit("Aucune paire daily/monthly trouvee -- la DB a-t-elle des lignes "
                         "oos horizon_type='monthly' ?")

    assert (pairs["cutoff_date_daily"] == pairs["cutoff_date_monthly"]).all(), \
        "cutoff_date devrait etre identique entre regime B et regime C (meme cible, brief §1)"
    pairs["cutoff_date"] = pairs["cutoff_date_daily"]
    pairs["h"] = (pd.to_datetime(pairs["target_date"]) - pd.to_datetime(pairs["cutoff_date"])).dt.days

    uniq = pairs.drop_duplicates(subset=["asset", "cutoff_date", "h"])[
        ["asset", "cutoff_date", "h", "last_close_daily", "y_true_daily"]
    ].copy()

    returns_cache: dict = {}
    rw_rows = []
    for _, r in uniq.iterrows():
        asset = r["asset"]
        price = price_cache[asset]
        rw_lo, rw_hi = dash.rw_pi_bounds(returns_cache, asset, price, r["h"], r["cutoff_date"], r["last_close_daily"])
        rw_rows.append({
            "asset": asset, "cutoff_date": r["cutoff_date"], "h": r["h"],
            "rw_point": r["last_close_daily"], "rw_lower": rw_lo, "rw_upper": rw_hi,
        })
    rw_df = pd.DataFrame(rw_rows)
    pairs = pairs.merge(rw_df, on=["asset", "cutoff_date", "h"], how="left")

    pairs["winkler_daily"] = dash.winkler_score(pairs["y_true_daily"], pairs["y_lower_daily"], pairs["y_upper_daily"])
    pairs["winkler_monthly"] = dash.winkler_score(pairs["y_true_monthly"], pairs["y_lower_monthly"], pairs["y_upper_monthly"])
    pairs["rw_sqerror"] = (pairs["rw_point"] - pairs["y_true_daily"]) ** 2
    pairs["rw_winkler"] = dash.winkler_score(pairs["y_true_daily"], pairs["rw_lower"], pairs["rw_upper"])

    pairs["pi_width_daily"] = pairs["y_upper_daily"] - pairs["y_lower_daily"]
    pairs["pi_width_monthly"] = pairs["y_upper_monthly"] - pairs["y_lower_monthly"]
    pairs["direction_daily"] = dash.direction_correct(pairs["y_true_daily"], pairs["y_pred_daily"], pairs["last_close_daily"])
    pairs["direction_monthly"] = dash.direction_correct(pairs["y_true_monthly"], pairs["y_pred_monthly"], pairs["last_close_monthly"])

    # scale RW = mediane par (asset, horizon_unit) -- pas juste asset : a
    # effective_n~10, la mediane sur M+1 seul (peu de points) est deja
    # fragile ; la calculer par (asset,h) evite en plus de melanger l'echelle
    # RW de M+1 (30j) avec celle de M+3 (90j) si jamais horizon_units>1.
    scale_sqerror = pairs.groupby(["asset", "horizon_unit"])["rw_sqerror"].transform("median")
    scale_winkler = pairs.groupby(["asset", "horizon_unit"])["rw_winkler"].transform("median")
    pairs["rw_scale_sqerror"] = scale_sqerror
    pairs["rw_scale_winkler"] = scale_winkler

    pairs["skill_sqerror_daily"] = 1.0 - pairs["sq_error_daily"] / pairs["rw_scale_sqerror"]
    pairs["skill_sqerror_monthly"] = 1.0 - pairs["sq_error_monthly"] / pairs["rw_scale_sqerror"]
    pairs["skill_winkler_daily"] = 1.0 - pairs["winkler_daily"] / pairs["rw_scale_winkler"]
    pairs["skill_winkler_monthly"] = 1.0 - pairs["winkler_monthly"] / pairs["rw_scale_winkler"]
    pairs["skill_diff_sqerror"] = pairs["skill_sqerror_daily"] - pairs["skill_sqerror_monthly"]
    pairs["skill_diff_winkler"] = pairs["skill_winkler_daily"] - pairs["skill_winkler_monthly"]

    pairs["asset_class"] = pairs["asset"].map(mpt.ASSET_CLASS)
    return pairs


def build_cell_table_monthly(df: pd.DataFrame, pairs: pd.DataFrame) -> list:
    """Mirror of `dashboard_d7_w1.build_cell_table`, generalised over the 3
    monthly horizon_units (the weekly original is single-horizon by
    construction)."""
    all_tests = bmp.comparison_3_daily_vs_monthly(df)
    cell_tests = {(r["model"], r["asset"], r["horizon_unit"]): r for r in all_tests}
    rows = []
    for (model, asset, h), g in pairs.groupby(["model", "asset", "horizon_unit"]):
        test = cell_tests.get((model, asset, h), {"status": "insufficient_data", "n": int(len(g))})
        rows.append({
            "model": model, "asset": asset, "horizon_unit": h,
            "asset_class": mpt.ASSET_CLASS.get(asset, "?"),
            "n": int(len(g)),
            "rmse_daily": float(np.sqrt(g["sq_error_daily"].mean())),
            "rmse_monthly": float(np.sqrt(g["sq_error_monthly"].mean())),
            "winkler_daily": float(g["winkler_daily"].mean()),
            "winkler_monthly": float(g["winkler_monthly"].mean()),
            "cov95_daily": float(g["in_interval_daily"].mean()),
            "cov95_monthly": float(g["in_interval_monthly"].mean()),
            "pi_width_daily": float(g["pi_width_daily"].mean()),
            "pi_width_monthly": float(g["pi_width_monthly"].mean()),
            "direction_daily": float(g["direction_daily"].mean()),
            "direction_monthly": float(g["direction_monthly"].mean()),
            "status": test.get("status"), "verdict": test.get("verdict"),
            "p_value": test.get("p_value"), "mean_diff": test.get("mean_diff"),
            "ci95_lo": test.get("ci95_lo"), "ci95_hi": test.get("ci95_hi"),
            "effective_n": test.get("effective_n"), "block_length": test.get("block_length"),
        })
    rows.sort(key=lambda r: (r["horizon_unit"], r["asset"]))
    return rows


def main():
    df = load_monthly_predictions(str(DB_PATH))
    print(f"Loaded {len(df)} monthly NsDiff oos rows "
          f"({df['frequence'].value_counts().to_dict()}).")
    if df.empty:
        raise SystemExit("Aucune ligne monthly NsDiff en base -- lancer oos_nsdiff_monthly.py d'abord.")

    assets = sorted(df["asset"].unique())
    max_target = df["target_date"].max()
    price_cache = dash.load_price_history_cache(assets, max_target, refresh=False)

    # --- table par actif, M+1/M+2/M+3 (brief §7 livrable 3) ---
    pairs_all = build_enriched_pairs_monthly(df, price_cache, horizon_units=list(MONTHLY_HORIZON_UNITS))
    cell_table = build_cell_table_monthly(df, pairs_all)
    for r in cell_table:
        print(f"  {r['horizon_unit']} {r['asset']:8s}: verdict={r['verdict']!s:45s} p={r['p_value']} "
              f"n={r['n']} eff_n={r['effective_n']} | RMSE {r['rmse_daily']:.4g}->{r['rmse_monthly']:.4g}")

    # --- agrégat pooled skill-score par classe (scope M+1, mirror du scope
    # W+1 du dashboard weekly -- brief ne demande pas explicitement M+2/M+3
    # poolés, seulement la table cellule ci-dessus pour ceux-là) ---
    pairs_m1 = pairs_all[pairs_all["horizon_unit"] == "M+1"].copy()
    aggregate_m1 = dash.build_aggregate(pairs_m1, SEED_POOLED)
    for cls, label in [("global", "Global"), ("crypto", "Crypto"), ("index", "Actions"), ("bond", "Obligations")]:
        agg = aggregate_m1[cls]
        if agg["status"] != "tested":
            print(f"  M+1 skill {label}: insuffisant (n_origines={agg.get('n_origins')})")
            continue
        sq, wk = agg["skill_sqerror"], agg["skill_winkler"]
        print(f"  M+1 skill {label}: n_origines={agg['n_origins']} (eff_n~{sq['effective_n']}) | "
              f"skill RMSE: {sq['verdict']} (p={sq['p_value']:.4f}) | "
              f"skill Winkler: {wk['verdict']} (p={wk['p_value']:.4f})")

    # --- multi-graines (brief §3, "aucun verdict mensuel sans sa stabilite inter-graines") ---
    multiseed = None
    if MULTISEED_JSON.exists():
        multiseed = json.loads(MULTISEED_JSON.read_text())
        print(f"\nMulti-graines chargé depuis {MULTISEED_JSON.name} (seeds={multiseed['seeds']}).")
    else:
        print(f"\nATTENTION: {MULTISEED_JSON.name} introuvable -- lancer nsdiff_monthly_multiseed.py d'abord "
              "(non-négociable brief §3).", file=sys.stderr)

    out = {
        "note": "Extraction pour NOTE_compare_daily_vs_monthly_nsdiff.md -- aucun modele relance ici, "
                "aucun test reimplemente (paired_block_bootstrap_test/winkler_score/rw_pi_bounds/"
                "build_pooled_series/run_pooled_test tous reutilises tels quels).",
        "seed_pooled": SEED_POOLED,
        "block_length": mpt.BLOCK_LENGTH,
        "min_paired_points": mpt.MIN_PAIRED_POINTS,
        "n_rows_loaded": int(len(df)),
        "cell_table_m1_m2_m3": cell_table,
        "aggregate_skill_m1": aggregate_m1,
        "multiseed": multiseed,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
