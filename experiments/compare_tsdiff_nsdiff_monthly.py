"""
compare_tsdiff_nsdiff_monthly.py -- TSDiff vs NsDiff AT MONTHLY HORIZON,
same regime (mirrors the spirit of `NOTE_compare_weekly_tsdiff_nsdiff.md`,
adapted to what's actually available). Extends
`analyze_nsdiff_monthly.py`/`NOTE_compare_daily_vs_monthly_nsdiff.md`
(daily-vs-monthly, single model) with the orthogonal question:
model-vs-model, at a fixed regime.

Reuses `build_enriched_pairs_monthly`/`build_cell_table_monthly` from
`analyze_nsdiff_monthly.py` UNCHANGED (both are already model-generic --
verified: neither hard-codes 'NsDiff' anywhere, they group by whatever
`model` values are present in the DataFrame -- so loading BOTH models'
monthly rows through them, with no code change, yields per-model cell
tables for free).

NOT reproduced here: the weekly note's CRPS/PIT-based comparison
(`NOTE_compare_weekly_tsdiff_nsdiff.md` §1) needs the full sample cloud per
forecast (`keep_samples=True` runs) -- neither `oos_nsdiff_monthly.py` nor
`oos_tsdiff_monthly.py` stores that (only point + 2.5/97.5 quantiles, same
documented limitation as `nsdiff_daily_weekly_multiseed.py`'s own note on
this). This script's model-vs-model comparison is therefore RMSE/Cov95/
Winkler/direction-based (paired block-bootstrap on squared-error and on
Winkler difference), not CRPS/PIT -- a real scope reduction from the weekly
note, stated plainly, not glossed over.
"""

import json
import sys
from pathlib import Path

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
from paired_test import paired_block_bootstrap_test          # noqa: E402
from analyze_nsdiff_monthly import (                          # noqa: E402
    build_enriched_pairs_monthly, build_cell_table_monthly, MONTHLY_HORIZON_UNITS,
)

DB_PATH = ROOT / "validation" / "tracking.db"
SEED_POOLED = 42
NSDIFF_MULTISEED_JSON = EXPERIMENTS_DIR / "nsdiff_monthly_multiseed.json"
TSDIFF_MULTISEED_JSON = EXPERIMENTS_DIR / "tsdiff_monthly_multiseed.json"
OUT_PATH = EXPERIMENTS_DIR / "tsdiff_vs_nsdiff_monthly_extract.json"

REGIME_LABEL = {"daily": "B (daily -> fin de mois)", "monthly": "C (mensuel natif)"}


def load_all_monthly_predictions(db_path: str) -> pd.DataFrame:
    """Same SQL as `analyze_nsdiff_monthly.load_monthly_predictions` but NO
    model filter -- loads NsDiff AND TSDiff monthly rows together."""
    df = mpt.load_predictions(str(db_path))
    return df[df["horizon_type"] == "monthly"].reset_index(drop=True)


def build_model_vs_model_pairs(df: pd.DataFrame, frequence: str, model_a: str, model_b: str,
                               horizon_units=None) -> pd.DataFrame:
    """Pair `model_a` vs `model_b` WITHIN the same regime (`frequence`), by
    (asset, cutoff_date, target_date, horizon_unit) -- the model-vs-model
    analogue of `build_monthly_pairs.build_daily_monthly_pairs` (which pairs
    frequence-vs-frequence within ONE model instead)."""
    sub = df[df["frequence"] == frequence]
    if horizon_units is not None:
        sub = sub[sub["horizon_unit"].isin(horizon_units)]
    a = sub[sub["model"] == model_a]
    b = sub[sub["model"] == model_b]

    frames = []
    for (asset, h), g_a in a.groupby(["asset", "horizon_unit"]):
        g_b = b[(b["asset"] == asset) & (b["horizon_unit"] == h)]
        merged = g_a.merge(g_b, on=["cutoff_date", "target_date"], suffixes=(f"_{model_a}", f"_{model_b}"))
        merged.insert(0, "asset", asset)
        merged.insert(1, "horizon_unit", h)
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["asset", "horizon_unit", "target_date"])
    pairs = pd.concat(frames, ignore_index=True)

    pairs[f"winkler_{model_a}"] = dash.winkler_score(pairs[f"y_true_{model_a}"], pairs[f"y_lower_{model_a}"], pairs[f"y_upper_{model_a}"])
    pairs[f"winkler_{model_b}"] = dash.winkler_score(pairs[f"y_true_{model_b}"], pairs[f"y_lower_{model_b}"], pairs[f"y_upper_{model_b}"])
    return pairs


def compare_models_cell(pairs: pd.DataFrame, model_a: str, model_b: str) -> list:
    """Per (asset, horizon_unit): paired block-bootstrap test on squared-error
    difference (model_a - model_b), mirror of
    `build_monthly_pairs.comparison_3_daily_vs_monthly`'s test logic, applied
    to a model-vs-model pairing instead of a regime-vs-regime one."""
    results = []
    for (asset, h), g in pairs.groupby(["asset", "horizon_unit"]):
        n = len(g)
        if n < mpt.MIN_PAIRED_POINTS:
            results.append({"asset": asset, "horizon_unit": h, "status": "insufficient_data", "n": int(n)})
            continue
        diffs = (g[f"sq_error_{model_a}"] - g[f"sq_error_{model_b}"]).values   # >0 => model_b lower sq error
        test = paired_block_bootstrap_test(diffs, block_length=min(mpt.BLOCK_LENGTH, n))
        if test["significant_at_05"]:
            verdict = f"{model_b}_significantly_better" if test["mean_diff"] > 0 else f"{model_a}_significantly_better"
        else:
            verdict = "indistinguishable"
        rmse_a = float((g[f"sq_error_{model_a}"].mean()) ** 0.5)
        rmse_b = float((g[f"sq_error_{model_b}"].mean()) ** 0.5)
        results.append({
            "asset": asset, "horizon_unit": h, "status": "tested", "verdict": verdict,
            "n": int(n), f"rmse_{model_a}": rmse_a, f"rmse_{model_b}": rmse_b,
            f"winkler_{model_a}": float(g[f"winkler_{model_a}"].mean()),
            f"winkler_{model_b}": float(g[f"winkler_{model_b}"].mean()),
            f"cov95_{model_a}": float(g[f"in_interval_{model_a}"].mean()),
            f"cov95_{model_b}": float(g[f"in_interval_{model_b}"].mean()),
            **test,
        })
    results.sort(key=lambda r: (r["horizon_unit"], r["asset"]))
    return results


def main():
    df = load_all_monthly_predictions(str(DB_PATH))
    print(f"Loaded {len(df)} monthly oos rows, models={sorted(df['model'].unique())} "
          f"({df.groupby(['model', 'frequence']).size().to_dict()}).")
    if df.empty or set(df["model"].unique()) != {"NsDiff", "TSDiff"}:
        raise SystemExit("Attendu exactement NsDiff + TSDiff en base (horizon_type='monthly') -- "
                         "lancer oos_nsdiff_monthly.py ET oos_tsdiff_monthly.py d'abord.")

    assets = sorted(df["asset"].unique())
    max_target = df["target_date"].max()
    price_cache = dash.load_price_history_cache(assets, max_target, refresh=False)

    # --- cell table (RMSE testé, regime-vs-regime, PAR MODELE -- fonctions génériques réutilisées telles quelles) ---
    pairs_all = build_enriched_pairs_monthly(df, price_cache, horizon_units=list(MONTHLY_HORIZON_UNITS))
    cell_table = build_cell_table_monthly(df, pairs_all)   # contains BOTH models' cells, unmodified function
    cell_table_by_model = {"NsDiff": [r for r in cell_table if r["model"] == "NsDiff"],
                           "TSDiff": [r for r in cell_table if r["model"] == "TSDiff"]}

    # --- agrégat skill-score poolé (M+1), PAR MODELE (filtrer AVANT le pooling -- cf. docstring) ---
    pairs_m1 = pairs_all[pairs_all["horizon_unit"] == "M+1"].copy()
    aggregate_skill_m1 = {}
    for model in ("NsDiff", "TSDiff"):
        aggregate_skill_m1[model] = dash.build_aggregate(pairs_m1[pairs_m1["model"] == model].copy(), SEED_POOLED)

    # --- NOUVEAU : TSDiff vs NsDiff, MEME regime, par cellule ---
    model_vs_model = {}
    for regime in ("daily", "monthly"):
        pairs_mm = build_model_vs_model_pairs(df, regime, "TSDiff", "NsDiff", horizon_units=list(MONTHLY_HORIZON_UNITS))
        cells = compare_models_cell(pairs_mm, "TSDiff", "NsDiff")
        model_vs_model[regime] = cells
        print(f"\n--- TSDiff vs NsDiff, régime {REGIME_LABEL[regime]} ---")
        for r in cells:
            if r["status"] != "tested":
                print(f"  {r['horizon_unit']} {r['asset']:8s}: insuffisant (n={r['n']})")
                continue
            print(f"  {r['horizon_unit']} {r['asset']:8s}: verdict={r['verdict']!s:35s} p={r['p_value']} "
                  f"n={r['n']} eff_n={r['effective_n']} | RMSE TSDiff {r['rmse_TSDiff']:.4g} vs "
                  f"NsDiff {r['rmse_NsDiff']:.4g}")

    # --- multi-graines, les DEUX artefacts ---
    multiseed = {}
    for model, path in (("NsDiff", NSDIFF_MULTISEED_JSON), ("TSDiff", TSDIFF_MULTISEED_JSON)):
        if path.exists():
            multiseed[model] = json.loads(path.read_text())
            print(f"\nMulti-graines {model} chargé depuis {path.name} (seeds={multiseed[model]['seeds']}).")
        else:
            multiseed[model] = None
            print(f"\nATTENTION: {path.name} introuvable pour {model}.", file=sys.stderr)

    out = {
        "note": "Extraction pour la comparaison TSDiff-vs-NsDiff mensuelle -- aucun modele relance, "
                "aucun test reimplemente. PAS de CRPS/PIT ici (pas de nuage d'echantillons stocke en DB) -- "
                "RMSE/Cov95/Winkler/direction uniquement, cf. docstring du script.",
        "seed_pooled": SEED_POOLED,
        "block_length": mpt.BLOCK_LENGTH,
        "min_paired_points": mpt.MIN_PAIRED_POINTS,
        "n_rows_loaded": int(len(df)),
        "cell_table_by_model": cell_table_by_model,
        "aggregate_skill_m1_by_model": aggregate_skill_m1,
        "model_vs_model_by_regime": model_vs_model,
        "multiseed": multiseed,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
