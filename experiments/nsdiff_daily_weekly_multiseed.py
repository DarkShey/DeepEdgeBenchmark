"""
nsdiff_daily_weekly_multiseed.py -- multi-seed robustness table for NsDiff's
daily(B)-vs-weekly(C) verdict (BRIEF_nsdiff_ameliorer_limites.md Fix 2).

Cadrage (non-negociable) : la piste `oos`/dashboard reste SINGLE-SEED (seed
42, comparable aux 6 autres modèles) -- ce script n'écrit JAMAIS dans
`tracking.db`. Il relance NsDiff daily(B) + weekly(C) sur les graines 42-46,
mêmes origines (lues verbatim, comme oos_nsdiff_daily_weekly.py), mêmes
budgets (epochs=40, seq_len=30, k_denoise=20, n_samples=50), dans un artefact
JSON isolé (`nsdiff_daily_weekly_multiseed.json`) + un fichier par
(seed, asset) sous `experiments/checkpoints_nsdiff_multiseed/` (reprenable,
même convention que `duel_multiseed.py`).

Réutilise tel quel : `oos_nsdiff_daily_weekly.generate_nsdiff_asset` (fit +
forecast), `oos_nsdiff_daily_weekly.fetch_verified` (prix, 4 actifs),
`oos_nsdiff_tlt.fetch_tlt_patched` (prix TLT, cf. Fix 1),
`matrice_paired_tests.comparison_3_daily_vs_weekly` (test par cellule, seed
interne 0 -- comme le dashboard), `dashboard_d7_w1.winkler_score`. Aucune
fonction de test réimplémentée.

Prix et origines chargés UNE SEULE FOIS (ne dépendent pas de la graine) ;
seul le fit/forecast NsDiff est répété par graine.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matrice_paired_tests as mpt                                      # noqa: E402
import dashboard_d7_w1 as dash                                          # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS                   # noqa: E402
from backtest_rolling_tsdiffw import load_baseline_triplets             # noqa: E402
from oos_nsdiff_daily_weekly import (                                   # noqa: E402
    load_baseline_triplets_daily, fetch_verified, generate_nsdiff_asset,
    DEFAULT_N_SAMPLES, DEFAULT_K_DENOISE,
)
from oos_nsdiff_tlt import fetch_tlt_patched                            # noqa: E402
from weekly_headtohead import build_weekly                              # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W as NSDIFF_EPOCHS   # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())  # BTC-USD, ETH-USD, SPY, ZN=F, TLT
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_nsdiff_multiseed"
OUT_PATH = Path(__file__).resolve().parent / "nsdiff_daily_weekly_multiseed.json"


def checkpoint_path(seed: int, asset: str) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"


def load_price_data():
    """Origins + verified price series, per asset -- computed once (seed-independent)."""
    origins_c_all = load_baseline_triplets(ASSETS)
    origins_b_all = load_baseline_triplets_daily(ASSETS)
    prices = {}
    for asset in ASSETS:
        origins_c = origins_c_all[origins_c_all["asset"] == asset].reset_index(drop=True)
        origins_b = origins_b_all[origins_b_all["asset"] == asset].reset_index(drop=True)
        if asset == "TLT":
            from backtest_rolling_tsdiffw import FETCH_START, FETCH_END
            daily = fetch_tlt_patched(asset, FETCH_START, FETCH_END)
            weekly, weekly_dates = build_weekly(daily)
        else:
            fetched = fetch_verified(asset, origins_c, origins_b)
            if fetched is None:
                raise SystemExit(f"[{asset}] price verification failed -- cannot run multiseed.")
            daily, weekly, weekly_dates, _label = fetched
        prices[asset] = (daily, weekly, weekly_dates, origins_c, origins_b)
        print(f"[{asset}] price data ready ({len(origins_c)} regime-C / {len(origins_b)} regime-B rows).")
    return prices


def run_seed(seed: int, prices: dict) -> list:
    all_rows = []
    for asset, (daily, weekly, weekly_dates, origins_c, origins_b) in prices.items():
        path = checkpoint_path(seed, asset)
        if path.exists():
            rows = json.loads(path.read_text())
            print(f"[seed={seed}][{asset}] loaded from checkpoint")
        else:
            t0 = time.time()
            rows_c, rows_b = generate_nsdiff_asset(
                asset, daily, weekly, weekly_dates, origins_c, origins_b,
                NSDIFF_EPOCHS, seed, DEFAULT_N_SAMPLES, DEFAULT_K_DENOISE,
            )
            rows = rows_c + rows_b
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, default=str))
            print(f"[seed={seed}][{asset}] {len(rows)} rows in {time.time() - t0:.0f}s -> checkpoint saved")
        all_rows.extend(rows)
    return all_rows


def build_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return df


def per_seed_summary(seed: int, df: pd.DataFrame) -> dict:
    """Per (asset, horizon_unit=='W+1'): verdict/p_value (reuse
    comparison_3_daily_vs_weekly, seed interne 0, comme le dashboard) +
    Winkler descriptif daily/weekly (reuse dashboard_d7_w1.winkler_score)."""
    cell_tests = {(r["asset"], r["horizon_unit"]): r for r in mpt.comparison_3_daily_vs_weekly(df)
                  if r["horizon_unit"] == "W+1"}
    pairs = mpt.build_daily_weekly_pairs(df, horizon_units=["W+1"])
    out = {}
    for asset in ASSETS:
        g = pairs[pairs["asset"] == asset]
        if g.empty:
            continue
        wd = dash.winkler_score(g["y_true_daily"], g["y_lower_daily"], g["y_upper_daily"])
        ww = dash.winkler_score(g["y_true_weekly"], g["y_lower_weekly"], g["y_upper_weekly"])
        test = cell_tests.get((asset, "W+1"), {})
        out[asset] = {
            "seed": seed,
            "rmse_daily": float((g["sq_error_daily"].mean()) ** 0.5),
            "rmse_weekly": float((g["sq_error_weekly"].mean()) ** 0.5),
            "winkler_daily": float(wd.mean()), "winkler_weekly": float(ww.mean()),
            "verdict_rmse": test.get("verdict"), "p_value_rmse": test.get("p_value"),
            "n": test.get("n"), "effective_n": test.get("effective_n"),
        }
    return out


def main():
    t_start = time.time()
    print("=== loading verified price data (once, seed-independent) ===")
    prices = load_price_data()

    per_seed = {}
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        rows = run_seed(seed, prices)
        df = build_df(rows)
        per_seed[seed] = per_seed_summary(seed, df)

    # --- table verdict x graine + CV inter-graines (RMSE, Winkler) ---
    by_asset = {asset: [] for asset in ASSETS}
    for seed, summary in per_seed.items():
        for asset, row in summary.items():
            by_asset[asset].append(row)

    cv_table = {}
    for asset, rows in by_asset.items():
        if not rows:
            continue
        rmse_d = np.array([r["rmse_daily"] for r in rows])
        rmse_w = np.array([r["rmse_weekly"] for r in rows])
        wk_d = np.array([r["winkler_daily"] for r in rows])
        wk_w = np.array([r["winkler_weekly"] for r in rows])
        verdicts = [r["verdict_rmse"] for r in rows]
        cv_table[asset] = {
            "verdicts_by_seed": {r["seed"]: r["verdict_rmse"] for r in rows},
            "p_values_by_seed": {r["seed"]: r["p_value_rmse"] for r in rows},
            "verdict_stable": len(set(verdicts)) == 1,
            "n": rows[0]["n"], "effective_n": rows[0]["effective_n"],
            "cv_rmse_daily": float(rmse_d.std(ddof=1) / rmse_d.mean()),
            "cv_rmse_weekly": float(rmse_w.std(ddof=1) / rmse_w.mean()),
            "cv_winkler_daily": float(wk_d.std(ddof=1) / wk_d.mean()),
            "cv_winkler_weekly": float(wk_w.std(ddof=1) / wk_w.mean()),
            "rmse_daily_mean": float(rmse_d.mean()), "rmse_weekly_mean": float(rmse_w.mean()),
            "winkler_daily_mean": float(wk_d.mean()), "winkler_weekly_mean": float(wk_w.mean()),
        }

    payload = {
        "seeds": SEEDS, "assets": ASSETS,
        "config": {"epochs": NSDIFF_EPOCHS, "n_samples": DEFAULT_N_SAMPLES, "k_denoise": DEFAULT_K_DENOISE,
                  "horizon_scope": "W+1 only (scope de la note, cf. dashboard_d7_w1)"},
        "note": "Table isolee (jamais ecrite dans tracking.db oos). CV = std inter-graines / mean "
                "inter-graines, calcule sur RMSE et Winkler (pas de CRPS ici -- les rows generees ne "
                "stockent que point+quantiles 2.5/97.5, pas le nuage complet d'echantillons, cf. "
                "NOTE_compare_weekly_tsdiff_nsdiff.md Etape 0 pour la meme limite cote duel).",
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "cv_table": cv_table,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {OUT_PATH}  ({(time.time() - t_start) / 60:.1f} min)")
    for asset, row in cv_table.items():
        print(f"  {asset}: verdicts={row['verdicts_by_seed']} stable={row['verdict_stable']} "
              f"CV(Winkler daily/weekly)={row['cv_winkler_daily']:.3f}/{row['cv_winkler_weekly']:.3f}")


if __name__ == "__main__":
    main()
