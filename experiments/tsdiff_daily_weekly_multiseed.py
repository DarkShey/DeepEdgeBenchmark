"""
tsdiff_daily_weekly_multiseed.py -- multi-seed robustness table for TSDiff's
daily(B)-vs-weekly(C) verdict + badge source for the dashboard.

MIRROR of `nsdiff_daily_weekly_multiseed.py` (declared, not silently
duplicated -- see BRIEF_dashboard_multiseed_200.md §4). Never writes to
`tracking.db` -- produces only the isolated JSON artifact
(`tsdiff_daily_weekly_multiseed.json`), the badge's source (dashboard §5).
The `oos` ENSEMBLE line (one per origin, concatenation of the 5 clouds) is
written by a separate script, `oos_ensemble_tsdiff_daily_weekly.py`, which
reuses (does not refit) the checkpoints this script produces.

IMPORTANT -- written but NOT executed on this machine (compute lourd chez le
tuteur). See RUNBOOK_regeneration_multiseed_200.md.

Relance TSDiff-W (regime C) + TSDiff-D (regime B) sur les graines 42-46, mêmes
origines (lues verbatim, comme oos_tsdiff_daily_weekly.py), n_samples=200
(ensemble 5x200, tâche 6), avec collect_samples=True (nuages bruts conservés,
requis par oos_ensemble_tsdiff_daily_weekly.py). Checkpoints reprenables sous
`experiments/checkpoints_tsdiff_multiseed_200/` (un fichier par (seed, asset)).

Réutilise tel quel : `oos_tsdiff_daily_weekly.generate_tsdiff_asset` (fit +
forecast), `oos_tsdiff_daily_weekly.load_baseline_triplets_daily` (regime B
origins), `backtest_rolling_tsdiffw.load_baseline_triplets` (regime C
origins), `oos_nsdiff_daily_weekly.fetch_verified` (prix, 5 actifs),
`weekly_headtohead_v2.load_selected_epochs` (budgets par actif),
`matrice_paired_tests.comparison_3_daily_vs_weekly` (test par cellule, seed
interne 0 -- comme le dashboard), `dashboard_d7_w1.winkler_score`. Aucune
fonction de test réimplémentée.
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
from oos_nsdiff_daily_weekly import fetch_verified, load_baseline_triplets_daily  # noqa: E402
from oos_nsdiff_tlt import fetch_tlt_patched                            # noqa: E402
from weekly_headtohead import build_weekly                              # noqa: E402
from weekly_headtohead_v2 import load_selected_epochs                   # noqa: E402
from oos_tsdiff_daily_weekly import generate_tsdiff_asset, DEFAULT_K_DENOISE, DEFAULT_SWEEP_FILE  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())  # BTC-USD, ETH-USD, SPY, ZN=F, TLT
N_SAMPLES = 200   # ensemble 5x200 (tâche 6, BRIEF_dashboard_multiseed_200.md)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_tsdiff_multiseed_200"
OUT_PATH = Path(__file__).resolve().parent / "tsdiff_daily_weekly_multiseed.json"


def checkpoint_path(seed: int, asset: str) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"


def load_price_data():
    """Origins + verified price series + per-asset epochs*, per asset --
    computed once (seed-independent). Mirror of the NsDiff version, plus the
    selected epochs (TSDiff-W/TSDiff-D, swept independently per asset)."""
    selected = load_selected_epochs(DEFAULT_SWEEP_FILE)
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
        epochs_w = selected[f"{asset}|TSDiff-W"]["epochs"]
        epochs_d = selected[f"{asset}|TSDiff-D"]["epochs"]
        prices[asset] = (daily, weekly, weekly_dates, origins_c, origins_b, epochs_w, epochs_d)
        print(f"[{asset}] price data ready ({len(origins_c)} regime-C / {len(origins_b)} regime-B rows, "
             f"epochs W={epochs_w}/D={epochs_d}).")
    return prices


def run_seed(seed: int, prices: dict) -> list:
    all_rows = []
    for asset, (daily, weekly, weekly_dates, origins_c, origins_b, epochs_w, epochs_d) in prices.items():
        path = checkpoint_path(seed, asset)
        if path.exists():
            rows = json.loads(path.read_text())
            print(f"[seed={seed}][{asset}] loaded from checkpoint")
        else:
            t0 = time.time()
            rows_c, rows_b = generate_tsdiff_asset(
                asset, daily, weekly, weekly_dates, origins_c, origins_b,
                epochs_w, epochs_d, seed, N_SAMPLES, DEFAULT_K_DENOISE,
                collect_samples=True,
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
    print("=== loading verified price data + selected epochs (once, seed-independent) ===")
    prices = load_price_data()

    per_seed = {}
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        rows = run_seed(seed, prices)
        df = build_df(rows)
        per_seed[seed] = per_seed_summary(seed, df)

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
        "config": {"n_samples": N_SAMPLES, "k_denoise": DEFAULT_K_DENOISE,
                  "horizon_scope": "W+1 only (scope de la note, cf. dashboard_d7_w1)",
                  "epochs_source": str(DEFAULT_SWEEP_FILE)},
        "note": "Table isolee (jamais ecrite dans tracking.db oos). CV = std inter-graines / mean "
                "inter-graines, calcule sur RMSE et Winkler.",
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
