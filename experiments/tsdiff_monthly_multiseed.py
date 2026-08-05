"""
tsdiff_monthly_multiseed.py -- multi-seed robustness table for the monthly
TSDiff daily(B)-vs-monthly(C) verdict, TSDiff mirror of
`nsdiff_monthly_multiseed.py` (same non-negotiable: brief §3, "aucun
verdict mensuel sans sa stabilité inter-graines" -- applies to TSDiff too,
not just NsDiff, once TSDiff is added to the monthly comparison).

Cadrage identique à `nsdiff_monthly_multiseed.py` : la piste `oos`/
tracking.db reste single-seed (`oos_tsdiff_monthly.py`, seed=42) -- ce
script n'écrit JAMAIS dans `tracking.db`. Graines 42-46, mêmes origines
(seed-indépendantes), mêmes budgets (epochs=40, seq_len_monthly=18,
k_denoise=20, ddim_eta=1.0, n_samples=50), artefact JSON isolé +
checkpoints par (seed, asset) sous
`experiments/checkpoints_tsdiff_monthly_multiseed/` (reprenable).

NOTE COÛT (à lire avant de relancer) : TSDiff est nettement plus lourd que
NsDiff sur ce run (UNet + T_DIFFUSION=1000 pas d'entraînement vs les petits
MLP de NsDiff) -- mesuré empiriquement : régime B (daily) seul ~176s (SPY)
à ~364s (BTC-USD) PAR GRAINE PAR ACTIF, contre <10s côté NsDiff. Budget
total mesuré pour les 5 graines x 5 actifs : de l'ordre de 2h-2h30 sur cette
machine. Les checkpoints par (seed, asset) rendent le run reprenable si
interrompu (RAM/disque, cf. contraintes machine documentées) -- relancer le
script reprend où il s'est arrêté, ne repart pas de zéro.
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

import dashboard_d7_w1 as dash                                          # noqa: E402
import build_monthly_pairs as bmp                                       # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS                   # noqa: E402
from oos_nsdiff_monthly import (                                        # noqa: E402
    build_monthly, three_way_split_monthly, FETCH_START,
    SEQ_LEN_MONTHLY, N_VAL_MONTHLY, N_TEST_MONTHLY,
)
from oos_tsdiff_monthly import (                                        # noqa: E402
    fetch_prices, generate_tsdiff_asset_monthly, DEFAULT_N_SAMPLES,
    DEFAULT_K_DENOISE, DEFAULT_DDIM_ETA, DEFAULT_EPOCHS,
)

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_tsdiff_monthly_multiseed"
OUT_PATH = Path(__file__).resolve().parent / "tsdiff_monthly_multiseed.json"


def checkpoint_path(seed: int, asset: str) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"


def load_price_data():
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = {}
    for asset in ASSETS:
        daily, price_source = fetch_prices(asset, FETCH_START, end)
        monthly, monthly_dates = build_monthly(daily)
        _, _, test_pos = three_way_split_monthly(monthly, N_VAL_MONTHLY, N_TEST_MONTHLY)
        data[asset] = (daily, monthly, monthly_dates, test_pos)
        print(f"[{asset}] price data ready ({price_source}, {len(test_pos)} test origins).")
    return data


def run_seed(seed: int, data: dict) -> list:
    all_rows = []
    for asset, (daily, monthly, monthly_dates, test_pos) in data.items():
        path = checkpoint_path(seed, asset)
        if path.exists():
            rows = json.loads(path.read_text())
            print(f"[seed={seed}][{asset}] loaded from checkpoint")
        else:
            t0 = time.time()
            rows_c, rows_b, _meta = generate_tsdiff_asset_monthly(
                asset, daily, monthly, monthly_dates, test_pos, DEFAULT_EPOCHS, seed,
                DEFAULT_N_SAMPLES, DEFAULT_K_DENOISE, DEFAULT_DDIM_ETA, SEQ_LEN_MONTHLY,
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
    cell_tests = {(r["asset"], r["horizon_unit"]): r for r in bmp.comparison_3_daily_vs_monthly(df)
                  if r["horizon_unit"] == "M+1"}
    pairs = bmp.build_daily_monthly_pairs(df, horizon_units=["M+1"])
    out = {}
    for asset in ASSETS:
        g = pairs[pairs["asset"] == asset]
        if g.empty:
            continue
        wd = dash.winkler_score(g["y_true_daily"], g["y_lower_daily"], g["y_upper_daily"])
        wm = dash.winkler_score(g["y_true_monthly"], g["y_lower_monthly"], g["y_upper_monthly"])
        test = cell_tests.get((asset, "M+1"), {})
        out[asset] = {
            "seed": seed,
            "rmse_daily": float((g["sq_error_daily"].mean()) ** 0.5),
            "rmse_monthly": float((g["sq_error_monthly"].mean()) ** 0.5),
            "winkler_daily": float(wd.mean()), "winkler_monthly": float(wm.mean()),
            "verdict_rmse": test.get("verdict"), "p_value_rmse": test.get("p_value"),
            "n": test.get("n"), "effective_n": test.get("effective_n"),
        }
    return out


def main():
    t_start = time.time()
    print("=== loading price data + test origins (once, seed-independent) ===")
    data = load_price_data()

    per_seed = {}
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        rows = run_seed(seed, data)
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
        rmse_m = np.array([r["rmse_monthly"] for r in rows])
        wk_d = np.array([r["winkler_daily"] for r in rows])
        wk_m = np.array([r["winkler_monthly"] for r in rows])
        verdicts = [r["verdict_rmse"] for r in rows]
        cv_table[asset] = {
            "verdicts_by_seed": {r["seed"]: r["verdict_rmse"] for r in rows},
            "p_values_by_seed": {r["seed"]: r["p_value_rmse"] for r in rows},
            "verdict_stable": len(set(verdicts)) == 1,
            "n": rows[0]["n"], "effective_n": rows[0]["effective_n"],
            "cv_rmse_daily": float(rmse_d.std(ddof=1) / rmse_d.mean()),
            "cv_rmse_monthly": float(rmse_m.std(ddof=1) / rmse_m.mean()),
            "cv_winkler_daily": float(wk_d.std(ddof=1) / wk_d.mean()),
            "cv_winkler_monthly": float(wk_m.std(ddof=1) / wk_m.mean()),
            "rmse_daily_mean": float(rmse_d.mean()), "rmse_monthly_mean": float(rmse_m.mean()),
            "winkler_daily_mean": float(wk_d.mean()), "winkler_monthly_mean": float(wk_m.mean()),
        }

    payload = {
        "seeds": SEEDS, "assets": ASSETS,
        "config": {"epochs": DEFAULT_EPOCHS, "n_samples": DEFAULT_N_SAMPLES, "k_denoise": DEFAULT_K_DENOISE,
                  "ddim_eta": DEFAULT_DDIM_ETA, "seq_len_monthly": SEQ_LEN_MONTHLY,
                  "n_test": N_TEST_MONTHLY, "n_val": N_VAL_MONTHLY,
                  "horizon_scope": "M+1 only (mirror du scope NsDiff monthly / W+1 weekly)"},
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
              f"CV(Winkler daily/monthly)={row['cv_winkler_daily']:.3f}/{row['cv_winkler_monthly']:.3f}")


if __name__ == "__main__":
    main()
