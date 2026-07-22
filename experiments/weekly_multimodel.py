"""
weekly_multimodel.py — extend ARIMA-GARCH / SARIMA / Prophet / LSTM / Naive to the
weekly regimes B (daily-trained, multi-step to weekly horizon) and C (weekly-native),
same protocol as TSDiff (BRIEF_audit_combinaisons.md step 3): same 5 assets, same 30
test origins / target dates (three_way_split + week_targets, reused unchanged from
epoch_sweep.py / weekly_headtohead.py), no lookahead.

Unlike TSDiff, these models are cheap to fit (no epoch-collapse pathology observed or
expected for ARIMA/SARIMA/Prophet/LSTM/Naive) -- no epoch/validation sweep here, "clean
selection" means: reuse each model's EXISTING, already-in-production configuration
(benchmarks/multi_horizon.py's fit_<model>/forecast_horizons_<model>), just fed
weekly-resampled data (regime C) or extended-horizon daily data (regime B). Refit at
EVERY walk-forward origin (no train-once-forward reuse across origins) -- these models
are fast enough that the TSDiff compute-saving trick isn't needed, and this matches
each model's own existing daily walk-forward convention (e.g. run_sarima refits every
step already).

Regime B: the existing daily forecast_horizons_<model> functions are used AS-IS, just
called with horizons=[5,10,15]-ish (daily-step distance to each weekly target) instead
of [1,7] -- zero model-code changes needed, this is exactly what TSDiff-D already did.

Regime C: SARIMA and Prophet have daily-specific assumptions baked into
multi_horizon.py that don't transfer to weekly-resampled data, so this file defines
weekly-specific variants for those two only (see forecast_horizons_sarima_weekly /
forecast_horizons_prophet_weekly below) -- documented design decisions, not arbitrary:
  - SARIMA: seasonal_order=(1,0,1,5) encodes a 5-TRADING-DAY cycle; meaningless on
    data already resampled to one point/week -- disabled (0,0,0,0) for regime C.
  - Prophet: weekly_seasonality=True (day-of-week effect) is meaningless with one
    point/week, and future dates must land on Fridays (W-FRI), not business days.
ARIMA-GARCH (frequency-agnostic log-return ARMA), Naive (frequency-agnostic sigma*sqrt(h)),
and LSTM (SEQ_LEN=30 lags, works the same whether lags are days or weeks) need no
changes for regime C -- their daily functions are reused unchanged.

Usage:
    python weekly_multimodel.py --models ARIMA-GARCH Naive              # checkpoint batch
    python weekly_multimodel.py --models ARIMA-GARCH Naive --dry-run-tiny  # fast smoke test
"""

import os

# Must run BEFORE any yfinance/statsmodels import (tsdiff_model, sarima_model below
# pull those in at module level): importing tensorflow for the first time AFTER
# yfinance/statsmodels have already been imported in this process triggers a
# confirmed deadlock (stack frozen in TFE_Execute, 0% CPU, never returns) -- same
# root cause documented and worked around in models/conftest.py. LSTM is the only
# model here that needs tensorflow, but the import order poisons the whole process
# regardless of which model runs first, so this guard is unconditional.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
    _tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import tsdiff_model as td                                        # noqa: E402 (fetch_data reused)
import multi_horizon as mh                                       # noqa: E402
import sarima_model                                               # noqa: E402

from weekly_headtohead import ASSETS, HORIZON_LABELS, build_weekly  # noqa: E402
from epoch_sweep import three_way_split, week_targets, DEFAULT_N_VAL, DEFAULT_N_TEST  # noqa: E402

MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive")
LSTM_SEED = 42

# LSTM régime C (BRIEF_lstm_weekly_retune.md) : SEQ_LEN choisi par actif sur le bloc de
# validation (experiments/lstm_weekly_sweep.py, sélection CRPS + règle 1-SE), JAMAIS sur
# le test régénéré ci-dessous. Régime B garde le défaut daily (lstm_model.SEQ_LEN=30,
# non retouché) -- asymétrie assumée, cf. METHODOLOGIE_weekly_vs_daily.md.
LSTM_WEEKLY_SWEEP_FILE = Path(__file__).resolve().parent / "lstm_weekly_sweep.json"


def load_lstm_weekly_seq_len(path=LSTM_WEEKLY_SWEEP_FILE) -> dict:
    """{asset_short: SEQ_LEN*} depuis experiments/lstm_weekly_sweep.py. Lève une erreur
    claire si le sweep n'a pas été lancé -- pas de valeur par défaut silencieuse qui
    court-circuiterait la sélection sur validation (garde-fou du brief)."""
    if not Path(path).exists():
        raise SystemExit(f"{path} introuvable -- lance experiments/lstm_weekly_sweep.py d'abord.")
    return json.loads(Path(path).read_text())["selected_seq_len"]


def forecast_horizons_sarima_weekly(train: pd.Series, horizons: list) -> dict:
    """SARIMA on weekly-resampled data -- seasonal_order disabled (see module
    docstring): reduces to plain ARIMA(1,1,1) on weekly closes."""
    history = train.astype(float).values.tolist()
    result = sarima_model.SARIMAX(
        history, order=sarima_model.ORDER, seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return mh.forecast_from_fitted_sarima(result, horizons)


def forecast_horizons_prophet_weekly(train: pd.Series, horizons: list) -> dict:
    """Prophet on weekly-resampled data -- weekly_seasonality disabled, future dates
    on W-FRI instead of business days (see module docstring)."""
    import prophet_model
    df_train = pd.DataFrame({"ds": pd.to_datetime(train.index),
                             "y": train.astype(float).values.flatten()})
    model = prophet_model.Prophet(
        interval_width=1 - prophet_model.PI_ALPHA,
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
    )
    model.fit(df_train)
    max_h = max(horizons)
    last_date = pd.to_datetime(train.index[-1])
    future_dates = pd.date_range(start=last_date, periods=max_h + 1, freq="W-FRI")[1:]
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))
    results = {}
    for h in horizons:
        row = forecast.iloc[h - 1]
        results[h] = (float(row["yhat"]), float(row["yhat_lower"]), float(row["yhat_upper"]))
    return results


REGIME_B_FORECAST = {
    "ARIMA-GARCH": mh.forecast_horizons_arima,
    "SARIMA": mh.forecast_horizons_sarima,
    "Prophet": mh.forecast_horizons_prophet,
    "LSTM": functools.partial(mh.forecast_horizons_lstm, epochs=None, seed=LSTM_SEED),
    "Naive": mh.forecast_horizons_naive,
}
REGIME_C_FORECAST = {
    "ARIMA-GARCH": mh.forecast_horizons_arima,
    "SARIMA": forecast_horizons_sarima_weekly,
    "Prophet": forecast_horizons_prophet_weekly,
    "LSTM": functools.partial(mh.forecast_horizons_lstm, epochs=None, seed=LSTM_SEED),
    "Naive": mh.forecast_horizons_naive,
}


def run_model_asset(model_name: str, asset_short: str, ticker: str, regime: str,
                    n_val: int, n_test: int, start: str, end: str,
                    lstm_weekly_seq_len: dict = None) -> dict:
    """regime: 'B' (daily-trained, multi-step) or 'C' (weekly-native).
    `lstm_weekly_seq_len` : {asset_short: SEQ_LEN*}, requis seulement pour
    model_name="LSTM" + regime="C" (cf. BRIEF_lstm_weekly_retune.md) -- construit le
    partial forecast_horizons_lstm avec le SEQ_LEN retenu POUR CET ACTIF, au lieu du
    défaut daily partagé par tous les autres modèles/régime B.
    Returns {"records": [...], "n_failed": int, "T0": str}."""
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = three_way_split(weekly, n_val, n_test)
    T0_date = weekly_dates.iloc[train_end_pos]

    if model_name == "LSTM" and regime == "C":
        if lstm_weekly_seq_len is None or asset_short not in lstm_weekly_seq_len:
            raise SystemExit(f"pas de SEQ_LEN* pour {asset_short} -- lance "
                            f"experiments/lstm_weekly_sweep.py --assets {asset_short} d'abord.")
        forecast_fn = functools.partial(mh.forecast_horizons_lstm, epochs=None, seed=LSTM_SEED,
                                        seq_len=lstm_weekly_seq_len[asset_short])
    else:
        forecast_fn = (REGIME_C_FORECAST if regime == "C" else REGIME_B_FORECAST)[model_name]
    records, n_failed = [], 0

    for k, m in enumerate(test_pos):
        origin_date, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        last_price = float(weekly.iloc[m])
        actuals = [float(weekly.iloc[m + h]) for h in (1, 2, 3)]

        if regime == "C":
            train_series = weekly.iloc[:m + 1]
            horizons = [1, 2, 3]
        else:
            train_series = daily.iloc[:daily_pos + 1]
            horizons = daily_horizons

        try:
            result = forecast_fn(train_series, horizons)
        except Exception as exc:
            n_failed += 1
            print(f"    [{model_name}/{asset_short}/{regime}] origin {k} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        for wi, w_label in enumerate(HORIZON_LABELS):
            h = horizons[wi]
            point, lo, hi = result[h]
            actual = actuals[wi]
            records.append({
                "asset": asset_short, "horizon": w_label, "model": model_name, "regime": regime,
                "origin": k, "origin_date": str(origin_date.date()),
                "target_date": str(target_dates[wi].date()),
                "daily_steps": h if regime == "B" else None,
                "last_close": last_price, "actual": actual,
                "point": float(point), "lower": float(lo), "upper": float(hi),
            })

    return {"records": records, "n_failed": n_failed, "T0": str(T0_date.date())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--regimes", nargs="+", default=["B", "C"], choices=["B", "C"])
    p.add_argument("--assets", nargs="+", default=["SPY", "BTC", "ETH", "ZN", "TLT"],
                   choices=list(ASSETS))
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "weekly_multimodel_results.json"))
    p.add_argument("--lstm-sweep-file", default=str(LSTM_WEEKLY_SWEEP_FILE),
                   help="SEQ_LEN* par actif pour LSTM régime C (experiments/lstm_weekly_sweep.py)")
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # Chargé seulement si nécessaire : les autres modèles/régime B n'en dépendent pas,
    # pas de raison de faire échouer un run qui ne touche pas au LSTM régime C.
    needs_lstm_c = "LSTM" in args.models and "C" in args.regimes
    lstm_weekly_seq_len = load_lstm_weekly_seq_len(args.lstm_sweep_file) if needs_lstm_c else None

    all_records, meta = [], {}
    t0 = time.time()
    for model_name in args.models:
        for asset in args.assets:
            for regime in args.regimes:
                t_cell = time.time()
                print(f"[{model_name}/{asset}/{regime}] running {args.n_test} test origins ...")
                res = run_model_asset(model_name, asset, ASSETS[asset], regime,
                                      args.n_val, args.n_test, args.start, end,
                                      lstm_weekly_seq_len=lstm_weekly_seq_len)
                elapsed_cell = time.time() - t_cell
                all_records.extend(res["records"])
                meta[f"{model_name}|{asset}|{regime}"] = {
                    "T0": res["T0"], "n_records": len(res["records"]),
                    "n_failed": res["n_failed"], "elapsed_s": round(elapsed_cell, 1),
                }
                print(f"  -> {len(res['records'])} records, {res['n_failed']} failed origins, "
                      f"{elapsed_cell:.1f}s")
    elapsed = time.time() - t0

    payload = {
        "config": {"models": args.models, "regimes": args.regimes, "assets": args.assets,
                  "n_val": args.n_val, "n_test": args.n_test, "start": args.start, "end": end,
                  "elapsed_s": round(elapsed, 1)},
        "meta": meta,
        "records": all_records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min, {len(all_records)} records)")


if __name__ == "__main__":
    main()
