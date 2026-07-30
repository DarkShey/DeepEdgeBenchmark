"""
generate_baseline_weekly_bootstrap.py — VOLET A.1 (MESSAGE_claude_code_etape2_equite_crps.md):
real empirical (bootstrap / MC-Dropout) N=500 sample clouds for the 5 baselines
(ARIMA-GARCH, SARIMA, Prophet, Naive, LSTM) at WEEKLY horizons (W+1/W+2/W+3),
on the EXACT SAME 78-origin x 5-asset triplets as step1_final -- to neutralise
the CRPS asymmetry (baselines Gaussian-parametric vs TSDiff-W native diffusion
samples) that the step1 verdict was built on.

Each model already has a DAILY 1-step bootstrap mechanism (`n_ensemble` in
models/*.py, cf. model_artifacts/crps_kpis.py's DEFAULT_N_ENSEMBLE): residual
bootstrap around the already-fitted 1-step forecast (ARIMA-GARCH/SARIMA/
Prophet), or MC-Dropout (LSTM) -- "not a fresh distributional assumption".
This script is the MULTI-STEP (h=1,2,3 weeks) generalisation of that SAME
mechanism, reusing the SAME already-fitted objects
(benchmarks/multi_horizon.fit_arima/fit_sarima/fit_prophet/fit_lstm,
weekly_multimodel.forecast_horizons_sarima_weekly/_prophet_weekly for the
weekly-native variants) -- no new modelling assumption, just h-step compounding:

  - ARIMA-GARCH: sum of h standardized-GARCH-residual draws (same pool as the
    daily n_ensemble), EACH scaled by that step's OWN GARCH-forecasted sigma
    (already computed by forecast_from_fitted_arima's var_per_step), added to
    the h-step cumulative mean forecast. GARCH standardized residuals are ~iid
    by construction -- no approximation caveat here.
  - SARIMA: sum of h in-sample-residual draws (same pool as daily, `result.
    resid`), added to predicted_mean[h]. APPROXIMATION (documented, not
    hidden): treats the residual pool as iid across the h compounded steps,
    ignoring the AR/MA weighting a fully correct h-step innovation variance
    would apply. Validity check: report the resulting bootstrap interval
    coverage (50/80/95%) in the KPI output -- if close to nominal, the
    approximation holds; if not, that's exactly where it breaks.
  - Prophet: sum of h in-sample-residual draws (same pool as daily, yhat -
    y over history). Same APPROXIMATION and same validity check as SARIMA.
  - Naive: reuses weekly_headtohead_v2.random_walk_samples VERBATIM -- already
    a genuine empirical h-week cumulative-return distribution from realised
    history, nothing to generalise.
  - LSTM: MC-Dropout RECURSIVE rollout -- at EACH of the h steps, one batched
    dropout-active forward pass (N=500 rows, independent dropout masks,
    mirroring the daily model(x_batch, training=True) trick), feeding each
    row's OWN stochastic prediction back into ITS OWN buffer for the next
    step (N independent stochastic paths, not one path replicated).

N=500 for every baseline -- MUST match TSDiff-W's N=500 (Claude cowork's tutor
review, point 1): crps_empirical is finite-N biased, and a different N per
side would silently reintroduce an estimator-level version of the very
asymmetry this script exists to remove.

Anti-leakage: identical to step1_final / backtest_rolling_tsdiffw.py -- each
origin's fit uses only weekly data <= that origin (train_series = weekly.iloc
[:m+1]), matching weekly_multimodel.py's regime-C convention exactly.

Storage: point+PI (mean, 2.5%/97.5% quantiles of the bootstrap cloud) upserted
into tracking.db as source='step2_baseline_empirical' (own partial unique
index, isolated from oos/live/backtest_rolling*). Full clouds + per-row KPIs
go to the JSON deliverable only (same convention as TSDiff: raw samples are
never bloat tracking.db).

Usage:
    python generate_baseline_weekly_bootstrap.py --smoke          # 1 asset, 2 origins, tiny -- plumbing only
    python generate_baseline_weekly_bootstrap.py                  # full run, all 5 assets
    python generate_baseline_weekly_bootstrap.py --assets SPY     # subset
"""

import os

# Must run BEFORE any yfinance/statsmodels import (tsdiff_model, arima_model below pull
# those in at module level): importing tensorflow for the first time AFTER yfinance/
# statsmodels have already been imported in this process triggers a confirmed deadlock
# (stack frozen in TFE_Execute, 0% CPU, never returns) -- same root cause documented and
# worked around in models/conftest.py and experiments/weekly_multimodel.py. This script
# needs tensorflow (bootstrap_lstm_weekly), so the guard is unconditional, same as there.
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
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "experiments"))

import tsdiff_model as td                                        # noqa: E402 (fetch_data reused)
import multi_horizon as mh                                        # noqa: E402
import arima_model                                                 # noqa: E402
import naive_model                                                  # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly  # noqa: E402
from weekly_headtohead_v2 import random_walk_samples                # noqa: E402
from weekly_multimodel import (                                     # noqa: E402
    forecast_horizons_sarima_weekly, forecast_horizons_prophet_weekly,
    load_lstm_weekly_seq_len, LSTM_WEEKLY_SWEEP_FILE, LSTM_SEED,
)
from backtest_rolling_tsdiffw import load_baseline_triplets, select_origins  # noqa: E402
from validation import tracking_db as tdb                           # noqa: E402

DB_PATH = str(ROOT / "validation" / "tracking.db")
BASELINE_MODELS = ("ARIMA-GARCH", "SARIMA", "Prophet", "Naive", "LSTM")
HORIZON_UNITS = {1: "W+1", 2: "W+2", 3: "W+3"}
DEFAULT_N_SAMPLES = 500     # MUST match TSDiff-W's N=500 -- non-negotiable (finite-N CRPS bias)
FETCH_START = "2015-01-01"
FETCH_END = "2026-07-24"    # matches step1_final -- fixed, reproducible
SOURCE_TAG = "step2_baseline_empirical"
RUN_ID = "step2-baseline-weekly-bootstrap"


# ── per-model bootstrap (h-step generalisation of each model's daily n_ensemble) ──

def bootstrap_arima_garch(train_weekly: pd.Series, horizons: list, n_samples: int,
                          rng: np.random.Generator) -> dict:
    """Sum of h standardized-GARCH-residual draws, each scaled by that step's
    OWN GARCH-forecasted sigma -- iid by construction, no approximation caveat."""
    arima_res, garch_res = mh.fit_arima(train_weekly)
    last_price = float(train_weekly.iloc[-1])
    max_h = max(horizons)

    mean_fc = np.asarray(arima_res.forecast(steps=max_h), dtype=float) / 100.0
    garch_fc = garch_res.forecast(horizon=max_h, reindex=False)
    var_per_step = garch_fc.variance.values[-1, :] / (100.0 ** 2)
    sigma_per_step = np.sqrt(var_per_step)
    cum_return = np.cumsum(mean_fc)

    std_resid = np.asarray(garch_res.resid, dtype=float) / \
        np.asarray(garch_res.conditional_volatility, dtype=float)

    out = {}
    for h in horizons:
        z_boot = rng.choice(std_resid, size=(n_samples, h), replace=True)     # [n_samples, h]
        cum_shock = (z_boot * sigma_per_step[:h]).sum(axis=1)                  # [n_samples]
        out[h] = last_price * np.exp(cum_return[h - 1] + cum_shock)
    return out


def bootstrap_sarima_weekly(train_weekly: pd.Series, horizons: list, n_samples: int,
                            rng: np.random.Generator) -> dict:
    """Sum of h in-sample-residual draws added to predicted_mean[h].
    APPROXIMATION: treats the residual pool as iid across compounded steps --
    validity checked via bootstrap interval coverage (see module docstring)."""
    import sarima_model
    history = train_weekly.astype(float).values.tolist()
    result = sarima_model.SARIMAX(
        history, order=sarima_model.ORDER, seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    max_h = max(horizons)
    fc = result.get_forecast(steps=max_h)
    pred_mean = np.asarray(fc.predicted_mean, dtype=float)
    residuals = np.asarray(result.resid, dtype=float)

    out = {}
    for h in horizons:
        draws = rng.choice(residuals, size=(n_samples, h), replace=True)
        out[h] = pred_mean[h - 1] + draws.sum(axis=1)
    return out


def bootstrap_prophet_weekly(train_weekly: pd.Series, horizons: list, n_samples: int,
                             rng: np.random.Generator) -> dict:
    """Sum of h in-sample-residual draws added to yhat[h]. Same approximation +
    validity check as SARIMA."""
    import prophet_model
    df_train = pd.DataFrame({"ds": pd.to_datetime(train_weekly.index),
                             "y": train_weekly.astype(float).values.flatten()})
    model = prophet_model.Prophet(
        interval_width=1 - prophet_model.PI_ALPHA,
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
    )
    model.fit(df_train)
    in_sample = model.predict(df_train)["yhat"].values
    residual_pool = in_sample - df_train["y"].values

    max_h = max(horizons)
    last_date = pd.to_datetime(train_weekly.index[-1])
    future_dates = pd.date_range(start=last_date, periods=max_h + 1, freq="W-FRI")[1:]
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))

    out = {}
    for h in horizons:
        point = float(forecast.iloc[h - 1]["yhat"])
        draws = rng.choice(residual_pool, size=(n_samples, h), replace=True)
        out[h] = point + draws.sum(axis=1)
    return out


def bootstrap_naive_weekly(weekly_r_raw: np.ndarray, last_price: float, horizons: list,
                           n_samples: int, rng: np.random.Generator) -> dict:
    """Verbatim reuse of weekly_headtohead_v2.random_walk_samples -- already a
    genuine empirical h-week cumulative-return distribution, nothing to bootstrap."""
    out = {}
    for h in horizons:
        pool = random_walk_samples(weekly_r_raw, h)
        draws = rng.choice(pool, size=n_samples, replace=True)
        out[h] = last_price * np.exp(draws)
    return out


def bootstrap_lstm_weekly(train_weekly: pd.Series, horizons: list, n_samples: int,
                          seq_len: int) -> dict:
    """MC-Dropout RECURSIVE rollout: at each of h steps, one batched dropout-active
    forward pass (n_samples independent dropout masks), each row fed back into ITS
    OWN buffer for the next step -- n_samples independent stochastic paths."""
    import lstm_model
    lstm_model.set_seed(LSTM_SEED)
    scaler = lstm_model.MinMaxScaler()
    scaled = scaler.fit_transform(train_weekly.values.reshape(-1, 1)).flatten()
    X, y = lstm_model.make_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = lstm_model.build_lstm(seq_len)
    es = lstm_model.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, y, epochs=lstm_model.EPOCHS, batch_size=lstm_model.BATCH_SIZE,
             validation_split=0.1, callbacks=[es], verbose=0)

    max_h = max(horizons)
    buffers = np.tile(scaled[-seq_len:], (n_samples, 1))            # [n_samples, seq_len]
    rollout_scaled = np.empty((n_samples, max_h), dtype=np.float64)
    for step in range(max_h):
        x_batch = buffers[:, -seq_len:].reshape(n_samples, seq_len, 1).astype(np.float32)
        p_scaled = model(x_batch, training=True).numpy().flatten()   # dropout active
        rollout_scaled[:, step] = p_scaled
        buffers = np.concatenate([buffers, p_scaled.reshape(-1, 1)], axis=1)

    rollout_prices = scaler.inverse_transform(
        rollout_scaled.reshape(-1, 1)).reshape(n_samples, max_h)
    return {h: rollout_prices[:, h - 1] for h in horizons}


# ── per-asset driver ──────────────────────────────────────────────────────────

def generate_asset(asset: str, ticker: str, origin_rows: pd.DataFrame, n_samples: int,
                   seed: int, lstm_seq_len: int) -> list:
    print(f"[{asset}] downloading {ticker} ({FETCH_START} -> {FETCH_END}) ...")
    daily = td.fetch_data(ticker, FETCH_START, FETCH_END)
    weekly, weekly_dates = build_weekly(daily)
    wd_values = weekly_dates.values
    weekly_r_raw = td._log_returns(weekly.values.astype(float))   # RAW (non-standardized), for Naive

    cutoffs = sorted(origin_rows["cutoff_date"].unique())
    records = []
    t0 = time.time()
    for k, cutoff_date in enumerate(cutoffs):
        matches = np.where(wd_values == np.datetime64(pd.Timestamp(cutoff_date)))[0]
        if not len(matches):
            raise SystemExit(f"[{asset}] cutoff {cutoff_date} not on the weekly grid.")
        m = int(matches[0])
        if m == 0:
            continue
        train_weekly = weekly.iloc[:m + 1]
        last_price = float(weekly.iloc[m])
        grp = origin_rows[origin_rows["cutoff_date"] == cutoff_date]
        horizons = sorted(int(h) for h in grp["horizon"].unique())

        rng = np.random.default_rng((seed, asset, cutoff_date).__hash__() % (2**32))
        model_samples = {
            "ARIMA-GARCH": bootstrap_arima_garch(train_weekly, horizons, n_samples, rng),
            "SARIMA": bootstrap_sarima_weekly(train_weekly, horizons, n_samples, rng),
            "Prophet": bootstrap_prophet_weekly(train_weekly, horizons, n_samples, rng),
            "Naive": bootstrap_naive_weekly(weekly_r_raw[:m], last_price, horizons, n_samples, rng),
            "LSTM": bootstrap_lstm_weekly(train_weekly, horizons, n_samples, lstm_seq_len),
        }
        for model_name, samples_by_h in model_samples.items():
            for _, row in grp.iterrows():
                h = int(row["horizon"])
                if h not in samples_by_h:
                    continue
                records.append({
                    "model": model_name, "asset": asset, "horizon": h,
                    "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": HORIZON_UNITS[h],
                    "cutoff_date": cutoff_date, "target_date": row["target_date"],
                    "last_close": last_price, "y_true": row["y_true"],
                    "method": "empirical_bootstrap_weekly", "n_samples": n_samples,
                    "samples": samples_by_h[h],
                })
        if (k + 1) % 10 == 0 or k == len(cutoffs) - 1:
            print(f"[{asset}] origin {k + 1}/{len(cutoffs)} done ({time.time() - t0:.0f}s elapsed)")
    return records


# ── DB storage ────────────────────────────────────────────────────────────────

def ensure_index(db_path: str = DB_PATH) -> None:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        con.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_{SOURCE_TAG}_unique
            ON predictions (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
            WHERE source = '{SOURCE_TAG}'
        """)
        con.commit()
    finally:
        con.close()


def save_rows(index: pd.DataFrame, samples: np.ndarray, db_path: str = DB_PATH) -> int:
    ensure_index(db_path)
    import sqlite3
    con = sqlite3.connect(db_path)
    n = 0
    try:
        for i, row in index.iterrows():
            s = samples[i]
            y_pred = float(np.mean(s))
            y_lower, y_upper = (float(q) for q in np.quantile(s, [0.025, 0.975]))
            rec = {
                "run_id": RUN_ID, "model": row["model"], "asset": row["asset"],
                "horizon": int(row["horizon"]), "regime": "unknown",
                "cutoff_date": row["cutoff_date"], "target_date": row["target_date"],
                "last_close": float(row["last_close"]), "y_pred": y_pred,
                "y_lower": y_lower, "y_upper": y_upper, "y_true": float(row["y_true"]),
                "source": SOURCE_TAG, "frequence": "weekly", "horizon_type": "weekly",
                "horizon_unit": row["horizon_unit"],
                "real_flag": tdb.compute_real_flag(row["model"], row["cutoff_date"]),
            }
            cols = list(rec.keys())
            placeholders = ", ".join(f":{c}" for c in cols)
            columns = ", ".join(cols)
            con.execute(f"""
                INSERT INTO predictions ({columns}) VALUES ({placeholders})
                ON CONFLICT (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
                WHERE source='{SOURCE_TAG}'
                DO UPDATE SET
                    run_id=excluded.run_id, target_date=excluded.target_date,
                    last_close=excluded.last_close, y_pred=excluded.y_pred,
                    y_lower=excluded.y_lower, y_upper=excluded.y_upper,
                    y_true=excluded.y_true, real_flag=excluded.real_flag
            """, rec)
            n += 1
        con.commit()
    finally:
        con.close()
    return n


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS.values()))
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 asset (SPY), 2 origins only -- plumbing check")
    p.add_argument("--out-prefix", default=str(ROOT / "experiments" / "baseline_weekly_bootstrap"))
    args = p.parse_args()

    assets = ["SPY"] if args.smoke else args.assets
    n_samples = args.n_samples

    lstm_seq_len = load_lstm_weekly_seq_len(LSTM_WEEKLY_SWEEP_FILE)

    triplets = load_baseline_triplets(assets)
    origins = select_origins(triplets, assets)
    if args.smoke:
        origins = origins[origins["cutoff_date"].isin(sorted(origins["cutoff_date"].unique())[:2])]

    all_records = []
    t0 = time.time()
    for asset in assets:
        origin_rows = origins[origins["asset"] == asset]
        short = [s for s, t in ASSET_TICKERS.items() if t == asset][0]
        all_records += generate_asset(asset, asset, origin_rows, n_samples, args.seed,
                                      lstm_seq_len[short])
    elapsed = time.time() - t0

    samples = np.stack([r.pop("samples") for r in all_records]).astype(np.float64)
    index = pd.DataFrame(all_records)
    print(f"\n{len(index)} rows x {n_samples} samples generated in {elapsed / 60:.1f} min")

    if args.dry_run or args.smoke:
        print("--dry-run/--smoke: skipping DB write.")
    else:
        n_written = save_rows(index, samples)
        print(f"Upserted {n_written} rows into {DB_PATH} (source='{SOURCE_TAG}').")

    np.savez_compressed(f"{args.out_prefix}_samples.npz", samples=samples)
    index.to_json(f"{args.out_prefix}_index.json", orient="records", indent=2)
    print(f"Saved -> {args.out_prefix}_index.json / _samples.npz")


if __name__ == "__main__":
    main()
