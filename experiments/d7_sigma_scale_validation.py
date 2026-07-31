"""
d7_sigma_scale_validation.py — the dedicated 7-step backtest required before
activating the EWMA sigma correction on D+7.

Context: BRIEF_branchement_prod_calibration_sigma.md deliberately limited the
live EWMA correction to D+1 ("D+7 à activer après un backtest 7-pas dédié"),
and pipeline._run_model_d7_rolling never passes sigma_scale. This script IS
that dedicated backtest, for the 4 models whose D+7 width is still a raw
normal band (SARIMA / Prophet / Naive / LSTM -- ARIMA-GARCH already ships
skew-t quantiles with a GARCH-dynamic sigma at D+7 and is excluded from the
EWMA mechanism by design, same as D+1).

Protocol (mirrors production exactly):
  - Rolling origins over the W1 test window (2020-2024, test 15%): at origin i
    the model is fitted on prices[:split+i] and asked for the h=7 trading-day
    forecast via benchmarks.multi_horizon's own forecast functions (the very
    code the pipeline calls), production defaults on (Prophet log-space,
    ARIMA n/a here).
  - The EWMA correction is then applied causally WITH THE RESOLUTION LAG:
    origin i's standardized residual z_i = (y_true - pred)/sigma_own only
    enters the EWMA state once resolved, i.e. from origin i+7 onward --
    exactly what validation/sigma_scale.py's `cutoff_date < as_of AND y_true
    IS NOT NULL` filter produces in production. State starts at 1.0
    (neutral), lambda = 0.94 (RiskMetrics, same as everywhere else).
  - LSTM: network fitted ONCE per asset on the train window (seed 42), then
    rolled origin by origin on realized history -- production D+7 live path
    also forecasts from a single fitted artifact per day, and the sigma
    mechanism under test is independent of refit cadence.

Scored on the eval suffix (same calibration_eval_split as the option-1/2/3
comparison, applied to the ORIGIN axis): strict MACE at 50/80/95 (normal
quantiles from the 95% band) and per-level coverage, raw vs EWMA-corrected.

Resumable under the 45 s cap: re-invoke until DONE.
Usage:  python experiments/d7_sigma_scale_validation.py --budget-s 36
Output: experiments/d7_sigma_scale_validation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import ASSETS, TEST_RATIO, CALIB_FRAC  # noqa: E402
from offline_prices import fetch_data_offline  # noqa: E402

START, END = "2020-01-01", "2024-12-31"   # W1, the reference window
H = 7                                     # trading-day horizon (D+7)
EWMA_LAMBDA = 0.94
SEED = 42
LEVELS_PCT = (50, 80, 95)

MODELS = ("Naive", "SARIMA", "Prophet", "LSTM")

STATE_PATH = ROOT / "experiments" / "d7_sigma_state.json"
OUT_PATH = ROOT / "experiments" / "d7_sigma_scale_validation.json"
CKPT_DIR = ROOT / "experiments" / "d7_sigma_ckpt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _prices(asset):
    prices = fetch_data_offline(ASSETS[asset], START, END)
    split = int(len(prices) * (1 - TEST_RATIO))
    return prices, split


def _forecast_step(model_name, prices, split, i, lstm_ctx):
    """(point, lo, hi, y_true) for origin i, horizon H trading days."""
    import multi_horizon as mh
    train_hist = prices.iloc[:split + i]
    y_true = float(prices.iloc[split + i + H - 1])
    if model_name == "Naive":
        point, lo, hi = mh.forecast_horizons_naive(train_hist, [H])[H]
    elif model_name == "SARIMA":
        point, lo, hi = mh.forecast_horizons_sarima(train_hist, [H])[H]
    elif model_name == "Prophet":
        point, lo, hi = mh.forecast_horizons_prophet(train_hist, [H])[H]
    else:  # LSTM -- from the per-asset fitted artifact
        model, scaler, std = lstm_ctx
        scaled_hist = scaler.transform(
            train_hist.values.reshape(-1, 1)).flatten()
        point, lo, hi = mh.forecast_from_fitted_lstm(
            model, scaler, std, scaled_hist, [H])[H]
    return float(point), float(lo), float(hi), y_true


def _lstm_ctx(asset, a, prices, split):
    """Fit once per asset (resumable via saved weights), return (model, scaler, std)."""
    import lstm_model as lm
    import multi_horizon as mh
    from sklearn.preprocessing import MinMaxScaler
    train = prices.iloc[:split]
    CKPT_DIR.mkdir(exist_ok=True)
    wpath = CKPT_DIR / f"{asset}.weights.h5"
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    if not wpath.exists():
        lm.set_seed(SEED)
        model, scaler2, std, _scaled = mh.fit_lstm(train, seed=SEED)
        model.save_weights(str(wpath))
        a["lstm_std"] = float(std)
        # mh.fit_lstm fitted its own scaler on the same train -> identical params
        return model, scaler2, float(std)
    model = lm.build_lstm(lm.SEQ_LEN)
    model.load_weights(str(wpath))
    X_train, _ = lm.make_sequences(train_scaled, lm.SEQ_LEN)
    if "lstm_std" not in a:
        train_preds = scaler.inverse_transform(
            model.predict(X_train.reshape(-1, lm.SEQ_LEN, 1),
                          verbose=0).reshape(-1, 1)).flatten()
        a["lstm_std"] = float(np.std(train.values[lm.SEQ_LEN:] - train_preds))
    return model, scaler, a["lstm_std"]


def ewma_scale_path(mu, sigma_own, actual, lag=H, lam=EWMA_LAMBDA):
    """Causal multiplicative scale per origin, with resolution lag: z_i enters
    the state at origin i+lag (when its target date has passed)."""
    z2 = ((np.asarray(actual) - np.asarray(mu)) / np.asarray(sigma_own)) ** 2
    s2 = 1.0
    out = np.empty(len(z2))
    for i in range(len(z2)):
        if i >= lag:
            s2 = lam * s2 + (1 - lam) * z2[i - lag]
        out[i] = np.sqrt(s2)
    return out


def evaluate(raw, n_calib):
    """Strict MACE + coverages on the eval suffix, raw vs EWMA-corrected."""
    from scipy import stats
    mu = np.asarray(raw["preds"], float)
    lo = np.asarray(raw["lower"], float)
    hi = np.asarray(raw["upper"], float)
    actual = np.asarray(raw["actual"], float)
    sigma_own = doc.sigma_from_pi95(lo, hi)
    out = {}
    for name, sig in (("raw", sigma_own),
                      ("ewma", sigma_own * ewma_scale_path(mu, sigma_own, actual))):
        mu_e, sig_e, y_e = mu[n_calib:], sig[n_calib:], actual[n_calib:]
        bounds = doc.quantile_bounds(mu_e, sig_e, lambda p: stats.norm.ppf(p))
        entry = {}
        for lvl in doc.LEVELS:
            blo, bhi = bounds[lvl]
            cov = float(np.mean((y_e >= blo) & (y_e <= bhi))) * 100
            entry[f"cov_{int(lvl * 100)}"] = round(cov, 2)
        entry["mace_strict"] = round(float(np.mean(
            [abs(entry[f"cov_{p}"] - p) for p in LEVELS_PCT])), 2)
        out[name] = entry
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-s", type=float, default=36.0)
    args = p.parse_args()
    deadline = time.time() + args.budget_s

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}

    def save():
        STATE_PATH.write_text(json.dumps(state, indent=2))

    for asset in ASSETS:
        prices, split = _prices(asset)
        n_origins = (len(prices) - split) - H + 1
        a = state.setdefault(asset, {})
        lstm_ctx = None
        for model_name in MODELS:
            r = a.setdefault(model_name, {"preds": [], "lower": [],
                                          "upper": [], "actual": [],
                                          "elapsed_s": 0.0})
            per_step = {"Naive": 0.01, "SARIMA": 0.7,
                        "Prophet": 1.1, "LSTM": 0.25}[model_name]
            while len(r["preds"]) < n_origins:
                if time.time() + max(3.0, per_step * 2) > deadline:
                    save()
                    log(f"PROGRESS {asset}/{model_name} "
                        f"{len(r['preds'])}/{n_origins} -- re-invoke")
                    return
                if model_name == "LSTM" and lstm_ctx is None:
                    if time.time() + 40 > deadline and not \
                            (CKPT_DIR / f"{asset}.weights.h5").exists():
                        save(); log("PROGRESS (lstm fit pending) -- re-invoke")
                        return
                    lstm_ctx = _lstm_ctx(asset, a, prices, split)
                    save()
                i = len(r["preds"])
                t0 = time.time()
                point, lo, hi, y_true = _forecast_step(
                    model_name, prices, split, i, lstm_ctx)
                r["preds"].append(point)
                r["lower"].append(lo)
                r["upper"].append(hi)
                r["actual"].append(y_true)
                r["elapsed_s"] += time.time() - t0
                per_step = max(per_step, time.time() - t0)
                if (i + 1) % 25 == 0:
                    log(f"  {asset}/{model_name}: {i + 1}/{n_origins}")
                    save()
            save()

    # ── final evaluation ─────────────────────────────────────────────────────
    out = {"config": {"window": f"{START} -> {END}", "h_trading_days": H,
                      "ewma_lambda": EWMA_LAMBDA, "resolution_lag": H,
                      "test_ratio": TEST_RATIO, "calib_frac": CALIB_FRAC,
                      "seed": SEED, "models": list(MODELS)},
           "assets": {}}
    acc = {}
    for asset in ASSETS:
        prices, split = _prices(asset)
        n_origins = (len(prices) - split) - H + 1
        n_calib = doc.calibration_eval_split(n_origins, CALIB_FRAC)
        out["assets"][asset] = {}
        for model_name in MODELS:
            res = evaluate(state[asset][model_name], n_calib)
            res["elapsed_s"] = round(state[asset][model_name]["elapsed_s"], 1)
            out["assets"][asset][model_name] = res
            for variant in ("raw", "ewma"):
                acc.setdefault((model_name, variant), []).append(
                    res[variant]["mace_strict"])
    out["summary_mace_strict_mean_5assets"] = {
        f"{m}/{v}": round(float(np.mean(vals)), 2)
        for (m, v), vals in sorted(acc.items())}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    log(f"DONE -> {OUT_PATH}")
    log(json.dumps(out["summary_mace_strict_mean_5assets"], indent=2))


if __name__ == "__main__":
    main()
