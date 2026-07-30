"""
prophet_sigma_investigation.py — WHY is Prophet's sigma ~2x too small?

Background (HANDOFF_dist_options_comparison.md): Prophet massively under-covers
(24.3 / 46.1 / 70.4 % for 50/80/95 targets on 2020-2024). Swapping the tail
SHAPE (student-t / GED) changes nothing (MACE 28.1 -> 28.2), CQR only patches it
to 20.6 by inflating widths 20-32%. Implied sigma ratio from those coverages is
~0.5 at every level -> the problem is the LEVEL of sigma, uniformly, not the
shape. This script isolates the mechanism and tests the candidate causes.

Working hypothesis (tested by phase A): at horizon 1 Prophet's interval is
essentially +/- z * sigma_obs (trend uncertainty ~ 0 one step out), and
sigma_obs is the std of IN-SAMPLE residuals around a flexible trend +
seasonality fit over the whole history. A flexible trend tracks prices closely
in-sample, so in-sample residuals are much smaller than genuine 1-step-ahead
forecast errors (whose floor is the daily-move scale of a near-random-walk
series). If that's right, sigma_pi(h=1) will sit far BELOW even the train
window's own 1-day change std.

Phases (all resumable under a hard 45 s execution cap; re-invoke until DONE):

  A. Diagnostic decomposition (cheap, all 5 assets, one fit each): sigma_obs vs
     in-sample residual std vs 1-step PI width vs the naive daily-change floor.

  B. Config sweep (walk-forward backtests, chunked exactly like the production
     run_prophet -- refit every step): candidate fixes/causes:
       base      -- production config (reference)
       cps_low   -- changepoint_prior_scale=0.001 (stiff trend: bigger in-sample
                    residuals -> bigger sigma_obs; does coverage recover?)
       cps_high  -- changepoint_prior_scale=0.5 (even more flexible: should get
                    WORSE if the overfit mechanism is right)
       no_seas   -- weekly/yearly seasonality off (how much sigma do the
                    seasonal terms absorb?)
       flat      -- growth='flat' (no trend at all)
       log       -- fit on log(price), exponentiate bounds (multiplicative
                    noise; the correct scale-free specification for prices)

  C. Post-hoc variance rescale (cheap, from phase-B "base" arrays): multiply
     sigma_hat by std(z_calib) fitted on the calibration prefix (the intern's
     option-1 machinery fits SHAPE only, unit variance, by design -- so it
     structurally CANNOT fix a variance-level error; this one-line scale fix is
     the natural "option 0" and the fair baseline for any Prophet adoption
     decision). Evaluated with the same MACE/coverage KPIs on the eval suffix.

Usage:
    python experiments/prophet_sigma_investigation.py --budget-s 30
Output: experiments/prophet_sigma_investigation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import (  # noqa: E402
    evaluate_variant, ASSETS, START, END, TEST_RATIO, CALIB_FRAC,
)
from offline_prices import fetch_data_offline  # noqa: E402

SWEEP_ASSETS = ("SPY", "BTC")  # equity + crypto; extend with --assets
Z95 = doc.Z95

CONFIGS = {
    "base": {},
    "cps_low": {"changepoint_prior_scale": 0.001},
    "cps_high": {"changepoint_prior_scale": 0.5},
    "no_seas": {"weekly_seasonality": False, "yearly_seasonality": False},
    "flat": {"growth": "flat"},
    "log": {"_log_space": True},
}

from robustness_windows import WINDOWS  # noqa: E402

_WINDOW = "W1"  # W1 keeps the legacy file names / START-END defaults


def _paths():
    suffix = "" if _WINDOW == "W1" else f"_{_WINDOW}"
    return (ROOT / "experiments" / f"prophet_invest_state{suffix}.json",
            ROOT / "experiments" / f"prophet_sigma_investigation{suffix}.json")


def _window_dates():
    return WINDOWS.get(_WINDOW, (START, END))


STATE_PATH, OUT_PATH = _paths()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_model(cfg):
    from prophet import Prophet
    kwargs = dict(interval_width=0.95, daily_seasonality=False,
                  weekly_seasonality=True, yearly_seasonality=True)
    kwargs.update({k: v for k, v in cfg.items() if not k.startswith("_")})
    return Prophet(**kwargs)


def _fit_predict_next(cfg, hist_ds, hist_y, next_ds):
    """One walk-forward step: fit on history, predict next date.
    Returns (yhat, lo, hi) in PRICE space (log configs are exponentiated)."""
    log_space = cfg.get("_log_space", False)
    y = np.log(hist_y) if log_space else np.asarray(hist_y, float)
    m = _make_model(cfg)
    m.fit(pd.DataFrame({"ds": hist_ds, "y": y}))
    fc = m.predict(pd.DataFrame({"ds": [next_ds]}))
    vals = (float(fc["yhat"].iloc[0]), float(fc["yhat_lower"].iloc[0]),
            float(fc["yhat_upper"].iloc[0]))
    if log_space:
        vals = tuple(np.exp(v) for v in vals)
    return vals


# ── Phase A ──────────────────────────────────────────────────────────────────
def phase_a_asset(ticker):
    prices = fetch_data_offline(ticker, *_window_dates())
    split = int(len(prices) * (1 - TEST_RATIO))
    train = prices.iloc[:split]
    t0 = time.time()
    m = _make_model({})
    df_train = pd.DataFrame({"ds": pd.to_datetime(train.index),
                             "y": train.astype(float).values})
    m.fit(df_train)
    in_sample = m.predict(df_train)
    resid = train.values - in_sample["yhat"].values
    sigma_obs_scaled = float(m.params["sigma_obs"][0]) * float(m.y_scale)
    next_ds = pd.to_datetime(train.index[-1]) + pd.tseries.offsets.BDay(1)
    fc = m.predict(pd.DataFrame({"ds": [next_ds]}))
    pi_sigma_h1 = float((fc["yhat_upper"].iloc[0] - fc["yhat_lower"].iloc[0]) / (2 * Z95))
    daily_change_std = float(np.std(np.diff(train.values)))
    daily_change_std_90d = float(np.std(np.diff(train.values[-90:])))
    return {
        "sigma_obs_scaled": round(sigma_obs_scaled, 6),
        "in_sample_resid_std": round(float(np.std(resid)), 6),
        "in_sample_resid_std_last90": round(float(np.std(resid[-90:])), 6),
        "pi_sigma_h1": round(pi_sigma_h1, 6),
        "train_daily_change_std": round(daily_change_std, 6),
        "train_daily_change_std_last90": round(daily_change_std_90d, 6),
        "ratio_pi_sigma_vs_daily_floor": round(pi_sigma_h1 / daily_change_std, 3),
        "n_changepoints_default": int(m.n_changepoints),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── Phase C ──────────────────────────────────────────────────────────────────
def phase_c_from_base(raw, n_calib):
    """Post-hoc sigma rescale (option 0) + shape/CQR variants on top of the
    RESCALED sigma, from the phase-B base arrays."""
    mu = np.asarray(raw["preds"], float)
    actual = np.asarray(raw["actual"], float)
    sigma_hat = doc.sigma_from_pi95(raw["lower"], raw["upper"])
    z_cal = (actual[:n_calib] - mu[:n_calib]) / sigma_hat[:n_calib]
    scale = float(np.std(z_cal))
    sigma_resc = sigma_hat * scale
    out = {"sigma_scale_factor": round(scale, 3)}
    mu_e, sig_e, y_e = mu[n_calib:], sigma_resc[n_calib:], actual[n_calib:]
    bounds = doc.quantile_bounds(mu_e, sig_e, lambda p: stats.norm.ppf(p))
    out["rescaled_normal"] = evaluate_variant(
        y_e, bounds, lambda p: stats.norm.ppf(p), mu_e, sig_e)
    # shape swap ON TOP of the variance fix (re-standardized calib residuals)
    z_cal2 = z_cal / scale
    dof, ppf_t = doc.fit_student_t(z_cal2)
    out["rescaled_student_t"] = evaluate_variant(
        y_e, doc.quantile_bounds(mu_e, sig_e, ppf_t), ppf_t, mu_e, sig_e)
    out["rescaled_student_t_dof"] = round(dof, 2)
    return out


# ── Phase B chunked walk-forward ─────────────────────────────────────────────
def run_chunk(cfg_name, asset_short, raw, prices, split, deadline):
    test = prices.iloc[split:]
    n_test = len(test)
    cfg = CONFIGS[cfg_name]
    while len(raw["preds"]) < n_test:
        if time.time() + 3.0 > deadline:
            return False
        i = len(raw["preds"])
        hist = prices.iloc[:split + i]
        t0 = time.time()
        yhat, lo, hi = _fit_predict_next(
            cfg, list(pd.to_datetime(hist.index)), hist.values,
            pd.to_datetime(test.index[i]))
        raw["preds"].append(yhat)
        raw["lower"].append(lo)
        raw["upper"].append(hi)
        raw["actual"].append(float(test.iloc[i]))
        raw["elapsed_s"] += time.time() - t0
        if (i + 1) % 25 == 0:
            log(f"  {asset_short}/{cfg_name}: {i + 1}/{n_test}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-s", type=float, default=30.0)
    p.add_argument("--assets", nargs="+", default=list(SWEEP_ASSETS),
                   choices=list(ASSETS))
    p.add_argument("--window", default="W1", choices=list(WINDOWS),
                   help="W1 = legacy file names / original 2020-2024 window")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS),
                   help="restrict the sweep (e.g. --configs base log for a "
                        "cross-window validation of the log fix)")
    args = p.parse_args()
    deadline = time.time() + args.budget_s

    global _WINDOW, STATE_PATH, OUT_PATH
    _WINDOW = args.window
    STATE_PATH, OUT_PATH = _paths()
    run_configs = {k: CONFIGS[k] for k in args.configs}

    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    state.setdefault("phaseA", {})
    state.setdefault("runs", {})

    def save():
        STATE_PATH.write_text(json.dumps(state, indent=2))

    # Phase A
    for asset_short, ticker in ASSETS.items():
        if asset_short in state["phaseA"]:
            continue
        if time.time() + 8 > deadline:
            save(); log("PROGRESS phaseA -- re-invoke"); return
        log(f"phase A: {asset_short}")
        state["phaseA"][asset_short] = phase_a_asset(ticker)
        save()

    # Phase B
    for asset_short in args.assets:
        ticker = ASSETS[asset_short]
        prices = fetch_data_offline(ticker, *_window_dates())
        split = int(len(prices) * (1 - TEST_RATIO))
        for cfg_name in run_configs:
            key = f"{asset_short}|{cfg_name}"
            raw = state["runs"].setdefault(
                key, {"preds": [], "lower": [], "upper": [],
                      "actual": [], "elapsed_s": 0.0})
            try:
                done = run_chunk(cfg_name, asset_short, raw, prices, split, deadline)
            finally:
                save()
            if not done:
                n_done = sum(1 for k, r in state["runs"].items()
                             if len(r["preds"]) and len(r["preds"]) >= 1)
                log(f"PROGRESS phaseB {key}: {len(raw['preds'])} steps -- re-invoke")
                return

    # Assemble output
    w_start, w_end = _window_dates()
    out = {"config": {"window": _WINDOW, "start": w_start, "end": w_end, "test_ratio": TEST_RATIO,
                      "calib_frac": CALIB_FRAC, "sweep_assets": args.assets,
                      "configs": {k: {kk: vv for kk, vv in v.items()}
                                  for k, v in run_configs.items()}},
           "phaseA": state["phaseA"], "phaseB": {}, "phaseC": {}}
    for asset_short in args.assets:
        prices = fetch_data_offline(ASSETS[asset_short], *_window_dates())
        split = int(len(prices) * (1 - TEST_RATIO))
        n_test = len(prices) - split
        n_calib = doc.calibration_eval_split(n_test, CALIB_FRAC)
        out["phaseB"][asset_short] = {}
        for cfg_name in run_configs:
            raw = state["runs"][f"{asset_short}|{cfg_name}"]
            mu = np.asarray(raw["preds"], float)
            actual = np.asarray(raw["actual"], float)
            sigma_hat = doc.sigma_from_pi95(raw["lower"], raw["upper"])
            mu_e, sig_e, y_e = mu[n_calib:], sigma_hat[n_calib:], actual[n_calib:]
            bounds = doc.quantile_bounds(mu_e, sig_e, lambda p: stats.norm.ppf(p))
            kpi = evaluate_variant(y_e, bounds, lambda p: stats.norm.ppf(p),
                                   mu_e, sig_e)
            err_e = y_e - mu_e
            kpi["rmse_eval"] = round(float(np.sqrt(np.mean(err_e ** 2))), 6)
            kpi["mean_sigma_hat_eval"] = round(float(np.mean(sig_e)), 6)
            kpi["realized_err_std_eval"] = round(float(np.std(err_e)), 6)
            kpi["sigma_ratio"] = round(float(np.mean(sig_e) / np.std(err_e)), 3)
            kpi["elapsed_s"] = round(raw["elapsed_s"], 1)
            out["phaseB"][asset_short][cfg_name] = kpi
        if "base" in run_configs:
            out["phaseC"][asset_short] = phase_c_from_base(
                state["runs"][f"{asset_short}|base"], n_calib)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    log(f"DONE -> {OUT_PATH}")


if __name__ == "__main__":
    main()
