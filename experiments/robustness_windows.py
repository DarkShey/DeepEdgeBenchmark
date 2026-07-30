"""
robustness_windows.py — re-run the PI-calibration options comparison
(all_models_dist_options.py, options 1/2 + ARIMA-GARCH native rows) on
ADDITIONAL time windows, to check that the per-model winners observed on the
single 2020-2024 window (HANDOFF_dist_options_comparison.md) hold elsewhere
before anything is wired into models/*.py for real.

Scope: the 3 models where the 2020-2024 comparison produced a clear winner and
adoption is actually on the table -- ARIMA-GARCH, SARIMA, Naive. (Prophet and
LSTM are excluded on purpose: both are known-broken at the sigma level itself,
see experiments/prophet_sigma_investigation.py and lstm_sigma_variants.py.)

Data comes from the committed DONNEE~1.XLS (see offline_prices.py), verified
identical to yfinance -- so runs are reproducible offline.

Chunked / resumable execution: this script is designed to survive hard 45-second
execution caps. It processes work units (whole ARIMA/Naive backtests; SARIMA
walk-forward in --chunk-steps slices) until --budget-s seconds have elapsed,
checkpointing raw walk-forward arrays to a state JSON after every unit. SARIMA
slicing is EXACT: run_sarima refits on the full history at every step, so
running steps [k, k+c) with history prices[:split+k] reproduces the monolithic
run bit-for-bit. Re-invoke until it prints ALL_DONE.

Output: robustness_<window>_results.json, same schema as
all_models_dist_options_results.json, so summarize_dist_options.py-style
aggregation and side-by-side comparison with the 2020-2024 reference apply.

Usage (repeat until ALL_DONE):
    python experiments/robustness_windows.py --window W2 --budget-s 30
    python experiments/robustness_windows.py --window W3 --budget-s 30
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

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import (  # noqa: E402
    evaluate_variant, option12_variants, ASSETS, TEST_RATIO, CALIB_FRAC,
    CRPS_N_SAMPLES, SEED,
)
from offline_prices import fetch_data_offline  # noqa: E402

# Reference window W1 = 2020-01-01 -> 2024-12-31 (already run by the intern:
# all_models_dist_options_results.json). W2 is an earlier regime mix (includes
# the 2018 rate cycle + covid crash in TRAIN and scores on 2022's bear market);
# W3 is entirely fresh data ending 2026 (its eval suffix was never seen by ANY
# earlier comparison).
WINDOWS = {
    # W1 = the intern's reference window. Re-run here too (results should agree
    # with all_models_dist_options_results.json for the shared models) because
    # the original run only kept KPIs -- the RAW per-step arrays checkpointed by
    # this script are needed by dynamic_sigma_variants.py.
    "W1": ("2020-01-01", "2024-12-31"),
    "W2": ("2018-03-01", "2022-12-31"),
    "W3": ("2022-01-01", "2026-06-30"),
}

MODELS = ("ARIMA-GARCH", "SARIMA", "Naive")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _state_path(window):
    return ROOT / "experiments" / f"robustness_state_{window}.json"


def _results_path(window):
    return ROOT / "experiments" / f"robustness_{window}_results.json"


def _load(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def _split(prices):
    split = int(len(prices) * (1 - TEST_RATIO))
    return prices.iloc[:split], prices.iloc[split:], split


def _finalize_asset(state_a, prices, n_calib):
    """All raw runs done for this asset -> compute option-1/2 variants exactly
    like all_models_dist_options.run_asset."""
    out = {"n_test": state_a["n_test"], "n_calib": n_calib, "models": {}}
    for model_name in ("SARIMA", "Naive"):
        raw = state_a[model_name.lower()]
        mu = np.asarray(raw["preds"], float)
        actual = np.asarray(raw["actual"], float)
        sigma_hat = doc.sigma_from_pi95(raw["lower"], raw["upper"])
        variants, fitted, overhead = option12_variants(mu, sigma_hat, actual, n_calib)
        out["models"][model_name] = {
            "base_train_time_s": round(raw["elapsed_s"], 2),
            "fitted_shapes": fitted, "overhead_s": overhead, "variants": variants,
        }
    raw = state_a["arima_normal"]
    mu = np.asarray(raw["preds"], float)
    actual = np.asarray(raw["actual"], float)
    sigma_hat = doc.sigma_from_pi95(raw["lower"], raw["upper"])
    variants, fitted, overhead = option12_variants(mu, sigma_hat, actual, n_calib)
    for dist_name in ("ged", "skewt"):
        raw_n = state_a[f"arima_native_{dist_name}"]
        bounds = {lvl: (np.asarray(raw_n["pi_bounds"][str(lvl)]["lower"], float),
                        np.asarray(raw_n["pi_bounds"][str(lvl)]["upper"], float))
                  for lvl in doc.LEVELS}
        kpi = evaluate_variant(np.asarray(raw_n["actual"], float), bounds,
                               crps_ensemble=[np.asarray(e, float)
                                              for e in raw_n["crps_ensemble"]])
        variants[f"native_{dist_name}"] = kpi
        overhead[f"native_{dist_name}"] = round(raw_n["elapsed_s"], 2)
    out["models"]["ARIMA-GARCH"] = {
        "base_train_time_s": round(state_a["arima_normal"]["elapsed_s"], 2),
        "fitted_shapes": fitted, "overhead_s": overhead, "variants": variants,
    }
    return out


def process_asset(asset_short, ticker, start, end, state, deadline, chunk_steps):
    """Returns True if the asset is fully finalized."""
    import arima_model as am
    import sarima_model as sm
    import naive_model as nm

    prices = fetch_data_offline(ticker, start, end)
    train, test, split = _split(prices)
    n_test = len(test)
    n_calib = doc.calibration_eval_split(n_test, CALIB_FRAC)

    a = state.setdefault(asset_short, {"n_test": n_test})
    if a.get("finalized"):
        return True

    def has_time(est):
        return time.time() + est < deadline

    # 1. Naive (instant)
    if "naive" not in a:
        t0 = time.time()
        res = nm.run_naive(train, test)
        a["naive"] = {"preds": list(map(float, res["predictions"])),
                      "lower": list(map(float, res["lower"])),
                      "upper": list(map(float, res["upper"])),
                      "actual": list(map(float, res["actual"])),
                      "elapsed_s": time.time() - t0}
        log(f"  {asset_short}: naive done")

    # 2. ARIMA-GARCH normal baseline (~3-6 s)
    if "arima_normal" not in a:
        if not has_time(10):
            return False
        t0 = time.time()
        res = am.run_arima_garch(train, test, dist="normal")
        a["arima_normal"] = {"preds": list(map(float, res["predictions"])),
                             "lower": list(map(float, res["lower"])),
                             "upper": list(map(float, res["upper"])),
                             "actual": list(map(float, res["actual"])),
                             "elapsed_s": time.time() - t0}
        log(f"  {asset_short}: arima_normal done ({a['arima_normal']['elapsed_s']:.1f}s)")

    # 3. ARIMA-GARCH native rows (eval window only, ~2-4 s each)
    train_ext = prices.iloc[:split + n_calib]
    test_eval = test.iloc[n_calib:]
    for dist_name in ("ged", "skewt"):
        key = f"arima_native_{dist_name}"
        if key not in a:
            if not has_time(8):
                return False
            t0 = time.time()
            res = am.run_arima_garch(train_ext, test_eval, dist=dist_name,
                                     n_crps_samples=CRPS_N_SAMPLES,
                                     ensemble_seed=SEED)
            a[key] = {
                "pi_bounds": {str(lvl): {"lower": list(map(float, res["pi_bounds"][lvl]["lower"])),
                                         "upper": list(map(float, res["pi_bounds"][lvl]["upper"]))}
                              for lvl in doc.LEVELS},
                "actual": list(map(float, res["actual"])),
                "crps_ensemble": [list(map(float, e)) for e in res["crps_ensemble"]],
                "elapsed_s": time.time() - t0,
            }
            log(f"  {asset_short}: {key} done ({a[key]['elapsed_s']:.1f}s)")

    # 4. SARIMA, chunked (exact: refit-per-step on full history)
    s = a.setdefault("sarima", {"preds": [], "lower": [], "upper": [],
                                "actual": [], "elapsed_s": 0.0})
    while len(s["preds"]) < n_test:
        done = len(s["preds"])
        # estimate the next chunk's cost from the last measured chunk (fallback
        # 0.35 s/step before any measurement), +30% safety margin
        est = s.get("last_chunk_s", chunk_steps * 0.35) * 1.3
        if not has_time(max(6, est)):
            return False
        k_end = min(done + chunk_steps, n_test)
        hist = prices.iloc[:split + done]
        chunk = test.iloc[done:k_end]
        t0 = time.time()
        res = sm.run_sarima(hist, chunk)
        s["preds"].extend(map(float, res["predictions"]))
        s["lower"].extend(map(float, res["lower"]))
        s["upper"].extend(map(float, res["upper"]))
        s["actual"].extend(map(float, res["actual"]))
        s["elapsed_s"] += time.time() - t0
        s["last_chunk_s"] = time.time() - t0
        log(f"  {asset_short}: sarima {k_end}/{n_test} "
            f"(+{k_end - done} steps, {time.time() - t0:.1f}s)")

    a["finalized"] = True
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", required=True, choices=list(WINDOWS))
    p.add_argument("--budget-s", type=float, default=30.0)
    p.add_argument("--chunk-steps", type=int, default=40)
    args = p.parse_args()

    start, end = WINDOWS[args.window]
    deadline = time.time() + args.budget_s
    spath, rpath = _state_path(args.window), _results_path(args.window)
    state = _load(spath)

    all_done = True
    for asset_short, ticker in ASSETS.items():
        try:
            done = process_asset(asset_short, ticker, start, end, state,
                                 deadline, args.chunk_steps)
        finally:
            _save(spath, state)
        if not done:
            all_done = False
            break  # budget exhausted

    if not all_done:
        n = sum(1 for a in state.values() if isinstance(a, dict) and a.get("finalized"))
        log(f"PROGRESS window={args.window}: {n}/{len(ASSETS)} assets finalized "
            f"-- re-invoke to continue")
        return

    # Assemble final results file (idempotent)
    payload = {"config": {"assets": list(ASSETS), "start": start, "end": end,
                          "test_ratio": TEST_RATIO, "calib_frac": CALIB_FRAC,
                          "crps_n_samples": CRPS_N_SAMPLES, "seed": SEED,
                          "models": list(MODELS), "window": args.window},
               "assets": {}}
    for asset_short, ticker in ASSETS.items():
        prices = fetch_data_offline(ticker, start, end)
        _, test, _ = _split(prices)
        n_calib = doc.calibration_eval_split(len(test), CALIB_FRAC)
        payload["assets"][asset_short] = _finalize_asset(state[asset_short],
                                                         prices, n_calib)
    _save(rpath, payload)
    log(f"ALL_DONE window={args.window} -> {rpath}")


if __name__ == "__main__":
    main()
