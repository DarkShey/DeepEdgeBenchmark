"""
all_models_dist_options.py — extend alternatives_distributions_pi.pdf's option 1
(alternate innovation distribution) to all 5 classical models, and add options
2 (CQR) and 3 (MDN, LSTM only) as A/B tests -- NOT productionised. Goal: measure
whether the extra compute is worth it, per model, before deciding to wire any of
this into models/*.py for real.

Every model's own existing walk-forward backtest (models.<model>.run_<model>,
production code, unmodified except models/arima_model.py's already-committed
`dist=`/`pi_levels`/`n_crps_samples` support) is called EXACTLY ONCE per asset for
the Gaussian baseline. All the option-1/2 variants below are then pure
post-processing on top of that single run's (mu, sigma_hat, actual) triple --
see experiments/dist_options_common.py's module docstring for why sigma_hat is
backed out from the existing 95% PI width rather than re-derived per model (same
convention already used by generate_distributions_dashboard.py / honest_eval).
ARIMA-GARCH additionally gets two "native" rows (dist='ged'/'skewt', its own
step-exact GARCH refit) for comparison against the uniform manual-fit method
applied to every model.

The walk-forward TEST window is split chronologically into a calibration prefix
(fits alt-distribution shapes / conformal scores -- never scored) and an eval
suffix (the only window any KPI is computed on) -- see
dist_options_common.calibration_eval_split.

Usage (from repo root; set CURL_CA_BUNDLE first if yfinance SSL fails locally):
    python experiments/all_models_dist_options.py
    python experiments/all_models_dist_options.py --assets SPY BTC
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402

ASSETS = {"SPY": "SPY", "BTC": "BTC-USD", "ETH": "ETH-USD", "ZN": "ZN=F", "TLT": "TLT"}
START, END = "2020-01-01", "2024-12-31"
TEST_RATIO = 0.15
CALIB_FRAC = 0.4
CRPS_N_SAMPLES = 300
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_variant(actual, bounds, ppf_fn=None, mu=None, sigma=None,
                     n_crps_samples=CRPS_N_SAMPLES, crps_ensemble=None):
    """bounds: {level: (lower_arr, upper_arr)} on the EVAL window. Returns a KPI
    dict. CRPS from `ppf_fn` (parametric variants) or a precomputed
    `crps_ensemble` (native ARIMA-GARCH rows) if given; omitted for CQR (a shift
    of existing bounds, not a full predictive distribution)."""
    out = {}
    for level in doc.LEVELS:
        lo, hi = bounds[level]
        cov, width = doc.coverage_width(actual, lo, hi)
        pct = int(round(level * 100))
        out[f"cov_{pct}"] = round(cov, 2)
        out[f"width_{pct}"] = round(width, 6)
    out["pinball"] = round(doc.avg_pinball(actual, bounds), 6)
    if crps_ensemble is not None:
        scores = [doc.crps_empirical(e, a) for e, a in zip(crps_ensemble, actual)]
        out["crps"] = round(float(np.mean(scores)), 6)
    elif ppf_fn is not None and mu is not None and sigma is not None:
        out["crps"] = round(doc.crps_from_ppf(ppf_fn, mu, sigma, actual,
                                              n_samples=n_crps_samples, seed=SEED), 6)
    return out


def option12_variants(mu, sigma_hat, actual, n_calib):
    """Baseline (normal) + option-1 manual (student_t/ged) + option-2 (CQR),
    all derived from the SAME (mu, sigma_hat, actual) triple -- see module
    docstring. Returns {variant_name: kpi_dict}, plus timing overhead per option
    (fit/conformal cost only, in seconds -- the base model's own Train Time is
    reported separately)."""
    mu_c, sig_c, y_c = mu[:n_calib], sigma_hat[:n_calib], actual[:n_calib]
    mu_e, sig_e, y_e = mu[n_calib:], sigma_hat[n_calib:], actual[n_calib:]

    t0 = time.time()
    bounds_normal_full = doc.quantile_bounds(mu, sigma_hat, lambda p: stats.norm.ppf(p))
    bounds_normal_cal = {lvl: (b[0][:n_calib], b[1][:n_calib]) for lvl, b in bounds_normal_full.items()}
    bounds_normal_eval = {lvl: (b[0][n_calib:], b[1][n_calib:]) for lvl, b in bounds_normal_full.items()}
    t_normal = time.time() - t0

    z_cal = (y_c - mu_c) / sig_c
    t0 = time.time()
    dof, ppf_t = doc.fit_student_t(z_cal)
    bounds_t_eval = doc.quantile_bounds(mu_e, sig_e, ppf_t)
    t_student_t = time.time() - t0

    t0 = time.time()
    beta, ppf_g = doc.fit_ged(z_cal)
    bounds_g_eval = doc.quantile_bounds(mu_e, sig_e, ppf_g)
    t_ged = time.time() - t0

    t0 = time.time()
    bounds_cqr_eval = doc.conformalize_bounds(bounds_normal_cal, y_c, bounds_normal_eval)
    t_cqr = time.time() - t0

    results = {
        "normal": evaluate_variant(y_e, bounds_normal_eval, lambda p: stats.norm.ppf(p), mu_e, sig_e),
        "student_t": evaluate_variant(y_e, bounds_t_eval, ppf_t, mu_e, sig_e),
        "ged": evaluate_variant(y_e, bounds_g_eval, ppf_g, mu_e, sig_e),
        "cqr": evaluate_variant(y_e, bounds_cqr_eval),
    }
    fitted = {"student_t_dof": round(float(dof), 3), "ged_beta": round(float(beta), 3)}
    overhead_s = {"normal": round(t_normal, 4), "student_t": round(t_student_t, 4),
                 "ged": round(t_ged, 4), "cqr": round(t_cqr, 4)}
    return results, fitted, overhead_s


def run_asset(ticker: str, asset_short: str) -> dict:
    import arima_model as am
    import sarima_model as sm
    import prophet_model as pm
    import lstm_model as lm
    import naive_model as nm

    prices = am.fetch_data(ticker, START, END)
    split = int(len(prices) * (1 - TEST_RATIO))
    train, test = prices.iloc[:split], prices.iloc[split:]
    n_test = len(test)
    n_calib = doc.calibration_eval_split(n_test, CALIB_FRAC)
    log(f"  {asset_short}: train={len(train)} test={n_test} (calib={n_calib}, eval={n_test - n_calib})")

    asset_out = {"n_test": n_test, "n_calib": n_calib, "models": {}}

    # ── SARIMA / Prophet / LSTM / Naive: single baseline run, sigma backed out ──
    baseline_runners = {
        "SARIMA": lambda: sm.run_sarima(train, test),
        "Prophet": lambda: pm.run_prophet(train, test),
        "LSTM": lambda: lm.run_lstm(train, test),
        "Naive": lambda: nm.run_naive(train, test),
    }
    for model_name, runner in baseline_runners.items():
        log(f"    [{asset_short}/{model_name}] baseline walk-forward ...")
        t0 = time.time()
        res = runner()
        base_time = time.time() - t0
        mu = np.asarray(res["predictions"], float)
        actual = np.asarray(res["actual"], float)
        sigma_hat = doc.sigma_from_pi95(res["lower"], res["upper"])
        variants, fitted, overhead = option12_variants(mu, sigma_hat, actual, n_calib)
        asset_out["models"][model_name] = {
            "base_train_time_s": round(base_time, 2),
            "fitted_shapes": fitted,
            "overhead_s": overhead,
            "variants": variants,
        }
        log(f"    [{asset_short}/{model_name}] done ({base_time:.1f}s base)  "
           f"normal cov50={variants['normal']['cov_50']}  "
           f"student_t cov50={variants['student_t']['cov_50']}  "
           f"cqr cov50={variants['cqr']['cov_50']}")

    # ── ARIMA-GARCH: uniform manual method (same as above) + native ged/skewt ──
    log(f"    [{asset_short}/ARIMA-GARCH] baseline walk-forward ...")
    t0 = time.time()
    res = am.run_arima_garch(train, test, dist="normal")
    base_time = time.time() - t0
    mu = np.asarray(res["predictions"], float)
    actual = np.asarray(res["actual"], float)
    sigma_hat = doc.sigma_from_pi95(res["lower"], res["upper"])
    variants, fitted, overhead = option12_variants(mu, sigma_hat, actual, n_calib)

    # Native rows: absorb the calibration window into training (mirrors a real
    # deployment re-fitting on the most recent data), walk-forward only the eval
    # window, using run_arima_garch's own step-exact dist=/pi_bounds/n_crps_samples.
    train_ext = prices.iloc[:split + n_calib]
    test_eval = test.iloc[n_calib:]
    native = {}
    for dist_name in ("ged", "skewt"):
        log(f"    [{asset_short}/ARIMA-GARCH] native dist={dist_name} (eval window only) ...")
        t0 = time.time()
        res_native = am.run_arima_garch(train_ext, test_eval, dist=dist_name,
                                        n_crps_samples=CRPS_N_SAMPLES, ensemble_seed=SEED)
        native_time = time.time() - t0
        bounds_native = {lvl: (res_native["pi_bounds"][lvl]["lower"], res_native["pi_bounds"][lvl]["upper"])
                         for lvl in doc.LEVELS}
        kpi = evaluate_variant(np.asarray(res_native["actual"], float), bounds_native,
                               crps_ensemble=res_native["crps_ensemble"])
        native[f"native_{dist_name}"] = kpi
        overhead[f"native_{dist_name}"] = round(native_time, 2)

    variants.update(native)
    asset_out["models"]["ARIMA-GARCH"] = {
        "base_train_time_s": round(base_time, 2),
        "fitted_shapes": fitted,
        "overhead_s": overhead,
        "variants": variants,
    }
    log(f"    [{asset_short}/ARIMA-GARCH] done  normal cov50={variants['normal']['cov_50']}  "
       f"native_ged cov50={variants['native_ged']['cov_50']}  "
       f"native_skewt cov50={variants['native_skewt']['cov_50']}")

    return asset_out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "all_models_dist_options_results.json"))
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip assets already present in --out (default on -- this "
                        "is a long multi-asset run meant to survive interruption/"
                        "handoff; pass --no-resume to force a clean rerun")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    existing = {}
    if args.resume and Path(args.out).exists():
        try:
            existing = json.loads(Path(args.out).read_text())
        except Exception as exc:
            log(f"  --resume: couldn't read existing {args.out} ({exc}), starting fresh")

    payload = {
        "config": {"assets": args.assets, "start": START, "end": END,
                  "test_ratio": TEST_RATIO, "calib_frac": CALIB_FRAC,
                  "crps_n_samples": CRPS_N_SAMPLES, "seed": SEED},
        "assets": existing.get("assets", {}),
    }
    done = set(payload["assets"])
    if done:
        log(f"--resume: {sorted(done)} already in {args.out}, skipping")

    t_start = time.time()
    for asset_short in args.assets:
        if asset_short in done:
            continue
        ticker = ASSETS[asset_short]
        log(f"=== {asset_short} ({ticker}) ===")
        payload["assets"][asset_short] = run_asset(ticker, asset_short)
        Path(args.out).write_text(json.dumps(payload, indent=2))  # incremental save
        log(f"  saved progress -> {args.out}")

    elapsed = time.time() - t_start
    payload["config"]["elapsed_s"] = round(elapsed, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    log(f"DONE in {elapsed/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
