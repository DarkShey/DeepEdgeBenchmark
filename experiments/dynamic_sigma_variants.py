"""
dynamic_sigma_variants.py — does a TIME-VARYING sigma fix Naive and SARIMA the
way it fixes the LSTM?

Motivation: the multi-window robustness runs (robustness_windows.py, W1/W2/W3)
show that NO static post-hoc option (student-t / GED shape swap, CQR) survives
a regime change for SARIMA and Naive -- each window has a different "winner"
because the miscalibration is a sigma-LEVEL error that moves with volatility
regimes, exactly the diagnosis already made for the LSTM (frozen sigma) where
an EWMA sigma path halved the calibration error (lstm_sigma_variants.py).

This script re-scores the SAME raw walk-forward arrays checkpointed by
robustness_windows.py (no model is re-run) with dynamic sigma paths:

  Naive  : sigma is frozen in production (std of train daily changes). Replace
           it with EWMA(lambda=0.94) / GARCH-free rolling paths on OBSERVED
           daily changes (strictly causal, warm-started on the train window).
  SARIMA : sigma_t (Kalman conf_int) is already dynamic but its LEVEL drifts.
           Correct it multiplicatively: sigma'_t = sigma_t * sqrt(EWMA(z_t^2)),
           z_t = e_t / sigma_t -- a causal variance-targeting correction that
           preserves SARIMA's own dynamics. (The raw-EWMA path is also scored
           for symmetry.)

Each sigma path is then pushed through the intern's option-1/2 machinery
(normal / student_t / ged / cqr on the same calib/eval split), for all three
windows. Both aggregations are reported:
  mace_strict = mean over assets of |cov - target|  (used in this follow-up)
  mace_loose  = |mean over assets of cov - target|  (the aggregation used by
                summarize_dist_options.py -- NOTE: it lets opposite-signed
                per-asset errors cancel and flatters every option)

Prereq: robustness_state_W{1,2,3}.json fully populated (ALL_DONE).
Usage:  python experiments/dynamic_sigma_variants.py
Output: experiments/dynamic_sigma_variants_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import option12_variants, ASSETS, TEST_RATIO, CALIB_FRAC  # noqa: E402
from robustness_windows import WINDOWS  # noqa: E402
from offline_prices import fetch_data_offline  # noqa: E402

EWMA_LAMBDA = 0.94
OUT_PATH = ROOT / "experiments" / "dynamic_sigma_variants_results.json"
LEVELS_PCT = (50, 80, 95)


def ewma_path(warm_resid, test_resid, lam=EWMA_LAMBDA):
    """Causal EWMA sigma: sigma_t uses residuals < t, seeded on warm_resid."""
    s2 = float(np.var(warm_resid))
    for e in np.asarray(warm_resid, float)[-250:]:
        s2 = lam * s2 + (1 - lam) * e * e
    out = np.empty(len(test_resid))
    for t, e in enumerate(np.asarray(test_resid, float)):
        out[t] = np.sqrt(s2)
        s2 = lam * s2 + (1 - lam) * e * e
    return out


def scaled_ewma_path(sigma_own, test_resid, lam=EWMA_LAMBDA):
    """sigma'_t = sigma_own_t * sqrt(EWMA(z^2)), z_t = e_t / sigma_own_t.
    Causal; EWMA of squared standardized residuals seeded at 1 (i.e. 'trust the
    model's own sigma until told otherwise')."""
    sigma_own = np.asarray(sigma_own, float)
    e = np.asarray(test_resid, float)
    s2 = 1.0
    out = np.empty(len(e))
    for t in range(len(e)):
        out[t] = sigma_own[t] * np.sqrt(s2)
        z = e[t] / sigma_own[t]
        s2 = lam * s2 + (1 - lam) * z * z
    return out


def mace_pair(cov_rows):
    """cov_rows: list over assets of {50: c, 80: c, 95: c}."""
    strict = float(np.mean([abs(r[l] - l) for r in cov_rows for l in LEVELS_PCT]))
    loose = float(np.mean([abs(np.mean([r[l] for r in cov_rows]) - l)
                           for l in LEVELS_PCT]))
    return round(strict, 2), round(loose, 2)


def main():
    t0 = time.time()
    out = {"config": {"ewma_lambda": EWMA_LAMBDA, "test_ratio": TEST_RATIO,
                      "calib_frac": CALIB_FRAC, "windows": WINDOWS},
           "windows": {}, "summary": {}}

    for wname, (start, end) in WINDOWS.items():
        spath = ROOT / "experiments" / f"robustness_state_{wname}.json"
        state = json.loads(spath.read_text())
        wout = {}
        acc = {}  # (model, path, variant) -> list of cov dicts
        for asset in ASSETS:
            a = state[asset]
            prices = fetch_data_offline(ASSETS[asset], start, end)
            split = int(len(prices) * (1 - TEST_RATIO))
            train = prices.iloc[:split]
            n_test = a["n_test"]
            n_calib = doc.calibration_eval_split(n_test, CALIB_FRAC)
            aout = {}
            for model, raw_key in (("Naive", "naive"), ("SARIMA", "sarima")):
                raw = a[raw_key]
                mu = np.asarray(raw["preds"], float)
                actual = np.asarray(raw["actual"], float)
                resid = actual - mu
                sigma_own = doc.sigma_from_pi95(raw["lower"], raw["upper"])
                paths = {"own": sigma_own}
                if model == "Naive":
                    warm = np.diff(train.values.astype(float))
                    paths["ewma94"] = ewma_path(warm, resid)
                else:
                    paths["ewma94"] = ewma_path(resid[:20], resid)  # crude seed
                    paths["scaled_ewma"] = scaled_ewma_path(sigma_own, resid)
                mout = {}
                for pname, sig in paths.items():
                    variants, fitted, _ = option12_variants(mu, sig, actual, n_calib)
                    mout[pname] = {"fitted_shapes": fitted, "variants": variants}
                    for vname, kpi in variants.items():
                        acc.setdefault((model, pname, vname), []).append(
                            {l: kpi[f"cov_{l}"] for l in LEVELS_PCT})
                aout[model] = mout
            wout[asset] = aout
        out["windows"][wname] = wout
        out["summary"][wname] = {}
        for (model, pname, vname), rows in sorted(acc.items()):
            strict, loose = mace_pair(rows)
            out["summary"][wname][f"{model}/{pname}/{vname}"] = {
                "mace_strict": strict, "mace_loose": loose}

    # cross-window mean of mace_strict, the adoption-decision number
    cross = {}
    for wname in WINDOWS:
        for key, v in out["summary"][wname].items():
            cross.setdefault(key, []).append(v["mace_strict"])
    out["summary"]["cross_window_mace_strict_mean"] = {
        k: round(float(np.mean(v)), 2) for k, v in sorted(cross.items())}
    out["config"]["elapsed_s"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"DONE -> {OUT_PATH}")
    best = sorted(out["summary"]["cross_window_mace_strict_mean"].items(),
                  key=lambda kv: kv[1])
    for k, v in best[:12]:
        print(f"  {v:6.2f}  {k}")


if __name__ == "__main__":
    main()
