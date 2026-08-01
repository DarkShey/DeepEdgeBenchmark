"""
d7_guard_analysis.py — design of the D+7 EWMA guard (LSTM activation + lambda
choice), from the raw arrays of the dedicated 7-step backtest
(d7_sigma_scale_validation.py's state file; no model is re-run here).

Problem: the D+7 validation showed the plain lambda=0.94 EWMA severely
degrading LSTM on SPY (MACE 5.0 -> 9.5, cov95 99 -> 83) while helping the
other assets -- which kept LSTM out of the D+7 activation. Diagnosis: D+7
residuals from consecutive daily origins OVERLAP by 6 days, so their z^2 are
strongly autocorrelated; a per-origin lambda of 0.94 effectively discounts a
week of independent information in one step and over-reacts to what is mostly
one repeated shock.

Guard candidates evaluated per model x asset (strict MACE 50/80/95, same
calib/eval split as everywhere):
  raw       -- no correction (reference)
  ewma94    -- per-origin lambda 0.94 (the mechanism as first validated)
  lam_adj   -- lambda = 0.94**(1/7) per origin (= 0.94 per INDEPENDENT
               observation -- the structurally right discount for h=7 overlap)
  damped.5  -- scale = sqrt(EWMA)^0.5 (shrink toward 1)
  adj+damp  -- both
  clamp     -- scale clipped to [0.5, 2]

Findings (see d7_guard_analysis.json):
  - lam_adj FIXES the SPY/LSTM pathology (9.55 -> 1.36) and is best-mean for
    LSTM (5.41), SARIMA (2.91) and Naive (2.81) -- adopted for those three at
    D+7 (LSTM thereby joins the D+7 activation).
  - Prophet is the documented exception: its raw D+7 sigma is so massively and
    persistently wrong (ZN/TLT MACE 54-56) that tracking speed dominates
    smoothness -- lam_adj under-corrects within the window (12.72 vs 9.42).
    Prophet keeps lambda=0.94 at D+7.

Usage:  python experiments/d7_guard_analysis.py
Output: experiments/d7_guard_analysis.json
Prereq: experiments/d7_sigma_state.json (gitignored; re-run
        d7_sigma_scale_validation.py to regenerate).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402

H, LAM = 7, 0.94
STATE_PATH = ROOT / "experiments" / "d7_sigma_state.json"
OUT_PATH = ROOT / "experiments" / "d7_guard_analysis.json"
MODELS = ("Naive", "SARIMA", "Prophet", "LSTM")
ASSETS = ("SPY", "BTC", "ETH", "ZN", "TLT")

VARIANTS = {
    "raw": dict(lam=LAM, gamma=0.0),
    "ewma94": dict(lam=LAM, gamma=1.0),
    "lam_adj": dict(lam=LAM ** (1.0 / H), gamma=1.0),
    "damped.5": dict(lam=LAM, gamma=0.5),
    "adj+damp": dict(lam=LAM ** (1.0 / H), gamma=0.5),
    "clamp": dict(lam=LAM, gamma=1.0, clamp=(0.5, 2.0)),
}


def scale_path(mu, sig, act, lam, gamma=1.0, clamp=None):
    """Causal per-origin scale with the 7-origin resolution lag (same as the
    validation backtest and the live tracking.db query)."""
    z2 = ((np.asarray(act) - np.asarray(mu)) / np.asarray(sig)) ** 2
    s2, out = 1.0, np.empty(len(z2))
    for i in range(len(z2)):
        if i >= H:
            s2 = lam * s2 + (1 - lam) * z2[i - H]
        s = np.sqrt(s2) ** gamma
        if clamp:
            s = min(max(s, clamp[0]), clamp[1])
        out[i] = s
    return out


def main():
    state = json.loads(STATE_PATH.read_text())
    out = {"config": {"h": H, "base_lambda": LAM,
                      "variants": {k: {kk: vv for kk, vv in v.items()}
                                   for k, v in VARIANTS.items()}},
           "mace_strict": {}, "summary_mean_5assets": {}}
    for model in MODELS:
        for asset in ASSETS:
            r = state[asset][model]
            mu = np.asarray(r["preds"], float)
            act = np.asarray(r["actual"], float)
            sig = doc.sigma_from_pi95(r["lower"], r["upper"])
            n_calib = doc.calibration_eval_split(len(act), 0.4)
            for name, kw in VARIANTS.items():
                s = sig * scale_path(mu, sig, act, **kw)
                mu_e, s_e, y_e = mu[n_calib:], s[n_calib:], act[n_calib:]
                b = doc.quantile_bounds(mu_e, s_e, lambda p: stats.norm.ppf(p))
                mace = float(np.mean(
                    [abs(np.mean((y_e >= b[l][0]) & (y_e <= b[l][1])) * 100 - l * 100)
                     for l in doc.LEVELS]))
                out["mace_strict"][f"{model}/{asset}/{name}"] = round(mace, 2)
    for model in MODELS:
        for name in VARIANTS:
            vals = [out["mace_strict"][f"{model}/{a}/{name}"] for a in ASSETS]
            out["summary_mean_5assets"][f"{model}/{name}"] = round(float(np.mean(vals)), 2)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary_mean_5assets"], indent=2))
    print(f"DONE -> {OUT_PATH}")


if __name__ == "__main__":
    main()
