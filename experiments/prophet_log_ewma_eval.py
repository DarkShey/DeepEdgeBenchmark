"""
prophet_log_ewma_eval.py — answer to the open question in Kyrio's 2026-07-31
session recap (§2) and commit 280e0da: the 5-asset sweep showed the adopted
log-space fix helping SPY/BTC/ETH but NOT ZN, and actively hurting TLT --
while noting that the sweep tested log ALONE, not the log+EWMA combination
actually adopted in production (prophet_model.py: LOG_SPACE=True AND
calibrate_sigma="ewma").

This script scores that full combination. It reads the RAW walk-forward
arrays checkpointed by prophet_sigma_investigation.py (state file, W1
2020-2024, configs base/log for the 5 assets) and applies the exact
production EWMA correction (sigma'_t = sigma_t * sqrt(EWMA(z^2)), causal,
scaled_ewma_path from dynamic_sigma_variants.py) on top of each config's own
sigma path, then reports strict MACE with normal quantiles.

Result (see prophet_log_ewma_eval.json): log+EWMA is the best combination on
EVERY asset -- including ZN (35.8 log-alone -> 8.4) and TLT (36.1 -> 9.9),
where it also beats base+EWMA (15.5 / 14.0). The production default
(log_space=True + calibrate_sigma="ewma" for all assets) therefore stands; no
per-asset-class split is needed. Residual weakness: ZN/TLT keep an
under-covered 50% level (~28-32% vs 50) -- the remaining error is central
sharpness, not the log transform.

Prereq: experiments/prophet_invest_state.json containing base+log runs for
all 5 assets (re-run prophet_sigma_investigation.py --assets ... --configs
base log if missing; the state file is gitignored).
Usage: python experiments/prophet_log_ewma_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "experiments"))

import dist_options_common as doc  # noqa: E402
from all_models_dist_options import option12_variants, ASSETS, CALIB_FRAC  # noqa: E402
from dynamic_sigma_variants import scaled_ewma_path  # noqa: E402

STATE_PATH = ROOT / "experiments" / "prophet_invest_state.json"
OUT_PATH = ROOT / "experiments" / "prophet_log_ewma_eval.json"
LEVELS_PCT = (50, 80, 95)


def main():
    state = json.loads(STATE_PATH.read_text())
    out = {"window": "W1 (2020-01-01 -> 2024-12-31)", "assets": {}}
    for asset in ASSETS:
        out["assets"][asset] = {}
        for cfg in ("base", "log"):
            key = f"{asset}|{cfg}"
            if key not in state["runs"]:
                print(f"  {key}: missing from state -- skipped")
                continue
            raw = state["runs"][key]
            mu = np.asarray(raw["preds"], float)
            actual = np.asarray(raw["actual"], float)
            sig = doc.sigma_from_pi95(raw["lower"], raw["upper"])
            n_calib = doc.calibration_eval_split(len(actual), CALIB_FRAC)
            for pname, s in (("alone", sig),
                             ("ewma", scaled_ewma_path(sig, actual - mu))):
                variants, _, _ = option12_variants(mu, s, actual, n_calib)
                k = variants["normal"]
                mace = float(np.mean([abs(k[f"cov_{p}"] - p)
                                      for p in LEVELS_PCT]))
                out["assets"][asset][f"{cfg}_{pname}"] = {
                    "mace_strict": round(mace, 2),
                    **{f"cov_{p}": k[f"cov_{p}"] for p in LEVELS_PCT},
                }
    combos = ("base_alone", "base_ewma", "log_alone", "log_ewma")
    out["mace_strict_mean_5assets"] = {
        c: round(float(np.mean([out["assets"][a][c]["mace_strict"]
                                for a in ASSETS if c in out["assets"][a]])), 2)
        for c in combos}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["mace_strict_mean_5assets"], indent=2))
    print(f"DONE -> {OUT_PATH}")


if __name__ == "__main__":
    main()
