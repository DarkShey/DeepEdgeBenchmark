"""
lstm_weekly_sweep.py — LSTM regime-C (weekly-native) hyperparameter selection on
a held-out validation block, per BRIEF_lstm_weekly_retune.md.

Mirrors epoch_sweep.py's discipline (selection on CRPS validation, test block
NEVER touched here) but sweeps `SEQ_LEN` for LSTM regime C instead of TSDiff
`epochs`. Same three_way_split/week_targets as the rest of the weekly protocol
(epoch_sweep.py, weekly_multimodel.py) -- same validation block already used
for TSDiff, disjoint from the 30 test origins.

Grid (reduced after review from the brief's {8,12,16,26}x{30,60}): with
n_val=12, an 8-candidate grid has too little power to trust small CRPS
differences. Only SEQ_LEN varies -- SEQ_LEN_CANDIDATES = (8, 16, 26), EPOCHS
stays at the daily default (lstm_model.EPOCHS, via epochs=None).

Selection: 1-SE rule, not raw argmin-of-12-origins. Per (asset, regime C
candidate): mean + standard error of the per-row CRPS across the n_val*3
(origins x W1/W2/W3) validation rows. Among candidates within one standard
error of the minimum-mean candidate, keep the most PARSIMONIOUS one (smallest
SEQ_LEN) -- avoids overfitting a hyperparameter choice to noise at this sample
size.

Regime B parity check (not a re-tune -- regime B keeps SEQ_LEN=30/lstm_model
defaults in production unconditionally): the SAME validation origins and the
SAME CRPS computation are also run for LSTM regime B at its unchanged daily
default, so the sweep's summary can honestly state whether that untouched
default is itself close to ITS OWN validation optimum -- same selection rule
(CRPS validation, 1-SE) applied per regime, just not acted on for B.

CRPS: closed-form Gaussian (honest_eval.metrics.crps_gaussian) -- LSTM's own PI
IS point +/- 1.96*sigma (forecast_from_fitted_lstm), so this is EXACT for LSTM,
not an approximation (same convention as weekly_vs_daily_pooled.py).

Usage:
    python experiments/lstm_weekly_sweep.py
    python experiments/lstm_weekly_sweep.py --assets SPY BTC --n-val 4   # smoke test
"""

# Must run BEFORE any yfinance/statsmodels import (epoch_sweep.py -> tsdiff_model
# pulls those in at module level): importing tensorflow for the first time AFTER
# yfinance/statsmodels have already been imported in this process triggers a
# confirmed deadlock (see weekly_multimodel.py's module docstring / models/conftest.py).
import os
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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import multi_horizon as mh                                       # noqa: E402
from honest_eval.metrics import crps_gaussian                     # noqa: E402
from weekly_headtohead import ASSETS, build_weekly                # noqa: E402
from epoch_sweep import three_way_split, week_targets, DEFAULT_N_VAL, DEFAULT_N_TEST, DEFAULT_SEED  # noqa: E402

SEQ_LEN_CANDIDATES = (8, 16, 26)   # regime C only -- daily default (30) NOT retuned
LSTM_SEED = 42                     # same seed as weekly_multimodel.py's LSTM_SEED


def _row_crps(point: float, lo: float, hi: float, actual: float) -> float:
    sigma = (hi - lo) / (2 * 1.96)
    return crps_gaussian(point, sigma, actual)


def _mean_se(values: list) -> tuple:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def sweep_regime_c(weekly: pd.Series, val_pos: list, candidates=SEQ_LEN_CANDIDATES) -> dict:
    """{seq_len: crps_values (list, n_val*3 rows)} -- refit at every validation
    origin (same convention as weekly_multimodel.py::run_model_asset regime C:
    LSTM is cheap enough that the TSDiff train-once-forward trick isn't needed)."""
    out = {}
    for seq_len in candidates:
        crps_vals = []
        for m in val_pos:
            train_series = weekly.iloc[:m + 1]
            result = mh.forecast_horizons_lstm(train_series, horizons=[1, 2, 3],
                                               epochs=None, seed=LSTM_SEED, seq_len=seq_len)
            for h in (1, 2, 3):
                point, lo, hi = result[h]
                actual = float(weekly.iloc[m + h])
                crps_vals.append(_row_crps(point, lo, hi, actual))
        out[seq_len] = crps_vals
    return out


def sweep_regime_b_default(weekly: pd.Series, weekly_dates: pd.Series, daily: pd.Series,
                           val_pos: list) -> list:
    """LSTM regime B (daily-trained, multi-step to weekly targets) at the
    UNCHANGED daily default (seq_len=None -> lstm_model.SEQ_LEN) -- parity
    check only, this config is never selected/overridden for production."""
    crps_vals = []
    for m in val_pos:
        _, daily_pos, _, daily_horizons = week_targets(weekly_dates, daily, m)
        train_series = daily.iloc[:daily_pos + 1]
        result = mh.forecast_horizons_lstm(train_series, horizons=daily_horizons,
                                           epochs=None, seed=LSTM_SEED)
        for wi, h in enumerate(daily_horizons):
            point, lo, hi = result[h]
            actual = float(weekly.iloc[m + wi + 1])
            crps_vals.append(_row_crps(point, lo, hi, actual))
    return crps_vals


def select_seq_len(candidate_stats: dict) -> dict:
    """1-SE rule: among candidates within one standard error of the minimum-mean
    candidate, keep the most parsimonious (smallest SEQ_LEN, i.e. shortest
    lookback). `candidate_stats`: {seq_len: (mean, se)}."""
    best_seq_len = min(candidate_stats, key=lambda k: candidate_stats[k][0])
    best_mean, best_se = candidate_stats[best_seq_len]
    threshold = best_mean + best_se
    qualifying = [sl for sl, (mean, _) in candidate_stats.items() if mean <= threshold]
    selected = min(qualifying)   # most parsimonious among those statistically tied with the min
    sel_mean, sel_se = candidate_stats[selected]
    return {"seq_len": selected, "mean_crps_val": sel_mean, "se_crps_val": sel_se,
            "argmin_seq_len": best_seq_len, "argmin_mean_crps_val": best_mean,
            "threshold_1se": threshold}


def sweep_asset(asset: str, ticker: str, n_val: int, n_test: int, start: str, end: str,
                candidates=SEQ_LEN_CANDIDATES) -> dict:
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    import tsdiff_model as td   # fetch_data reused, TF already imported above (safe order)
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = three_way_split(weekly, n_val, n_test)
    T0_date = weekly_dates.iloc[train_end_pos]
    print(f"[{asset}] train <= {T0_date.date()} | validation "
          f"{weekly_dates.iloc[val_pos[0]].date()} -> {weekly_dates.iloc[val_pos[-1]].date()} "
          f"({len(val_pos)}) [test block reserved, not touched]")

    t0 = time.time()
    regime_c_raw = sweep_regime_c(weekly, val_pos, candidates=candidates)
    regime_c_stats = {sl: _mean_se(vals) for sl, vals in regime_c_raw.items()}
    selection = select_seq_len(regime_c_stats)

    regime_b_raw = sweep_regime_b_default(weekly, weekly_dates, daily, val_pos)
    regime_b_mean, regime_b_se = _mean_se(regime_b_raw)
    elapsed = time.time() - t0

    print(f"[{asset}] done in {elapsed:.0f}s")
    for sl, (mean, se) in sorted(regime_c_stats.items()):
        mark = " *" if sl == selection["seq_len"] else ""
        print(f"    regime C  seq_len={sl:<4} CRPS_val={mean:8.4f} +/- {se:.4f}{mark}")
    print(f"    regime B  seq_len=30(default) CRPS_val={regime_b_mean:8.4f} +/- {regime_b_se:.4f}"
          f"  [unchanged, parity check only]")

    return {
        "T0": str(T0_date.date()),
        "n_val": len(val_pos),
        "regime_c": {str(sl): {"mean_crps_val": m, "se_crps_val": s}
                    for sl, (m, s) in regime_c_stats.items()},
        "selected": selection,
        "regime_b_default_crps_val": {"seq_len": 30, "mean_crps_val": regime_b_mean,
                                      "se_crps_val": regime_b_se},
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--candidates", nargs="+", type=int, default=list(SEQ_LEN_CANDIDATES))
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "lstm_weekly_sweep.json"))
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = tuple(args.candidates)

    results = {}
    t0 = time.time()
    for asset in args.assets:
        results[asset] = sweep_asset(asset, ASSETS[asset], args.n_val, args.n_test,
                                     args.start, end, candidates=candidates)
    elapsed = time.time() - t0

    payload = {
        "config": {"candidates": list(candidates), "assets": args.assets,
                  "n_val": args.n_val, "n_test": args.n_test, "start": args.start,
                  "end": end, "elapsed_s": round(elapsed, 1)},
        "results": results,
        "selected_seq_len": {asset: r["selected"]["seq_len"] for asset, r in results.items()},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")
    print(f"\n{'Actif':<6}{'SEQ_LEN*':>10}{'CRPS_val':>12}{'B default':>12}")
    print("-" * 40)
    for asset, r in results.items():
        sel = r["selected"]
        print(f"{asset:<6}{sel['seq_len']:>10}{sel['mean_crps_val']:>12.4f}"
              f"{r['regime_b_default_crps_val']['mean_crps_val']:>12.4f}")


if __name__ == "__main__":
    main()
