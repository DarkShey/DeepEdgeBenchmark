"""
epoch_sweep.py — per-(asset, model) epoch selection on a held-out validation
block, strictly before any test-set evaluation (see BRIEF_weekly_prediction_v2.md).

Three chronological blocks, oldest to newest, NEVER overlapping:

    [ ---- train (<= T0) ---- | -- validation (n_val origins) -- | -- test (n_test origins) -- ]
                              T0                                 T1

Only the validation block is used here. For each (asset, model in {TSDiff-W,
TSDiff-D}) and each epoch candidate in EPOCH_CANDIDATES, train on data <= T0 and
score on the validation origins: CRPS (selection criterion), Cov95 and rel_std%
(diagnostics only, not used for selection). Saves raw candidate scores to
experiments/epoch_sweep_results.json and prints the argmin-CRPS epoch* per
(asset, model) — this file's output is what the user reviews before the final
head-to-head (weekly_headtohead_v2.py) is allowed to run.

Efficiency note: candidates are evaluated via INCREMENTAL checkpointing (train
40 epochs -> evaluate -> train 20 more -> evaluate at 60 -> ...) instead of 5
independent from-scratch fits. TSDiff.train() already leaves the model in a
ready-to-sample state (EMA weights applied to net) after every call, and its
optimizer holds live references to net's parameters, so calling train() again
simply resumes optimization from there — training cost for the whole sweep is
sum(candidates growth) = 120 epoch-units instead of 40+60+80+100+120 = 400.
The only deviation from 5 independent from-scratch fits: each checkpoint's
"resume" starts from the previous checkpoint's EMA-applied weights rather than
its raw SGD weights (train() always ends with self.ema.apply(self.net)) — a
minor smoothing, not a different training regime.

Usage:
    python epoch_sweep.py                       # SPY+BTC, candidates {40,60,80,100,120}
    python epoch_sweep.py --candidates 2 4 --n-val 3 --n-test 5 --n-samples 6 --k-denoise 3
                                                  # fast smoke run, plumbing only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))
import tsdiff_model as td                                     # noqa: E402

from crps_metrics import crps_empirical                        # noqa: E402
from weekly_headtohead import (                                # noqa: E402
    ASSETS, HORIZON_LABELS, HORIZON_WEEKLY, HORIZON_DAILY, WEEK_MARGIN,
    build_weekly, standardized_returns,
)

EPOCH_CANDIDATES = (40, 60, 80, 100, 120)
DEFAULT_N_VAL = 12    # "~10-15 origines" (BRIEF_weekly_prediction_v2.md §3)
DEFAULT_N_TEST = 30
DEFAULT_SEED = 42


def three_way_split(weekly: pd.Series, n_val: int, n_test: int):
    """Positions into `weekly`: (train_end_pos, val_origins_pos, test_origins_pos),
    three disjoint, chronologically ordered blocks with zero overlap. train_end_pos
    is T0 (last position included in training); the n_val validation origins come
    immediately after T0; the n_test test origins come immediately after the last
    validation origin, each still leaving WEEK_MARGIN weekly points for its W1-W3
    targets."""
    n_w = len(weekly)
    needed = n_val + n_test + WEEK_MARGIN
    if n_w < needed + 1:
        raise ValueError(f"only {n_w} weekly points available, need >= {needed + 1} "
                          f"for {n_val} validation + {n_test} test origins with a "
                          f"{WEEK_MARGIN}-week margin.")
    train_end_pos = n_w - needed - 1
    val_origins_pos = list(range(train_end_pos + 1, train_end_pos + 1 + n_val))
    test_origins_pos = list(range(val_origins_pos[-1] + 1, val_origins_pos[-1] + 1 + n_test))
    return train_end_pos, val_origins_pos, test_origins_pos


def week_targets(weekly_dates: pd.Series, daily: pd.Series, m: int):
    """For weekly-origin position m: (origin_date, daily_pos, target_dates[3],
    daily_horizons[3]) — target_dates are the actual trading days realising
    W1/W2/W3, daily_horizons their trading-day distance from the origin."""
    origin_date = weekly_dates.iloc[m]
    daily_pos = int(daily.index.get_loc(origin_date))
    target_dates = [weekly_dates.iloc[m + h] for h in (1, 2, 3)]
    daily_horizons = [int(daily.index.get_loc(d) - daily_pos) for d in target_dates]
    return origin_date, daily_pos, target_dates, daily_horizons


def fit_checkpoints(train: pd.Series, horizon: int, candidates, hidden=td.HIDDEN,
                    depth=td.DEPTH, cond_dim=td.COND_DIM, T=td.T_DIFFUSION,
                    batch_size=td.BATCH_SIZE):
    """Generator yielding (epochs_so_far, model, mu, sd) at each candidate epoch
    count, training incrementally (see module docstring) — no model.train() call
    is ever made with data beyond `train` (guardrail: this is the ONLY training
    data used across the whole sweep for this (asset, model))."""
    train_p = train.values.astype(float)
    r = td._log_returns(train_p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd
    H_win, T_win = td._make_windows(z, td.SEQ_LEN, horizon)
    if len(H_win) == 0:
        raise ValueError("not enough return history to build training windows.")
    model = td.TSDiff(td.SEQ_LEN, horizon, hidden, depth, cond_dim, T)
    done = 0
    for target in sorted(candidates):
        model.train(H_win, T_win, epochs=target - done, batch_size=batch_size)
        done = target
        yield target, model, mu, sd


def _sweep_one_model(asset: str, model_type: str, train_series: pd.Series, horizon: int,
                     full_series: pd.Series, seed: int, candidates, val_pos: list,
                     weekly: pd.Series, weekly_dates: pd.Series, daily: pd.Series,
                     n_samples: int, k_denoise: int) -> list:
    results = []
    td.set_seed(seed)
    for epochs, model, mu, sd in fit_checkpoints(train_series, horizon, candidates):
        buf = standardized_returns(full_series, mu, sd)
        crps_vals, cov_flags, rel_stds = [], [], []
        for k, m in enumerate(val_pos):
            _, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
            last_price = float(weekly.iloc[m])
            if model_type == "TSDiff-W":
                pos, horizons, keymap = m, [1, 2, 3], {1: 1, 2: 2, 3: 3}
            else:
                pos, horizons = daily_pos, daily_horizons
                keymap = {1: daily_horizons[0], 2: daily_horizons[1], 3: daily_horizons[2]}

            td.set_seed(seed + k)
            samples_by_h = td.forecast_from_fitted(model, buf[:pos], mu, sd, last_price,
                                                   horizons=horizons, n_samples=n_samples,
                                                   k_denoise=k_denoise)
            for wi in range(3):
                samples = samples_by_h[keymap[wi + 1]]
                actual = float(weekly.iloc[m + wi + 1])
                crps_vals.append(crps_empirical(samples, actual))
                lo, hi = np.quantile(samples, [0.025, 0.975])
                cov_flags.append(bool(lo <= actual <= hi))
                rel_stds.append(float(samples.std() / last_price * 100))

        rec = {
            "asset": asset, "model": model_type, "epochs": epochs,
            "crps_val": float(np.mean(crps_vals)), "cov95_val": float(np.mean(cov_flags)),
            "rel_std_pct_val": float(np.mean(rel_stds)), "n_val_origins": len(val_pos),
        }
        results.append(rec)
        print(f"[{asset}] {model_type:<9} epochs={epochs:<4} "
              f"CRPS_val={rec['crps_val']:9.4f}  Cov95_val={rec['cov95_val']:.2f}  "
              f"rel_std%={rec['rel_std_pct_val']:.3f}")
    return results


def sweep_asset(asset: str, ticker: str, candidates, n_val: int, n_test: int, seed: int,
                n_samples: int, k_denoise: int, start: str, end: str,
                models=("TSDiff-W", "TSDiff-D"), candidates_d=None):
    """`models` restricts which model types are swept (e.g. ("TSDiff-D",) to
    re-sweep only the daily model with a different candidate list). `candidates_d`,
    if given, overrides `candidates` for TSDiff-D only -- both read the SAME
    validation origins (val_pos, derived from the same start/end/n_val/n_test),
    so results merge cleanly with a prior sweep run under the same config."""
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = three_way_split(weekly, n_val, n_test)
    T0_date = weekly_dates.iloc[train_end_pos]
    train_daily = daily.loc[:T0_date]
    train_weekly = weekly.iloc[:train_end_pos + 1]
    print(f"[{asset}] train <= {T0_date.date()} ({len(train_daily)}d / {len(train_weekly)}w) | "
          f"validation {weekly_dates.iloc[val_pos[0]].date()} -> "
          f"{weekly_dates.iloc[val_pos[-1]].date()} ({len(val_pos)}) | "
          f"test {weekly_dates.iloc[test_pos[0]].date()} -> "
          f"{weekly_dates.iloc[test_pos[-1]].date()} ({len(test_pos)}) [not touched here]")

    records = []
    if "TSDiff-W" in models:
        records += _sweep_one_model(asset, "TSDiff-W", train_weekly, HORIZON_WEEKLY, weekly,
                                    seed, candidates, val_pos, weekly, weekly_dates, daily,
                                    n_samples, k_denoise)
    if "TSDiff-D" in models:
        records += _sweep_one_model(asset, "TSDiff-D", train_daily, HORIZON_DAILY, daily,
                                    seed, candidates_d or candidates, val_pos, weekly,
                                    weekly_dates, daily, n_samples, k_denoise)
    meta = {
        "train_end": str(T0_date.date()),
        "val_origins": [str(weekly_dates.iloc[m].date()) for m in val_pos],
        "test_origins": [str(weekly_dates.iloc[m].date()) for m in test_pos],
    }
    return records, meta


def select_epochs(sweep_records: list) -> dict:
    """{(asset, model): epochs*} — argmin CRPS_val, independently per (asset, model).
    Guardrail: only crps_val (computed on the validation block) feeds this — the
    test block is never touched by epoch_sweep.py at all."""
    df = pd.DataFrame(sweep_records)
    selected = {}
    for (asset, model), g in df.groupby(["asset", "model"]):
        best = g.loc[g["crps_val"].idxmin()]
        selected[f"{asset}|{model}"] = {
            "epochs": int(best["epochs"]),
            "crps_val": float(best["crps_val"]),
            "cov95_val": float(best["cov95_val"]),
            "rel_std_pct_val": float(best["rel_std_pct_val"]),
        }
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--candidates", nargs="+", type=int, default=list(EPOCH_CANDIDATES))
    p.add_argument("--candidates-d", nargs="+", type=int, default=None,
                   help="override --candidates for TSDiff-D only")
    p.add_argument("--models", nargs="+", default=["TSDiff-W", "TSDiff-D"],
                   choices=["TSDiff-W", "TSDiff-D"], help="restrict which model(s) to sweep")
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=max(50, td.N_SAMPLES))
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "epoch_sweep_results.json"))
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    all_records, meta = [], {}
    t0 = time.time()
    for asset in args.assets:
        records, asset_meta = sweep_asset(asset, ASSETS[asset], args.candidates, args.n_val,
                                          args.n_test, args.seed, args.n_samples,
                                          args.k_denoise, args.start, end,
                                          models=args.models, candidates_d=args.candidates_d)
        all_records.extend(records)
        meta[asset] = asset_meta
    elapsed = time.time() - t0

    selected = select_epochs(all_records)

    payload = {
        "config": {
            "candidates": args.candidates, "candidates_d": args.candidates_d,
            "models": args.models, "n_val": args.n_val, "n_test": args.n_test,
            "seed": args.seed, "n_samples": args.n_samples, "k_denoise": args.k_denoise,
            "assets": args.assets, "start": args.start, "end": end,
            "elapsed_s": round(elapsed, 1),
        },
        "meta": meta,
        "records": all_records,
        "selected_epochs": selected,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")
    print(f"\n{'Actif':<6}{'Modele':<10}{'Epochs*':>8}{'CRPS_val':>12}{'Cov95_val':>11}{'rel_std%':>10}")
    print("-" * 57)
    for key, sel in selected.items():
        asset, model = key.split("|")
        print(f"{asset:<6}{model:<10}{sel['epochs']:>8}{sel['crps_val']:>12.4f}"
              f"{sel['cov95_val']:>11.2f}{sel['rel_std_pct_val']:>10.3f}")


if __name__ == "__main__":
    main()
