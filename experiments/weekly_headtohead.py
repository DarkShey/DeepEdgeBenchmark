"""
weekly_headtohead.py — TSDiff-W vs TSDiff-D, train-once-forward (see
BRIEF_weekly_prediction.md for the full protocol and rationale).

Trains each model exactly ONCE per asset (on data <= the first walk-forward
origin), then loops over N walk-forward origins re-sampling from the fitted
model via forecast_from_fitted — no retraining inside the loop. TSDiff-W is
trained natively on weekly returns (horizon=3 -> W1/W2/W3 directly); TSDiff-D
is trained on daily returns with horizon=15 trading days (~3 weeks) and its
forecast is read off at the exact daily-step distance to each weekly target,
then compared on the SAME target dates as TSDiff-W.

Guardrails (do not violate — see BRIEF_weekly_prediction.md §7):
  - mu/sd are frozen to the first origin's training stats, never recomputed.
  - every history window fed to forecast_from_fitted only ever contains
    returns realised <= the origin (buffer[:pos] slicing below).
  - same seed / n_samples / k_denoise / target dates for both models.

Usage:
    python weekly_headtohead.py                         # SPY+BTC, 300 epochs, 30 origins
    python weekly_headtohead.py --epochs 2 --n-origins 3 --n-samples 6 --k-denoise 3
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
import tsdiff_model as td                                    # noqa: E402

from crps_metrics import crps_empirical                       # noqa: E402

ASSETS = {"SPY": "SPY", "BTC": "BTC-USD", "ETH": "ETH-USD", "ZN": "ZN=F", "TLT": "TLT"}
HORIZON_LABELS = ("W1", "W2", "W3")

DEFAULT_N_ORIGINS = 30
DEFAULT_EPOCHS    = 300
DEFAULT_SEED      = 42
HORIZON_WEEKLY    = 3
HORIZON_DAILY     = 15    # 3 weeks x 5 trading days — plan item 1: a run param, not a model default
WEEK_MARGIN       = 3     # weekly points reserved past the last origin, for its W1-W3 targets


def build_weekly(daily: pd.Series):
    """Friday-anchored weekly close series (`.last()` within each week), plus,
    aligned 1:1 to it, the ACTUAL trading date realising that close. The W-FRI
    bin label is always the calendar Friday, even when Friday is a market
    holiday (e.g. Juneteenth) and the week's last trading day was Thursday —
    so the label itself must never be used to index the daily series."""
    weekly = daily.resample("W-FRI").last().dropna()
    weekly_dates = daily.index.to_series().resample("W-FRI").last().dropna()
    return weekly, weekly_dates


def pick_origins(weekly: pd.Series, n_origins: int) -> list:
    """Positions (into `weekly`) of the walk-forward origins: the `n_origins` most
    recent weekly points that still leave WEEK_MARGIN future weekly points (W1-W3)
    to evaluate the *last* origin against."""
    n_w = len(weekly)
    needed = n_origins + WEEK_MARGIN
    if n_w < needed + 1:
        raise ValueError(f"only {n_w} weekly points available, need >= {needed + 1} "
                          f"for {n_origins} origins with a {WEEK_MARGIN}-week margin.")
    start = n_w - needed
    return list(range(start, start + n_origins))


def standardized_returns(prices: pd.Series, mu: float, sd: float) -> np.ndarray:
    """Log-returns of `prices` standardized with a FIXED (mu, sd) — never
    recomputed per origin, so no lookahead leaks into the standardization."""
    r = td._log_returns(prices.values.astype(float))
    return (r - mu) / sd


def run_pair(asset: str, ticker: str, n_origins: int, epochs: int, seed: int,
            n_samples: int, k_denoise: int, start: str, end: str) -> dict:
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)
    origins_pos = pick_origins(weekly, n_origins)
    first_origin_date = weekly_dates.iloc[origins_pos[0]]     # actual trading day, not the W-FRI label

    train_daily  = daily.loc[:first_origin_date]              # exact label match, inclusive
    train_weekly = weekly.iloc[:origins_pos[0] + 1]            # position-based: same inclusivity
    print(f"[{asset}] train <= {first_origin_date.date()} "
          f"({len(train_daily)} daily / {len(train_weekly)} weekly obs) | "
          f"{n_origins} origins {weekly.index[origins_pos[0]].date()} -> "
          f"{weekly.index[origins_pos[-1]].date()}")

    # ── train-once-forward: fit each model exactly once (guardrail: same seed) ──
    t0 = time.time()
    td.set_seed(seed)
    model_w, mu_w, sd_w = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs)
    print(f"[{asset}] TSDiff-W fitted in {time.time() - t0:.0f}s "
          f"(mu={mu_w:.6f}, sd={sd_w:.6f})")

    t0 = time.time()
    td.set_seed(seed)
    model_d, mu_d, sd_d = td.fit_tsdiff(train_daily, horizon=HORIZON_DAILY, epochs=epochs)
    print(f"[{asset}] TSDiff-D fitted in {time.time() - t0:.0f}s "
          f"(mu={mu_d:.6f}, sd={sd_d:.6f})")

    # Full-range standardized-return buffers, built ONCE with the frozen mu/sd.
    # Guardrail: every origin below slices buffer[:pos] — realised-only history.
    weekly_z = standardized_returns(weekly, mu_w, sd_w)
    daily_z  = standardized_returns(daily, mu_d, sd_d)

    records = []
    for k, m in enumerate(origins_pos):
        origin_date  = weekly_dates.iloc[m]                    # actual trading day
        daily_pos    = int(daily.index.get_loc(origin_date))
        last_price_w = float(weekly.iloc[m])
        last_price_d = float(daily.iloc[daily_pos])
        assert abs(last_price_w - last_price_d) < 1e-6, \
            "weekly/daily close mismatch at a supposedly shared origin date"

        target_dates    = [weekly_dates.iloc[m + h] for h in (1, 2, 3)]   # actual trading days
        actuals          = [float(weekly.iloc[m + h]) for h in (1, 2, 3)]
        daily_horizons   = [int(daily.index.get_loc(d) - daily_pos) for d in target_dates]

        # Guardrail: identical seed for both models at this origin (paired draws) —
        # varies across origins for statistical independence, matched within an origin.
        td.set_seed(seed + k)
        samples_w = td.forecast_from_fitted(model_w, weekly_z[:m], mu_w, sd_w, last_price_w,
                                            horizons=[1, 2, 3], n_samples=n_samples,
                                            k_denoise=k_denoise)
        td.set_seed(seed + k)
        samples_d = td.forecast_from_fitted(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price_d,
                                            horizons=daily_horizons, n_samples=n_samples,
                                            k_denoise=k_denoise)

        for wi, w_label in enumerate(HORIZON_LABELS):
            actual = actuals[wi]
            h_d = daily_horizons[wi]
            for model_name, samples in (("TSDiff-W", samples_w[wi + 1]),
                                        ("TSDiff-D", samples_d[h_d])):
                point = float(np.mean(samples))
                lo = float(np.quantile(samples, 0.025))
                hi = float(np.quantile(samples, 0.975))
                records.append({
                    "asset": asset, "horizon": w_label, "model": model_name,
                    "origin": k, "origin_date": str(origin_date.date()),
                    "target_date": str(target_dates[wi].date()),
                    "daily_steps": h_d if model_name == "TSDiff-D" else None,
                    "actual": actual, "point": point, "lower": lo, "upper": hi,
                    "crps": crps_empirical(samples, actual),
                    "in_interval": bool(lo <= actual <= hi),
                })
        print(f"[{asset}] origin {k + 1}/{n_origins} ({origin_date.date()}) done")

    return {
        "records": records,
        "n_origins": n_origins,
        "first_origin": str(first_origin_date.date()),
        "mu_w": mu_w, "sd_w": sd_w, "mu_d": mu_d, "sd_d": sd_d,
    }


def summarize(records: list) -> dict:
    """{asset: {horizon: {model: {RMSE, Cov95, CRPS, n_origins}}}}"""
    df = pd.DataFrame(records)
    out = {}
    for (asset, horizon, model), g in df.groupby(["asset", "horizon", "model"]):
        rmse = float(np.sqrt(np.mean((g["point"] - g["actual"]) ** 2)))
        cov95 = float(g["in_interval"].mean())
        crps = float(g["crps"].mean())
        out.setdefault(asset, {}).setdefault(horizon, {})[model] = {
            "RMSE": round(rmse, 4), "Cov95": round(cov95, 4), "CRPS": round(crps, 4),
            "n_origins": int(len(g)),
        }
    return out


def print_table(summary: dict) -> None:
    print(f"\n{'Actif':<6}{'Horizon':<8}{'Modele':<10}{'RMSE':>10}{'Cov95':>8}{'CRPS':>10}")
    print("-" * 52)
    for asset, by_h in summary.items():
        for horizon in HORIZON_LABELS:
            for model in ("TSDiff-W", "TSDiff-D"):
                m = by_h.get(horizon, {}).get(model)
                if not m:
                    continue
                print(f"{asset:<6}{horizon:<8}{model:<10}"
                      f"{m['RMSE']:>10.2f}{m['Cov95']:>8.2f}{m['CRPS']:>10.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--n-origins", type=int, default=DEFAULT_N_ORIGINS)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=max(50, td.N_SAMPLES))
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "weekly_headtohead_results.json"))
    args = p.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    all_records, meta = [], {}
    t0 = time.time()
    for asset in args.assets:
        result = run_pair(asset, ASSETS[asset], args.n_origins, args.epochs, args.seed,
                          args.n_samples, args.k_denoise, args.start, end)
        all_records.extend(result["records"])
        meta[asset] = {k: v for k, v in result.items() if k != "records"}
    elapsed = time.time() - t0

    summary = summarize(all_records)
    payload = {
        "config": {
            "n_origins": args.n_origins, "epochs": args.epochs, "seed": args.seed,
            "n_samples": args.n_samples, "k_denoise": args.k_denoise,
            "horizon_weekly": HORIZON_WEEKLY, "horizon_daily": HORIZON_DAILY,
            "assets": args.assets, "start": args.start, "end": end,
            "elapsed_s": round(elapsed, 1),
        },
        "meta": meta,
        "summary": summary,
        "records": all_records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")
    print_table(summary)


if __name__ == "__main__":
    main()
