"""
weekly_tsdiff_production.py — recurring TSDiff regime-C (weekly-native) forward
generation, per BRIEF_weekly_pipeline_integration.md Couche B.

Extracts the fit+forecast logic already validated in weekly_headtohead_v2.run_pair_v2
/ weekly_multiasset.py --phase final into a PRODUCTION function: instead of scoring
30 held-out test origins, fits TSDiff-W on the ENTIRE available weekly history up to
today and forecasts W+1/W+2/W+3 from the LAST origin (today) -- no test/validation
split here, this is forward generation, not backtesting.

Guardrail (anti-lookahead): fit on the full `weekly` series as returned by
build_weekly (everything realised up to and including the origin), exactly the same
construction as run_pair_v2's per-origin slicing, just at the single most-recent
origin instead of 30 historical ones -- nothing beyond the current close is ever
used.

epochs* per asset come from experiments/epoch_sweep_results.json's "selected_epochs"
(argmin CRPS on validation, already chosen -- NOT re-swept here, cf.
BRIEF_weekly_pipeline_integration.md §4 step 4).

Insertion: validation/tracking_db.save_prediction() -- the daily "live" path used by
model_artifacts/pipeline.py::_save_business_predictions -- NOT
sim_trades.insert_oos_predictions: that upsert's ON CONFLICT target is scoped
`WHERE source='oos'` (validation/tracking_db.py's idx_predictions_oos_unique), so a
repeated source='live' insert would NOT be deduplicated by it and would silently
duplicate on every weekly re-run. save_prediction's INSERT OR IGNORE on
UNIQUE(tc_id, model, cutoff_date) is the correct idempotent path for recurring live
generation -- the same guarantee daily live rows already get. tc_id=f"TC_{ticker}_W{h}"
is auto-registered by save_prediction (register_test_case, upsert on tc_id) before
the insert -- no orphan risk, verified against validation/tracking_db.py (no other
query joins test_cases assuming it is 100% daily).

verdict_plausibilite reuses validation/verdict_rules.check_plausibility as-is
(already generic: scales by sqrt(horizon_trading_days)) -- passed
TRADING_DAYS_PER_WEEK * h as the trading-day equivalent of a week horizon, no change
to verdict_rules.py needed.

Usage:
    python experiments/weekly_tsdiff_production.py                    # all 5 assets
    python experiments/weekly_tsdiff_production.py --assets SPY BTC --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import tsdiff_model as td                                            # noqa: E402
from weekly_headtohead import ASSETS, build_weekly, HORIZON_WEEKLY, standardized_returns  # noqa: E402
from validation import tracking_db as tdb                            # noqa: E402
from validation import verdict_rules                                  # noqa: E402

try:
    from calibration.regime.assets import ASSETS as _ASSET_REGISTRY
    TICKER_ASSET_CLASS = {a["ticker"]: a["asset_class"] for a in _ASSET_REGISTRY}
except Exception:
    TICKER_ASSET_CLASS = {"BTC-USD": "crypto", "ETH-USD": "crypto", "SPY": "index",
                          "ZN=F": "bond", "TLT": "bond"}

MODEL_NAME = "TSDiff"
TRADING_DAYS_PER_WEEK = 5
DEFAULT_SWEEP_FILE = ROOT / "experiments" / "epoch_sweep_results.json"
DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50
DEFAULT_K_DENOISE = td.K_DENOISE


def load_epochs(sweep_file=DEFAULT_SWEEP_FILE) -> dict:
    """{asset_short: epochs*} -- already-selected TSDiff-W epoch per asset
    (argmin CRPS on the validation block, cf. epoch_sweep.py /
    weekly_multiasset.py --phase sweep). Raises if an asset was never swept."""
    selected = json.loads(Path(sweep_file).read_text())["selected_epochs"]
    out = {}
    for asset in ASSETS:
        key = f"{asset}|TSDiff-W"
        if key not in selected:
            raise SystemExit(f"No selected epoch* for {key} in {sweep_file} -- "
                            f"run epoch_sweep.py / weekly_multiasset.py --phase sweep first.")
        out[asset] = selected[key]["epochs"]
    return out


def generate_asset(asset: str, ticker: str, epochs: int, seed: int, n_samples: int,
                   k_denoise: int, start: str, end: str) -> list:
    """Fit TSDiff-W on ALL available weekly history and forecast W+1/2/3 from
    today (the most recent W-FRI close). Returns `save_prediction`-ready records
    (verdict_integrite/verdict_plausibilite already computed)."""
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    daily = td.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)

    td.set_seed(seed)
    model, mu, sd = td.fit_tsdiff(weekly, horizon=HORIZON_WEEKLY, epochs=epochs)
    weekly_z = standardized_returns(weekly, mu, sd)

    origin_date = weekly_dates.iloc[-1]
    last_price = float(weekly.iloc[-1])
    samples_by_h = td.forecast_from_fitted(model, weekly_z, mu, sd, last_price,
                                           horizons=[1, 2, 3], n_samples=n_samples,
                                           k_denoise=k_denoise)

    asset_class = TICKER_ASSET_CLASS.get(ticker, "")
    now = datetime.now()
    records = []
    for h in (1, 2, 3):
        samples = samples_by_h[h]
        point = float(np.mean(samples))
        lo, hi = (float(q) for q in np.quantile(samples, [0.025, 0.975]))
        target_date = origin_date + pd.Timedelta(weeks=h)
        record = {
            "run_id": f"weekly-tsdiff-{origin_date.date()}",
            "tc_id": f"TC_{ticker}_W{h}",
            "model": MODEL_NAME, "asset": ticker, "horizon": h,
            "cutoff_date": str(origin_date.date()), "target_date": str(target_date.date()),
            "regime": "unknown", "last_close": last_price,
            "y_pred": point, "y_lower": lo, "y_upper": hi,
            "frequence": "weekly", "horizon_type": "weekly", "horizon_unit": f"W+{h}",
            "created_at": now.isoformat(timespec="seconds"),
        }
        record["verdict_integrite"] = verdict_rules.check_integrity(record)
        record["verdict_plausibilite"] = (
            verdict_rules.check_plausibility(record, asset_class, TRADING_DAYS_PER_WEEK * h)
            if record["verdict_integrite"] else 0
        )
        records.append(record)
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--sweep-file", default=str(DEFAULT_SWEEP_FILE))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--k-denoise", type=int, default=DEFAULT_K_DENOISE)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    epochs_by_asset = load_epochs(args.sweep_file)

    all_records = []
    for asset in args.assets:
        ep = epochs_by_asset[asset]
        records = generate_asset(asset, ASSETS[asset], ep, args.seed, args.n_samples,
                                 args.k_denoise, args.start, end)
        all_records.extend(records)
        for r in records:
            print(f"  [{asset}] {r['horizon_unit']} cutoff={r['cutoff_date']} "
                  f"target={r['target_date']} pred={r['y_pred']:.4f} "
                  f"[{r['y_lower']:.4f}, {r['y_upper']:.4f}] "
                  f"integrite={r['verdict_integrite']} plausibilite={r['verdict_plausibilite']}")

    if args.dry_run:
        print(f"\n--dry-run: {len(all_records)} record(s) not written.")
        return

    n_inserted = 0
    for record in all_records:
        if tdb.save_prediction(record, db_path=args.db_path):
            n_inserted += 1
    print(f"\nInserted {n_inserted}/{len(all_records)} new rows into {args.db_path} "
         f"(source='live', model='{MODEL_NAME}', frequence='weekly'). "
         f"Rows already present for the same (tc_id, model, cutoff_date) are ignored "
         f"(idempotent -- see save_prediction).")


if __name__ == "__main__":
    main()
