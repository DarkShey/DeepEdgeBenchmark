"""
weekly_nsdiff_production.py — recurring NsDiff-W (weekly-native, via the daily
NsDiff module fed weekly data -- same construction as TSDiff-prod) forward
generation. Mirror of weekly_tsdiff_production.py, per
BRIEF_nsdiff_weekly_parite_et_compa.md §4.1 -- NsDiff-W is already evaluated
in the duel (duel_backtest_nsdiff.json, nsdiff_weekly.fit_weekly, seq_len=26)
but was never wired into the production/dashboard chain, where TSDiff-W was
until now the only weekly diffusion model.

Swap vs weekly_tsdiff_production.py: `td.fit_tsdiff`/`td.forecast_from_fitted`
-> `nm.fit_nsdiff`/`nm.forecast_from_fitted` (nsdiff_model.py) -- signatures
are identical (`nm.forecast_from_fitted(model, hist_window, mu, sd, last_price,
horizons=, n_samples=, **_ignored)`, the `k_denoise` kwarg absorbed by
`**_ignored` since NsDiff's diffusion step count is fixed at fit time, not at
sampling time). This is `nm.fit_nsdiff` (seq_len=30, the SAME daily-module-fed-
weekly construction TSDiff-prod uses), NOT `nsdiff_weekly.fit_weekly`
(seq_len=26, the module used by the duel) -- a DELIBERATE definition choice
(brief §4, "décision de définition à trancher"): mirrors TSDiff-prod line-for-
line for an apples-to-apples production comparison, at the cost of a declared
mismatch with the duel's own NsDiff-W definition (seq_len 30 here vs 26 there)
-- dette déclarée, cf. brief §5 (aligning seq_len between prod and duel is
future work, not done today).

Epochs: NsDiff has no entry in epoch_sweep_results.json (that file only has
"<asset>|TSDiff-W" keys) -- brief §4.1 decides (a) for today: a fixed,
declared epoch budget (NSDIFF_EPOCHS_W=40, same value the duel's original
non-swept NsDiff arm used, cf. duel_backtest.py's own NSDIFF_EPOCHS_W
constant) rather than (b) adding a validation-sweep entry to
epoch_sweep_results.json (dette déclarée, brief §5).

Insertion: validation/tracking_db.save_prediction(), source='live' -- exactly
like weekly_tsdiff_production.py (same idempotent INSERT OR IGNORE on
UNIQUE(tc_id, model, cutoff_date)). tc_id=f"TC_{ticker}_W{h}" is IDENTICAL to
TSDiff's own tc_id for the same (ticker, horizon) -- no collision because the
UNIQUE constraint also keys on `model`, and register_test_case's upsert on
tc_id alone is asset/horizon-only (model-agnostic), so re-registering the same
tc_id with the same (asset, horizon) is a no-op, not a conflict (verified
against validation/tracking_db.py: register_test_case only touches
`test_cases`, `save_prediction` scopes uniqueness by (tc_id, model,
cutoff_date) on `predictions`).

Usage:
    python experiments/weekly_nsdiff_production.py                    # all 5 assets
    python experiments/weekly_nsdiff_production.py --assets SPY BTC --dry-run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import nsdiff_model as nm                                            # noqa: E402
from weekly_headtohead import ASSETS, build_weekly, HORIZON_WEEKLY, standardized_returns  # noqa: E402
from validation import tracking_db as tdb                            # noqa: E402
from validation import verdict_rules                                  # noqa: E402

try:
    from calibration.regime.assets import ASSETS as _ASSET_REGISTRY
    TICKER_ASSET_CLASS = {a["ticker"]: a["asset_class"] for a in _ASSET_REGISTRY}
except Exception:
    TICKER_ASSET_CLASS = {"BTC-USD": "crypto", "ETH-USD": "crypto", "SPY": "index",
                          "ZN=F": "bond", "TLT": "bond"}

MODEL_NAME = "NsDiff"
TRADING_DAYS_PER_WEEK = 5
DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 50
# Fixed, declared epoch budget (brief §4.1 option (a) -- no epoch_sweep_results.json
# entry exists for NsDiff-W; option (b), a validation sweep entry, is dette déclarée
# per brief §5). Same value as the duel's original non-swept NsDiff arm
# (duel_backtest.NSDIFF_EPOCHS_W).
NSDIFF_EPOCHS_W = 40


def generate_asset(asset: str, ticker: str, epochs: int, seed: int, n_samples: int,
                   start: str, end: str) -> list:
    """Fit NsDiff-W on ALL available weekly history and forecast W+1/2/3 from
    today (the most recent W-FRI close). Returns `save_prediction`-ready records
    (verdict_integrite/verdict_plausibilite already computed)."""
    print(f"[{asset}] downloading {ticker} ({start} -> {end}) ...")
    daily = nm.fetch_data(ticker, start, end)
    weekly, weekly_dates = build_weekly(daily)

    nm.set_seed(seed)
    model, mu, sd = nm.fit_nsdiff(weekly, horizon=HORIZON_WEEKLY, epochs=epochs)
    weekly_z = standardized_returns(weekly, mu, sd)

    origin_date = weekly_dates.iloc[-1]
    last_price = float(weekly.iloc[-1])
    samples_by_h = nm.forecast_from_fitted(model, weekly_z, mu, sd, last_price,
                                           horizons=[1, 2, 3], n_samples=n_samples)

    asset_class = TICKER_ASSET_CLASS.get(ticker, "")
    now = datetime.now()
    records = []
    for h in (1, 2, 3):
        samples = samples_by_h[h]
        point = float(np.mean(samples))
        lo, hi = (float(q) for q in np.quantile(samples, [0.025, 0.975]))
        target_date = origin_date + pd.Timedelta(weeks=h)
        record = {
            "run_id": f"weekly-nsdiff-{origin_date.date()}",
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
    p.add_argument("--epochs", type=int, default=NSDIFF_EPOCHS_W,
                   help="fixed epoch budget (declared, brief §4.1 option (a) -- "
                       "no epoch_sweep_results.json entry for NsDiff-W)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    all_records = []
    for asset in args.assets:
        records = generate_asset(asset, ASSETS[asset], args.epochs, args.seed, args.n_samples,
                                 args.start, end)
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
