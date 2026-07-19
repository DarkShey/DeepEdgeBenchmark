"""
backfill_weekly_predictions.py — load the already-produced TSDiff weekly results
(experiments/weekly_multiasset_results.json, experiments/weekly_headtohead_v2_results.json)
into validation/tracking.db as source='oos' rows, using the new frequence/horizon_type/
horizon_unit attributes (BRIEF_audit_combinaisons.md).

Loads TWO regimes:
  - Regime C (weekly native)      : TSDiff-W, all 5 assets, from weekly_multiasset_results.json.
  - Regime B (daily -> weekly)    : TSDiff-D, SPY+BTC only, from weekly_headtohead_v2_results.json
                                    (the only assets it was ever run for -- ETH/ZN/TLT never had
                                    a properly-tuned TSDiff-D run, by design, cf. brief history).

Deliberately mirrors sim_trades.build_oos_prediction_rows()'s OOS row shape and goes
through sim_trades.insert_oos_predictions() (not a hand-rolled INSERT) so this backfill
respects the exact same upsert / conflict-target guarantees as the daily OOS pipeline --
verdict_integrite/verdict_plausibilite/tc_id/created_at stay NULL, matching the
established convention for source='oos' rows (never computed for OOS, cf.
build_oos_prediction_rows docstring).

last_close is not stored directly in the weekly result JSONs -- recovered from the
RandomWalk sibling record's "point" field (RandomWalk's point IS last_price by
construction, cf. weekly_headtohead_v2.run_pair_v2).

Usage:
    python backfill_weekly_predictions.py            # loads both regimes, both files
    python backfill_weekly_predictions.py --dry-run   # print what would be inserted, no write
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from validation import sim_trades as st                        # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS           # noqa: E402

RUN_ID = "20260717-weekly-backfill"


def _last_close_lookup(records: list) -> dict:
    """{(asset, origin, horizon): last_close} from each combo's RandomWalk sibling."""
    out = {}
    for r in records:
        if r["model"] == "RandomWalk":
            out[(r["asset"], r["origin"], r["horizon"])] = r["point"]
    return out


def _rows_from_file(path: Path, model_filter: str, frequence: str) -> list:
    payload = json.loads(path.read_text())
    records = payload["records"]
    last_close = _last_close_lookup(records)

    rows = []
    for r in records:
        if r["model"] != model_filter:
            continue
        asset_short = r["asset"]
        ticker = ASSET_TICKERS[asset_short]
        week_n = int(r["horizon"][1:])   # "W1" -> 1
        key = (asset_short, r["origin"], r["horizon"])
        if key not in last_close:
            raise SystemExit(f"no RandomWalk sibling for {key} in {path} -- cannot derive last_close")

        rows.append({
            "run_id": RUN_ID, "model": "TSDiff", "asset": ticker, "horizon": week_n,
            "regime": "unknown", "cutoff_date": r["origin_date"], "target_date": r["target_date"],
            "last_close": float(last_close[key]), "y_pred": float(r["point"]),
            "y_lower": float(r["lower"]), "y_upper": float(r["upper"]),
            "y_true": float(r["actual"]), "source": "oos",
            "frequence": frequence, "horizon_type": "weekly", "horizon_unit": f"W+{week_n}",
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multiasset-file", default=str(Path(__file__).resolve().parent
                                                    / "weekly_multiasset_results.json"))
    p.add_argument("--v2-file", default=str(Path(__file__).resolve().parent
                                           / "weekly_headtohead_v2_results.json"))
    p.add_argument("--db-path", default="validation/tracking.db")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows_c = _rows_from_file(Path(args.multiasset_file), "TSDiff-W", "weekly")     # regime C
    rows_b = _rows_from_file(Path(args.v2_file), "TSDiff-D", "daily")              # regime B
    all_rows = rows_c + rows_b

    by_asset_freq = {}
    for r in all_rows:
        by_asset_freq.setdefault((r["asset"], r["frequence"]), 0)
        by_asset_freq[(r["asset"], r["frequence"])] += 1

    print(f"Regime C (weekly native, TSDiff-W): {len(rows_c)} rows from {args.multiasset_file}")
    print(f"Regime B (daily->weekly, TSDiff-D): {len(rows_b)} rows from {args.v2_file}")
    print(f"Total: {len(all_rows)} rows\n")
    for (asset, freq), n in sorted(by_asset_freq.items()):
        print(f"  {asset:<10} frequence={freq:<7} n={n}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    n = st.insert_oos_predictions(all_rows, db_path=args.db_path)
    print(f"\nInserted/updated {n} rows into {args.db_path} (source='oos', run_id={RUN_ID}).")


if __name__ == "__main__":
    main()
