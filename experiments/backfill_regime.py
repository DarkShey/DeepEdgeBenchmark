"""
backfill_regime.py — retroactively compute the market regime for existing OOS rows
in validation/tracking.db (currently regime='unknown' for all 4801 OOS rows, cf.
build_oos_prediction_rows() which never sets a real regime), via
calibration.regime.regime_agent.RegimeAgent.predict_history() -- one fit + one
batch classification per asset, not one fit per row (RegimeAgent.fit() runs a full
GARCH fit; doing that per-row would be ~4800x more expensive for no benefit).

Caveat (inherited from RegimeAgent.predict_history()'s own docstring, "pas de
contrainte point-in-time ici -- réservé à la visualisation"): this is NOT a
point-in-time re-simulation. The HMM/BOCPD are fit once per asset on its FULL
available history, then every date is classified against that single fit -- a date
from years ago is classified with the benefit of the whole later history informing
the model's regime definitions. Acceptable for retrospective segmentation of
already-realised backtest predictions (which is exactly what this backfill is for),
not a re-run of what the regime engine would have said in real time back then.

Usage:
    python backfill_regime.py                 # all assets present in OOS rows
    python backfill_regime.py --dry-run        # print counts, no write
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import download_full_data          # noqa: E402
from calibration.regime.regime_agent import RegimeAgent           # noqa: E402


def backfill_asset(conn: sqlite3.Connection, asset: str, dry_run: bool) -> dict:
    rows = conn.execute(
        "SELECT DISTINCT cutoff_date FROM predictions WHERE source='oos' AND asset=?",
        (asset,),
    ).fetchall()
    cutoff_dates = sorted(r[0] for r in rows)
    if not cutoff_dates:
        return {"asset": asset, "n_dates": 0, "n_updated": 0, "n_missing_in_history": 0}

    start, end = cutoff_dates[0], cutoff_dates[-1]
    ohlcv = download_full_data(asset, start="2010-01-01", end=end)   # ample warm-up before `start`

    agent = RegimeAgent()
    agent.fit(ohlcv, train_end=ohlcv.index[-1].strftime("%Y-%m-%d"))
    hist = agent.predict_history(ohlcv)
    regime_by_date = {ts.strftime("%Y-%m-%d"): r for ts, r in hist["regime"].items()}

    n_updated, n_missing = 0, 0
    for cutoff in cutoff_dates:
        regime = regime_by_date.get(cutoff)
        if regime is None:
            n_missing += 1
            continue
        if not dry_run:
            conn.execute(
                "UPDATE predictions SET regime=? WHERE source='oos' AND asset=? AND cutoff_date=?",
                (regime, asset, cutoff),
            )
        n_updated += 1

    return {"asset": asset, "n_dates": len(cutoff_dates), "n_updated": n_updated,
            "n_missing_in_history": n_missing}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        assets = [r[0] for r in conn.execute(
            "SELECT DISTINCT asset FROM predictions WHERE source='oos'")]
        print(f"Assets to backfill: {assets}\n")

        results = []
        for asset in assets:
            print(f"[{asset}] fitting RegimeAgent + classifying history ...")
            res = backfill_asset(conn, asset, args.dry_run)
            results.append(res)
            print(f"  {res}")

        if args.dry_run:
            print("\n--dry-run: nothing written.")
        else:
            conn.commit()
            print("\nCommitted.")

        n_before = conn.execute(
            "SELECT regime, COUNT(*) FROM predictions WHERE source='oos' GROUP BY regime").fetchall()
        print(f"\nregime distribution (source='oos') now: {n_before}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
