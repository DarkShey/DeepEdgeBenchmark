import argparse
import csv
import json
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from benchmarks import config
from benchmarks.multi_horizon import MODEL_ADAPTERS
from benchmarks.regime_overlay import fit_predict_regime
from benchmarks.run_benchmark import download_full_data
from calibration.regime.assets import ASSETS

from validation import tracking_db as td
from validation import verdict_rules

D_TO_TRADING_DAYS = {1: 1, 7: 5}

DEFAULT_DB_PATH = "validation/tracking.db"
DEFAULT_RUN_DIR = "Run"


def export_run_bundle(run_id, all_records, visible_by_asset, args, run_dir_root=DEFAULT_RUN_DIR):
    date_str = datetime.now().strftime("%Y%m%d")
    run_folder = Path(run_dir_root) / f"{date_str}-run-complet"
    (run_folder / "training_data").mkdir(parents=True, exist_ok=True)
    (run_folder / "scripts").mkdir(parents=True, exist_ok=True)

    for ticker, series in visible_by_asset.items():
        series.rename("close").to_csv(run_folder / "training_data" / f"{ticker}.csv")

    if all_records:
        with open(run_folder / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    meta = {
        "run_id": run_id,
        "date": date_str,
        "holdout_days": args.holdout_days,
        "db_path": args.db_path,
        "assets": sorted(visible_by_asset.keys()),
        "models": sorted({r["model"] for r in all_records}),
        "horizons_D": sorted({r["horizon"] for r in all_records}),
        "cutoff_dates": {t: str(s.index[-1].date()) for t, s in visible_by_asset.items()},
        "n_test_cases": len(all_records),
        "source": "yfinance (daily, benchmarks.run_benchmark.download_full_data)",
    }
    with open(run_folder / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    this_dir = Path(__file__).resolve().parent
    for fname in ("tracking_db.py", "verdict_rules.py", "generate_test_cases.py"):
        src = this_dir / fname
        if src.exists():
            shutil.copy(src, run_folder / "scripts" / fname)

    print(f"[generate_test_cases] bundle Run exporte -> {run_folder}")
    return run_folder


def build_records(ticker, asset_class, model_name, adapter_fn, visible_close,
                  full_close, regime_tag, run_id, epochs, seed):
    trading_days = sorted(set(D_TO_TRADING_DAYS.values()))
    if model_name == "LSTM":
        raw = adapter_fn(visible_close, trading_days, epochs=epochs, seed=seed)
    else:
        raw = adapter_fn(visible_close, trading_days)

    cutoff_date = visible_close.index[-1].date()
    last_close = float(visible_close.iloc[-1])
    now = datetime.now().isoformat(timespec="seconds")
    n_visible = len(visible_close)

    records = []
    for d, h_days in D_TO_TRADING_DAYS.items():
        point, lo, hi = raw[h_days]
        target_date = cutoff_date + timedelta(days=d)

        future_idx = n_visible - 1 + h_days
        if future_idx < len(full_close):
            actual = float(full_close.iloc[future_idx])
            evaluated_at = now
        else:
            actual = None
            evaluated_at = None

        record = {
            "run_id": run_id,
            "tc_id": f"TC_{ticker}_D{d}",
            "model": model_name,
            "asset": ticker,
            "horizon": d,
            "cutoff_date": str(cutoff_date),
            "target_date": str(target_date),
            "regime": regime_tag,
            "last_close": last_close,
            "y_pred": float(point),
            "y_lower": float(lo),
            "y_upper": float(hi),
            "created_at": now,
            "actual": actual,
            "evaluated_at": evaluated_at,
        }
        record["verdict_integrite"] = verdict_rules.check_integrity(record)
        record["verdict_plausibilite"] = (
            verdict_rules.check_plausibility(record, asset_class, h_days)
            if record["verdict_integrite"] else 0
        )
        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser(description="Test cases de base")
    p.add_argument("--assets", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--holdout-days", type=int, default=7)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--no-run-export", action="store_true")
    args = p.parse_args()

    selected_assets = ASSETS
    if args.assets:
        wanted = {t.strip() for t in args.assets.split(",")}
        selected_assets = [a for a in ASSETS if a["ticker"] in wanted]

    selected_models = list(MODEL_ADAPTERS.items())
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        selected_models = [(n, f) for n, f in selected_models if n in wanted]

    if not selected_assets or not selected_models:
        raise SystemExit("Aucun actif ou aucun modele selectionne.")

    run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    data_end = config.DATA_END or datetime.today().strftime("%Y-%m-%d")
    print(f"[generate_test_cases] run_id={run_id} fenetre {config.DATA_START} -> {data_end}")

    inserted = duplicates = errors = 0
    all_records = []
    visible_by_asset = {}
    for asset in selected_assets:
        ticker, asset_class = asset["ticker"], asset["asset_class"]
        print(f"\n[generate_test_cases] {ticker} : telechargement ...")
        try:
            full_data = download_full_data(ticker, config.DATA_START, data_end)
        except SystemExit as exc:
            print(f"  ECHEC telechargement : {exc}")
            errors += len(selected_models) * len(D_TO_TRADING_DAYS)
            continue

        if args.holdout_days > 0:
            cutoff_limit = full_data.index[-1] - timedelta(days=args.holdout_days)
            visible_data = full_data[full_data.index <= cutoff_limit]
        else:
            visible_data = full_data
        if visible_data.empty:
            print(f"  ECHEC : holdout trop grand pour {ticker}")
            errors += len(selected_models) * len(D_TO_TRADING_DAYS)
            continue
        visible_close, full_close = visible_data["Close"], full_data["Close"]
        visible_by_asset[ticker] = visible_close
        print(f"  {len(full_data)} jours -> {len(visible_data)} visibles "
              f"(cutoff {visible_data.index[-1].date()})")

        try:
            regime_state = fit_predict_regime(visible_data, visible_data.index[-1])
            regime_tag = regime_state.dominant_regime()
        except Exception as exc:
            print(f"  regime indisponible ({exc}) -> regime='unknown'")
            regime_tag = "unknown"

        for model_name, adapter_fn in selected_models:
            try:
                records = build_records(ticker, asset_class, model_name, adapter_fn,
                                        visible_close, full_close, regime_tag, run_id,
                                        args.epochs, args.seed)
            except Exception:
                print(f"  {model_name:<12} ECHEC generation prediction :")
                traceback.print_exc()
                errors += len(D_TO_TRADING_DAYS)
                continue

            all_records.extend(records)
            for record in records:
                try:
                    if td.save_prediction(record, db_path=args.db_path):
                        inserted += 1
                    else:
                        duplicates += 1
                except ValueError as exc:
                    print(f"  {model_name:<12} {record['tc_id']} REJETE : {exc}")
                    errors += 1
            print(f"  {model_name:<12} ok -- {len(records)} test cases traites")

    print(f"\n[generate_test_cases] termine -> {inserted} inseres, "
          f"{duplicates} doublons ignores, {errors} echecs. DB : {args.db_path}")

    if not args.no_run_export and all_records:
        export_run_bundle(run_id, all_records, visible_by_asset, args, run_dir_root=args.run_dir)


if __name__ == "__main__":
    main()
