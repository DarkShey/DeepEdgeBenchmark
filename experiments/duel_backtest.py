"""
duel_backtest.py — the duel itself: 6 models (TSDiff, ARIMA-GARCH, SARIMA,
Prophet, LSTM, Naive), one shared set of origins, identical re-estimation
rule, identical `m` sample count, fair CRPS, paired tests, Holm correction and
Model Confidence Set per (asset, horizon). BRIEF_unification_protocole_duel.md
§3 (the deliverable) — this is the assembly the brief's §1 says is missing;
every brick it calls (duel_origins, duel_sampling_adapters, crps_fair,
duel_pairwise_tests, mcs) is reused as-is, nothing here reimplements them.

Protocol, enforced by construction (not by convention):
  - Origins/target dates: duel_origins.build_common_origins (rolling origin,
    purge structural + embargo for W1-W3 overlap, three_way_split-anchored).
  - Re-estimation: FROZEN AT T0 for all 6 models (duel_origins.py's own
    declared choice, preserved here) -- every adapter's `fit_*_state` is
    called exactly once per asset, on data <= T0; every test origin only
    advances each model's conditioning state with realised data.
  - Hyperparameters: TSDiff-W's epoch count AND the ARIMA-GARCH innovation
    spec (normal/t/gjr-t, BRIEF_baselines_fortes.md §2.1-2.4) are BOTH
    selected on the VALIDATION block ONLY (verrou E1) -- TSDiff-W by reusing
    epoch_sweep.py's own incremental-checkpoint sweep (`_sweep_one_model`/
    `select_epochs`) unchanged, GARCH by `select_garch_spec` below (argmin
    mean fair CRPS over val_pos, same pattern) -- both restricted to MY
    origins (which already embed the embargo). The TEST block (test_pos) is
    never touched until final scoring below.
  - Sampling: every model draws `m` (default 500) trajectories per origin via
    duel_sampling_adapters' 5 classic adapters + tsdiff_model.forecast_from_fitted
    for TSDiff -- never a Gaussian/log-normal reconstruction (audit reserve N1).
  - Scoring: crps_fair (Ferro 2014, experiments/crps_metrics.py), `m` identical
    across all 6 models by construction (same `--m-samples` passed everywhere).
  - Tests: duel_pairwise_tests.pairwise_crps_tests (dm_hac_test + block
    bootstrap, per pair x asset x horizon), Holm-corrected across the full
    grid per pair (duel_pairwise_tests.holm_correct_grid), pooled_pair_verdict
    (MASE-scaled, asset-class-fused) per pair x horizon, clark_west_test of
    every non-naive model against Naive on POINT forecasts, and a fresh
    Model Confidence Set + SPA-vs-GARCH(1,1) per (asset, horizon) (mcs.py).

Usage:
    python duel_backtest.py                                   # all 5 assets, m=500, n_test=30
    python duel_backtest.py --assets SPY BTC --n-test 8 --m-samples 100 --n-boot 500
                                                                # fast smoke run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must run before EVERYTHING else that might import statsmodels/arima before
# tensorflow (models/conftest.py, experiments/conftest.py): duel_sampling_
# adapters.py imports arima_model/sarima_model eagerly and only imports
# lstm_model (hence tensorflow) lazily inside fit_lstm_state() -- if TF's
# first op happens without single-threaded config already set, the first
# model.fit() of the process deadlocks indefinitely (confirmed, 0% CPU
# afterwards, not a slow run). Those conftest.py guards only fire under
# pytest; this script runs standalone, so the guard is duplicated here.
os.environ.setdefault("PYTHONHASHSEED", "0")
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tsdiff_model as td                                        # noqa: E402
import epoch_sweep as es                                         # noqa: E402
import duel_sampling_adapters as dsa                              # noqa: E402
from weekly_headtohead import ASSETS, HORIZON_WEEKLY, build_weekly, standardized_returns  # noqa: E402
from duel_origins import build_common_origins, HORIZON_MAX        # noqa: E402
from crps_metrics import crps_empirical, crps_fair                # noqa: E402
from duel_pairwise_tests import (                                 # noqa: E402
    pairwise_crps_tests, holm_correct_grid, pooled_pair_verdict, clark_west_test,
)
from pooled_analysis import compute_asset_scales                  # noqa: E402
from mcs import model_confidence_set, spa_test                    # noqa: E402

MODELS = ("TSDiff", "ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive")
CLASSICS = ("ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive")
HORIZONS = ("W1", "W2", "W3")
H_BY_HORIZON = {"W1": 1, "W2": 2, "W3": 3}
BENCHMARK_FOR_SPA = "ARIMA-GARCH"
PAIRS = tuple((("TSDiff", c) for c in CLASSICS))


def select_garch_spec(asset_code: str, train_daily: pd.Series, daily: pd.Series,
                      weekly: pd.Series, weekly_dates: pd.Series, val_pos: list,
                      T0_daily_pos: int, args) -> tuple:
    """GARCH innovation-law/asymmetry spec selection, VALIDATION BLOCK ONLY
    (BRIEF_baselines_fortes.md §2.4, verrou E1) -- mirrors the TSDiff-W epoch
    sweep just above: each candidate spec is fit ONCE on data <= T0, scored by
    mean fair CRPS over `val_pos` (never `test_pos`), argmin wins. Returns
    (best_spec, {spec: mean_crps_val}) so the choice is auditable, not just
    asserted."""
    scores = {}
    for spec in args.garch_spec_candidates:
        state = dsa.fit_garch_state(train_daily, spec=spec)
        crps_vals = []
        for k, m_pos in enumerate(val_pos):
            _, daily_pos, target_dates, daily_horizons = es.week_targets(weekly_dates, daily, m_pos)
            last_price = float(weekly.iloc[m_pos])
            actuals = [float(weekly.iloc[m_pos + h]) for h in (1, 2, 3)]
            new_daily_prices = daily.iloc[T0_daily_pos:daily_pos + 1].values.astype(float)
            realized_returns_pct = np.diff(np.log(new_daily_prices)) * 100.0
            samples_g = dsa.garch_trajectory_samples(state, realized_returns_pct, last_price,
                                                     horizons=daily_horizons, m=args.m_samples,
                                                     seed=args.seed + k)
            for wi in range(3):
                crps_vals.append(crps_fair(samples_g[daily_horizons[wi]], actuals[wi]))
        scores[spec] = float(np.mean(crps_vals))
    best_spec = min(scores, key=scores.get)
    print(f"[{asset_code}] GARCH spec*={best_spec} (val-selected, candidates={scores})")
    return best_spec, scores


def run_asset_duel(asset_code: str, ticker: str, args) -> tuple:
    print(f"[{asset_code}] downloading {ticker} ({args.start} -> {args.end}) ...")
    daily = td.fetch_data(ticker, args.start, args.end)
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = build_common_origins(
        weekly, args.n_val, args.n_test, embargo=args.embargo)
    T0_date = weekly_dates.iloc[train_end_pos]
    train_daily = daily.loc[:T0_date]
    train_weekly = weekly.iloc[:train_end_pos + 1]
    print(f"[{asset_code}] train <= {T0_date.date()} | validation {len(val_pos)} origins | "
          f"test {weekly_dates.iloc[test_pos[0]].date()} -> {weekly_dates.iloc[test_pos[-1]].date()} "
          f"({len(test_pos)})")

    # ── TSDiff-W epoch selection, validation block ONLY (verrou E1) ────────
    # Reuses epoch_sweep._sweep_one_model/select_epochs unchanged, restricted
    # to THIS asset's own (embargoed) val_pos -- the test block is untouched.
    t0 = time.time()
    sweep_records = es._sweep_one_model(
        asset_code, "TSDiff-W", train_weekly, HORIZON_WEEKLY, weekly, args.seed,
        args.tsdiff_epoch_candidates, val_pos, weekly, weekly_dates, daily,
        args.tsdiff_hp_samples, args.k_denoise)
    epochs_w = es.select_epochs(sweep_records)[f"{asset_code}|TSDiff-W"]["epochs"]
    print(f"[{asset_code}] TSDiff-W epochs*={epochs_w} (val-selected, {time.time() - t0:.0f}s)")

    # ── Fit all 6 models EXACTLY ONCE on data <= T0 (frozen-at-T0 rule) ─────
    t0 = time.time()
    td.set_seed(args.seed)
    model_w, mu_w, sd_w = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs_w)
    weekly_z = standardized_returns(weekly, mu_w, sd_w)
    weekly_r_raw = td._log_returns(weekly.values.astype(float))
    print(f"[{asset_code}] TSDiff-W fit in {time.time() - t0:.0f}s")

    T0_daily_pos = int(daily.index.get_loc(T0_date))
    t0 = time.time()
    garch_spec, garch_spec_scores = select_garch_spec(
        asset_code, train_daily, daily, weekly, weekly_dates, val_pos, T0_daily_pos, args)
    garch_state = dsa.fit_garch_state(train_daily, spec=garch_spec)
    print(f"[{asset_code}] ARIMA-GARCH ({garch_spec}) fit in {time.time() - t0:.0f}s")

    t0 = time.time()
    sarima_state = dsa.fit_sarima_state(train_daily)
    print(f"[{asset_code}] SARIMA fit in {time.time() - t0:.0f}s")

    t0 = time.time()
    prophet_state = dsa.fit_prophet_state(train_daily, m=args.m_samples)
    print(f"[{asset_code}] Prophet fit in {time.time() - t0:.0f}s")

    t0 = time.time()
    lstm_state = dsa.fit_lstm_state(train_daily, seed=args.seed)
    print(f"[{asset_code}] LSTM fit in {time.time() - t0:.0f}s")

    seq_len = lstm_state["seq_len"]

    records = []
    t0 = time.time()
    for k, m_pos in enumerate(test_pos):
        origin_date, daily_pos, target_dates, daily_horizons = es.week_targets(weekly_dates, daily, m_pos)
        last_price = float(weekly.iloc[m_pos])
        actuals = [float(weekly.iloc[m_pos + h]) for h in (1, 2, 3)]
        hist_weekly_returns = weekly_r_raw[:m_pos]

        new_daily_prices = daily.iloc[T0_daily_pos:daily_pos + 1].values.astype(float)
        realized_returns_pct = np.diff(np.log(new_daily_prices)) * 100.0
        realized_prices_since_fit = new_daily_prices[1:]
        tail_prices = daily.iloc[:daily_pos + 1].values[-seq_len:]

        seed_k = args.seed + k
        td.set_seed(seed_k)
        samples_w = td.forecast_from_fitted(model_w, weekly_z[:m_pos], mu_w, sd_w, last_price,
                                            horizons=[1, 2, 3], n_samples=args.m_samples,
                                            k_denoise=args.k_denoise)
        samples_g = dsa.garch_trajectory_samples(garch_state, realized_returns_pct, last_price,
                                                 horizons=daily_horizons, m=args.m_samples, seed=seed_k)
        samples_s = dsa.sarima_trajectory_samples(sarima_state, realized_prices_since_fit,
                                                  horizons=daily_horizons, m=args.m_samples, seed=seed_k)
        samples_p = dsa.prophet_trajectory_samples(prophet_state, target_dates,
                                                    horizons=[1, 2, 3], seed=seed_k)
        samples_l = dsa.lstm_trajectory_samples(lstm_state, tail_prices, horizons=daily_horizons,
                                                m=args.m_samples, seed=seed_k)
        samples_n = dsa.naive_trajectory_samples(hist_weekly_returns, last_price,
                                                 horizons=[1, 2, 3], m=args.m_samples, seed=seed_k)

        for wi, h_label in enumerate(HORIZONS):
            actual = actuals[wi]
            h_d = daily_horizons[wi]
            per_model_samples = {
                "TSDiff": samples_w[wi + 1], "ARIMA-GARCH": samples_g[h_d],
                "SARIMA": samples_s[h_d], "Prophet": samples_p[wi + 1],
                "LSTM": samples_l[h_d], "Naive": samples_n[wi + 1],
            }
            for model_name, samples in per_model_samples.items():
                records.append({
                    "asset": ticker, "asset_code": asset_code, "horizon": h_label,
                    "model": model_name, "origin": k, "origin_date": str(origin_date.date()),
                    "target_date": str(target_dates[wi].date()), "actual": actual,
                    "point": float(np.mean(samples)),
                    "crps": crps_fair(samples, actual),
                    "crps_empirical": crps_empirical(samples, actual),
                })
        if (k + 1) % max(1, len(test_pos) // 5) == 0 or k == len(test_pos) - 1:
            print(f"[{asset_code}] test origin {k + 1}/{len(test_pos)} "
                 f"({origin_date.date()}) done ({time.time() - t0:.0f}s elapsed)")

    meta = {
        "train_end": str(T0_date.date()), "epochs_tsdiff_w": epochs_w,
        "garch_spec": garch_spec, "garch_spec_val_scores": garch_spec_scores,
        "val_origins": [str(weekly_dates.iloc[m].date()) for m in val_pos],
        "test_origins": [str(weekly_dates.iloc[m].date()) for m in test_pos],
    }
    return records, meta


def build_grid_analysis(df: pd.DataFrame, all_meta: dict, args) -> dict:
    tickers = sorted(df["asset"].unique())

    cell_results = pairwise_crps_tests(df, list(PAIRS), H_BY_HORIZON)
    holm_by_pair = {}
    for a, b in PAIRS:
        pair_label = f"{a} vs {b}"
        adj = holm_correct_grid(cell_results, tickers, list(HORIZONS), pair_label)
        holm_by_pair[pair_label] = adj
        for key, p_holm in adj.items():
            cell_results[key]["p_value_bootstrap_holm"] = p_holm
            cell_results[key]["significant_after_holm"] = bool(p_holm < 0.05)

    # No-lookahead scale window: end at the EARLIEST first-test-origin date
    # across every asset actually run (conservative -- guarantees no asset's
    # in-sample scale window peeks into ITS OWN test block).
    earliest_test_start = min(meta["test_origins"][0] for meta in all_meta.values())
    scales = compute_asset_scales(args.scale_start, earliest_test_start)

    pooled = {}
    for a, b in PAIRS:
        pooled[f"{a} vs {b}"] = {}
        for h_label, h_val in H_BY_HORIZON.items():
            df_h = df[df["horizon"] == h_label]
            pooled[f"{a} vs {b}"][h_label] = pooled_pair_verdict(df_h, (a, b), scales, h=h_val)

    cw_results = {}
    for ticker in tickers:
        for h_label, h_val in H_BY_HORIZON.items():
            sub = df[(df["asset"] == ticker) & (df["horizon"] == h_label)]
            piv_actual = sub.pivot_table(index="origin", columns="model", values="actual").sort_index()
            piv_point = sub.pivot_table(index="origin", columns="model", values="point").sort_index()
            if "Naive" not in piv_point.columns:
                continue
            naive_point = piv_point["Naive"].values
            actual_vals = piv_actual.iloc[:, 0].values
            for model_name in MODELS:
                if model_name == "Naive" or model_name not in piv_point.columns:
                    continue
                cw_results[f"{ticker}|{h_label}|{model_name} vs Naive"] = clark_west_test(
                    actual_vals, naive_point, piv_point[model_name].values, h=h_val)

    mcs_results, spa_results = {}, {}
    for ticker in tickers:
        for h_label in HORIZONS:
            sub = df[(df["asset"] == ticker) & (df["horizon"] == h_label)]
            piv = sub.pivot_table(index="origin", columns="model", values="crps").sort_index().dropna()
            key = f"{ticker}|{h_label}"
            if len(piv) < 8 or piv.shape[1] < 2:
                mcs_results[key] = {"status": "insufficient_data", "n": int(len(piv))}
                spa_results[key] = {"status": "insufficient_data", "n": int(len(piv))}
                continue
            mcs_results[key] = model_confidence_set(piv, alpha=0.05, block_length=3,
                                                     n_boot=args.n_boot, seed=args.seed)
            if BENCHMARK_FOR_SPA in piv.columns:
                bench = piv[BENCHMARK_FOR_SPA].values
                others = {m: piv[m].values for m in piv.columns if m != BENCHMARK_FOR_SPA}
                spa_results[key] = spa_test(bench, others, block_length=3,
                                            n_boot=args.n_boot, seed=args.seed)
            else:
                spa_results[key] = {"status": "benchmark_missing"}

    crps_summary = (df.groupby(["asset", "horizon", "model"])["crps"].mean()
                    .reset_index().to_dict(orient="records"))

    return {
        "asset_scales_mase": scales, "scale_window": [args.scale_start, earliest_test_start],
        "crps_fair_mean_by_cell": crps_summary,
        "pairwise_vs_diffusion": cell_results,
        "pooled_pair_verdict_by_horizon": pooled,
        "clark_west_vs_naive": cw_results,
        "model_confidence_set": mcs_results,
        "spa_vs_garch": spa_results,
    }


def print_summary_table(df: pd.DataFrame, mcs_results: dict) -> None:
    print(f"\n{'Actif':<9}{'Horizon':<8}{'Modele':<14}{'CRPS_fair':>11}{'MCS':>6}")
    print("-" * 48)
    for (asset, horizon), g in df.groupby(["asset", "horizon"]):
        mcs_set = set(mcs_results.get(f"{asset}|{horizon}", {}).get("mcs", []))
        means = g.groupby("model")["crps"].mean().sort_values()
        for model, crps_val in means.items():
            flag = "oui" if model in mcs_set else "non"
            print(f"{asset:<9}{horizon:<8}{model:<14}{crps_val:>11.4f}{flag:>6}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--n-val", type=int, default=es.DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=es.DEFAULT_N_TEST)
    p.add_argument("--embargo", type=int, default=None, help="default: HORIZON_MAX - 1 (weeks)")
    p.add_argument("--m-samples", type=int, default=500, help="trajectories per model per origin (brief: >=500, identical across models)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsdiff-epoch-candidates", nargs="+", type=int, default=[40, 60, 80])
    p.add_argument("--tsdiff-hp-samples", type=int, default=100, help="sample count for epoch selection only (val block), separate from --m-samples")
    p.add_argument("--garch-spec-candidates", nargs="+", default=["normal", "t", "gjr-t"],
                   choices=list(dsa.GARCH_SPECS), help="GARCH innovation/asymmetry specs "
                   "selected on the validation block only (BRIEF_baselines_fortes.md §2.4)")
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates for MCS/SPA")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--scale-start", default="2015-01-01")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "duel_backtest.json"))
    args = p.parse_args()
    args.end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    t_start = time.time()
    all_records, all_meta = [], {}
    for asset_code in args.assets:
        records, meta = run_asset_duel(asset_code, ASSETS[asset_code], args)
        all_records.extend(records)
        all_meta[asset_code] = meta

    df = pd.DataFrame(all_records)
    analysis = build_grid_analysis(df, all_meta, args)
    elapsed = time.time() - t_start

    payload = {
        "config": {
            "assets": args.assets, "n_val": args.n_val, "n_test": args.n_test,
            "embargo_weeks": args.embargo if args.embargo is not None else HORIZON_MAX - 1,
            "m_samples": args.m_samples, "seed": args.seed, "k_denoise": args.k_denoise,
            "n_boot": args.n_boot, "start": args.start, "end": args.end,
            "elapsed_s": round(elapsed, 1),
            "garch_spec_candidates": args.garch_spec_candidates,
            "garch_spec_selection": "BRIEF_baselines_fortes.md §2.4: selected per asset on the "
                                    "validation block only (verrou E1), argmin mean fair CRPS "
                                    "over val_pos -- see meta_by_asset[asset].garch_spec(_val_scores).",
            "reestimation_rule": "frozen at T0 for all 6 models (declared, duel_origins.py) -- "
                                 "no residual asymmetry: every adapter is fit exactly once on "
                                 "data <= T0, only conditioning state advances at later origins.",
            "sampling": "genuine model-native trajectory sampling (duel_sampling_adapters.py) "
                       "for all 5 classics, never a Gaussian/log-normal reconstruction of stored "
                       "CI bounds (audit reserve N1).",
            "scoring": "crps_fair (Ferro 2014), m identical across all 6 models.",
            "verdict_reading_rule": "read as Model Confidence Set membership per (asset, horizon), "
                                    "never as \"jeu egal\"/\"globalement moins bon\" (audit reserve N3).",
        },
        "meta_by_asset": all_meta,
        "records": all_records,
        **analysis,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")
    print_summary_table(df, analysis["model_confidence_set"])


if __name__ == "__main__":
    main()
