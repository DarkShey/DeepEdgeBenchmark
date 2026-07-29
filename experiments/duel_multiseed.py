"""
duel_multiseed.py — multi-seed robustness loop around the duel
(BRIEF_multigraines.md §2.1-2.2), plus the TSDiff global-vs-per-asset
training comparison (§2.3). Reuses duel_backtest.run_asset_duel/
build_grid_analysis and duel_global_training UNCHANGED -- this file only
loops them over seeds, checkpoints, and aggregates the dispersion.

Why re-arming the seed alone is enough (nothing else needs to change):
every source of model stochasticity in this duel is already threaded
through a single top-level `args.seed` (duel_backtest.py's own convention:
TSDiff init/training/sampling via td.set_seed, LSTM weight init + MC-Dropout
via seed=args.seed, GARCH/SARIMA/Prophet/Naive trajectory draws via
seed=args.seed+k per origin). Looping `args.seed = seed` before each full
backtest re-arms ALL of it -- origins (duel_origins.build_common_origins)
are date-position-based and carry NO randomness, so they are identical
across seeds by construction, exactly as the brief requires ("rien d'autre
ne change"). The test-side bootstraps (mcs.py, duel_pairwise_tests.py) are
ALSO seeded from args.seed -- brief §2.1 explicitly declares this acceptable
("les bootstraps de test dérivent de façon déterministe de la graine du
run"): we are measuring MODEL variance across seeds, and each seed's own
test excercises its own models, so conflating the two would need a second,
independent bootstrap seed axis that the brief does not ask for.

Checkpointing (brief §5, "run reprenable"): each (seed, asset) result is
saved to experiments/checkpoints/seed{seed}_{asset_code}.json as soon as
computed; a rerun skips any (seed, asset) whose checkpoint file already
exists, so an interrupted multi-seed run resumes without recomputing
already-finished (seed, asset) pairs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

import duel_backtest as db                                        # noqa: E402
import duel_global_training as dgt                                 # noqa: E402
import tsdiff_model as td                                          # noqa: E402
from weekly_headtohead import ASSETS, build_weekly                 # noqa: E402
from duel_origins import build_common_origins                      # noqa: E402

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


# ══════════════════════════════════════════════════════════════════════════
#  Checkpointed per-(seed, asset) run
# ══════════════════════════════════════════════════════════════════════════

def checkpoint_path(seed: int, asset_code: str) -> Path:
    return CHECKPOINT_DIR / f"seed{seed}_{asset_code}.json"


def load_or_run_asset(seed: int, asset_code: str, ticker: str, args) -> tuple:
    path = checkpoint_path(seed, asset_code)
    if path.exists():
        data = json.loads(path.read_text())
        print(f"[seed={seed}][{asset_code}] loaded from checkpoint ({path.name})")
        return data["records"], data["meta"]
    records, meta = db.run_asset_duel(asset_code, ticker, args)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records, "meta": meta}, default=str))
    print(f"[seed={seed}][{asset_code}] checkpoint saved -> {path.name}")
    return records, meta


def run_one_seed(seed: int, args) -> tuple:
    """Full duel (all assets) at one seed. Returns (df, all_meta, analysis)."""
    args.seed = seed
    all_records, all_meta = [], {}
    for asset_code in args.assets:
        records, meta = load_or_run_asset(seed, asset_code, ASSETS[asset_code], args)
        all_records.extend(records)
        all_meta[asset_code] = meta
    df = pd.DataFrame(all_records)
    analysis = db.build_grid_analysis(df, all_meta, args)
    return df, all_meta, analysis


# ══════════════════════════════════════════════════════════════════════════
#  Dispersion aggregation across seeds (brief §2.2, the actual deliverable)
# ══════════════════════════════════════════════════════════════════════════

def aggregate_crps_dispersion(per_seed_df: dict) -> list:
    """Per (asset, horizon, model): mean/std of the PER-SEED mean CRPS (one
    number per seed first, then dispersion of that number across seeds --
    this measures seed-to-seed model variance, not origin-to-origin
    variance within a seed, which is a different, already-reported axis)."""
    frames = []
    for seed, df in per_seed_df.items():
        g = df.groupby(["asset", "horizon", "model"])["crps"].mean().reset_index()
        g["seed"] = seed
        frames.append(g)
    allg = pd.concat(frames, ignore_index=True)
    agg = (allg.groupby(["asset", "horizon", "model"])["crps"]
           .agg(crps_mean="mean", crps_std="std", n_seeds="count").reset_index())
    return agg.to_dict(orient="records")


def aggregate_mcs_stability(per_seed_analysis: dict) -> dict:
    """{cell: {model: fraction of seeds where model is in that cell's MCS}}."""
    n_seeds = len(per_seed_analysis)
    cells = set()
    for a in per_seed_analysis.values():
        cells |= set(a["model_confidence_set"])
    out = {}
    for cell in sorted(cells):
        counts = {m: 0 for m in db.MODELS}
        for a in per_seed_analysis.values():
            for m in a["model_confidence_set"].get(cell, {}).get("mcs", []):
                counts[m] = counts.get(m, 0) + 1
        out[cell] = {m: c / n_seeds for m, c in counts.items()}
    return out


def aggregate_holm_stability(per_seed_analysis: dict) -> dict:
    """{cell_key: {"fraction_significant", "seeds_significant", "stable"}} for
    every pairwise-test cell key seen in ANY seed. `stable` = fraction is
    exactly 0.0 or 1.0 (same significance call in every seed) -- any other
    value is a case that BASCULE from one seed to another, flagged, not
    hidden."""
    n_seeds = len(per_seed_analysis)
    keys = set()
    for a in per_seed_analysis.values():
        keys |= set(a["pairwise_vs_diffusion"])
    out = {}
    for key in sorted(keys):
        seeds_sig = []
        for seed, a in per_seed_analysis.items():
            cell = a["pairwise_vs_diffusion"].get(key, {})
            if cell.get("significant_after_holm"):
                seeds_sig.append(seed)
        frac = len(seeds_sig) / n_seeds
        out[key] = {
            "fraction_significant": frac, "seeds_significant": seeds_sig,
            "stable": bool(frac in (0.0, 1.0)),
        }
    return out


def aggregate_pooled_stability(per_seed_analysis: dict) -> dict:
    """{pair: {horizon: {"fraction_significant", "seeds_significant", "stable"}}}."""
    n_seeds = len(per_seed_analysis)
    pairs = set()
    for a in per_seed_analysis.values():
        pairs |= set(a["pooled_pair_verdict_by_horizon"])
    out = {}
    for pair in sorted(pairs):
        out[pair] = {}
        for h in db.HORIZONS:
            seeds_sig = []
            for seed, a in per_seed_analysis.items():
                cell = a["pooled_pair_verdict_by_horizon"].get(pair, {}).get(h, {})
                if cell.get("significant_bootstrap"):
                    seeds_sig.append(seed)
            frac = len(seeds_sig) / n_seeds
            out[pair][h] = {
                "fraction_significant": frac, "seeds_significant": seeds_sig,
                "stable": bool(frac in (0.0, 1.0)),
            }
    return out


def aggregate_spa_stability(per_seed_analysis: dict) -> dict:
    """{cell: {"fraction_reject", "seeds_reject", "stable"}} -- does the SPA
    non-rejection (or rejection) of "nothing beats GARCH(1,1)" hold on every
    seed?"""
    n_seeds = len(per_seed_analysis)
    cells = set()
    for a in per_seed_analysis.values():
        cells |= set(a["spa_vs_garch"])
    out = {}
    for cell in sorted(cells):
        seeds_rej = []
        for seed, a in per_seed_analysis.items():
            if a["spa_vs_garch"].get(cell, {}).get("reject_no_model_beats_benchmark"):
                seeds_rej.append(seed)
        frac = len(seeds_rej) / n_seeds
        out[cell] = {
            "fraction_reject": frac, "seeds_reject": seeds_rej,
            "stable": bool(frac in (0.0, 1.0)),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Global vs per-asset TSDiff training comparison (brief §2.3)
# ══════════════════════════════════════════════════════════════════════════

def run_global_training_comparison(seeds: list, args) -> dict:
    """For each seed in `seeds` (a DECLARED subset, brief §2.3 allows >= 2 for
    cost): fit ONE shared TSDiff model on the pooled training windows of all
    `args.assets`, epoch count selected on the pooled validation blocks only
    (verrou E1), then score it on each asset's own test_pos. Compares its
    standing (MCS membership, mean CRPS) against the per-asset TSDiff of the
    SAME seed -- reuses that seed's already-computed classics from the
    per-(seed, asset) checkpoints (armes egales: same origins, same classics,
    only TSDiff's training regime differs)."""
    results_by_seed = {}
    for seed in seeds:
        args.seed = seed
        print(f"[global][seed={seed}] fetching/building per-asset weekly series ...")
        daily_by_asset, weekly_by_asset, weekly_dates_by_asset = {}, {}, {}
        val_pos_by_asset, train_weekly_by_asset = {}, {}
        for asset_code in args.assets:
            ticker = ASSETS[asset_code]
            daily = td.fetch_data(ticker, args.start, args.end)
            weekly, weekly_dates = build_weekly(daily)
            train_end_pos, val_pos, _ = build_common_origins(
                weekly, args.n_val, args.n_test, embargo=args.embargo)
            daily_by_asset[asset_code] = daily
            weekly_by_asset[asset_code] = weekly
            weekly_dates_by_asset[asset_code] = weekly_dates
            val_pos_by_asset[asset_code] = val_pos
            train_weekly_by_asset[asset_code] = weekly.iloc[:train_end_pos + 1]

        t0 = time.time()
        best_epochs, val_scores = dgt.select_global_tsdiff_epochs(
            train_weekly_by_asset, weekly_by_asset, weekly_dates_by_asset, daily_by_asset,
            val_pos_by_asset, args.tsdiff_epoch_candidates, seed, args.tsdiff_hp_samples,
            args.k_denoise)
        print(f"[global][seed={seed}] epochs*={best_epochs} (val-selected pooled, "
              f"candidates={val_scores}, {time.time() - t0:.0f}s)")

        t0 = time.time()
        model, mu_sd_by_asset = dgt.fit_tsdiff_global(train_weekly_by_asset, db.HORIZON_WEEKLY,
                                                       best_epochs, seed)
        print(f"[global][seed={seed}] shared TSDiff fit in {time.time() - t0:.0f}s")

        global_records = []
        for asset_code in args.assets:
            mu, sd = mu_sd_by_asset[asset_code]
            recs = dgt.evaluate_tsdiff_on_test(asset_code, ASSETS[asset_code], model, mu, sd, args)
            global_records.extend(recs)
            print(f"[global][seed={seed}][{asset_code}] scored on test_pos "
                  f"({len(recs) // 3} origins)")

        results_by_seed[seed] = {
            "epochs_global": best_epochs, "epochs_val_scores": val_scores,
            "records": global_records,
        }
    return results_by_seed


# (the standing comparison itself -- swap TSDiff-global in for TSDiff, rerun
# the same MCS/pairwise analysis, diff the MCS membership -- is done inline
# in main(), where per_seed_df/per_seed_analysis are already in scope.)


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                   help="model seeds re-armed for each full duel run (brief: S >= 5)")
    p.add_argument("--global-seeds", nargs="+", type=int, default=None,
                   help="declared subset of --seeds used for the global-training "
                        "comparison (brief §2.3 allows >= 2 for cost); default: first 2 of --seeds")
    p.add_argument("--n-val", type=int, default=db.es.DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=db.es.DEFAULT_N_TEST)
    p.add_argument("--embargo", type=int, default=None)
    p.add_argument("--m-samples", type=int, default=500)
    p.add_argument("--tsdiff-epoch-candidates", nargs="+", type=int, default=[40, 60, 80])
    p.add_argument("--tsdiff-hp-samples", type=int, default=100)
    p.add_argument("--garch-spec-candidates", nargs="+", default=["normal", "t", "gjr-t"])
    p.add_argument("--k-denoise", type=int, default=td.K_DENOISE)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--scale-start", default="2015-01-01")
    p.add_argument("--skip-global", action="store_true", help="skip the global-training comparison")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "duel_backtest.json"))
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    global_seeds = args.global_seeds or args.seeds[:2]

    t_start = time.time()
    per_seed_df, per_seed_meta, per_seed_analysis = {}, {}, {}
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        df, all_meta, analysis = run_one_seed(seed, args)
        per_seed_df[seed] = df
        per_seed_meta[seed] = all_meta
        per_seed_analysis[seed] = analysis

    print("\n=== aggregating dispersion across seeds ===")
    crps_dispersion = aggregate_crps_dispersion(per_seed_df)
    mcs_stability = aggregate_mcs_stability(per_seed_analysis)
    holm_stability = aggregate_holm_stability(per_seed_analysis)
    pooled_stability = aggregate_pooled_stability(per_seed_analysis)
    spa_stability = aggregate_spa_stability(per_seed_analysis)

    global_comparison = {}
    if not args.skip_global:
        print(f"\n=== global-training comparison (seeds={global_seeds}) ===")
        global_results = run_global_training_comparison(global_seeds, args)
        for seed, gres in global_results.items():
            base_df = per_seed_df[seed]
            classics_df = base_df[base_df["model"] != "TSDiff"]
            global_df = pd.DataFrame(gres["records"])
            swapped_df = pd.concat([classics_df, global_df.assign(model="TSDiff")], ignore_index=True)
            analysis_global = db.build_grid_analysis(swapped_df, per_seed_meta[seed], args)
            crps_per_asset = (base_df[base_df["model"] == "TSDiff"]
                              .groupby(["asset", "horizon"])["crps"].mean())
            crps_global = global_df.groupby(["asset", "horizon"])["crps"].mean()
            mcs_per_asset = {cell: "TSDiff" in v.get("mcs", [])
                            for cell, v in per_seed_analysis[seed]["model_confidence_set"].items()}
            mcs_global = {cell: "TSDiff" in v.get("mcs", [])
                         for cell, v in analysis_global["model_confidence_set"].items()}
            global_comparison[str(seed)] = {
                "epochs_global": gres["epochs_global"], "epochs_val_scores": gres["epochs_val_scores"],
                "crps_per_asset_mean": {f"{a}|{h}": v for (a, h), v in crps_per_asset.items()},
                "crps_global_mean": {f"{a}|{h}": v for (a, h), v in crps_global.items()},
                "mcs_membership_per_asset": mcs_per_asset, "mcs_membership_global": mcs_global,
                "standing_changed_cells": [c for c in mcs_per_asset
                                           if mcs_per_asset[c] != mcs_global.get(c)],
            }

    elapsed = time.time() - t_start
    budgets = {
        "TSDiff (par actif)": {
            "epochs": "candidats 40/60/80, sélection sur validation par actif (verrou É1)",
            "data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
            "hpo": "époques (validation uniquement)", "n_seeds": len(args.seeds),
        },
        "TSDiff (global)": {
            "epochs": "candidats 40/60/80, sélection pooled sur validation de tous les actifs",
            "data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
            "hpo": "époques (validation pooled uniquement)",
            "n_seeds": f"{len(global_seeds)} (sous-ensemble déclaré de {args.seeds})",
        },
        "ARIMA-GARCH": {
            "spec": "normal/t/gjr-t, sélection sur validation par actif (verrou É1)",
            "data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
            "hpo": "spec d'innovation (validation uniquement)", "n_seeds": len(args.seeds),
        },
        "SARIMA": {"order": "fixe (sarima_model.ORDER/SEASONAL_ORDER)",
                  "data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
                  "hpo": "aucun", "n_seeds": len(args.seeds)},
        "Prophet": {"data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
                   "hpo": "aucun", "n_seeds": len(args.seeds)},
        "LSTM": {"epochs": "fixe (lstm_model.EPOCHS), MC-Dropout re-armé par graine",
                "data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
                "hpo": "aucun", "n_seeds": len(args.seeds)},
        "Naive": {"data_window": f"{args.start} -> {args.end}", "m": args.m_samples,
                 "hpo": "aucun", "n_seeds": len(args.seeds)},
    }

    payload = {
        "config": {
            "assets": args.assets, "seeds": args.seeds, "global_seeds": global_seeds,
            "n_val": args.n_val, "n_test": args.n_test, "m_samples": args.m_samples,
            "k_denoise": args.k_denoise, "n_boot": args.n_boot, "start": args.start, "end": args.end,
            "elapsed_s": round(elapsed, 1), "budgets": budgets,
            "multiseed_rule": "each seed re-arms ALL model stochasticity (TSDiff init/train/sample, "
                              "LSTM init/MC-Dropout, classics' trajectory draws) via a single "
                              "top-level args.seed, threaded through duel_backtest.py unchanged; "
                              "origins carry no randomness so are identical across seeds; test-side "
                              "bootstraps derive deterministically from the run seed (declared, "
                              "brief §2.1).",
        },
        "per_seed": {
            str(seed): {
                "meta_by_asset": per_seed_meta[seed], "records": per_seed_df[seed].to_dict(orient="records"),
                **per_seed_analysis[seed],
            } for seed in args.seeds
        },
        "aggregate": {
            "crps_dispersion": crps_dispersion, "mcs_stability": mcs_stability,
            "holm_stability": holm_stability, "pooled_stability": pooled_stability,
            "spa_stability": spa_stability,
        },
        "global_vs_per_asset": global_comparison,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
