"""
summarize_dist_options.py — aggregate experiments/all_models_dist_options_results.json
+ experiments/lstm_mdn_results.json across the 5-asset roster into one comparison
table per (model, variant), for the final cost/benefit report.

Aggregation choice: coverage is already asset-scale-free (a percentage), so
cov_50/80/95 are averaged directly across assets. Width/pinball/CRPS are NOT
asset-scale-free (BTC prices are ~1e4, ZN=F prices are ~0.2) -- comparing raw
values across assets would be dominated by BTC, so each is expressed as a RATIO
to that (asset, model)'s own `normal` baseline before averaging across assets. A
ratio < 1 means the option improves on today's Gaussian baseline for that KPI;
> 1 means it's worse -- this is the direct "is it worth it" number the
comparison is for.

MACE (mean absolute calibration error) -- two variants, keep both:
  - `mace_loose`  : abs(mean_across_assets(coverage) - target). BUGGY as the sole
    number: this lets errors of opposite sign cancel between assets (BTC
    over-covers, SPY under-covers -> averages out to "calibrated" when neither
    asset actually is). This is what earlier versions of this script reported
    under the plain key `mean_abs_calibration_error`.
  - `mace_strict` : mean_across_assets(abs(coverage - target)) -- the honest one,
    no cancellation possible. Flagged by a colleague's independent robustness
    re-run (HANDOFF_sigma_calibration_suivi.md, 2026-07-30 -- see its §1 warning)
    after noticing this script's headline numbers were flattering relative to a
    per-asset breakdown. `mace_strict` is the one to use going forward; `mace_loose`
    is kept only so old reports referencing it stay reproducible/comparable.
`mean_abs_calibration_error` is kept as a deprecated alias for `mace_loose`
(equal value) so any existing consumer of this JSON does not silently break --
new code should read `mace_strict`.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MAIN_JSON = ROOT / "all_models_dist_options_results.json"
MDN_JSON = ROOT / "lstm_mdn_results.json"
OUT_JSON = ROOT / "dist_options_summary.json"

LEVELS = (50, 80, 95)
VARIANTS_COMMON = ("normal", "student_t", "ged", "cqr")
VARIANTS_ARIMA_NATIVE = ("native_ged", "native_skewt")


def summarize_main():
    data = json.loads(MAIN_JSON.read_text())
    assets = list(data["assets"])
    models = list(data["assets"][assets[0]]["models"])

    summary = {}
    for model in models:
        variants = list(data["assets"][assets[0]]["models"][model]["variants"])
        summary[model] = {}
        # overhead / base time, averaged across assets
        base_times = [data["assets"][a]["models"][model]["base_train_time_s"] for a in assets]
        summary[model]["base_train_time_s_mean"] = round(float(np.mean(base_times)), 2)

        for variant in variants:
            cov = {lvl: [] for lvl in LEVELS}
            rel_width = {lvl: [] for lvl in LEVELS}
            rel_pinball, rel_crps = [], []
            overhead = []
            for a in assets:
                mv = data["assets"][a]["models"][model]
                v = mv["variants"][variant]
                v_norm = mv["variants"]["normal"]
                for lvl in LEVELS:
                    cov[lvl].append(v[f"cov_{lvl}"])
                    rel_width[lvl].append(v[f"width_{lvl}"] / v_norm[f"width_{lvl}"])
                rel_pinball.append(v["pinball"] / v_norm["pinball"])
                if "crps" in v and "crps" in v_norm:
                    rel_crps.append(v["crps"] / v_norm["crps"])
                oh = mv.get("overhead_s", {}).get(variant)
                if oh is not None:
                    overhead.append(oh)

            mace_loose = float(np.mean([abs(np.mean(cov[lvl]) - lvl) for lvl in LEVELS]))
            mace_strict = float(np.mean([np.mean([abs(c - lvl) for c in cov[lvl]])
                                        for lvl in LEVELS]))
            summary[model][variant] = {
                "cov_mean": {lvl: round(float(np.mean(cov[lvl])), 2) for lvl in LEVELS},
                "cov_std_across_assets": {lvl: round(float(np.std(cov[lvl])), 2) for lvl in LEVELS},
                "mace_strict": round(mace_strict, 2),
                "mace_loose": round(mace_loose, 2),
                "mean_abs_calibration_error": round(mace_loose, 2),  # deprecated alias, see module docstring
                "rel_width_mean": {lvl: round(float(np.mean(rel_width[lvl])), 4) for lvl in LEVELS},
                "rel_pinball_mean": round(float(np.mean(rel_pinball)), 4),
                "rel_crps_mean": round(float(np.mean(rel_crps)), 4) if rel_crps else None,
                "overhead_s_mean": round(float(np.mean(overhead)), 4) if overhead else 0.0,
            }
    return summary, assets


def summarize_mdn():
    data = json.loads(MDN_JSON.read_text())
    assets = list(data["assets"])
    cov = {lvl: [] for lvl in LEVELS}
    rel_crps_vals, train_times, cov50_stds = [], [], []
    per_asset = {}
    for a in assets:
        agg = data["assets"][a]["kpi_agg"]
        for lvl in LEVELS:
            cov[lvl].append(agg[f"cov_{lvl}"]["mean"])
        train_times.append(float(np.mean(data["assets"][a]["train_time_s_per_seed"])))
        cov50_stds.append(agg["cov_50"]["std"])
        per_asset[a] = {
            "cov_50_mean": agg["cov_50"]["mean"], "cov_50_std": agg["cov_50"]["std"],
            "cov_95_mean": agg["cov_95"]["mean"],
            "crps_mean": agg["crps"]["mean"], "crps_std": agg["crps"]["std"],
            "rmse_mean": agg["rmse_eval"]["mean"],
            "train_time_s_mean": round(float(np.mean(data["assets"][a]["train_time_s_per_seed"])), 1),
        }
    mace_loose = float(np.mean([abs(np.mean(cov[lvl]) - lvl) for lvl in LEVELS]))
    mace_strict = float(np.mean([np.mean([abs(c - lvl) for c in cov[lvl]])
                                for lvl in LEVELS]))
    return {
        "per_asset": per_asset,
        "cov_mean": {lvl: round(float(np.mean(cov[lvl])), 2) for lvl in LEVELS},
        "mace_strict": round(mace_strict, 2),
        "mace_loose": round(mace_loose, 2),
        "mean_abs_calibration_error": round(mace_loose, 2),  # deprecated alias, see module docstring
        "mean_train_time_s": round(float(np.mean(train_times)), 1),
        "mean_seed_std_cov50": round(float(np.mean(cov50_stds)), 2),
        "seeds": data["config"]["seeds"],
    }


def main():
    main_summary, assets = summarize_main()
    mdn_summary = summarize_mdn()
    # baseline LSTM (from the main run) for direct MDN comparison
    lstm_baseline_train_time = main_summary["LSTM"]["base_train_time_s_mean"]
    lstm_baseline_normal = main_summary["LSTM"]["normal"]

    out = {
        "assets": assets,
        "models": main_summary,
        "mdn": mdn_summary,
        "lstm_baseline_for_mdn_comparison": {
            "train_time_s": lstm_baseline_train_time,
            "cov_mean": lstm_baseline_normal["cov_mean"],
            "mace_strict": lstm_baseline_normal["mace_strict"],
            "mace_loose": lstm_baseline_normal["mace_loose"],
            "mean_abs_calibration_error": lstm_baseline_normal["mean_abs_calibration_error"],
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT_JSON}")
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
