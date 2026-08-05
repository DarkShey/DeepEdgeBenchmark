"""
compare_tsdiff_nsdiff_monthly_multiseed.py -- seed-stability check for the
TSDiff-vs-NsDiff model comparison (`compare_tsdiff_nsdiff_monthly.py`),
which is otherwise seed=42-only (reads `tracking.db`, single-seed by
design). Same non-negotiable as everywhere else in this body of work
(brief §3): no model-vs-model verdict presented without its multi-seed
stability.

Pure post-hoc reuse -- NO retraining. Both `nsdiff_monthly_multiseed.py` and
`tsdiff_monthly_multiseed.py` already computed and CHECKPOINTED all 5 seeds'
raw rows to disk (`experiments/checkpoints_{ns,ts}diff_monthly_multiseed/
seed{S}_{asset}.json`, each row already carrying its own `model` field).
This script just loads those checkpoints back (seconds, no compute), tags
them together per seed, and reruns `compare_tsdiff_nsdiff_monthly.
build_model_vs_model_pairs`/`compare_models_cell` (imported, not
reimplemented) per seed -- the exact same test used for the seed=42 DB
comparison, just fed from the checkpoint files for seeds 43-46 (which never
went into `tracking.db`, by design, brief §3 "multi-graines en artefact
isolé jamais dans la DB").

Scope: M+1 only (mirrors every other multiseed table in this body of work),
BOTH regimes (daily/monthly) since that's exactly where the two models'
comparison diverges sharply (see `NOTE_compare_tsdiff_nsdiff_monthly.md`).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from weekly_headtohead import ASSETS as ASSET_TICKERS                    # noqa: E402
from compare_tsdiff_nsdiff_monthly import (                              # noqa: E402
    build_model_vs_model_pairs, compare_models_cell,
)

SEEDS = [42, 43, 44, 45, 46]
ASSETS = list(ASSET_TICKERS.values())
NSDIFF_CKPT_DIR = EXPERIMENTS_DIR / "checkpoints_nsdiff_monthly_multiseed"
TSDIFF_CKPT_DIR = EXPERIMENTS_DIR / "checkpoints_tsdiff_monthly_multiseed"
OUT_PATH = EXPERIMENTS_DIR / "tsdiff_vs_nsdiff_monthly_multiseed.json"


def load_seed_rows(seed: int) -> pd.DataFrame:
    rows = []
    for asset in ASSETS:
        ns_path = NSDIFF_CKPT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"
        ts_path = TSDIFF_CKPT_DIR / f"seed{seed}_{asset.replace('=', '_')}.json"
        if not ns_path.exists() or not ts_path.exists():
            raise SystemExit(f"Checkpoint manquant pour seed={seed} asset={asset} "
                             f"({ns_path.name} / {ts_path.name}) -- lancer les deux "
                             "scripts multiseed en entier d'abord.")
        rows += json.loads(ns_path.read_text())
        rows += json.loads(ts_path.read_text())
    df = pd.DataFrame(rows)
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return df


def main():
    per_seed = {regime: {} for regime in ("daily", "monthly")}
    for seed in SEEDS:
        df = load_seed_rows(seed)
        for regime in ("daily", "monthly"):
            pairs = build_model_vs_model_pairs(df, regime, "TSDiff", "NsDiff", horizon_units=["M+1"])
            cells = compare_models_cell(pairs, "TSDiff", "NsDiff")
            per_seed[regime][seed] = {c["asset"]: c for c in cells}
        print(f"seed={seed} done.")

    cv_table = {regime: {} for regime in ("daily", "monthly")}
    for regime in ("daily", "monthly"):
        for asset in ASSETS:
            rows = [per_seed[regime][s][asset] for s in SEEDS if asset in per_seed[regime][s]]
            rows = [r for r in rows if r.get("status") == "tested"]
            if not rows:
                continue
            verdicts = [r["verdict"] for r in rows]
            rmse_ts = np.array([r["rmse_TSDiff"] for r in rows])
            rmse_ns = np.array([r["rmse_NsDiff"] for r in rows])
            cv_table[regime][asset] = {
                "verdicts_by_seed": {r0: r1["verdict"] for r0, r1 in zip(SEEDS, [per_seed[regime][s].get(asset, {}) for s in SEEDS])},
                "p_values_by_seed": {r0: per_seed[regime][s].get(asset, {}).get("p_value") for r0, s in zip(SEEDS, SEEDS)},
                "verdict_stable": len(set(verdicts)) == 1,
                "rmse_tsdiff_mean": float(rmse_ts.mean()), "rmse_nsdiff_mean": float(rmse_ns.mean()),
                "cv_rmse_tsdiff": float(rmse_ts.std(ddof=1) / rmse_ts.mean()) if len(rmse_ts) > 1 else None,
                "cv_rmse_nsdiff": float(rmse_ns.std(ddof=1) / rmse_ns.mean()) if len(rmse_ns) > 1 else None,
            }

    payload = {"seeds": SEEDS, "assets": ASSETS, "horizon_scope": "M+1 only",
              "note": "Post-hoc, reconstruit depuis les checkpoints deja calcules par "
                      "nsdiff_monthly_multiseed.py / tsdiff_monthly_multiseed.py -- aucun "
                      "reentrainement ici.",
              "cv_table_by_regime": cv_table}
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {OUT_PATH}")
    for regime in ("daily", "monthly"):
        print(f"\n--- regime {regime} ---")
        for asset, row in cv_table[regime].items():
            print(f"  {asset}: stable={row['verdict_stable']} verdicts={row['verdicts_by_seed']}")


if __name__ == "__main__":
    main()
