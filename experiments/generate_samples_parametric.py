"""
generate_samples_parametric.py — N=500 samples for ARIMA-GARCH / SARIMA /
Prophet / Naive / LSTM, drawn from each model's own predictive distribution as
implied by its already-stored (y_pred, y_lower, y_upper) 95% PI (see
prob_kpi_common.sample_parametric's docstring for the per-model construction).
Zero retraining, zero new fits -- BRIEF_kpi_probabilistes.md Sec.3.

Persists to experiments/samples/<ASSET>_parametric.{index.parquet,npz} — NOT
tracking.db (brief guardrail: don't bloat the binary DB with sample clouds).

Usage:
    python generate_samples_parametric.py --asset SPY
    python generate_samples_parametric.py --asset SPY --n-samples 500 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prob_kpi_common import (                                    # noqa: E402
    PARAMETRIC_MODELS, load_matrix_rows, sample_parametric,
)

SAMPLES_DIR = ROOT / "samples"


def generate(asset: str, n_samples: int = 500, seed: int = 42,
             db_path: str = None) -> tuple:
    kwargs = {} if db_path is None else {"db_path": db_path}
    df = load_matrix_rows(asset, models=list(PARAMETRIC_MODELS), **kwargs)
    if df.empty:
        raise SystemExit(f"no rows found for asset={asset!r} among {PARAMETRIC_MODELS}")

    n = len(df)
    samples = np.empty((n, n_samples), dtype=np.float64)
    # Deterministic per-row seed (model, horizon cell, cutoff_date) -> reproducible,
    # independent draws across rows regardless of iteration order.
    for i, row in df.iterrows():
        row_seed = (seed, row["model"], row["frequence"], row["horizon_type"],
                    row["horizon_unit"], row["cutoff_date"])
        rng = np.random.default_rng(abs(hash(row_seed)) % (2**32))
        samples[i] = sample_parametric(
            row["model"], row["y_pred"], row["y_lower"], row["y_upper"],
            row["last_close"], n_samples, rng,
        )

    index = df.copy()
    index["method"] = "parametric"
    index["n_samples"] = n_samples
    return index.reset_index(drop=True), samples


def save(asset: str, index: pd.DataFrame, samples: np.ndarray, suffix: str = "parametric") -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    index_path = SAMPLES_DIR / f"{asset}_{suffix}.index.parquet"
    samples_path = SAMPLES_DIR / f"{asset}_{suffix}.samples.npz"
    index.to_parquet(index_path)
    np.savez_compressed(samples_path, samples=samples)
    print(f"[{asset}] {suffix}: {len(index)} rows x {samples.shape[1]} samples -> "
          f"{index_path.name}, {samples_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", required=True)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-path", default=None)
    args = p.parse_args()

    index, samples = generate(args.asset, args.n_samples, args.seed, args.db_path)
    save(args.asset, index, samples)


if __name__ == "__main__":
    main()
