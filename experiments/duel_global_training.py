"""
duel_global_training.py — TSDiff trained GLOBALLY (once, on all 5 assets
pooled) vs the duel's default PER-ASSET training (BRIEF_multigraines.md
§2.3, second half of audit reco #5). Reuses duel_backtest/duel_origins/
epoch_sweep/tsdiff_model as-is; only the training regime is new here.

Per-asset (existing, duel_backtest.run_asset_duel): one TSDiff model per
asset, its own weights, fit on that asset's own z-scored return windows.

Global (this module): ONE shared set of weights, fit on the CONCATENATION
of all 5 assets' z-scored training windows (each asset still standardized
by its OWN mu/sd -- same convention as per-asset, so the model trains in a
shared unit-variance return space, not raw price units). Evaluated on each
asset's OWN test origins, de-standardized with that asset's OWN (mu, sd) --
only the trained weights are shared, never the standardization stats.

Epoch selection for the global model mirrors epoch_sweep.py's own
validation-only criterion (verrou E1), pooled across all 5 assets' val_pos
instead of one asset's: argmin MEAN fair CRPS over every asset's validation
origins, never the test block. A fresh independent final fit at the
selected epoch count is then done (mirrors weekly_headtohead_v2.py's own
"sweep decides, then a fresh fit runs" convention for TSDiff-W/D) rather
than reusing the incremental sweep's already-mutated model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tsdiff_model as td                                        # noqa: E402
import epoch_sweep as es                                         # noqa: E402
from weekly_headtohead import HORIZON_WEEKLY, HORIZON_LABELS, build_weekly, standardized_returns  # noqa: E402
from duel_origins import build_common_origins                     # noqa: E402
from crps_metrics import crps_empirical, crps_fair                # noqa: E402


def fit_tsdiff_checkpoints_pooled(train_weekly_by_asset: dict, horizon: int, candidates, seed: int):
    """Incremental-checkpoint training (epoch_sweep.fit_checkpoints' own
    trick: train partial epochs, yield, resume) on the POOLED z-scored
    windows of every asset in `train_weekly_by_asset` -- ONE shared model,
    each asset's own (mu, sd) kept alongside for standardization. Yields
    (epochs_so_far, model, mu_sd_by_asset) at each candidate epoch count."""
    td.set_seed(seed)
    all_H, all_T, mu_sd_by_asset = [], [], {}
    for asset_code, train_weekly in train_weekly_by_asset.items():
        prices = train_weekly.values.astype(float)
        r = td._log_returns(prices)
        mu, sd = float(r.mean()), float(r.std())
        sd = sd if sd > 1e-8 else 1.0
        z = (r - mu) / sd
        H_win, T_win = td._make_windows(z, td.SEQ_LEN, horizon)
        if len(H_win) == 0:
            raise ValueError(f"{asset_code}: not enough history to build training windows.")
        all_H.append(H_win)
        all_T.append(T_win)
        mu_sd_by_asset[asset_code] = (mu, sd)
    H_pooled = np.concatenate(all_H, axis=0)
    T_pooled = np.concatenate(all_T, axis=0)

    model = td.TSDiff(td.SEQ_LEN, horizon, td.HIDDEN, td.DEPTH, td.COND_DIM, td.T_DIFFUSION)
    done = 0
    for target in sorted(candidates):
        model.train(H_pooled, T_pooled, epochs=target - done, batch_size=td.BATCH_SIZE)
        done = target
        yield target, model, mu_sd_by_asset


def select_global_tsdiff_epochs(train_weekly_by_asset: dict, weekly_by_asset: dict,
                                weekly_dates_by_asset: dict, daily_by_asset: dict,
                                val_pos_by_asset: dict, candidates, seed: int,
                                n_samples: int, k_denoise: int) -> tuple:
    """Argmin MEAN fair CRPS pooled over every asset's val_pos (validation
    block only, verrou E1 -- test_pos is never touched here or anywhere in
    this module). Returns (best_epochs, {epochs: mean_crps_val})."""
    scores = {}
    for epochs, model, mu_sd_by_asset in fit_tsdiff_checkpoints_pooled(
            train_weekly_by_asset, HORIZON_WEEKLY, candidates, seed):
        crps_vals = []
        for asset_code, val_pos in val_pos_by_asset.items():
            weekly = weekly_by_asset[asset_code]
            weekly_dates = weekly_dates_by_asset[asset_code]
            daily = daily_by_asset[asset_code]
            mu, sd = mu_sd_by_asset[asset_code]
            weekly_z = standardized_returns(weekly, mu, sd)
            for k, m_pos in enumerate(val_pos):
                es.week_targets(weekly_dates, daily, m_pos)  # guardrail: raises if misaligned
                last_price = float(weekly.iloc[m_pos])
                td.set_seed(seed + k)
                samples = td.forecast_from_fitted(model, weekly_z[:m_pos], mu, sd, last_price,
                                                  horizons=[1, 2, 3], n_samples=n_samples,
                                                  k_denoise=k_denoise)
                for wi in range(3):
                    actual = float(weekly.iloc[m_pos + wi + 1])
                    crps_vals.append(crps_fair(samples[wi + 1], actual))
        scores[epochs] = float(np.mean(crps_vals))
    best_epochs = min(scores, key=scores.get)
    return best_epochs, scores


def fit_tsdiff_global(train_weekly_by_asset: dict, horizon: int, epochs: int, seed: int) -> tuple:
    """Fresh, independent final fit at exactly `epochs` (not reusing the
    incremental sweep's mutated model, epoch_sweep's own convention) -- ONE
    shared model trained on the pooled windows of every asset. Returns
    (model, {asset_code: (mu, sd)})."""
    gen = fit_tsdiff_checkpoints_pooled(train_weekly_by_asset, horizon, [epochs], seed)
    _, model, mu_sd_by_asset = next(gen)
    return model, mu_sd_by_asset


def evaluate_tsdiff_on_test(asset_code: str, ticker: str, model, mu: float, sd: float, args) -> list:
    """Scores an already-fitted TSDiff model (global OR per-asset) on
    `asset_code`'s OWN test_pos -- same origins duel_backtest.run_asset_duel
    uses, same record schema (model label "TSDiff-global"), so results plug
    directly into duel_pairwise_tests/mcs alongside the classics' records
    already computed for the SAME seed (fair, "armes egales" comparison:
    identical origins, identical seed-derived per-origin sampling seeds,
    identical m)."""
    daily = td.fetch_data(ticker, args.start, args.end)
    weekly, weekly_dates = build_weekly(daily)
    _, _, test_pos = build_common_origins(weekly, args.n_val, args.n_test, embargo=args.embargo)
    weekly_z = standardized_returns(weekly, mu, sd)

    records = []
    for k, m_pos in enumerate(test_pos):
        origin_date, _, target_dates, _ = es.week_targets(weekly_dates, daily, m_pos)
        last_price = float(weekly.iloc[m_pos])
        seed_k = args.seed + k
        td.set_seed(seed_k)
        samples = td.forecast_from_fitted(model, weekly_z[:m_pos], mu, sd, last_price,
                                          horizons=[1, 2, 3], n_samples=args.m_samples,
                                          k_denoise=args.k_denoise)
        for wi, h_label in enumerate(HORIZON_LABELS):
            actual = float(weekly.iloc[m_pos + wi + 1])
            s = samples[wi + 1]
            records.append({
                "asset": ticker, "asset_code": asset_code, "horizon": h_label,
                "model": "TSDiff-global", "origin": k, "origin_date": str(origin_date.date()),
                "target_date": str(target_dates[wi].date()), "actual": actual,
                "point": float(np.mean(s)), "crps": crps_fair(s, actual),
                "crps_empirical": crps_empirical(s, actual),
            })
    return records
