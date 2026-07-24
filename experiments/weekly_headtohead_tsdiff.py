"""
Tête-à-tête TSDiff-W (weekly-natif) vs TSDiff-D (daily-multistep) — Phase 0
==========================================================================
Teste l'hypothèse : un TSDiff entraîné sur rendements HEBDO (1 pas = 1 semaine)
prédit W+1/W+2/W+3 mieux qu'un TSDiff entraîné en DAILY et extrapolé sur ~5/10/15
pas de bourse (multi-step).

Protocole : **train-once-forward** (OOS, sans lookahead). Chaque modèle est entraîné
UNE fois sur les données <= première origine, puis prévoit aux N origines en ne
faisant que ré-échantillonner (TSDiff est conditionné sur la fenêtre d'historique,
donc il forecast depuis n'importe quelle origine sans réentraînement). Standardisation
figée sur la fenêtre d'entraînement (aucune fuite). Identique pour les deux modèles
(comparaison équitable) → permet 30 origines × 300 epochs pour ~6 entraînements.

Comparaison équitable : à chaque vendredi-origine, les deux modèles prévoient les
MÊMES dates-cibles (vendredis t+1/t+2/t+3) :
  - TSDiff-W lit l'horizon en semaines (k = 1,2,3) ;
  - TSDiff-D lit son chemin daily à l'offset = nb de jours de bourse origine->cible.
On agrège RMSE, couverture PI 95%, CRPS par horizon et par modèle.

Usage :
    python experiments/weekly_headtohead_tsdiff.py --assets SPY,BTC-USD,TLT \
        --origins 30 --epochs 300 --n-samples 200
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, "models")   # même convention d'import que le pipeline
import numpy as np
import pandas as pd

import tsdiff_model as td
import tsdiff_weekly as tw

WEEKS = (1, 2, 3)
DAILY_SEQ_LEN = 30
DAILY_HORIZON = 16   # couvre ~3 semaines (~15 j de bourse) + marge jours fériés


def _daily_pos(daily_index: pd.DatetimeIndex, date) -> int:
    """Position du dernier jour coté <= `date` (le vendredi étiquette W-FRI peut être
    férié, donc absent de l'index daily)."""
    return int(daily_index.searchsorted(date, side="right")) - 1


def empirical_crps(samples: np.ndarray, y: float) -> float:
    """CRPS empirique (estimateur énergie) : E|X-y| - 0.5 E|X-X'|."""
    s = np.asarray(samples, dtype=float)
    return float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])))


# ── Daily : fit une fois, forecast depuis un lookback arbitraire ────────────────
def fit_daily(train_daily: pd.Series, epochs, seq_len=DAILY_SEQ_LEN,
              horizon=DAILY_HORIZON, hidden=64, depth=2):
    """Entraîne un TSDiff daily (horizon jours). Retourne (model, mu, sd)."""
    p = train_daily.values.astype(float)
    r = td._log_returns(p)
    mu, sd = float(r.mean()), float(r.std())
    sd = sd if sd > 1e-8 else 1.0
    z = (r - mu) / sd
    H_win, T_win = td._make_windows(z, seq_len, horizon)
    model = td.TSDiff(seq_len=seq_len, horizon=horizon, hidden=hidden, depth=depth)
    model.train(H_win, T_win, epochs=epochs)
    return model, mu, sd


def forecast_daily_from_fitted(model, mu, sd, lookback_z, last_price, offsets,
                               n_samples, k_denoise, horizon=DAILY_HORIZON):
    """Échantillonne le chemin daily depuis un modèle déjà entraîné + lookback, lit le
    prix à chaque offset (jours de bourse). Retourne {offset: samples_prix}."""
    paths = model.sample_paths(np.asarray(lookback_z, dtype=np.float32),
                               n_samples=n_samples, k_denoise=k_denoise)  # [N, horizon]
    out = {}
    for off in offsets:
        k = min(int(off), horizon)
        cum_r = paths[:, :k].sum(axis=1) * sd + k * mu
        out[off] = last_price * np.exp(cum_r)
    return out


def _stats(samples: np.ndarray, target: float) -> dict:
    lo, hi = np.quantile(samples, 0.025), np.quantile(samples, 0.975)
    return {
        "sq_err": float((np.mean(samples) - target) ** 2),
        "cover": int(lo <= target <= hi),
        "crps": empirical_crps(samples, target),
    }


def run_asset(ticker, start, end, origins, epochs, n_samples, k_denoise, seed):
    td.set_seed(seed)
    daily = td.fetch_data(ticker, start, end)
    weekly = tw.to_weekly(daily)
    n = len(weekly)

    last_origin = n - 1 - max(WEEKS)
    first_origin = max(tw.SEQ_LEN_W, DAILY_SEQ_LEN, last_origin - origins + 1)
    origin_idx = list(range(first_origin, last_origin + 1))

    # ── train-once : chaque modèle entraîné UNE fois sur data <= première origine ──
    cutoff_date = weekly.index[first_origin]
    print(f"  {ticker}: {len(daily)} barres daily -> {n} hebdo ; "
          f"{len(origin_idx)} origines (train <= {str(cutoff_date.date())})")
    model_w, _z_w, mu_w, sd_w, _last_w = tw.fit_weekly(weekly.iloc[:first_origin + 1], epochs=epochs)
    model_d, mu_d, sd_d = fit_daily(daily.loc[:cutoff_date], epochs=epochs)

    r_week = td._log_returns(weekly.values)     # r_week[i] = ret weekly[i]->weekly[i+1]
    r_day = td._log_returns(daily.values)

    acc = {m: {k: [] for k in WEEKS} for m in ("TSDiff-W", "TSDiff-D")}
    for oi in origin_idx:
        origin_date = weekly.index[oi]
        targets = {k: (weekly.index[oi + k], float(weekly.iloc[oi + k])) for k in WEEKS}

        # TSDiff-W : lookback = seq_len derniers rendements hebdo, standardisés (mu/sd train)
        lb_w = (r_week[oi - tw.SEQ_LEN_W:oi] - mu_w) / sd_w
        wk = tw.forecast_from_fitted_weekly(model_w, mu_w, sd_w, lb_w, float(weekly.iloc[oi]),
                                            weeks=WEEKS, n_samples=n_samples, k_denoise=k_denoise)
        for k in WEEKS:
            acc["TSDiff-W"][k].append(_stats(wk[k]["samples"], targets[k][1]))

        # TSDiff-D : même origine, offsets des mêmes cibles
        pos0 = _daily_pos(daily.index, origin_date)
        offsets = {k: (_daily_pos(daily.index, targets[k][0]) - pos0) for k in WEEKS}
        lb_d = (r_day[pos0 - DAILY_SEQ_LEN:pos0] - mu_d) / sd_d
        dsamp = forecast_daily_from_fitted(model_d, mu_d, sd_d, lb_d, float(daily.iloc[pos0]),
                                           list(offsets.values()), n_samples=n_samples,
                                           k_denoise=k_denoise)
        for k in WEEKS:
            acc["TSDiff-D"][k].append(_stats(dsamp[offsets[k]], targets[k][1]))

    res = {}
    for m in acc:
        res[m] = {}
        for k in WEEKS:
            rows = acc[m][k]
            res[m][f"W{k}"] = {
                "rmse": round(float(np.sqrt(np.mean([r["sq_err"] for r in rows]))), 4),
                "coverage_95": round(float(np.mean([r["cover"] for r in rows])), 3),
                "crps": round(float(np.mean([r["crps"] for r in rows])), 4),
                "n": len(rows),
            }
    return res


def main():
    p = argparse.ArgumentParser(description="Tête-à-tête TSDiff weekly vs daily-multistep")
    p.add_argument("--assets", default="SPY,BTC-USD,TLT")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2026-07-16")
    p.add_argument("--origins", type=int, default=30, help="nb d'origines walk-forward hebdo")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--k-denoise", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/weekly_headtohead_results.json")
    args = p.parse_args()

    tickers = [t.strip() for t in args.assets.split(",") if t.strip()]
    all_res = {}
    for tk in tickers:
        print(f"\n=== {tk} ===")
        all_res[tk] = run_asset(tk, args.start, args.end, args.origins, args.epochs,
                                args.n_samples, args.k_denoise, args.seed)

    print("\n" + "=" * 72)
    print(f"{'Actif':<10}{'Horizon':<9}{'Modèle':<11}{'RMSE':>12}{'Cov95':>8}{'CRPS':>12}")
    print("-" * 72)
    for tk in tickers:
        for k in WEEKS:
            for m in ("TSDiff-W", "TSDiff-D"):
                r = all_res[tk][m][f"W{k}"]
                print(f"{tk:<10}{'W'+str(k):<9}{m:<11}{r['rmse']:>12}{r['coverage_95']:>8}{r['crps']:>12}")
        print("-" * 72)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "protocol": "train-once-forward", "results": all_res}, f, indent=2)
    print(f"\nRésultats -> {args.out}")

    print("\n=== Verdict (CRPS weekly-natif vs daily-multistep) ===")
    wins = {"W": 0, "D": 0}
    for tk in tickers:
        for k in WEEKS:
            w = all_res[tk]["TSDiff-W"][f"W{k}"]["crps"]
            d = all_res[tk]["TSDiff-D"][f"W{k}"]["crps"]
            winner = "W" if w < d else "D"
            wins[winner] += 1
            print(f"  {tk} W{k}: TSDiff-W={w} vs TSDiff-D={d}  -> {winner} gagne")
    print(f"\n  Bilan CRPS : TSDiff-W gagne {wins['W']}/{sum(wins.values())} cas.")


if __name__ == "__main__":
    main()
