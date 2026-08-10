"""
cost_grid_2020.py -- chantier B du BRIEF extension/puissance, ETAPE PREALABLE :
« chiffrer le cout de regeneration complet avant de lancer, et le decouper si
necessaire ».

Le brief demande de regenerer la grille `oos` a depart 2020-01 (340 origines,
`effective_n` ~ 113) pour TOUS les modeles de reference restants, la
comparabilite mutuelle l'exigeant. Sur 7 actifs et 2 regimes, cela fait
340 x 7 x 2 = 4 760 origines-cellules PAR MODELE refit-par-origine. Avant de
lancer plusieurs jours de calcul a l'aveugle, on mesure.

METHODE : on ne devine pas, on CHRONOMETRE. Chaque modele est execute sur un
petit echantillon d'origines reelles, dans les deux regimes, sur les series
gelees de `prices_v3/`, et le cout est extrapole lineairement au nombre
d'origines de la grille. L'extrapolation est lineaire parce que ces modeles sont
refit a chaque origine independamment -- sauf NsDiff, train-once-forward, dont le
cout se decompose en (fits fixes) + (forecast x origines) et est traite a part.

CE QUE LE CHIFFRAGE SERT A DECIDER, et c'est declare avant de le lire :
  * si le total est sous ~2 h, on lance tout ;
  * sinon on decoupe, et l'ordre de decoupe est fixe par le brief lui-meme :
    l'hypothese primaire pre-declaree est `var_limit` sur SPY, W+2/W+3, regime
    WEEKLY, NsDiff-ensemble vs ARIMA-GARCH. Le sous-ensemble minimal qui la
    tranche est donc {NsDiff, ARIMA-GARCH} x {weekly} x {SPY}. Tout ce qui
    depasse sert les analyses exploratoires, pas le test confirmatoire.

Sortie : experiments/cost_grid_2020.json
Usage   : python cost_grid_2020.py [--n-probe 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmarks.multi_horizon as mh                                 # noqa: E402
import nsdiff_model as nm                                             # noqa: E402
from epoch_sweep import week_targets                                  # noqa: E402
from prices_v3 import ORIGIN_START, OUT_DIR as PRICES_V3, PANEL, slug  # noqa: E402
from weekly_headtohead import build_weekly, HORIZON_WEEKLY, HORIZON_DAILY  # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W                  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "cost_grid_2020.json"
N_SEEDS = 5
N_SAMPLES = 200
BUDGET_HOURS_ALL_IN = 2.0        # seuil de decision declare

# Modeles refit-par-origine encore au benchmark (TSDiff retire).
PER_ORIGIN_MODELS = {
    "ARIMA-GARCH": mh.forecast_horizons_arima,
    "SARIMA": mh.forecast_horizons_sarima,
    "Prophet": mh.forecast_horizons_prophet,
    "Naive": None,        # analytique et instantane, mesure quand meme
}


def load_grid(asset: str):
    daily = pd.read_parquet(PRICES_V3 / f"{slug(asset)}.parquet")["close"]
    weekly, weekly_dates = build_weekly(daily)
    origin = pd.Timestamp(ORIGIN_START)
    test_pos = [i for i, d in enumerate(weekly_dates) if d >= origin][:-3]
    return daily, weekly, weekly_dates, test_pos


def probe_per_origin(name: str, fn, asset: str, n_probe: int) -> dict:
    """Chronometre `n_probe` origines reelles, dans les deux regimes."""
    daily, weekly, weekly_dates, test_pos = load_grid(asset)
    picks = np.linspace(0, len(test_pos) - 1, n_probe).astype(int)
    per_regime, failures = {}, []
    for regime in ("weekly", "daily"):
        times = []
        for i in picks:
            m = test_pos[i]
            _, daily_pos, _, daily_horizons = week_targets(weekly_dates, daily, m)
            train = weekly.iloc[:m + 1] if regime == "weekly" else daily.iloc[:daily_pos + 1]
            horizons = [1, 2, 3] if regime == "weekly" else daily_horizons
            t0 = time.time()
            try:
                if fn is None:                     # Naive : cout de reference
                    _ = float(np.std(np.diff(np.log(train.values))))
                else:
                    fn(train, horizons)
            except Exception as exc:
                failures.append(f"{name}/{regime}/origine {m}: {type(exc).__name__}: {exc}")
                continue
            times.append(time.time() - t0)
        per_regime[regime] = {"n_probe": len(times),
                              "median_s_per_origin": float(np.median(times)) if times else None,
                              "n_train_obs": int(len(train))}
    return {"per_regime": per_regime, "failures": failures}


def probe_nsdiff(asset: str, n_probe: int) -> dict:
    """NsDiff est train-once-forward : cout = fits FIXES + forecast x origines.
    Les deux sont mesures separement, sinon l'extrapolation lineaire du refit-
    par-origine lui serait appliquee a tort et sur-estimerait d'un facteur ~300."""
    daily, weekly, weekly_dates, test_pos = load_grid(asset)
    origin = pd.Timestamp(ORIGIN_START)
    train_w = weekly[weekly.index < origin]
    train_d = daily[daily.index < origin]

    out = {}
    for regime, train, horizon in (("weekly", train_w, HORIZON_WEEKLY),
                                   ("daily", train_d, HORIZON_DAILY)):
        nm.set_seed(42)
        t0 = time.time()
        model, mu, sd = nm.fit_nsdiff(train, horizon=horizon, epochs=NSDIFF_EPOCHS_W)
        fit_s = time.time() - t0

        z = (nm._log_returns((weekly if regime == "weekly" else daily).values.astype(float)) - mu) / sd
        times = []
        for i in np.linspace(0, len(test_pos) - 1, n_probe).astype(int):
            m = test_pos[i]
            _, daily_pos, _, daily_horizons = week_targets(weekly_dates, daily, m)
            pos = m if regime == "weekly" else daily_pos
            hs = [1, 2, 3] if regime == "weekly" else daily_horizons
            t0 = time.time()
            nm.forecast_from_fitted(model, z[:pos], mu, sd, float(weekly.iloc[m]),
                                    horizons=hs, n_samples=N_SAMPLES)
            times.append(time.time() - t0)
        out[regime] = {"fit_s": round(fit_s, 2), "n_train_obs": int(len(train)),
                       "median_s_per_origin_forecast": float(np.median(times))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-asset", default="SPY")
    p.add_argument("--n-probe", type=int, default=3)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    _, _, _, test_pos = load_grid(args.probe_asset)
    n_origins = len(test_pos)
    n_assets = len(args.assets)
    print(f"grille : {n_origins} origines x {n_assets} actifs x 2 regimes "
          f"= {n_origins * n_assets * 2} origines-cellules par modele\n")

    per_model, total_s = {}, 0.0
    for name, fn in PER_ORIGIN_MODELS.items():
        print(f"[{name}] chronometrage sur {args.n_probe} origines ...")
        probe = probe_per_origin(name, fn, args.probe_asset, args.n_probe)
        secs = 0.0
        for regime, r in probe["per_regime"].items():
            if r["median_s_per_origin"] is None:
                continue
            secs += r["median_s_per_origin"] * n_origins * n_assets
        per_model[name] = {"protocol": "refit par origine", **probe,
                           "extrapolated_s": secs, "extrapolated_h": secs / 3600.0}
        total_s += secs
        print(f"   -> {secs / 3600:.2f} h extrapolees"
              + (f"  [ECHECS: {probe['failures'][:1]}]" if probe["failures"] else ""))

    print(f"\n[NsDiff] chronometrage (train-once-forward) ...")
    ns = probe_nsdiff(args.probe_asset, args.n_probe)
    ns_s = sum((r["fit_s"] + r["median_s_per_origin_forecast"] * n_origins)
               for r in ns.values()) * N_SEEDS * n_assets
    per_model["NsDiff"] = {"protocol": f"train-once-forward, {N_SEEDS} graines, {N_SAMPLES} tirages",
                           "per_regime": ns, "extrapolated_s": ns_s,
                           "extrapolated_h": ns_s / 3600.0}
    total_s += ns_s
    print(f"   -> {ns_s / 3600:.2f} h extrapolees")

    # LSTM : mesure a part, c'est le modele dont le cout est le plus incertain
    print(f"\n[LSTM] chronometrage sur {min(args.n_probe, 2)} origines (le plus lourd) ...")
    lstm = probe_per_origin("LSTM", mh.forecast_horizons_lstm, args.probe_asset, min(args.n_probe, 2))
    lstm_s = sum(r["median_s_per_origin"] * n_origins * n_assets
                 for r in lstm["per_regime"].values() if r["median_s_per_origin"])
    per_model["LSTM"] = {"protocol": "refit par origine", **lstm,
                         "extrapolated_s": lstm_s, "extrapolated_h": lstm_s / 3600.0}
    total_s += lstm_s
    print(f"   -> {lstm_s / 3600:.2f} h extrapolees"
          + (f"  [ECHECS: {lstm['failures'][:1]}]" if lstm["failures"] else ""))

    # sous-ensemble minimal qui tranche l'hypothese primaire
    def _sub(model, regimes, n_a):
        e = per_model[model]
        if model == "NsDiff":
            return sum((e["per_regime"][r]["fit_s"]
                        + e["per_regime"][r]["median_s_per_origin_forecast"] * n_origins)
                       for r in regimes) * N_SEEDS * n_a
        return sum(e["per_regime"][r]["median_s_per_origin"] * n_origins * n_a
                   for r in regimes if e["per_regime"][r]["median_s_per_origin"])

    primary_s = _sub("NsDiff", ["weekly"], 1) + _sub("ARIMA-GARCH", ["weekly"], 1)
    econ_s = (_sub("NsDiff", ["weekly", "daily"], n_assets)
              + _sub("ARIMA-GARCH", ["weekly", "daily"], n_assets))

    payload = {
        "scope": "chantier B, etape prealable : chiffrage AVANT lancement",
        "grid": {"origin_start": ORIGIN_START, "n_origins": n_origins,
                 "n_assets": n_assets, "assets": args.assets,
                 "n_origin_cells_per_model": n_origins * n_assets * 2},
        "method": "chronometrage sur origines reelles (series gelees prices_v3), extrapolation "
                  "lineaire pour les modeles refit-par-origine ; NsDiff traite a part "
                  "(fits fixes + forecast x origines)",
        "probe_asset": args.probe_asset, "n_probe": args.n_probe,
        "per_model": per_model,
        "totals": {
            "full_regeneration_h": total_s / 3600.0,
            "budget_threshold_h": BUDGET_HOURS_ALL_IN,
            "fits_in_budget": bool(total_s / 3600.0 <= BUDGET_HOURS_ALL_IN),
            "primary_hypothesis_subset_h": primary_s / 3600.0,
            "primary_hypothesis_subset": "{NsDiff, ARIMA-GARCH} x {weekly} x {SPY} -- le minimum "
                                         "qui tranche l'hypothese primaire pre-declaree",
            "economic_arms_all_assets_h": econ_s / 3600.0,
            "economic_arms_scope": "{NsDiff, ARIMA-GARCH} x {weekly, daily} x tous actifs -- de "
                                   "quoi rejouer TOUT le volet economique et le match de "
                                   "calibration, sans les 4 modeles de reference classiques",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== chiffrage (heures extrapolees, grille complete 340 x 7 x 2) ===")
    for name, e in sorted(per_model.items(), key=lambda kv: -kv[1]["extrapolated_h"]):
        print(f"  {name:<14}{e['extrapolated_h']:>8.2f} h   ({e['protocol']})")
    t = payload["totals"]
    print(f"  {'TOTAL':<14}{t['full_regeneration_h']:>8.2f} h"
          f"   -> {'dans le budget' if t['fits_in_budget'] else 'HORS BUDGET, decoupage requis'} "
          f"(seuil {BUDGET_HOURS_ALL_IN} h)")
    print(f"\n  sous-ensemble hypothese primaire      {t['primary_hypothesis_subset_h']:>8.3f} h")
    print(f"  bras economiques, tous actifs         {t['economic_arms_all_assets_h']:>8.2f} h")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
