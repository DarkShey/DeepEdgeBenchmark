"""
nsdiff_refit_cadence.py -- chantier A3 : chiffrer l'asymetrie de protocole que
tout classement inter-modeles de ce repo traine depuis le debut.

LE PROBLEME. ARIMA-GARCH est refit a CHAQUE origine sur fenetre glissante ;
NsDiff est entraine UNE fois avant la premiere origine et roule 90 origines
(soit ~21 mois) sur ce fit. Quand GARCH gagne, on ne sait pas s'il gagne parce
qu'il est meilleur ou parce qu'il est plus frais. Le caveat est declare partout
(`matrice_paired_tests.comparison_1_ranking`, `nsdiff_vs_garch_w23`) et n'a
jamais ete mesure.

LA QUESTION POSEE, en deux temps :
  1. un refit PERIODIQUE de NsDiff (pas par origine -- personne ne reentrainerait
     un modele de diffusion chaque semaine en production) ameliore-t-il ses
     propres previsions, et de combien ?
  2. si oui, cela change-t-il le verdict vs GARCH sur l'actif concerne ?

C'est aussi, litteralement, la question operationnelle : a quelle cadence
faut-il reentrainer en production ?

PROTOCOLE. A chaque point de refit, le modele est reajuste sur TOUT l'historique
strictement anterieur (fenetre EXPANSIVE, pas glissante -- c'est ce que fait le
train-once-forward, en dosant simplement 1 seul point de refit ; changer aussi
la forme de la fenetre melangerait deux effets). Entre deux refits, on roule
forward exactement comme le bras de reference. Le bras `train_once` regenere ici
est donc, par construction, le cas `cadence = +inf`.

APPARIEMENT AU NIVEAU DU GENERATEUR ALEATOIRE, volontaire : a l'origine k, le
forecast est tire apres `set_seed(seed + k)` dans TOUS les bras -- exactement la
convention de `oos_nsdiff_daily_weekly.generate_nsdiff_asset`. Deux bras qui ne
different que par leur cadence de refit voient donc la meme sequence aleatoire a
chaque origine : l'ecart mesure est l'effet du refit, pas du bruit Monte-Carlo.

Budgets inchanges et declares : 40 epoques (plat, `NSDIFF_EPOCHS_W`), seq_len=30,
k_denoise=20, n_samples=200 -- la reference actee. Prix relus depuis les series
GELEES de `diffusion_multiseed_v2/prices/` : aucun appel reseau, donc aucun risque
de desalignement avec les bras deja produits.

Sortie : experiments/nsdiff_refit_cadence.json (+ checkpoints .npz reprenables)
Usage :
    python nsdiff_refit_cadence.py --smoke                        # plomberie
    python nsdiff_refit_cadence.py                                # SPY + BTC-USD, 5 graines
    python nsdiff_refit_cadence.py --assets SPY --cadences 13     # trimestriel seul
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

import calibration_tests as ct                                        # noqa: E402
import dashboard_d7_w1 as dash                                        # noqa: E402
import diffusion_headtohead as h2h                                    # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from backtest_rolling_tsdiffw import load_baseline_triplets, HORIZON_UNITS   # noqa: E402
from epoch_sweep import week_targets                                  # noqa: E402
from nsdiff_vs_garch_w23 import load_challenger                       # noqa: E402
from oos_nsdiff_daily_weekly import (                                 # noqa: E402
    DEFAULT_K_DENOISE, load_baseline_triplets_daily, nsdiff_engine, _standardized_returns,
)
from paired_test import paired_block_bootstrap_test                    # noqa: E402
from weekly_headtohead import HORIZON_WEEKLY, HORIZON_DAILY, build_weekly    # noqa: E402
from weekly_nsdiff_production import NSDIFF_EPOCHS_W                   # noqa: E402
from backtest_rolling_tsdiffw import _weekly_position                  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_refit_cadence.json"
CKPT_DIR = Path(__file__).resolve().parent / "nsdiff_refit_cadence"
PRICE_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"

SEEDS = [42, 43, 44, 45, 46]
DEFAULT_ASSETS = ["SPY", "BTC-USD"]     # le mieux couvert et le plus volatil : les deux extremes
# Cadences en SEMAINES entre deux refits. 13 = trimestriel, 4 = mensuel.
# `None` = train-once-forward (cadence infinie), le bras de reference.
DEFAULT_CADENCES = [13, 4]
N_SAMPLES = 200
POOL_SEED = 42
TARGET_COVERAGE = 0.95
ROW_COLS = ["asset", "frequence", "horizon", "horizon_unit", "cutoff_date",
            "target_date", "last_close", "y_pred", "y_lower", "y_upper", "y_true"]


# ── generation avec refit periodique ────────────────────────────────────────

def refit_points(n_origins: int, cadence) -> list:
    """Indices d'origine (0-based) auxquels on reentraine. `cadence=None` ->
    [0] seulement, c'est-a-dire le train-once-forward."""
    if cadence is None:
        return [0]
    if cadence < 1:
        raise ValueError(f"cadence doit valoir >= 1 semaine (ou None), recu {cadence}")
    return list(range(0, n_origins, int(cadence)))


def generate_with_cadence(asset: str, daily: pd.Series, weekly: pd.Series, weekly_dates: pd.Series,
                          origins_c: pd.DataFrame, origins_b: pd.DataFrame, seed: int,
                          cadence, epochs: int = NSDIFF_EPOCHS_W, n_samples: int = N_SAMPLES,
                          k_denoise: int = DEFAULT_K_DENOISE, verbose: bool = True) -> pd.DataFrame:
    """Boucle walk-forward a refit periodique. Aux points de refit, le modele est
    reajuste sur tout l'historique STRICTEMENT anterieur a l'origine courante ;
    ailleurs on roule forward sur le dernier fit, exactement comme le bras de
    reference. `cadence=None` reproduit le train-once-forward."""
    engine = nsdiff_engine()
    cutoffs = sorted(origins_c["cutoff_date"].unique())
    pos_of = {c: _weekly_position(weekly_dates, c) for c in cutoffs}
    if any(p is None for p in pos_of.values()):
        raise SystemExit(f"[{asset}] origines absentes de la grille W-FRI")
    cutoffs = sorted(cutoffs, key=lambda c: pos_of[c])
    points = set(refit_points(len(cutoffs), cadence))

    origins_c_by = dict(tuple(origins_c.groupby("cutoff_date")))
    origins_b_by = dict(tuple(origins_b.groupby("cutoff_date")))

    model_w = model_d = None
    mu_w = sd_w = mu_d = sd_d = None
    weekly_z = daily_z = None
    n_fits, fit_seconds = 0, 0.0
    rows = []

    for k, cutoff_date in enumerate(cutoffs):
        m = pos_of[cutoff_date]
        _, daily_pos, target_dates, daily_horizons = week_targets(weekly_dates, daily, m)
        last_price = float(weekly.iloc[m])

        if k in points:
            t_fit = time.time()
            # entrainement sur tout ce qui precede STRICTEMENT l'origine courante
            train_weekly = weekly.iloc[:m]
            train_daily = daily.iloc[:daily_pos]
            engine.module.set_seed(seed)
            model_w, mu_w, sd_w = engine.fit(train_weekly, HORIZON_WEEKLY, epochs, k_denoise)
            engine.module.set_seed(seed)
            model_d, mu_d, sd_d = engine.fit(train_daily, HORIZON_DAILY, epochs, k_denoise)
            weekly_z = _standardized_returns(weekly, mu_w, sd_w, engine.module)
            daily_z = _standardized_returns(daily, mu_d, sd_d, engine.module)
            n_fits += 1
            fit_seconds += time.time() - t_fit
            if verbose:
                print(f"[{asset}][seed={seed}][cadence={cadence}] refit #{n_fits} a l'origine "
                      f"{k + 1}/{len(cutoffs)} ({cutoff_date}) -- {len(train_weekly)} obs hebdo, "
                      f"{len(train_daily)} obs quotidiennes, {time.time() - t_fit:.0f}s")

        grp_c, grp_b = origins_c_by[cutoff_date], origins_b_by[cutoff_date]
        needed_h = sorted(int(h) for h in set(grp_c["horizon"]) | set(grp_b["horizon"]))
        h_d_needed = [daily_horizons[h - 1] for h in needed_h]

        # meme convention de graine que generate_nsdiff_asset : les bras ne
        # different que par la cadence, jamais par la sequence aleatoire
        engine.module.set_seed(seed + k)
        samples_w = engine.forecast(model_w, weekly_z[:m], mu_w, sd_w, last_price,
                                    needed_h, n_samples, k_denoise)
        engine.module.set_seed(seed + k)
        samples_d = engine.forecast(model_d, daily_z[:daily_pos], mu_d, sd_d, last_price,
                                    h_d_needed, n_samples, k_denoise)

        for h in needed_h:
            for regime, grp, s in (("weekly", grp_c, samples_w[h]),
                                   ("daily", grp_b, samples_d[daily_horizons[h - 1]])):
                sel = grp[grp["horizon"] == h]
                if not len(sel):
                    continue
                stored_target = sel["target_date"].iloc[0]
                if str(target_dates[h - 1].date()) != stored_target:
                    raise SystemExit(f"[{asset}] target_date incoherent a {cutoff_date} h={h}")
                lo, hi = (float(q) for q in np.quantile(s, [0.025, 0.975]))
                rows.append({
                    "asset": asset, "frequence": regime, "horizon": h,
                    "horizon_unit": HORIZON_UNITS[h], "cutoff_date": cutoff_date,
                    "target_date": stored_target, "last_close": last_price,
                    "y_pred": float(np.mean(s)), "y_lower": lo, "y_upper": hi,
                    "y_true": float(sel["y_true"].iloc[0]),
                })

    out = pd.DataFrame(rows)[ROW_COLS]
    out.attrs["n_fits"] = n_fits
    out.attrs["fit_seconds"] = round(fit_seconds, 1)
    return out


def run_arm(asset: str, seed: int, cadence, prices: dict, out_dir: Path, verbose=True) -> pd.DataFrame:
    tag = "trainonce" if cadence is None else f"w{cadence}"
    path = out_dir / f"{asset.replace('=', '_')}_seed{seed}_{tag}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        df = pd.DataFrame(json.loads(str(blob["rows"])))[ROW_COLS]
        df.attrs["n_fits"] = int(blob["n_fits"])
        df.attrs["fit_seconds"] = float(blob["fit_seconds"])
        print(f"[{asset}][seed={seed}][cadence={cadence}] checkpoint relu ({len(df)} lignes, "
              f"{df.attrs['n_fits']} fits)")
        return df

    daily, weekly, weekly_dates, origins_c, origins_b = prices[asset]
    df = generate_with_cadence(asset, daily, weekly, weekly_dates, origins_c, origins_b,
                               seed, cadence, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=json.dumps(df.to_dict(orient="list")),
                        n_fits=df.attrs["n_fits"], fit_seconds=df.attrs["fit_seconds"])
    return df


# ── mise en forme + tests ───────────────────────────────────────────────────

def enrich(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    r = df.copy()
    r["model"] = "NsDiff"
    r["horizon_type"] = "weekly"
    r["seed"] = seed
    r["sq_error"] = (r["y_pred"] - r["y_true"]) ** 2
    r["in_interval"] = ((r["y_true"] >= r["y_lower"]) & (r["y_true"] <= r["y_upper"])).astype(float)
    return h2h.with_winkler(r)


def compare_arms(refit: pd.DataFrame, base: pd.DataFrame, asset: str, regime: str,
                 horizon_unit: str) -> dict:
    """refit - train_once, apparie par origine. Metriques a MINIMISER : un
    mean_diff negatif favorise le refit."""
    def _sel(d):
        return (d[(d["frequence"] == regime) & (d["horizon_unit"] == horizon_unit)]
                .sort_values("cutoff_date").reset_index(drop=True))
    a, b = _sel(refit), _sel(base)
    if len(a) != len(b) or not (a["cutoff_date"].values == b["cutoff_date"].values).all():
        raise SystemExit(f"[{asset}/{regime}/{horizon_unit}] origines desalignees entre bras")

    def _test(x, y):
        t = paired_block_bootstrap_test(np.asarray(x) - np.asarray(y), block_length=3, seed=POOL_SEED)
        verdict = ("indistinguishable" if not t["significant_at_05"]
                   else ("refit_significantly_better" if t["mean_diff"] < 0 else "train_once_significantly_better"))
        return {**t, "verdict": verdict}

    return {
        "n": int(len(a)),
        "rmse_refit": float(np.sqrt(a["sq_error"].mean())),
        "rmse_train_once": float(np.sqrt(b["sq_error"].mean())),
        "rmse_test": _test(a["sq_error"], b["sq_error"]),
        "winkler_refit": float(a["winkler"].mean()),
        "winkler_train_once": float(b["winkler"].mean()),
        "winkler_test": _test(a["winkler"], b["winkler"]),
        "cov95_refit": float(a["in_interval"].mean()),
        "cov95_train_once": float(b["in_interval"].mean()),
        "coverage_gap_refit": ct.coverage_gap_block_test(a["in_interval"].values,
                                                         target=TARGET_COVERAGE, seed=POOL_SEED),
    }


def vs_garch(arm: pd.DataFrame, garch: pd.DataFrame, cache: dict, assets: list,
             horizon_units: list, label_a: str) -> dict:
    """Le verdict vs GARCH rejoue sur ce bras -- meme machinerie que la tache 7.
    Le bras est poole sur les graines (`seed_pooled_rows`), donc la question
    reste "un run a graine tiree au hasard bat-il GARCH ?"."""
    arm_a = pd.concat([arm, h2h.seed_pooled_rows(arm)], ignore_index=True)
    arm_b = h2h.broadcast_seeds(garch, sorted(arm["seed"].unique()))
    return {hu: h2h.run_match(arm_a, arm_b, cache, hu, [], assets, label_a, "ARIMA-GARCH")
            for hu in horizon_units}


def load_prices(assets: list) -> dict:
    origins_c_all = load_baseline_triplets(assets)
    origins_b_all = load_baseline_triplets_daily(assets)
    out = {}
    for asset in assets:
        frozen = PRICE_DIR / f"{asset.replace('=', '_')}.parquet"
        if not frozen.exists():
            raise SystemExit(f"prix geles absents pour {asset} ({frozen}) -- lancer d'abord "
                             "diffusion_multiseed_v2.py (refus de retelecharger : desalignerait les bras)")
        daily = pd.read_parquet(frozen)["close"]
        weekly, weekly_dates = build_weekly(daily)
        out[asset] = (daily, weekly, weekly_dates,
                      origins_c_all[origins_c_all["asset"] == asset].reset_index(drop=True),
                      origins_b_all[origins_b_all["asset"] == asset].reset_index(drop=True))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--cadences", type=int, nargs="+", default=DEFAULT_CADENCES,
                   help="ecart en SEMAINES entre deux refits (13=trimestriel, 4=mensuel)")
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out-dir", default=str(CKPT_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true", help="SPY, 1 graine, cadence trimestrielle seule")
    args = p.parse_args()

    assets, seeds, cadences = args.assets, args.seeds, args.cadences
    out_dir = Path(args.out_dir)
    if args.smoke:
        assets, seeds, cadences = ["SPY"], [42], [13]
        out_dir = out_dir.parent / "nsdiff_refit_cadence_smoke"

    t0 = time.time()
    prices = load_prices(assets)
    garch = load_challenger(assets, args.horizons)
    cache = v2.price_cache(pd.concat(
        [pd.DataFrame({"asset": [a], "target_date": [prices[a][2].max()]}) for a in assets],
        ignore_index=True))

    arms, cost = {}, {}
    for cadence in [None, *cadences]:
        tag = "train_once" if cadence is None else f"every_{cadence}w"
        frames, fits, secs = [], [], []
        for seed in seeds:
            for asset in assets:
                df = run_arm(asset, seed, cadence, prices, out_dir)
                frames.append(enrich(df, seed))
                fits.append(df.attrs["n_fits"])
                secs.append(df.attrs["fit_seconds"])
        arms[tag] = pd.concat(frames, ignore_index=True)
        cost[tag] = {
            "n_fits_per_asset_seed": int(np.mean(fits)),
            "fit_seconds_per_asset_seed": round(float(np.mean(secs)), 1),
            "fit_seconds_total": round(float(np.sum(secs)), 1),
            "cost_multiplier_vs_train_once": None,
        }
        print(f"\n[{tag}] {int(np.mean(fits))} refit(s) par (actif, graine), "
              f"{np.mean(secs):.0f}s de fit chacun")

    base_cost = cost["train_once"]["fit_seconds_per_asset_seed"] or 1.0
    for tag in cost:
        cost[tag]["cost_multiplier_vs_train_once"] = round(
            cost[tag]["fit_seconds_per_asset_seed"] / base_cost, 2)

    # 1. le refit ameliore-t-il NsDiff lui-meme ?
    improvement = {}
    for tag, arm in arms.items():
        if tag == "train_once":
            continue
        improvement[tag] = {}
        for seed in seeds:
            for asset in assets:
                for regime in ("weekly", "daily"):
                    for hu in args.horizons:
                        key = f"{asset}|{regime}|{hu}|seed{seed}"
                        improvement[tag][key] = compare_arms(
                            arm[(arm["seed"] == seed) & (arm["asset"] == asset)],
                            arms["train_once"][(arms["train_once"]["seed"] == seed)
                                               & (arms["train_once"]["asset"] == asset)],
                            asset, regime, hu)

    # 2. cela change-t-il le verdict vs GARCH ?
    garch_verdicts = {tag: vs_garch(arm, garch, cache, assets, args.horizons, f"NsDiff-{tag}")
                      for tag, arm in arms.items()}

    # correction de Holm sur la famille "amelioration par refit", par cadence et
    # par metrique : chaque cadence est une famille (actif x regime x horizon x graine)
    holm = {}
    for tag, cells in improvement.items():
        holm[tag] = {}
        for metric in ("rmse_test", "winkler_test"):
            fam = {k: v[metric] for k, v in cells.items()}
            holm[tag][metric] = mt.family_summary(mt.correct_family(fam))

    payload = {
        "question": "un refit PERIODIQUE de NsDiff (a) l'ameliore-t-il, (b) change-t-il le "
                    "verdict vs ARIMA-GARCH ? -- et a quel cout ?",
        "config": {
            "assets": assets, "seeds": seeds,
            "cadences_weeks": {"train_once": None, **{f"every_{c}w": c for c in cadences}},
            "window": "EXPANSIVE (tout l'historique strictement anterieur), comme le train-once",
            "epochs": NSDIFF_EPOCHS_W, "n_samples": N_SAMPLES, "k_denoise": DEFAULT_K_DENOISE,
            "rng_pairing": "set_seed(seed + k) a chaque origine dans TOUS les bras -- l'ecart "
                           "mesure est l'effet du refit, pas du bruit Monte-Carlo",
            "prices": "series gelees de diffusion_multiseed_v2/prices/ (aucun appel reseau)",
            "multiple_testing": "Holm par (cadence, metrique) sur la famille actif x regime x "
                                "horizon x graine",
        },
        "cost": cost,
        "refit_vs_train_once": improvement,
        "holm": holm,
        "vs_garch": garch_verdicts,
    }
    payload["config"]["elapsed_min"] = round((time.time() - t0) / 60, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== 1. le refit ameliore-t-il NsDiff ? (Holm par cadence/metrique) ===")
    for tag, metrics in holm.items():
        for metric, s in metrics.items():
            print(f"  {tag:<12} {metric:<14} m={s['m']:<4} {s['n_significant_raw']} rejets bruts -> "
                  f"{s['n_significant_holm']} apres Holm")
    print("\n=== 2. verdict poole global vs ARIMA-GARCH, par bras ===")
    for tag, hz in garch_verdicts.items():
        for hu in args.horizons:
            for regime in ("weekly", "daily"):
                g = hz[hu]["per_regime"][regime]["pooled_across_assets"].get("global", {})
                if g.get("status") != "tested":
                    continue
                print(f"  {tag:<12} {hu} {regime:<7} RMSE {g['skill_sqerror']['verdict']:<38} "
                      f"(p={g['skill_sqerror']['p_value']:.4f}) | Winkler "
                      f"{g['skill_winkler']['verdict']:<38} (p={g['skill_winkler']['p_value']:.4f})")
    print("\n=== 3. cout ===")
    for tag, c in cost.items():
        print(f"  {tag:<12} {c['n_fits_per_asset_seed']:>3} fit(s)/actif/graine  "
              f"{c['fit_seconds_per_asset_seed']:>7.0f}s  (x{c['cost_multiplier_vs_train_once']} "
              "vs train-once)")
    print(f"\n-> {args.out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
