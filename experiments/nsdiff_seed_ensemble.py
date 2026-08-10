"""
nsdiff_seed_ensemble.py -- tache 6 du BRIEF "NsDiff : consolider le verdict
daily vs weekly" : exploiter la variance inter-graines au lieu de la subir.

Principe : a chaque (actif, horizon, regime, origine), les 5 graines ont
produit 5 nuages predictifs de 200 tirages. L'ensemble les CONCATENE en un
seul nuage de 1000 tirages, dont on retire point (moyenne) et bandes
(quantiles 2.5%/97.5%) exactement comme le fait `oos_nsdiff_daily_weekly.
generate_nsdiff_asset` pour un run simple -- meme formule, plus de tirages
issus de plus de modeles. C'est une melange (mixture) des 5 predictives, pas
une moyenne de bornes : melanger les nuages ELARGIT la bande quand les
graines sont en desaccord, ce qui est precisement l'effet recherche cote
calibration, alors que moyenner les bornes l'aurait moyenne.

Reference de comparaison : la performance ATTENDUE d'un run a graine unique,
c'est-a-dire la metrique moyennee sur les 5 graines origine par origine (la
convention de pooling declaree dans `nsdiff_v2_data`). C'est la bonne
reference : la question n'est pas "l'ensemble bat-il la pire graine" mais
"l'ensemble vaut-il mieux que tirer une graine au hasard", qui est ce que
fait la production aujourd'hui.

Tests : bootstrap par blocs apparie par origine (`paired_test`, via
`calibration_tests.coverage_gap_block_test` pour la couverture et
`paired_block_bootstrap_test` directement pour sq_error/Winkler) ; aucun test
reimplemente. Cout nul en calcul de modele : aucun refit, les nuages
existent deja.

Sortie : experiments/nsdiff_seed_ensemble.json
Usage   : python nsdiff_seed_ensemble.py
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
import nsdiff_production_spec as spec                                 # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_seed_ensemble.json"
TARGET_COVERAGE = 0.95
POOL_SEED = 42
CELL_KEYS = ["model", "asset", "frequence", "horizon_type", "horizon_unit",
             "cutoff_date", "target_date"]


def build_ensemble_rows(rows: pd.DataFrame, samples: np.ndarray) -> pd.DataFrame:
    """Une ligne par (actif, regime, horizon, origine) : nuage des 5 graines
    concatene, puis point/bandes lus dessus avec la MEME formule que le run
    simple (moyenne, quantiles 2.5/97.5).

    La concatenation ET la lecture passent par `nsdiff_production_spec`, qui
    porte desormais la specification de la config candidate production (chantier
    A2) : la formule n'existe qu'a un seul endroit, et ce qui est teste ici est
    litteralement ce qui serait deploye."""
    rows = rows.reset_index(drop=True)
    out = []
    for keys, g in rows.groupby(CELL_KEYS, sort=False):
        forecast = spec.production_forecast(samples[g.index.to_numpy()])
        row = dict(zip(CELL_KEYS, keys))
        row.update({
            "y_pred": forecast["y_pred"], "y_lower": forecast["y_lower"], "y_upper": forecast["y_upper"],
            "y_true": float(g["y_true"].iloc[0]), "last_close": float(g["last_close"].iloc[0]),
            "n_seeds": int(g["seed"].nunique()), "n_samples_total": int(forecast["n_samples"]),
            "seed": -1,     # marqueur "ensemble", jamais une vraie graine
        })
        out.append(row)
    df = pd.DataFrame(out)
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return df


def _paired(a, b, label_better_when_negative: str, label_better_when_positive: str) -> dict:
    """a - b, bootstrappe par blocs. Convention: a = ensemble, b = graine
    unique attendue ; pour une METRIQUE A MINIMISER (sq_error, Winkler), un
    mean_diff negatif favorise l'ensemble."""
    test = paired_block_bootstrap_test(np.asarray(a) - np.asarray(b), block_length=3, seed=POOL_SEED)
    if not test["significant_at_05"]:
        verdict = "indistinguishable"
    else:
        verdict = label_better_when_negative if test["mean_diff"] < 0 else label_better_when_positive
    return {**test, "verdict": verdict}


def compare_regime(ens: pd.DataFrame, single: pd.DataFrame, asset: str, regime: str) -> dict:
    """ens / single : lignes d'UNE cellule (actif, regime, horizon), triees par
    origine. `single` porte les metriques moyennees sur les 5 graines."""
    e = ens.sort_values("cutoff_date")
    s = single.sort_values("cutoff_date")
    if not np.array_equal(e["cutoff_date"].values, s["cutoff_date"].values):
        raise SystemExit(f"[{asset}/{regime}] origines desalignees entre ensemble et graine unique")

    wink_e = dash.winkler_score(e["y_true"], e["y_lower"], e["y_upper"])
    return {
        "rmse_single_seed_expected": float(np.sqrt(s["sq_error"].mean())),
        "rmse_ensemble": float(np.sqrt(e["sq_error"].mean())),
        "sq_error_test": _paired(e["sq_error"].values, s["sq_error"].values,
                                 "ensemble_significantly_better", "single_seed_significantly_better"),
        "winkler_single_seed_expected": float(s["winkler"].mean()),
        "winkler_ensemble": float(wink_e.mean()),
        "winkler_test": _paired(wink_e, s["winkler"].values,
                                "ensemble_significantly_better", "single_seed_significantly_better"),
        "pi_width_single_seed_expected": float(s["pi_width"].mean()),
        "pi_width_ensemble": float((e["y_upper"] - e["y_lower"]).mean()),
        "cov95_single_seed_expected": float(s["in_interval"].mean()),
        "cov95_ensemble": float(e["in_interval"].mean()),
        "coverage_gap_single_seed": ct.coverage_gap_block_test(s["in_interval"].values,
                                                               target=TARGET_COVERAGE, seed=POOL_SEED),
        "coverage_gap_ensemble": ct.coverage_gap_block_test(e["in_interval"].values,
                                                            target=TARGET_COVERAGE, seed=POOL_SEED),
    }


def single_seed_expected(rows: pd.DataFrame) -> pd.DataFrame:
    """Metriques moyennees sur les graines, origine par origine."""
    r = rows.copy()
    r["winkler"] = dash.winkler_score(r["y_true"], r["y_lower"], r["y_upper"])
    r["pi_width"] = r["y_upper"] - r["y_lower"]
    return (r.groupby(CELL_KEYS, as_index=False)
            .agg(sq_error=("sq_error", "mean"), winkler=("winkler", "mean"),
                 in_interval=("in_interval", "mean"), pi_width=("pi_width", "mean"),
                 y_true=("y_true", "first"), last_close=("last_close", "first")))


def run_horizon(rows: pd.DataFrame, samples: np.ndarray, cache: dict, horizon_unit: str) -> dict:
    print(f"\n=== {horizon_unit} ===")
    mask = (rows["horizon_unit"] == horizon_unit).to_numpy()
    sub, sub_samples = rows[mask].reset_index(drop=True), samples[mask]

    ens_rows = build_ensemble_rows(sub, sub_samples)
    single = single_seed_expected(sub)

    per_cell = {}
    for asset in sorted(sub["asset"].unique()):
        per_cell[asset] = {
            regime: compare_regime(ens_rows[(ens_rows["asset"] == asset) & (ens_rows["frequence"] == regime)],
                                   single[(single["asset"] == asset) & (single["frequence"] == regime)],
                                   asset, regime)
            for regime in ("daily", "weekly")
        }

    # le verdict daily-vs-weekly rejoue SUR l'ensemble (memes briques que la tache 1)
    ens_pairs = dash.build_enriched_pairs(ens_rows, cache, horizon_unit=horizon_unit)
    res = {"per_cell": per_cell,
           "daily_vs_weekly_on_ensemble": v2.pooled_skill_test(ens_pairs, seed=POOL_SEED)}
    _print_horizon(horizon_unit, res)
    return res


def _print_horizon(horizon_unit: str, res: dict) -> None:
    print(f"\n[{horizon_unit}] ensemble 5 graines (1000 tirages) vs graine unique attendue")
    print(f"{'Actif':<10}{'regime':<8}{'RMSE 1g -> ens':>22}{'Cov95 1g -> ens':>20}"
          f"{'Winkler 1g -> ens':>26}{'test Winkler':>34}")
    for asset, cell in res["per_cell"].items():
        for regime, c in cell.items():
            print(f"{asset:<10}{regime:<8}"
                  f"{c['rmse_single_seed_expected']:>10.4g} -> {c['rmse_ensemble']:<9.4g}"
                  f"{c['cov95_single_seed_expected']:>9.3f} -> {c['cov95_ensemble']:<8.3f}"
                  f"{c['winkler_single_seed_expected']:>12.4g} -> {c['winkler_ensemble']:<11.4g}"
                  f"{c['winkler_test']['verdict']:>34}")
    g = res["daily_vs_weekly_on_ensemble"]["global"]
    if g.get("status") == "tested":
        print(f"\n[{horizon_unit}] daily vs weekly REJOUE sur l'ensemble -- skill Winkler global : "
              f"{g['skill_winkler']['verdict']} (p={g['skill_winkler']['p_value']:.4f}) | "
              f"skill RMSE : {g['skill_sqerror']['verdict']} (p={g['skill_sqerror']['p_value']:.4f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True)
    cache = v2.price_cache(rows)
    payload = {
        "config": {
            **v2.load_config(),
            "ensemble": "concatenation des nuages des 5 graines (5 x 200 = 1000 tirages), "
                        "point = moyenne, bandes = quantiles 2.5/97.5 -- meme lecture que pour "
                        "un run simple, aucune formule nouvelle",
            "reference": "performance ATTENDUE d'un run a graine unique = metrique moyennee sur "
                        "les 5 graines, origine par origine (pas la meilleure graine, pas la pire)",
            "cost": "aucun refit -- les nuages existent deja (artefact v2)",
        },
        "horizons": {h: run_horizon(rows, samples, cache, h) for h in args.horizons},
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
