"""
nsamples_sweep.py -- chantier 3 du BRIEF "NsDiff : rouvrir la question
economique par le rapport edge/frais" : fermer la reserve « 200 tirages est un
point de fonctionnement raisonne, pas un optimum certifie ».

CE QUE LE BALAYAGE MESURE. Les bornes de NsDiff ne sont pas produites par le
modele : elles sont LUES sur le nuage, en quantiles empiriques 2.5/97.5. Le
budget de tirages N est donc un parametre d'ESTIMATION, pas de modele --
augmenter N ne change pas la loi predictive, seulement la precision avec
laquelle on en lit les bords. La question est : a partir de quel N la lecture
a-t-elle convergee ?

METHODE : SOUS-ECHANTILLONNAGE EMBOITE, pas cinq runs independants.
Un artefact unique a 800 tirages est genere (memes graines, memes epoques,
memes prix geles, meme boucle -- seul `n_samples` change), puis chaque budget
N est evalue en decoupant les 800 tirages de chaque nuage en `800 // N` blocs
DISJOINTS, en lisant les metriques sur chaque bloc, et en moyennant.

  * Pourquoi disjoint et non « les N premiers » : les tirages sont i.i.d.
    conditionnellement au modele, donc tout bloc de taille N est un echantillon
    valide de budget N -- mais n'en prendre qu'un seul par nuage laisserait du
    bruit Monte-Carlo qui se confondrait avec l'effet de N. Moyenner les
    `800 // N` blocs disjoints (16 a N=50, 1 a N=800) l'elimine sans reutiliser
    un seul tirage.
  * Pourquoi pas cinq runs separes : deux runs a budgets differents tirent des
    sequences aleatoires differentes, donc leur ecart melange l'effet de N et le
    bruit de tirage. Ici tous les budgets sont lus SUR LE MEME materiel.

Consequence assumee : a N=800 il n'y a qu'un bloc, donc la valeur de reference
porte tout le bruit Monte-Carlo residuel d'un nuage de 800 -- c'est precisement
pourquoi le critere d'arret ci-dessous se compare a un DEMI ECART-TYPE bootstrap
et non a l'egalite.

CRITERE D'ARRET, declare dans le brief avant tout run : le budget retenu pour un
regime est le plus petit N pour lequel la couverture ET le Winkler sont a moins
d'un demi ecart-type bootstrap de leur valeur a N=800. L'ecart-type est celui de
la moyenne par origine, estime par le bootstrap PAR BLOCS deja utilise partout
(`paired_test.paired_block_bootstrap_test`, block_length=3) -- aucune nouvelle
machinerie de reechantillonnage.

HYPOTHESE A TRANCHER, declaree dans `NOTE_nsdiff_consolidation_daily_vs_weekly.md`
§1 et jamais testee : le nuage DAILY, propage sur ~5 pas quotidiens et a queues
plus lourdes, convergerait plus lentement que le nuage WEEKLY (il gagnait +4,2
points de couverture en passant de 50 a 200 tirages, contre +2,4 au weekly). Le
balayage la teste directement, en comparant les deux courbes de convergence.

BORNE THEORIQUE, pour lire les courbes : le biais du quantile empirique est
mecanique et se calcule sans simuler (`oos_reference_audit.expected_quantile_level`).
Couverture reelle d'un PI etiquete 95 % : 91,3 % a N=50, 94,05 % a 200, 94,6 % a
500, 94,8 % a 800. Toute convergence observee PLUS LENTE que cette borne vient du
modele (queues, propagation), pas de l'estimateur.

Sortie : experiments/nsamples_sweep.json
Usage   : python nsamples_sweep.py
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

import dashboard_d7_w1 as dash                                        # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from oos_reference_audit import two_sided_coverage                    # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsamples_sweep.json"
N800_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_n800" / "NsDiff"
BUDGETS = (50, 100, 200, 400, 800)
Q_LOW, Q_HIGH = 0.025, 0.975
TARGET = 0.95
BLOCK_LENGTH = 3
POOL_SEED = 42
STOP_TOL_SD = 0.5          # critere d'arret : un DEMI ecart-type bootstrap
REGIMES = ("weekly", "daily")


def block_metrics(samples: np.ndarray, y_true: np.ndarray, last_close: np.ndarray,
                  budget: int) -> dict:
    """Metriques par origine a budget `budget`, moyennees sur les `n_blocks`
    blocs disjoints de chaque nuage. `samples` : (n_rows, 800)."""
    n_total = samples.shape[1]
    n_blocks = n_total // budget
    if n_blocks < 1:
        raise ValueError(f"budget {budget} > {n_total} tirages disponibles")

    cov = np.zeros((n_blocks, len(y_true)))
    width = np.zeros_like(cov)
    winkler = np.zeros_like(cov)
    for b in range(n_blocks):
        cloud = samples[:, b * budget:(b + 1) * budget]
        lo = np.quantile(cloud, Q_LOW, axis=1)
        hi = np.quantile(cloud, Q_HIGH, axis=1)
        cov[b] = ((y_true >= lo) & (y_true <= hi)).astype(float)
        width[b] = (hi - lo) / last_close * 100.0
        winkler[b] = dash.winkler_score(y_true, lo, hi)
    return {"n_blocks": int(n_blocks),
            "cov_per_origin": cov.mean(axis=0),
            "width_per_origin": width.mean(axis=0),
            "winkler_per_origin": winkler.mean(axis=0)}


def bootstrap_sd(series: np.ndarray) -> float:
    """Ecart-type bootstrap PAR BLOCS de la moyenne de `series` (une valeur par
    origine, ordre chronologique). Deduit de l'IC95 renvoye par la brique
    existante -- aucun reechantillonnage reimplemente : (hi - lo) / (2 x 1.96)."""
    t = paired_block_bootstrap_test(series, block_length=min(BLOCK_LENGTH, series.size),
                                    seed=POOL_SEED)
    return float((t["ci95_hi"] - t["ci95_lo"]) / (2.0 * 1.959963984540054))


def sweep_cell(rows: pd.DataFrame, samples: np.ndarray, regime: str, horizon_unit: str,
               budgets=BUDGETS) -> dict:
    """Une cellule = (regime, horizon), tous actifs et toutes graines confondus.
    Les origines sont ordonnees par (actif, graine, date) : le bootstrap par
    blocs y respecte donc la contiguite temporelle a l'interieur de chaque
    (actif, graine), qui est ce que les blocs doivent capturer."""
    mask = ((rows["frequence"] == regime) & (rows["horizon_unit"] == horizon_unit)).to_numpy()
    sub = rows[mask].sort_values(["asset", "seed", "cutoff_date"])
    idx = sub.index.to_numpy()
    s = samples[idx]
    y_true = sub["y_true"].to_numpy(dtype=float)
    last_close = sub["last_close"].to_numpy(dtype=float)

    per_budget = {}
    for n in budgets:
        m = block_metrics(s, y_true, last_close, n)
        per_budget[n] = {
            "n_blocks_averaged": m["n_blocks"],
            "cov95": float(m["cov_per_origin"].mean()),
            "pi_width_pct_of_price": float(m["width_per_origin"].mean()),
            "winkler_mean": float(m["winkler_per_origin"].mean()),
            "_cov_series": m["cov_per_origin"],
            "_wink_series": m["winkler_per_origin"],
            "theoretical_coverage_of_a_perfect_model": round(two_sided_coverage(n), 6),
        }

    ref = per_budget[max(budgets)]
    sd_cov = bootstrap_sd(ref["_cov_series"])
    sd_wink = bootstrap_sd(ref["_wink_series"])
    tol_cov, tol_wink = STOP_TOL_SD * sd_cov, STOP_TOL_SD * sd_wink

    converged = []
    for n in budgets:
        b = per_budget[n]
        ok_cov = abs(b["cov95"] - ref["cov95"]) <= tol_cov
        ok_wink = abs(b["winkler_mean"] - ref["winkler_mean"]) <= tol_wink
        b["within_half_sd_coverage"] = bool(ok_cov)
        b["within_half_sd_winkler"] = bool(ok_wink)
        b["converged"] = bool(ok_cov and ok_wink)
        if b["converged"]:
            converged.append(n)
        for k in ("_cov_series", "_wink_series"):
            b.pop(k)

    return {
        "n_origins": int(len(sub)),
        "reference_budget": max(budgets),
        "bootstrap_sd_at_reference": {"cov95": sd_cov, "winkler": sd_wink},
        "stop_tolerance": {"cov95": tol_cov, "winkler": tol_wink,
                           "rule": f"|metrique(N) - metrique(800)| <= {STOP_TOL_SD} x ecart-type bootstrap"},
        "by_budget": {str(n): per_budget[n] for n in budgets},
        "smallest_converged_budget": min(converged) if converged else None,
    }


def convergence_gain(cell: dict, a: int, b: int) -> dict:
    """Gain de couverture en passant du budget a au budget b, en POINTS."""
    x = cell["by_budget"][str(a)]["cov95"]
    y = cell["by_budget"][str(b)]["cov95"]
    return {"from": a, "to": b, "cov95_gain_points": round(100 * (y - x), 3)}


# ── le bras ENSEMBLE : la configuration reellement deployee ─────────────────

ENSEMBLE_BUDGETS = (200, 400, 1000, 2000, 4000)


def interleave_seeds(rows: pd.DataFrame, samples: np.ndarray, regime: str,
                     horizon_unit: str) -> tuple:
    """Nuage d'ENSEMBLE par origine : les nuages des 5 graines ENTRELACES
    (tirage i de la graine 42, puis de la 43, ... puis tirage i+1 de la 42, ...).

    L'entrelacement n'est pas cosmetique. Concatener graine par graine ferait
    qu'un bloc contigu de 800 tirages serait EXACTEMENT une graine -- on
    mesurerait alors un run simple en croyant mesurer l'ensemble. Entrelace, tout
    bloc contigu de taille multiple de 5 contient le meme nombre de tirages de
    chaque graine : c'est un melange stratifie des 5 lois predictives, ce qu'est
    la configuration production (`nsdiff_production_spec`).
    """
    sub = rows[(rows["frequence"] == regime) & (rows["horizon_unit"] == horizon_unit)]
    keys = ["asset", "cutoff_date"]
    clouds, y_true, last_close = [], [], []
    for _, g in sub.groupby(keys, sort=True):
        g = g.sort_values("seed")
        block = samples[g.index.to_numpy()]                 # (n_seeds, n_draws)
        clouds.append(block.T.reshape(-1))                  # entrelace : (n_draws*n_seeds,)
        y_true.append(float(g["y_true"].iloc[0]))
        last_close.append(float(g["last_close"].iloc[0]))
    return (np.stack(clouds), np.asarray(y_true, dtype=float),
            np.asarray(last_close, dtype=float))


def sweep_ensemble(rows: pd.DataFrame, samples: np.ndarray, regime: str,
                   horizon_unit: str, budgets=ENSEMBLE_BUDGETS) -> dict:
    """Meme balayage, mais sur le nuage d'ensemble. Repond a la question que le
    balayage par graine ne pose pas : la CONFIGURATION PRODUCTION (5 x 200 =
    1000 tirages) est-elle, elle, convergee ?"""
    clouds, y_true, last_close = interleave_seeds(rows, samples, regime, horizon_unit)
    budgets = tuple(b for b in budgets if b <= clouds.shape[1])
    per_budget = {}
    for n in budgets:
        m = block_metrics(clouds, y_true, last_close, n)
        per_budget[n] = {"n_blocks_averaged": m["n_blocks"],
                         "cov95": float(m["cov_per_origin"].mean()),
                         "pi_width_pct_of_price": float(m["width_per_origin"].mean()),
                         "winkler_mean": float(m["winkler_per_origin"].mean()),
                         "_cov": m["cov_per_origin"], "_wk": m["winkler_per_origin"]}
    ref = per_budget[max(budgets)]
    tol_cov = STOP_TOL_SD * bootstrap_sd(ref["_cov"])
    tol_wk = STOP_TOL_SD * bootstrap_sd(ref["_wk"])
    converged = []
    for n in budgets:
        b = per_budget[n]
        b["converged"] = bool(abs(b["cov95"] - ref["cov95"]) <= tol_cov
                              and abs(b["winkler_mean"] - ref["winkler_mean"]) <= tol_wk)
        if b["converged"]:
            converged.append(n)
        for k in ("_cov", "_wk"):
            b.pop(k)
    return {"n_origins": int(len(y_true)), "reference_budget": max(budgets),
            "stop_tolerance": {"cov95": tol_cov, "winkler": tol_wk},
            "by_budget": {str(n): per_budget[n] for n in budgets},
            "smallest_converged_budget": min(converged) if converged else None,
            "production_budget": 1000,
            "production_is_converged": bool(1000 in converged)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", default=str(N800_DIR))
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.artifact), model="NsDiff")
    rows = rows.reset_index(drop=True)
    n_draws = samples.shape[1]
    print(f"artefact : {len(rows)} lignes x {n_draws} tirages, graines {v2.seeds(rows)}")
    if n_draws != max(args.budgets):
        raise SystemExit(f"artefact a {n_draws} tirages, budget max demande {max(args.budgets)} -- "
                         "le sous-echantillonnage emboite exige que l'artefact PORTE le budget max")

    cells, ensemble = {}, {}
    for regime in REGIMES:
        for hu in args.horizons:
            cells[f"{regime}|{hu}"] = sweep_cell(rows, samples, regime, hu, tuple(args.budgets))
            ensemble[f"{regime}|{hu}"] = sweep_ensemble(rows, samples, regime, hu)

    # l'hypothese declaree : le daily converge-t-il plus lentement que le weekly ?
    hypothesis = {}
    for hu in args.horizons:
        d, w = cells[f"daily|{hu}"], cells[f"weekly|{hu}"]
        hypothesis[hu] = {
            "daily_gain_50_to_200_points": convergence_gain(d, 50, 200)["cov95_gain_points"],
            "weekly_gain_50_to_200_points": convergence_gain(w, 50, 200)["cov95_gain_points"],
            "daily_gain_200_to_800_points": convergence_gain(d, 200, 800)["cov95_gain_points"],
            "weekly_gain_200_to_800_points": convergence_gain(w, 200, 800)["cov95_gain_points"],
            "daily_smallest_converged": d["smallest_converged_budget"],
            "weekly_smallest_converged": w["smallest_converged_budget"],
        }
        h = hypothesis[hu]
        h["daily_converges_slower"] = bool(
            (h["daily_gain_200_to_800_points"] > h["weekly_gain_200_to_800_points"])
            and (h["daily_smallest_converged"] or 0) >= (h["weekly_smallest_converged"] or 0))

    # budget retenu par regime : le plus grand des « plus petits N converges »
    # a travers les horizons -- un budget de production doit tenir au pire des
    # horizons servis, pas au meilleur.
    retained = {}
    for regime in REGIMES:
        per_h = [cells[f"{regime}|{hu}"]["smallest_converged_budget"] for hu in args.horizons]
        retained[regime] = None if any(x is None for x in per_h) else max(per_h)

    payload = {
        "question": "200 tirages est-il un optimum, ou seulement un point de fonctionnement ?",
        "config": {
            "budgets": list(args.budgets), "artifact": str(args.artifact),
            "method": "sous-echantillonnage EMBOITE : un artefact unique a 800 tirages, decoupe en "
                      "800//N blocs disjoints par nuage, metriques moyennees sur les blocs -- tous "
                      "les budgets sont lus sur le MEME materiel, l'ecart entre eux ne melange pas "
                      "l'effet de N avec du bruit de tirage",
            "stop_rule": f"plus petit N dont la couverture ET le Winkler sont a moins de "
                         f"{STOP_TOL_SD} ecart-type bootstrap de leur valeur a {max(args.budgets)}",
            "bootstrap": "paired_test.paired_block_bootstrap_test (block_length=3), ecart-type "
                         "deduit de l'IC95 -- aucun reechantillonnage reimplemente",
            "theoretical_bound": "couverture d'un modele PARFAIT lue sur N tirages : "
                                 + ", ".join(f"N={n} -> {100 * two_sided_coverage(n):.2f} %"
                                             for n in args.budgets),
        },
        "cells": cells,
        "ensemble_cells": ensemble,
        "hypothesis_daily_converges_slower": hypothesis,
        "retained_budget_by_regime": retained,
        "production_config_converged": {k: v["production_is_converged"] for k, v in ensemble.items()},
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for regime in REGIMES:
        print(f"\n=== regime {regime} ===")
        print(f"{'cellule':<12}{'N':>6}{'blocs':>7}{'Cov95':>9}{'(modele parfait)':>18}"
              f"{'largeur %':>12}{'Winkler':>12}{'converge':>10}")
        for hu in args.horizons:
            c = cells[f"{regime}|{hu}"]
            for n in args.budgets:
                b = c["by_budget"][str(n)]
                print(f"{hu if n == args.budgets[0] else '':<12}{n:>6}{b['n_blocks_averaged']:>7}"
                      f"{b['cov95']:>9.4f}{100 * b['theoretical_coverage_of_a_perfect_model']:>17.2f}%"
                      f"{b['pi_width_pct_of_price']:>12.2f}{b['winkler_mean']:>12.4g}"
                      f"{('oui' if b['converged'] else '-'):>10}")
            print(f"{'':<12}{'->':>6} plus petit budget converge : {c['smallest_converged_budget']} "
                  f"(tolerance : cov +/-{c['stop_tolerance']['cov95']:.4f}, "
                  f"Winkler +/-{c['stop_tolerance']['winkler']:.4g})")

    print("\n=== hypothese : le daily converge-t-il plus lentement que le weekly ? ===")
    for hu, h in hypothesis.items():
        print(f"  {hu}  gain 50->200 : daily {h['daily_gain_50_to_200_points']:+.2f} pts / "
              f"weekly {h['weekly_gain_50_to_200_points']:+.2f} pts | "
              f"gain 200->800 : daily {h['daily_gain_200_to_800_points']:+.2f} / "
              f"weekly {h['weekly_gain_200_to_800_points']:+.2f} | "
              f"plus petit N converge : daily {h['daily_smallest_converged']} / "
              f"weekly {h['weekly_smallest_converged']}  -> "
              f"{'CONFIRMEE' if h['daily_converges_slower'] else 'non confirmee'}")

    print("\n=== bras ENSEMBLE : la configuration reellement deployee (5 graines entrelacees) ===")
    print(f"{'cellule':<14}{'N':>7}{'blocs':>7}{'Cov95':>9}{'largeur %':>12}{'Winkler':>12}{'converge':>10}")
    for regime in REGIMES:
        for hu in args.horizons:
            e = ensemble[f"{regime}|{hu}"]
            for n, b in e["by_budget"].items():
                lbl = f"{regime}|{hu}" if n == list(e["by_budget"])[0] else ""
                star = " <- production" if int(n) == 1000 else ""
                print(f"{lbl:<14}{n:>7}{b['n_blocks_averaged']:>7}{b['cov95']:>9.4f}"
                      f"{b['pi_width_pct_of_price']:>12.2f}{b['winkler_mean']:>12.4g}"
                      f"{('oui' if b['converged'] else '-'):>10}{star}")
            print(f"{'':<14}{'->':>7} plus petit budget converge : {e['smallest_converged_budget']}"
                  f" | production (1000) convergee : {e['production_is_converged']}")

    print(f"\n=== budget retenu par regime, PAR GRAINE (au pire horizon servi) ===")
    for regime, n in retained.items():
        print(f"  {regime:<8} {n}")
    n_ok = sum(v["production_is_converged"] for v in ensemble.values())
    print(f"\n=== configuration production (ensemble 1000 tirages) : convergee sur "
          f"{n_ok}/{len(ensemble)} cellules ===")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
