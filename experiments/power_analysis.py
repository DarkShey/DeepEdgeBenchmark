"""
power_analysis.py -- chantier 2 du BRIEF "NsDiff : rouvrir la question
economique par le rapport edge/frais" : de combien d'origines a-t-on BESOIN, et
l'extension envisageable les fournit-elle ?

Le brief pose la question dans cet ordre, et c'est le bon : « avant de lancer,
calculer l'effective_n necessaire pour detecter un edge de 3 bps a 80 % de
puissance. Si la cible est inatteignable meme avec l'extension, le dire -- c'est
un resultat. »

METHODE, sans nouvelle hypothese distributionnelle. Le test du programme est le
bootstrap par blocs sur la moyenne des differences de PnL par origine. Son
erreur-type est donc DEJA estimee, blocs compris, par la brique existante --
inutile de postuler une loi ou une autocorrelation. Elle decroit en 1/sqrt(n) :

    n_requis = n_observe x [ SE_observe x (z_{1-a/2} + z_{1-b}) / delta ]^2

  * `SE_observe` : ecart-type bootstrap par blocs de la moyenne, sur les
    90 origines actuelles (`paired_test`, block_length=3) ;
  * `delta` : la taille d'effet a detecter ;
  * a = 0,05 bilateral, b = 0,20 (puissance 80 %) -- les valeurs declarees dans
    le brief.

DEUX CIBLES, et la distinction est decisive :
  1. `delta = 3 bps`, la cible declaree par le brief. Elle vient de l'edge POOLE
     ENTRE ACTIFS mesure au chantier B (+2 a +5 bps) ;
  2. `delta = |effet observe|` de la cellule elle-meme. Le chantier 1 a montre
     que l'edge par INSTRUMENT est bien plus grand que sa moyenne cross-actifs
     (jusqu'a 16 bps sur SPY/var_limit) : demander la puissance de detecter
     3 bps la ou l'effet en vaut 16 sur-estime massivement le besoin.
La colonne qui decide est la seconde ; la premiere est rapportee parce que c'est
celle que le brief nomme.

CORRECTION POUR TESTS MULTIPLES. La puissance calculee a alpha=0,05 est celle
d'un test isole. Le programme corrige par Holm sur des familles de 6, donc le
seuil effectif au rang le plus strict est 0,05/6 = 0,00833. Les deux colonnes
sont donnees : sans correction, et au seuil de Holm -- c'est la seconde qui dit
ce qu'il faut vraiment pour qu'un resultat SURVIVE.

CE QUE L'EXTENSION PEUT FOURNIR. Compte reel des origines hebdomadaires
disponibles dans les series GELEES, par date de depart, avec ce qu'il reste pour
l'entrainement en face -- parce qu'avancer la premiere origine de test prend les
donnees a l'entrainement, et qu'un NsDiff entraine sur 260 semaines n'est pas le
meme modele que sur 511.

Sortie : experiments/power_analysis.json
Usage   : python power_analysis.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import econ_backtest as eb                                            # noqa: E402
import real_fees as rf                                                # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402
from weekly_headtohead import ASSETS as ASSET_TICKERS, build_weekly   # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "power_analysis.json"
ECON_JSON = Path(__file__).resolve().parent / "nsdiff_edge_vs_fees.json"
PRICE_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "prices"

ALPHA = 0.05
HOLM_ALPHA = 0.05 / 6.0        # rang le plus strict d'une famille de 6
POWER = 0.80
TARGET_BPS = 3.0               # cible declaree par le brief
BLOCK_LENGTH = 3
N_ORIGINS_NOW = 90
POOL_SEED = 42
BPS = 1e4

# Candidats d'extension du panier, criteres d'inclusion DECLARES dans le brief :
# liquidite, historique >= aux actifs actuels, frais reels <= 5 bps aller-retour.
CANDIDATE_ASSETS = {
    "QQQ":  {"exposition": "actions technologiques (Nasdaq-100)", "vehicule": "ETF ou future NQ",
             "frais_ar_bps": 3.5, "historique": "1999-", "correle_a": "SPY (~0,9)"},
    "GLD":  {"exposition": "or", "vehicule": "ETF ou future GC",
             "frais_ar_bps": 4.0, "historique": "2004-", "correle_a": "aucun actif du panier"},
    "IEF":  {"exposition": "taux 7-10 ans (2e point de courbe)", "vehicule": "ETF",
             "frais_ar_bps": 4.0, "historique": "2002-", "correle_a": "TLT, ZN=F (~0,9)"},
    "EFA":  {"exposition": "actions hors Amerique du Nord", "vehicule": "ETF",
             "frais_ar_bps": 5.0, "historique": "2001-", "correle_a": "SPY (~0,8)"},
    "USO":  {"exposition": "petrole", "vehicule": "ETF ou future CL",
             "frais_ar_bps": 5.0, "historique": "2006-", "correle_a": "aucun actif du panier"},
}


def required_n(se_observed: float, delta: float, n_observed: int = N_ORIGINS_NOW,
               alpha: float = ALPHA, power: float = POWER) -> dict:
    """n d'origines requis pour detecter `delta` a la puissance voulue, en
    partant de l'erreur-type DEJA mesuree par le bootstrap par blocs."""
    if delta <= 0 or not np.isfinite(se_observed) or se_observed <= 0:
        return {"n_required": None, "effective_n_required": None, "factor_vs_now": None}
    z = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    factor = (se_observed * z / delta) ** 2
    n_req = n_observed * factor
    return {"n_required": float(n_req),
            "effective_n_required": float(n_req / BLOCK_LENGTH),
            "factor_vs_now": float(factor)}


def origins_available(assets: list, starts=("2015-01-01", "2018-01-01", "2020-01-01",
                                            "2022-01-01", "2024-10-18")) -> dict:
    """Origines hebdomadaires disponibles selon la date de premiere origine de
    test, ET ce qui reste en face pour l'entrainement. Lu sur les series GELEES."""
    out = {}
    for asset in assets:
        frozen = PRICE_DIR / f"{asset.replace('=', '_')}.parquet"
        if not frozen.exists():
            continue
        daily = pd.read_parquet(frozen)["close"]
        weekly, weekly_dates = build_weekly(daily)
        n_total = len(weekly)
        per_start = {}
        for s in starts:
            first = pd.Timestamp(s)
            n_train = int((weekly_dates < first).sum())
            n_test = int(((weekly_dates >= first).sum()) - 3)     # W+3 exige 3 semaines de marge
            per_start[s] = {"n_train_weeks": n_train, "n_test_origins": max(0, n_test),
                            "effective_n": max(0, n_test) // BLOCK_LENGTH}
        out[asset] = {"n_weekly_total": n_total,
                      "span": f"{weekly_dates.iloc[0].date()} -> {weekly_dates.iloc[-1].date()}",
                      "by_start": per_start}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--econ", default=str(ECON_JSON))
    p.add_argument("--target-bps", type=float, default=TARGET_BPS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    econ = json.loads(Path(args.econ).read_text())
    cells = econ["cells"]
    lvl = rf.DECISION_LEVEL

    # 1. puissance requise, cellule par cellule
    per_cell = {}
    for key, c in cells.items():
        if not key.endswith(f"|{lvl}") or "|filtered_direction|" in key:
            continue
        t = c["test_vs_garch"]
        if t.get("status") != "tested":
            continue
        # erreur-type de la moyenne, deduite de l'IC95 bootstrap deja calcule
        se = (t["ci95_hi"] - t["ci95_lo"]) / (2.0 * 1.959963984540054) * BPS
        observed = abs(t["mean_diff"] * BPS)
        per_cell[key] = {
            "instrument": c["instrument"], "edge_net_bps": c["edge_net_bps"],
            "observed_effect_bps": observed, "se_bps": se,
            "p_value": t["p_value"], "n_now": t["n"],
            "for_declared_target_3bps": required_n(se, args.target_bps),
            "for_observed_effect": required_n(se, observed),
            "for_observed_effect_at_holm": required_n(se, observed, alpha=HOLM_ALPHA),
        }

    # 2. ce que l'extension peut fournir
    availability = origins_available(list(ASSET_TICKERS.values()))

    # 3. confrontation : les cellules a edge net positif sont-elles atteignables ?
    ref = availability.get("SPY", {}).get("by_start", {})
    reachable = {}
    for start, info in ref.items():
        n_max = info["n_test_origins"]
        ok_declared = [k for k, v in per_cell.items()
                       if (v["for_declared_target_3bps"]["n_required"] or 1e18) <= n_max]
        ok_observed = [k for k, v in per_cell.items()
                       if v["edge_net_bps"] > 0
                       and (v["for_observed_effect_at_holm"]["n_required"] or 1e18) <= n_max]
        reachable[start] = {
            "n_test_origins": n_max, "effective_n": info["effective_n"],
            "n_train_weeks": info["n_train_weeks"],
            "n_cells_detectable_at_3bps": len(ok_declared),
            "n_cells_detectable_at_observed_effect_under_holm": len(ok_observed),
            "cells_detectable_under_holm": sorted(ok_observed)[:12],
        }

    payload = {
        "question": "combien d'origines faut-il pour trancher, et l'extension les fournit-elle ?",
        "config": {
            "method": "n_requis = n_observe x [SE_observe x (z_{1-a/2} + z_{1-b}) / delta]^2 ; "
                      "SE_observe vient du bootstrap PAR BLOCS deja utilise (aucune nouvelle "
                      "hypothese de loi ni d'autocorrelation)",
            "alpha": ALPHA, "alpha_holm_rank1": HOLM_ALPHA, "power": POWER,
            "block_length": BLOCK_LENGTH, "n_origins_now": N_ORIGINS_NOW,
            "declared_target_bps": args.target_bps,
            "why_two_targets": "3 bps est l'edge POOLE ENTRE ACTIFS du chantier B ; l'edge par "
                               "INSTRUMENT mesure au chantier 1 est bien plus grand (jusqu'a "
                               "16 bps), donc exiger la puissance de detecter 3 bps sur-estime "
                               "massivement le besoin",
            "inclusion_criteria": "liquidite, historique >= aux actifs actuels, frais reels "
                                  "<= 5 bps aller-retour (declares dans le brief)",
        },
        "per_cell": per_cell,
        "origins_available": availability,
        "reachability_by_start_date": reachable,
        "candidate_assets": CANDIDATE_ASSETS,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print(f"=== puissance requise (alpha={ALPHA}, puissance={POWER:.0%}), "
          f"cellules a edge net POSITIF, cout '{lvl}' ===")
    print(f"{'cellule':<44}{'edge net':>10}{'SE':>8}{'n pour 3 bps':>14}"
          f"{'n pour effet obs.':>19}{'n sous Holm':>13}")
    pos = sorted((k for k, v in per_cell.items() if v["edge_net_bps"] > 0),
                 key=lambda k: -per_cell[k]["edge_net_bps"])
    for k in pos[:14]:
        v = per_cell[k]
        f = lambda d: ("inatteignable" if d["n_required"] is None or d["n_required"] > 1e5
                       else f"{d['n_required']:.0f}")
        print(f"{k.replace('|central', ''):<44}{v['edge_net_bps']:>10.2f}{v['se_bps']:>8.2f}"
              f"{f(v['for_declared_target_3bps']):>14}{f(v['for_observed_effect']):>19}"
              f"{f(v['for_observed_effect_at_holm']):>13}")

    print(f"\n=== origines disponibles selon la date de premiere origine (SPY, series gelees) ===")
    print(f"{'depart':<14}{'origines test':>15}{'effective_n':>13}{'semaines train':>16}"
          f"{'cellules detectables sous Holm':>32}")
    for s, r in reachable.items():
        print(f"{s:<14}{r['n_test_origins']:>15}{r['effective_n']:>13}{r['n_train_weeks']:>16}"
              f"{r['n_cells_detectable_at_observed_effect_under_holm']:>32}")

    print(f"\n=== candidats d'extension du panier (criteres du brief) ===")
    for a, c in CANDIDATE_ASSETS.items():
        ok = c["frais_ar_bps"] <= 5.0
        print(f"  {a:<5}{c['exposition']:<38}{c['frais_ar_bps']:>5.1f} bps  "
              f"{'retenu' if ok else 'ecarte':<8} | correle a : {c['correle_a']}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
