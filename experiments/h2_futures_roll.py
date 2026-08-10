"""
h2_futures_roll.py -- chantier H2 du BRIEF « regeneration oos et famille 3 » :
l'edge net survit-il quand on paie le ROULEMENT des futures ?

LA RESERVE QU'ON LEVE. Les deux vehicules « future » du benchmark (SPY-ES pour
l'exposition actions, ZN-FUT pour les taux) etaient jusqu'ici factures a leur
seul aller-retour outright, 1,5 bps -- soit le vehicule le moins cher du panier,
et celui qui portait l'essentiel de l'espoir economique du programme. Un contrat
a terme expire : le tenir au-dela d'une echeance coute un roulement de plus. La
simplification etait declaree et FAVORABLE au future ; H2 la chiffre.

CE QUE LE ROULEMENT DEPLACE, ET CE QU'IL NE DEPLACE PAS. Les deux bras tradent le
MEME instrument : ils paient donc le meme tarif de roulement. Mais ils ne le
paient pas dans la meme quantite -- le cout est proportionnel a |position|, et les
deux bras n'ont pas la meme exposition moyenne. Le roulement ne s'annule donc
PAS dans l'edge differentiel : il penalise le bras le plus expose. C'est
exactement ce que ce script mesure, cellule par cellule.

TROIS CHIFFRES PAR CELLULE, et le troisieme est le seul qui decide :
  * PnL net de chaque bras, sans puis avec roulement ;
  * edge net differentiel (NsDiff - GARCH), sans puis avec ;
  * le test apparie par blocs sur les PnL AVEC roulement -- meme moteur, meme
    longueur de bloc, meme graine que partout ailleurs.

BORDEREAUX REELS. Le brief demande aussi de « remplacer la grille de frais
hypothetique par des bordereaux reels des qu'ils existent ». Ils n'existent pas :
aucun releve de courtage n'est verse au depot. La grille reste donc declaree comme
hypothese, a son emplacement unique (`real_fees.INSTRUMENTS`) -- et ce script
mesure la sensibilite de la conclusion aux trois niveaux bas/central/haut, ce qui
est la seule reponse honnete disponible tant que les bordereaux manquent.

Sortie : experiments/h2_futures_roll.json
Usage   : python h2_futures_roll.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import grid2020_tests as g2t                                          # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import real_fees as rf                                                # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "h2_futures_roll.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
BPS = 1e4
HORIZON_WEEKS = {"W+1": 1, "W+2": 2, "W+3": 3}
STRATEGIES = ("inverse_width", "var_limit")     # les deux familles qui prennent des positions
REGIME = "weekly"                                # perimetre de la phase W


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples, garch = g2t.load_arms(Path(args.grid_dir))
    arms = g2t.align({"nsdiff_ensemble": g2t.nsdiff_bands(rows, samples, 0.95),
                      "garch": g2t.garch_bands(garch, 0.95)})

    cells, fam = {}, {}
    for inst in sorted(rf.ROLLED_INSTRUMENTS):
        asset = rf.INSTRUMENTS[inst]["asset"]
        for strategy in STRATEGIES:
            for hu, hw in HORIZON_WEEKS.items():
                data = {n: g2t.cell(df, asset, REGIME, hu) for n, df in arms.items()}
                if data["garch"]["y_true"].size == 0:
                    continue
                out = {"instrument": inst, "asset": asset, "strategy": strategy,
                       "horizon_unit": hu, "horizon_weeks": hw,
                       "round_trip_bps": rf.round_trip_bps(inst),
                       "roll_cost_bps": rf.roll_cost_bps(inst, hw),
                       "total_round_trip_bps": rf.total_round_trip_bps(inst, hw),
                       "roll_share_of_cost": rf.roll_cost_bps(inst, hw)
                                             / rf.total_round_trip_bps(inst, hw),
                       "par_niveau": {}}
                for level in rf.LEVELS:
                    dec = {tag: {n: g2t.decompose(strategy, d, cost)
                                 for n, d in data.items()}
                           for tag, cost in (("sans_roll", rf.one_way_bps(inst, level)),
                                             ("avec_roll", rf.one_way_total_bps(inst, hw, level)))}
                    blk = {}
                    for tag, dd in dec.items():
                        ns, gc = dd["nsdiff_ensemble"], dd["garch"]
                        blk[tag] = {
                            "pnl_net_nsdiff_bps": float(np.mean(ns["net"]) * BPS),
                            "pnl_net_garch_bps": float(np.mean(gc["net"]) * BPS),
                            "edge_net_bps": float(np.mean(ns["net"] - gc["net"]) * BPS),
                            "exposition_nsdiff": float(np.mean(np.abs(ns["positions"]))),
                            "exposition_garch": float(np.mean(np.abs(gc["positions"]))),
                        }
                    blk["delta_edge_bps"] = (blk["avec_roll"]["edge_net_bps"]
                                             - blk["sans_roll"]["edge_net_bps"])
                    blk["test_vs_garch_avec_roll"] = g2t.paired(
                        dec["avec_roll"]["nsdiff_ensemble"]["net"],
                        dec["avec_roll"]["garch"]["net"], "nsdiff", "garch")
                    blk["edge_survit"] = bool(blk["avec_roll"]["edge_net_bps"] > 0)
                    out["par_niveau"][level] = blk
                key = f"{inst}|{strategy}|{hu}"
                cells[key] = out
                fam[key] = out["par_niveau"][rf.DECISION_LEVEL]["test_vs_garch_avec_roll"]

    corrected = mt.correct_family(fam)
    summary = mt.family_summary(corrected)

    lvl = rf.DECISION_LEVEL
    n_pos = sum(1 for c in cells.values() if c["par_niveau"][lvl]["edge_survit"])
    payload = {
        "scope": "chantier H2 -- realisme d'execution des futures : cout de roulement",
        "declared": {
            "rolled_instruments": rf.ROLLED_INSTRUMENTS,
            "cost_model": "un roulement coute un aller-retour complet (hypothese conservatrice) ; "
                          "un sleeve tenu h semaines en traverse h/13 en esperance",
            "basis_not_modelled": "la base ES/SPY n'est pas modelisee : dans une comparaison "
                                  "FINANCEE (sans levier, |w| <= 1), le portage du future est "
                                  "compense au premier ordre par l'interet sur le capital non "
                                  "immobilise. Reserve residuelle : l'ecart a l'equilibre de la "
                                  "base a court terme n'est pas mesure.",
            "real_statements": "aucun bordereau de courtage reel n'est verse au depot ; la grille "
                               "reste declaree comme hypothese, et la sensibilite aux trois "
                               "niveaux est rapportee cellule par cellule",
            "regime": REGIME, "decision_level": lvl,
        },
        "grid": {"n_origins": int(arms["garch"]["cutoff_date"].nunique()),
                 "source": str(args.grid_dir)},
        "per_cell": cells,
        "holm_family_avec_roll": {"family": corrected["family"], "summary": summary},
        "verdict": {
            "n_cells": len(cells), "n_edge_positif_avec_roll": n_pos,
            "n_significatifs_holm": summary["n_significant_holm"],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"=== H2 : cout de roulement des futures, regime {REGIME}, niveau {lvl} ===")
    for key, c in sorted(cells.items()):
        b = c["par_niveau"][lvl]
        print(f"  {key:<34} outright {c['round_trip_bps']:.2f} + roll "
              f"{c['roll_cost_bps']:.2f} = {c['total_round_trip_bps']:.2f} bps  |  "
              f"edge {b['sans_roll']['edge_net_bps']:+7.2f} -> "
              f"{b['avec_roll']['edge_net_bps']:+7.2f} bps "
              f"({b['delta_edge_bps']:+.2f})  {'survit' if b['edge_survit'] else 'ne survit pas'}")
    print(f"\n  edge positif apres roulement : {n_pos}/{len(cells)} cellules")
    print(f"  famille Holm m={summary['m']} : {summary['n_significant_raw']} bruts -> "
          f"{summary['n_significant_holm']} Holm")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
