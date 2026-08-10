"""
famille3_v2.py -- chantier F3 du BRIEF « regeneration oos et famille 3 » : la
famille 3 REPOSEE, pas rejouee.

CE QUI EST CLOS, ET POURQUOI CE N'EST PAS UN ECHEC DE PUISSANCE. La famille 3
originelle prend position « si le PI exclut le prix courant ». Elle n'emet JAMAIS :
0 position sur 3 240 origines-instruments, a 95 % comme a 80 %. La cause est
mecanique et se lit sans test -- la largeur mediane du PI (8 % du prix a 80 %)
depasse toujours le drift median a 1-3 semaines. Descendre le niveau a 70 %, 60 %,
apres avoir vu ce resultat, serait du p-hacking pur, et ne changerait rien au
mecanisme : la comparaison reste binaire « tout l'intervalle d'un cote ou rien ».
La famille est close DANS SA FORMULATION, et remplacee ici.

LA NOUVELLE FAMILLE, DECLAREE A PRIORI (tout ce paragraphe est anterieur au
premier calcul de ce fichier ; il reprend le brief mot pour mot la ou le brief
tranche, et declare explicitement ce qu'il laissait ouvert) :

  * REGLE     position si |mediane predictive - prix courant| > k x largeur du
              nuage, signe donne par la mediane. C'est le ratio drift/incertitude
              que la famille 3 mesurait maladroitement par exclusion binaire.
  * k         0,25 et 0,5. Deux valeurs, declarees par le brief. Aucun balayage.
  * LARGEUR   l'echelle interquantile a 80 %, c'est-a-dire l'ecart entre les
              quantiles 10 % et 90 %. DECLARE ICI, parce que le brief fixe k mais
              ecrit seulement « largeur du nuage (echelle interquantile) ». Le
              niveau retenu est celui de la famille QU'ON REMPLACE
              (`filtered_direction` tournait a 80 %, declare a priori au chantier
              precedent) : garder la meme echelle est la seule facon de dire que
              c'est la REGLE qui change, et pas l'echelle sous elle.
              Ce n'est pas un choix sans consequence, et l'etage 1 le montre : la
              lecture concurrente -- « largeur » = largeur du PI a 95 % --
              reproduit trait pour trait la pathologie de la famille close (1
              position emise sur 2 700 origines-instruments). Les deux lectures
              sont donc rapportees a l'etage 1, DESCRIPTIF ; une seule, declaree
              ci-dessus, alimente l'etage confirmatoire.
  * POINT     la MEDIANE, des deux cotes. Cote NsDiff, la mediane du nuage
              agrege (pas sa moyenne) ; cote GARCH, sa prevision centrale, qui
              EST la mediane de sa loi predictive. Symetrie complete, exigee par
              le brief.
  * SYMETRIE  le bras GARCH est evalue avec le MEME signal sur SES propres
              quantiles. Aucune regle de ce fichier ne connait le modele qui
              l'appelle.

DEUX ETAGES, ET UN SEUL EST CONFIRMATOIRE.

  ETAGE 1 -- EXPLORATOIRE, grille actuelle (90 origines). Une seule question,
  descriptive : la famille EMET-ELLE des signaux ? C'est le controle que la
  famille 3 originelle n'avait pas passe. Aucun test, aucune p-value, aucune
  conclusion -- un comptage.

  ETAGE 2 -- CONFIRMATOIRE, nouvelle grille (340 origines). Famille de Holm
  DEDIEE, declaree ici avant tout calcul :

      SPY-ETF x {W+1, W+2, W+3} x {k=0,25 ; k=0,5} x {vs GARCH, vs B&H}
      regime weekly  ->  m = 12 tests, et eux seuls.

  Perimetre choisi AVANT lecture, et pour la meme raison que l'hypothese primaire
  du programme : SPY weekly est la seule zone ou l'analyse de puissance donnait un
  n requis atteignable. SPY-ES partage les memes previsions que SPY-ETF et n'en
  differe que par les frais -- l'inclure doublerait la famille sans ajouter
  d'information, il est rapporte en exploratoire. Les autres actifs, le regime
  daily et les autres vehicules sont exploratoires et etiquetes tels quels.

CRITERE DE CLOTURE, declare par le brief et applique ici sans reinterpretation :
si aux DEUX valeurs de k la famille n'emet pas de signaux exploitables nets de
frais sur la nouvelle grille, la question « les intervalles de NsDiff portent-ils
un signal directionnel ? » est close DEFINITIVEMENT, sans autre reformulation.
« Exploitable » est operationnalise ici, avant lecture : au moins un test de la
famille rejette sous Holm en faveur de NsDiff, ET la cellule correspondante a un
PnL net positif.

Briques reutilisees telles quelles : `econ_backtest` (la nouvelle regle y vit,
comme les trois autres familles), `grid2020_tests` (construction des bras,
alignement, decomposition, test apparie par blocs), `real_fees`,
`multiple_testing`, `nsdiff_v2_data`.

Sortie : experiments/famille3_v2.json
Usage   : python famille3_v2.py
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

import econ_backtest as eb                                            # noqa: E402
import grid2020_tests as g2t                                          # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
import real_fees as rf                                                # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "famille3_v2.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
GARCH_PI80 = Path(__file__).resolve().parent / "garch_pi80" / "bands.parquet"

LEVEL = 0.80                       # echelle de largeur, declaree (cf. docstring)
LEVEL_ALT = 0.95                   # lecture concurrente, rapportee en descriptif seulement
BPS = 1e4
HORIZONS = ("W+1", "W+2", "W+3")
STRATEGIES = tuple(f"normalised_direction_k{str(k).replace('.', '')}"
                   for k in eb.NORMALISED_DIRECTION_K)

# ── LA FAMILLE CONFIRMATOIRE, EN DUR, AVANT TOUT CALCUL ─────────────────────
FAMILY = {
    "instrument": "SPY-ETF",
    "regime": "weekly",
    "horizons": list(HORIZONS),
    "strategies": list(STRATEGIES),
    "comparisons": ["test_vs_garch", "test_vs_bh"],
    "m": len(HORIZONS) * len(STRATEGIES) * 2,
    "declared_in": "BRIEF_nsdiff_regeneration_oos_et_famille3.md, chantier F3",
    "rationale": "meme perimetre que l'hypothese primaire du programme (SPY weekly, seule zone "
                 "a n requis atteignable). SPY-ES partage les memes previsions et n'en differe "
                 "que par les frais : exploratoire, pour ne pas doubler la famille a information "
                 "constante.",
}


def arms_from_grid(rows: pd.DataFrame, samples: np.ndarray, garch: pd.DataFrame,
                   level: float = LEVEL) -> dict:
    """Les deux bras au niveau demande, alignes origine par origine. Le point est
    la MEDIANE des deux cotes (cf. docstring) : cote GARCH la prevision centrale
    l'est deja, cote NsDiff il faut le demander explicitement."""
    return g2t.align({"nsdiff_ensemble": g2t.nsdiff_bands(rows, samples, level, point="median"),
                      "garch": g2t.garch_bands(garch, level)})


def emission_counts(arms: dict, assets, regimes) -> dict:
    """ETAGE 1 : la famille emet-elle ? Comptage descriptif, aucun test."""
    out = {}
    for strategy in STRATEGIES:
        for arm_name, df in arms.items():
            n_active = n_total = 0
            per_cell = {}
            for asset in assets:
                for regime in regimes:
                    for hu in HORIZONS:
                        d = g2t.cell(df, asset, regime, hu)
                        if d["y_true"].size == 0:
                            continue
                        w = eb.STRATEGIES[strategy]["fn"](d)
                        a, t = int((np.abs(w) > 0).sum()), int(w.size)
                        per_cell[f"{asset}|{regime}|{hu}"] = {"n_active": a, "n": t}
                        n_active += a
                        n_total += t
            out[f"{strategy}|{arm_name}"] = {
                "n_active": n_active, "n_total": n_total,
                "activation_rate": (n_active / n_total) if n_total else float("nan"),
                "per_cell": per_cell,
            }
    return out


def economic_cells(arms: dict, assets, regimes) -> dict:
    """PnL net de frais par cellule, pour les deux bras et pour acheter-et-garder,
    avec les tests apparies par blocs. Moteur inchange (`grid2020_tests`)."""
    cells = {}
    for strategy in STRATEGIES:
        for inst, sp in rf.INSTRUMENTS.items():
            if sp["asset"] not in assets:
                continue
            cost = rf.one_way_bps(inst, rf.DECISION_LEVEL)
            for regime in regimes:
                for hu in HORIZONS:
                    data = {n: g2t.cell(df, sp["asset"], regime, hu) for n, df in arms.items()}
                    if data["garch"]["y_true"].size == 0:
                        continue
                    ns = g2t.decompose(strategy, data["nsdiff_ensemble"], cost)
                    gc = g2t.decompose(strategy, data["garch"], cost)
                    bh = g2t.decompose("buy_and_hold", data["garch"], cost)
                    cells[f"{inst}|{strategy}|{hu}|{regime}"] = {
                        "instrument": inst, "asset": sp["asset"], "strategy": strategy,
                        "horizon_unit": hu, "regime": regime, "n": int(ns["net"].size),
                        "n_origines_actives_nsdiff": int((np.abs(ns["positions"]) > 0).sum()),
                        "n_origines_actives_garch": int((np.abs(gc["positions"]) > 0).sum()),
                        "pnl_net_nsdiff_bps": float(np.mean(ns["net"]) * BPS),
                        "pnl_net_garch_bps": float(np.mean(gc["net"]) * BPS),
                        "pnl_net_bh_bps": float(np.mean(bh["net"]) * BPS),
                        "edge_net_bps": float(np.mean(ns["net"] - gc["net"]) * BPS),
                        "test_vs_garch": g2t.paired(ns["net"], gc["net"], "nsdiff", "garch"),
                        "test_vs_bh": g2t.paired(ns["net"], bh["net"], "nsdiff", "buy_and_hold"),
                    }
    return cells


def confirmatory(cells: dict) -> dict:
    """ETAGE 2 : la famille de Holm declaree, et le critere de cloture."""
    fam = {}
    for strategy in FAMILY["strategies"]:
        for hu in FAMILY["horizons"]:
            key = f"{FAMILY['instrument']}|{strategy}|{hu}|{FAMILY['regime']}"
            if key not in cells:
                continue
            for comp in FAMILY["comparisons"]:
                fam[f"{strategy}|{hu}|{comp}"] = cells[key][comp]
    corrected = mt.correct_family(fam)
    summary = mt.family_summary(corrected)

    survivors = []
    for name in summary["survivors"]:
        strategy, hu, comp = name.split("|")
        c = cells[f"{FAMILY['instrument']}|{strategy}|{hu}|{FAMILY['regime']}"]
        if fam[name]["verdict"].startswith("nsdiff") and c["pnl_net_nsdiff_bps"] > 0:
            survivors.append(name)

    per_k = {}
    for strategy in FAMILY["strategies"]:
        emitted = any(cells[f"{FAMILY['instrument']}|{strategy}|{hu}|{FAMILY['regime']}"]
                      ["n_origines_actives_nsdiff"] > 0
                      for hu in FAMILY["horizons"]
                      if f"{FAMILY['instrument']}|{strategy}|{hu}|{FAMILY['regime']}" in cells)
        per_k[strategy] = {"emits_signals": bool(emitted),
                           "exploitable_survivors": [s for s in survivors if s.startswith(strategy)]}

    return {
        **FAMILY, "family": corrected["family"], "summary": summary,
        "exploitable_survivors": survivors,
        "per_k": per_k,
        "closure_criterion": "si aux DEUX valeurs de k la famille n'emet pas de signaux "
                             "exploitables nets de frais (au moins un rejet Holm en faveur de "
                             "NsDiff sur une cellule a PnL net positif), la question du signal "
                             "directionnel est close definitivement, sans reformulation",
        "family_positive": bool(survivors),
        "question_closed": not bool(survivors),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--garch-pi80", default=str(GARCH_PI80))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()

    # ── etage 1 : exploratoire, grille actuelle ─────────────────────────────
    rows_old, samples_old = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir),
                                         model="NsDiff")
    rows_old = rows_old.reset_index(drop=True)
    garch_old = pd.read_parquet(args.garch_pi80)
    emission_old = {}
    for level in (LEVEL, LEVEL_ALT):
        a = arms_from_grid(rows_old, samples_old, garch_old, level)
        assets_old = sorted(a["garch"]["asset"].unique())
        n_old = a["garch"]["cutoff_date"].nunique()
        emission_old[f"largeur@{level:.0%}"] = emission_counts(a, assets_old, ("weekly", "daily"))
    print(f"=== ETAGE 1 (EXPLORATOIRE) -- grille actuelle : {n_old} origines, "
          f"{len(assets_old)} actifs ===")
    for level_name, block in emission_old.items():
        tag = " [DECLARE]" if level_name.endswith(f"{LEVEL:.0%}") else " [lecture concurrente]"
        print(f"  -- echelle {level_name}{tag}")
        for name, e in block.items():
            print(f"     {name:<40}{e['n_active']:>6} / {e['n_total']:<6} positions "
                  f"({e['activation_rate']:.1%})")

    # ── etage 2 : confirmatoire, nouvelle grille ────────────────────────────
    rows, samples, garch = g2t.load_arms(Path(args.grid_dir))
    arms = arms_from_grid(rows, samples, garch)
    assets = sorted(arms["garch"]["asset"].unique())
    n_new = arms["garch"]["cutoff_date"].nunique()
    emission_new = emission_counts(arms, assets, ("weekly", "daily"))
    cells = economic_cells(arms, assets, ("weekly", "daily"))
    conf = confirmatory(cells)

    print(f"\n=== emission sur la nouvelle grille : {n_new} origines, {len(assets)} actifs ===")
    for name, e in emission_new.items():
        print(f"  {name:<40}{e['n_active']:>6} / {e['n_total']:<6} positions "
              f"({e['activation_rate']:.1%})")

    print(f"\n=== ETAGE 2 (CONFIRMATOIRE) -- famille Holm dediee, m={conf['summary']['m']} : "
          f"{FAMILY['instrument']}, {FAMILY['regime']} ===")
    for name in sorted(conf["family"]):
        t = conf["family"][name]
        strategy, hu, comp = name.split("|")
        c = cells[f"{FAMILY['instrument']}|{strategy}|{hu}|{FAMILY['regime']}"]
        print(f"  {name:<46} PnL net {c['pnl_net_nsdiff_bps']:+8.2f} bps  "
              f"actives {c['n_origines_actives_nsdiff']:>3}/{c['n']:<3}  "
              f"p={t['p_value']:.4f}  p_aj={t['holm_p_adjusted']:.4f}  -> {t['holm_verdict']}")
    s = conf["summary"]
    print(f"  famille m={s['m']}, seuil le plus strict {s['smallest_threshold']:.4f} | "
          f"{s['n_significant_raw']} bruts -> {s['n_significant_holm']} Holm")
    for strategy, r in conf["per_k"].items():
        print(f"  {strategy:<32} emet : {r['emits_signals']} | survivants exploitables : "
              f"{r['exploitable_survivors'] or 'aucun'}")
    print(f"  >>> QUESTION DU SIGNAL DIRECTIONNEL : "
          f"{'CLOSE DEFINITIVEMENT' if conf['question_closed'] else 'OUVERTE (famille positive)'}")

    payload = {
        "scope": "chantier F3 -- famille 3 reposee : signal directionnel normalise",
        "declared_rule": {
            "position": "si |mediane predictive - prix courant| > k x largeur du PI",
            "k_values": list(eb.NORMALISED_DIRECTION_K),
            "width_level": LEVEL,
            "width_level_declared_here": "le brief fixe k mais ecrit seulement « echelle "
                                         "interquantile » : le niveau retenu est celui de la "
                                         "famille remplacee (filtered_direction, 80 %), pour que "
                                         "ce soit la REGLE qui change et pas l'echelle sous elle. "
                                         "La lecture concurrente (largeur du PI a 95 %) est "
                                         "rapportee a l'etage 1, en descriptif seulement -- elle "
                                         "reproduit la non-emission de la famille close.",
            "point": "mediane des deux cotes -- mediane du nuage agrege cote NsDiff, prevision "
                     "centrale (= mediane de la loi predictive) cote GARCH",
            "symmetry": "GARCH evalue avec le meme signal sur ses propres quantiles",
        },
        "closed_formulation": {
            "family": "filtered_direction -- position si le PI exclut le prix courant",
            "status": "close dans sa formulation, remplacee ; aucun balayage de niveau",
        },
        "stage1_exploratory": {"grid": {"n_origins": n_old, "assets": assets_old,
                                        "source": str(args.v2_dir)},
                               "emission": emission_old,
                               "status": "descriptif -- aucun test, aucune conclusion"},
        "stage2_confirmatory": {"grid": {"n_origins": n_new, "assets": assets,
                                         "effective_n": n_new // g2t.BLOCK_LENGTH,
                                         "source": str(args.grid_dir)},
                                "emission": emission_new,
                                "holm_family": conf},
        "exploratory_cells": cells,
        "note": "Seule la famille de Holm declaree est confirmatoire. Tout le reste -- autres "
                "actifs, regime daily, vehicule SPY-ES -- est EXPLORATOIRE et ne fonde aucune "
                "conclusion.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
