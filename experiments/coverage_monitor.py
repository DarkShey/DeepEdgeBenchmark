"""
coverage_monitor.py -- chantier H3 du BRIEF « regeneration oos et famille 3 » :
le suivi de couverture EN LIGNE, la ou le benchmark s'arrete.

Tout le programme mesure une couverture RETROSPECTIVE sur grille figee : on
attend d'avoir 90 ou 340 origines resolues, puis on prononce un verdict sur le
bloc entier. Un usage reel n'a pas ce luxe -- il faut savoir, la semaine ou ca
arrive, qu'un modele a cesse de couvrir. C'est la version operationnelle de
Kupiec : meme quantite mesuree, mais lue en fenetre glissante et assortie d'une
bande d'alerte, au lieu d'un test unique sur tout l'echantillon.

CE QUI EST DECLARE, avant toute mesure :

  * FENETRE      26 origines (~ deux trimestres en hebdo). Compromis assume :
                 plus court, la couverture empirique est trop bruitee pour
                 declencher quoi que ce soit (a 13 origines, l'ecart-type d'une
                 couverture 95 % vaut ~6 points) ; plus long, on detecte la
                 derive un trimestre trop tard.
  * BANDE        [0,88 ; 0,99] sur la couverture 95 % glissante. Sortir de la
                 bande n'est PAS un verdict statistique : c'est un declencheur
                 d'investigation. La borne basse est a ~2,6 ecarts-types sous 95
                 pour n=26 ; la borne haute attrape la sur-couverture, qui est
                 un defaut symetrique (des intervalles inutilement larges ne
                 coutent rien en couverture et tout en Winkler).
  * ALERTE       une origine est en alerte des que sa fenetre sort de la bande.
                 Un episode est une suite d'alertes consecutives -- ce qui compte
                 operationnellement, c'est le nombre d'EPISODES et leur duree,
                 pas le nombre d'origines en alerte.

PAS DE FUITE, par construction : la fenetre a l'origine t ne contient que des
origines DEJA RESOLUES a t. Une origine dont la cible W+3 n'est pas encore
tombee n'entre dans aucune fenetre -- c'est le meme lag de resolution que la
calibration sigma du walk-forward hebdo.

Les fonctions de ce module ne font AUCUNE entree-sortie et sont testables a la
main (`test_coverage_monitor.py`) ; le runner en bas lit la piste `oos` de
`tracking.db` en LECTURE SEULE.

DEUXIEME CONSOMMATEUR : `permanent_defect_cells` (option 2 de
DECISION_derive_couverture_daily.md) distingue le DEFAUT PERMANENT -- une cellule
dont le plein echantillon est hors bande, elle n'a jamais couvert -- de la DERIVE
que ce module suit. Le dashboard D7/W1 s'en sert pour marquer ses cellules non
fiables ; la bande est partagee, pas recopiee (`test_permanent_defect_cells.py`).

Sortie : experiments/coverage_monitor.json
Usage   : python coverage_monitor.py [--window 26] [--models NsDiff ARIMA-GARCH]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# DB_PATH est resolu ICI plutot qu'importe de `backtest_rolling_tsdiffw` : cette
# brique tourne EN ROUTINE (job quotidien), et une brique de monitoring n'a aucune
# raison de tirer la pile de modeles derriere elle pour une constante de chemin.
# `test_db_path_reste_aligne_sur_la_convention_du_depot` interdit la derive.
DB_PATH = str(ROOT / "validation" / "tracking.db")

OUT_PATH = Path(__file__).resolve().parent / "coverage_monitor.json"
WINDOW = 26
BAND = (0.88, 0.99)
TARGET = 0.95
MODELS = ["NsDiff", "ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive"]


def rolling_coverage(in_interval, window: int = WINDOW) -> np.ndarray:
    """Couverture sur les `window` dernieres origines resolues, alignee a droite.
    Les `window - 1` premieres valent NaN : la fenetre n'est pas pleine, et une
    fenetre partielle declencherait des alertes de bruit pur au demarrage."""
    x = np.asarray(in_interval, dtype=float)
    if window <= 0:
        raise ValueError("window doit etre >= 1")
    out = np.full(x.size, np.nan)
    if x.size < window:
        return out
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def alerts(coverage, band: tuple = BAND) -> list:
    """Indices ou la couverture glissante sort de la bande, avec le cote. NaN =
    fenetre incomplete, jamais une alerte."""
    cov = np.asarray(coverage, dtype=float)
    lo, hi = band
    out = []
    for i, c in enumerate(cov):
        if np.isnan(c):
            continue
        if c < lo:
            out.append({"index": int(i), "coverage": float(c), "side": "sous_couverture"})
        elif c > hi:
            out.append({"index": int(i), "coverage": float(c), "side": "sur_couverture"})
    return out


def episodes(alert_list: list) -> list:
    """Suites d'alertes consecutives. C'est l'unite operationnelle : trois
    episodes d'une origine ne se lisent pas comme un episode de trois."""
    out = []
    for a in alert_list:
        if out and a["index"] == out[-1]["end"] + 1 and a["side"] == out[-1]["side"]:
            out[-1]["end"] = a["index"]
            out[-1]["length"] += 1
            out[-1]["worst_coverage"] = (min(out[-1]["worst_coverage"], a["coverage"])
                                         if a["side"] == "sous_couverture"
                                         else max(out[-1]["worst_coverage"], a["coverage"]))
        else:
            out.append({"start": a["index"], "end": a["index"], "length": 1,
                        "side": a["side"], "worst_coverage": a["coverage"]})
    return out


def monitor_series(in_interval, window: int = WINDOW, band: tuple = BAND) -> dict:
    """Le suivi complet d'une cellule : couverture glissante, alertes, episodes,
    et l'etat COURANT -- la seule chose qu'un operateur regarde vraiment."""
    cov = rolling_coverage(in_interval, window)
    al = alerts(cov, band)
    current = float(cov[-1]) if cov.size and not np.isnan(cov[-1]) else None
    return {
        "n": int(np.asarray(in_interval).size), "window": window, "band": list(band),
        "coverage_full_sample": float(np.mean(in_interval)) if np.size(in_interval) else None,
        "coverage_current_window": current,
        "status": ("fenetre_incomplete" if current is None
                   else "OK" if band[0] <= current <= band[1]
                   else "sous_couverture" if current < band[0] else "sur_couverture"),
        "n_alert_origins": len(al), "episodes": episodes(al),
        "rolling_coverage": [None if np.isnan(c) else round(float(c), 4) for c in cov],
    }


# ── defaut permanent vs derive (DECISION_derive_couverture_daily, option 2) ──

def permanent_defect(result: dict, band: tuple = BAND) -> dict | None:
    """Traduit un resultat de `monitor_series` en DEFAUT PERMANENT, ou None.

    Le critere est repris tel quel de la note de decision : la cellule est en
    alerte (fenetre glissante hors bande) ET son plein echantillon est LUI AUSSI
    hors bande -- autrement dit elle n'a jamais couvert, pas une seule fenetre.
    Une cellule dont seule la fenetre courante sort de la bande est une DERIVE :
    elle n'est PAS marquee ici, elle reste du ressort du resume de job quotidien.
    Confondre les deux ramenerait le comptage « 51 cellules en alerte » que la
    note a precisement separe en 35 defauts + 16 derives.

    Le `side` est lu sur le PLEIN echantillon (c'est ce que le marquage annonce),
    pas sur la fenetre courante : sous-couverture = intervalle trop etroit, le cas
    dangereux ; sur-couverture = intervalle inutilement large, couteux en Winkler
    mais sans risque de sous-estimation."""
    lo, hi = band
    full = result.get("coverage_full_sample")
    if full is None or result.get("status") in ("OK", "fenetre_incomplete"):
        return None
    if lo <= full <= hi:
        return None
    return {
        "coverage_full_sample": float(full),
        "coverage_current_window": result.get("coverage_current_window"),
        "n_origins": int(result["n"]),
        "side": "sous_couverture" if full < lo else "sur_couverture",
        "status_current_window": result["status"],
    }


def permanent_defect_cells(source: str = "oos", band: tuple = BAND,
                           db_path: str = DB_PATH, models: list = None,
                           window: int = WINDOW) -> list:
    """Les cellules a defaut permanent d'une piste, relues depuis tracking.db en
    LECTURE SEULE. Aucune liste codee en dur : le marquage est DERIVE de la piste
    a chaque appel, donc il suit la piste si elle evolue. `band` par defaut =
    `BAND`, la constante du suivi H3 -- il n'existe pas de seconde bande."""
    df = load_oos(db_path, list(models or MODELS), source)
    out = []
    for ident, r in monitor_cells(df, window, tuple(band)):
        defect = permanent_defect(r, tuple(band))
        if defect is not None:
            out.append({**ident, **defect, "last_cutoff": r["last_cutoff"]})
    return out


# ── runner : la piste `oos` du dashboard, en lecture seule ──────────────────

def load_oos(db_path: str, models: list, source: str = "oos") -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            "SELECT model, asset, frequence, horizon_unit, cutoff_date, y_true, y_lower, y_upper "
            "FROM predictions WHERE source=? AND horizon_type='weekly' AND y_true IS NOT NULL "
            "AND model IN ({}) ORDER BY cutoff_date".format(",".join("?" * len(models))),
            con, params=[source] + list(models))
    finally:
        con.close()


def monitor_cells(df: pd.DataFrame, window: int = WINDOW, band: tuple = BAND):
    """Le suivi de chaque cellule (modele, actif, regime, horizon) d'une piste
    deja chargee -- yield (identite, resultat de monitor_series). Extrait du
    runner pour que le marquage des defauts permanents lise EXACTEMENT la meme
    chose que H3, sans machinerie parallele qui pourrait en diverger."""
    for (model, asset, regime, hu), g in df.groupby(["model", "asset", "frequence",
                                                     "horizon_unit"], sort=True):
        g = g.sort_values("cutoff_date")
        inside = ((g["y_true"] >= g["y_lower"]) & (g["y_true"] <= g["y_upper"])).astype(float)
        r = monitor_series(inside.to_numpy(), window, tuple(band))
        r["last_cutoff"] = str(g["cutoff_date"].iloc[-1])
        yield {"key": f"{model}|{asset}|{regime}|{hu}", "model": model, "asset": asset,
               "regime": regime, "horizon_unit": hu}, r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--source", default="oos",
                   help="piste a suivre (defaut : oos ; 'oos2020' = grille regeneree)")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--band", type=float, nargs=2, default=list(BAND))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    df = load_oos(args.db_path, args.models, args.source)
    print(f"piste {args.source} : {len(df)} lignes resolues, {df['model'].nunique()} modeles "
          f"(LECTURE SEULE)")

    per_cell, flagged = {}, []
    for ident, r in monitor_cells(df, args.window, tuple(args.band)):
        per_cell[ident["key"]] = r
        if r["status"] not in ("OK", "fenetre_incomplete"):
            flagged.append((ident["key"], r))

    payload = {
        "scope": f"chantier H3 -- monitoring de couverture en ligne sur la piste {args.source}",
        "read_only": True, "source": args.source,
        "declared": {"window": args.window, "band": list(args.band), "target": TARGET,
                     "rule": "sortir de la bande declenche une INVESTIGATION, ce n'est pas un "
                             "verdict statistique ; le test formel reste Kupiec "
                             "(calibration_tests.kupiec_lr_uc) sur l'echantillon complet",
                     "no_leak": "la fenetre a l'origine t ne contient que des origines deja "
                                "resolues a t"},
        "n_cells": len(per_cell), "n_flagged": len(flagged),
        "flagged": [k for k, _ in flagged],
        "per_cell": per_cell,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"\n=== fenetre {args.window} origines, bande [{args.band[0]:.2f} ; {args.band[1]:.2f}] "
          f"=== {len(flagged)} cellules en alerte sur {len(per_cell)}")
    for key, r in sorted(flagged, key=lambda kv: kv[1]["coverage_current_window"]):
        print(f"  {key:<34} couverture courante {r['coverage_current_window']:.3f} "
              f"({r['status']}, plein echantillon {r['coverage_full_sample']:.3f}, "
              f"{len(r['episodes'])} episodes)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
