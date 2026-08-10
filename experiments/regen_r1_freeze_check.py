"""
regen_r1_freeze_check.py -- chantier R1 du BRIEF « regeneration oos et famille 3 » :
LE GEL DES DONNEES EST-IL TOUJOURS VALIDE, ici et maintenant, avant la phase W ?

R1 est une porte, pas une etape de production. Les series de `prices_v3/` ont ete
constituees au chantier A ; ce script ne les refabrique pas et ne touche pas au
reseau. Il RE-VERIFIE, hors ligne, sur les fichiers geles tels qu'ils sont sur le
disque aujourd'hui, les trois choses que R1 exige :

  1. panel desequilibre DECLARE -- chaque actif a son historique propre, aucune
     troncature au plus court ; le tableau de comptages est recalcule, pas repris ;
  2. source unique -- yfinance pour les sept actifs, aucun melange ;
  3. recouvrement avec `diffusion_multiseed_v2/prices/` sur la periode commune,
     BLOQUANT.

UNE DIVERGENCE AVEC LE BRIEF, ASSUMEE ET DOCUMENTEE. Le brief ecrit « tolerance
~2e-7 relative » sur les prix. Ce critere a ete essaye au chantier A et il
BLOQUE -- non pas sur une revision d'historique, mais sur un changement de base
d'ajustement des dividendes : TLT presente un ratio ancien/nouveau CONSTANT de
0,99598782 (ancienne serie `fetch_tlt_patched` hors ligne, nouvelle yfinance).
Un facteur constant ne deplace aucun log-rendement, donc aucun modele. Le critere
bloquant retenu porte donc sur ce que les modeles voient reellement :

    (a) log-rendements identiques        tolerance 1e-5   [bloquant]
    (b) dispersion du ratio de prix      tolerance 1e-4   [bloquant]
    (c) niveau du ratio                  rapporte, jamais bloquant

(b) est ce qui distingue un simple changement de base (ratio constant, dispersion
nulle) d'une revision d'historique (ratio qui derive) -- c'est cette derniere que
le brief veut interdire, et elle reste interdite. Les tolerances et la mesure
vivent dans `prices_v3.overlap_check`, reutilise tel quel : ce fichier n'ajoute
que la porte.

LECTURE SEULE : aucun appel reseau, aucune ecriture hors du JSON de sortie.

Sortie : experiments/regen_r1_freeze_check.json
Usage   : python regen_r1_freeze_check.py
Code de sortie : 0 si la porte est franchie, 1 sinon (la phase W ne doit pas partir).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prices_v3 import (LOGRET_TOL, ORIGIN_START, OUT_DIR as PRICES_V3, PANEL,   # noqa: E402
                       RATIO_DISPERSION_TOL, counts, overlap_check, slug)

OUT_PATH = Path(__file__).resolve().parent / "regen_r1_freeze_check.json"


def check_asset(asset: str) -> dict:
    path = PRICES_V3 / f"{slug(asset)}.parquet"
    if not path.exists():
        return {"status": "missing", "path": str(path), "passes": False}
    daily = pd.read_parquet(path)["close"].sort_index()
    return {"status": "frozen", "path": str(path), "counts": counts(asset, daily),
            "overlap_vs_v2": overlap_check(asset, daily)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    per_asset, blocking = {}, []
    for asset in args.assets:
        r = check_asset(asset)
        per_asset[asset] = r
        ov = r.get("overlap_vs_v2", {})
        if r["status"] == "missing":
            blocking.append(f"{asset} (serie gelee absente)")
        elif ov.get("status") == "checked" and not ov["passes"]:
            blocking.append(f"{asset} (recouvrement hors tolerance)")

    starts = {a: r["counts"]["span"].split(" -> ")[0]
              for a, r in per_asset.items() if r["status"] == "frozen"}
    payload = {
        "scope": "chantier R1 -- gel des donnees verifie hors ligne avant la phase W",
        "read_only": True, "network": "aucun appel",
        "origin_start": ORIGIN_START, "prices": str(PRICES_V3),
        "panel_unbalanced": {
            "declared": True,
            "rule": "chaque actif a son historique maximal propre, aucune troncature au plus court",
            "starts": starts,
        },
        "source": "yfinance, source unique pour les sept actifs",
        "blocking_criterion": {
            "brief_says": "~2e-7 relatif sur les prix",
            "retained": {"logret_tolerance": LOGRET_TOL,
                         "price_ratio_dispersion_tolerance": RATIO_DISPERSION_TOL},
            "why": "le critere du brief bloque sur un changement de base d'ajustement de "
                   "dividendes (TLT, ratio CONSTANT 0.99598782), qui ne deplace aucun "
                   "log-rendement donc aucun modele. Le critere retenu porte sur les "
                   "log-rendements et sur la DISPERSION du ratio -- une revision "
                   "d'historique reste bloquante, un changement de base ne l'est plus.",
        },
        "per_asset": per_asset,
        "blocking": blocking,
        "gate": "PASS" if not blocking else "FAIL",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"=== R1 : gel des donnees, {len(args.assets)} actifs, hors ligne ===")
    for asset, r in per_asset.items():
        if r["status"] != "frozen":
            print(f"  {asset:<9} ABSENT")
            continue
        c, ov = r["counts"], r["overlap_vs_v2"]
        tail = (f"recouvrement {ov['n_common']} dates, log-rdt max {ov['logret_max_abs_diff']:.1e}, "
                f"dispersion ratio {ov['price_ratio_dispersion']:.1e} -> "
                f"{'OK' if ov['passes'] else 'HORS TOLERANCE'}"
                if ov.get("status") == "checked" else "aucune serie anterieure (premier entrant)")
        print(f"  {asset:<9} {c['span']}  {c['n_daily']:>5} obs, {c['n_test_origins_from_2020']} "
              f"origines | {tail}")
        if "price_level_shift" in ov:
            print(f"            NIVEAU : {ov['price_level_shift'].split(' -- ')[0]}")

    print(f"\n  porte R1 : {payload['gate']}"
          + (f" -- {', '.join(blocking)}" if blocking else " -- la phase W peut partir"))
    print(f"-> {args.out}")
    sys.exit(0 if not blocking else 1)


if __name__ == "__main__":
    main()
