"""
prices_v4.py -- chantier 0-bis-A du BRIEF « porte 0-bis : le CTA corrige, juge
dans son habitat » : UN SOCLE DE PRIX UNIQUE pour le signal ET le PnL, sur
l'univers elargi.

LA LECON QU'ON APPLIQUE. A la porte 0, le signal ne pouvait pas etre produit sur
l'univers complet de DEITA parce que sa base de prix et `prices_v3` divergent de
2,6e-03 sur SPY -- un signal calcule sur une serie et trade sur une autre est
interdit, et cette seule raison a suffi a rabattre le programme sur 7 actifs.
`prices_v4` supprime l'obstacle a la racine : les 18 actifs sont telecharges par
le MEME pipeline que `prices_v3` (meme source, meme fenetre, calendriers de
cotation propres), geles et hashes ensemble.

L'UNIVERS, DECLARE AVANT TOUT CALCUL. Union de l'univers DEITA (16 actifs) et du
panel du benchmark (7), avec la regle de substitution que le brief exige --
« une seule convention par exposition, pas les deux » :

    REGLE : quand une exposition existe des deux cotes sous deux vehicules
    differents, c'est l'INSTRUMENT DU BENCHMARK qui est retenu.

    or       GC=F (DEITA, future)   -> GLD (benchmark, ETF)
    petrole  CL=F (DEITA, future)   -> USO (benchmark, ETF)

Motif, anterieur a tout resultat : `prices_v4` doit servir le signal ET le PnL, et
le chantier 1 conditionnel ne peut trader que les instruments du panel -- les
nuages NsDiff n'existent que la. Retenir GLD/USO fait du panel un SOUS-ENSEMBLE
STRICT de l'univers 0-bis, ce qui garantit qu'aucune exposition n'apparait deux
fois et que le chantier 1 se branche sans re-gel. Le prix a payer est declare :
USO est un ETF a roulement mensuel, moins pur que CL=F comme proxy du petrole.

    18 actifs = 16 (DEITA) - 2 (GC=F, CL=F substitues) + 4 (ZN=F, TLT, GLD, USO)

LES QUATRE CLASSES de la branche 2, figees ici : actions / taux / matieres
premieres-or / crypto. **VXX est declare hors classes** : c'est un ETN de
volatilite court terme a decroissance structurelle de roulement, il n'appartient
a aucune des quatre, et le forcer dans l'une contaminerait son verdict. Il reste
dans l'univers de conviction (moteur DEITA tel quel) et il est rapporte a part.
DEITA le range sous le secteur « Crypto », ce qui est une erreur d'etiquetage de
son cote -- hors perimetre du ticket, signalee, non corrigee ici.

VERIFICATION BLOQUANTE. Sur les 7 actifs partages avec `prices_v3`, le nouveau gel
doit REPRODUIRE l'ancien -- memes tolerances qu'au chantier R1 (log-rendements
1e-5, dispersion du ratio de prix 1e-4). Sans cela, la grille 0-bis ne se
comparerait plus a la grille `oos2020`, et le chantier 1 conditionnel trouverait
sous ses nuages NsDiff des prix qui ne sont plus ceux qui les ont produits.

Sortie : experiments/prices_v4/
    <ACTIF>.parquet   serie quotidienne gelee
    config.json       univers, substitutions, classes, comptages, hashes, verifications
Usage   : python prices_v4.py
Code de sortie : 0 si le gel est valide, 1 sinon.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prices_v3 import (COMMON_START, FETCH_END, LOGRET_TOL, ORIGIN_START,       # noqa: E402
                       OUT_DIR as PRICES_V3, PANEL as PANEL_V3,
                       RATIO_DISPERSION_TOL, counts, fetch_one, overlap_check, slug)

OUT_DIR = Path(__file__).resolve().parent / "prices_v4"

# Substitutions actees (cf. docstring) : une seule convention par exposition.
SUBSTITUTIONS = {
    "GC=F": {"remplace_par": "GLD", "exposition": "or",
             "motif": "le panel du benchmark trade l'ETF ; le chantier 1 conditionnel ne peut "
                      "trader que les instruments du panel (les nuages NsDiff n'existent que la)"},
    "CL=F": {"remplace_par": "USO", "exposition": "petrole",
             "motif": "idem or ; reserve declaree : USO porte un roulement mensuel, proxy moins "
                      "pur que CL=F"},
}

# Les 18 actifs, avec secteur/sous-secteur (pour la conviction) et classe (pour la
# branche 2). `classe = None` -> hors des quatre classes declarees.
PANEL = {
    # actions
    "SPY":      {"sector": "Equity", "subsector": "US Large-Cap", "classe": "actions",
                 "start": None, "origine": "DEITA + benchmark"},
    "QQQ":      {"sector": "Equity", "subsector": "US Tech Index", "classe": "actions",
                 "start": None, "origine": "DEITA"},
    "IWM":      {"sector": "Equity", "subsector": "US Small-Cap", "classe": "actions",
                 "start": None, "origine": "DEITA"},
    "EFA":      {"sector": "Equity", "subsector": "International Developed", "classe": "actions",
                 "start": None, "origine": "DEITA"},
    "EEM":      {"sector": "Equity", "subsector": "Emerging Markets", "classe": "actions",
                 "start": None, "origine": "DEITA"},
    # taux
    "ZN=F":     {"sector": "Bond", "subsector": "US Treasury Future", "classe": "taux",
                 "start": None, "origine": "benchmark"},
    "TLT":      {"sector": "Bond", "subsector": "US Treasury ETF", "classe": "taux",
                 "start": None, "origine": "benchmark"},
    # matieres premieres - or
    "GLD":      {"sector": "Commodity", "subsector": "Precious Metals", "classe": "matieres",
                 "start": None, "origine": "benchmark (substitue a GC=F)"},
    "SI=F":     {"sector": "Commodity", "subsector": "Precious Metals", "classe": "matieres",
                 "start": None, "origine": "DEITA"},
    "HG=F":     {"sector": "Commodity", "subsector": "Industrial Metals", "classe": "matieres",
                 "start": None, "origine": "DEITA"},
    "USO":      {"sector": "Commodity", "subsector": "Energy", "classe": "matieres",
                 "start": None, "origine": "benchmark (substitue a CL=F)"},
    "ZC=F":     {"sector": "Commodity", "subsector": "Agriculture", "classe": "matieres",
                 "start": None, "origine": "DEITA"},
    # crypto
    "BTC-USD":  {"sector": "Crypto", "subsector": "Large-Cap Crypto", "classe": "crypto",
                 "start": "2014-09-01", "origine": "DEITA + benchmark"},
    "ETH-USD":  {"sector": "Crypto", "subsector": "L1 Smart Contract", "classe": "crypto",
                 "start": "2017-11-01", "origine": "DEITA + benchmark"},
    "SOL-USD":  {"sector": "Crypto", "subsector": "L1 Smart Contract", "classe": "crypto",
                 "start": "2020-04-01", "origine": "DEITA"},
    "BNB-USD":  {"sector": "Crypto", "subsector": "Exchange Token", "classe": "crypto",
                 "start": "2017-11-01", "origine": "DEITA"},
    "LINK-USD": {"sector": "Crypto", "subsector": "DeFi Oracle", "classe": "crypto",
                 "start": "2017-11-01", "origine": "DEITA"},
    # hors classes, declare
    "VXX":      {"sector": "Crypto", "subsector": "Volatility", "classe": None,
                 "start": "2018-01-01", "origine": "DEITA",
                 "note": "ETN de volatilite court terme, decroissance structurelle de roulement : "
                         "n'appartient a aucune des quatre classes declarees. Reste dans l'univers "
                         "de conviction, rapporte a part, exclu du test de classes et du "
                         "portefeuille equipondere. L'etiquette secteur 'Crypto' vient de DEITA et "
                         "est une erreur de son cote -- signalee, non corrigee ici."},
}
CLASSES = ("actions", "taux", "matieres", "crypto")
ASSET_MAP = {a: {"sector": v["sector"], "subsector": v["subsector"]} for a, v in PANEL.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"univers 0-bis : {len(args.assets)} actifs "
          f"({sum(1 for a in args.assets if PANEL[a]['classe'])} dans les 4 classes, "
          f"{sum(1 for a in args.assets if not PANEL[a]['classe'])} hors classes)")
    print(f"substitutions : " + " | ".join(f"{k} -> {v['remplace_par']}"
                                           for k, v in SUBSTITUTIONS.items()) + "\n")

    per_asset, blocking = {}, []
    for asset in args.assets:
        spec = PANEL[asset]
        daily = fetch_one(asset, spec["start"] or COMMON_START).sort_index()
        path = out_dir / f"{slug(asset)}.parquet"
        daily.to_frame("close").to_parquet(path)
        ov = overlap_check(asset, daily, old_dir=PRICES_V3)
        per_asset[asset] = {**{k: v for k, v in spec.items() if k != "start"},
                            "requested_start": spec["start"] or COMMON_START,
                            "counts": counts(asset, daily),
                            "sha256": sha256(path),
                            "in_benchmark_panel": asset in PANEL_V3,
                            "overlap_vs_prices_v3": ov}
        if ov.get("status") == "checked" and not ov["passes"]:
            blocking.append(asset)

    payload = {
        "scope": "chantier 0-bis-A -- socle de prix unique pour le signal ET le PnL",
        "config": {
            "common_start": COMMON_START, "fetch_end": FETCH_END,
            "origin_start": ORIGIN_START,
            "source": "yfinance, source unique -- meme pipeline que prices_v3",
            "calendars": "calendriers de cotation propres a chaque actif ; aucun ffill "
                         "de prix (cf. bug 2 du ticket DEITA)",
            "substitutions": SUBSTITUTIONS,
            "classes_branche_2": list(CLASSES),
            "hors_classes": [a for a, v in PANEL.items() if not v["classe"]],
            "blocking_tolerances": {"logret": LOGRET_TOL,
                                    "price_ratio_dispersion": RATIO_DISPERSION_TOL},
            "panel_is_strict_subset": sorted(PANEL_V3) == sorted(
                a for a in PANEL if a in PANEL_V3),
        },
        "per_asset": per_asset,
        "blocking": blocking,
        "gate": "PASS" if not blocking else "FAIL",
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str,
                                                    ensure_ascii=False))

    print(f"{'actif':<10}{'classe':<10}{'origines':>9}{'obs':>7}  recouvrement vs prices_v3")
    for a, r in per_asset.items():
        ov, c = r["overlap_vs_prices_v3"], r["counts"]
        tail = (f"{ov['n_common']} dates, log-rdt max {ov['logret_max_abs_diff']:.1e} -> "
                f"{'OK' if ov['passes'] else 'HORS TOLERANCE'}"
                if ov.get("status") == "checked" else "nouvel actif")
        print(f"  {a:<10}{str(r['classe'] or '(hors)'):<10}"
              f"{c['n_test_origins_from_2020']:>7}{c['n_daily']:>8}  {tail}")

    print(f"\n  panel benchmark = sous-ensemble strict : "
          f"{payload['config']['panel_is_strict_subset']}")
    print(f"  porte 0-bis-A : {payload['gate']}"
          + (f" -- {', '.join(blocking)}" if blocking else " -- le signal peut etre gele"))
    print(f"-> {out_dir}")
    sys.exit(0 if not blocking else 1)


if __name__ == "__main__":
    main()
