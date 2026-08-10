"""
nsdiff_vs_garch_w23.py -- tache 7 du BRIEF "NsDiff : consolider le verdict
daily vs weekly" : le match INTER-MODELES contre le modele volatilite
(ARIMA-GARCH) a W+2/W+3, l'ecart que NsDiff etait cense refermer. Tout le
reste du chantier etait intra-NsDiff.

Depend des taches 2 et 4 (resultats NsDiff multi-graines fiabilises a ces
horizons) : ce script lit l'artefact v2 (5 graines, n_samples=200) et les
lignes `source='oos'` d'ARIMA-GARCH dans `tracking.db` (lecture seule).
W+1 est calcule aussi, comme point de reference -- la question du brief porte
sur W+2/W+3.

ARIMA-GARCH N'EST PAS CONCERNE PAR LE BIAIS D'ECHANTILLONNAGE de la tache 4 :
ses bornes sont ANALYTIQUES (`last_price * exp(mu + sigma * dist.ppf(...))`,
cf. `models/arima_model.py`), il ne tire aucun echantillon. Comparer un NsDiff
a 200 tirages (dont le PI 95% couvre reellement ~94.1%) a un GARCH analytique
(94.95%) est donc le sens DEFAVORABLE a NsDiff -- avec l'ancien artefact a 50
tirages, NsDiff aurait paru encore plus faible. Le verdict est conservateur.

ASYMETRIE DE PROTOCOLE, a citer avec tout resultat de ce fichier (meme caveat
que `matrice_paired_tests.comparison_1_ranking`) : ARIMA-GARCH est refit a
CHAQUE origine sur fenetre glissante et n'a pas de graine (il est
deterministe) ; NsDiff est entraine UNE fois avant la premiere origine
(train-once-forward) et depend d'une graine. Ce n'est donc pas un match a
budget d'entrainement egal -- l'ecart mesure melange "modele" et "protocole
d'entrainement", exactement la reserve que porte deja tout classement
inter-modeles de ce repo (BRIEF_comparaison_rigoureuse.md §3).

Multi-graines (non negociable du brief) : chaque test est rejoue graine par
graine ET sur les 5 graines poolees. GARCH n'ayant pas de graine, il est
replique a l'identique sur chacune (`diffusion_headtohead.broadcast_seeds`) --
la question posee est "un run NsDiff a graine tiree au hasard bat-il GARCH ?".

Toute la machinerie de comparaison vit dans `diffusion_headtohead.py`, partage
avec `nsdiff_vs_tsdiff_v2.py` : les deux matchs sont litteralement le meme
test, pas deux implementations.

Sortie : experiments/nsdiff_vs_garch_w23.json
Usage   : python nsdiff_vs_garch_w23.py
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import diffusion_headtohead as h2h                                    # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                          # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_vs_garch_w23.json"
LABEL_A, LABEL_B = "NsDiff", "ARIMA-GARCH"


def load_challenger(assets: list, horizon_units: list) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, frequence, horizon_type, horizon_unit, cutoff_date,
                   target_date, last_close, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE source='oos' AND model=? AND horizon_type='weekly'
                  AND asset IN ({}) AND horizon_unit IN ({}) AND y_true IS NOT NULL
            """.format(",".join("?" * len(assets)), ",".join("?" * len(horizon_units))),
            con, params=[LABEL_B, *assets, *horizon_units])
    finally:
        con.close()
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    df["in_interval"] = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return h2h.with_winkler(df)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows = h2h.with_winkler(v2.load_rows())
    assets, seeds = v2.assets(rows), v2.seeds(rows)
    arm_a = pd.concat([rows, h2h.seed_pooled_rows(rows)], ignore_index=True)
    arm_b = h2h.broadcast_seeds(load_challenger(assets, args.horizons), seeds)
    print(f"{LABEL_B}: {len(arm_b) // (len(seeds) + 1)} lignes oos (repliquees sur {len(seeds)} graines) | "
          f"{LABEL_A} v2: {len(rows)} lignes, graines {seeds}")

    cache = v2.price_cache(rows)
    horizons = {}
    for hu in args.horizons:
        print(f"\n=== {hu} ===")
        res = h2h.run_match(arm_a, arm_b, cache, hu, seeds, assets, LABEL_A, LABEL_B)
        h2h.print_match(hu, res, seeds, LABEL_A, LABEL_B)
        horizons[hu] = res

    payload = {
        "config": {
            **v2.load_config(),
            "challenger": LABEL_B,
            "challenger_source": "tracking.db source='oos' (lecture seule)",
            "challenger_intervals": "ANALYTIQUES (arima_model.py : last_price*exp(mu+sigma*ppf)) -- "
                                    "aucun tirage, donc aucun biais de quantile empirique",
            "pairing": "par (actif, horizon, target_date), a l'interieur d'un meme regime",
            "protocol_asymmetry": (f"{LABEL_B} refit a chaque origine, deterministe (pas de graine) ; "
                                   f"{LABEL_A} train-once-forward, dependant d'une graine. L'ecart mesure "
                                   "melange modele et protocole d'entrainement -- meme reserve que tout "
                                   "classement inter-modeles de ce repo."),
        },
        "horizons": horizons,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
