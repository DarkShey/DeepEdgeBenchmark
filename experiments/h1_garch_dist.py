"""
h1_garch_dist.py -- chantier H1 du BRIEF « regeneration oos et famille 3 » :
LE CHAMPION DU BENCHMARK JOUE-T-IL AVEC LA BONNE LOI ?

LE CONSTAT, etabli par mesure au chantier precedent et pas par lecture de code :
les lignes `oos` du bras ARIMA-GARCH sont la variante GAUSSIENNE (elles se
reproduisent a 1,45e-06 avec `dist="normal"`, contre 2,3e-02 avec le skew-t),
alors que `models/arima_model.py` declare aujourd'hui `GARCH_DIST="skewt"`. Le
benchmark fait donc jouer son champion avec des queues fines sur des actifs a
queues epaisses -- et c'est le champion qui a tenu le mur contre NsDiff.

CE QUI EST COMPARE, ET CE QUI NE L'EST PAS. Les deux bras sont produits par le
MEME code, sur la MEME grille, avec les MEMES prix geles : `grid2020.run_garch`
prend la loi en argument depuis ce chantier, et rien d'autre ne change. La
comparaison mesure donc la loi d'innovation, et elle seule.

TROIS AXES, et ils ne disent pas la meme chose :
  * COUVERTURE -- la loi ne deplace que les BORNES, jamais le point : une
    difference de couverture est entierement imputable a la loi ;
  * WINKLER -- regle de score propre : c'est lui qui tranche, parce qu'une
    couverture plus haute obtenue en elargissant n'est pas une amelioration ;
  * RMSE -- doit etre IDENTIQUE aux deux lois (meme equation de moyenne ARIMA).
    C'est un controle de plomberie : si le RMSE bouge, les deux bras ne sont pas
    le meme modele et la comparaison ne porte pas sur ce qu'on croit.

FAMILLE DE HOLM, declaree avant lecture : le test Winkler apparie sur les
cellules (actif x horizon) du regime de la phase, et elles seules. La couverture
est rapportee avec son test de Kupiec par cellule, en descriptif -- deux familles
sur la meme question seraient une double lecture.

VERDICT : la loi CHAMPIONNE est celle qui gagne le Winkler poole ; a egalite
statistique, la gaussienne est conservee -- c'est la config qui a produit tous les
verdicts publies, et un changement de champion doit etre gagne, pas obtenu par
defaut.

Sortie : experiments/h1_garch_dist.json
Usage   : python h1_garch_dist.py [--regime weekly]
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
import multiple_testing as mt                                         # noqa: E402
from grid2020_tests import cell, garch_bands, paired                  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "h1_garch_dist.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
ARMS = {"normal": "ARIMA-GARCH", "skewt": "ARIMA-GARCH[skewt]"}
HORIZONS = ("W+1", "W+2", "W+3")
POOL_SEED = 42
RMSE_IDENTITY_TOL = 1e-9      # les deux lois partagent l'equation de moyenne


def load(grid_dir: Path, arm: str) -> pd.DataFrame:
    path = grid_dir / arm / "bands.parquet"
    if not path.exists():
        raise SystemExit(f"artefact manquant : {path}")
    return garch_bands(pd.read_parquet(path), 0.95)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--regime", default="weekly", choices=["weekly", "daily"])
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    arms = {dist: load(Path(args.grid_dir), arm) for dist, arm in ARMS.items()}
    assets = sorted(set.intersection(*(set(df["asset"]) for df in arms.values())))
    n_origins = arms["normal"]["cutoff_date"].nunique()
    print(f"grille : {n_origins} origines, {len(assets)} actifs, regime {args.regime}")

    cells, fam, rmse_drift = {}, {}, []
    for asset in assets:
        for hu in HORIZONS:
            d = {dist: cell(df, asset, args.regime, hu) for dist, df in arms.items()}
            if d["normal"]["y_true"].size == 0:
                continue
            m = {}
            for dist, x in d.items():
                inside = ((x["y_true"] >= x["y_lower"]) & (x["y_true"] <= x["y_upper"]))
                w = dash.winkler_score(x["y_true"], x["y_lower"], x["y_upper"])
                m[dist] = {
                    "cov95": float(inside.mean()),
                    "winkler": float(np.mean(w)),
                    "pi_width_pct": float(np.mean((x["y_upper"] - x["y_lower"])
                                                  / x["last_close"]) * 100),
                    "rmse": float(np.sqrt(np.mean((x["y_pred"] - x["y_true"]) ** 2))),
                    "kupiec": ct.kupiec_lr_uc(inside.astype(int), alpha_target=0.05),
                    "_winkler_series": w, "_inside": inside.astype(float),
                }
            drift = abs(m["normal"]["rmse"] - m["skewt"]["rmse"]) / max(m["normal"]["rmse"], 1e-12)
            rmse_drift.append(drift)
            key = f"{asset}|{hu}"
            test = paired(-m["skewt"]["_winkler_series"], -m["normal"]["_winkler_series"],
                          "skewt", "normal")
            cells[key] = {
                "asset": asset, "horizon_unit": hu, "n": int(d["normal"]["y_true"].size),
                **{f"{k}_{dist}": m[dist][k] for dist in m
                   for k in ("cov95", "winkler", "pi_width_pct", "rmse")},
                "kupiec_normal": m["normal"]["kupiec"], "kupiec_skewt": m["skewt"]["kupiec"],
                "rmse_rel_drift": drift,
                "winkler_test": test,
            }
            fam[key] = test

    corrected = mt.correct_family(fam)
    summary = mt.family_summary(corrected)
    wins_skewt = sum(1 for k in corrected["family"]
                     if corrected["family"][k]["holm_reject"]
                     and corrected["family"][k]["verdict"] == "skewt_significantly_better")
    wins_normal = sum(1 for k in corrected["family"]
                      if corrected["family"][k]["holm_reject"]
                      and corrected["family"][k]["verdict"] == "normal_significantly_better")

    all_w = {dist: np.concatenate([np.asarray(cells[k][f"winkler_{dist}"]).reshape(1)
                                   for k in cells]) for dist in ARMS}
    pooled_mean = {dist: float(np.mean(all_w[dist])) for dist in ARMS}
    champion = ("skewt" if wins_skewt > wins_normal else "normal")
    max_drift = max(rmse_drift) if rmse_drift else 0.0

    payload = {
        "scope": "chantier H1 -- loi d'innovation du bras GARCH : gaussienne vs skew-t",
        "grid": {"n_origins": n_origins, "assets": assets, "regime": args.regime,
                 "source": str(args.grid_dir)},
        "arms": ARMS,
        "identity_control": {
            "rule": "les deux lois partagent l'equation de moyenne ARIMA : le RMSE doit etre "
                    "identique. S'il bouge, les deux bras ne sont pas le meme modele.",
            "max_relative_rmse_drift": max_drift,
            "tolerance": RMSE_IDENTITY_TOL,
            "passes": bool(max_drift <= RMSE_IDENTITY_TOL),
        },
        "holm_family": {"scope": f"test Winkler apparie sur les cellules actif x horizon, "
                                 f"regime {args.regime}",
                        "family": corrected["family"], "summary": summary},
        "per_cell": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in cells.items()},
        "pooled_winkler_mean": pooled_mean,
        "verdict": {
            "cells_won_skewt": wins_skewt, "cells_won_normal": wins_normal,
            "champion": champion,
            "rule": "la loi championne est celle qui gagne le Winkler ; a egalite statistique, "
                    "la gaussienne est conservee -- c'est la config qui a produit les verdicts "
                    "publies, et un changement de champion doit etre gagne, pas obtenu par defaut",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"\n=== controle d'identite : derive RMSE max {max_drift:.2e} -> "
          f"{'OK (meme modele de moyenne)' if max_drift <= RMSE_IDENTITY_TOL else 'ANOMALIE'}")
    print(f"\n=== Cov95 / Winkler / largeur, par cellule (regime {args.regime}) ===")
    for k in sorted(cells):
        c = cells[k]
        print(f"  {k:<16} Cov95 {c['cov95_normal']:.3f} -> {c['cov95_skewt']:.3f} | "
              f"Winkler {c['winkler_normal']:.4g} -> {c['winkler_skewt']:.4g} | "
              f"largeur {c['pi_width_pct_normal']:.2f} % -> {c['pi_width_pct_skewt']:.2f} % | "
              f"{corrected['family'][k]['holm_verdict']}")
    print(f"\n  famille Holm m={summary['m']} | {summary['n_significant_raw']} bruts -> "
          f"{summary['n_significant_holm']} Holm  (skew-t {wins_skewt}, gaussien {wins_normal})")
    print(f"  >>> CONFIG CHAMPIONNE ACTEE : dist={champion!r}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
