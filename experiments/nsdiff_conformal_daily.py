"""
nsdiff_conformal_daily.py -- tache 5 du BRIEF "NsDiff : consolider le verdict
daily vs weekly" : la sous-couverture du regime daily est-elle CORRIGEABLE, et
si oui, le weekly conserve-t-il son avantage de calibration une fois la
correction appliquee ?

Methode : recalibration conformale par split, la brique DEJA presente dans le
repo (`tsdiff_recalibrate.py`), importee et appelee telle quelle --
`chronological_split`, `row_median_lo_hi`, `nonconformity_scores`,
`conformal_k`. Rien n'est reimplemente ; seul le perimetre change (les nuages
NsDiff v2 au lieu des nuages TSDiff persistes).

Decoupage : par (actif, graine, horizon, regime), les origines uniques sont
coupees en deux dans l'ordre chronologique -- premiere moitie = bloc de
calibration (d'ou sort k), seconde moitie = bloc de test (ou tout est mesure).
Aucune fuite : k ne voit jamais le bloc de test. Consequence a assumer et
declaree partout : le bloc de test ne fait que ~45 origines, donc
`effective_n` ~15 -- la moitie de la puissance du corps de l'etude.

Deux scenarios, tous deux rapportes :
  A. daily recalibre / weekly INTACT -- la question exacte du brief ("la
     sous-couverture du daily est corrigeable : ... puis re-tester si le
     weekly conserve un avantage").
  B. daily ET weekly recalibres -- controle d'equite. Sans lui, un scenario A
     favorable au daily ne prouverait rien : on aurait donne a un seul des
     deux regimes une couche de correction que l'autre n'a pas eue. C'est le
     scenario B qui dit si l'avantage du weekly est intrinseque ou seulement
     un defaut de calibration du daily.

Sortie : experiments/nsdiff_conformal_daily.json
Usage   : python nsdiff_conformal_daily.py
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
import nsdiff_v2_data as v2                                           # noqa: E402
import tsdiff_recalibrate as tsr                                      # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_conformal_daily.json"
LEVEL = 0.95              # le niveau des bandes stockees (q2.5 / q97.5)
TARGET_COVERAGE = LEVEL
POOL_SEED = 42


def recalibrate_cell(idx: pd.DataFrame, samples: np.ndarray) -> dict:
    """Split conformal sur UNE cellule (actif, graine, horizon, regime).
    `idx` doit etre trie chronologiquement et aligne sur `samples`."""
    calib_rows, test_rows, calib_range, test_range = tsr.chronological_split(idx.reset_index(drop=True))
    med, lo, hi = tsr.row_median_lo_hi(samples, LEVEL)
    y = idx["y_true"].to_numpy(dtype=float)

    scores_c = tsr.nonconformity_scores(y[calib_rows], med[calib_rows], lo[calib_rows], hi[calib_rows])
    k = tsr.conformal_k(scores_c, LEVEL)

    new_lo = med + k * (lo - med)
    new_hi = med + k * (hi - med)
    return {
        "k": float(k), "n_calib": int(len(calib_rows)), "n_test": int(len(test_rows)),
        "calib_size_flag": "ok" if len(calib_rows) >= tsr.MIN_CALIB_FLAG else
                           f"n_calib={len(calib_rows)} < {tsr.MIN_CALIB_FLAG} -- k potentiellement bruite",
        "calib_range": [str(calib_range[0]), str(calib_range[1])],
        "test_range": [str(test_range[0]), str(test_range[1])],
        "test_rows": test_rows, "new_lo": new_lo, "new_hi": new_hi,
    }


def build_recalibrated_rows(rows: pd.DataFrame, samples: np.ndarray, horizon_unit: str,
                            regimes: tuple) -> tuple:
    """Renvoie (lignes du BLOC DE TEST avec bandes recalibrees pour les regimes
    demandes, lignes du bloc de test inchangees, diagnostics par cellule)."""
    rows = rows.reset_index(drop=True)
    sel = rows.index[rows["horizon_unit"] == horizon_unit].to_numpy()
    sub, sub_samples = rows.loc[sel].reset_index(drop=True), samples[sel]

    keep_parts, diagnostics = [], {}
    for (asset, seed, regime), g in sub.groupby(["asset", "seed", "frequence"], sort=True):
        order = g.sort_values("cutoff_date")
        pos = order.index.to_numpy()
        cell = recalibrate_cell(order, sub_samples[pos])

        block = order.iloc[cell["test_rows"]].copy()
        block["k_conformal"] = cell["k"]
        block["y_lower_orig"], block["y_upper_orig"] = block["y_lower"], block["y_upper"]
        if regime in regimes:
            block["y_lower"] = cell["new_lo"][cell["test_rows"]]
            block["y_upper"] = cell["new_hi"][cell["test_rows"]]
        keep_parts.append(block)
        diagnostics[f"{asset}|seed{seed}|{regime}"] = {
            key: cell[key] for key in ("k", "n_calib", "n_test", "calib_size_flag",
                                       "calib_range", "test_range")
        }

    out = pd.concat(keep_parts, ignore_index=True)
    out["in_interval"] = ((out["y_true"] >= out["y_lower"]) & (out["y_true"] <= out["y_upper"])).astype(float)
    return out, diagnostics


def evaluate(rows_test: pd.DataFrame, cache: dict, horizon_unit: str) -> dict:
    """Meme batterie que la tache 1, restreinte au bloc de test : skill Winkler
    poole (graines moyennees) + couverture par (actif, regime)."""
    seeds = v2.seeds(rows_test)
    pairs_by_seed = {s: v2.enriched_pairs(rows_test, cache, horizon_unit, seed=s) for s in seeds}
    pooled_pairs = v2.seed_average(pairs_by_seed)

    coverage = {}
    for asset, g in pooled_pairs.groupby("asset"):
        g = g.sort_values("cutoff_date")
        coverage[asset] = {
            regime: ct.coverage_gap_block_test(g[f"in_interval_{regime}"].values,
                                               target=TARGET_COVERAGE, seed=POOL_SEED)
            for regime in ("daily", "weekly")
        }
        coverage[asset]["winkler_daily"] = float(g["winkler_daily"].mean())
        coverage[asset]["winkler_weekly"] = float(g["winkler_weekly"].mean())
        coverage[asset]["pi_width_daily"] = float(g["pi_width_daily"].mean())
        coverage[asset]["pi_width_weekly"] = float(g["pi_width_weekly"].mean())
    return {"pooled_skill": v2.pooled_skill_test(pooled_pairs, seed=POOL_SEED), "coverage": coverage}


def run_horizon(rows: pd.DataFrame, samples: np.ndarray, cache: dict, horizon_unit: str) -> dict:
    print(f"\n=== {horizon_unit} ===")
    baseline_rows, diagnostics = build_recalibrated_rows(rows, samples, horizon_unit, regimes=())
    daily_rows, _ = build_recalibrated_rows(rows, samples, horizon_unit, regimes=("daily",))
    both_rows, _ = build_recalibrated_rows(rows, samples, horizon_unit, regimes=("daily", "weekly"))

    res = {
        "conformal_diagnostics": diagnostics,
        "before": evaluate(baseline_rows, cache, horizon_unit),
        "scenario_A_daily_recalibrated": evaluate(daily_rows, cache, horizon_unit),
        "scenario_B_both_recalibrated": evaluate(both_rows, cache, horizon_unit),
    }
    _print_horizon(horizon_unit, res)
    return res


def _print_horizon(horizon_unit: str, res: dict) -> None:
    print(f"\n[{horizon_unit}] bloc de TEST uniquement -- couverture (cible 95%) et largeur PI")
    print(f"{'Actif':<10}{'cov daily av->ap':>18}{'largeur daily av->ap':>26}"
          f"{'cov weekly av->ap':>19}{'k daily (median graines)':>26}")
    for asset in res["before"]["coverage"]:
        b = res["before"]["coverage"][asset]
        a = res["scenario_A_daily_recalibrated"]["coverage"][asset]
        bb = res["scenario_B_both_recalibrated"]["coverage"][asset]
        ks = [d["k"] for key, d in res["conformal_diagnostics"].items()
              if key.startswith(f"{asset}|") and key.endswith("|daily")]
        cov_d = f"{b['daily']['coverage']:.3f} -> {a['daily']['coverage']:.3f}"
        wid_d = f"{b['pi_width_daily']:.4g} -> {a['pi_width_daily']:.4g}"
        cov_w = f"{b['weekly']['coverage']:.3f} -> {bb['weekly']['coverage']:.3f}"
        print(f"{asset:<10}{cov_d:>18}{wid_d:>26}{cov_w:>19}{np.median(ks):>26.3f}")

    print(f"\n[{horizon_unit}] skill Winkler poole (weekly vs daily), bloc de test")
    print(f"{'Scenario':<32}{'global verdict':>36}{'p':>9}")
    for label, key in (("avant recalibration", "before"),
                       ("A: daily recalibre seul", "scenario_A_daily_recalibrated"),
                       ("B: daily ET weekly recalibres", "scenario_B_both_recalibrated")):
        g = res[key]["pooled_skill"]["global"]
        if g.get("status") != "tested":
            print(f"{label:<32}{g.get('status'):>36}")
            continue
        w = g["skill_winkler"]
        print(f"{label:<32}{w['verdict']:>36}{w['p_value']:>9.4f}")


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
            "method": "split conformal, brique tsdiff_recalibrate.py importee telle quelle "
                      "(chronological_split / row_median_lo_hi / nonconformity_scores / conformal_k)",
            "level": LEVEL,
            "k_grouping": "un k par (actif, graine, horizon, regime) -- jamais un facteur global",
            "split": "50/50 par cutoff_date unique, calibration STRICTEMENT avant le test",
            "power_caveat": "toutes les mesures de ce fichier portent sur le seul bloc de TEST "
                            "(~45 origines, effective_n ~15) : deux fois moins puissant que le "
                            "corps de l'etude (90 origines). Un 'indistinguable' y est encore "
                            "moins informatif qu'ailleurs.",
        },
        "horizons": {h: run_horizon(rows, samples, cache, h) for h in args.horizons},
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
