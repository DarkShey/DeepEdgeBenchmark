"""
grid_old_vs_new.py -- chantier R3 du BRIEF « regeneration oos et famille 3 » :
LE TABLEAU ANCIENNE GRILLE / NOUVELLE GRILLE, et l'explication des renversements.

R3 pose une regle simple et couteuse a tenir : « aucun melange ancien/nouveau ».
Les verdicts issus de la grille 90 origines restent cites comme tels, et la note
doit porter un tableau qui met les deux grilles cote a cote sur les conclusions
cles -- couverture, Winkler, TOST, PnL -- en expliquant tout renversement.

Ce script produit ce tableau. Il ne juge pas : il aligne, il compare, et il
ETIQUETTE les renversements pour que la note ait a les expliquer un par un.

LES DEUX GRILLES, et pourquoi elles ne sont pas comparables ligne a ligne :

                       ancienne                    nouvelle
    origines           90  (effective_n 30)        340 (effective_n 113)
    depart             fenetre recente             2020-01
    actifs             5                           7  (+ GLD, USO)
    prix               diffusion_multiseed_v2      prices_v3 (re-geles)
    NsDiff             ensemble 5 x 200            ensemble 5 x 200
    GARCH              refit/origine, gaussien     refit/origine, config championne H1

Elles ne partagent NI les memes origines, NI les memes prix (TLT porte un ratio de
niveau constant de 0,99598782 entre les deux jeux, cf. R1). Comparer origine par
origine n'aurait donc pas de sens ; ce qui se compare, ce sont les VERDICTS et les
grandeurs agregees par cellule. Les deux entrants GLD et USO n'existent que d'un
cote : ils sont rapportes a part, jamais melanges au tableau de comparaison.

CE QU'EST UN RENVERSEMENT, declare avant lecture :
  * VERDICT     le test apparie change de camp, ou passe de « significatif » a
                « indistinguable » (ou l'inverse) ;
  * SIGNE       l'edge net change de signe ;
  * EQUIVALENCE le TOST conclut a l'equivalence d'un cote et pas de l'autre.
Un simple deplacement de valeur, sans changement de verdict, n'est pas un
renversement -- c'est ce qu'on attend d'un echantillon quatre fois plus long.

Sortie : experiments/grid_old_vs_new.json
Usage   : python grid_old_vs_new.py
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
import grid2020_tests as g2t                                          # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
import real_fees as rf                                                # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "grid_old_vs_new.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
GARCH_PI80 = Path(__file__).resolve().parent / "garch_pi80" / "bands.parquet"
HORIZONS = ("W+1", "W+2", "W+3")
REGIMES = ("weekly", "daily")
STRATEGY = "var_limit"          # la famille qui porte l'hypothese primaire du programme
BPS = 1e4
TOST_MARGIN = 0.05


def metrics(arms: dict, asset: str, regime: str, hu: str, cost_one_way: float) -> dict:
    a = g2t.cell(arms["nsdiff_ensemble"], asset, regime, hu)
    b = g2t.cell(arms["garch"], asset, regime, hu)
    if a["y_true"].size == 0:
        return None
    wa = dash.winkler_score(a["y_true"], a["y_lower"], a["y_upper"])
    wb = dash.winkler_score(b["y_true"], b["y_lower"], b["y_upper"])
    ia = ((a["y_true"] >= a["y_lower"]) & (a["y_true"] <= a["y_upper"])).astype(float)
    ib = ((b["y_true"] >= b["y_lower"]) & (b["y_true"] <= b["y_upper"])).astype(float)
    ns = g2t.decompose(STRATEGY, a, cost_one_way)
    gc = g2t.decompose(STRATEGY, b, cost_one_way)
    tost = ct.tost_relative_rmse((a["y_pred"] - a["y_true"]) ** 2,
                                 (b["y_pred"] - b["y_true"]) ** 2,
                                 margin_rel=TOST_MARGIN, seed=g2t.POOL_SEED)
    return {
        "n": int(a["y_true"].size),
        "cov95_nsdiff": float(ia.mean()), "cov95_garch": float(ib.mean()),
        "winkler_nsdiff": float(wa.mean()), "winkler_garch": float(wb.mean()),
        "winkler_verdict": g2t.paired(-wa, -wb, "nsdiff", "garch")["verdict"],
        "tost_equivalent": bool(tost.get("equivalent", False)),
        "edge_net_bps": float(np.mean(ns["net"] - gc["net"]) * BPS),
        "pnl_net_nsdiff_bps": float(np.mean(ns["net"]) * BPS),
    }


def reversal(old: dict, new: dict) -> list:
    out = []
    if old["winkler_verdict"] != new["winkler_verdict"]:
        out.append(f"verdict Winkler : {old['winkler_verdict']} -> {new['winkler_verdict']}")
    if np.sign(old["edge_net_bps"]) != np.sign(new["edge_net_bps"]):
        out.append(f"signe de l'edge net : {old['edge_net_bps']:+.2f} -> "
                   f"{new['edge_net_bps']:+.2f} bps")
    if old["tost_equivalent"] != new["tost_equivalent"]:
        out.append(f"TOST : equivalence {old['tost_equivalent']} -> {new['tost_equivalent']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--garch-pi80", default=str(GARCH_PI80))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows_o, samples_o = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir),
                                     model="NsDiff")
    old = g2t.align({"nsdiff_ensemble": g2t.nsdiff_bands(rows_o.reset_index(drop=True),
                                                         samples_o, 0.95),
                     "garch": g2t.garch_bands(pd.read_parquet(args.garch_pi80), 0.95)})
    rows_n, samples_n, garch_n = g2t.load_arms(Path(args.grid_dir))
    new = g2t.align({"nsdiff_ensemble": g2t.nsdiff_bands(rows_n, samples_n, 0.95),
                     "garch": g2t.garch_bands(garch_n, 0.95)})

    assets_old = set(old["garch"]["asset"])
    assets_new = set(new["garch"]["asset"])
    common = sorted(assets_old & assets_new)
    newcomers = sorted(assets_new - assets_old)
    n_old = old["garch"]["cutoff_date"].nunique()
    n_new = new["garch"]["cutoff_date"].nunique()
    print(f"ancienne grille : {n_old} origines, {len(assets_old)} actifs | "
          f"nouvelle : {n_new} origines, {len(assets_new)} actifs "
          f"(entrants : {', '.join(newcomers) or 'aucun'})")

    def cost(asset):
        insts = rf.instruments_for_asset(asset)
        return rf.one_way_bps(insts[0]) if insts else 0.0

    table, reversals, newcomer_cells = {}, {}, {}
    for asset in common:
        for regime in REGIMES:
            for hu in HORIZONS:
                o = metrics(old, asset, regime, hu, cost(asset))
                n = metrics(new, asset, regime, hu, cost(asset))
                if o is None or n is None:
                    continue
                key = f"{asset}|{regime}|{hu}"
                table[key] = {"ancienne": o, "nouvelle": n}
                r = reversal(o, n)
                if r:
                    reversals[key] = r
    for asset in newcomers:
        for regime in REGIMES:
            for hu in HORIZONS:
                m = metrics(new, asset, regime, hu, cost(asset))
                if m:
                    newcomer_cells[f"{asset}|{regime}|{hu}"] = m

    def pooled(side):
        vals = [v[side] for v in table.values()]
        return {
            "cov95_nsdiff": float(np.mean([x["cov95_nsdiff"] for x in vals])),
            "cov95_garch": float(np.mean([x["cov95_garch"] for x in vals])),
            "edge_net_bps": float(np.mean([x["edge_net_bps"] for x in vals])),
            "n_cells_winkler_nsdiff": sum(1 for x in vals
                                          if x["winkler_verdict"] == "nsdiff_significantly_better"),
            "n_cells_winkler_garch": sum(1 for x in vals
                                         if x["winkler_verdict"] == "garch_significantly_better"),
            "n_cells_tost_equivalent": sum(1 for x in vals if x["tost_equivalent"]),
            "n_cells": len(vals),
        }

    payload = {
        "scope": "chantier R3 -- tableau ancienne grille vs nouvelle grille",
        "grids": {
            "ancienne": {"n_origins": n_old, "effective_n": n_old // g2t.BLOCK_LENGTH,
                         "assets": sorted(assets_old), "prices": "diffusion_multiseed_v2/prices",
                         "source": str(args.v2_dir)},
            "nouvelle": {"n_origins": n_new, "effective_n": n_new // g2t.BLOCK_LENGTH,
                         "assets": sorted(assets_new), "prices": "prices_v3",
                         "source": str(args.grid_dir)},
        },
        "not_comparable_row_by_row": "origines, prix et perimetre different ; seuls les VERDICTS "
                                     "et les grandeurs agregees par cellule se comparent",
        "strategy_for_pnl": STRATEGY, "tost_margin_rel": TOST_MARGIN,
        "per_cell": table,
        "pooled": {"ancienne": pooled("ancienne"), "nouvelle": pooled("nouvelle")},
        "reversals": reversals,
        "n_reversals": len(reversals),
        "newcomers": {"assets": newcomers, "per_cell": newcomer_cells,
                      "status": "premiers entrants -- rapportes a part, jamais melanges au "
                                "tableau de comparaison"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    po, pn = payload["pooled"]["ancienne"], payload["pooled"]["nouvelle"]
    print(f"\n=== agregat sur {po['n_cells']} cellules communes ===")
    print(f"  Cov95 NsDiff   {po['cov95_nsdiff']:.3f} -> {pn['cov95_nsdiff']:.3f}")
    print(f"  Cov95 GARCH    {po['cov95_garch']:.3f} -> {pn['cov95_garch']:.3f}")
    print(f"  edge net moyen {po['edge_net_bps']:+.2f} -> {pn['edge_net_bps']:+.2f} bps")
    print(f"  Winkler gagne par NsDiff  {po['n_cells_winkler_nsdiff']} -> "
          f"{pn['n_cells_winkler_nsdiff']} cellules")
    print(f"  Winkler gagne par GARCH   {po['n_cells_winkler_garch']} -> "
          f"{pn['n_cells_winkler_garch']} cellules")
    print(f"  TOST equivalent           {po['n_cells_tost_equivalent']} -> "
          f"{pn['n_cells_tost_equivalent']} cellules")
    print(f"\n=== renversements : {len(reversals)} cellules sur {len(table)} ===")
    for key in sorted(reversals):
        print(f"  {key:<24} {' ; '.join(reversals[key])}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
