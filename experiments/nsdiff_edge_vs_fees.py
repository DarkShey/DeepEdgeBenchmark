"""
nsdiff_edge_vs_fees.py -- chantiers 1.2 et 1.3 du BRIEF "NsDiff : rouvrir la
question economique par le rapport edge/frais".

LA QUESTION, ecrite telle que le brief la pose : « au niveau de frais REEL de
chaque instrument, l'edge net par origine devient-il positif, et le PnL
distinguable de B&H / de GARCH ? »

CE QUI CHANGE PAR RAPPORT AU CHANTIER B, et rien d'autre :

  1. LES FRAIS. Les trois niveaux forfaitaires PAR CLASSE d'actif (1/5/10 bps
     unidirectionnels pour actions-obligations, 10/30/60 pour la crypto) sont
     remplaces par une grille PAR INSTRUMENT (`real_fees.py`), exprimee en
     ALLER-RETOUR tout compris. L'exposition actions y est evaluee sous DEUX
     vehicules -- l'ETF (SPY, 3,5 bps) et le future (ES, 1,5 bps) -- qui
     partagent exactement les memes previsions : le meme edge, execute autrement.
  2. LE NIVEAU DE L'INTERVALLE, pour la famille 3 seulement (chantier 1.3) : 80 %
     au lieu de 95 %. Un seul niveau, declare a priori dans le brief, pas de
     balayage. Les familles 1 et 2 restent a 95 % -- la VaR a 2,5 % de la famille
     2 EST le quantile bas d'un PI a 95 %, la deplacer changerait la strategie.

TOUT LE RESTE EST IDENTIQUE et importe : `econ_backtest` (memes trois familles,
memes regles de decision, memes formules de PnL, meme plafond de position),
`paired_test` (bootstrap par blocs), `mcs.spa_test`, `multiple_testing` (Holm).

LECTURE DE LA MATRICE, dans l'ordre voulu par le brief : la decision se lit sur
l'EDGE NET, la p-value vient ensuite. Trois colonnes portent la decision :

    edge_brut_bps   = moyenne par origine de (PnL brut NsDiff - PnL brut GARCH)
    surcout_bps     = moyenne par origine de (frais NsDiff - frais GARCH)
    edge_net_bps    = edge_brut - surcout

`surcout_bps` est un ECART de frais, pas un niveau : les deux bras paient des
frais sur leur propre exposition, et celles-ci different. Un edge brut positif
peut donc etre annule par une exposition plus lourde -- c'est exactement ce que
cette colonne rend visible. Le niveau de frais absolu de chaque bras est
rapporte a cote (`frais_nsdiff_bps`, `frais_garch_bps`), parce que c'est lui
qui decide si la strategie gagne de l'argent DANS L'ABSOLU.

FAMILLES DE HOLM, declarees avant le run et transposees telles quelles du
chantier B : pour un (strategie, comparaison, INSTRUMENT) donne, la famille est
l'ensemble de ses 6 cellules (3 horizons x 2 regimes). L'instrument est l'unite
de decision -- c'est toute la question du chantier -- donc on ne poole pas entre
instruments, on corrige a l'interieur de chacun.

ASYMETRIE DE PROTOCOLE, inchangee et toujours declaree : ARIMA-GARCH est refit a
chaque origine, NsDiff est train-once-forward. Chiffree au chantier A3-ii : une
cadence de refit jusqu'a 24x plus rapide ne deplace aucun verdict.

Sortie : experiments/nsdiff_edge_vs_fees.json
Usage   : python nsdiff_edge_vs_fees.py
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
import multiple_testing as mt                                         # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
import real_fees as rf                                                # noqa: E402
from mcs import spa_test                                              # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_edge_vs_fees.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
GARCH_BANDS = Path(__file__).resolve().parent / "garch_pi80" / "bands.parquet"
POOL_SEED = 42
BLOCK_LENGTH = 3
HORIZON_WEEKS = {"W+1": 1, "W+2": 2, "W+3": 3}
REGIMES = ("weekly", "daily")
BPS = 1e4

# Famille 3 seule tourne au niveau 80 % (declare a priori dans le brief).
LEVEL_BY_STRATEGY = {"inverse_width": 0.95, "var_limit": 0.95, "filtered_direction": 0.80}
CELL_KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date", "target_date"]


# ── construction des bras, aux deux niveaux d'intervalle ────────────────────

def nsdiff_bands(rows: pd.DataFrame, samples: np.ndarray, level: float) -> pd.DataFrame:
    """Bandes de l'ENSEMBLE production (5 graines concatenees) au niveau demande.
    A 95 % on passe par `nsdiff_production_spec` (la spec, inchangee) ; a 80 % on
    lit les quantiles 10/90 du MEME nuage concatene -- meme materiel, meme
    formule, seul le niveau change."""
    alpha = (1.0 - level) / 2.0
    rows = rows.reset_index(drop=True)
    out = []
    for keys, g in rows.groupby(CELL_KEYS, sort=False):
        cloud = spec.aggregate_cloud(samples[g.index.to_numpy()])
        lo, hi = (float(q) for q in np.quantile(cloud, [alpha, 1.0 - alpha]))
        out.append({**dict(zip(CELL_KEYS, keys)),
                    "y_pred": float(cloud.mean()), "y_lower": lo, "y_upper": hi,
                    "y_true": float(g["y_true"].iloc[0]),
                    "last_close": float(g["last_close"].iloc[0])})
    return pd.DataFrame(out)


def nsdiff_bands_by_seed(rows: pd.DataFrame, samples: np.ndarray, level: float,
                         seed: int) -> pd.DataFrame:
    alpha = (1.0 - level) / 2.0
    sub = rows[rows["seed"] == seed].reset_index(drop=True)
    idx = rows.index[rows["seed"] == seed].to_numpy()
    cloud = samples[idx]
    lo = np.quantile(cloud, alpha, axis=1)
    hi = np.quantile(cloud, 1.0 - alpha, axis=1)
    out = sub[CELL_KEYS + ["y_true", "last_close"]].copy()
    out["y_pred"] = cloud.mean(axis=1)
    out["y_lower"], out["y_upper"] = lo, hi
    return out


def garch_bands(bands: pd.DataFrame, level: float) -> pd.DataFrame:
    lo, hi = ("y_lower", "y_upper") if level == 0.95 else ("y_lower80", "y_upper80")
    out = bands[CELL_KEYS + ["y_true", "last_close", "y_pred"]].copy()
    out["y_lower"], out["y_upper"] = bands[lo].values, bands[hi].values
    return out


def align(frames: dict) -> dict:
    common = None
    for df in frames.values():
        k = set(zip(df["asset"], df["frequence"], df["horizon_unit"], df["cutoff_date"]))
        common = k if common is None else (common & k)
    out = {}
    for name, df in frames.items():
        key = list(zip(df["asset"], df["frequence"], df["horizon_unit"], df["cutoff_date"]))
        out[name] = df[[k in common for k in key]].sort_values(
            ["asset", "frequence", "horizon_unit", "cutoff_date"]).reset_index(drop=True)
    ref = next(iter(out.values()))
    for name, df in out.items():
        if not np.allclose(df["y_true"].values, ref["y_true"].values):
            raise SystemExit(f"bras '{name}' ne voit pas la meme cible")
    return out


def cell_slice(df: pd.DataFrame, asset: str, regime: str, hu: str) -> dict:
    s = df[(df["asset"] == asset) & (df["frequence"] == regime)
           & (df["horizon_unit"] == hu)].sort_values("cutoff_date")
    return {c: s[c].to_numpy(dtype=float)
            for c in ("last_close", "y_pred", "y_lower", "y_upper", "y_true")}


# ── PnL decompose : brut / frais / net ──────────────────────────────────────

def decompose(strategy: str, data: dict, cost_bps_one_way: float) -> dict:
    """Positions, PnL BRUT, frais et PnL NET, separement -- le chantier B ne
    renvoyait que le net, or la question ici porte sur la decomposition."""
    if strategy == "buy_and_hold":
        w = eb.positions_buy_and_hold(data["last_close"])
    else:
        w = eb.STRATEGIES[strategy]["fn"](data)
    w = np.asarray(w, dtype=float)
    r = eb.gross_returns(data["last_close"], data["y_true"])
    gross = w * r
    fees = 2.0 * (cost_bps_one_way * 1e-4) * np.abs(w)
    return {"positions": w, "gross": gross, "fees": fees, "net": gross - fees}


def paired(a, b, label_a: str, label_b: str) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 5:
        return {"status": "insufficient_data", "n": int(d.size)}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    verdict = ("indistinguishable" if not t["significant_at_05"]
               else (f"{label_a}_significantly_better" if t["mean_diff"] > 0
                     else f"{label_b}_significantly_better"))
    return {"status": "tested", "verdict": verdict, **t}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", nargs="+", default=list(eb.STRATEGIES))
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--levels", nargs="+", default=list(rf.LEVELS))
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--garch-bands", default=str(GARCH_BANDS))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir), model="NsDiff")
    seeds = v2.seeds(rows)
    if (len(seeds), samples.shape[1]) != (len(spec.SEEDS), spec.N_SAMPLES_PER_SEED):
        raise SystemExit("artefact hors spec production")
    bands = pd.read_parquet(args.garch_bands)
    print(f"NsDiff : {len(rows)} lignes / {len(seeds)} graines | GARCH : {len(bands)} lignes "
          f"(bandes 95 % et 80 %)")
    print(f"grille de frais (aller-retour, bps) : "
          + " | ".join(f"{k} {rf.round_trip_bps(k):.1f}" for k in rf.INSTRUMENTS))
    print(f"instruments sous l'edge de reference ({rf.EDGE_REFERENCE_BPS} bps) : "
          f"{rf.viable_instruments()}")

    # bras, par niveau d'intervalle
    arms_by_level = {}
    for level in sorted(set(LEVEL_BY_STRATEGY.values())):
        frames = {"nsdiff_ensemble": nsdiff_bands(rows, samples, level),
                  "garch": garch_bands(bands, level)}
        for s in seeds:
            frames[f"nsdiff_seed{s}"] = nsdiff_bands_by_seed(rows, samples, level, s)
        arms_by_level[level] = align(frames)
        print(f"  niveau {level:.0%} : {len(arms_by_level[level]['garch'])} origines alignees")

    seed_arms = [f"nsdiff_seed{s}" for s in seeds]
    cells, pooled_sig = {}, {}
    for strategy in args.strategies:
        level = LEVEL_BY_STRATEGY[strategy]
        arms = arms_by_level[level]
        for inst, spec_inst in rf.INSTRUMENTS.items():
            asset = spec_inst["asset"]
            for hu in args.horizons:
                for regime in REGIMES:
                    data = {n: cell_slice(df, asset, regime, hu) for n, df in arms.items()}
                    for lvl in args.levels:
                        cost = rf.one_way_bps(inst, lvl)
                        dec = {n: decompose(strategy, d, cost) for n, d in data.items()}
                        dec["buy_and_hold"] = decompose("buy_and_hold", data["garch"], cost)
                        dec["nsdiff_pooled"] = {
                            k: np.mean([dec[a][k] for a in seed_arms], axis=0)
                            for k in ("positions", "gross", "fees", "net")}

                        ns, gc, bh = dec["nsdiff_ensemble"], dec["garch"], dec["buy_and_hold"]
                        edge_brut = float(np.mean(ns["gross"] - gc["gross"]) * BPS)
                        surcout = float(np.mean(ns["fees"] - gc["fees"]) * BPS)
                        cells[f"{inst}|{strategy}|{hu}|{regime}|{lvl}"] = {
                            "instrument": inst, "asset": asset, "vehicle": spec_inst["vehicle"],
                            "pi_level": level, "cost_level": lvl,
                            "round_trip_bps": rf.round_trip_bps(inst, lvl),
                            "n": int(ns["net"].size),
                            "edge_brut_bps": edge_brut,
                            "surcout_bps": surcout,
                            "edge_net_bps": edge_brut - surcout,
                            "frais_nsdiff_bps": float(np.mean(ns["fees"]) * BPS),
                            "frais_garch_bps": float(np.mean(gc["fees"]) * BPS),
                            "pnl_net_nsdiff_bps": float(np.mean(ns["net"]) * BPS),
                            "pnl_net_garch_bps": float(np.mean(gc["net"]) * BPS),
                            "pnl_net_bh_bps": float(np.mean(bh["net"]) * BPS),
                            "exposition_moyenne": float(np.mean(np.abs(ns["positions"]))),
                            "n_origines_actives": int((np.abs(ns["positions"]) > 0).sum()),
                            "test_vs_garch": paired(ns["net"], gc["net"], "nsdiff", "garch"),
                            "test_vs_bh": paired(ns["net"], bh["net"], "nsdiff", "buy_and_hold"),
                            "test_pooled_vs_garch": paired(dec["nsdiff_pooled"]["net"], gc["net"],
                                                           "nsdiff_pooled", "garch"),
                        }

    # Holm : famille = les 6 cellules (3 horizons x 2 regimes) d'un
    # (strategie, comparaison, instrument), au niveau de cout de decision
    holm = {}
    for strategy in args.strategies:
        for comp in ("test_vs_garch", "test_vs_bh", "test_pooled_vs_garch"):
            for inst in rf.INSTRUMENTS:
                fam = {k.split("|", 1)[1].rsplit("|", 1)[0]: v[comp] for k, v in cells.items()
                       if v["instrument"] == inst and k.endswith(f"|{rf.DECISION_LEVEL}")
                       and f"|{strategy}|" in k}
                if not fam:
                    continue
                c = mt.correct_family(fam)
                holm[f"{strategy}|{comp}|{inst}"] = {"family": c["family"],
                                                     "summary": mt.family_summary(c)}

    lvl = rf.DECISION_LEVEL
    payload = {
        "question": "au niveau de frais REEL de chaque instrument, l'edge net par origine "
                    "devient-il positif et le PnL distinguable de B&H / de GARCH ?",
        "config": {
            "fee_grid": rf.summary_table(),
            "fee_unit": "aller-retour, bps du notionnel ; converti en cout unidirectionnel "
                        "par real_fees.one_way_bps (le moteur le double lui-meme)",
            "decision_cost_level": lvl,
            "pi_level_by_strategy": LEVEL_BY_STRATEGY,
            "pi_level_rationale": "famille 3 seule a 80 % (declare a priori dans le brief, un "
                                  "seul niveau, pas de balayage) ; familles 1 et 2 restent a "
                                  "95 % -- la VaR a 2,5 % de la famille 2 EST le quantile bas "
                                  "d'un PI a 95 %",
            "arms": {"nsdiff_ensemble": spec.PRODUCTION_SPEC["name"],
                     "nsdiff_pooled": "PnL moyen sur les 5 graines, origine par origine",
                     "garch": "ARIMA-GARCH oos (95 % de la base ; 80 % derive, cf. garch_pi80)",
                     "buy_and_hold": "reference de contexte"},
            "decomposition": "edge_net = edge_brut - surcout ; surcout = ECART de frais entre "
                             "les deux bras (leurs expositions different), pas un niveau",
            "holm_family": "les 6 cellules (3 horizons x 2 regimes) d'un (strategie, "
                           "comparaison, instrument) au niveau de cout de decision -- "
                           "l'instrument est l'unite de decision, on ne poole pas entre eux",
            "protocol_asymmetry": "GARCH refit par origine, NsDiff train-once-forward ; chiffree "
                                  "au chantier A3-ii (refit x24,6 ne deplace aucun verdict)",
        },
        "cells": cells,
        "holm": holm,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    for strategy in args.strategies:
        print(f"\n=== {strategy} (PI {LEVEL_BY_STRATEGY[strategy]:.0%}, cout '{lvl}') ===")
        print(f"{'instrument':<11}{'A/R bps':>9}{'cellule':<16}{'edge brut':>11}{'surcout':>9}"
              f"{'EDGE NET':>10}{'PnL net ns':>12}{'PnL net B&H':>12}{'p vs GARCH':>12}")
        for inst in rf.INSTRUMENTS:
            for hu in args.horizons:
                for regime in REGIMES:
                    c = cells.get(f"{inst}|{strategy}|{hu}|{regime}|{lvl}")
                    if c is None:
                        continue
                    t = c["test_vs_garch"]
                    print(f"{inst:<11}{c['round_trip_bps']:>9.1f}{regime + '|' + hu:<16}"
                          f"{c['edge_brut_bps']:>11.2f}{c['surcout_bps']:>9.2f}"
                          f"{c['edge_net_bps']:>10.2f}{c['pnl_net_nsdiff_bps']:>12.2f}"
                          f"{c['pnl_net_bh_bps']:>12.2f}"
                          f"{(t.get('p_value', float('nan'))):>12.3f}")

    print("\n=== Holm, par (strategie, comparaison, instrument) ===")
    for key, b in sorted(holm.items()):
        s = b["summary"]
        if s["n_significant_raw"] or s["n_significant_holm"]:
            print(f"  {key:<48} m={s['m']:<3} {s['n_significant_raw']} bruts -> "
                  f"{s['n_significant_holm']} Holm"
                  + (f"  survivants : {', '.join(s['survivors'])}" if s["survivors"] else ""))
    total_raw = sum(b["summary"]["n_significant_raw"] for b in holm.values())
    total_holm = sum(b["summary"]["n_significant_holm"] for b in holm.values())
    print(f"  TOTAL sur {len(holm)} familles : {total_raw} rejets bruts -> {total_holm} apres Holm")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
