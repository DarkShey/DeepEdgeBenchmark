"""
nsdiff_vs_garch_econ.py -- chantier B : le juge de paix. NsDiff apporte-t-il de
la VALEUR ECONOMIQUE au-dela d'ARIMA-GARCH, ou, et combien net de couts ?

Le contexte qui rend cette question decisive : depuis le chantier A2, la
configuration candidate production (ensemble 5 graines x 200 tirages) n'est plus
battue par GARCH sur aucun des 6 tests pooles globaux -- elle n'en gagne aucun
non plus. Sur la calibration, les deux modeles sont a PARITE. Quand deux modeles
sont a parite statistique, c'est l'argent qui tranche.

CE QUI EST COMPARE. Trois bras, tous evalues avec LES MEMES regles de decision
(`econ_backtest`, qui ne sait pas quel modele l'appelle) :
  * `nsdiff_ensemble` -- la configuration production (`nsdiff_production_spec`) ;
  * `nsdiff_pooled`   -- la performance ATTENDUE d'un run a graine tiree au
    hasard : PnL moyenne sur les 5 graines, origine par origine. C'est la
    convention de pooling du programme, appliquee a la metrique economique ;
  * `garch`           -- ARIMA-GARCH, lignes `oos` en lecture seule.
Les 5 graines individuelles sont evaluees aussi, pour verifier que le verdict
poole n'est pas porte par une seule d'entre elles (non-negociable du brief).

REFERENCE DE CONTEXTE : `buy_and_hold`. Sans elle, "NsDiff bat GARCH de 40 bps"
ne dit pas si l'une ou l'autre valait mieux que ne rien faire.

COUTS. Trois niveaux declares (`econ_backtest.COST_LEVELS`), par classe
d'actif -- de SPY a ETH, un chiffre unique serait faux partout. Le niveau
`central` porte le verdict ; `faible` et `eleve` sont la robustesse.

POOLING ENTRE ACTIFS. Le PnL est deja sans echelle (rendement par unite de
capital), donc la moyenne cross-sectionnelle par origine est licite -- contrairement
au RMSE, qui exige un skill-score. Le dedoublonnage ZN=F + TLT en une seule
contribution "taux" est repris tel quel de `dashboard_d7_w1.build_pooled_series`
(les deux actifs sont correles, ils ne valent pas deux voix).

TESTS. Bootstrap par blocs apparie par origine (`paired_test`, block_length=3 --
obligatoire, les sleeves se chevauchent a W+2/W+3), et SPA de Hansen
(`mcs.spa_test`, deja dans le repo) avec GARCH en reference. Correction de Holm
sur la famille de decision : les 6 tests pooles (3 horizons x 2 regimes) par
strategie, au niveau de cout central.

Sortie : experiments/nsdiff_vs_garch_econ.json
Usage   : python nsdiff_vs_garch_econ.py
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
import econ_backtest as eb                                            # noqa: E402
import matrice_paired_tests as mpt                                    # noqa: E402
import multiple_testing as mt                                         # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from mcs import spa_test                                              # noqa: E402
from nsdiff_seed_ensemble import build_ensemble_rows                  # noqa: E402
from nsdiff_vs_garch_w23 import load_challenger                       # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_vs_garch_econ.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
POOL_SEED = 42
BLOCK_LENGTH = 3
HORIZON_WEEKS = {"W+1": 1, "W+2": 2, "W+3": 3}
REGIMES = ("weekly", "daily")
BOND_ASSETS = {"ZN=F", "TLT"}
COLS = ["last_close", "y_pred", "y_lower", "y_upper", "y_true"]


# ── assemblage des bras ─────────────────────────────────────────────────────

def cell_data(df: pd.DataFrame, asset: str, regime: str, hu: str) -> pd.DataFrame:
    sub = df[(df["asset"] == asset) & (df["frequence"] == regime) & (df["horizon_unit"] == hu)]
    return sub.sort_values("cutoff_date").reset_index(drop=True)


def align(frames: dict) -> dict:
    """Restreint tous les bras aux origines communes, et verifie qu'ils voient la
    MEME cible. Sans ce controle, un bras qui aurait une origine de plus
    fausserait toutes les differences appariees."""
    common = None
    for df in frames.values():
        keys = set(zip(df["cutoff_date"], df["target_date"]))
        common = keys if common is None else (common & keys)
    common = sorted(common)
    out = {}
    for name, df in frames.items():
        mask = [k in set(common) for k in zip(df["cutoff_date"], df["target_date"])]
        out[name] = df[mask].sort_values("cutoff_date").reset_index(drop=True)
    ref = next(iter(out.values()))
    for name, df in out.items():
        if not np.allclose(df["y_true"].values, ref["y_true"].values):
            raise SystemExit(f"bras '{name}' ne voit pas la meme cible -- non comparable")
    return out


def as_arrays(df: pd.DataFrame) -> dict:
    return {c: df[c].to_numpy(dtype=float) for c in COLS}


# ── evaluation d'une cellule ────────────────────────────────────────────────

def paired_pnl_test(pnl_a, pnl_b, label_a: str, label_b: str) -> dict:
    """a - b sur le PnL : metrique a MAXIMISER, donc mean_diff > 0 favorise a
    (inverse de la convention des metriques d'erreur du repo -- signale)."""
    d = np.asarray(pnl_a, dtype=float) - np.asarray(pnl_b, dtype=float)
    if d.size < 5:
        return {"status": "insufficient_data", "n": int(d.size)}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    if not t["significant_at_05"]:
        verdict = "indistinguishable"
    else:
        verdict = f"{label_a}_significantly_better" if t["mean_diff"] > 0 else f"{label_b}_significantly_better"
    return {"status": "tested", "verdict": verdict, **t}


def evaluate_cell(arms: dict, strategy: str, cost_bps: float, horizon_weeks: int) -> dict:
    """Une cellule = (actif, regime, horizon, strategie, niveau de cout). Tous
    les bras y passent par la meme fonction de decision."""
    results, pnl = {}, {}
    for name, data in arms.items():
        if name == "buy_and_hold":
            positions = eb.positions_buy_and_hold(data["last_close"])
            r = eb.gross_returns(data["last_close"], data["y_true"])
            res = eb.evaluate(positions, r, cost_bps, horizon_weeks)
            res["positions"] = positions.tolist()
        else:
            res = eb.run_strategy(strategy, data, cost_bps, horizon_weeks)
        pnl[name] = np.asarray(res.pop("pnl_series"), dtype=float)
        res.pop("positions")
        results[name] = res
    return results, pnl


def var_block(arms: dict) -> dict:
    """Diagnostics de VaR : violations realisees, cout des depassements, et le
    test formel de Kupiec (`calibration_tests`, importe, jamais recopie) a
    alpha=2.5 % -- le quantile bas d'un PI 95 % EST la VaR a 2.5 %."""
    out = {}
    for name, data in arms.items():
        if name == "buy_and_hold":
            continue
        var = eb.var_from_lower(data["last_close"], data["y_lower"])
        r = eb.gross_returns(data["last_close"], data["y_true"])
        diag = eb.var_diagnostics(r, var, level=eb.VAR_LEVEL)
        flags = diag.pop("breach_flags")
        out[name] = {**diag, "kupiec": ct.kupiec_lr_uc(flags, alpha_target=eb.VAR_LEVEL)}
    return out


# ── pooling entre actifs ────────────────────────────────────────────────────

def pool_across_assets(pnl_by_asset: dict) -> dict:
    """Moyenne cross-sectionnelle du PnL par origine, apres dedoublonnage des
    deux actifs de taux en une seule contribution (meme convention que
    `dashboard_d7_w1.build_pooled_series`). Renvoie {bras: serie par origine}."""
    if not pnl_by_asset:
        return {}
    arms = list(next(iter(pnl_by_asset.values())))
    n = len(next(iter(next(iter(pnl_by_asset.values())).values())))
    out = {}
    for arm in arms:
        contributions = []
        bonds = [pnl_by_asset[a][arm] for a in pnl_by_asset if a in BOND_ASSETS]
        for asset, per_arm in pnl_by_asset.items():
            if asset in BOND_ASSETS:
                continue
            if len(per_arm[arm]) != n:
                return {}          # origines non alignees entre actifs : pas de pooling
            contributions.append(per_arm[arm])
        if bonds:
            if any(len(b) != n for b in bonds):
                return {}
            contributions.append(np.mean(bonds, axis=0))
        out[arm] = np.mean(contributions, axis=0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--strategies", nargs="+", default=list(eb.STRATEGIES))
    p.add_argument("--cost-levels", nargs="+", default=list(eb.COST_LEVELS))
    p.add_argument("--decision-cost-level", default="central")
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir), model="NsDiff")
    seeds = v2.seeds(rows)
    assets = v2.assets(rows)
    if (len(seeds), samples.shape[1]) != (len(spec.SEEDS), spec.N_SAMPLES_PER_SEED):
        raise SystemExit("artefact hors spec production -- refus de produire un chiffre 'production'")
    ensemble = build_ensemble_rows(rows, samples)
    garch = load_challenger(assets, args.horizons)
    print(f"NsDiff : {len(rows)} lignes / {len(seeds)} graines | ensemble : {len(ensemble)} | "
          f"ARIMA-GARCH : {len(garch)} lignes")

    arm_frames = {"nsdiff_ensemble": ensemble, "garch": garch,
                  **{f"nsdiff_seed{s}": rows[rows["seed"] == s] for s in seeds}}
    seed_arms = [f"nsdiff_seed{s}" for s in seeds]

    cells, pooled, var_out = {}, {}, {}
    for hu in args.horizons:
        h_weeks = HORIZON_WEEKS[hu]
        for regime in REGIMES:
            per_asset_arms = {}
            for asset in assets:
                frames = {n: cell_data(df, asset, regime, hu) for n, df in arm_frames.items()}
                aligned = align(frames)
                per_asset_arms[asset] = {n: as_arrays(df) for n, df in aligned.items()}
                per_asset_arms[asset]["buy_and_hold"] = per_asset_arms[asset]["garch"]

            var_out[f"{regime}|{hu}"] = {
                asset: var_block({k: v_ for k, v_ in per_asset_arms[asset].items()
                                  if k in ("nsdiff_ensemble", "garch")})
                for asset in assets}

            for strategy in args.strategies:
                for level in args.cost_levels:
                    pnl_by_asset = {}
                    for asset in assets:
                        cost_bps = eb.COST_LEVELS[level][mpt.ASSET_CLASS[asset]]
                        res, pnl = evaluate_cell(per_asset_arms[asset], strategy, cost_bps, h_weeks)
                        # bras "graine tiree au hasard" : PnL moyenne sur les graines,
                        # origine par origine -- la convention de pooling du programme
                        pnl["nsdiff_pooled"] = np.mean([pnl[a] for a in seed_arms], axis=0)
                        res["nsdiff_pooled"] = {
                            "n": int(pnl["nsdiff_pooled"].size),
                            "pnl_total": float(pnl["nsdiff_pooled"].sum()),
                            "pnl_mean_per_origin": float(pnl["nsdiff_pooled"].mean()),
                            "sharpe_annualised": eb.sharpe(pnl["nsdiff_pooled"], h_weeks),
                            "max_drawdown": eb.max_drawdown(pnl["nsdiff_pooled"]),
                        }
                        pnl_by_asset[asset] = pnl
                        # controle multi-graines : le verdict poole est-il porte par
                        # une seule graine ? Un test par graine, pas deux.
                        seed_verdicts = [paired_pnl_test(pnl[a], pnl["garch"], "s", "g").get("verdict")
                                         for a in seed_arms]
                        cells[f"{asset}|{regime}|{hu}|{strategy}|{level}"] = {
                            "cost_bps_one_way": cost_bps,
                            "arms": res,
                            "test_ensemble_vs_garch": paired_pnl_test(
                                pnl["nsdiff_ensemble"], pnl["garch"], "nsdiff_ensemble", "garch"),
                            "test_pooled_vs_garch": paired_pnl_test(
                                pnl["nsdiff_pooled"], pnl["garch"], "nsdiff_pooled", "garch"),
                            "n_seeds_beating_garch": sum(v == "s_significantly_better" for v in seed_verdicts),
                            "n_seeds_losing_to_garch": sum(v == "g_significantly_better" for v in seed_verdicts),
                        }

                    pooled_pnl = pool_across_assets(pnl_by_asset)
                    key = f"{regime}|{hu}|{strategy}|{level}"
                    if not pooled_pnl:
                        pooled[key] = {"status": "not_poolable"}
                        continue
                    entry = {
                        "status": "tested",
                        "summary": {a: {"pnl_total": float(s.sum()),
                                        "sharpe_annualised": eb.sharpe(s, h_weeks),
                                        "max_drawdown": eb.max_drawdown(s)}
                                    for a, s in pooled_pnl.items()},
                        "test_ensemble_vs_garch": paired_pnl_test(
                            pooled_pnl["nsdiff_ensemble"], pooled_pnl["garch"], "nsdiff_ensemble", "garch"),
                        "test_pooled_vs_garch": paired_pnl_test(
                            pooled_pnl["nsdiff_pooled"], pooled_pnl["garch"], "nsdiff_pooled", "garch"),
                        "test_ensemble_vs_buy_and_hold": paired_pnl_test(
                            pooled_pnl["nsdiff_ensemble"], pooled_pnl["buy_and_hold"],
                            "nsdiff_ensemble", "buy_and_hold"),
                        "test_garch_vs_buy_and_hold": paired_pnl_test(
                            pooled_pnl["garch"], pooled_pnl["buy_and_hold"], "garch", "buy_and_hold"),
                    }
                    # SPA de Hansen : GARCH en reference, perte = -PnL
                    entry["spa_vs_garch"] = spa_test(
                        -pooled_pnl["garch"],
                        {"nsdiff_ensemble": -pooled_pnl["nsdiff_ensemble"],
                         "nsdiff_pooled": -pooled_pnl["nsdiff_pooled"]},
                        block_length=BLOCK_LENGTH, seed=POOL_SEED)
                    pooled[key] = entry

    # Holm sur la famille de decision : 6 tests pooles par strategie, cout central.
    # Le SPA de Hansen y passe AUSSI -- il est teste sur les memes 6 series et pose
    # la meme question ; l'exempter de la correction reviendrait a se reserver le
    # test le plus favorable.
    holm = {}
    for strategy in args.strategies:
        for arm in ("test_ensemble_vs_garch", "test_pooled_vs_garch"):
            fam = {k.rsplit("|", 2)[0]: v[arm] for k, v in pooled.items()
                   if v.get("status") == "tested"
                   and k.endswith(f"|{strategy}|{args.decision_cost_level}")}
            corrected = mt.correct_family(fam)
            holm[f"{strategy}|{arm}"] = {"family": corrected["family"],
                                         "summary": mt.family_summary(corrected)}
        spa_fam = {k.rsplit("|", 2)[0]: {
                       "p_value": v["spa_vs_garch"]["p_value"],
                       "verdict": ("a_model_beats_garch"
                                   if v["spa_vs_garch"]["reject_no_model_beats_benchmark"]
                                   else "indistinguishable"),
                       "per_model_mean_gain_vs_benchmark":
                           v["spa_vs_garch"]["per_model_mean_gain_vs_benchmark"]}
                   for k, v in pooled.items()
                   if v.get("status") == "tested"
                   and k.endswith(f"|{strategy}|{args.decision_cost_level}")}
        corrected = mt.correct_family(spa_fam)
        holm[f"{strategy}|spa_vs_garch"] = {"family": corrected["family"],
                                            "summary": mt.family_summary(corrected)}

    payload = {
        "question": "NsDiff apporte-t-il de la valeur economique au-dela d'ARIMA-GARCH, "
                    "ou, et combien net de couts ?",
        "config": {
            "arms": {"nsdiff_ensemble": spec.PRODUCTION_SPEC["name"],
                     "nsdiff_pooled": "PnL moyenne sur les 5 graines, origine par origine",
                     "nsdiff_seedNN": "les 5 graines individuelles (controle : le verdict poole "
                                      "n'est-il porte que par une graine ?)",
                     "garch": "ARIMA-GARCH, tracking.db source='oos' (lecture seule)",
                     "buy_and_hold": "reference de contexte, pas une famille du brief"},
            "strategies": {k: eb.STRATEGIES[k]["label"] for k in args.strategies},
            "cost_levels_bps_one_way": eb.COST_LEVELS,
            "cost_application": "aller-retour = 2 x bps x |position|, par sleeve",
            "decision_cost_level": args.decision_cost_level,
            "position_cap": eb.W_MAX, "warmup_origins": eb.WARMUP_ORIGINS,
            "var_level": eb.VAR_LEVEL, "var_budget": eb.VAR_BUDGET,
            "overlap": "a l'horizon h, chaque origine ouvre un sleeve tenu h semaines : les "
                       "PnL par origine se chevauchent pour h>1. Traite par le bootstrap PAR "
                       "BLOCS (block_length=3), jamais par un test i.i.d. Sharpe annualise "
                       "par sqrt(52/h).",
            "protocol_asymmetry": "ARIMA-GARCH refit a chaque origine ; NsDiff train-once-forward. "
                                  "Meme reserve que tout le programme -- chiffree par "
                                  "nsdiff_refit_cadence.py.",
            "multiple_testing": f"Holm sur les 6 tests pooles (3 horizons x 2 regimes) par "
                                f"strategie, au niveau de cout '{args.decision_cost_level}'.",
        },
        "cells": cells, "pooled": pooled, "var_diagnostics": var_out, "holm": holm,
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    lvl = args.decision_cost_level
    print(f"\n=== verdicts pooles (tous actifs), niveau de cout '{lvl}' ===")
    for strategy in args.strategies:
        print(f"\n--- {strategy} : {eb.STRATEGIES[strategy]['label']} ---")
        print(f"{'cellule':<16}{'PnL ens/GARCH/B&H':>30}{'Sharpe ens/GARCH':>22}{'verdict ens vs GARCH':>38}")
        for hu in args.horizons:
            for regime in REGIMES:
                e = pooled.get(f"{regime}|{hu}|{strategy}|{lvl}", {})
                if e.get("status") != "tested":
                    continue
                s = e["summary"]
                pnl_txt = (f"{s['nsdiff_ensemble']['pnl_total']:+.3f}/"
                           f"{s['garch']['pnl_total']:+.3f}/{s['buy_and_hold']['pnl_total']:+.3f}")
                sh_txt = f"{s['nsdiff_ensemble']['sharpe_annualised']:+.2f}/{s['garch']['sharpe_annualised']:+.2f}"
                t = e["test_ensemble_vs_garch"]
                verdict_txt = "{} (p={:.3f})".format(t["verdict"], t["p_value"])
                print(f"{regime + '|' + hu:<16}{pnl_txt:>30}{sh_txt:>22}{verdict_txt:>38}")

    print("\n=== Holm sur la famille de decision (6 tests pooles par strategie) ===")
    for key, block in holm.items():
        s = block["summary"]
        print(f"  {key:<48} m={s['m']:<3} {s['n_significant_raw']} bruts -> "
              f"{s['n_significant_holm']} apres Holm"
              + (f"  survivants : {', '.join(s['survivors'])}" if s["survivors"] else ""))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
