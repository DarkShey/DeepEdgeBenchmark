"""
nsdiff_consolidation_tests.py -- taches 1, 2 et 3 du BRIEF "NsDiff :
consolider le verdict daily vs weekly", sur l'artefact multi-graines v2
(n_samples=200, tache 4 faite en amont comme le brief le conseille).

  Tache 1 -- tester FORMELLEMENT le signal calibration (priorite max).
      Le seul apport defendable de la synthese precedente (le weekly est
      mieux calibre) n'y etait lu qu'en descriptif multi-graines. Ici :
        (a) test poole PAR GRAINE sur le skill Winkler (weekly vs daily),
            via `dashboard_d7_w1.build_pooled_series`/`run_pooled_test`
            reutilises tels quels -- exactement ce que la note precedente
            declarait hors budget (§6, "pas de re-test du skill-score poole
            par graine") ;
        (b) le meme test avec les 5 GRAINES POOLEES (metrique moyennee par
            origine, cf. la convention declaree dans nsdiff_v2_data) -- le
            verdict principal ;
        (c) couverture : ecart de couverture bootstrappe par blocs (test de
            reference, sans hypothese d'independance) + Kupiec (POF) et
            Christoffersen (independance / conditionnelle) par graine, en
            complement de manuel ;
        (d) tete-a-tete de couverture weekly - daily, appariee par origine.

  Tache 2 -- W+2 et W+3 en multi-graines : tout ce qui precede est calcule
      aux TROIS horizons, plus le test RMSE par cellule et le TOST. C'est la
      ou se logeait l'ecart residuel vs le modele volatilite (cf. tache 7,
      script separe).

  Tache 3 -- TOST sur le RMSE (marge +/-5% de RMSE relatif, justifiee dans
      le JSON et la note) : "indistinguable" n'est pas "equivalent" ; sans
      TOST, l'absence de difference a effective_n~30 ne prouve rien.

Aucun test reimplemente : bootstrap par blocs, Winkler, skill-score RW,
pooling par classe et test RMSE par cellule viennent de `paired_test`,
`dashboard_d7_w1` et `matrice_paired_tests`. Kupiec / Christoffersen / TOST
sont dans `calibration_tests` (code nouveau, teste unitairement dans
`test_calibration_tests.py`).

Sortie : experiments/nsdiff_consolidation_tests.json
Usage   : python nsdiff_consolidation_tests.py
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

OUT_PATH = Path(__file__).resolve().parent / "nsdiff_consolidation_tests.json"
TARGET_COVERAGE = 0.95
TOST_MARGIN = 0.05        # +/-5% de RMSE relatif -- justification dans MARGIN_RATIONALE
POOL_SEED = 42            # graine du bootstrap poole, comme dashboard_d7_w1.main

MARGIN_RATIONALE = (
    "Marge d'equivalence = +/-5% de RMSE relatif. Justification declaree : "
    "(i) c'est l'ordre de grandeur de la variabilite INTER-GRAINES deja mesuree "
    "sur ce meme modele (CV du RMSE 0.8%-7.3% selon l'actif, "
    "NOTE_compare_daily_vs_weekly_nsdiff.md §6bis) -- une difference de regime "
    "plus petite que le bruit de graine n'est pas exploitable en production ; "
    "(ii) elle est plus stricte que l'ecart RMSE daily->weekly observe sur les "
    "cellules ou la synthese concluait deja 'indistinguable' ; (iii) elle est "
    "fixee A PRIORI, avant de regarder les p-values, et la meme pour les 5 "
    "actifs et les 3 horizons -- aucune marge choisie apres coup pour faire "
    "passer une cellule."
)


# ── tache 1a/1b : skill Winkler (et RMSE) poole ─────────────────────────────

def pooled_skill_block(pairs_by_seed: dict, pairs_pooled: pd.DataFrame) -> dict:
    """(a) un test poole par graine + (b) le test poole sur les graines
    moyennees. Meme fonction de test dans les deux cas."""
    per_seed = {str(s): v2.pooled_skill_test(p, seed=POOL_SEED) for s, p in pairs_by_seed.items()}
    pooled = v2.pooled_skill_test(pairs_pooled, seed=POOL_SEED)

    summary = {}
    for group in ("global", "crypto", "index", "bond"):
        for metric in ("skill_winkler", "skill_sqerror"):
            verdicts, pvals = [], []
            for s, res in per_seed.items():
                if res[group].get("status") != "tested":
                    continue
                verdicts.append(res[group][metric]["verdict"])
                pvals.append(res[group][metric]["p_value"])
            n_sig_weekly = sum(v == "weekly_native_significantly_better" for v in verdicts)
            n_sig_daily = sum(v == "daily_significantly_better" for v in verdicts)
            summary[f"{group}.{metric}"] = {
                "n_seeds_tested": len(verdicts),
                "n_seeds_weekly_significant": n_sig_weekly,
                "n_seeds_daily_significant": n_sig_daily,
                "p_values_by_seed": {s: per_seed[s][group][metric]["p_value"]
                                     for s in per_seed if per_seed[s][group].get("status") == "tested"},
                "median_p_across_seeds": float(np.median(pvals)) if pvals else None,
                "pooled_over_seeds": {k: pooled[group][metric][k] for k in
                                      ("verdict", "p_value", "mean_diff", "ci95_lo", "ci95_hi",
                                       "n", "effective_n")}
                if pooled[group].get("status") == "tested" else pooled[group],
            }
    return {"per_seed": per_seed, "pooled_over_seeds": pooled, "summary": summary}


# ── tache 1c/1d : couverture ────────────────────────────────────────────────

def coverage_block(pairs_by_seed: dict, pairs_pooled: pd.DataFrame) -> dict:
    """Par (actif, regime) : couverture moyenne, test d'ecart bootstrappe par
    blocs sur l'indicateur moyenne-graines, Kupiec/Christoffersen par graine,
    et le tete-a-tete weekly-daily apparie par origine."""
    out = {}
    for asset, g_pooled in pairs_pooled.groupby("asset"):
        g_pooled = g_pooled.sort_values("cutoff_date")
        cell = {"n_origins": int(len(g_pooled))}
        for regime in ("daily", "weekly"):
            col = f"in_interval_{regime}"
            gap = ct.coverage_gap_block_test(g_pooled[col].values, target=TARGET_COVERAGE,
                                             seed=POOL_SEED)

            kupiec, chris = {}, {}
            for s, p in pairs_by_seed.items():
                hits_in = p[p["asset"] == asset].sort_values("cutoff_date")[col].values
                violations = 1.0 - hits_in       # 1 = violation, convention Kupiec
                kupiec[str(s)] = ct.kupiec_lr_uc(violations, alpha_target=1.0 - TARGET_COVERAGE)
                chris[str(s)] = ct.christoffersen_lr_cc(violations, alpha_target=1.0 - TARGET_COVERAGE)

            k_p = [v["p_value"] for v in kupiec.values() if v["status"] == "tested"]
            c_p = [v["p_value"] for v in chris.values() if v["status"] == "tested"]
            cell[regime] = {
                "coverage_seed_averaged": gap.get("coverage"),
                "coverage_gap_block_test": gap,
                "kupiec_by_seed": {s: {"violation_rate": v["violation_rate"], "p_value": v["p_value"]}
                                   for s, v in kupiec.items()},
                "kupiec_n_seeds_rejecting_05": int(sum(p < 0.05 for p in k_p)),
                "kupiec_median_p": float(np.median(k_p)) if k_p else None,
                "christoffersen_cc_n_seeds_rejecting_05": int(sum(p < 0.05 for p in c_p)),
                "christoffersen_cc_median_p": float(np.median(c_p)) if c_p else None,
                "christoffersen_status_by_seed": {s: v["status"] for s, v in chris.items()},
            }

        # tete-a-tete : (couverture weekly - couverture daily) par origine
        diff = (g_pooled["in_interval_weekly"] - g_pooled["in_interval_daily"]).values
        head = ct.coverage_gap_block_test(diff, target=0.0, seed=POOL_SEED)
        head["verdict"] = ("weekly_covers_significantly_more" if head["significant_at_05"] and head["coverage_gap"] > 0
                           else "daily_covers_significantly_more" if head["significant_at_05"]
                           else "indistinguishable")
        cell["head_to_head_coverage"] = head
        out[asset] = cell
    return out


# ── tache 3 : TOST sur le RMSE ──────────────────────────────────────────────

def tost_block(pairs_by_seed: dict, pairs_pooled: pd.DataFrame, cell_tests: dict) -> dict:
    """Par actif (jamais entre actifs : le RMSE est a l'echelle du prix).
    `a` = daily, `b` = weekly -> rmse_ratio > 1 signifie daily moins precis."""
    out = {}
    for asset, g in pairs_pooled.groupby("asset"):
        g = g.sort_values("cutoff_date")
        pooled = ct.tost_relative_rmse(g["sq_error_daily"].values, g["sq_error_weekly"].values,
                                       margin_rel=TOST_MARGIN, seed=POOL_SEED)
        per_seed = {}
        for s, p in pairs_by_seed.items():
            gs = p[p["asset"] == asset].sort_values("cutoff_date")
            per_seed[str(s)] = ct.tost_relative_rmse(gs["sq_error_daily"].values,
                                                     gs["sq_error_weekly"].values,
                                                     margin_rel=TOST_MARGIN, seed=POOL_SEED)
        diff_test = cell_tests.get(asset, {})
        out[asset] = {
            "tost_pooled_over_seeds": pooled,
            "tost_n_seeds_equivalent": int(sum(v["verdict"] == "equivalent" for v in per_seed.values())),
            "tost_verdict_by_seed": {s: v["verdict"] for s, v in per_seed.items()},
            "tost_p_by_seed": {s: v["p_tost"] for s, v in per_seed.items()},
            "difference_test_seed_averaged": {
                "verdict": diff_test.get("verdict"), "p_value": diff_test.get("p_value"),
                "n": diff_test.get("n"), "effective_n": diff_test.get("effective_n"),
            },
            "combined_reading": _combine(pooled, diff_test),
        }
    return out


def _combine(tost: dict, diff_test: dict) -> str:
    """La lecture que le brief demande : difference etablie / equivalence
    etablie / ni l'un ni l'autre (puissance insuffisante)."""
    sig_diff = diff_test.get("verdict") not in (None, "indistinguishable")
    equiv = tost.get("verdict") == "equivalent"
    if sig_diff and equiv:
        return "difference statistiquement significative mais PLUS PETITE que la marge -- effet reel, negligeable en pratique"
    if sig_diff:
        return "difference etablie (et non bornee sous la marge)"
    if equiv:
        return "EQUIVALENCE etablie a la marge declaree -- interchangeables"
    return "ni difference ni equivalence etablies -- puissance insuffisante, ne PAS lire comme 'interchangeables'"


# ── orchestration ───────────────────────────────────────────────────────────

def run_horizon(rows: pd.DataFrame, cache: dict, horizon_unit: str) -> dict:
    print(f"\n=== {horizon_unit} ===")
    seeds = v2.seeds(rows)
    pairs_by_seed = {s: v2.enriched_pairs(rows, cache, horizon_unit, seed=s) for s in seeds}
    pairs_pooled = v2.seed_average(pairs_by_seed)

    # test RMSE par cellule sur les lignes moyennees-graines : on reconstruit des
    # lignes au format DB (y_pred moyen par origine/regime) pour appeler
    # comparison_3_daily_vs_weekly TEL QUEL.
    rows_h = rows[rows["horizon_unit"] == horizon_unit]
    avg_rows = (rows_h.groupby(["model", "asset", "frequence", "horizon_type", "horizon_unit",
                                "cutoff_date", "target_date"], as_index=False)
                .agg(y_pred=("y_pred", "mean"), y_lower=("y_lower", "mean"),
                     y_upper=("y_upper", "mean"), y_true=("y_true", "first"),
                     last_close=("last_close", "first"), sq_error=("sq_error", "mean"),
                     in_interval=("in_interval", "mean")))
    cell_tests = v2.cell_rmse_test(avg_rows, horizon_unit)

    result = {
        "task1_pooled_skill": pooled_skill_block(pairs_by_seed, pairs_pooled),
        "task1_coverage": coverage_block(pairs_by_seed, pairs_pooled),
        "task3_tost_rmse": tost_block(pairs_by_seed, pairs_pooled, cell_tests),
        "descriptive": {
            asset: {
                "rmse_daily": float(np.sqrt(g["sq_error_daily"].mean())),
                "rmse_weekly": float(np.sqrt(g["sq_error_weekly"].mean())),
                "cov95_daily": float(g["in_interval_daily"].mean()),
                "cov95_weekly": float(g["in_interval_weekly"].mean()),
                "winkler_daily": float(g["winkler_daily"].mean()),
                "winkler_weekly": float(g["winkler_weekly"].mean()),
                "pi_width_daily": float(g["pi_width_daily"].mean()),
                "pi_width_weekly": float(g["pi_width_weekly"].mean()),
            } for asset, g in pairs_pooled.groupby("asset")
        },
    }
    _print_horizon(horizon_unit, result)
    return result


def _print_horizon(horizon_unit: str, res: dict) -> None:
    s = res["task1_pooled_skill"]["summary"]
    print(f"\n[{horizon_unit}] Tache 1 -- skill Winkler poole (weekly vs daily)")
    print(f"{'Groupe':<10}{'graines sig. weekly':>22}{'p poole (5 graines)':>22}{'verdict poole':>34}")
    for group in ("global", "crypto", "index", "bond"):
        row = s[f"{group}.skill_winkler"]
        pooled = row["pooled_over_seeds"]
        p = pooled.get("p_value")
        print(f"{group:<10}{row['n_seeds_weekly_significant']}/{row['n_seeds_tested']:<20}"
              f"{(f'{p:.4f}' if p is not None else 'n/a'):>22}{str(pooled.get('verdict')):>34}")

    print(f"\n[{horizon_unit}] Tache 1 -- couverture (cible 95%, graines moyennees)")
    print(f"{'Actif':<10}{'cov daily':>11}{'p ecart':>9}{'cov weekly':>12}{'p ecart':>9}"
          f"{'Kupiec rejets d/w':>19}{'tete-a-tete':>34}")
    for asset, cell in res["task1_coverage"].items():
        d, w = cell["daily"], cell["weekly"]
        kupiec = f"{d['kupiec_n_seeds_rejecting_05']}/5 vs {w['kupiec_n_seeds_rejecting_05']}/5"
        print(f"{asset:<10}{d['coverage_seed_averaged']:>11.3f}"
              f"{d['coverage_gap_block_test']['p_value']:>9.3f}"
              f"{w['coverage_seed_averaged']:>12.3f}"
              f"{w['coverage_gap_block_test']['p_value']:>9.3f}"
              f"{kupiec:>19}{cell['head_to_head_coverage']['verdict']:>34}")

    print(f"\n[{horizon_unit}] Tache 3 -- TOST RMSE (marge +/-{TOST_MARGIN:.0%})")
    print(f"{'Actif':<10}{'ratio RMSE d/w':>16}{'p_TOST':>9}{'graines equiv.':>16}  lecture")
    for asset, cell in res["task3_tost_rmse"].items():
        t = cell["tost_pooled_over_seeds"]
        n_equiv = f"{cell['tost_n_seeds_equivalent']}/5"
        print(f"{asset:<10}{t['rmse_ratio']:>16.4f}{t['p_tost']:>9.4f}{n_equiv:>16}"
              f"  {cell['combined_reading']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", nargs="+", default=v2.HORIZON_UNITS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows = v2.load_rows()
    cfg = v2.load_config()
    print(f"Artefact v2 : {len(rows)} lignes, graines {v2.seeds(rows)}, "
          f"n_samples={cfg['n_samples']}, epochs={cfg['epochs']}")
    cache = v2.price_cache(rows)

    payload = {
        "config": {
            **cfg,
            "target_coverage": TARGET_COVERAGE,
            "tost_margin_rel": TOST_MARGIN,
            "tost_margin_rationale": MARGIN_RATIONALE,
            "pool_seed_bootstrap": POOL_SEED,
            "seed_pooling_convention": v2.__doc__.split("Convention de POOLING")[1].strip(),
            "caveat_chi2": ct.__doc__.split("CAVEAT commun")[1].strip(),
        },
        "horizons": {h: run_horizon(rows, cache, h) for h in args.horizons},
    }
    payload["config"]["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
