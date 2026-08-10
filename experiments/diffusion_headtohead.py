"""
diffusion_headtohead.py -- machinerie de comparaison INTER-MODELES, partagee
par les matchs du chantier de consolidation NsDiff :

  * `nsdiff_vs_garch_w23.py`   -- NsDiff vs le modele volatilite (tache 7)
  * `nsdiff_vs_tsdiff_v2.py`   -- NsDiff vs TSDiff a budget d'echantillonnage egal

Extraite telle quelle du premier (aucune formule modifiee), generalisee a un
couple (A, B) quelconque au lieu du couple code en dur -- pour que les deux
matchs soient litteralement le meme test, et pas deux implementations qu'il
faudrait comparer avant de comparer les modeles.

Conventions, valables pour les deux matchs :
  * appariement par (actif, horizon, `target_date`), A L'INTERIEUR d'un meme
    regime (weekly-vs-weekly, daily-vs-daily) -- melanger les regimes
    confondrait la question inter-modeles avec la question daily-vs-weekly ;
  * signe : toutes les metriques sont A MINIMISER, donc `mean_diff = A - B`
    negatif favorise A ;
  * tests : bootstrap par blocs (`paired_test.paired_block_bootstrap_test`,
    block_length=3) partout ; ecart de couverture via
    `calibration_tests.coverage_gap_block_test` ; pooling entre actifs via
    les skill-scores vs marche aleatoire et `dashboard_d7_w1.
    build_pooled_series`/`run_pooled_test` reutilises tels quels ;
  * multi-graines : chaque test est rejoue graine par graine ET sur les
    graines poolees (metrique moyennee par origine, `seed_pooled_rows`).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import calibration_tests as ct                                        # noqa: E402
import dashboard_d7_w1 as dash                                        # noqa: E402
import matrice_paired_tests as mpt                                    # noqa: E402
from paired_test import paired_block_bootstrap_test                    # noqa: E402

TARGET_COVERAGE = 0.95
POOL_SEED = 42
BLOCK_LENGTH = 3
MIN_PAIRED_POINTS = 5
POOL_KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date", "target_date"]

# `run_pooled_test` parle le vocabulaire du chantier daily-vs-weekly ; on le
# reutilise TEL QUEL et on retraduit ses verdicts (A joue le role "daily", B
# celui du "weekly" dans la difference de skill).
_POOLED_VERDICT = {
    "daily_significantly_better": "{a}_significantly_better",
    "weekly_native_significantly_better": "{b}_significantly_better",
    "indistinguishable": "indistinguishable",
}


def with_winkler(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows.copy()
    r["winkler"] = dash.winkler_score(r["y_true"], r["y_lower"], r["y_upper"])
    return r


def seed_pooled_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Lignes `seed=-1` portant la metrique MOYENNEE sur les graines, origine
    par origine -- la convention de pooling declaree dans `nsdiff_v2_data`
    (l'unite d'inference reste l'origine, moyenner sur les graines ne fait que
    retirer le bruit Monte-Carlo). `rows` doit deja porter `winkler`."""
    pooled = (rows.groupby(POOL_KEYS, as_index=False)
              .agg(sq_error=("sq_error", "mean"), winkler=("winkler", "mean"),
                   in_interval=("in_interval", "mean"), y_true=("y_true", "first"),
                   last_close=("last_close", "first")))
    pooled["seed"] = -1
    return pooled


def broadcast_seeds(rows: pd.DataFrame, seeds: list) -> pd.DataFrame:
    """Un modele DETERMINISTE (ARIMA-GARCH : pas de graine) produit la meme
    prevision quelle que soit la graine de son adversaire. On le replique donc
    a l'identique sur chaque graine, plus la ligne `seed=-1` du poolage --
    autrement dit, la question posee devient "un run <adversaire> a graine
    tiree au hasard bat-il ce modele ?", et non l'inverse."""
    return pd.concat([rows.assign(seed=s) for s in (-1, *seeds)], ignore_index=True)


def merge_pair(rows_a: pd.DataFrame, rows_b: pd.DataFrame, asset: str, regime: str,
               horizon_unit: str, label_a: str = "A", label_b: str = "B") -> pd.DataFrame:
    """Une ligne par origine partagee, suffixes `_a` / `_b`. Verifie que les
    deux cotes voient la MEME verite terrain -- si `y_true` diverge, les deux
    bras ne regardent pas la meme cible et rien de ce qui suit n'a de sens."""
    cols = ["target_date", "cutoff_date", "last_close", "y_true", "sq_error", "in_interval", "winkler"]

    def _sel(df):
        return df[(df["asset"] == asset) & (df["frequence"] == regime)
                  & (df["horizon_unit"] == horizon_unit)][cols]

    merged = _sel(rows_a).merge(_sel(rows_b), on="target_date", suffixes=("_a", "_b"))
    merged = merged.sort_values("cutoff_date_a")
    if len(merged) and not np.allclose(merged["y_true_a"], merged["y_true_b"]):
        raise SystemExit(f"[{asset}/{regime}/{horizon_unit}] y_true divergent entre "
                         f"{label_a} et {label_b} -- artefacts non comparables")
    return merged


def paired_verdict(diffs, label_a: str, label_b: str, suffix: str = "significantly_better") -> dict:
    """Metrique a MINIMISER : mean_diff < 0 favorise A."""
    d = np.asarray(diffs, dtype=float)
    if d.size < MIN_PAIRED_POINTS:
        return {"status": "insufficient_data", "n": int(d.size)}
    t = paired_block_bootstrap_test(d, block_length=min(BLOCK_LENGTH, d.size), seed=POOL_SEED)
    if not t["significant_at_05"]:
        verdict = "indistinguishable"
    else:
        verdict = f"{label_a}_{suffix}" if t["mean_diff"] < 0 else f"{label_b}_{suffix}"
    return {"status": "tested", "verdict": verdict, **t}


def compare_cell(merged: pd.DataFrame, label_a: str, label_b: str) -> dict:
    if merged.empty:
        return {"status": "no_overlap", "n": 0}
    return {
        "status": "tested", "n": int(len(merged)),
        f"rmse_{label_a}": float(np.sqrt(merged["sq_error_a"].mean())),
        f"rmse_{label_b}": float(np.sqrt(merged["sq_error_b"].mean())),
        "rmse_test": paired_verdict(merged["sq_error_a"] - merged["sq_error_b"], label_a, label_b),
        f"winkler_{label_a}": float(merged["winkler_a"].mean()),
        f"winkler_{label_b}": float(merged["winkler_b"].mean()),
        "winkler_test": paired_verdict(merged["winkler_a"] - merged["winkler_b"], label_a, label_b),
        f"cov95_{label_a}": float(merged["in_interval_a"].mean()),
        f"cov95_{label_b}": float(merged["in_interval_b"].mean()),
        f"coverage_gap_{label_a}": ct.coverage_gap_block_test(merged["in_interval_a"].values,
                                                              target=TARGET_COVERAGE, seed=POOL_SEED),
        f"coverage_gap_{label_b}": ct.coverage_gap_block_test(merged["in_interval_b"].values,
                                                              target=TARGET_COVERAGE, seed=POOL_SEED),
        # couverture : metrique a MAXIMISER -> on teste B - A pour garder
        # "negatif favorise A" et reutiliser paired_verdict sans exception
        "coverage_head_to_head": paired_verdict(merged["in_interval_b"] - merged["in_interval_a"],
                                                label_a, label_b, suffix="covers_more"),
    }


def add_rw_skill(merged: pd.DataFrame, asset: str, cache: dict) -> pd.DataFrame:
    """Skill-scores sans echelle vs marche aleatoire, pour pouvoir POOLER entre
    actifs de prix differents. Assemblage des memes briques que
    `dashboard_d7_w1.build_enriched_pairs` (`rw_pi_bounds`, `winkler_score`,
    mediane par actif comme echelle) ; cette fonction-la n'est pas reutilisable
    telle quelle car elle apparie regime B contre regime C, pas deux modeles."""
    m = merged.copy()
    m["h"] = (pd.to_datetime(m["target_date"]) - pd.to_datetime(m["cutoff_date_a"])).dt.days
    returns_cache: dict = {}
    lo_hi = [dash.rw_pi_bounds(returns_cache, asset, cache[asset], int(r.h), r.cutoff_date_a,
                               float(r.last_close_a)) for r in m.itertuples()]
    m["rw_lower"], m["rw_upper"] = zip(*lo_hi)
    m["rw_sqerror"] = (m["last_close_a"] - m["y_true_a"]) ** 2
    m["rw_winkler"] = dash.winkler_score(m["y_true_a"], m["rw_lower"], m["rw_upper"])

    scale_sq, scale_wk = m["rw_sqerror"].median(), m["rw_winkler"].median()
    m["skill_diff_sqerror"] = (1 - m["sq_error_a"] / scale_sq) - (1 - m["sq_error_b"] / scale_sq)
    m["skill_diff_winkler"] = (1 - m["winkler_a"] / scale_wk) - (1 - m["winkler_b"] / scale_wk)
    m["asset"], m["asset_class"], m["model"] = asset, mpt.ASSET_CLASS[asset], "pair"
    m["cutoff_date"] = m["cutoff_date_a"]
    return m


def pooled_across_assets(frames: list, label_a: str, label_b: str) -> dict:
    """`build_pooled_series` + `run_pooled_test` reutilises tels quels
    (dedoublonnage ZN=F/TLT en une contribution "taux" inclus), verdicts
    retraduits dans le vocabulaire du couple (A, B)."""
    if not frames:
        return {}
    pooled = dash.build_pooled_series(pd.concat(frames, ignore_index=True))
    out = {}
    for label, cls in (("global", None), ("crypto", "crypto"), ("index", "index"), ("bond", "bond")):
        res = dash.run_pooled_test(pooled, cls, POOL_SEED)
        if res.get("status") == "tested":
            for metric in ("skill_sqerror", "skill_winkler"):
                res[metric]["verdict"] = _POOLED_VERDICT[res[metric]["verdict"]].format(
                    a=label_a, b=label_b)
        out[label] = res
    return out


def run_match(rows_a: pd.DataFrame, rows_b: pd.DataFrame, cache: dict, horizon_unit: str,
              seeds: list, assets: list, label_a: str, label_b: str,
              regimes=("weekly", "daily")) -> dict:
    """Le match complet a un horizon : par cellule sur les graines poolees, par
    graine, et poole entre actifs -- pour les deux regimes."""
    out = {"per_regime": {}}
    for regime in regimes:
        per_cell, per_seed, skill_frames = {}, {}, []
        for asset in assets:
            merged = merge_pair(rows_a[rows_a["seed"] == -1], rows_b[rows_b["seed"] == -1],
                                asset, regime, horizon_unit, label_a, label_b)
            per_cell[asset] = compare_cell(merged, label_a, label_b)
            if not merged.empty:
                skill_frames.append(add_rw_skill(merged, asset, cache))
            per_seed[asset] = {}
            for s in seeds:
                m = merge_pair(rows_a[rows_a["seed"] == s], rows_b[rows_b["seed"] == s],
                               asset, regime, horizon_unit, label_a, label_b)
                c = compare_cell(m, label_a, label_b)
                per_seed[asset][str(s)] = {
                    "rmse": c.get("rmse_test", {}).get("verdict"),
                    "rmse_p": c.get("rmse_test", {}).get("p_value"),
                    "winkler": c.get("winkler_test", {}).get("verdict"),
                    "winkler_p": c.get("winkler_test", {}).get("p_value"),
                }
        out["per_regime"][regime] = {
            "per_cell_seed_pooled": per_cell,
            "per_seed_verdicts": per_seed,
            "pooled_across_assets": pooled_across_assets(skill_frames, label_a, label_b),
        }
    return out


def print_match(horizon_unit: str, res: dict, seeds: list, label_a: str, label_b: str) -> None:
    for regime, block in res["per_regime"].items():
        print(f"\n[{horizon_unit}] {label_a}-{regime} vs {label_b}-{regime} (graines poolees)")
        print(f"{'Actif':<10}{f'RMSE {label_a}/{label_b}':>24}{'verdict RMSE':>32}{'graines sig.':>14}"
              f"{f'Cov95 {label_a}/{label_b}':>18}{'verdict Winkler':>32}")
        for asset, c in block["per_cell_seed_pooled"].items():
            if c.get("status") != "tested":
                print(f"{asset:<10}{c.get('status'):>24}")
                continue
            rmse = f"{c[f'rmse_{label_a}']:.4g} / {c[f'rmse_{label_b}']:.4g}"
            cov = f"{c[f'cov95_{label_a}']:.3f} / {c[f'cov95_{label_b}']:.3f}"
            n_sig = sum(v["rmse"] not in (None, "indistinguishable")
                        for v in block["per_seed_verdicts"][asset].values())
            print(f"{asset:<10}{rmse:>24}{str(c['rmse_test']['verdict']):>32}"
                  f"{f'{n_sig}/{len(seeds)}':>14}{cov:>18}{str(c['winkler_test']['verdict']):>32}")
        g = block["pooled_across_assets"].get("global", {})
        if g.get("status") == "tested":
            print(f"  -> poole tous actifs : RMSE {g['skill_sqerror']['verdict']} "
                  f"(p={g['skill_sqerror']['p_value']:.4f}) | Winkler {g['skill_winkler']['verdict']} "
                  f"(p={g['skill_winkler']['p_value']:.4f})")
