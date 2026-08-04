"""
compare_weekly_diffusion.py — BRIEF_nsdiff_weekly_parite_et_compa.md, Axe 1:
approfondir la comparaison weekly TSDiff-W vs NsDiff-W au-delà du CRPS/MCS/CV
déjà produits hier (NOTE_duel_nsdiff.md) : calibration 50/80/95, sharpness,
Winkler, PIT, décomposition par actif ET par classe d'actif.

Étape 0 (obligatoire avant d'écrire ce script) : audit de
`duel_backtest_nsdiff.json` / `_swept.json` / `checkpoints_nsdiff/*.json` --
AUCUN des trois ne stocke les intervalles/échantillons par origine, seulement
`point`/`crps`/`crps_empirical` (vérifié directement, pas supposé). Aucun poids
de modèle n'est non plus sérialisé (pas de .pt/.pth dans le repo). Donc
`prob_kpi_common.row_kpis` (qui a besoin du nuage d'échantillons complet, pas
seulement des quantiles -- le PIT empirique est une fraction du nuage) ne peut
PAS être calculé directement depuis les artefacts existants : "calcul direct
des KPI, zéro re-run" (l'option §3.3 idéale) n'est pas disponible.

Décision prise ici (l'autre branche du §3.3) : ni un re-run du duel complet
(6 modèles, sweep d'époques, bootstrap MCS/SPA -- ça, c'est le run coûteux à
NE PAS relancer), ni une réutilisation de checkpoints qui n'existent pas côté
échantillons -- mais une PASSE CIBLÉE, limitée aux 2 modèles de diffusion
(TSDiff-W, NsDiff-W), qui rejoue exactement `duel_backtest.run_asset_duel` /
`run_nsdiff_records` (mêmes fonctions `td.fit_tsdiff`/`nw.fit_weekly`, mêmes
origines déterministes `duel_origins.build_common_origins`, même convention de
seed `seed + k` par origine) SANS re-sweeper les époques -- les époques déjà
sélectionnées hier sont relues telles quelles depuis
`meta_by_asset[asset]["epochs_tsdiff_w"/"epochs_nsdiff_w"]` de
`duel_backtest_nsdiff.json` -- et qui garde le nuage d'échantillons au lieu de
le collapser à `point`/`crps`. Coût : le duel complet (6 modèles, sweep 40/60/
80 par graine) a pris 95s/graine (cf. config.elapsed_s) ; cette passe est 2
modèles sur des époques déjà fixées (pas de sweep), donc nettement moins chère
-- mesurée en smoke test avant le run complet (voir --smoke).

Ne réimplémente rien : row_kpis (coverage/sharpness/Winkler/PIT) vient de
prob_kpi_common.py, fit/forecast viennent de tsdiff_model.py/nsdiff_weekly.py
tels quels -- ce script ne fait qu'orchestrer + agréger.

Recoupement (garde-fou) : chaque ligne recalculée est aussi comparée à la
`crps_empirical` originale stockée dans `duel_backtest_nsdiff.json` pour le
MÊME (asset, seed, horizon, origine) -- si le re-fit ne reproduit pas des
ordres de grandeur proches (même seed, mêmes époques, entraînement néanmoins
non bit-exact -- torch n'est pas déterministe à 100% même graine fixée), c'est
signalé, pas juste silencieusement pris pour argent comptant.

Usage:
    python experiments/compare_weekly_diffusion.py --smoke      # 1 actif, 1 graine, m=20 -- plomberie seule
    python experiments/compare_weekly_diffusion.py               # run complet (5 actifs x 5 graines, m=500)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tsdiff_model as td                                        # noqa: E402
import nsdiff_model as nm                                         # noqa: E402
import nsdiff_weekly as nw                                        # noqa: E402
from weekly_headtohead import ASSETS, HORIZON_WEEKLY, build_weekly, standardized_returns  # noqa: E402
from duel_origins import build_common_origins                     # noqa: E402
from crps_metrics import crps_fair, crps_empirical                # noqa: E402
from prob_kpi_common import row_kpis                               # noqa: E402
from build_kpi_probabilistes import ASSET_CLASS                    # noqa: E402

HORIZONS = ("W1", "W2", "W3")
H_OF = {"W1": 1, "W2": 2, "W3": 3}
DEFAULT_DUEL_JSON = ROOT / "experiments" / "duel_backtest_nsdiff.json"
DEFAULT_OUT = ROOT / "experiments" / "compare_weekly_diffusion.json"


def run_asset_seed(asset_code: str, ticker: str, seed: int, meta: dict, cfg: dict,
                    m_samples: int, k_denoise: int) -> tuple:
    """Rejoue TSDiff-W + NsDiff-W pour (asset, seed) EXACTEMENT comme
    duel_backtest.run_asset_duel/run_nsdiff_records (mêmes origines
    déterministes, mêmes époques déjà sélectionnées -- lues dans `meta`, pas
    re-sweepées), mais retourne le nuage d'échantillons par origine au lieu de
    le réduire à point/crps."""
    daily = td.fetch_data(ticker, cfg["start"], cfg["end"])
    weekly, weekly_dates = build_weekly(daily)
    train_end_pos, val_pos, test_pos = build_common_origins(
        weekly, cfg["n_val"], cfg["n_test"], embargo=cfg.get("embargo_weeks"))
    T0_date = weekly_dates.iloc[train_end_pos]
    train_weekly = weekly.iloc[:train_end_pos + 1]

    if str(T0_date.date()) != meta["train_end"]:
        raise SystemExit(f"[{asset_code}] train_end mismatch: recomputed={T0_date.date()} "
                         f"vs duel meta={meta['train_end']} -- origins not reproducible, abort.")
    test_dates_recomputed = [str(weekly_dates.iloc[m].date()) for m in test_pos]
    if test_dates_recomputed != meta["test_origins"]:
        raise SystemExit(f"[{asset_code}] test_origins mismatch vs duel meta -- abort.")

    epochs_w = meta["epochs_tsdiff_w"]
    epochs_ns = meta["epochs_nsdiff_w"]

    td.set_seed(seed)
    model_w, mu_w, sd_w = td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs_w)
    weekly_z = standardized_returns(weekly, mu_w, sd_w)

    nm.set_seed(seed)
    model_ns, _, mu_ns, sd_ns, _ = nw.fit_weekly(train_weekly, horizon=HORIZON_WEEKLY, epochs=epochs_ns)
    weekly_z_ns = standardized_returns(weekly, mu_ns, sd_ns)

    records = []
    for k, m_pos in enumerate(test_pos):
        origin_date = weekly_dates.iloc[m_pos]
        last_price = float(weekly.iloc[m_pos])
        actuals = [float(weekly.iloc[m_pos + h]) for h in (1, 2, 3)]

        seed_k = seed + k
        td.set_seed(seed_k)
        samples_w = td.forecast_from_fitted(model_w, weekly_z[:m_pos], mu_w, sd_w, last_price,
                                            horizons=[1, 2, 3], n_samples=m_samples,
                                            k_denoise=k_denoise)
        nm.set_seed(seed_k)
        lookback_z = weekly_z_ns[:m_pos][-model_ns.seq_len:]
        samples_ns = nw.forecast_from_fitted_weekly(model_ns, mu_ns, sd_ns, lookback_z, last_price,
                                                     weeks=(1, 2, 3), n_samples=m_samples)

        for wi, h_label in enumerate(HORIZONS):
            actual = actuals[wi]
            for model_name, samples in (("TSDiff", samples_w[wi + 1]),
                                        ("NsDiff", samples_ns[wi + 1]["samples"])):
                kpi = row_kpis(np.asarray(samples, dtype=float), actual)
                records.append({
                    "asset": ticker, "asset_code": asset_code, "asset_class": ASSET_CLASS.get(ticker, "?"),
                    "seed": seed, "horizon": h_label, "model": model_name, "origin": k,
                    "origin_date": str(origin_date.date()), "actual": actual,
                    "crps_fair": crps_fair(np.asarray(samples, dtype=float), actual),
                    **kpi,
                })
    return records, {"epochs_tsdiff_w": epochs_w, "epochs_nsdiff_w": epochs_ns,
                     "train_end": str(T0_date.date())}


def cross_check_vs_yesterday(df: pd.DataFrame, duel_json: dict) -> dict:
    """Compare, par (asset, seed, horizon, origine, model), le crps_empirical
    recalculé ici contre celui stocké hier dans duel_backtest_nsdiff.json --
    garde-fou de fidélité du re-fit (même seed/époques, mais torch non garanti
    bit-exact). Rapporte la corrélation et l'écart relatif médian, PAS juste un
    "OK" muet."""
    orig_rows = []
    for seed_str, seed_blob in duel_json["per_seed"].items():
        for r in seed_blob["records"]:
            if r["model"] in ("TSDiff", "NsDiff"):
                orig_rows.append({
                    "asset_code": r["asset_code"], "seed": int(seed_str), "horizon": r["horizon"],
                    "model": r["model"], "origin": r["origin"], "crps_empirical_orig": r["crps_empirical"],
                })
    orig_df = pd.DataFrame(orig_rows)
    df2 = df.copy()
    df2["crps_empirical_new"] = df2["crps"]   # row_kpis' "crps" IS crps_empirical (prob_kpi_common.crps_from_samples)
    merged = df2.merge(orig_df, on=["asset_code", "seed", "horizon", "model", "origin"], how="inner")
    if merged.empty:
        return {"status": "no_overlap", "n": 0}
    diff = merged["crps_empirical_new"] - merged["crps_empirical_orig"]
    rel = diff.abs() / merged["crps_empirical_orig"].abs().clip(lower=1e-9)
    corr = float(np.corrcoef(merged["crps_empirical_new"], merged["crps_empirical_orig"])[0, 1])
    return {
        "status": "ok", "n": int(len(merged)), "pearson_corr": corr,
        "median_abs_rel_diff": float(rel.median()), "p90_abs_rel_diff": float(rel.quantile(0.9)),
        "note": "corrélation attendue élevée (>0.9) et écart relatif médian modéré (torch "
               "n'est pas garanti bit-exact même à seed fixée) -- pas une reproduction exacte, "
               "une reproduction du MEME protocole (mêmes origines, mêmes époques déjà "
               "sélectionnées, même convention de seed par origine).",
    }


def aggregate(df: pd.DataFrame, group_cols: list) -> list:
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        entry = dict(zip(group_cols, keys))
        entry.update({
            "n": int(len(g)),
            "crps_fair_mean": float(g["crps_fair"].mean()),
            "crps_mean": float(g["crps"].mean()),
            "coverage_50": float(g["cov50"].mean()), "coverage_80": float(g["cov80"].mean()),
            "coverage_95": float(g["cov95"].mean()),
            "sharpness_50": float(g["sharp50"].mean()), "sharpness_80": float(g["sharp80"].mean()),
            "sharpness_95": float(g["sharp95"].mean()),
            "winkler_50": float(g["winkler50"].mean()), "winkler_80": float(g["winkler80"].mean()),
            "winkler_95": float(g["winkler95"].mean()),
            "pit_mean": float(g["pit"].mean()), "pit_std": float(g["pit"].std()),
        })
        rows.append(entry)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="défaut: toutes les graines de --duel-json (config.seeds)")
    p.add_argument("--duel-json", default=str(DEFAULT_DUEL_JSON))
    p.add_argument("--m-samples", type=int, default=None, help="défaut: m_samples du duel (config)")
    p.add_argument("--k-denoise", type=int, default=None, help="défaut: k_denoise du duel (config)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true",
                   help="1 actif (SPY), 1 graine (première de --duel-json), m=20 -- plomberie seule")
    args = p.parse_args()

    duel_json = json.loads(Path(args.duel_json).read_text())
    cfg = duel_json["config"]
    m_samples = args.m_samples or cfg["m_samples"]
    k_denoise = args.k_denoise or cfg["k_denoise"]
    seeds = args.seeds or cfg["seeds"]
    assets = args.assets

    if args.smoke:
        assets = ["SPY"]
        seeds = seeds[:1]
        m_samples = 20
        print(f"--smoke: assets={assets} seeds={seeds} m_samples={m_samples}")

    t_start = time.time()
    all_records = []
    meta_report = {}
    for seed in seeds:
        seed_meta = duel_json["per_seed"][str(seed)]["meta_by_asset"]
        for asset_code in assets:
            ticker = ASSETS[asset_code]
            t0 = time.time()
            recs, m = run_asset_seed(asset_code, ticker, seed, seed_meta[asset_code], cfg,
                                     m_samples, k_denoise)
            all_records.extend(recs)
            meta_report[f"{asset_code}|seed{seed}"] = m
            print(f"[seed={seed}][{asset_code}] {len(recs)} lignes en {time.time() - t0:.0f}s "
                 f"(epochs TSDiff-W={m['epochs_tsdiff_w']}, NsDiff-W={m['epochs_nsdiff_w']})")

    df = pd.DataFrame(all_records)
    elapsed = time.time() - t_start
    print(f"\n{len(df)} lignes générées en {elapsed:.0f}s ({elapsed/60:.1f} min).")

    cross_check = cross_check_vs_yesterday(df, duel_json)
    print(f"Recoupement vs duel_backtest_nsdiff.json: n={cross_check.get('n')} "
         f"pearson_r={cross_check.get('pearson_corr')} "
         f"median_abs_rel_diff={cross_check.get('median_abs_rel_diff')}")

    by_asset_horizon = aggregate(df, ["asset_code", "horizon", "model"])
    by_class_horizon = aggregate(df, ["asset_class", "horizon", "model"])
    by_horizon = aggregate(df, ["horizon", "model"])

    payload = {
        "config": {
            "assets": assets, "seeds": seeds, "m_samples": m_samples, "k_denoise": k_denoise,
            "duel_json_source": args.duel_json,
            "epoch_selection": "réutilise epochs_tsdiff_w/epochs_nsdiff_w déjà sélectionnés dans "
                               "duel_json (aucun re-sweep) -- voir meta_by_asset_seed.",
            "protocol_asymmetry_caveat": (
                "TSDiff-W (duel) = module daily nourri en weekly (tsdiff_model.fit_tsdiff), "
                "seq_len=30 (tsdiff_model.SEQ_LEN), époques sweepées par graine/actif (candidats "
                "40/60/80, verrou E1) -- verrou de sélection retiré ici (on relit la sélection "
                "d'hier, on ne re-sweepe pas). NsDiff-W = module weekly dédié "
                "(nsdiff_weekly.fit_weekly), seq_len=26 (nsdiff_weekly.SEQ_LEN_W), budget "
                "d'époques FIXE et déclaré (pas de sélection par graine/actif). L'avantage CRPS "
                "de NsDiff observé hier (NOTE_duel_nsdiff.md) est donc en partie un effet de "
                "budget/lookback, PAS seulement de modèle -- cf. BRIEF_nsdiff_weekly_parite_et_"
                "compa.md §2. Ne pas conclure 'NsDiff > TSDiff' sans cette réserve."
            ),
            "elapsed_s": round(elapsed, 1),
        },
        "meta_by_asset_seed": meta_report,
        "cross_check_vs_yesterday": cross_check,
        "aggregate_by_asset_horizon_model": by_asset_horizon,
        "aggregate_by_class_horizon_model": by_class_horizon,
        "aggregate_by_horizon_model": by_horizon,
        "per_row": json.loads(df.astype(object).where(pd.notnull(df), None).to_json(orient="records")),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {args.out}")

    print("\n=== Résumé par horizon x modèle (poolé tous actifs/graines) ===")
    for row in by_horizon:
        print(f"{row['horizon']:<4}{row['model']:<8} n={row['n']:<5} "
             f"CRPS={row['crps_mean']:.4f}  Cov50={row['coverage_50']:.2f} "
             f"Cov80={row['coverage_80']:.2f} Cov95={row['coverage_95']:.2f} "
             f"Sharp95={row['sharpness_95']:.3f} Winkler95={row['winkler_95']:.3f} "
             f"PIT_mean={row['pit_mean']:.3f}")


if __name__ == "__main__":
    main()
