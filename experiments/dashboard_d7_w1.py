"""
dashboard_d7_w1.py — mini-dashboard autonome, focalisé sur la comparaison
Daily (régime B, modèle daily évalué à son horizon natif W+1) vs Weekly natif
(régime C), sur l'horizon W+1 -- "pour prévoir 1 semaine, vaut-il mieux un
modèle daily ou un modèle weekly natif ?" (BRIEF_dashboard_D7_vs_W1.md).

Isolé de tout le reste : ne touche ni Run/dashboard.html, ni model_artifacts/
generate_dashboard.py, ni le pipeline, ni le schema DB. Lit validation/tracking.db
(lecture seule) + télécharge l'historique de prix (yfinance, via
models/arima_model.fetch_data, réutilisé tel quel) pour construire la baseline
random walk. Écrit un unique fichier HTML autonome (aucun CDN/fetch requis pour
l'ouvrir en file://) + un JSON de traçabilité.

Choix de l'appariement (v2) : d'abord tenté sur D+7 (régime A, daily projeté à
7 jours calendaires) vs W+1 (régime C), apparié uniquement sur les
origines-vendredi -- ~9-14 paires/cellule, effective_n~3-4, et un écart de
définition d'horizon (crypto: D+7 cible réellement cutoff+5j, pas +7j). Recablé
sur regime B (frequence=daily, horizon_type=weekly, horizon_unit=W+1) vs regime C
(frequence=weekly) : les deux côtés partagent EXACTEMENT le même target_date ET
le même cutoff_date par construction (vérifié : 100% des paires), donnant 30
paires/cellule (effective_n~10) sans aucune approximation d'horizon -- plus
rigoureux, et ça retire l'essentiel des deux pièges méthodologiques (puissance,
confusion horizon x régime) plutôt que de les afficher en gros encadrés.

Appariement importé de experiments/matrice_paired_tests.py
(build_daily_weekly_pairs, comparison_3_daily_vs_weekly) -- pas recopié -- pour
garantir un recoupement exact avec matrice_paired_tests.json
(comparison_3_daily_vs_weekly_per_model, filtré horizon_unit=W+1).

Tests : bootstrap par blocs (experiments/paired_test.py, réutilisé, jamais
réimplémenté), seed fixe. Le test par cellule (badge RMSE) appelle
comparison_3_daily_vs_weekly tel quel (seed interne = 0, comme
matrice_paired_tests.json, pour un recoupement exact). Les tests poolés
utilisent --seed (défaut 42) -- distinction documentée dans le pied de page.

Usage:
    python -m experiments.dashboard_d7_w1 --db-path validation/tracking.db \\
        --out experiments/dashboard_d7_w1.html --seed 42
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
for _p in (EXPERIMENTS_DIR, MODELS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matrice_paired_tests as mpt          # noqa: E402  (path patched above)
from paired_test import paired_block_bootstrap_test  # noqa: E402
from arima_model import fetch_data          # noqa: E402

ALPHA = 0.05                       # PI @ 95% -> Winkler/interval-score alpha
BLOCK_LENGTH = mpt.BLOCK_LENGTH     # 3, identique à matrice_paired_tests
MIN_PAIRED_POINTS = mpt.MIN_PAIRED_POINTS
HORIZON_UNIT = "W+1"                # scope de ce dashboard : "1 semaine" uniquement
CELL_TEST_SEED = 0                 # comparison_3_daily_vs_weekly (importé) ne paramètre pas le seed -> toujours 0
MIN_RW_QUANTILE_SAMPLES = 20       # taille mini d'historique de rendements avant de faire confiance aux quantiles 2.5/97.5
PRICE_HISTORY_START = "2015-01-01"  # yfinance tronque automatiquement si le ticker est plus jeune (ex: ETH-USD ~2017-11)

ASSET_CLASS_LABEL = {"crypto": "Crypto", "index": "Actions", "bond": "Obligations (taux)"}
BOND_ASSETS = {"ZN=F", "TLT"}      # dédoublonnées en une contribution "taux" avant pooling (corrélées)


# ── Winkler / interval score @ (1 - alpha) ───────────────────────────────────
def winkler_score(y_true, lower, upper, alpha: float = ALPHA):
    """Interval Score de Winkler (Gneiting & Raftery 2007, eq. proper scoring
    rule) : pénalise la largeur de l'intervalle, PLUS une pénalité supplémentaire
    si y_true est en dehors, proportionnelle au dépassement et à 2/alpha.
    Seule métrique probabiliste calculable ici : la DB stocke (y_lower, y_upper),
    pas d'échantillons -> pas de vrai CRPS."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score = np.where(below, width + (2.0 / alpha) * (lower - y_true), score)
    score = np.where(above, width + (2.0 / alpha) * (y_true - upper), score)
    return score


def _selftest_winkler() -> None:
    """Winkler vérifié sur un cas jouet (borne connue) avant de peupler la page.
    l=90, u=110 (largeur 20) ; alpha=0.05 -> 2/alpha=40."""
    cases = [
        (100.0, 90.0, 110.0, 20.0),               # dans l'intervalle -> juste la largeur
        (115.0, 90.0, 110.0, 20.0 + 40 * 5.0),     # 5 au-dessus de u -> 220.0
        (80.0, 90.0, 110.0, 20.0 + 40 * 10.0),     # 10 en dessous de l -> 420.0
    ]
    for y, lo, hi, expected in cases:
        got = float(winkler_score(np.array([y]), np.array([lo]), np.array([hi]))[0])
        assert abs(got - expected) < 1e-9, f"Winkler self-test failed: y={y} l={lo} u={hi} got={got} expected={expected}"
    print("Winkler self-test (cas jouet, l=90/u=110/alpha=0.05) : OK "
          "(dans PI=20.0, +5 au-dessus=220.0, -10 en dessous=420.0)")


def direction_correct(y_true, y_pred, last_close):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    last_close = np.asarray(last_close, dtype=float)
    return (np.sign(y_true - last_close) == np.sign(y_pred - last_close)).astype(float)


# ── Baseline random walk : point = dernier close, PI = quantiles empiriques ──
def historical_h_day_returns(price: pd.Series, h_days: int) -> pd.Series:
    """r_h(t) = price[t] / price[asof(t - h_days)] - 1, pour chaque date t de la
    série avec une date antérieure disponible (asof, gère les jours non tradés :
    week-ends/jours fériés pour les actions/obligations). Vectorisé via
    searchsorted sur l'index trié."""
    idx = price.index
    vals = price.values.astype(float)
    target = idx - pd.Timedelta(days=int(h_days))
    pos = idx.searchsorted(target, side="right") - 1
    valid = pos >= 0
    r = pd.Series(vals[valid] / vals[pos[valid]] - 1.0, index=idx[valid])
    return r.sort_index()


def rw_pi_bounds(returns_cache: dict, asset: str, price: pd.Series, h_days: int,
                  cutoff_date: str, last_close: float):
    """PI RW @95% : quantiles [2.5%, 97.5%] des rendements cumulés à h_days
    calendaires, calculés sur la fenêtre <= cutoff_date (walk-forward honnête,
    aucune fuite de future)."""
    key = (asset, int(h_days))
    if key not in returns_cache:
        returns_cache[key] = historical_h_day_returns(price, h_days)
    r = returns_cache[key]
    cutoff_dt = pd.Timestamp(cutoff_date)
    subset = r[r.index <= cutoff_dt].values
    if len(subset) < MIN_RW_QUANTILE_SAMPLES:
        raise ValueError(f"Historique insuffisant pour la baseline RW: {asset} h={h_days}j "
                          f"cutoff={cutoff_date} n={len(subset)} < {MIN_RW_QUANTILE_SAMPLES}")
    q_lo, q_hi = np.quantile(subset, [0.025, 0.975])
    return float(last_close * (1.0 + q_lo)), float(last_close * (1.0 + q_hi))


PRICE_CACHE_DIR = EXPERIMENTS_DIR / ".price_cache_d7_w1"


def load_price_history_cache(assets, end_date: str, refresh: bool = False) -> dict:
    """Un téléchargement yfinance par actif (via arima_model.fetch_data, réutilisé,
    pas recopié), couvrant toute la fenêtre nécessaire aux quantiles walk-forward
    de tous les cutoffs -- puis mis en cache localement (CSV). Nécessaire pour la
    reproductibilité ("deux runs à seed égal -> mêmes p-values") : les métriques
    de cellule (RMSE, mean_diff du test par cellule) ne dépendent que de la DB et
    sont déjà bit-à-bit stables d'un run à l'autre, mais sans ce cache, chaque run
    refait un appel réseau à Yahoo Finance dont les tout derniers jours peuvent
    être resservis avec un bruit de dernière décimale d'un appel à l'autre --
    assez pour faire dériver le 6e-7e chiffre des quantiles RW (donc du skill
    poolé). Le cache élimine cette source de non-déterminisme externe au seed."""
    PRICE_CACHE_DIR.mkdir(exist_ok=True)
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    cache = {}
    for asset in assets:
        cache_file = PRICE_CACHE_DIR / f"{asset.replace('=', '_')}.csv"
        series = None
        if cache_file.exists() and not refresh:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)["close"]
            if cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7):
                series = cached
        if series is None:
            print(f"  Téléchargement historique prix {asset} ({PRICE_HISTORY_START} -> {end}) ...")
            series = fetch_data(asset, PRICE_HISTORY_START, end)
            series.to_frame("close").to_csv(cache_file)
        else:
            print(f"  Historique prix {asset} : cache local ({cache_file.name}) -- reproductibilité d'un run à l'autre.")
        cache[asset] = series
    return cache


# ── Assemblage des lignes appariées (une ligne = un (model, asset, origine W+1)) ──
def build_enriched_pairs(df: pd.DataFrame, price_cache: dict) -> pd.DataFrame:
    pairs = mpt.build_daily_weekly_pairs(df, horizon_units=[HORIZON_UNIT])
    if pairs.empty:
        raise SystemExit(f"Aucune paire daily/weekly trouvée à l'horizon {HORIZON_UNIT} -- "
                          "la DB a-t-elle des lignes OOS ?")

    # côté daily et côté weekly partagent le même target_date (clé de jointure) ET
    # le même cutoff_date (vérifié 100% identiques) -- un seul horizon réel par
    # ligne, une seule baseline RW nécessaire (contrairement à D+7/W+1 où les
    # deux côtés pouvaient cibler des horizons réels différents).
    assert (pairs["cutoff_date_daily"] == pairs["cutoff_date_weekly"]).all(), \
        "cutoff_date devrait être identique entre regime B et regime C à horizon natif -- vérifier la jointure"
    pairs["cutoff_date"] = pairs["cutoff_date_daily"]
    pairs["h"] = (pd.to_datetime(pairs["target_date"]) - pd.to_datetime(pairs["cutoff_date"])).dt.days

    # RW calculée une fois par (asset, cutoff_date) unique, puis rejointe.
    uniq = pairs.drop_duplicates(subset=["asset", "cutoff_date"])[
        ["asset", "cutoff_date", "h", "last_close_daily", "y_true_daily"]
    ].copy()

    returns_cache: dict = {}
    rw_rows = []
    for _, r in uniq.iterrows():
        asset = r["asset"]
        price = price_cache[asset]
        rw_lo, rw_hi = rw_pi_bounds(returns_cache, asset, price, r["h"], r["cutoff_date"], r["last_close_daily"])
        rw_rows.append({
            "asset": asset, "cutoff_date": r["cutoff_date"],
            "rw_point": r["last_close_daily"], "rw_lower": rw_lo, "rw_upper": rw_hi,
        })
    rw_df = pd.DataFrame(rw_rows)
    pairs = pairs.merge(rw_df, on=["asset", "cutoff_date"], how="left")

    pairs["winkler_daily"] = winkler_score(pairs["y_true_daily"], pairs["y_lower_daily"], pairs["y_upper_daily"])
    pairs["winkler_weekly"] = winkler_score(pairs["y_true_weekly"], pairs["y_lower_weekly"], pairs["y_upper_weekly"])
    pairs["rw_sqerror"] = (pairs["rw_point"] - pairs["y_true_daily"]) ** 2   # y_true_daily == y_true_weekly (même target_date)
    pairs["rw_winkler"] = winkler_score(pairs["y_true_daily"], pairs["rw_lower"], pairs["rw_upper"])

    pairs["pi_width_daily"] = pairs["y_upper_daily"] - pairs["y_lower_daily"]
    pairs["pi_width_weekly"] = pairs["y_upper_weekly"] - pairs["y_lower_weekly"]
    pairs["direction_daily"] = direction_correct(pairs["y_true_daily"], pairs["y_pred_daily"], pairs["last_close_daily"])
    pairs["direction_weekly"] = direction_correct(pairs["y_true_weekly"], pairs["y_pred_weekly"], pairs["last_close_weekly"])

    # skill sans échelle : 1 - score_modèle/scale_RW(asset). scale_RW est la
    # MEDIANE (pas la valeur à cette origine précise) du score RW de l'actif --
    # diviser par le score RW de l'origine elle-même est numériquement instable :
    # la RW naïve (persistance) tombe occasionnellement quasi pile sur y_true à
    # une origine donnée (rw_sqerror proche de 0), ce qui fait exploser le ratio
    # à cette seule origine et domine toute moyenne poolée. La médiane par actif
    # reste "sans échelle entre actifs" tout en étant robuste à ces coups de chance.
    scale_sqerror = pairs.groupby("asset")["rw_sqerror"].median()
    scale_winkler = pairs.groupby("asset")["rw_winkler"].median()
    pairs["rw_scale_sqerror"] = pairs["asset"].map(scale_sqerror)
    pairs["rw_scale_winkler"] = pairs["asset"].map(scale_winkler)

    pairs["skill_sqerror_daily"] = 1.0 - pairs["sq_error_daily"] / pairs["rw_scale_sqerror"]
    pairs["skill_sqerror_weekly"] = 1.0 - pairs["sq_error_weekly"] / pairs["rw_scale_sqerror"]
    pairs["skill_winkler_daily"] = 1.0 - pairs["winkler_daily"] / pairs["rw_scale_winkler"]
    pairs["skill_winkler_weekly"] = 1.0 - pairs["winkler_weekly"] / pairs["rw_scale_winkler"]
    pairs["skill_diff_sqerror"] = pairs["skill_sqerror_daily"] - pairs["skill_sqerror_weekly"]   # >0 => daily relativement meilleur
    pairs["skill_diff_winkler"] = pairs["skill_winkler_daily"] - pairs["skill_winkler_weekly"]

    pairs["asset_class"] = pairs["asset"].map(mpt.ASSET_CLASS)
    return pairs


# ── Panneau 2 : verdict par cellule (model x asset) ──────────────────────────
def build_cell_table(df: pd.DataFrame, pairs: pd.DataFrame) -> list:
    all_tests = mpt.comparison_3_daily_vs_weekly(df)
    cell_tests = {(r["model"], r["asset"]): r for r in all_tests if r["horizon_unit"] == HORIZON_UNIT}
    rows = []
    for (model, asset), g in pairs.groupby(["model", "asset"]):
        test = cell_tests.get((model, asset), {"status": "insufficient_data", "n": int(len(g))})
        row = {
            "model": model, "asset": asset, "asset_class": mpt.ASSET_CLASS.get(asset, "?"),
            "n": int(len(g)),
            "rmse_daily": float(np.sqrt(g["sq_error_daily"].mean())),
            "rmse_weekly": float(np.sqrt(g["sq_error_weekly"].mean())),
            "winkler_daily": float(g["winkler_daily"].mean()),
            "winkler_weekly": float(g["winkler_weekly"].mean()),
            "cov95_daily": float(g["in_interval_daily"].mean()),
            "cov95_weekly": float(g["in_interval_weekly"].mean()),
            "pi_width_daily": float(g["pi_width_daily"].mean()),
            "pi_width_weekly": float(g["pi_width_weekly"].mean()),
            "direction_daily": float(g["direction_daily"].mean()),
            "direction_weekly": float(g["direction_weekly"].mean()),
            "status": test.get("status"),
            "verdict": test.get("verdict"),
            "p_value": test.get("p_value"),
            "mean_diff": test.get("mean_diff"),
            "ci95_lo": test.get("ci95_lo"),
            "ci95_hi": test.get("ci95_hi"),
            "effective_n": test.get("effective_n"),
            "block_length": test.get("block_length"),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["model"], r["asset"]))
    return rows


# ── Panneau 4 : trajectoires par origine, pour la cellule sélectionnée ───────
def build_trajectories(pairs: pd.DataFrame) -> dict:
    traj = {}
    for (model, asset), g in pairs.groupby(["model", "asset"]):
        g = g.sort_values("cutoff_date")
        key = f"{model}||{asset}"
        traj[key] = [
            {
                "cutoff_date": row["cutoff_date"],
                "sq_error_diff": float(row["sq_error_daily"] - row["sq_error_weekly"]),
                "pi_width_daily": float(row["pi_width_daily"]),
                "pi_width_weekly": float(row["pi_width_weekly"]),
            }
            for _, row in g.iterrows()
        ]
    return traj


# ── Panneau 3 : agrégat poolé (skill-score sans échelle, groupé par classe) ──
def build_pooled_series(pairs: pd.DataFrame) -> pd.DataFrame:
    """Dédoublonne ZN=F & TLT (corrélées, obligations) en UNE contribution
    "taux" par (model, cutoff_date) -- moyenne des deux, pas deux voix
    indépendantes -- avant tout pooling."""
    cols = ["model", "asset", "asset_class", "cutoff_date", "skill_diff_sqerror", "skill_diff_winkler"]
    base = pairs[cols].copy()
    bonds = base[base["asset"].isin(BOND_ASSETS)]
    non_bonds = base[~base["asset"].isin(BOND_ASSETS)]
    bonds_taux = (bonds.groupby(["model", "cutoff_date"], as_index=False)[["skill_diff_sqerror", "skill_diff_winkler"]]
                  .mean())
    bonds_taux["asset"] = "ZNF+TLT(taux)"
    bonds_taux["asset_class"] = "bond"
    return pd.concat([non_bonds, bonds_taux[cols]], ignore_index=True)


def run_pooled_test(pooled: pd.DataFrame, asset_class: str | None, seed: int) -> dict:
    """Regroupe par ORIGINE (cutoff_date) -- moyenne cross-sectionnelle des
    diffs de skill de tous les (model, asset-ou-taux) contribuant à cette
    origine dans la classe visée (ou toutes classes si asset_class=None) --
    puis bootstrap par blocs (paired_test.py, réutilisé) sur cette série
    chronologique. "Blocs = origines consécutives" : chaque élément du vecteur
    passé au test EST une origine, jamais un mélange."""
    sub = pooled if asset_class is None else pooled[pooled["asset_class"] == asset_class]
    n_contributions = int(len(sub))
    if sub.empty:
        return {"status": "insufficient_data", "n_origins": 0, "n_contributions": 0}
    by_origin = sub.groupby("cutoff_date")[["skill_diff_sqerror", "skill_diff_winkler"]].mean().sort_index()
    n_origins = len(by_origin)
    if n_origins < MIN_PAIRED_POINTS:
        return {"status": "insufficient_data", "n_origins": int(n_origins), "n_contributions": n_contributions}

    block_length = min(BLOCK_LENGTH, n_origins)
    test_sqerror = paired_block_bootstrap_test(by_origin["skill_diff_sqerror"].values,
                                               block_length=block_length, seed=seed)
    test_winkler = paired_block_bootstrap_test(by_origin["skill_diff_winkler"].values,
                                               block_length=block_length, seed=seed)

    def _verdict(test):
        if not test["significant_at_05"]:
            return "indistinguishable"
        # mean_diff > 0 => skill_daily > skill_weekly en moyenne => daily relativement meilleur
        return "daily_significantly_better" if test["mean_diff"] > 0 else "weekly_native_significantly_better"

    return {
        "status": "tested", "n_origins": int(n_origins), "n_contributions": n_contributions,
        "skill_sqerror": {**test_sqerror, "verdict": _verdict(test_sqerror)},
        "skill_winkler": {**test_winkler, "verdict": _verdict(test_winkler)},
    }


def build_aggregate(pairs: pd.DataFrame, seed: int) -> dict:
    pooled = build_pooled_series(pairs)
    result = {"global": run_pooled_test(pooled, None, seed)}
    for cls in ("crypto", "index", "bond"):
        result[cls] = run_pooled_test(pooled, cls, seed)
    return result


# ── Rendu HTML autonome ───────────────────────────────────────────────────────
def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, default=str)
    from dashboard_d7_w1_template import PAGE_TEMPLATE
    return PAGE_TEMPLATE.replace("__DATA_JSON__", data_json)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=str(ROOT / "validation" / "tracking.db"))
    p.add_argument("--out", default=str(EXPERIMENTS_DIR / "dashboard_d7_w1.html"))
    p.add_argument("--data-out", default=str(EXPERIMENTS_DIR / "dashboard_d7_w1_data.json"))
    p.add_argument("--seed", type=int, default=42, help="seed du test poolé (le test par cellule recoupe "
                   "comparison_3_daily_vs_weekly, dont le seed interne est fixé à 0, non paramétrable).")
    p.add_argument("--refresh-prices", action="store_true",
                   help="force le retéléchargement de l'historique de prix (yfinance) au lieu du cache local "
                        f"({PRICE_CACHE_DIR}) -- par défaut le cache est réutilisé s'il couvre la fenêtre requise.")
    args = p.parse_args()

    _selftest_winkler()

    print(f"Chargement des prédictions OOS depuis {args.db_path} ...")
    df = mpt.load_predictions(args.db_path)
    print(f"  {len(df)} lignes OOS chargées.")

    assets = sorted(df["asset"].unique())
    max_target = df["target_date"].max()
    print("Construction de la baseline random walk (yfinance, quantiles empiriques) ...")
    price_cache = load_price_history_cache(assets, max_target, refresh=args.refresh_prices)

    print(f"Appariement daily/weekly à l'horizon natif {HORIZON_UNIT} "
          "(via matrice_paired_tests.build_daily_weekly_pairs) ...")
    pairs = build_enriched_pairs(df, price_cache)
    print(f"  {len(pairs)} paires (model, asset, origine).")

    cells = build_cell_table(df, pairs)
    trajectories = build_trajectories(pairs)
    aggregate = build_aggregate(pairs, args.seed)

    n_sig_cells = sum(1 for c in cells if c.get("verdict") not in (None, "indistinguishable"))
    print(f"  {len(cells)} cellules, {n_sig_cells} verdicts significatifs (RMSE, seed={CELL_TEST_SEED}).")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": args.db_path,
        "horizon_unit": HORIZON_UNIT,
        "seed_pooled": args.seed,
        "seed_cell_tests": CELL_TEST_SEED,
        "block_length": BLOCK_LENGTH,
        "min_paired_points": MIN_PAIRED_POINTS,
        "alpha": ALPHA,
        "min_rw_quantile_samples": MIN_RW_QUANTILE_SAMPLES,
        "price_history_start": PRICE_HISTORY_START,
        "asset_class_label": ASSET_CLASS_LABEL,
        "cells": cells,
        "trajectories": trajectories,
        "aggregate": aggregate,
    }

    Path(args.data_out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"Données -> {args.data_out}")

    html = render_html(payload)
    Path(args.out).write_text(html)
    print(f"Page -> {args.out}")

    print("\nPoint d'étape :")
    print(f"  Cellules testées : {len(cells)} (30 attendues = 6 modèles x 5 actifs)")
    print(f"  Verdicts significatifs par cellule (RMSE) : {n_sig_cells}/{len(cells)}")
    for cls, label in [("global", "Global"), *[(c, ASSET_CLASS_LABEL[c]) for c in ("crypto", "index", "bond")]]:
        agg = aggregate[cls]
        if agg["status"] != "tested":
            print(f"  {label}: insuffisant (n_origines={agg.get('n_origins')})")
            continue
        sq = agg["skill_sqerror"]; wk = agg["skill_winkler"]
        print(f"  {label}: n_origines={agg['n_origins']} (n_contributions={agg['n_contributions']}) "
              f"| skill RMSE: {sq['verdict']} (p={sq['p_value']:.4f}) "
              f"| skill Winkler: {wk['verdict']} (p={wk['p_value']:.4f})")


if __name__ == "__main__":
    main()
