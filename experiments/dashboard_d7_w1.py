"""
dashboard_d7_w1.py — mini-dashboard autonome, focalisé sur la seule comparaison
D+7 (régime A, daily projeté à 7 jours calendaires) vs W+1 (régime C, weekly
natif), sur les origines-vendredi (BRIEF_dashboard_D7_vs_W1.md).

Isolé de tout le reste : ne touche ni Run/dashboard.html, ni model_artifacts/
generate_dashboard.py, ni le pipeline, ni le schema DB. Lit validation/tracking.db
(lecture seule) + télécharge l'historique de prix (yfinance, via
models/arima_model.fetch_data, réutilisé tel quel) pour construire la baseline
random walk. Écrit un unique fichier HTML autonome (aucun CDN/fetch requis pour
l'ouvrir en file://) + un JSON de traçabilité.

Appariement D+7/W+1 : importé de experiments/matrice_paired_tests.py
(build_d7_w1_pairs, comparison_4_d7_vs_w1) -- pas recopié -- pour garantir un
recoupement exact avec matrice_paired_tests.json (comparison_4_d7_vs_w1_friday_aligned).

Note sur les deux horizons réels : pour les cryptos (BTC-USD/ETH-USD, marché
7j/7), le D+7 documenté par matrice_paired_tests.verify_d7_definition cible en
réalité cutoff+5 jours calendaires (pas +7) alors que W+1 cible bien cutoff+7 --
c'est un écart de définition assumé et documenté ailleurs, pas une erreur ici.
Chaque côté (D+7 / W+1) est donc évalué contre SA PROPRE baseline RW, calculée
à SON PROPRE horizon réel (target_date - cutoff_date), jamais contre un horizon
supposé commun de 7 jours.

Tests : bootstrap par blocs (experiments/paired_test.py, réutilisé, jamais
réimplémenté), seed fixe. Le test par cellule (badge RMSE) appelle
comparison_4_d7_vs_w1 tel quel (seed interne = 0, comme matrice_paired_tests.json,
pour un recoupement exact). Les tests poolés (§2.2 du brief) utilisent --seed
(défaut 42) -- cette distinction de seed est documentée dans le pied de page de
la page générée.

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
CELL_TEST_SEED = 0                 # comparison_4_d7_vs_w1 (importé) ne paramètre pas le seed -> toujours 0
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
    pas d'échantillons -> pas de vrai CRPS (cf. brief §2.1)."""
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
    """Brief §6 : "Winkler vérifié sur un cas jouet (borne connue) avant de
    peupler la page." l=90, u=110 (largeur 20) ; alpha=0.05 -> 2/alpha=40."""
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
    aucune fuite de future) -- brief §2.2.1."""
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
    reproductibilité (brief §6, "deux runs à seed égal -> mêmes p-values") : les
    métriques de cellule (RMSE, mean_diff du test par cellule) ne dépendent que de
    la DB et sont déjà bit-à-bit stables d'un run à l'autre, mais sans ce cache,
    chaque run refait un appel réseau à Yahoo Finance dont les tout derniers jours
    peuvent être resservis avec un bruit de dernière décimale d'un appel à l'autre
    -- assez pour faire dériver le 6e-7e chiffre des quantiles RW (donc du skill
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


# ── Assemblage des lignes appariées (une ligne = un (model, asset, origine-vendredi)) ──
def build_enriched_pairs(df: pd.DataFrame, price_cache: dict) -> pd.DataFrame:
    pairs = mpt.build_d7_w1_pairs(df)
    if pairs.empty:
        raise SystemExit("Aucune paire D+7/W+1 trouvée -- la DB a-t-elle des lignes OOS ?")

    pairs["h_d7"] = (pd.to_datetime(pairs["target_date_d7"]) - pd.to_datetime(pairs["cutoff_date"])).dt.days
    pairs["h_w1"] = (pd.to_datetime(pairs["target_date_w1"]) - pd.to_datetime(pairs["cutoff_date"])).dt.days

    # RW calculée une fois par (asset, cutoff_date) unique (indépendante du modèle), puis rejointe.
    uniq = pairs.drop_duplicates(subset=["asset", "cutoff_date"])[
        ["asset", "cutoff_date", "target_date_d7", "target_date_w1", "h_d7", "h_w1",
         "last_close_d7", "last_close_w1", "y_true_d7", "y_true_w1"]
    ].copy()

    returns_cache: dict = {}
    rw_rows = []
    for _, r in uniq.iterrows():
        asset = r["asset"]
        price = price_cache[asset]
        rw_lo_d7, rw_hi_d7 = rw_pi_bounds(returns_cache, asset, price, r["h_d7"], r["cutoff_date"], r["last_close_d7"])
        rw_lo_w1, rw_hi_w1 = rw_pi_bounds(returns_cache, asset, price, r["h_w1"], r["cutoff_date"], r["last_close_w1"])
        rw_rows.append({
            "asset": asset, "cutoff_date": r["cutoff_date"],
            "rw_point_d7": r["last_close_d7"], "rw_lower_d7": rw_lo_d7, "rw_upper_d7": rw_hi_d7,
            "rw_point_w1": r["last_close_w1"], "rw_lower_w1": rw_lo_w1, "rw_upper_w1": rw_hi_w1,
        })
    rw_df = pd.DataFrame(rw_rows)
    pairs = pairs.merge(rw_df, on=["asset", "cutoff_date"], how="left")

    pairs["winkler_d7"] = winkler_score(pairs["y_true_d7"], pairs["y_lower_d7"], pairs["y_upper_d7"])
    pairs["winkler_w1"] = winkler_score(pairs["y_true_w1"], pairs["y_lower_w1"], pairs["y_upper_w1"])
    pairs["rw_sqerror_d7"] = (pairs["rw_point_d7"] - pairs["y_true_d7"]) ** 2
    pairs["rw_sqerror_w1"] = (pairs["rw_point_w1"] - pairs["y_true_w1"]) ** 2
    pairs["rw_winkler_d7"] = winkler_score(pairs["y_true_d7"], pairs["rw_lower_d7"], pairs["rw_upper_d7"])
    pairs["rw_winkler_w1"] = winkler_score(pairs["y_true_w1"], pairs["rw_lower_w1"], pairs["rw_upper_w1"])

    pairs["pi_width_d7"] = pairs["y_upper_d7"] - pairs["y_lower_d7"]
    pairs["pi_width_w1"] = pairs["y_upper_w1"] - pairs["y_lower_w1"]
    pairs["direction_d7"] = direction_correct(pairs["y_true_d7"], pairs["y_pred_d7"], pairs["last_close_d7"])
    pairs["direction_w1"] = direction_correct(pairs["y_true_w1"], pairs["y_pred_w1"], pairs["last_close_w1"])

    # skill sans échelle : 1 - score_modèle/scale_RW(asset). scale_RW est la
    # MEDIANE (pas la valeur à cette origine précise) du score RW de l'actif,
    # poolée sur les deux côtés (d7+w1) et toutes ses origines -- diviser par le
    # score RW de l'origine elle-même est numériquement instable : la RW naïve
    # (persistance) tombe occasionnellement quasi pile sur y_true à une origine
    # donnée (rw_sqerror proche de 0), ce qui fait exploser le ratio à cette
    # seule origine et domine toute moyenne poolée (observé : jusqu'à ~1e7 sur
    # un skill_diff avant ce correctif). La médiane par actif reste "sans
    # échelle entre actifs" (chaque actif normalisé par SA propre difficulté
    # RW typique) tout en étant robuste aux origines où la RW a eu de la chance.
    scale_sqerror = (pd.concat([pairs["rw_sqerror_d7"], pairs["rw_sqerror_w1"]])
                     .groupby(pd.concat([pairs["asset"], pairs["asset"]]).values).median())
    scale_winkler = (pd.concat([pairs["rw_winkler_d7"], pairs["rw_winkler_w1"]])
                     .groupby(pd.concat([pairs["asset"], pairs["asset"]]).values).median())
    pairs["rw_scale_sqerror"] = pairs["asset"].map(scale_sqerror)
    pairs["rw_scale_winkler"] = pairs["asset"].map(scale_winkler)

    pairs["skill_sqerror_d7"] = 1.0 - pairs["sq_error_d7"] / pairs["rw_scale_sqerror"]
    pairs["skill_sqerror_w1"] = 1.0 - pairs["sq_error_w1"] / pairs["rw_scale_sqerror"]
    pairs["skill_winkler_d7"] = 1.0 - pairs["winkler_d7"] / pairs["rw_scale_winkler"]
    pairs["skill_winkler_w1"] = 1.0 - pairs["winkler_w1"] / pairs["rw_scale_winkler"]
    pairs["skill_diff_sqerror"] = pairs["skill_sqerror_d7"] - pairs["skill_sqerror_w1"]   # >0 => D+7 relativement meilleur (skill sans échelle)
    pairs["skill_diff_winkler"] = pairs["skill_winkler_d7"] - pairs["skill_winkler_w1"]

    pairs["asset_class"] = pairs["asset"].map(mpt.ASSET_CLASS)
    return pairs


# ── Panneau 2 : verdict par cellule (model x asset) ──────────────────────────
def build_cell_table(df: pd.DataFrame, pairs: pd.DataFrame) -> list:
    cell_tests = {(r["model"], r["asset"]): r for r in mpt.comparison_4_d7_vs_w1(df)}
    rows = []
    for (model, asset), g in pairs.groupby(["model", "asset"]):
        test = cell_tests.get((model, asset), {"status": "insufficient_data", "n": int(len(g))})
        row = {
            "model": model, "asset": asset, "asset_class": mpt.ASSET_CLASS.get(asset, "?"),
            "n": int(len(g)),
            "rmse_d7": float(np.sqrt(g["sq_error_d7"].mean())),
            "rmse_w1": float(np.sqrt(g["sq_error_w1"].mean())),
            "winkler_d7": float(g["winkler_d7"].mean()),
            "winkler_w1": float(g["winkler_w1"].mean()),
            "cov95_d7": float(g["in_interval_d7"].mean()),
            "cov95_w1": float(g["in_interval_w1"].mean()),
            "pi_width_d7": float(g["pi_width_d7"].mean()),
            "pi_width_w1": float(g["pi_width_w1"].mean()),
            "direction_d7": float(g["direction_d7"].mean()),
            "direction_w1": float(g["direction_w1"].mean()),
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
                "sq_error_diff": float(row["sq_error_d7"] - row["sq_error_w1"]),
                "pi_width_d7": float(row["pi_width_d7"]),
                "pi_width_w1": float(row["pi_width_w1"]),
            }
            for _, row in g.iterrows()
        ]
    return traj


# ── Panneau 3 : agrégat poolé (skill-score sans échelle, groupé par classe) ──
def build_pooled_series(pairs: pd.DataFrame) -> pd.DataFrame:
    """Dédoublonne ZN=F & TLT (corrélées, obligations) en UNE contribution
    "taux" par (model, cutoff_date) -- moyenne des deux, pas deux voix
    indépendantes -- avant tout pooling (brief §2.2 point 2)."""
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
    chronologique. "Blocs = origines consécutives" (brief §2.2.3) : chaque
    élément du vecteur passé au test EST une origine, jamais un mélange."""
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
        # mean_diff > 0 => skill_d7 > skill_w1 en moyenne => D+7 relativement meilleur
        return "daily_D+7_significantly_better" if test["mean_diff"] > 0 else "weekly_native_significantly_better"

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
                   "comparison_4_d7_vs_w1, dont le seed interne est fixé à 0, non paramétrable).")
    p.add_argument("--refresh-prices", action="store_true",
                   help="force le retéléchargement de l'historique de prix (yfinance) au lieu du cache local "
                        f"({PRICE_CACHE_DIR}) -- par défaut le cache est réutilisé s'il couvre la fenêtre requise.")
    args = p.parse_args()

    _selftest_winkler()

    print(f"Chargement des prédictions OOS depuis {args.db_path} ...")
    df = mpt.load_predictions(args.db_path)
    print(f"  {len(df)} lignes OOS chargées.")

    d7_check = mpt.verify_d7_definition(df)

    assets = sorted(df["asset"].unique())
    max_target = df["target_date"].max()
    print("Construction de la baseline random walk (yfinance, quantiles empiriques) ...")
    price_cache = load_price_history_cache(assets, max_target, refresh=args.refresh_prices)

    print("Appariement D+7/W+1 (origines-vendredi, via matrice_paired_tests.build_d7_w1_pairs) ...")
    pairs = build_enriched_pairs(df, price_cache)
    print(f"  {len(pairs)} paires (model, asset, origine-vendredi).")

    cells = build_cell_table(df, pairs)
    trajectories = build_trajectories(pairs)
    aggregate = build_aggregate(pairs, args.seed)

    n_sig_cells = sum(1 for c in cells if c.get("verdict") not in (None, "indistinguishable"))
    print(f"  {len(cells)} cellules, {n_sig_cells} verdicts significatifs (RMSE, seed={CELL_TEST_SEED}).")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": args.db_path,
        "seed_pooled": args.seed,
        "seed_cell_tests": CELL_TEST_SEED,
        "block_length": BLOCK_LENGTH,
        "min_paired_points": MIN_PAIRED_POINTS,
        "alpha": ALPHA,
        "min_rw_quantile_samples": MIN_RW_QUANTILE_SAMPLES,
        "price_history_start": PRICE_HISTORY_START,
        "asset_class_label": ASSET_CLASS_LABEL,
        "d7_definition_check": d7_check,
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
