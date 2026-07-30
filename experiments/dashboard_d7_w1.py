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


# ── Mini-rapport développé par verdict (§6bis BRIEF_extension_fenetre_validation.md) ──
# Fonctions PURES : n'utilisent que les champs déjà calculés (row de build_cell_table /
# agg de build_aggregate), aucun accès DB/prix supplémentaire, aucun texte figé
# indépendant des chiffres -- le rapport se régénère automatiquement (mêmes fonctions,
# nouveaux chiffres) après un backfill ou un retraining.
REL_GAP_TIE_THRESHOLD = 0.01   # écart relatif en-dessous duquel on affiche "quasi identique" plutôt qu'un camp


def _leaning(value_daily: float, value_weekly: float, mode: str, target: float | None = None):
    """mode: 'lower_better' | 'higher_better' | 'closer_to_target'. Retourne
    (leaning in {'daily','weekly','tie'}, rel_gap >= 0)."""
    if mode == "lower_better":
        daily_better = value_daily < value_weekly
    elif mode == "higher_better":
        daily_better = value_daily > value_weekly
    elif mode == "closer_to_target":
        daily_better = abs(value_daily - target) < abs(value_weekly - target)
    else:
        raise ValueError(mode)
    denom = max(abs(value_daily), abs(value_weekly), 1e-9)
    rel_gap = abs(value_daily - value_weekly) / denom
    if rel_gap < REL_GAP_TIE_THRESHOLD:
        return "tie", rel_gap
    return ("daily" if daily_better else "weekly"), rel_gap


def _verdict_basis_report(status: str, n, effective_n, block_length, p_value,
                          mean_diff, ci95_lo, ci95_hi, metric_label: str) -> dict:
    """Base statistique du verdict, en clair : n/effective_n, IC95, et ce que
    « significatif » veut dire ici (bootstrap par blocs, IC95 excluant 0)."""
    if status != "tested":
        return {
            "text": (f"Base insuffisante pour tester ({metric_label}) : n={n} paire(s) -- "
                     f"minimum requis={MIN_PAIRED_POINTS} -- aucun verdict statistique rendu, "
                     "lecture ci-dessous purement descriptive."),
            "p_value": None, "ci95_lo": None, "ci95_hi": None,
            "n": n, "effective_n": None, "block_length": None,
        }
    sig = p_value < 0.05
    text = (
        f"Test bootstrap par blocs sur {metric_label} : {n} origines appariées "
        f"(blocs de {block_length} origines consécutives -> {effective_n} blocs quasi "
        f"indépendants, c'est la puissance statistique réelle, pas n={n}). "
        f"Différence moyenne (daily − weekly) = {mean_diff:+.4f}, IC95% bootstrap = "
        f"[{ci95_lo:+.4f}, {ci95_hi:+.4f}] : cet intervalle {'exclut' if sig else 'inclut'} 0 "
        f"(p={p_value:.4f}) -> résultat {'significatif' if sig else 'non significatif'}. "
        "« Significatif » signifie ici que l'intervalle de confiance à 95% (obtenu par "
        f"rééchantillonnage par blocs de {block_length} origines, pour respecter l'autocorrélation "
        "entre origines hebdomadaires qui se chevauchent) de la différence d'erreur appariée "
        "exclut zéro."
    )
    return {
        "text": text, "p_value": p_value, "ci95_lo": ci95_lo, "ci95_hi": ci95_hi,
        "n": n, "effective_n": effective_n, "block_length": block_length,
    }


CELL_KPI_SPECS = [
    {"key": "rmse", "label": "RMSE (précision du point -- décide le verdict)", "mode": "lower_better", "fmt": "{:.3f}"},
    {"key": "winkler", "label": "Winkler / Interval Score (fiabilité probabiliste)", "mode": "lower_better", "fmt": "{:.3f}"},
    {"key": "cov95", "label": "Cov95 (calibration, cible 95%)", "mode": "closer_to_target", "target": 0.95, "fmt": "{:.1%}"},
    {"key": "pi_width", "label": "Largeur PI 95% (finesse, à couverture comparable)", "mode": "lower_better", "fmt": "{:.3f}"},
    {"key": "direction", "label": "Direction correcte (diagnostic, bruité à n≈30)", "mode": "higher_better", "fmt": "{:.1%}"},
]


def _lean_phrase(leaning: str, side_word: str, comparative: str, gap_txt: str) -> str:
    """'{daily/weekly} plus {bas/haut/étroit} ({gap})' ou 'quasi identique entre les deux côtés
    ({gap})' si tie -- évite l'accord bancal 'quasi identiques plus bas' d'un gabarit unique."""
    if leaning == "tie":
        return f"quasi identique entre les deux côtés ({gap_txt})"
    return f"{side_word} {comparative} ({gap_txt})"


def _kpi_note(key: str, leaning: str, rel_gap: float, value_daily: float, value_weekly: float, row: dict) -> str:
    side_word = {"daily": "daily", "weekly": "weekly natif"}.get(leaning)
    gap_txt = f"écart relatif {rel_gap*100:.1f}%"
    if key == "rmse":
        phrase = _lean_phrase(leaning, side_word, "plus bas", gap_txt)
        if row["status"] == "tested":
            agree = ("cohérent avec le verdict testé ci-dessus" if row.get("verdict") != "indistinguishable"
                     else "mais le verdict reste indistinguable (l'écart n'est pas significatif)")
            return f"RMSE {phrase} -- {agree}."
        return f"RMSE {phrase} -- lecture descriptive, non testée (n insuffisant)."
    if key == "winkler":
        phrase = _lean_phrase(leaning, side_word, "plus bas", gap_txt)
        return (f"Winkler {phrase} -- lecture descriptive, "
                "non testée à ce niveau de cellule (seul le RMSE est testé statistiquement par cellule).")
    if key == "cov95":
        gap_pt = abs(value_daily - value_weekly) * 100
        lean_txt = ("quasi identiques, toutes deux" if leaning == "tie" else f"{side_word}")
        return (f"Cov95 daily={value_daily*100:.1f}% / weekly={value_weekly*100:.1f}% (cible 95%) -- "
                f"{lean_txt} plus proche de la cible (écart entre les deux : {gap_pt:.1f} pt).")
    if key == "pi_width":
        phrase = _lean_phrase(leaning, side_word, "plus étroit", gap_txt)
        cov_gap = abs(row["cov95_daily"] - row["cov95_weekly"])
        caveat = (" ATTENTION : couverture (Cov95) sensiblement différente entre les deux côtés -> comparer "
                  "la largeur seule n'est pas informatif ici." if cov_gap > 0.05 else
                  " Couverture comparable des deux côtés -> comparaison de largeur informative.")
        return f"PI {phrase}.{caveat}"
    if key == "direction":
        phrase = _lean_phrase(leaning, side_word, "plus haut", gap_txt)
        return (f"Direction correcte daily={value_daily*100:.1f}% / weekly={value_weekly*100:.1f}% "
                f"({phrase}) -- diagnostic à lire avec prudence, proche du hasard (50%) à n≈{row['n']}.")
    raise ValueError(key)


def build_cell_report(row: dict) -> dict:
    basis = _verdict_basis_report(row["status"], row["n"], row.get("effective_n"), row.get("block_length"),
                                  row.get("p_value"), row.get("mean_diff"), row.get("ci95_lo"), row.get("ci95_hi"),
                                  "erreur quadratique (RMSE)")
    kpi_readings = []
    for spec in CELL_KPI_SPECS:
        vd, vw = row[f"{spec['key']}_daily"], row[f"{spec['key']}_weekly"]
        leaning, rel_gap = _leaning(vd, vw, spec["mode"], spec.get("target"))
        kpi_readings.append({
            "key": spec["key"], "label": spec["label"],
            "value_daily_display": spec["fmt"].format(vd), "value_weekly_display": spec["fmt"].format(vw),
            "leaning": leaning,
            "note": _kpi_note(spec["key"], leaning, rel_gap, vd, vw, row),
        })

    descriptive = [k for k in kpi_readings if k["key"] != "rmse"]
    n_daily = sum(1 for k in descriptive if k["leaning"] == "daily")
    n_weekly = sum(1 for k in descriptive if k["leaning"] == "weekly")
    n_tie = sum(1 for k in descriptive if k["leaning"] == "tie")
    rmse_leaning = kpi_readings[0]["leaning"]
    winkler_leaning = kpi_readings[1]["leaning"]

    if row["status"] != "tested" or row.get("verdict") in (None, "indistinguishable"):
        conflict = False
        arbitrage_text = ("Pas de gagnant : verdict indistinguable (ou non testé) au niveau de la cellule, "
                          "même si certains KPI penchent d'un côté ci-dessus -- ces écarts descriptifs ne "
                          "sont pas garantis significatifs à cette échelle.")
    elif rmse_leaning != "tie" and winkler_leaning != "tie" and rmse_leaning != winkler_leaning:
        conflict = True
        arbitrage_text = (f"Arbitrage point/incertitude : le point (RMSE) favorise {rmse_leaning} tandis que "
                          f"l'incertitude (Winkler) favorise {winkler_leaning} -- un modèle peut être plus "
                          "précis en moyenne mais moins fiable dans son intervalle, ou l'inverse.")
    elif rmse_leaning == winkler_leaning and rmse_leaning != "tie":
        conflict = False
        arbitrage_text = (f"Concordant : le point (RMSE) et l'incertitude (Winkler) favorisent tous deux {rmse_leaning}.")
    else:
        conflict = False
        arbitrage_text = "Winkler quasi identique entre les deux côtés -- pas d'arbitrage à signaler."

    return {
        "verdict_basis": basis,
        "kpi_readings": kpi_readings,
        "arbitrage": {"n_daily": n_daily, "n_weekly": n_weekly, "n_tie": n_tie,
                     "conflict": conflict, "text": arbitrage_text},
    }


def build_aggregate_report(agg: dict) -> dict:
    sq_report = _verdict_basis_report("tested", agg["skill_sqerror"]["n"], agg["skill_sqerror"]["effective_n"],
                                      agg["skill_sqerror"]["block_length"], agg["skill_sqerror"]["p_value"],
                                      agg["skill_sqerror"]["mean_diff"], agg["skill_sqerror"]["ci95_lo"],
                                      agg["skill_sqerror"]["ci95_hi"], "skill RMSE (précision du point, sans échelle vs RW)")
    wk_report = _verdict_basis_report("tested", agg["skill_winkler"]["n"], agg["skill_winkler"]["effective_n"],
                                      agg["skill_winkler"]["block_length"], agg["skill_winkler"]["p_value"],
                                      agg["skill_winkler"]["mean_diff"], agg["skill_winkler"]["ci95_lo"],
                                      agg["skill_winkler"]["ci95_hi"], "skill Winkler (fiabilité de l'incertitude, sans échelle vs RW)")
    sq_v, wk_v = agg["skill_sqerror"]["verdict"], agg["skill_winkler"]["verdict"]
    if sq_v == wk_v:
        synth = ("Concordant : les deux axes sont indistinguables, pas de gagnant." if sq_v == "indistinguishable"
                 else f"Concordant : précision et incertitude désignent toutes deux « {sq_v} ».")
    elif sq_v == "indistinguishable" or wk_v == "indistinguishable":
        winner = sq_v if sq_v != "indistinguishable" else wk_v
        axis_name = "précision (RMSE)" if sq_v != "indistinguishable" else "incertitude (Winkler)"
        synth = (f"Partiel : seul l'axe {axis_name} est significatif (« {winner} »), l'autre axe reste "
                "indistinguable -- pas de synthèse unique, lire les deux verdicts séparément.")
    else:
        synth = (f"Arbitrage : la précision (RMSE) désigne « {sq_v} » tandis que la fiabilité de "
                f"l'incertitude (Winkler) désigne « {wk_v} » -- les deux axes ne s'accordent pas.")
    return {"skill_sqerror": sq_report, "skill_winkler": wk_report, "synthesis": synth}


# ── Traduction en langage clair (BRIEF_dashboard_clarte.md) ──────────────────
# Même règle que §6bis : fonctions PURES des métriques déjà calculées (aucun
# recalcul, aucun accès DB/prix supplémentaire, aucun texte figé indépendant des
# chiffres) -- seule la PRÉSENTATION change, calculs/verdicts/chiffres identiques.
# Un "match nul" reste un "match nul" même si un KPI penche légèrement (§3, §6bis).
RELIABILITY_LOW, RELIABILITY_MID = 15, 40   # seuils effective_n (brief §4)


def reliability_gauge(effective_n) -> dict:
    if effective_n is None:
        return {"level": "inconnue", "emoji": "⚪",
                "label": "Pas assez de recul pour évaluer la fiabilité du verdict"}
    if effective_n < RELIABILITY_LOW:
        return {"level": "faible", "emoji": "🔴",
                "label": "Fiabilité faible -- peu de recul, à prendre avec prudence"}
    if effective_n < RELIABILITY_MID:
        return {"level": "moyenne", "emoji": "🟠", "label": "Fiabilité moyenne"}
    return {"level": "forte", "emoji": "🟢", "label": "Fiabilité forte -- on peut s'appuyer dessus"}


def _axis_outcome(status: str, verdict) -> str:
    """'daily' | 'weekly' | 'tie' | 'insufficient' à partir d'un statut+verdict de
    test (peu importe la variante de nommage -- daily_multistep_* inclus, cf. le
    commentaire équivalent dans le template pour le badge technique)."""
    if status != "tested" or verdict is None:
        return "insufficient"
    if verdict in ("daily_significantly_better", "daily_multistep_significantly_better"):
        return "daily"
    if verdict == "weekly_native_significantly_better":
        return "weekly"
    return "tie"


def _decapitalize(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _join_fr(items: list) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} et {items[1]}"
    return f"{', '.join(items[:-1])} et {items[-1]}"


CANONICAL_HEADLINE = {
    "daily": "Le modèle quotidien est meilleur",
    "weekly": "Le modèle hebdomadaire est meilleur",
    "tie": "Match nul — aucune différence fiable",
    "insufficient": "Pas assez de recul pour trancher",
}
# KPI 1 -- "le prix prévu est-il proche de la réalité ?" (précision, ex-RMSE)
PRECISION_PHRASE = {
    "daily": "le quotidien est un peu plus précis",
    "weekly": "l'hebdomadaire est un peu plus précis",
    "tie": "prévision aussi précise des deux côtés",
    "insufficient": "pas assez de recul pour comparer la précision",
}
# KPI 2 -- "la fourchette de prix est-elle fiable ?" (ex-Winkler + ex-Cov95, fusionnés)
RELIABILITY_PHRASE = {
    "daily": "fourchette de prix plus fiable côté quotidien",
    "weekly": "fourchette de prix plus fiable côté hebdomadaire",
    "tie": "fourchette de prix tout aussi fiable des deux côtés",
    "insufficient": "pas assez de recul pour juger la fourchette de prix",
}
PRECISION_QUESTION_ANSWER = {
    "daily": "Le modèle quotidien prévoit un prix un peu plus proche de la réalité.",
    "weekly": "Le modèle hebdomadaire prévoit un prix un peu plus proche de la réalité.",
    "tie": "Les deux modèles prévoient le prix avec une précision comparable.",
    "insufficient": "On n'a pas encore assez de recul pour comparer la précision.",
}
RELIABILITY_QUESTION_ANSWER = {
    "daily": ("La fourchette de prix est plus fiable côté quotidien (le vrai prix tombe dedans "
             "plus souvent, sans qu'elle soit trop large)."),
    "weekly": ("La fourchette de prix est plus fiable côté hebdomadaire (le vrai prix tombe dedans "
              "plus souvent, sans qu'elle soit trop large)."),
    "tie": "La fourchette de prix est tout aussi fiable des deux côtés.",
    "insufficient": "On n'a pas encore assez de recul pour juger la fiabilité de la fourchette.",
}


def _headline(precision: str, reliability: str) -> str:
    """Verdict en gros (§2). Concordant -> phrase unique (§3). Sinon, composé à
    partir des 2 questions KPI (§5), comme l'exemple du brief ("aussi précis des
    deux côtés, mais fourchette plus fiable côté hebdo")."""
    if precision == reliability:
        return CANONICAL_HEADLINE[precision]
    phrase = PRECISION_PHRASE[precision]
    return f"{phrase[0].upper()}{phrase[1:]}, mais {RELIABILITY_PHRASE[reliability]}."


def _one_liner(precision: str, reliability: str) -> str:
    """Phrase de « pourquoi » courte affichée par défaut sous le verdict (niveau 2, §2)."""
    if precision == reliability:
        if precision == "tie":
            return "Précision comparable et fourchette tout aussi fiable des deux côtés."
        if precision == "insufficient":
            return "Pas encore assez de semaines de recul pour comparer."
        side = "quotidien" if precision == "daily" else "hebdomadaire"
        return f"Plus précis, et fourchette plus fiable, côté {side}."
    return f"{PRECISION_PHRASE[precision][0].upper()}{PRECISION_PHRASE[precision][1:]}, {RELIABILITY_PHRASE[reliability]}."


def _certainty_sentence(effective_n) -> str:
    """Traduit la jauge 🔴🟠🟢 en phrase, sans chiffre de p/IC (§6bis point 3)."""
    gauge = reliability_gauge(effective_n)
    if gauge["level"] == "inconnue":
        return "On n'a pas encore assez de semaines de recul pour juger la fiabilité de ce verdict."
    qualifier = {"faible": "à prendre avec prudence", "moyenne": "raisonnablement fiable",
                "forte": "solide"}[gauge["level"]]
    return f"On a {effective_n} semaines de recul réellement exploitables pour ce verdict : {qualifier}."


def build_plain_explanation(group_label: str, precision: str, reliability: str, effective_n) -> list:
    """3 phrases en clair (§6bis) : 1) ce que dit le verdict, 2) pourquoi (les 2
    questions KPI), 3) à quel point on est sûr. Gabarit de phrases + valeurs
    injectées -- rien écrit à la main indépendamment des chiffres."""
    if precision == reliability:
        verdict_sentence = f"Sur {group_label}, {_decapitalize(CANONICAL_HEADLINE[precision])}."
    else:
        verdict_sentence = (f"Sur {group_label}, {PRECISION_PHRASE[precision]}, mais "
                            f"{RELIABILITY_PHRASE[reliability]}.")
    return [
        verdict_sentence,
        PRECISION_QUESTION_ANSWER[precision] + " " + RELIABILITY_QUESTION_ANSWER[reliability],
        _certainty_sentence(effective_n),
    ]


GROUP_SENTENCE_LABEL = {"crypto": "la crypto", "index": "les actions", "bond": "les obligations"}
GROUP_CARD_TITLE = {"crypto": "Crypto (BTC-USD, ETH-USD)", "index": "Actions (SPY)",
                    "bond": "Obligations (ZN=F + TLT)"}


def build_global_answer(aggregate: dict) -> str:
    """Réponse courte de l'en-tête (§2) : regroupe les classes qui partagent le
    même verdict (précision, fiabilité) plutôt que de répéter 3 fois la même
    phrase -- cf. l'exemple du brief ("sur les actions et les obligations, le
    quotidien est meilleur ; sur la crypto, ...")."""
    order = [("crypto", GROUP_SENTENCE_LABEL["crypto"]), ("index", GROUP_SENTENCE_LABEL["index"]),
             ("bond", GROUP_SENTENCE_LABEL["bond"])]
    groups_by_outcome: dict = {}
    for cls, label in order:
        agg = aggregate[cls]
        if agg["status"] != "tested":
            key = ("insufficient", "insufficient")
        else:
            key = (_axis_outcome("tested", agg["skill_sqerror"]["verdict"]),
                   _axis_outcome("tested", agg["skill_winkler"]["verdict"]))
        groups_by_outcome.setdefault(key, []).append(label)

    sentences = []
    for (precision, reliability), labels in groups_by_outcome.items():
        group_label = _join_fr(labels)
        if precision == reliability:
            sentences.append(f"Sur {group_label}, {_decapitalize(CANONICAL_HEADLINE[precision])}.")
        else:
            sentences.append(f"Sur {group_label}, {PRECISION_PHRASE[precision]}, mais "
                             f"{RELIABILITY_PHRASE[reliability]}.")
    return " ".join(sentences)


def build_aggregate_plain(agg: dict, group_key: str) -> dict:
    label = GROUP_SENTENCE_LABEL[group_key]
    if agg["status"] != "tested":
        precision = reliability = "insufficient"
        effective_n = None
    else:
        precision = _axis_outcome("tested", agg["skill_sqerror"]["verdict"])
        reliability = _axis_outcome("tested", agg["skill_winkler"]["verdict"])
        effective_n = agg["skill_sqerror"]["effective_n"]
    return {
        "title": GROUP_CARD_TITLE[group_key],
        "headline": _headline(precision, reliability),
        "one_liner": _one_liner(precision, reliability),
        "gauge": reliability_gauge(effective_n),
        "explanation": build_plain_explanation(label, precision, reliability, effective_n),
    }


def build_cell_plain(row: dict) -> dict:
    """Cellule = un seul axe testé (précision/RMSE) -- la fiabilité de la
    fourchette (Winkler) y reste purement descriptive (pas de p-value à ce
    niveau, cf. build_cell_report), donc l'explication le dit explicitement au
    lieu de prétendre à un test qui n'existe pas à cette granularité."""
    precision = _axis_outcome(row["status"], row.get("verdict"))
    winkler_leaning, _ = _leaning(row["winkler_daily"], row["winkler_weekly"], "lower_better")
    effective_n = row.get("effective_n")
    headline = CANONICAL_HEADLINE[precision]
    subject = f"{row['model']} sur {row['asset']}"
    verdict_sentence = f"Pour {subject}, {_decapitalize(headline)}."
    reliability_clause = {
        "daily": "elle penche plutôt du côté quotidien",
        "weekly": "elle penche plutôt du côté hebdomadaire",
        "tie": "les deux se valent",
    }[winkler_leaning]
    why_sentence = (f"{PRECISION_QUESTION_ANSWER[precision]} Côté fourchette de prix (lecture "
                    f"indicative, pas testée statistiquement à ce niveau détaillé), {reliability_clause}.")
    return {
        "headline": headline,
        "gauge": reliability_gauge(effective_n),
        "explanation": [verdict_sentence, why_sentence, _certainty_sentence(effective_n)],
    }


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
        row["report"] = build_cell_report(row)
        row["plain"] = build_cell_plain(row)
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
    for key, agg in result.items():
        if agg["status"] == "tested":
            agg["report"] = build_aggregate_report(agg)
        if key in ("crypto", "index", "bond"):
            agg["plain"] = build_aggregate_plain(agg, key)
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
        "plain": {
            "question": "Pour prévoir un prix à 1 semaine, vaut-il mieux un modèle hebdomadaire ou quotidien ?",
            "answer": build_global_answer(aggregate),
            "reliability_thresholds": {"low": RELIABILITY_LOW, "mid": RELIABILITY_MID},
        },
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
