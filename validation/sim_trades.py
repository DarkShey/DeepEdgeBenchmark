"""
sim_trades.py — Test cases Bull-Calm (`bull_calm_d1`) et Sideways (`sideways_d1`) à D+1,
cf. BRIEF_bull_calm_d1.md, BRIEF_sideways_d1.md, BRIEF_db_unification.md.

Le brut (une ligne = une prédiction D->D+1, live ou oos) vit dans l'unique table
`predictions` de `validation.tracking_db` (colonne `source` : 'live'/'oos', cf.
BRIEF_db_unification.md). Ce module lit ce brut via la vue `all_predictions` (créée
par `init_db()`, vocabulaire test-case : `d_date`/`reference_price`/`predicted`/
`pi_lower`/`pi_upper`/`realized_price`, filtrée `horizon=1` -- mes test cases sont D+1
uniquement, la table `predictions` contient aussi du live horizon=7 qui ne doit jamais
fuiter ici), applique la règle (`bull_calm_d1` + règle sœur `pi95_conf`, ou
`sideways_d1`) et persiste les signaux valides dans `sim_trades` (résultats calculés,
table distincte du brut -- inchangée par l'unification). Stdlib (sqlite3, json) +
pandas (lecture parquet) uniquement, même contrainte que `tracking_db.py`.

Ingestion OOS : lit `Run/*-D1/` ET `Run/*-D7/predictions.parquet`, insère directement
dans `predictions` (source='oos') -- plus de table séparée, `daily_oos_log` a été
supprimée (BRIEF_db_unification.md §2.3), c'était une redondance pure avec `predictions`.

DEUX chemins d'alignement anti-look-ahead distincts dans build_oos_prediction_rows(),
PAS le même code pour D1 et D7 (BRIEF_comparaison_rigoureuse.md -- bug trouvé et corrigé :
la version précédente appliquait le seul chemin D1 aux deux, silencieusement fausse pour
D7) :
  - **D1** (backtest quotidien dense, un point par jour de validation) : la ligne t-1 EST
    la veille de la ligne t -- `reference_price=actual[t-1]`, `cutoff_date=date[t-1]`.
  - **D7 et tout horizon multi-pas** (Gate2 "rolling origins", cf.
    model_artifacts/pipeline.py `_run_model_d7_rolling` / `MAX_D7_ROLLING_ORIGINS=10`) :
    au plus 10 origines ESPACÉES sur toute la fenêtre de validation (~17-18 jours d'écart
    typique, jamais 1 jour) -- la ligne t-1 est une origine antérieure SANS RAPPORT avec
    la cible de la ligne t, jamais sa veille. `reference_price`/`cutoff_date` sont
    reconstruits depuis `prices.parquet` (série de prix réelle gelée, écrite par
    write_prices_parquet, présente dans le même dossier) : on retrouve la position de
    `target_date` dans cette série et on recule de `HORIZON_TRADING_DAYS[label]` pas de
    bourse (5 pour "D7") -- exactement le calcul que `_run_model_d7_rolling` a fait à la
    génération (`extended_train.iloc[-1]` à `target_idx - h_days`), simplement jamais
    persisté tel quel dans predictions.parquet (seuls date/actual/predicted/pi_lower/
    pi_upper y sont écrits). Dossier sans prices.parquet -> toutes ses lignes sont
    droppées (aucune reconstruction fiable possible), pas de résultat silencieusement faux.

`regime` : forcé à `"unknown"` pour `source="oos"`. `business_validation.json` (quand il
existe) décrit la prévision LIVE la plus récente du combo, pas les ~112 jours historiques
du backtest — le propager sur tout le log serait une fuite d'information non temporelle
mais sémantique (fausse étiquette). Pour `source="live"`, `regime` vient directement de
`predictions.regime` (déjà correct par construction). Groupement KPI par `regime` :
autorisé uniquement pour `source="live"` (voir `kpi_report`).

Signaux valides mais non résolus (`status="open"`, live uniquement) : exclus de TOUS les
KPIs tant qu'ils ne sont pas résolus par `resolve_open_sim_trades` / `sync_live_trades`
(§6.5) — y compris du compte `n_signaux`, qui ne reflète donc que l'activité déjà
résolue de la règle (le nombre de signaux ouverts est reporté séparément, `n_open`).
Résolution live directe depuis `predictions` (plus de mirroring à rafraîchir : dès que
`tracking_db.evaluate_pending` écrit `y_true`, `all_predictions.realized_price` le
reflète immédiatement).
"""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from validation import tracking_db as td

DEFAULT_DB_PATH = "tracking.db"

VALID_SOURCES = {"oos", "live"}
_GROUP_BY_COLUMNS = {"asset", "model", "regime"}
_HORIZON_LABEL_TO_INT = {"D1": 1, "D7": 7}
# Pas de bourse entre l'origine et la cible pour un horizon Gate2 -- DISTINCT de
# _HORIZON_LABEL_TO_INT ci-dessus (qui n'est que l'étiquette business "D+7"/horizon=7).
# Source : model_artifacts/pipeline.py HORIZON_TRADING_DAYS.
_HORIZON_LABEL_TO_TRADING_STEPS = {"D1": 1, "D7": 5}


def _connect(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _drop_legacy_daily_oos_log(conn) -> None:
    """Supprime `daily_oos_log` si elle traîne encore (créée par une base alimentée
    avant BRIEF_db_unification.md, plus jamais recréée depuis -- ce module ne
    l'écrit plus). Garde-fou avant suppression : chaque ligne doit déjà avoir un
    équivalent exact dans `predictions` (même run_id/model/asset/horizon/cutoff_date) ;
    sinon on lève plutôt que de perdre des données silencieusement."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_oos_log'")}
    if not tables:
        return

    orphans = conn.execute("""
        SELECT COUNT(*) FROM daily_oos_log d
        WHERE NOT EXISTS (
            SELECT 1 FROM predictions p
            WHERE p.run_id = d.run_id AND p.model = d.model AND p.asset = d.asset
              AND p.horizon = d.horizon AND p.cutoff_date = d.d_date
        )
    """).fetchone()[0]
    if orphans:
        raise RuntimeError(
            f"daily_oos_log contient {orphans} ligne(s) sans équivalent dans predictions -- "
            "suppression bloquée (perte de données potentielle, à vérifier manuellement)."
        )
    conn.execute("DROP TABLE daily_oos_log")


def init_db(db_path=DEFAULT_DB_PATH) -> None:
    """Assure `predictions` (brut, table+migration possédées par tracking_db.py), crée
    `sim_trades` si absente (résultats calculés, CREATE TABLE IF NOT EXISTS, idempotent
    -- schéma §7 de BRIEF_bull_calm_d1.md, `in_band` ajoutée par BRIEF_sideways_d1.md
    §10) et (re)crée la vue `all_predictions` (BRIEF_db_unification.md §2.4) : lecture
    du brut au vocabulaire test-case, filtrée horizon=1 ET horizon_type='daily' (mes
    test cases sont D+1 uniquement ; `predictions` contient aussi du live horizon=7 ET,
    depuis BRIEF_audit_combinaisons.md, du weekly horizon=1 qui signifie "W+1" -- même
    valeur numérique que "D+1" mais une prédiction complètement différente. Sans le
    filtre horizon_type, un W+1 se glisserait dans all_predictions comme s'il s'agissait
    d'un vrai D+1 et pourrait finir dans sim_trades / les règles de trading, qui n'ont
    jamais été pensées pour un horizon hebdomadaire) et `daily_duplicate = 0` (BRIEF_correction_sim_trades.md
    §3 : ignore les copies OOS flaguées par tracking_db.flag_daily_duplicates -- le live
    reste entièrement visible, ses lignes sont toutes à 0 par construction). Vue recréée
    à chaque appel (DROP+CREATE, pas IF NOT EXISTS) pour ne jamais rester sur une
    définition périmée -- coût nul, une vue ne porte aucune donnée."""
    td.init_db(db_path)
    conn = _connect(db_path)
    try:
        _drop_legacy_daily_oos_log(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sim_trades (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_version      TEXT NOT NULL,
                run_id            TEXT NOT NULL,
                model             TEXT NOT NULL,
                asset             TEXT NOT NULL,
                horizon           INTEGER NOT NULL,
                regime            TEXT NOT NULL,
                source            TEXT NOT NULL,
                d_date            TEXT NOT NULL,
                target_date       TEXT NOT NULL,
                reference_price   REAL NOT NULL,
                predicted         REAL NOT NULL,
                pi_lower          REAL NOT NULL,
                pi_upper          REAL NOT NULL,
                realized_price    REAL,
                signal_valid      INTEGER NOT NULL,
                direction_ok      INTEGER,
                branch            INTEGER,
                counter           INTEGER,
                roi               REAL,
                in_band           INTEGER,
                degenerate_pi     INTEGER NOT NULL,
                status            TEXT NOT NULL,
                created_at        TEXT NOT NULL,
                evaluated_at      TEXT,
                UNIQUE (rule_version, source, run_id, model, asset, horizon, d_date)
            )
        """)
        # Migration légère (BRIEF_sideways_d1.md §10) : une sim_trades créée avant
        # l'ajout de in_band (Bull-Calm seul) n'a pas la colonne -- l'ajouter si absente,
        # jamais de perte de données puisqu'aucune ligne existante n'a besoin de backfill
        # (roi/in_band restent NULL par défaut pour les lignes déjà en base).
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sim_trades)")}
        if "in_band" not in existing_cols:
            conn.execute("ALTER TABLE sim_trades ADD COLUMN in_band INTEGER")
        # Migration lazy (BRIEF_sideways_v2.md §6) : vol_bucket/stress_score/gated_out,
        # propres à sideways_gated_d1 -- NULL par défaut pour toute ligne préexistante
        # (bull_calm_d1/pi95_conf/bear_*/sideways_d1 ne les renseignent jamais).
        if "vol_bucket" not in existing_cols:
            conn.execute("ALTER TABLE sim_trades ADD COLUMN vol_bucket INTEGER")
        if "stress_score" not in existing_cols:
            conn.execute("ALTER TABLE sim_trades ADD COLUMN stress_score REAL")
        if "gated_out" not in existing_cols:
            conn.execute("ALTER TABLE sim_trades ADD COLUMN gated_out INTEGER")

        conn.execute("DROP VIEW IF EXISTS all_predictions")
        conn.execute("""
            CREATE VIEW all_predictions AS
            SELECT
                id, run_id, model, asset, horizon, regime,
                cutoff_date AS d_date, target_date,
                last_close  AS reference_price, y_pred AS predicted,
                y_lower     AS pi_lower,        y_upper AS pi_upper,
                y_true      AS realized_price,  source
            FROM predictions
            WHERE horizon = 1
              AND horizon_type = 'daily'
              AND daily_duplicate = 0
        """)
        conn.commit()
    finally:
        conn.close()


# ── Règle bull_calm_d1 (§3, §4, §8) et règle sœur pi95_conf (§3) ────────────────

def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _resolve_branches(ref: float, pi_low: float, pi_high: float, realized: float):
    """Branches §4, évaluées dans l'ordre, exhaustives et sans recouvrement (§6.4).
    Retourne (branch, counter, exit_px)."""
    if realized > pi_high:                     # 1
        return 1, 2, pi_high
    if realized > ref:                          # 2
        return 2, 1, realized
    if realized >= pi_low:                       # 3 (PI_low <= realized <= ref)
        return 3, -1, realized
    return 4, -2, realized                       # 4 (v1 : close réalisé, PAS pi_low, cf. §11.1)


def bull_calm_d1(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0):
    """TC1.1 Bull-Calm — cf. §8 du brief. Retourne (signal_valid, branch, counter, roi, degenerate_pi)."""
    degenerate_pi = int(pi_high <= ref)

    # Garde-fou d'étanchéité (taxonomie TC1.1-TC1.5) : bull calm exclut les jours
    # ref < pi_low, qui relèvent de TC1.2 (bull stress). Sans `ref >= pi_low`, TC1.1
    # et TC1.2 compteraient deux fois les mêmes journées.
    signal_valid = (predicted > ref) and (ref >= pi_low)
    if not signal_valid:
        return False, None, 0, 0.0, degenerate_pi

    if realized is None:
        return True, None, None, None, degenerate_pi

    branch, counter, exit_px = _resolve_branches(ref, pi_low, pi_high, realized)
    roi = (exit_px - ref) / ref - fee_bps / 1e4
    return True, branch, counter, roi, degenerate_pi


def pi95_conf(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0):
    """TC1.2 Bull-Stress — règle sœur (§3) : signal plus strict (hausse quasi certaine
    même au pire bas de l'IC) -- `pi_low > ref` au lieu de `predicted > ref`. Résolution
    identique (mêmes branches §4)."""
    degenerate_pi = int(pi_high <= ref)

    signal_valid = pi_low > ref
    if not signal_valid:
        return False, None, 0, 0.0, degenerate_pi

    if realized is None:
        return True, None, None, None, degenerate_pi

    branch, counter, exit_px = _resolve_branches(ref, pi_low, pi_high, realized)
    roi = (exit_px - ref) / ref - fee_bps / 1e4
    return True, branch, counter, roi, degenerate_pi


def _resolve_branches_bear(ref: float, pi_low: float, pi_high: float, realized: float):
    """Miroir de `_resolve_branches` pour une position courte (short) : profit quand le
    prix baisse. Branches §3bis de BRIEF_bull_calm_d1.md, évaluées dans l'ordre,
    exhaustives et sans recouvrement. Retourne (branch, counter, exit_px)."""
    if realized < pi_low:                        # 1
        return 1, 2, pi_low
    if realized < ref:                            # 2
        return 2, 1, realized
    if realized <= pi_high:                        # 3 (ref <= realized <= PI_high)
        return 3, -1, realized
    return 4, -2, realized                          # 4 (v1 : close réalisé, PAS pi_high)


def bear_calm_d1(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0):
    """TC1.3 Bear-Calm — miroir de bull_calm_d1 (§3bis du brief) : position courte légère.
    Retourne (signal_valid, branch, counter, roi, degenerate_pi)."""
    # Miroir de bull_calm_d1 (degenerate_pi = pi_high<=ref, la borne "côté profit" pour un
    # long) : pour un short, la borne "côté profit" est PI_low -- dégénéré si elle
    # n'est même pas sous ref (incohérent avec une prévision de baisse).
    degenerate_pi = int(pi_low >= ref)

    # Garde-fou d'étanchéité (miroir de bull_calm_d1) : bear calm exclut les jours
    # ref > pi_high, qui relèvent de TC1.4 (bear stress). Sans `ref <= pi_high`, TC1.3
    # et TC1.4 compteraient deux fois les mêmes journées.
    signal_valid = (predicted < ref) and (ref <= pi_high)
    if not signal_valid:
        return False, None, 0, 0.0, degenerate_pi

    if realized is None:
        return True, None, None, None, degenerate_pi

    branch, counter, exit_px = _resolve_branches_bear(ref, pi_low, pi_high, realized)
    roi = (ref - exit_px) / ref - fee_bps / 1e4
    return True, branch, counter, roi, degenerate_pi


def bear_stress_d1(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0):
    """TC1.4 Bear-Stress — miroir de pi95_conf (§3bis du brief) : signal plus strict
    (baisse quasi certaine même au meilleur haut de l'IC) -- `pi_high < ref` au lieu de
    `predicted < ref`. Résolution identique (mêmes branches miroir)."""
    degenerate_pi = int(pi_low >= ref)

    signal_valid = pi_high < ref
    if not signal_valid:
        return False, None, 0, 0.0, degenerate_pi

    if realized is None:
        return True, None, None, None, degenerate_pi

    branch, counter, exit_px = _resolve_branches_bear(ref, pi_low, pi_high, realized)
    roi = (ref - exit_px) / ref - fee_bps / 1e4
    return True, branch, counter, roi, degenerate_pi


def sideways_d1(ref, predicted, pi_low, pi_high, realized, k=0.10, m_frac=0.25, h_frac=0.50):
    """TC1.5 Sideways — test de justesse d'une journée plate (BRIEF_sideways_d1.md §7,
    pseudo-code exact). Retourne (signal_sideways, branch, counter, roi, in_band,
    degenerate_pi). roi est TOUJOURS None (pas de position directionnelle) ; realized
    peut être None (live)."""
    W = pi_high - pi_low
    degenerate_pi = int(W <= 0)

    # --- Signal à D : P(D) dans la bande ET mouvement prédit négligeable ---
    eps = k * W
    signal = (pi_low <= ref <= pi_high) and (abs(predicted - ref) <= eps)
    if not signal:
        return False, None, 0, None, None, degenerate_pi

    if realized is None:
        return True, None, None, None, None, degenerate_pi   # jour plat, non résolu

    # --- Résolution à D+1 : counter symétrique, pas de ROI ---
    in_band = int(pi_low <= realized <= pi_high)
    m, h = m_frac * W, h_frac * W
    if in_band and abs(realized - ref) <= m:       # 1 : quasi immobile
        branch, counter = 1, 2
    elif in_band:                                    # 2 : resté dans la bande
        branch, counter = 2, 1
    else:
        dist = (pi_low - realized) if realized < pi_low else (realized - pi_high)
        if dist <= h:                                # 3 : petit breakout
            branch, counter = 3, -1
        else:                                          # 4 : gros breakout
            branch, counter = 4, -2

    return True, branch, counter, None, in_band, degenerate_pi


def sideways_gated_d1(ref, predicted, pi_low, pi_high, realized,
                      vol_bucket=None, stress_score=None, source="oos",
                      k=0.10, vb_max=1, stress_max=0.30, m_frac=0.25, h_frac=0.50):
    """TC1.5b Sideways gaté (BRIEF_sideways_v2.md §7) — extension de sideways_d1 : signal
    plat validé par le régime/volatilité + P&L short-vol borné. sideways_d1 reste
    strictement inchangée (§0 du brief, baseline). Retourne (signal_sideways, branch,
    counter, pnl_shortvol, in_band, gated_out, degenerate_pi). counter = justesse v1,
    identique à sideways_d1 à signal validé (le gate ne change jamais la résolution).
    pnl_shortvol est un PROXY d'évaluation short-straddle dans [-1, +1] (jamais un
    rendement exécuté -- ne jamais le libeller roi/rendement dans le reporting), None si
    non résolu, non tradable, ou PI dégénéré (§5.3, évite la division par W=0).
    direction_ok est géré à NULL par l'adaptateur, pas ici (non directionnel).
    realized/stress_score/vol_bucket peuvent être None."""
    W = pi_high - pi_low
    degenerate_pi = int(W <= 0)

    # --- Signal v1 (identique à sideways_d1, §7 de BRIEF_sideways_d1.md) ---
    eps = k * W
    signal_v1 = (pi_low <= ref <= pi_high) and (abs(predicted - ref) <= eps)
    if not signal_v1:
        return False, None, 0, None, None, 0, degenerate_pi

    # --- Gate régime/volatilité (§2) : vol_bucket/stress_score None -> permissif
    # (n_gate_undefined côté KPI, §5.1) ; stress ignoré hors live (§5.2) ---
    gate_vol = (vol_bucket is None) or (vol_bucket <= vb_max)
    gate_stress = (source != "live") or (stress_score is None) or (stress_score <= stress_max)
    if not (gate_vol and gate_stress):
        return False, None, 0, None, None, 1, degenerate_pi   # flat suspect, journalisé (gated_out=1)

    if realized is None:
        return True, None, None, None, None, 0, degenerate_pi   # live non résolu

    # --- Résolution : counter v1 (inchangé) + pnl short-vol (§3) ---
    in_band = int(pi_low <= realized <= pi_high)
    m, h = m_frac * W, h_frac * W
    move = abs(realized - ref)
    if in_band and move <= m:
        branch, counter = 1, 2
    elif in_band:
        branch, counter = 2, 1
    else:
        dist = (pi_low - realized) if realized < pi_low else (realized - pi_high)
        branch, counter = (3, -1) if dist <= h else (4, -2)

    # W<=0 (dégénéré) diviserait par zéro : pnl_shortvol non défini plutôt qu'un crash
    # (§5.3) -- la ligne reste exclue des KPI par défaut via degenerate_pi, comme v1.
    pnl_shortvol = None if degenerate_pi else max(1.0 - move / (W / 2.0), -1.0)
    return True, branch, counter, pnl_shortvol, in_band, 0, degenerate_pi


# ── Dispatch rule_version -> fonction, normalisé pour generate_sim_trades/resolve_open_
# sim_trades : (signal_valid, branch, counter, roi, direction_ok, in_band, degenerate_pi).
# bull_calm_d1/pi95_conf ne sont PAS modifiées : ces adaptateurs les enveloppent sans
# toucher à leur code ni à leur signature (appelables directement tels quels, cf. tests).

def _adapt_bull_like(fn):
    def call(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0, **_ignored):
        signal_valid, branch, counter, roi, degenerate = fn(
            ref, predicted, pi_low, pi_high, realized, fee_bps=fee_bps)
        direction_ok = int(realized > ref) if (signal_valid and realized is not None) else None
        return signal_valid, branch, counter, roi, direction_ok, None, degenerate
    return call


def _adapt_bear_like(fn):
    """Miroir de _adapt_bull_like : direction_ok = le prix a bien baissé sous ref
    (thèse courte confirmée), pas au-dessus."""
    def call(ref, predicted, pi_low, pi_high, realized, fee_bps=0.0, **_ignored):
        signal_valid, branch, counter, roi, degenerate = fn(
            ref, predicted, pi_low, pi_high, realized, fee_bps=fee_bps)
        direction_ok = int(realized < ref) if (signal_valid and realized is not None) else None
        return signal_valid, branch, counter, roi, direction_ok, None, degenerate
    return call


def _adapt_sideways(fn):
    def call(ref, predicted, pi_low, pi_high, realized, k=0.10, m_frac=0.25, h_frac=0.50, **_ignored):
        signal_valid, branch, counter, roi, in_band, degenerate = fn(
            ref, predicted, pi_low, pi_high, realized, k=k, m_frac=m_frac, h_frac=h_frac)
        return signal_valid, branch, counter, roi, None, in_band, degenerate
    return call


def _adapt_sideways_gated(fn):
    """Adaptateur sideways_gated_d1 -> (signal_valid, branch, counter, roi, direction_ok,
    in_band, degenerate_pi, gated_out) : roi <- pnl_shortvol (proxy short-straddle
    d'évaluation, JAMAIS un rendement exécuté), direction_ok toujours None (non
    directionnel, comme sideways_d1). 8e valeur (gated_out) propre à cette règle -- les
    autres RULES restent à 7 valeurs (comportement/signature inchangés, §0 du brief) ;
    le dispatch générique (generate_sim_trades/resolve_open_sim_trades) gère les deux
    arités sans jamais rompre les appels directs existants (ex. RULES["bear_calm_d1"])."""
    def call(ref, predicted, pi_low, pi_high, realized, k=0.10, vb_max=1, stress_max=0.30,
             m_frac=0.25, h_frac=0.50, vol_bucket=None, stress_score=None, source="oos",
             **_ignored):
        signal_valid, branch, counter, roi, in_band, gated_out, degenerate = fn(
            ref, predicted, pi_low, pi_high, realized, vol_bucket=vol_bucket,
            stress_score=stress_score, source=source, k=k, vb_max=vb_max,
            stress_max=stress_max, m_frac=m_frac, h_frac=h_frac)
        return signal_valid, branch, counter, roi, None, in_band, degenerate, gated_out
    return call


RULES = {
    "bull_calm_d1": _adapt_bull_like(bull_calm_d1),      # TC1.1 Bull-Calm
    "pi95_conf": _adapt_bull_like(pi95_conf),             # TC1.2 Bull-Stress
    "bear_calm_d1": _adapt_bear_like(bear_calm_d1),       # TC1.3 Bear-Calm
    "bear_stress_d1": _adapt_bear_like(bear_stress_d1),   # TC1.4 Bear-Stress
    "sideways_d1": _adapt_sideways(sideways_d1),          # TC1.5 Sideways
    "sideways_gated_d1": _adapt_sideways_gated(sideways_gated_d1),   # TC1.5b Sideways gaté
}


# ── Proxy vol_bucket OOS (BRIEF_sideways_v2.md §1, §5.1, §5.5) ──────────────────

def _vol_bucket_terciles(widths):
    """Terciles PAR RANG au sein d'un groupe déjà filtré asset×model (widths = W =
    pi_high - pi_low, la vol anticipée par le modèle -- connue à D, aucun look-ahead).
    None partout si le groupe a moins de 3 lignes (terciles non fiables, §5.1) -- compte
    dans n_gate_undefined côté KPI, jamais maquillé en signal validé."""
    n = len(widths)
    if n < 3:
        return [None] * n
    order = sorted(range(n), key=lambda i: widths[i])
    buckets = [None] * n
    for rank, idx in enumerate(order):
        buckets[idx] = min(2, rank * 3 // n)
    return buckets


def _vol_bucket_proxy_for_rows(rows):
    """Applique _vol_bucket_terciles par groupe (asset, model) sur des lignes
    all_predictions-like (n'a besoin que de id/asset/model/pi_lower/pi_upper -- jamais
    realized_price, aucun look-ahead possible par construction). Retourne {id: bucket}."""
    groups = {}
    for row in rows:
        groups.setdefault((row["asset"], row["model"]), []).append(row)
    result = {}
    for group_rows in groups.values():
        widths = [row["pi_upper"] - row["pi_lower"] for row in group_rows]
        for row, bucket in zip(group_rows, _vol_bucket_terciles(widths)):
            result[row["id"]] = bucket
    return result


# ── Ingestion OOS -> predictions (source="oos"), depuis Run/*-D1/predictions.parquet ─

def _date_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_oos_prediction_rows(run_dir, source="oos"):
    """Lit predictions.parquet + metrics.json d'un dossier Run/<...>-D1/ ou -D7/,
    applique l'alignement anti-look-ahead adapté à la structure de CE backtest (cf.
    docstring du module -- D1 dense vs D7/rolling-origin, deux chemins distincts,
    PAS le même code). Exclut les lignes NaN / bornes cassées (pi_lower > pi_upper).
    Retourne des lignes au format de la table `predictions` unifiée
    (BRIEF_db_unification.md §2.2) : `cutoff_date`/`last_close`/`y_pred`/`y_lower`/
    `y_upper`/`y_true`, `source='oos'`. Les colonnes métier live (tc_id, verdict_*,
    created_at, ...) sont absentes de ces dicts -> NULL par schéma une fois
    insérées (jamais renseignées pour l'OOS). Retourne (rows, n_dropped)."""
    run_dir = Path(run_dir)
    df = pd.read_parquet(run_dir / "predictions.parquet").sort_values("date").reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = metrics["model"]
    asset = metrics["asset"]
    horizon_label = metrics.get("horizon")
    horizon = _HORIZON_LABEL_TO_INT.get(horizon_label, 1)
    steps = _HORIZON_LABEL_TO_TRADING_STEPS.get(horizon_label, 1)

    run_id = run_dir.name
    rows = []
    n_dropped = 0

    if steps <= 1:
        # D1 : backtest quotidien dense -- t-1 EST la veille de t.
        for t in range(1, len(df)):
            reference_price = df.loc[t - 1, "actual"]
            predicted = df.loc[t, "predicted"]
            pi_lower = df.loc[t, "pi_lower"]
            pi_upper = df.loc[t, "pi_upper"]
            realized_price = df.loc[t, "actual"]

            values = (reference_price, predicted, pi_lower, pi_upper, realized_price)
            if not all(_is_finite(v) for v in values) or pi_lower > pi_upper:
                n_dropped += 1
                continue

            rows.append({
                "run_id": run_id, "model": model, "asset": asset, "horizon": horizon,
                "regime": "unknown",   # jamais business_validation.json ici, cf. docstring module
                "cutoff_date": _date_str(df.loc[t - 1, "date"]),
                "target_date": _date_str(df.loc[t, "date"]),
                "last_close": float(reference_price), "y_pred": float(predicted),
                "y_lower": float(pi_lower), "y_upper": float(pi_upper),
                "y_true": float(realized_price),
                "source": source,
            })
        return rows, n_dropped

    # D7 (et tout horizon Gate2 multi-pas) : origines glissantes ESPACÉES -- t-1 n'a
    # aucun rapport avec la cible de t. cutoff_date/last_close reconstruits depuis
    # prices.parquet (série de prix réelle gelée par le pipeline), en reculant de
    # `steps` pas de bourse depuis target_date -- exactement le calcul fait à la
    # génération (cf. docstring du module), jamais persisté tel quel ailleurs.
    prices_path = run_dir / "prices.parquet"
    if not prices_path.exists():
        return [], len(df)   # aucune reconstruction fiable possible -- tout dropper, rien de faux

    prices = pd.read_parquet(prices_path).sort_values("date").reset_index(drop=True)
    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(prices["date"])}

    for t in range(len(df)):
        target_ts = pd.Timestamp(df.loc[t, "date"])
        target_pos = date_to_pos.get(target_ts)
        if target_pos is None or target_pos - steps < 0:
            n_dropped += 1
            continue
        cutoff_pos = target_pos - steps

        reference_price = prices.loc[cutoff_pos, "close"]
        predicted = df.loc[t, "predicted"]
        pi_lower = df.loc[t, "pi_lower"]
        pi_upper = df.loc[t, "pi_upper"]
        realized_price = df.loc[t, "actual"]

        values = (reference_price, predicted, pi_lower, pi_upper, realized_price)
        if not all(_is_finite(v) for v in values) or pi_lower > pi_upper:
            n_dropped += 1
            continue

        rows.append({
            "run_id": run_id, "model": model, "asset": asset, "horizon": horizon,
            "regime": "unknown",
            "cutoff_date": _date_str(prices.loc[cutoff_pos, "date"]),
            "target_date": _date_str(df.loc[t, "date"]),
            "last_close": float(reference_price), "y_pred": float(predicted),
            "y_lower": float(pi_lower), "y_upper": float(pi_upper),
            "y_true": float(realized_price),
            "source": source,
        })

    return rows, n_dropped


def insert_oos_predictions(rows, db_path=DEFAULT_DB_PATH) -> int:
    """Upsert « garde le dernier run » dans `predictions` (source='oos') --
    BRIEF_prevention_doublons.md §5. La clé métier OOS est désormais unique par
    construction (`idx_predictions_oos_unique`, `(source, model, asset, horizon,
    cutoff_date)`, SANS `run_id` depuis ce brief) : un rejeu sur la même date ne
    s'empile plus (ancien comportement `INSERT OR IGNORE`, qui gardait le *premier*
    run et empilait les suivants -- cause racine des doublons OOS, cf.
    BRIEF_audit_tracking_db.md), il REMPLACE la ligne existante (`run_id`,
    `target_date`, valeurs) par celles du nouvel insert.

    `run_id` n'est plus qu'une métadonnée de provenance, pas une composante de clé.
    L'ordre d'appel compte : `ingest_oos` parcourt les dossiers `Run/*-D1` par ordre
    lexicographique (= chronologique, préfixe `YYYYMMDD` du nom de dossier), donc le
    dernier insert pour une date donnée est bien le run le plus récent -- « garde le
    dernier » n'est correct que si l'appelant ingère dans cet ordre.

    Retourne le nombre de lignes affectées (insérées OU mises à jour).

    `frequence`/`horizon_type`/`horizon_unit` (BRIEF_audit_combinaisons.md) : absents
    des rows produites par build_oos_prediction_rows() (daily uniquement, pas modifiée
    -- 100% des appelants actuels via ingest_oos() restent daily natif), donc défaultés
    ici à 'daily'/'daily'/'D+{horizon}' quand absents. Un futur appelant weekly les
    passe explicitement dans chaque row et ils sont respectés tels quels. Ajoutés à la
    clé ON CONFLICT (cf. idx_predictions_oos_unique dans tracking_db.py) : sans ça, une
    prédiction weekly et une prédiction daily sur le même (model, asset, horizon,
    cutoff_date) s'écraseraient l'une l'autre au lieu de coexister."""
    if not rows:
        return 0
    td.init_db(db_path)
    cols = ("run_id", "model", "asset", "horizon", "regime", "cutoff_date", "target_date",
            "last_close", "y_pred", "y_lower", "y_upper", "y_true", "source",
            "frequence", "horizon_type", "horizon_unit")
    placeholders = ", ".join(f":{c}" for c in cols)
    columns = ", ".join(cols)
    conn = _connect(db_path)
    try:
        n = 0
        for row in rows:
            horizon_type = row.get("horizon_type", "daily")
            row = {
                **row,
                "frequence": row.get("frequence", "daily"),
                "horizon_type": horizon_type,
                "horizon_unit": row.get("horizon_unit") or
                    f"{'W' if horizon_type == 'weekly' else 'D'}+{row['horizon']}",
            }
            cur = conn.execute(f"""
                INSERT INTO predictions ({columns}) VALUES ({placeholders})
                ON CONFLICT (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
                WHERE source='oos'
                DO UPDATE SET
                    run_id      = excluded.run_id,
                    target_date = excluded.target_date,
                    last_close  = excluded.last_close,
                    y_pred      = excluded.y_pred,
                    y_lower     = excluded.y_lower,
                    y_upper     = excluded.y_upper,
                    y_true      = excluded.y_true,
                    regime      = excluded.regime,
                    horizon_unit = excluded.horizon_unit
            """, row)
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def ingest_oos(run_root="Run", db_path=DEFAULT_DB_PATH) -> dict:
    """Parcourt tous les Run/*-D1/ ET Run/*-D7/ par ordre lexicographique (skip
    silencieusement si predictions.parquet ou metrics.json manque), construit et
    insère (upsert « garde le dernier », cf. insert_oos_predictions) dans
    `predictions` (source='oos'). L'ordre lexicographique = chronologique (préfixe
    `YYYYMMDD` du nom de dossier, BRIEF_prevention_doublons.md §5) : essentiel pour
    que le dernier combo traité pour une date donnée soit bien le run le plus récent
    -- D1 et D7 ne peuvent jamais se chevaucher sur la clé métier (horizon différent),
    donc les trier/traiter ensemble ou séparément est équivalent. Idempotent.

    D7 (BRIEF_audit_combinaisons.md) : 286 dossiers Run/*-D7/ avec un
    predictions.parquet complet existaient déjà sur disque, jamais ingérés
    auparavant (glob D1 seul, oubli historique) -- d'où la case regime A / D+7
    entièrement vide dans la matrice de couverture malgré le calcul déjà fait.
    build_oos_prediction_rows lit `metrics.json["horizon"]="D7"` -> horizon=7 sans
    changement de code ; frequence/horizon_type/horizon_unit prennent leurs défauts
    daily/daily/D+7 dans insert_oos_predictions (régime A, comme D+1).

    Reconstruit ensuite `sim_trades` (source='oos') depuis zéro (cf.
    rebuild_oos_sim_trades, §6 du brief) : l'upsert peut changer `run_id`/`y_pred`
    d'une ligne déjà en base, et `sim_trades` est indexé sur `run_id` -- plus sûr de
    tout regénérer (l'OOS est déterministe et entièrement résolu) que d'essayer de
    rapiécer les anciens signaux. `all_predictions` (lue par generate_sim_trades)
    filtre horizon=1 AND horizon_type='daily' -- le D+7 n'y apparaît jamais (attendu,
    les règles de trading sont D+1 uniquement), pas de risque de fuite ici."""
    td.init_db(db_path)
    inserted = dropped = combos = 0
    run_dirs = sorted(Path(run_root).glob("*-D1")) + sorted(Path(run_root).glob("*-D7"))
    for run_dir in run_dirs:
        if not (run_dir / "predictions.parquet").exists() or not (run_dir / "metrics.json").exists():
            continue
        rows, n_dropped = build_oos_prediction_rows(run_dir)
        inserted += insert_oos_predictions(rows, db_path=db_path)
        dropped += n_dropped
        combos += 1
    rebuild_stats = rebuild_oos_sim_trades(db_path=db_path)
    return {"combos": combos, "inserted": inserted, "dropped": dropped, **rebuild_stats}


# ── sim_trades : génération + résolution ────────────────────────────────────────

def generate_sim_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1",
                        fee_bps=0.0, source=None, regime_lookup=None, **rule_kwargs) -> int:
    """Applique la règle à chaque ligne `all_predictions` (brut unifié, horizon=1,
    cf. BRIEF_db_unification.md) n'ayant pas encore de sim_trade pour cette
    rule_version. Les flats (signal_valid=False) ne génèrent pas de ligne (§7.2 de
    BRIEF_bull_calm_d1.md) -- SAUF pour rule_version="sideways_gated_d1" quand
    gated_out=1 (« flat suspect » : signal_v1=True mais gate_regime=False, journalisé
    signal_valid=0/gated_out=1 pour le KPI "valeur ajoutée du gate", cf.
    BRIEF_sideways_v2.md §2/§6 -- jamais status='open', la décision est prise au signal,
    indépendamment de `realized`). Idempotent (UNIQUE + LEFT JOIN ... IS NULL).
    `**rule_kwargs` passe les paramètres spécifiques à la règle (k/m_frac/h_frac pour
    sideways_d1, k/vb_max/stress_max/m_frac/h_frac pour sideways_gated_d1) ; les
    adaptateurs de RULES ignorent ceux qui ne les concernent pas.

    Pour rule_version="sideways_gated_d1" uniquement : vol_bucket/stress_score sont
    calculés PAR LIGNE (jamais un rule_kwarg constant) et injectés avant l'appel de la
    règle -- source='oos' : proxy par terciles de W (asset×model, sans look-ahead,
    _vol_bucket_proxy_for_rows, calculé sur TOUT le groupe OOS, pas seulement les
    lignes "nouvelles", pour rester stable entre deux appels incrémentaux) ;
    source='live' : `regime_lookup(asset, d_date) -> (vol_bucket, stress_score)` si
    fourni (lecture du RegimeState de calibration/regime/, câblage laissé à l'appelant
    production -- ex. evaluate_daily.py -- pour ne dépendre d'aucun accès réseau ici),
    sinon None/None (gate dégénère en pass, transparent, §5.1).

    Retourne le nombre de lignes insérées (tradables + flats suspects journalisés)."""
    init_db(db_path)
    rule_fn = RULES[rule_version]
    conn = _connect(db_path)
    try:
        query = """
            SELECT l.* FROM all_predictions l
            LEFT JOIN sim_trades s
              ON s.rule_version = ? AND s.source = l.source AND s.run_id = l.run_id
             AND s.model = l.model AND s.asset = l.asset AND s.horizon = l.horizon
             AND s.d_date = l.d_date
            WHERE s.id IS NULL
        """
        params = [rule_version]
        if source is not None:
            query += " AND l.source = ?"
            params.append(source)
        rows = conn.execute(query, params).fetchall()

        vol_bucket_map = {}
        if rule_version == "sideways_gated_d1":
            oos_rows = conn.execute("SELECT * FROM all_predictions WHERE source='oos'").fetchall()
            vol_bucket_map = _vol_bucket_proxy_for_rows(oos_rows)

        now = datetime.now(timezone.utc).isoformat()
        n_inserted = 0
        for row in rows:
            realized = row["realized_price"]
            row_kwargs = dict(rule_kwargs)
            if rule_version == "sideways_gated_d1":
                row_kwargs["source"] = row["source"]
                if row["source"] == "oos":
                    row_kwargs["vol_bucket"] = vol_bucket_map.get(row["id"])
                    row_kwargs["stress_score"] = None   # §5.2 : jamais de stress fiable en OOS
                else:
                    vol_bucket = stress_score = None
                    if regime_lookup is not None:
                        vol_bucket, stress_score = regime_lookup(row["asset"], row["d_date"])
                    row_kwargs["vol_bucket"] = vol_bucket
                    row_kwargs["stress_score"] = stress_score

            result = rule_fn(row["reference_price"], row["predicted"], row["pi_lower"], row["pi_upper"],
                             realized, fee_bps=fee_bps, **row_kwargs)
            signal_valid, branch, counter, roi, direction_ok, in_band, degenerate = result[:7]
            gated_out = result[7] if len(result) > 7 else None
            if not signal_valid and not gated_out:
                continue

            if gated_out:
                status, evaluated_at = "closed", now   # flat suspect : décidé au signal, jamais "open"
            else:
                status = "closed" if branch is not None else "open"
                evaluated_at = now if status == "closed" else None

            cur = conn.execute("""
                INSERT OR IGNORE INTO sim_trades (
                    rule_version, run_id, model, asset, horizon, regime, source,
                    d_date, target_date, reference_price, predicted, pi_lower, pi_upper,
                    realized_price, signal_valid, direction_ok, branch, counter, roi, in_band,
                    degenerate_pi, status, created_at, evaluated_at,
                    vol_bucket, stress_score, gated_out
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (rule_version, row["run_id"], row["model"], row["asset"], row["horizon"],
                  row["regime"], row["source"], row["d_date"], row["target_date"],
                  row["reference_price"], row["predicted"], row["pi_lower"], row["pi_upper"],
                  realized, int(signal_valid), direction_ok, branch, counter, roi, in_band,
                  degenerate, status, now, evaluated_at,
                  row_kwargs.get("vol_bucket"), row_kwargs.get("stress_score"), gated_out))
            n_inserted += cur.rowcount
        conn.commit()
        return n_inserted
    finally:
        conn.close()


def resolve_open_sim_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1",
                            fee_bps=0.0, **rule_kwargs) -> int:
    """Réévalue les sim_trades status='open' dont le realized_price est maintenant
    connu dans `all_predictions` -- pour source='live', dès que
    `tracking_db.evaluate_pending` écrit `y_true`, la vue le reflète immédiatement
    (plus de mirroring séparé à rafraîchir, cf. BRIEF_db_unification.md). Idempotent :
    ne touche jamais un trade déjà 'closed' (les flats suspects sideways_gated_d1,
    gated_out=1, sont toujours 'closed' dès la génération -- jamais candidats ici).

    Pour rule_version="sideways_gated_d1" : vol_bucket/stress_score/source sont relus
    DEPUIS le trade déjà stocké (figés à la génération, §1 du brief -- jamais
    recalculés à la résolution) et réinjectés tels quels ; le gate est donc
    déterministe entre génération et résolution."""
    init_db(db_path)
    rule_fn = RULES[rule_version]
    conn = _connect(db_path)
    try:
        open_trades = conn.execute(
            "SELECT * FROM sim_trades WHERE status='open' AND rule_version=?", (rule_version,)
        ).fetchall()

        now = datetime.now(timezone.utc).isoformat()
        n_resolved = 0
        for trade in open_trades:
            log_row = conn.execute(
                "SELECT realized_price FROM all_predictions WHERE source=? AND run_id=? AND model=? "
                "AND asset=? AND horizon=? AND d_date=?",
                (trade["source"], trade["run_id"], trade["model"], trade["asset"],
                 trade["horizon"], trade["d_date"]),
            ).fetchone()
            if log_row is None or log_row["realized_price"] is None:
                continue

            realized = log_row["realized_price"]
            row_kwargs = dict(rule_kwargs)
            if rule_version == "sideways_gated_d1":
                row_kwargs["source"] = trade["source"]
                row_kwargs["vol_bucket"] = trade["vol_bucket"]
                row_kwargs["stress_score"] = trade["stress_score"]

            result = rule_fn(trade["reference_price"], trade["predicted"], trade["pi_lower"],
                             trade["pi_upper"], realized, fee_bps=fee_bps, **row_kwargs)
            _, branch, counter, roi, direction_ok, in_band, _ = result[:7]

            conn.execute("""
                UPDATE sim_trades SET realized_price=?, direction_ok=?, branch=?, counter=?,
                    roi=?, in_band=?, status='closed', evaluated_at=? WHERE id=?
            """, (realized, direction_ok, branch, counter, roi, in_band, now, trade["id"]))
            n_resolved += 1
        conn.commit()
        return n_resolved
    finally:
        conn.close()


def reconcile_oos_sim_trades(db_path=DEFAULT_DB_PATH) -> dict:
    """Réconcilie `sim_trades` (source='oos') avec les survivants `daily_duplicate=0`
    de `predictions` (BRIEF_correction_sim_trades.md §4, méthode 4.A -- suppression
    ciblée + régénération, jamais une reconstruction complète : préserve les
    `created_at` des sim_trades déjà présents pour les survivants). Ne touche jamais
    aux lignes `source='live'` (0 doublon, cf. tracking_db). Idempotent : un second
    appel ne supprime ni ne régénère rien de plus.

    Devenue **superflue pour le go-forward** depuis BRIEF_prevention_doublons.md :
    l'index OOS sans `run_id` + l'upsert de `insert_oos_predictions` rendent les
    doublons impossibles à créer, donc `daily_duplicate` reste toujours à 0 et cette
    fonction n'a plus jamais rien à faire (elle est conservée -- inoffensive, sans
    filtre : ne fait rien si aucune ligne n'est flaguée -- pour l'historique/rollback
    et au cas où l'approche par flag serait un jour réactivée). Le chemin d'ingestion
    normal utilise désormais `rebuild_oos_sim_trades` (ci-dessous), appelé par
    `ingest_oos`.

    1) Relève les `rule_version` distinctes déjà présentes en OOS *avant* suppression
       (sinon une rule_version dont tous les sim_trades étaient des doublons ne
       serait jamais régénérée pour ses survivants).
    2) Supprime les sim_trades OOS dont la prédiction source est flaguée
       `daily_duplicate=1` (jointure sur source/run_id/model/asset/horizon/
       cutoff_date=d_date), dans une transaction avec contrôle interne (le live ne
       doit jamais bouger) avant commit.
    3) Régénère (`generate_sim_trades`, idempotent) pour chaque `rule_version`
       relevée à l'étape 1, pour garantir que chaque survivant a bien son signal.

    Précondition : le schéma doit déjà exister (appeler init_db() avant, comme pour
    flag_daily_duplicates) -- volontairement pas de lazy-init ici, pour la même
    raison que flag_daily_duplicates.

    ⚠️ Limite depuis BRIEF_prevention_doublons.md : l'étape 3 (régénération) appelle
    `generate_sim_trades`, qui s'auto-initialise (`init_db()`) -- et `init_db()` tente
    désormais de (re)poser l'index dur `idx_predictions_oos_unique` à CHAQUE appel, ce
    qui échoue si des lignes `predictions` dupliquées (même flaguées, pas supprimées)
    sont encore physiquement présentes. Cette fonction ne peut donc plus être appelée
    en toute sécurité que sur une base qui n'a **aucun** doublon OOS restant en
    `predictions` -- ce qui est désormais toujours le cas une fois la suppression du
    brief appliquée, et rend cette fonction structurellement un no-op (0 supprimé,
    0 régénéré). Sur une base qui aurait encore des doublons flagués non supprimés
    (état intermédiaire pré-nettoyage), l'appeler lève une IntegrityError -- ce n'est
    plus un mode d'usage supporté.

    Retourne {"rule_versions": [...], "n_deleted": int, "n_regenerated": int}."""
    conn = _connect(db_path)
    try:
        rule_versions = sorted({
            row["rule_version"] for row in
            conn.execute("SELECT DISTINCT rule_version FROM sim_trades WHERE source='oos'")
        })

        n_before = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
        n_live_before = conn.execute(
            "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
        ).fetchone()[0]

        conn.execute("""
            DELETE FROM sim_trades
            WHERE source = 'oos'
              AND id IN (
                  SELECT s.id
                  FROM sim_trades s
                  JOIN predictions p
                    ON p.source = 'oos' AND p.run_id = s.run_id AND p.model = s.model
                   AND p.asset = s.asset AND p.horizon = s.horizon AND p.cutoff_date = s.d_date
                  WHERE p.daily_duplicate = 1
              )
        """)

        n_after_delete = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
        n_live_after_delete = conn.execute(
            "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
        ).fetchone()[0]

        if n_live_after_delete != n_live_before:
            conn.rollback()
            raise RuntimeError(
                "reconcile_oos_sim_trades : contrôle interne échoué, des lignes live "
                f"auraient été supprimées ({n_live_before} -> {n_live_after_delete}), "
                "rollback effectué."
            )

        n_deleted = n_before - n_after_delete
        conn.commit()
    finally:
        conn.close()

    n_regenerated = 0
    for rv in rule_versions:
        n_regenerated += generate_sim_trades(db_path=db_path, rule_version=rv, source="oos")

    return {
        "rule_versions": rule_versions,
        "n_deleted": n_deleted,
        "n_regenerated": n_regenerated,
    }


def rebuild_oos_sim_trades(db_path=DEFAULT_DB_PATH) -> dict:
    """Reconstruit `sim_trades` (source='oos') depuis zéro, pour TOUTES les
    rule_versions connues (`RULES`) -- BRIEF_prevention_doublons.md §6 : chemin normal
    d'après-ingestion désormais que l'index OOS empêche les doublons. Plus sûr qu'un
    rapiéçage ciblé (`reconcile_oos_sim_trades`) puisque l'upsert de
    `insert_oos_predictions` peut changer le `run_id` (et les valeurs) d'une prédiction
    déjà en base, et `sim_trades` est indexé sur `run_id` -- un ancien sim_trade lié au
    `run_id` remplacé deviendrait orphelin s'il n'était pas supprimé. L'OOS est
    déterministe et entièrement résolu (`y_true` toujours connu à l'ingestion) : une
    reconstruction complète est peu coûteuse et strictement équivalente à un rapiéçage
    correct. Ne touche jamais `source='live'` (`WHERE source='oos'` partout).
    Idempotent : rejouer sans nouvelle ingestion entre-temps supprime puis régénère
    exactement les mêmes lignes.

    Retourne {"n_deleted": int, "n_regenerated": int}."""
    init_db(db_path)

    conn = _connect(db_path)
    try:
        n_deleted = conn.execute("SELECT COUNT(*) FROM sim_trades WHERE source='oos'").fetchone()[0]
        conn.execute("DELETE FROM sim_trades WHERE source='oos'")
        conn.commit()
    finally:
        conn.close()

    n_regenerated = 0
    for rv in RULES:
        n_regenerated += generate_sim_trades(db_path=db_path, rule_version=rv, source="oos")

    return {"n_deleted": n_deleted, "n_regenerated": n_regenerated}


def sync_live_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1", fee_bps=0.0,
                     regime_lookup=None, **rule_kwargs) -> dict:
    """Point d'entrée appelé par evaluate_daily.py : génère les nouveaux sim_trades pour
    les prédictions live déjà dans `predictions` et résout les 'open' devenus
    résolubles. Idempotent de bout en bout. Depuis BRIEF_db_unification.md, le live vit
    nativement dans `predictions` (écrit par tracking_db.save_prediction) -- plus
    d'étape d'ingestion/rafraîchissement séparée à faire ici, `all_predictions` la
    reflète déjà.

    regime_lookup : uniquement consulté pour rule_version="sideways_gated_d1", cf.
    generate_sim_trades -- ignoré (et inutile) à la résolution, qui relit
    vol_bucket/stress_score déjà figés sur le trade."""
    init_db(db_path)
    n_new = generate_sim_trades(db_path=db_path, rule_version=rule_version, fee_bps=fee_bps,
                                source="live", regime_lookup=regime_lookup, **rule_kwargs)
    n_resolved = resolve_open_sim_trades(db_path=db_path, rule_version=rule_version,
                                         fee_bps=fee_bps, **rule_kwargs)
    return {"new_trades": n_new, "resolved": n_resolved}


# ── KPIs (§9) ────────────────────────────────────────────────────────────────────

def kpi_report(db_path=DEFAULT_DB_PATH, source="oos", rule_version="bull_calm_d1",
              group_by=("asset", "model"), include_degenerate=False,
              k_values=None, gate_values=None, m_frac=0.25, h_frac=0.50) -> list:
    """KPIs par group_by (sous-ensemble de {asset, model, regime}) — appeler avec
    group_by=() pour l'agrégat global (§11.5 Bull-Calm, les deux niveaux sont à
    reporter). Ne mélange jamais source='oos' et 'live' (paramètre obligatoire, pas de
    défaut combiné). group_by='regime' interdit pour source='oos' (regime y est
    toujours 'unknown', cf. docstring module). Les signaux non résolus (status='open')
    sont exclus de tous les KPIs, y compris n_signaux ; comptés séparément en n_open.

    Pour rule_version="bull_calm_d1"/"pi95_conf" (BRIEF_bull_calm_d1.md §9) : KPIs
    orientés ROI. Pour rule_version="sideways_d1" (BRIEF_sideways_d1.md §8) : variante
    "justesse" sans ROI (taux_justesse/immobile/breakout haussier-baissier, in_band).
    Pour rule_version="sideways_gated_d1" (BRIEF_sideways_v2.md §8) : justesse v1 +
    P&L short-vol (`pnl_shortvol_*` -- JAMAIS `roi_*`, proxy short-straddle d'évaluation
    non exécuté) + coupe par vol_bucket + garde-fous sharpness/queue + valeur ajoutée du
    gate (`n_gated_out`/`n_signal_v1`).

    k_values : balayage de sensibilité §8.7 du brief Sideways (uniquement valide avec
    rule_version="sideways_d1"). Recalcule taux_signal/taux_justesse pour chaque k
    DIRECTEMENT depuis all_predictions (le k des sim_trades déjà générés est figé à la
    génération) -- purement en lecture, n'écrit jamais dans sim_trades. Résultat
    attaché à chaque groupe sous la clé 'k_sensitivity'.

    gate_values : balayage (vb_max, stress_max) §8.8 (uniquement valide avec
    rule_version="sideways_gated_d1"). Ne peut que DURCIR le gate déjà appliqué à la
    génération -- filtre en mémoire les signaux déjà tradables (vol_bucket/stress_score
    sont figés par ligne, §1) ; n'élargit jamais au-delà (les lignes gated_out à la
    génération n'ont pas de pnl_shortvol connu). Résultat attaché sous 'gate_sensitivity'."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source invalide : {source!r} (attendu parmi {VALID_SOURCES})")
    invalid = set(group_by) - _GROUP_BY_COLUMNS
    if invalid:
        raise ValueError(f"group_by invalide : {invalid} (attendu parmi {_GROUP_BY_COLUMNS})")
    if source == "oos" and "regime" in group_by:
        raise ValueError(
            "group_by='regime' n'a pas de sens pour source='oos' (regime y est toujours "
            "'unknown' -- business_validation.json ne décrit que la prévision live la plus "
            "récente, pas les jours historiques du backtest). Utiliser source='live'.")
    if k_values is not None and rule_version != "sideways_d1":
        raise ValueError("k_values n'a de sens que pour rule_version='sideways_d1'")
    if gate_values is not None and rule_version != "sideways_gated_d1":
        raise ValueError("gate_values n'a de sens que pour rule_version='sideways_gated_d1'")

    init_db(db_path)
    conn = _connect(db_path)
    try:
        log_rows = conn.execute("SELECT * FROM all_predictions WHERE source=?", (source,)).fetchall()
        trade_rows = conn.execute(
            "SELECT * FROM sim_trades WHERE source=? AND rule_version=?", (source, rule_version)
        ).fetchall()
    finally:
        conn.close()

    def key_of(row):
        return tuple(row[c] for c in group_by)

    def _new_group():
        return {"log": [], "signals": [], "n_open": 0, "n_gated_out": 0}

    groups = {}
    for row in log_rows:
        groups.setdefault(key_of(row), _new_group())["log"].append(row)
    for row in trade_rows:
        if not include_degenerate and row["degenerate_pi"]:
            continue
        g = groups.setdefault(key_of(row), _new_group())
        if not row["signal_valid"]:
            g["n_gated_out"] += 1   # flat suspect (sideways_gated_d1 uniquement, §6)
        elif row["status"] == "closed":
            g["signals"].append(row)
        else:
            g["n_open"] += 1

    if rule_version == "sideways_gated_d1":
        summarize_fn = _summarize_group_sideways_gated
    elif rule_version == "sideways_d1":
        summarize_fn = _summarize_group_sideways
    else:
        summarize_fn = _summarize_group

    results = []
    for key, data in sorted(groups.items()):
        entry = dict(zip(group_by, key))
        if rule_version == "sideways_gated_d1":
            entry.update(summarize_fn(data["log"], data["signals"], data["n_open"], data["n_gated_out"]))
        else:
            entry.update(summarize_fn(data["log"], data["signals"], data["n_open"]))
        if k_values is not None:
            entry["k_sensitivity"] = _sideways_k_sweep(
                data["log"], k_values, m_frac, h_frac, include_degenerate)
        if gate_values is not None:
            entry["gate_sensitivity"] = _gate_sweep(data["signals"], gate_values)
        results.append(entry)
    return results


def _sideways_k_sweep(log_rows, k_values, m_frac, h_frac, include_degenerate) -> list:
    """Recalcule en mémoire, pour chaque k, le nombre de signaux et le taux de justesse
    sideways_d1 sur les lignes all_predictions fournies -- jamais persisté (§8.7)."""
    sweep = []
    for k in k_values:
        n_total = len(log_rows)
        n_signaux = 0
        n_juste = 0
        for row in log_rows:
            signal, branch, counter, roi, in_band, degenerate = sideways_d1(
                row["reference_price"], row["predicted"], row["pi_lower"], row["pi_upper"],
                row["realized_price"], k=k, m_frac=m_frac, h_frac=h_frac)
            if not signal or (not include_degenerate and degenerate):
                continue
            if branch is None:   # signal ouvert (live non résolu) : compte hors justesse
                continue
            n_signaux += 1
            if counter >= 1:
                n_juste += 1
        sweep.append({
            "k": k,
            "n_signaux": n_signaux,
            "taux_signal": round(n_signaux / n_total, 4) if n_total else None,
            "taux_justesse": round(n_juste / n_signaux, 4) if n_signaux else None,
        })
    return sweep


def _summarize_group(log_rows, signals, n_open) -> dict:
    n_total = len(log_rows)
    n_signaux = len(signals)
    entry = {
        "n_total": n_total,
        "n_signaux": n_signaux,
        "n_open": n_open,
        "n_flat": n_total - n_signaux - n_open,
        "taux_signal": round(n_signaux / n_total, 4) if n_total else None,
    }
    if not signals:
        entry.update({
            "precision_direction": None, "taux_realisation": None,
            "counter_sum": None, "counter_mean": None,
            "branch_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
            "roi_sum": None, "roi_compound": None, "roi_mean": None,
            "roi_median": None, "roi_min": None, "sharpe": None,
            "pi_coverage_95": None,
        })
        return entry

    n = len(signals)
    directions = [t["direction_ok"] for t in signals]
    counters = [t["counter"] for t in signals]
    rois = sorted(t["roi"] for t in signals)
    branch_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for t in signals:
        branch_dist[t["branch"]] += 1
    coverage = [1 if t["pi_lower"] <= t["realized_price"] <= t["pi_upper"] else 0 for t in signals]

    roi_mean = sum(rois) / n
    compound = 1.0
    for r in rois:
        compound *= (1 + r)
    compound -= 1.0
    median = rois[n // 2] if n % 2 == 1 else (rois[n // 2 - 1] + rois[n // 2]) / 2

    sharpe = None
    if n >= 2:
        variance = sum((r - roi_mean) ** 2 for r in rois) / (n - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = (roi_mean / std) * math.sqrt(252)

    entry.update({
        "precision_direction": round(sum(directions) / n, 4),
        "taux_realisation": round(sum(1 for c in counters if c >= 1) / n, 4),
        "counter_sum": sum(counters),
        "counter_mean": round(sum(counters) / n, 4),
        "branch_distribution": branch_dist,
        "roi_sum": round(sum(rois), 6),
        "roi_compound": round(compound, 6),
        "roi_mean": round(roi_mean, 6),
        "roi_median": round(median, 6),
        "roi_min": round(min(rois), 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "pi_coverage_95": round(sum(coverage) / n, 4),
    })
    return entry


def _summarize_group_sideways(log_rows, signals, n_open) -> dict:
    """Variante "justesse" (BRIEF_sideways_d1.md §8) : pas de ROI. Taux de breakout
    décomposé haussier (realized > pi_upper) / baissier (realized < pi_lower)."""
    n_total = len(log_rows)
    n_signaux = len(signals)
    entry = {
        "n_total": n_total,
        "n_signaux": n_signaux,
        "n_open": n_open,
        "n_flat": n_total - n_signaux - n_open,
        "taux_signal": round(n_signaux / n_total, 4) if n_total else None,
    }
    if not signals:
        entry.update({
            "taux_justesse": None, "taux_immobile": None,
            "taux_breakout": None, "taux_breakout_haussier": None, "taux_breakout_baissier": None,
            "counter_sum": None, "counter_mean": None,
            "branch_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
            "in_band_coverage": None,
        })
        return entry

    n = len(signals)
    counters = [t["counter"] for t in signals]
    branch_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for t in signals:
        branch_dist[t["branch"]] += 1
    in_band_vals = [t["in_band"] for t in signals]

    n_breakout_haussier = sum(
        1 for t in signals if t["branch"] in (3, 4) and t["realized_price"] > t["pi_upper"])
    n_breakout_baissier = sum(
        1 for t in signals if t["branch"] in (3, 4) and t["realized_price"] < t["pi_lower"])

    entry.update({
        "taux_justesse": round(sum(1 for c in counters if c >= 1) / n, 4),
        "taux_immobile": round(sum(1 for c in counters if c == 2) / n, 4),
        "taux_breakout": round(sum(1 for c in counters if c < 0) / n, 4),
        "taux_breakout_haussier": round(n_breakout_haussier / n, 4),
        "taux_breakout_baissier": round(n_breakout_baissier / n, 4),
        "counter_sum": sum(counters),
        "counter_mean": round(sum(counters) / n, 4),
        "branch_distribution": branch_dist,
        "in_band_coverage": round(sum(in_band_vals) / n, 4),
    })
    return entry


def _gate_sweep(signals, gate_values) -> list:
    """Balayage (vb_max, stress_max) (BRIEF_sideways_v2.md §8.8) : ne peut que DURCIR le
    gate déjà appliqué à la génération -- vol_bucket/stress_score sont figés par ligne
    (§1), donc ce balayage retrace, pour des seuils plus stricts, combien de signaux
    déjà tradables le resteraient et leur sharpe_shortvol. N'élargit jamais au-delà du
    gate de génération (les lignes gated_out à la génération n'ont pas de pnl_shortvol
    connu, cf. docstring kpi_report)."""
    sweep = []
    for vb_max, stress_max in gate_values:
        passing = [
            t for t in signals
            if (t["vol_bucket"] is None or t["vol_bucket"] <= vb_max)
            and (t["source"] != "live" or t["stress_score"] is None or t["stress_score"] <= stress_max)
        ]
        n = len(passing)
        sharpe = None
        if n >= 2:
            pnl = [t["roi"] for t in passing]
            mean = sum(pnl) / n
            variance = sum((p - mean) ** 2 for p in pnl) / (n - 1)
            std = math.sqrt(variance)
            if std > 0:
                sharpe = round((mean / std) * math.sqrt(252), 4)
        sweep.append({
            "vb_max": vb_max, "stress_max": stress_max,
            "n_signal_tradable": n, "sharpe_shortvol": sharpe,
        })
    return sweep


def _summarize_group_sideways_gated(log_rows, signals, n_open, n_gated_out) -> dict:
    """KPI sideways_gated_d1 (BRIEF_sideways_v2.md §8) : justesse v1 (identique à
    _summarize_group_sideways) + P&L short-vol (`pnl_shortvol_*` -- JAMAIS `roi_*`,
    `roi` stocke un proxy short-straddle d'évaluation, jamais un rendement exécuté,
    cf. docstring sideways_gated_d1) + coupe par vol_bucket + garde-fous sharpness
    (rel_width_mean/move_ratio_mean) + risque de queue (cvar_5/skew/freq_floor/calmar)
    + valeur ajoutée du gate (n_gated_out / n_signal_v1)."""
    n_total = len(log_rows)
    n_signaux = len(signals)
    n_signal_v1 = n_signaux + n_gated_out
    entry = {
        "n_total": n_total,
        "n_signaux": n_signaux,
        "n_open": n_open,
        "n_gated_out": n_gated_out,
        "n_signal_v1": n_signal_v1,
        "n_flat": n_total - n_signaux - n_open - n_gated_out,
        "taux_signal": round(n_signaux / n_total, 4) if n_total else None,
        "gated_out_ratio": round(n_gated_out / n_signal_v1, 4) if n_signal_v1 else None,
    }
    if not signals:
        entry.update({
            "taux_justesse": None, "taux_immobile": None,
            "taux_breakout": None, "taux_breakout_haussier": None, "taux_breakout_baissier": None,
            "counter_sum": None, "counter_mean": None,
            "branch_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
            "in_band_coverage": None, "n_gate_undefined": 0,
            "pnl_shortvol_mean": None, "pnl_shortvol_median": None, "pnl_shortvol_sum": None,
            "pnl_shortvol_min": None, "sharpe_shortvol": None,
            "rel_width_mean": None, "move_ratio_mean": None,
            "cvar_5_shortvol": None, "pnl_skew": None, "freq_floor": None, "calmar_shortvol": None,
            "by_vol_bucket": {},
        })
        return entry

    n = len(signals)
    counters = [t["counter"] for t in signals]
    branch_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for t in signals:
        branch_dist[t["branch"]] += 1
    in_band_vals = [t["in_band"] for t in signals]
    n_breakout_haussier = sum(
        1 for t in signals if t["branch"] in (3, 4) and t["realized_price"] > t["pi_upper"])
    n_breakout_baissier = sum(
        1 for t in signals if t["branch"] in (3, 4) and t["realized_price"] < t["pi_lower"])
    n_gate_undefined = sum(1 for t in signals if t["vol_bucket"] is None)

    pnl = [t["roi"] for t in signals]   # roi <- pnl_shortvol (proxy short-straddle, jamais un rendement)
    pnl_mean = sum(pnl) / n
    pnl_sorted = sorted(pnl)
    pnl_median = pnl_sorted[n // 2] if n % 2 == 1 else (pnl_sorted[n // 2 - 1] + pnl_sorted[n // 2]) / 2

    sharpe = None
    if n >= 2:
        variance = sum((p - pnl_mean) ** 2 for p in pnl) / (n - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = (pnl_mean / std) * math.sqrt(252)

    rel_width = [(t["pi_upper"] - t["pi_lower"]) / t["reference_price"] for t in signals]
    move_ratio = [
        abs(t["realized_price"] - t["reference_price"]) / ((t["pi_upper"] - t["pi_lower"]) / 2.0)
        for t in signals
    ]

    n_tail = max(1, math.ceil(0.05 * n))
    cvar_5 = sum(pnl_sorted[:n_tail]) / n_tail
    pop_variance = sum((p - pnl_mean) ** 2 for p in pnl) / n
    pop_std = math.sqrt(pop_variance)
    pnl_skew = (sum((p - pnl_mean) ** 3 for p in pnl) / n) / (pop_std ** 3) if pop_std > 0 else None
    freq_floor = round(branch_dist[4] / n, 4)

    cumulative, running = [], 0.0
    for t in sorted(signals, key=lambda t: t["d_date"]):
        running += t["roi"]
        cumulative.append(running)
    peak, max_drawdown = float("-inf"), 0.0
    for value in cumulative:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)
    calmar = round(pnl_mean / max_drawdown, 6) if max_drawdown > 0 else None

    by_vol_bucket = {}
    for bucket in (0, 1, 2, None):
        bucket_signals = [t for t in signals if t["vol_bucket"] == bucket]
        if not bucket_signals:
            continue
        bn = len(bucket_signals)
        bucket_counters = [t["counter"] for t in bucket_signals]
        bucket_pnl = [t["roi"] for t in bucket_signals]
        by_vol_bucket[bucket] = {
            "n": bn,
            "taux_justesse": round(sum(1 for c in bucket_counters if c >= 1) / bn, 4),
            "pnl_shortvol_mean": round(sum(bucket_pnl) / bn, 6),
        }

    entry.update({
        "taux_justesse": round(sum(1 for c in counters if c >= 1) / n, 4),
        "taux_immobile": round(sum(1 for c in counters if c == 2) / n, 4),
        "taux_breakout": round(sum(1 for c in counters if c < 0) / n, 4),
        "taux_breakout_haussier": round(n_breakout_haussier / n, 4),
        "taux_breakout_baissier": round(n_breakout_baissier / n, 4),
        "counter_sum": sum(counters),
        "counter_mean": round(sum(counters) / n, 4),
        "branch_distribution": branch_dist,
        "in_band_coverage": round(sum(in_band_vals) / n, 4),
        "n_gate_undefined": n_gate_undefined,
        "pnl_shortvol_mean": round(pnl_mean, 6),
        "pnl_shortvol_median": round(pnl_median, 6),
        "pnl_shortvol_sum": round(sum(pnl), 6),
        "pnl_shortvol_min": round(min(pnl), 6),
        "sharpe_shortvol": round(sharpe, 4) if sharpe is not None else None,
        "rel_width_mean": round(sum(rel_width) / n, 6),
        "move_ratio_mean": round(sum(move_ratio) / n, 6),
        "cvar_5_shortvol": round(cvar_5, 6),
        "pnl_skew": round(pnl_skew, 6) if pnl_skew is not None else None,
        "freq_floor": freq_floor,
        "calmar_shortvol": calmar,
        "by_vol_bucket": by_vol_bucket,
    })
    return entry


def naive_always_long_report(db_path=DEFAULT_DB_PATH, source="oos", model="Naive",
                             group_by=("asset",)) -> list:
    """KPI 8 (§9.8), benchmark -- décision tuteur (2026-07-13) : le prédicteur naïf pur
    (persistance stricte, predicted[t]==actual[t-1]) ne déclenche jamais bull_calm_d1
    (predicted>ref toujours faux) -- vérifié empiriquement sur les runs Naive-*-D1
    récents (0 signal). Le benchmark 'signal-filtré' serait donc vide et sans valeur de
    comparaison. On applique ici la résolution des branches (§4) à CHAQUE ligne du log
    Naive, sans filtre de signal, pour mesurer la valeur ajoutée du filtre predicted>ref
    des autres modèles (est-on meilleur qu'être long tous les jours ?). Jamais persisté
    dans sim_trades : ce n'est pas un 'signal' au sens du schéma §7.2, seulement un
    calcul de rapport."""
    invalid = set(group_by) - _GROUP_BY_COLUMNS
    if invalid:
        raise ValueError(f"group_by invalide : {invalid} (attendu parmi {_GROUP_BY_COLUMNS})")
    if source == "oos" and "regime" in group_by:
        raise ValueError("group_by='regime' n'a pas de sens pour source='oos'")

    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM all_predictions WHERE source=? AND model=? AND realized_price IS NOT NULL",
            (source, model),
        ).fetchall()
    finally:
        conn.close()

    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[c] for c in group_by), []).append(row)

    results = []
    for key, group_rows in sorted(groups.items()):
        entry = dict(zip(group_by, key))
        rois = []
        branch_dist = {1: 0, 2: 0, 3: 0, 4: 0}
        for row in group_rows:
            branch, _counter, exit_px = _resolve_branches(
                row["reference_price"], row["pi_lower"], row["pi_upper"], row["realized_price"])
            rois.append((exit_px - row["reference_price"]) / row["reference_price"])
            branch_dist[branch] += 1

        n = len(rois)
        compound = 1.0
        for r in rois:
            compound *= (1 + r)
        compound -= 1.0

        entry.update({
            "n_days": n,
            "roi_sum": round(sum(rois), 6) if n else None,
            "roi_mean": round(sum(rois) / n, 6) if n else None,
            "roi_compound": round(compound, 6) if n else None,
            "branch_distribution": branch_dist,
        })
        results.append(entry)
    return results


def naive_always_short_report(db_path=DEFAULT_DB_PATH, source="oos", model="Naive",
                              group_by=("asset",)) -> list:
    """Miroir de naive_always_long_report (KPI 8 côté Bear) : applique la résolution des
    branches courtes (§3bis) à CHAQUE ligne du log Naive, sans filtre de signal, pour
    mesurer la valeur ajoutée du filtre predicted<ref des autres modèles (est-on meilleur
    qu'être short tous les jours ?). Jamais persisté dans sim_trades."""
    invalid = set(group_by) - _GROUP_BY_COLUMNS
    if invalid:
        raise ValueError(f"group_by invalide : {invalid} (attendu parmi {_GROUP_BY_COLUMNS})")
    if source == "oos" and "regime" in group_by:
        raise ValueError("group_by='regime' n'a pas de sens pour source='oos'")

    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM all_predictions WHERE source=? AND model=? AND realized_price IS NOT NULL",
            (source, model),
        ).fetchall()
    finally:
        conn.close()

    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[c] for c in group_by), []).append(row)

    results = []
    for key, group_rows in sorted(groups.items()):
        entry = dict(zip(group_by, key))
        rois = []
        branch_dist = {1: 0, 2: 0, 3: 0, 4: 0}
        for row in group_rows:
            branch, _counter, exit_px = _resolve_branches_bear(
                row["reference_price"], row["pi_lower"], row["pi_upper"], row["realized_price"])
            rois.append((row["reference_price"] - exit_px) / row["reference_price"])
            branch_dist[branch] += 1

        n = len(rois)
        compound = 1.0
        for r in rois:
            compound *= (1 + r)
        compound -= 1.0

        entry.update({
            "n_days": n,
            "roi_sum": round(sum(rois), 6) if n else None,
            "roi_mean": round(sum(rois) / n, 6) if n else None,
            "roi_compound": round(compound, 6) if n else None,
            "branch_distribution": branch_dist,
        })
        results.append(entry)
    return results


RULE_TC_ID = {
    "bull_calm_d1": "TC1.1", "pi95_conf": "TC1.2",
    "bear_calm_d1": "TC1.3", "bear_stress_d1": "TC1.4", "sideways_d1": "TC1.5",
    "sideways_gated_d1": "TC1.5b",
}


def daily_detail(db_path=DEFAULT_DB_PATH, asset=None, models=None) -> list:
    """Vue jour par jour (une ligne = un jour de `all_predictions`, horizon=1) avec le(s)
    test case(s) TC1.1-TC1.5 qui ont généré un signal ce jour-là et leur `counter`, pour
    l'inspection détaillée (dashboard Run/, tableau "Test cases (D+1)"). `all_predictions`
    filtre déjà horizon=1 (BRIEF_db_unification.md §2.4 -- seul horizon supporté par les 5
    règles, l'alignement D->D+1 ne s'applique pas au backtest D+7 rolling-origin). Un jour
    peut ne correspondre à aucune règle (`signals` vide -- flat pour les 5) ou, en
    bordure, à plus d'une (léger recouvrement Bull-Calm/Sideways documenté dans
    BRIEF_sideways_d1.md §0) : `signals` est donc une liste, jamais une valeur unique
    supposée."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        query = "SELECT * FROM all_predictions"
        params: list = []
        clauses = []
        if asset is not None:
            clauses.append("asset = ?")
            params.append(asset)
        if models is not None:
            clauses.append(f"model IN ({','.join('?' * len(models))})")
            params += list(models)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        log_rows = conn.execute(query, params).fetchall()

        trades_by_key: dict = {}
        for rule_version in RULES:
            trade_query = "SELECT * FROM sim_trades WHERE rule_version = ? AND horizon = 1"
            trade_params: list = [rule_version]
            if asset is not None:
                trade_query += " AND asset = ?"
                trade_params.append(asset)
            if models is not None:
                trade_query += f" AND model IN ({','.join('?' * len(models))})"
                trade_params += list(models)
            for row in conn.execute(trade_query, trade_params).fetchall():
                key = (row["source"], row["run_id"], row["model"], row["asset"], row["d_date"])
                trades_by_key.setdefault(key, []).append(row)
    finally:
        conn.close()

    results = []
    for row in log_rows:
        key = (row["source"], row["run_id"], row["model"], row["asset"], row["d_date"])
        signals = [{
            "tc_id": RULE_TC_ID[m["rule_version"]], "rule_version": m["rule_version"],
            "branch": m["branch"], "counter": m["counter"], "roi": m["roi"], "status": m["status"],
            "gated_out": m["gated_out"],
        } for m in trades_by_key.get(key, [])]
        results.append({
            "source": row["source"], "d_date": row["d_date"], "target_date": row["target_date"],
            "model": row["model"], "asset": row["asset"],
            "reference_price": row["reference_price"], "predicted": row["predicted"],
            "pi_lower": row["pi_lower"], "pi_upper": row["pi_upper"],
            "realized_price": row["realized_price"],
            "signals": signals,
        })
    results.sort(key=lambda r: (r["d_date"], r["model"]), reverse=True)
    return results


# ── CLI (usage manuel, non couvert par les tests) ────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="bull_calm_d1 : ingestion + rapport KPI")
    p.add_argument("--db-path", default="validation/tracking.db")
    p.add_argument("--run-root", default="Run")
    p.add_argument("--ingest-oos", action="store_true", help="ingère les Run/*-D1/ dans predictions (source='oos')")
    p.add_argument("--sync-live", action="store_true", help="ingère/résout les prédictions live")
    p.add_argument("--report", choices=["oos", "live"], help="imprime le rapport KPI pour cette source")
    args = p.parse_args()

    init_db(args.db_path)

    if args.ingest_oos:
        stats = ingest_oos(run_root=args.run_root, db_path=args.db_path)
        print(f"[sim_trades] ingestion OOS : {stats}")
        n_trades = generate_sim_trades(db_path=args.db_path, source="oos")
        print(f"[sim_trades] {n_trades} nouveau(x) sim_trade(s) OOS (bull_calm_d1)")

    if args.sync_live:
        result = sync_live_trades(db_path=args.db_path)
        print(f"[sim_trades] sync live : {result}")

    if args.report:
        for row in kpi_report(db_path=args.db_path, source=args.report, group_by=("asset", "model")):
            print(row)
        print("--- agrégé ---")
        for row in kpi_report(db_path=args.db_path, source=args.report, group_by=()):
            print(row)


if __name__ == "__main__":
    main()
