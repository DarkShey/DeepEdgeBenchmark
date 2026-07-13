"""
sim_trades.py — Test Case Bull à D+1 (`bull_calm_d1`), cf. BRIEF_bull_calm_d1.md.

Construit `daily_oos_log` (vue normalisée D->D+1, alignement anti-look-ahead §2) à partir
des `Run/*-D1/predictions.parquet` (`source="oos"`) et de `validation.tracking_db`
(`source="live"`), applique la règle `bull_calm_d1` (+ règle sœur `pi95_conf`, §3) et
persiste les signaux valides dans `sim_trades`. Stdlib (sqlite3, json) + pandas (lecture
parquet) uniquement — même contrainte de dépendances que `tracking_db.py`.

Restreint aux combos horizon=1 jour de bourse (dossiers `*-D1`) : `predictions.parquet`
d'un dossier `*-D7` est un backtest rolling-origin espacé de plusieurs jours (pas un log
quotidien consécutif), l'alignement D->D+1 de bull_calm_d1 ne s'y applique pas.

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


def _connect(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH) -> None:
    """Crée daily_oos_log et sim_trades si absentes (CREATE TABLE IF NOT EXISTS,
    idempotent) — schéma §7 du brief."""
    conn = _connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_oos_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                model             TEXT NOT NULL,
                asset             TEXT NOT NULL,
                horizon           INTEGER NOT NULL,
                regime            TEXT NOT NULL,
                d_date            TEXT NOT NULL,
                target_date       TEXT NOT NULL,
                reference_price   REAL NOT NULL,
                predicted         REAL NOT NULL,
                pi_lower          REAL NOT NULL,
                pi_upper          REAL NOT NULL,
                realized_price    REAL,
                source            TEXT NOT NULL,
                created_at        TEXT NOT NULL,
                UNIQUE (source, run_id, model, asset, horizon, d_date)
            )
        """)
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


RULES = {
    "bull_calm_d1": _adapt_bull_like(bull_calm_d1),      # TC1.1 Bull-Calm
    "pi95_conf": _adapt_bull_like(pi95_conf),             # TC1.2 Bull-Stress
    "bear_calm_d1": _adapt_bear_like(bear_calm_d1),       # TC1.3 Bear-Calm
    "bear_stress_d1": _adapt_bear_like(bear_stress_d1),   # TC1.4 Bear-Stress
    "sideways_d1": _adapt_sideways(sideways_d1),          # TC1.5 Sideways
}


# ── Construction daily_oos_log — source="oos" (Run/*-D1/predictions.parquet) ───

def _date_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_daily_oos_log_rows(run_dir, source="oos"):
    """Lit predictions.parquet + metrics.json d'un dossier Run/<...>-D1/, applique
    l'alignement §2 (t>=1 ; t=0 ignoré faute de t-1, §6.1). Exclut les lignes NaN /
    bornes cassées (pi_lower > pi_upper, §6.3). Retourne (rows, n_dropped)."""
    run_dir = Path(run_dir)
    df = pd.read_parquet(run_dir / "predictions.parquet").sort_values("date").reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model = metrics["model"]
    asset = metrics["asset"]
    horizon = _HORIZON_LABEL_TO_INT.get(metrics.get("horizon"), 1)

    run_id = run_dir.name
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    n_dropped = 0

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
            "d_date": _date_str(df.loc[t - 1, "date"]),
            "target_date": _date_str(df.loc[t, "date"]),
            "reference_price": float(reference_price), "predicted": float(predicted),
            "pi_lower": float(pi_lower), "pi_upper": float(pi_upper),
            "realized_price": float(realized_price),
            "source": source, "created_at": now,
        })

    return rows, n_dropped


def insert_daily_oos_log(rows, db_path=DEFAULT_DB_PATH) -> int:
    """INSERT OR IGNORE (idempotent sur la contrainte UNIQUE). Retourne le nombre de
    lignes effectivement insérées."""
    if not rows:
        return 0
    init_db(db_path)
    cols = ("run_id", "model", "asset", "horizon", "regime", "d_date", "target_date",
            "reference_price", "predicted", "pi_lower", "pi_upper", "realized_price",
            "source", "created_at")
    placeholders = ", ".join(f":{c}" for c in cols)
    columns = ", ".join(cols)
    conn = _connect(db_path)
    try:
        n = 0
        for row in rows:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO daily_oos_log ({columns}) VALUES ({placeholders})", row)
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def ingest_oos(run_root="Run", db_path=DEFAULT_DB_PATH) -> dict:
    """Parcourt tous les Run/*-D1/ (skip silencieusement si predictions.parquet ou
    metrics.json manque), construit et insère daily_oos_log. Idempotent."""
    init_db(db_path)
    inserted = dropped = combos = 0
    for run_dir in sorted(Path(run_root).glob("*-D1")):
        if not (run_dir / "predictions.parquet").exists() or not (run_dir / "metrics.json").exists():
            continue
        rows, n_dropped = build_daily_oos_log_rows(run_dir)
        inserted += insert_daily_oos_log(rows, db_path=db_path)
        dropped += n_dropped
        combos += 1
    return {"combos": combos, "inserted": inserted, "dropped": dropped}


# ── Construction daily_oos_log — source="live" (validation.tracking_db) ────────

def build_live_daily_oos_log_rows(db_path=DEFAULT_DB_PATH):
    """Lit les prédictions live horizon=1 de tracking.db (déjà alignées par
    construction, §2) et les met en forme daily_oos_log. Exclut NaN / bornes cassées."""
    td.init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT run_id, model, asset, horizon, regime, cutoff_date, target_date, "
            "last_close, y_pred, y_lower, y_upper, y_true FROM predictions WHERE horizon = 1"
        ).fetchall()
    finally:
        conn.close()

    now = datetime.now(timezone.utc).isoformat()
    out = []
    for row in rows:
        values = (row["last_close"], row["y_pred"], row["y_lower"], row["y_upper"])
        if not all(_is_finite(v) for v in values) or row["y_lower"] > row["y_upper"]:
            continue
        out.append({
            "run_id": row["run_id"], "model": row["model"], "asset": row["asset"],
            "horizon": row["horizon"], "regime": row["regime"],
            "d_date": row["cutoff_date"], "target_date": row["target_date"],
            "reference_price": row["last_close"], "predicted": row["y_pred"],
            "pi_lower": row["y_lower"], "pi_upper": row["y_upper"],
            "realized_price": row["y_true"], "source": "live", "created_at": now,
        })
    return out


def ingest_live_daily_oos_log(db_path=DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    return insert_daily_oos_log(build_live_daily_oos_log_rows(db_path=db_path), db_path=db_path)


def refresh_live_realized_prices(db_path=DEFAULT_DB_PATH) -> int:
    """Reporte dans daily_oos_log(source='live') les y_true fraîchement résolus dans
    tracking.db (evaluate_pending) pour les lignes encore NULL. Idempotent."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        pending = conn.execute(
            "SELECT id, run_id, model, asset, d_date FROM daily_oos_log "
            "WHERE source='live' AND realized_price IS NULL"
        ).fetchall()
        n = 0
        for row in pending:
            pred = conn.execute(
                "SELECT y_true FROM predictions WHERE run_id=? AND model=? AND asset=? AND cutoff_date=?",
                (row["run_id"], row["model"], row["asset"], row["d_date"]),
            ).fetchone()
            if pred is None or pred["y_true"] is None:
                continue
            conn.execute("UPDATE daily_oos_log SET realized_price=? WHERE id=?",
                        (pred["y_true"], row["id"]))
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


# ── sim_trades : génération + résolution ────────────────────────────────────────

def generate_sim_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1",
                        fee_bps=0.0, source=None, **rule_kwargs) -> int:
    """Applique la règle à chaque ligne daily_oos_log n'ayant pas encore de sim_trade
    pour cette rule_version. Les flats (signal_valid=False) ne génèrent pas de ligne
    (§7.2). Idempotent (UNIQUE + LEFT JOIN ... IS NULL). `**rule_kwargs` passe les
    paramètres spécifiques à la règle (k/m_frac/h_frac pour sideways_d1) ; les
    adaptateurs de RULES ignorent ceux qui ne les concernent pas. Retourne le nombre
    de lignes insérées."""
    init_db(db_path)
    rule_fn = RULES[rule_version]
    conn = _connect(db_path)
    try:
        query = """
            SELECT l.* FROM daily_oos_log l
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

        now = datetime.now(timezone.utc).isoformat()
        n_inserted = 0
        for row in rows:
            realized = row["realized_price"]
            signal_valid, branch, counter, roi, direction_ok, in_band, degenerate = rule_fn(
                row["reference_price"], row["predicted"], row["pi_lower"], row["pi_upper"],
                realized, fee_bps=fee_bps, **rule_kwargs)
            if not signal_valid:
                continue

            status = "closed" if branch is not None else "open"
            evaluated_at = now if status == "closed" else None

            cur = conn.execute("""
                INSERT OR IGNORE INTO sim_trades (
                    rule_version, run_id, model, asset, horizon, regime, source,
                    d_date, target_date, reference_price, predicted, pi_lower, pi_upper,
                    realized_price, signal_valid, direction_ok, branch, counter, roi, in_band,
                    degenerate_pi, status, created_at, evaluated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (rule_version, row["run_id"], row["model"], row["asset"], row["horizon"],
                  row["regime"], row["source"], row["d_date"], row["target_date"],
                  row["reference_price"], row["predicted"], row["pi_lower"], row["pi_upper"],
                  realized, 1, direction_ok, branch, counter, roi, in_band, degenerate, status,
                  now, evaluated_at))
            n_inserted += cur.rowcount
        conn.commit()
        return n_inserted
    finally:
        conn.close()


def resolve_open_sim_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1",
                            fee_bps=0.0, **rule_kwargs) -> int:
    """Réévalue les sim_trades status='open' dont le realized_price est maintenant
    connu dans daily_oos_log (mis à jour au préalable par refresh_live_realized_prices).
    Idempotent : ne touche jamais un trade déjà 'closed'."""
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
                "SELECT realized_price FROM daily_oos_log WHERE source=? AND run_id=? AND model=? "
                "AND asset=? AND horizon=? AND d_date=?",
                (trade["source"], trade["run_id"], trade["model"], trade["asset"],
                 trade["horizon"], trade["d_date"]),
            ).fetchone()
            if log_row is None or log_row["realized_price"] is None:
                continue

            realized = log_row["realized_price"]
            _, branch, counter, roi, direction_ok, in_band, _ = rule_fn(
                trade["reference_price"], trade["predicted"], trade["pi_lower"],
                trade["pi_upper"], realized, fee_bps=fee_bps, **rule_kwargs)

            conn.execute("""
                UPDATE sim_trades SET realized_price=?, direction_ok=?, branch=?, counter=?,
                    roi=?, in_band=?, status='closed', evaluated_at=? WHERE id=?
            """, (realized, direction_ok, branch, counter, roi, in_band, now, trade["id"]))
            n_resolved += 1
        conn.commit()
        return n_resolved
    finally:
        conn.close()


def sync_live_trades(db_path=DEFAULT_DB_PATH, rule_version="bull_calm_d1", fee_bps=0.0,
                     **rule_kwargs) -> dict:
    """Point d'entrée appelé par evaluate_daily.py : ingère les nouvelles prédictions
    live horizon=1, rafraîchit les realized_price connus, génère les nouveaux sim_trades
    et résout les 'open' devenus résolubles. Idempotent de bout en bout."""
    init_db(db_path)
    n_log = ingest_live_daily_oos_log(db_path=db_path)
    refresh_live_realized_prices(db_path=db_path)
    n_new = generate_sim_trades(db_path=db_path, rule_version=rule_version, fee_bps=fee_bps,
                                source="live", **rule_kwargs)
    n_resolved = resolve_open_sim_trades(db_path=db_path, rule_version=rule_version,
                                         fee_bps=fee_bps, **rule_kwargs)
    return {"new_log_rows": n_log, "new_trades": n_new, "resolved": n_resolved}


# ── KPIs (§9) ────────────────────────────────────────────────────────────────────

def kpi_report(db_path=DEFAULT_DB_PATH, source="oos", rule_version="bull_calm_d1",
              group_by=("asset", "model"), include_degenerate=False,
              k_values=None, m_frac=0.25, h_frac=0.50) -> list:
    """KPIs par group_by (sous-ensemble de {asset, model, regime}) — appeler avec
    group_by=() pour l'agrégat global (§11.5 Bull-Calm, les deux niveaux sont à
    reporter). Ne mélange jamais source='oos' et 'live' (paramètre obligatoire, pas de
    défaut combiné). group_by='regime' interdit pour source='oos' (regime y est
    toujours 'unknown', cf. docstring module). Les signaux non résolus (status='open')
    sont exclus de tous les KPIs, y compris n_signaux ; comptés séparément en n_open.

    Pour rule_version="bull_calm_d1"/"pi95_conf" (BRIEF_bull_calm_d1.md §9) : KPIs
    orientés ROI. Pour rule_version="sideways_d1" (BRIEF_sideways_d1.md §8) : variante
    "justesse" sans ROI (taux_justesse/immobile/breakout haussier-baissier, in_band).

    k_values : balayage de sensibilité §8.7 du brief Sideways (uniquement valide avec
    rule_version="sideways_d1"). Recalcule taux_signal/taux_justesse pour chaque k
    DIRECTEMENT depuis daily_oos_log (le k des sim_trades déjà générés est figé à la
    génération) -- purement en lecture, n'écrit jamais dans sim_trades. Résultat
    attaché à chaque groupe sous la clé 'k_sensitivity'."""
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

    init_db(db_path)
    conn = _connect(db_path)
    try:
        log_rows = conn.execute("SELECT * FROM daily_oos_log WHERE source=?", (source,)).fetchall()
        trade_rows = conn.execute(
            "SELECT * FROM sim_trades WHERE source=? AND rule_version=?", (source, rule_version)
        ).fetchall()
    finally:
        conn.close()

    def key_of(row):
        return tuple(row[c] for c in group_by)

    groups = {}
    for row in log_rows:
        groups.setdefault(key_of(row), {"log": [], "signals": [], "n_open": 0})["log"].append(row)
    for row in trade_rows:
        if not include_degenerate and row["degenerate_pi"]:
            continue
        g = groups.setdefault(key_of(row), {"log": [], "signals": [], "n_open": 0})
        if row["status"] == "closed":
            g["signals"].append(row)
        else:
            g["n_open"] += 1

    summarize_fn = _summarize_group_sideways if rule_version == "sideways_d1" else _summarize_group

    results = []
    for key, data in sorted(groups.items()):
        entry = dict(zip(group_by, key))
        entry.update(summarize_fn(data["log"], data["signals"], data["n_open"]))
        if k_values is not None:
            entry["k_sensitivity"] = _sideways_k_sweep(
                data["log"], k_values, m_frac, h_frac, include_degenerate)
        results.append(entry)
    return results


def _sideways_k_sweep(log_rows, k_values, m_frac, h_frac, include_degenerate) -> list:
    """Recalcule en mémoire, pour chaque k, le nombre de signaux et le taux de justesse
    sideways_d1 sur les lignes daily_oos_log fournies -- jamais persisté (§8.7)."""
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
            "SELECT * FROM daily_oos_log WHERE source=? AND model=? AND realized_price IS NOT NULL",
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
            "SELECT * FROM daily_oos_log WHERE source=? AND model=? AND realized_price IS NOT NULL",
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
}


def daily_detail(db_path=DEFAULT_DB_PATH, asset=None, models=None) -> list:
    """Vue jour par jour (une ligne = un jour de `daily_oos_log`) avec le(s) test case(s)
    TC1.1-TC1.5 qui ont généré un signal ce jour-là et leur `counter`, pour l'inspection
    détaillée (dashboard Run/, tableau "Test cases (D+1)"). Restreint à `horizon=1` (seul
    horizon supporté par les 5 règles, cf. docstring de module -- l'alignement D->D+1 ne
    s'applique pas au backtest D+7 rolling-origin). Un jour peut ne correspondre à aucune
    règle (`signals` vide -- flat pour les 5) ou, en bordure, à plus d'une (léger
    recouvrement Bull-Calm/Sideways documenté dans BRIEF_sideways_d1.md §0) : `signals`
    est donc une liste, jamais une valeur unique supposée."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        query = "SELECT * FROM daily_oos_log WHERE horizon = 1"
        params: list = []
        if asset is not None:
            query += " AND asset = ?"
            params.append(asset)
        if models is not None:
            query += f" AND model IN ({','.join('?' * len(models))})"
            params += list(models)
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
    p.add_argument("--ingest-oos", action="store_true", help="ingère les Run/*-D1/ dans daily_oos_log")
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
