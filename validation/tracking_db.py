"""
tracking_db.py — Base de suivi des prédictions (Partie B, cf. BRIEF_tracking_db.md)

Bibliothèque standard uniquement (sqlite3, csv) : aucune dépendance externe, aucun
accès réseau. Le seul point de contact avec la partie entraînement/prédiction
(model_artifacts/pipeline.py, pipeline unique depuis la fusion Partie A + pipeline ML)
est save_prediction(record) et le contrat de champs de RECORD_FIELDS (§3 du brief).
Source unique : ce module remplace l'ancienne version partielle de validation/tracking_db.py
(save_prediction seul, sans evaluate_pending/report) suite à la consolidation Partie A + B.

Deux tables SQLite :
  - test_cases  : les 8 cas de référence (tc_id, asset, horizon, description).
  - predictions : une ligne par prédiction, contrat + champs d'évaluation remplis
    a posteriori par evaluate_pending(), avec UNIQUE(tc_id, model, cutoff_date)
    pour garantir l'idempotence (rejouer un run ne duplique rien).
"""

import csv
import sqlite3
from datetime import date

DEFAULT_DB_PATH = "tracking.db"

RECORD_FIELDS = (
    "run_id", "tc_id", "model", "asset", "horizon", "cutoff_date", "target_date",
    "regime", "last_close", "y_pred", "y_lower", "y_upper",
    "verdict_integrite", "verdict_plausibilite", "created_at",
)

EVAL_FIELDS = (
    "y_true", "in_interval", "abs_error", "abs_error_naif",
    "beats_naif", "direction_correct", "evaluated_at",
)

_GROUP_BY_COLUMNS = {"model", "asset", "horizon", "regime"}


def _connect(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH) -> None:
    """Crée les 2 tables si absentes (CREATE TABLE IF NOT EXISTS, idempotent)."""
    conn = _connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                tc_id       TEXT PRIMARY KEY,
                asset       TEXT NOT NULL,
                horizon     INTEGER NOT NULL,
                description TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                TEXT NOT NULL,
                tc_id                 TEXT NOT NULL,
                model                 TEXT NOT NULL,
                asset                 TEXT NOT NULL,
                horizon               INTEGER NOT NULL,
                cutoff_date           TEXT NOT NULL,
                target_date           TEXT NOT NULL,
                regime                TEXT NOT NULL,
                last_close            REAL NOT NULL,
                y_pred                REAL NOT NULL,
                y_lower               REAL NOT NULL,
                y_upper               REAL NOT NULL,
                verdict_integrite     INTEGER NOT NULL,
                verdict_plausibilite  INTEGER NOT NULL,
                created_at            TEXT NOT NULL,
                y_true                REAL,
                in_interval           INTEGER,
                abs_error             REAL,
                abs_error_naif        REAL,
                beats_naif            INTEGER,
                direction_correct     INTEGER,
                evaluated_at          TEXT,
                UNIQUE (tc_id, model, cutoff_date)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def register_test_case(tc_id, asset, horizon, description="", db_path=DEFAULT_DB_PATH) -> None:
    """Upsert d'un cas de référence (ON CONFLICT(tc_id) DO UPDATE)."""
    conn = _connect(db_path)
    try:
        conn.execute("""
            INSERT INTO test_cases (tc_id, asset, horizon, description)
            VALUES (:tc_id, :asset, :horizon, :description)
            ON CONFLICT(tc_id) DO UPDATE SET
                asset=excluded.asset, horizon=excluded.horizon, description=excluded.description
        """, {"tc_id": tc_id, "asset": asset, "horizon": horizon, "description": description})
        conn.commit()
    finally:
        conn.close()


def save_prediction(record: dict, db_path=DEFAULT_DB_PATH) -> bool:
    """Valide le contrat, auto-enregistre le test_case, puis INSERT OR IGNORE
    (idempotent sur tc_id/model/cutoff_date). Retourne True si insérée, False
    si doublon ignoré."""
    init_db(db_path)   # paresseux et idempotent : un run direct (ex. model_artifacts.pipeline,
                       # qui n'appelle jamais init_db lui-même) ne plante pas sur "table manquante"
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"record incomplet, champs manquants : {missing}")

    register_test_case(record["tc_id"], record["asset"], record["horizon"], db_path=db_path)

    conn = _connect(db_path)
    try:
        placeholders = ", ".join(f":{f}" for f in RECORD_FIELDS)
        columns = ", ".join(RECORD_FIELDS)
        cur = conn.execute(
            f"INSERT OR IGNORE INTO predictions ({columns}) VALUES ({placeholders})",
            {f: record[f] for f in RECORD_FIELDS},
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def evaluate_pending(price_fetcher, db_path=DEFAULT_DB_PATH, today=None) -> int:
    """Pour chaque prédiction non encore évaluée (y_true IS NULL) et échue
    (target_date <= today) : demande y_true à price_fetcher(asset, target_date).
    None -> on saute (donnée pas encore dispo, on réessaiera plus tard).
    Sinon -> calcule les métriques (§6) et met à jour la ligne.
    Retourne le nombre de prédictions effectivement évaluées."""
    init_db(db_path)
    if today is None:
        today = date.today().isoformat()

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, asset, target_date, last_close, y_pred, y_lower, y_upper "
            "FROM predictions WHERE y_true IS NULL AND target_date <= ?",
            (today,),
        ).fetchall()

        n_evaluated = 0
        for row in rows:
            y_true = price_fetcher(row["asset"], row["target_date"])
            if y_true is None:
                continue

            in_interval = 1 if row["y_lower"] <= y_true <= row["y_upper"] else 0
            abs_error = abs(y_true - row["y_pred"])
            abs_error_naif = abs(y_true - row["last_close"])
            beats_naif = 1 if abs_error <= abs_error_naif else 0
            direction_correct = 1 if _sign(row["y_pred"] - row["last_close"]) == _sign(y_true - row["last_close"]) else 0

            conn.execute("""
                UPDATE predictions SET
                    y_true=?, in_interval=?, abs_error=?, abs_error_naif=?,
                    beats_naif=?, direction_correct=?, evaluated_at=?
                WHERE id=?
            """, (y_true, in_interval, abs_error, abs_error_naif, beats_naif,
                  direction_correct, date.today().isoformat(), row["id"]))
            n_evaluated += 1

        conn.commit()
        return n_evaluated
    finally:
        conn.close()


def report(group_by=("model",), db_path=DEFAULT_DB_PATH) -> list:
    """Agrégation par group_by (sous-ensemble de {model, asset, horizon, regime}).
    Par groupe : n_total, n_evalues, taux_integrite, taux_plausibilite,
    couverture_ic95, taux_bat_naif, exactitude_dir.

    taux_integrite/taux_plausibilite portent sur TOUTES les prédictions du groupe
    (verdicts calculés par A dès la sauvegarde) ; couverture_ic95/taux_bat_naif/
    exactitude_dir ne portent que sur les prédictions évaluées (AVG() ignore les
    NULL en SQL, donc la moyenne se fait automatiquement sur les seules lignes
    évaluées, sans fausser le dénominateur avec les prédictions encore en attente).
    """
    init_db(db_path)
    invalid = set(group_by) - _GROUP_BY_COLUMNS
    if invalid:
        raise ValueError(f"group_by invalide : {invalid} (attendu parmi {_GROUP_BY_COLUMNS})")

    conn = _connect(db_path)
    try:
        group_cols = ", ".join(group_by)
        select_cols = (group_cols + ", ") if group_cols else ""
        query = f"""
            SELECT {select_cols}
                COUNT(*) AS n_total,
                SUM(CASE WHEN y_true IS NOT NULL THEN 1 ELSE 0 END) AS n_evalues,
                AVG(verdict_integrite) AS taux_integrite,
                AVG(verdict_plausibilite) AS taux_plausibilite,
                AVG(in_interval) AS couverture_ic95,
                AVG(beats_naif) AS taux_bat_naif,
                AVG(direction_correct) AS exactitude_dir
            FROM predictions
            {"GROUP BY " + group_cols if group_cols else ""}
        """
        rows = conn.execute(query).fetchall()

        results = []
        for row in rows:
            entry = {col: row[col] for col in group_by}
            entry["n_total"] = row["n_total"]
            entry["n_evalues"] = row["n_evalues"]
            for metric in ("taux_integrite", "taux_plausibilite", "couverture_ic95",
                          "taux_bat_naif", "exactitude_dir"):
                value = row[metric]
                entry[metric] = round(value, 4) if value is not None else None
            results.append(entry)
        return results
    finally:
        conn.close()


def fetch_predictions_for_run(run_id, db_path=DEFAULT_DB_PATH) -> list:
    """Toutes les lignes (contrat + évaluation) de ce run_id précis, une par
    (tc_id, model) — utilisé pour ranger les résultats de validation business dans
    les dossiers Run/<date>-<modèle>-<asset>-<horizon>/ du pipeline ML."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def pending_assets(db_path=DEFAULT_DB_PATH) -> list:
    """Actifs distincts ayant au moins une prédiction non résolue et déjà échue
    (target_date <= aujourd'hui) — utilisé par evaluate_daily.py (cron) pour savoir
    quels actifs re-télécharger avant d'appeler evaluate_pending()."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        today = date.today().isoformat()
        rows = conn.execute(
            "SELECT DISTINCT asset FROM predictions WHERE y_true IS NULL AND target_date <= ?",
            (today,),
        ).fetchall()
        return [row["asset"] for row in rows]
    finally:
        conn.close()


def run_ids_evaluated_on(day_iso, db_path=DEFAULT_DB_PATH) -> list:
    """run_id distincts ayant eu au moins une prédiction évaluée ce jour précis
    (evaluated_at = day_iso) — permet de ne rafraîchir que les bundles Run/ concernés
    après un evaluate_pending(), sans retraiter tous les runs historiques."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM predictions WHERE evaluated_at = ?", (day_iso,)
        ).fetchall()
        return [row["run_id"] for row in rows]
    finally:
        conn.close()


def export_csv(path, db_path=DEFAULT_DB_PATH) -> int:
    """Dump de la table predictions en CSV. Retourne le nombre de lignes écrites."""
    conn = _connect(db_path)
    try:
        cur = conn.execute("SELECT * FROM predictions ORDER BY id")
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(tuple(row) for row in rows)
        return len(rows)
    finally:
        conn.close()
