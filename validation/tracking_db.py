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
  - predictions : une ligne par prédiction (live ET oos depuis BRIEF_db_unification.md,
    cf. colonne `source`), contrat + champs d'évaluation remplis a posteriori par
    evaluate_pending(), avec UNIQUE(tc_id, model, cutoff_date) pour garantir
    l'idempotence des insertions live (rejouer un run ne duplique rien).

`source` (TEXT, défaut 'live') distingue les prédictions live (pipeline.py, via
save_prediction) des prédictions OOS (backtest historique, ingérées par
validation/sim_trades.py depuis Run/*-D1/predictions.parquet). RECORD_FIELDS et
save_prediction restent inchangés : la colonne omise de l'INSERT prend son DEFAULT
'live' automatiquement -- model_artifacts/pipeline.py n'a donc rien à changer.
Idempotence des lignes OOS (tc_id NULL, donc hors du UNIQUE ci-dessus puisque SQLite
traite chaque NULL comme distinct) : index partiel séparé, cf. init_db(). Clé métier
OOS (BRIEF_prevention_doublons.md) : `(source, model, asset, horizon, cutoff_date)`,
SANS `run_id` -- `run_id` n'est plus qu'une métadonnée de provenance (le run qui a
produit la ligne), pas une composante d'identité de la prédiction. Deux backtests sur
la même date ne peuvent donc plus jamais cohabiter : le second ÉCRASE le premier
(upsert « garde le dernier », cf. sim_trades.insert_oos_predictions), zéro doublon
possible par construction plutôt que par nettoyage a posteriori.

Toute requête "live" (résolution cron, export business_validation.json, rapport de
suivi) doit filtrer `source='live'` pour ignorer les lignes OOS qui partagent
désormais la même table (cf. BRIEF_db_unification.md §3).
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
    """Crée les tables si absentes (CREATE TABLE IF NOT EXISTS, idempotent). Si
    `predictions` existe déjà mais sans la colonne `source` (base pré-unification),
    migre en place (cf. _migrate_predictions_add_source). Si `daily_duplicate` est
    absente (base pré-BRIEF_correction_doublons.md), l'ajoute (cf.
    _migrate_predictions_add_daily_duplicate)."""
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
                tc_id                 TEXT,
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
                verdict_integrite     INTEGER,
                verdict_plausibilite  INTEGER,
                created_at            TEXT,
                y_true                REAL,
                in_interval           INTEGER,
                abs_error             REAL,
                abs_error_naif        REAL,
                beats_naif            INTEGER,
                direction_correct     INTEGER,
                evaluated_at          TEXT,
                source                TEXT NOT NULL DEFAULT 'live',
                daily_duplicate       INTEGER NOT NULL DEFAULT 0,
                UNIQUE (tc_id, model, cutoff_date)
            )
        """)
        _migrate_predictions_add_source(conn)
        _migrate_predictions_add_daily_duplicate(conn)
        _migrate_predictions_add_frequency_horizon(conn)
        # Idempotence des lignes OOS : tc_id y est NULL, donc hors de portée du
        # UNIQUE(tc_id, model, cutoff_date) ci-dessus (SQLite traite chaque NULL comme
        # distinct). Index partiel dédié, n'affecte jamais les lignes source='live'.
        #
        # BRIEF_prevention_doublons.md §4 : `run_id` retiré de la clé -- l'ancien index
        # (source, run_id, model, asset, horizon, cutoff_date) laissait deux backtests
        # sur les mêmes dates s'empiler (run_id différent à chaque rejeu = jamais de
        # collision). La clé métier réelle d'une prédiction OOS ne dépend pas du run qui
        # l'a produite : (source, model, asset, horizon, cutoff_date) la rend unique par
        # construction, plus aucun doublon possible. DROP puis CREATE IF NOT EXISTS :
        # idempotent, se rejoue sans erreur qu'on parte de l'ancien ou du nouvel index.
        #
        # BRIEF_audit_combinaisons.md : `frequence`/`horizon_type` ajoutés à la clé --
        # sans eux, une prédiction TSDiff-D quotidienne (frequence=daily,
        # horizon_type=daily, horizon=1 = "D+1") et une prédiction TSDiff-W hebdo
        # (frequence=weekly, horizon_type=weekly, horizon=1 = "W+1") sur le même actif
        # au même cutoff_date collisionneraient sur l'ancienne clé (source, model,
        # asset, horizon, cutoff_date) -- silencieusement ignorées ou écrasées selon
        # l'appelant (cf. insert_oos_predictions, ON CONFLICT DO UPDATE).
        conn.execute("DROP INDEX IF EXISTS idx_predictions_oos_unique")
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_oos_unique
            ON predictions (source, model, asset, horizon, frequence, horizon_type, cutoff_date)
            WHERE source = 'oos'
        """)
        conn.commit()
    finally:
        conn.close()


_PREDICTIONS_LEGACY_COLUMNS = (
    "id", "run_id", "tc_id", "model", "asset", "horizon", "cutoff_date", "target_date",
    "regime", "last_close", "y_pred", "y_lower", "y_upper", "verdict_integrite",
    "verdict_plausibilite", "created_at", "y_true", "in_interval", "abs_error",
    "abs_error_naif", "beats_naif", "direction_correct", "evaluated_at",
)


def _migrate_predictions_add_source(conn) -> None:
    """Reconstruction ponctuelle de `predictions` pour une base créée avant
    BRIEF_db_unification.md (pas de colonne `source`, tc_id/verdict_integrite/
    verdict_plausibilite/created_at encore NOT NULL) : ALTER TABLE ADD COLUMN seul ne
    peut ajouter que `source` mais ne peut pas relâcher un NOT NULL déjà posé sur les
    quatre autres colonnes -- SQLite ne supporte pas ALTER COLUMN DROP NOT NULL, d'où
    la reconstruction (CREATE + INSERT SELECT + DROP + RENAME), dans la transaction
    déjà ouverte par init_db() (rollback automatique si un COMMIT n'a jamais lieu).

    `id` est réinséré explicitement (pas de renumérotation) et le compteur
    AUTOINCREMENT (sqlite_sequence) est restauré à sa valeur historique -- jamais
    recalculé depuis MAX(id), qui peut être inférieur au compteur réel si des lignes
    ont été supprimées entre-temps (constaté : seq=392 pour 200 lignes restantes sur
    la base réelle). Vérifie ensuite que le nombre de lignes ET le contenu de CHAQUE
    ligne (pas un échantillon) sont strictement identiques avant/après ; lève sinon."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "source" in cols or not cols:
        return   # déjà migrée, ou table qui vient d'être créée fraîche (rien à migrer)

    col_list = ", ".join(_PREDICTIONS_LEGACY_COLUMNS)
    n_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows_before = conn.execute(
        f"SELECT {col_list} FROM predictions ORDER BY id"
    ).fetchall()
    rows_before = [tuple(row) for row in rows_before]
    seq_row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='predictions'").fetchone()
    old_seq = seq_row["seq"] if seq_row else None

    conn.execute("""
        CREATE TABLE predictions_new (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                TEXT NOT NULL,
            tc_id                 TEXT,
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
            verdict_integrite     INTEGER,
            verdict_plausibilite  INTEGER,
            created_at            TEXT,
            y_true                REAL,
            in_interval           INTEGER,
            abs_error             REAL,
            abs_error_naif        REAL,
            beats_naif            INTEGER,
            direction_correct     INTEGER,
            evaluated_at          TEXT,
            source                TEXT NOT NULL DEFAULT 'live',
            daily_duplicate       INTEGER NOT NULL DEFAULT 0,
            UNIQUE (tc_id, model, cutoff_date)
        )
    """)
    conn.execute(f"""
        INSERT INTO predictions_new ({col_list}, source)
        SELECT {col_list}, 'live' FROM predictions
    """)
    conn.execute("DROP TABLE predictions")
    conn.execute("ALTER TABLE predictions_new RENAME TO predictions")

    # ALTER TABLE RENAME met à jour le nom dans sqlite_sequence mais PAS la valeur de
    # seq (vérifié empiriquement : elle retombe au MAX(id) réellement inséré) -- il
    # faut la réécrire explicitement à sa valeur historique.
    if old_seq is not None:
        cur = conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name='predictions'", (old_seq,))
        if cur.rowcount == 0:
            conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('predictions', ?)", (old_seq,))

    n_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rows_after = conn.execute(
        f"SELECT {col_list} FROM predictions ORDER BY id"
    ).fetchall()
    rows_after = [tuple(row) for row in rows_after]
    if n_after != n_before or rows_after != rows_before:
        raise RuntimeError(
            "Migration predictions.source a corrompu la table : "
            f"n_before={n_before} n_after={n_after} "
            f"contenu_identique={rows_after == rows_before}"
        )


def _migrate_predictions_add_daily_duplicate(conn) -> None:
    """Ajout idempotent de `daily_duplicate` (BRIEF_correction_doublons.md §4.1) sur une
    base existante qui a déjà la colonne `source` (donc pas concernée par la
    reconstruction de _migrate_predictions_add_source, qui ne s'exécute que si
    `source` est absente) : ALTER TABLE ADD COLUMN suffit ici, SQLite l'autorise pour
    une colonne NOT NULL avec DEFAULT constant. Sans effet si la colonne existe déjà."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "daily_duplicate" in cols or not cols:
        return
    conn.execute("ALTER TABLE predictions ADD COLUMN daily_duplicate INTEGER NOT NULL DEFAULT 0")


def _migrate_predictions_add_frequency_horizon(conn) -> None:
    """Ajout idempotent de `frequence`/`horizon_type`/`horizon_unit` (BRIEF_audit_combinaisons.md
    §0) sur une base existante : trois attributs qui distinguent COMMENT une prédiction
    a été produite, jusqu'ici confondus dans la seule colonne `horizon` (INTEGER, un
    nombre de jours) :
      - `frequence`    : granularité d'ENTRAINEMENT du modèle ('daily' / 'weekly').
      - `horizon_type` : granularité de la CIBLE visée ('daily' / 'weekly') -- distincte
                         de `frequence` : un modèle entraîné en daily peut viser un
                         horizon weekly (regime B, ex. TSDiff-D multi-pas).
      - `horizon_unit` : le pas précis en toutes lettres ('D+1', 'D+7', 'W+1', 'W+2', 'W+3').

    `horizon` (INTEGER) N'EST PAS touché -- toutes les requêtes/le code existants qui le
    lisent comme un nombre de jours continuent de fonctionner sans changement.

    `frequence`/`horizon_type` : ALTER TABLE ADD COLUMN avec DEFAULT constant 'daily'
    (comme `daily_duplicate`) -- toutes les lignes existantes sont réellement du daily
    natif, donc ce défaut est correct pour elles, pas juste un remplissage arbitraire.
    `horizon_unit` n'a pas de defaut constant valide (dépend de `horizon`) : colonne
    nullable ajoutée puis backfillée explicitement pour les lignes existantes."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    if not cols:
        return   # table qui vient d'être créée fraîche (CREATE TABLE ci-dessus les a déjà)
    if "frequence" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN frequence TEXT NOT NULL DEFAULT 'daily'")
    if "horizon_type" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN horizon_type TEXT NOT NULL DEFAULT 'daily'")
    if "horizon_unit" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN horizon_unit TEXT")
    conn.execute("""
        UPDATE predictions SET horizon_unit = 'D+' || horizon
        WHERE horizon_unit IS NULL AND horizon_type = 'daily'
    """)
    conn.execute("""
        UPDATE predictions SET horizon_unit = 'W+' || horizon
        WHERE horizon_unit IS NULL AND horizon_type = 'weekly'
    """)


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
    si doublon ignoré.

    `frequence`/`horizon_type`/`horizon_unit` (BRIEF_audit_combinaisons.md) sont
    volontairement ABSENTS de RECORD_FIELDS : model_artifacts/pipeline.py (le seul
    appelant live actuel) ne les fournit pas et n'a pas à changer. S'ils sont absents
    du record, on applique le défaut 'daily natif' -- correct pour 100% des appelants
    actuels, qui produisent tous du daily. Un futur appelant weekly les passe
    explicitement dans `record` et ils sont respectés tels quels."""
    init_db(db_path)   # paresseux et idempotent : un run direct (ex. model_artifacts.pipeline,
                       # qui n'appelle jamais init_db lui-même) ne plante pas sur "table manquante"
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"record incomplet, champs manquants : {missing}")

    register_test_case(record["tc_id"], record["asset"], record["horizon"], db_path=db_path)

    frequence = record.get("frequence", "daily")
    horizon_type = record.get("horizon_type", "daily")
    horizon_unit = record.get("horizon_unit") or (
        f"{'W' if horizon_type == 'weekly' else 'D'}+{record['horizon']}")

    conn = _connect(db_path)
    try:
        insert_fields = RECORD_FIELDS + ("frequence", "horizon_type", "horizon_unit")
        placeholders = ", ".join(f":{f}" for f in insert_fields)
        columns = ", ".join(insert_fields)
        params = {f: record[f] for f in RECORD_FIELDS}
        params.update(frequence=frequence, horizon_type=horizon_type, horizon_unit=horizon_unit)
        cur = conn.execute(
            f"INSERT OR IGNORE INTO predictions ({columns}) VALUES ({placeholders})",
            params,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def flag_daily_duplicates(db_path=DEFAULT_DB_PATH) -> int:
    """Marque daily_duplicate=1 sur toutes les copies OOS sauf le dernier lancement
    de chaque prédiction (source, model, asset, horizon, cutoff_date, target_date).
    Idempotent (repart de 0 sur les lignes oos à chaque appel, cf. BRIEF_correction_doublons.md
    §4.2). Ne touche jamais les lignes source='live' (0 doublon, déjà protégées par
    UNIQUE(tc_id, model, cutoff_date)). Ne supprime aucune ligne.

    Le survivant de chaque groupe (daily_duplicate=0) est déterminé par
    ORDER BY run_id DESC, created_at DESC, id DESC : au sein d'un groupe, model/asset/
    horizon/cutoff_date/target_date sont fixes, donc le run_id oos ne varie que par son
    préfixe YYYYMMDD -- le tri lexicographique sur la chaîne complète est donc déjà un
    tri chronologique correct, pas besoin d'extraire le préfixe.

    Les deux UPDATE (reset puis flag) et les contrôles internes (§6) tournent dans la
    transaction implicite d'une connexion unique : si un contrôle échoue, rollback
    complet et exception, rien n'est persisté. Retourne le nombre de lignes passées à 1.

    Précondition : la colonne `daily_duplicate` doit déjà exister (appeler init_db()
    avant, comme le fait le script one-shot §5) -- volontairement pas de lazy-init ici
    pour rester une pure fonction de mise à jour, symétrique du bloc SQL du brief."""
    conn = _connect(db_path)
    try:
        n_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        conn.execute("UPDATE predictions SET daily_duplicate = 0 WHERE source = 'oos'")
        conn.execute("""
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY source, model, asset, horizon, cutoff_date, target_date
                           ORDER BY run_id DESC, created_at DESC, id DESC
                       ) AS rn
                FROM predictions
                WHERE source = 'oos'
            )
            UPDATE predictions
            SET daily_duplicate = 1
            WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
        """)
        # cur.rowcount vaut -1 ici : le driver sqlite3 de Python ne sait déterminer le
        # rowcount que pour un UPDATE/DELETE/INSERT dont le texte SQL commence
        # directement par ce mot-clé -- un UPDATE précédé d'un WITH (CTE) n'est pas
        # reconnu et rowcount reste à sa valeur par défaut. On recompte donc directement.
        n_flagged = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='oos' AND daily_duplicate=1"
        ).fetchone()[0]

        n_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_live_flagged = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='live' AND daily_duplicate=1"
        ).fetchone()[0]
        n_bad_groups = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM predictions WHERE source='oos' AND daily_duplicate=0
                GROUP BY source, model, asset, horizon, cutoff_date, target_date
                HAVING COUNT(*) <> 1
            )
        """).fetchone()[0]

        if n_after != n_before or n_live_flagged != 0 or n_bad_groups != 0:
            conn.rollback()
            raise RuntimeError(
                "flag_daily_duplicates : contrôle interne échoué, rollback effectué "
                f"(n_before={n_before} n_after={n_after} "
                f"live_flaggees={n_live_flagged} groupes_oos_invalides={n_bad_groups})"
            )

        conn.commit()
        return n_flagged
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
            "FROM predictions WHERE source='live' AND y_true IS NULL AND target_date <= ?",
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

    Filtré sur source='live' (BRIEF_db_unification.md §3) : ce rapport de suivi
    business ne doit jamais mélanger les lignes OOS du backtest historique.
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
            WHERE source='live'
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
    les dossiers Run/<date>-<modèle>-<asset>-<horizon>/ du pipeline ML. Filtré sur
    source='live' (BRIEF_db_unification.md §3) : un run OOS ne doit jamais être
    exporté comme prédiction business, même si (en pratique) les run_id live et oos
    ont des formats disjoints."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE run_id = ? AND source='live' ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def pending_assets(db_path=DEFAULT_DB_PATH) -> list:
    """Actifs distincts ayant au moins une prédiction non résolue et déjà échue
    (target_date <= aujourd'hui) — utilisé par evaluate_daily.py (cron) pour savoir
    quels actifs re-télécharger avant d'appeler evaluate_pending(). Filtré sur
    source='live' : l'OOS est déjà entièrement résolu par construction (backtest
    historique), le cron ne doit jamais tenter de le "compléter"."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        today = date.today().isoformat()
        rows = conn.execute(
            "SELECT DISTINCT asset FROM predictions "
            "WHERE source='live' AND y_true IS NULL AND target_date <= ?",
            (today,),
        ).fetchall()
        return [row["asset"] for row in rows]
    finally:
        conn.close()


def run_ids_evaluated_on(day_iso, db_path=DEFAULT_DB_PATH) -> list:
    """run_id distincts ayant eu au moins une prédiction évaluée ce jour précis
    (evaluated_at = day_iso) — permet de ne rafraîchir que les bundles Run/ concernés
    après un evaluate_pending(), sans retraiter tous les runs historiques. Filtre
    source='live' redondant avec evaluate_pending (seul writer de evaluated_at, déjà
    filtré) mais explicité par défense en profondeur (BRIEF_db_unification.md §3)."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM predictions WHERE source='live' AND evaluated_at = ?",
            (day_iso,),
        ).fetchall()
        return [row["run_id"] for row in rows]
    finally:
        conn.close()


def export_csv(path, db_path=DEFAULT_DB_PATH, source="live") -> int:
    """Dump de la table predictions en CSV. Retourne le nombre de lignes écrites.

    `source="live"` par défaut (comportement inchangé de tracking_export.csv avant
    BRIEF_db_unification.md, qui ne connaissait que des lignes live) : ne pas changer
    silencieusement ce que produit un export existant du seul fait que la table
    contient désormais aussi de l'OOS. `source=None` exporte tout (live + oos, avec
    la colonne `source` pour les distinguer) ; toute autre valeur filtre sur cette
    source précise.

    Dès que source != 'live' (donc source=None ou source='oos'), les lignes oos
    flaguées daily_duplicate=1 sont exclues (BRIEF_correction_doublons.md §4.3) pour
    qu'une même prédiction métier ne soit comptée qu'une fois dans les
    stats/exports OOS ; les lignes live ne sont jamais filtrées sur ce critère
    (0 doublon par construction, cf. UNIQUE(tc_id, model, cutoff_date))."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        if source is None:
            cur = conn.execute(
                "SELECT * FROM predictions WHERE source='live' OR daily_duplicate=0 ORDER BY id"
            )
        elif source == "live":
            cur = conn.execute("SELECT * FROM predictions WHERE source=? ORDER BY id", (source,))
        else:
            cur = conn.execute(
                "SELECT * FROM predictions WHERE source=? AND daily_duplicate=0 ORDER BY id",
                (source,),
            )
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(tuple(row) for row in rows)
        return len(rows)
    finally:
        conn.close()
