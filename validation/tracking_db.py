"""
tracking_db.py — Persistance des test cases de validation business
======================================================================
Couche de stockage pour la validation "business" des prédictions (distincte du
benchmark statistique de `benchmarks/db.py` : ici on trace des test cases
individuels — une prédiction, un instant donné — avec deux verdicts calculables
IMMÉDIATEMENT (intégrité structurelle + plausibilité métier), sans attendre le
vrai futur. `evaluate_pending()` (remplit `actual` une fois le futur connu) et
`report()` (agrégats par modèle/actif/horizon/régime) sont gérés ailleurs — ce
module ne fait QUE la persistance idempotente d'un enregistrement de prédiction.

Champs requis d'un `record` (voir REQUIRED_FIELDS) :
    run_id, tc_id, model, asset, horizon, cutoff_date, target_date, regime,
    last_close, y_pred, y_lower, y_upper, verdict_integrite, verdict_plausibilite,
    created_at

Convention tc_id recommandée : encoder actif + horizon dans l'id (ex.
"TC_BTC-USD_D1") pour que la clé de dédoublonnage (tc_id, model, cutoff_date)
identifie sans ambiguïté "quel test case x quel modèle x quel jour de calcul" —
si tc_id est un simple compteur (TC1, TC2...) partagé entre actifs, deux
prédictions différentes sur deux actifs peuvent se retrouver avec la même clé
et une des deux sera silencieusement ignorée comme "doublon". C'est la
responsabilité de l'appelant (voir generate_test_cases.py), pas de ce module.
"""

import sqlite3
from pathlib import Path

REQUIRED_FIELDS = [
    "run_id", "tc_id", "model", "asset", "horizon", "cutoff_date", "target_date",
    "regime", "last_close", "y_pred", "y_lower", "y_upper",
    "verdict_integrite", "verdict_plausibilite", "created_at",
]

# Colonnes remplies plus tard par evaluate_pending() (hors scope de ce module) :
# actual, verdict_precision (ou équivalent) — la table les prévoit dès la
# création pour qu'evaluate_pending() puisse faire un simple UPDATE.
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    tc_id TEXT NOT NULL,
    model TEXT NOT NULL,
    asset TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    cutoff_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    regime TEXT NOT NULL,
    last_close REAL NOT NULL,
    y_pred REAL NOT NULL,
    y_lower REAL NOT NULL,
    y_upper REAL NOT NULL,
    verdict_integrite INTEGER NOT NULL,
    verdict_plausibilite INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    actual REAL,
    evaluated_at TEXT,
    UNIQUE(tc_id, model, cutoff_date)
)
"""


def _validate(record: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in record or record[f] is None]
    if missing:
        raise ValueError(f"Champ(s) manquant(s) dans le record : {missing}")


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def save_prediction(record: dict, db_path: str = "tracking.db") -> bool:
    """Insère `record` dans `predictions` (table créée si absente).

    Retourne True si la ligne a été insérée, False si elle existait déjà
    (même tc_id/model/cutoff_date — doublon ignoré silencieusement, comme
    demandé). Lève ValueError si un champ de REQUIRED_FIELDS manque.
    """
    _validate(record)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            f"""INSERT OR IGNORE INTO predictions ({", ".join(REQUIRED_FIELDS)})
                VALUES ({", ".join("?" for _ in REQUIRED_FIELDS)})""",
            tuple(record[f] for f in REQUIRED_FIELDS),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()
