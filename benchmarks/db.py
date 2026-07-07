"""
db.py — Sortie SQLite du benchmark
=====================================
Stockage plat de chaque verdict (modèle x actif x horizon) pour un run donné.
SQLite (stdlib `sqlite3`) : aucune dépendance nouvelle, un seul fichier
(`benchmarks/benchmark_results.db`), suffisant pour un usage local/exploratoire.
"""

import sqlite3
from datetime import datetime

import pandas as pd

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    t_days INTEGER NOT NULL,
    data_start TEXT NOT NULL,
    data_end TEXT NOT NULL,
    train_val_split REAL NOT NULL
)
"""

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    ticker TEXT NOT NULL,
    horizon_label TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    anchor_date TEXT NOT NULL,
    target_date TEXT,
    point_forecast REAL,
    pi_lower REAL,
    pi_upper REAL,
    actual REAL,
    verdict TEXT NOT NULL,
    regime_tag TEXT,
    stress_score REAL,
    vol_bucket INTEGER,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
)
"""

RESULT_COLUMNS = [
    "run_id", "model", "ticker", "horizon_label", "horizon_days", "anchor_date",
    "target_date", "point_forecast", "pi_lower", "pi_upper", "actual", "verdict",
    "regime_tag", "stress_score", "vol_bucket",
]


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(RUNS_SCHEMA)
    conn.execute(RESULTS_SCHEMA)
    conn.commit()
    return conn


def insert_run(conn: sqlite3.Connection, t_days: int, data_start: str, data_end: str,
               train_val_split: float) -> int:
    cur = conn.execute(
        "INSERT INTO runs (timestamp, t_days, data_start, data_end, train_val_split) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), t_days, data_start, data_end,
         train_val_split),
    )
    conn.commit()
    return cur.lastrowid


def insert_results_df(conn: sqlite3.Connection, run_id: int, df: pd.DataFrame) -> None:
    """Appends the rows of `df` (one row per model x ticker x horizon) to `results`,
    stamped with `run_id`. `df` must already contain the non-run_id RESULT_COLUMNS."""
    out = df.copy()
    out.insert(0, "run_id", run_id)
    out[RESULT_COLUMNS].to_sql("results", conn, if_exists="append", index=False)
    conn.commit()


def load_results_df(conn: sqlite3.Connection, run_id: int = None) -> pd.DataFrame:
    query = "SELECT * FROM results"
    params = ()
    if run_id is not None:
        query += " WHERE run_id = ?"
        params = (run_id,)
    return pd.read_sql_query(query, conn, params=params)
