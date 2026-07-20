"""apply_live_supersedes_oos_flag.py — script one-shot : flague les lignes OOS de
`predictions` dont la date (model, asset, horizon, frequence, horizon_type,
cutoff_date) est aussi couverte par une ligne `live` réelle.

Une fois qu'un modèle a une vraie prédiction live pour une date donnée, le backtest
OOS correspondant est redondant (même date, en double dans les agrégats/stats) --
cf. validation.tracking_db.flag_oos_superseded_by_live pour la clé de correspondance
et pourquoi `target_date` en est volontairement exclue.

Applique dans l'ordre :
  1. sauvegarde `.bak_YYYYMMDD` avant toute écriture ;
  2. `init_db()` (idempotent) ;
  3. `flag_daily_duplicates()` (dédoublonnage OOS interne, doit tourner avant --
     cf. docstring de flag_oos_superseded_by_live) ;
  4. `flag_oos_superseded_by_live()` (le flag qui nous intéresse ici) ;
  5. rapport de contrôle de non-régression.

Ne supprime et ne modifie aucune ligne existante hors de la colonne `daily_duplicate`.
Idempotent : rejouable sans effet de bord.

Usage :
    python validation/correction/apply_live_supersedes_oos_flag.py --db validation/tracking.db
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo, pour `import validation`
from validation import tracking_db as td

BUSINESS_KEY = "model, asset, horizon, frequence, horizon_type, cutoff_date"


def backup_db(db_path: str) -> str:
    bak_path = f"{db_path}.bak_{date.today().strftime('%Y%m%d')}"
    shutil.copy2(db_path, bak_path)
    return bak_path


def count_total(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()


def run_regression_checks(db_path: str, n_before: int) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        n_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_live_flagged = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='live' AND daily_duplicate=1"
        ).fetchone()[0]
        n_overlap_remaining = conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT {BUSINESS_KEY}
                FROM predictions
                WHERE daily_duplicate = 0
                GROUP BY {BUSINESS_KEY}
                HAVING COUNT(DISTINCT source) = 2
            )
        """).fetchone()[0]
        n_oos_flagged_total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='oos' AND daily_duplicate=1"
        ).fetchone()[0]

        return {
            "n_before": n_before,
            "n_after": n_after,
            "n_live_flagged": n_live_flagged,
            "n_overlap_remaining": n_overlap_remaining,
            "n_oos_flagged_total": n_oos_flagged_total,
            "ok": (
                n_after == n_before
                and n_live_flagged == 0
                and n_overlap_remaining == 0
            ),
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="validation/tracking.db")
    args = ap.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Base introuvable : {db_path}", file=sys.stderr)
        return 1

    n_before = count_total(db_path)

    bak_path = backup_db(db_path)
    print(f"Sauvegarde effectuée : {bak_path}")
    print(f"(restauration manuelle si besoin : cp {bak_path} {db_path})\n")

    td.init_db(db_path)
    n_within_oos = td.flag_daily_duplicates(db_path=db_path)
    print(f"flag_daily_duplicates()        : {n_within_oos} lignes oos (doublons internes run_id)")

    try:
        n_superseded = td.flag_oos_superseded_by_live(db_path=db_path)
    except RuntimeError as exc:
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        print(f"Aucune modification n'a été persistée pour cette étape (rollback interne). "
              f"Restaurer {bak_path} si la base est dans un état incertain.", file=sys.stderr)
        return 1

    print(f"flag_oos_superseded_by_live()  : {n_superseded} lignes oos nouvellement flaguées "
          f"(date couverte par une ligne live)\n")

    print("── Contrôles de non-régression " + "─" * 45)
    checks = run_regression_checks(db_path, n_before)
    print(f"Total lignes avant / après                    : {checks['n_before']} / {checks['n_after']} "
          f"{'OK' if checks['n_after'] == checks['n_before'] else 'MISMATCH'}")
    print(f"Lignes live flaguées (attendu 0)               : {checks['n_live_flagged']}")
    print(f"Lignes oos flaguées au total (interne + live)   : {checks['n_oos_flagged_total']}")
    print(f"Groupes live+oos encore en chevauchement        : {checks['n_overlap_remaining']} (attendu 0)")

    # rejouabilité : un second passage complet (les deux fonctions, dans le même
    # ordre) doit reproduire EXACTEMENT le même état final -- flag_daily_duplicates()
    # réinitialise tous les oos à 0 avant de rejouer sa propre logique, donc
    # flag_oos_superseded_by_live() reflague nécessairement les mêmes 223 lignes
    # comme "nouvelles" à chaque repassage : ce n'est pas une régression, c'est l'état
    # final (l'ensemble des id flagués) qui doit être stable, pas le delta.
    flagged_ids_first = {r[0] for r in sqlite3.connect(db_path).execute(
        "SELECT id FROM predictions WHERE daily_duplicate=1"
    )}
    td.flag_daily_duplicates(db_path=db_path)
    td.flag_oos_superseded_by_live(db_path=db_path)
    checks_replay = run_regression_checks(db_path, n_before)
    flagged_ids_second = {r[0] for r in sqlite3.connect(db_path).execute(
        "SELECT id FROM predictions WHERE daily_duplicate=1"
    )}
    idempotent = (
        flagged_ids_first == flagged_ids_second
        and checks_replay["n_oos_flagged_total"] == checks["n_oos_flagged_total"]
        and checks_replay["n_after"] == checks["n_after"]
    )
    print(f"Idempotence (état final identique après 2e passage) : {'OK' if idempotent else 'MISMATCH'}")

    all_ok = checks["ok"] and checks_replay["ok"] and idempotent
    print("\n" + ("TOUS LES CONTRÔLES SONT PASSÉS." if all_ok else "AU MOINS UN CONTRÔLE A ÉCHOUÉ."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
