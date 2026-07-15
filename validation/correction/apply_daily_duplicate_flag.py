"""apply_daily_duplicate_flag.py — script one-shot (BRIEF_correction_doublons.md).

Applique le dédoublonnage OOS par flag `daily_duplicate` sur une base tracking.db :
  1. sauvegarde `.bak_YYYYMMDD` avant toute écriture ;
  2. appelle `tracking_db.init_db()` (ajoute la colonne si absente) PUIS
     `tracking_db.flag_daily_duplicates()` (met à jour les flags dans sa propre
     transaction -- rollback interne si un contrôle échoue, cf. tracking_db.py) ;
  3. affiche le rapport de contrôle de non-régression (§6 du brief).

Ne supprime et ne modifie aucune ligne existante hors de la colonne `daily_duplicate`.
Idempotent : rejouable sans effet de bord (flag_daily_duplicates repart de 0 sur les
lignes oos à chaque appel).

Usage :
    python validation/correction/apply_daily_duplicate_flag.py \
        --db validation/tracking.db \
        --audit-csv validation/audit/audit_keep_drop.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo, pour `import validation`
from validation import tracking_db as td

BUSINESS_KEY = "source, model, asset, horizon, cutoff_date, target_date"


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
    """§6 du brief : renvoie les compteurs + un flag `ok` global."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        n_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_flagged = conn.execute("SELECT COUNT(*) FROM predictions WHERE daily_duplicate=1").fetchone()[0]
        n_kept = conn.execute("SELECT COUNT(*) FROM predictions WHERE daily_duplicate=0").fetchone()[0]
        n_bad_groups = conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM predictions WHERE source='oos' AND daily_duplicate=0
                GROUP BY {BUSINESS_KEY}
                HAVING COUNT(*) <> 1
            )
        """).fetchone()[0]
        n_live_total = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='live'").fetchone()[0]
        n_live_flagged = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='live' AND daily_duplicate=1"
        ).fetchone()[0]
        n_oos_total = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]

        return {
            "n_before": n_before,
            "n_after": n_after,
            "n_flagged": n_flagged,
            "n_kept": n_kept,
            "n_bad_groups": n_bad_groups,
            "n_live_total": n_live_total,
            "n_live_flagged": n_live_flagged,
            "n_oos_total": n_oos_total,
            "ok": (
                n_after == n_before
                and n_bad_groups == 0
                and n_live_flagged == 0
                and n_flagged + n_kept == n_after
            ),
        }
    finally:
        conn.close()


def cross_check_with_audit_csv(db_path: str, csv_path: str) -> bool | None:
    """§6 dernier point : les id flaggés doivent correspondre à decision='drop' dans
    l'audit dry-run (validation/audit/audit_keep_drop.csv). Retourne None si le CSV
    n'est pas disponible (audit non exécuté / chemin différent), True/False sinon."""
    if not Path(csv_path).exists():
        print(f"(recoupement ignoré : {csv_path} introuvable)")
        return None

    audit_drop_ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["decision"] == "drop":
                audit_drop_ids.add(int(row["id"]))

    conn = sqlite3.connect(db_path)
    try:
        flagged_ids = {r[0] for r in conn.execute(
            "SELECT id FROM predictions WHERE daily_duplicate=1"
        ).fetchall()}
    finally:
        conn.close()

    match = flagged_ids == audit_drop_ids
    print(f"Recoupement avec {csv_path} (decision='drop') : "
          f"{'IDENTIQUE' if match else 'DIVERGENT'} "
          f"({len(flagged_ids)} id flaggés en base vs {len(audit_drop_ids)} 'drop' dans l'audit)")
    if not match:
        only_db = flagged_ids - audit_drop_ids
        only_csv = audit_drop_ids - flagged_ids
        print(f"  id flaggés en base mais absents de l'audit ('drop') : {sorted(only_db)[:10]}"
              f"{' ...' if len(only_db) > 10 else ''} (total {len(only_db)})")
        print(f"  id 'drop' dans l'audit mais non flaggés en base     : {sorted(only_csv)[:10]}"
              f"{' ...' if len(only_csv) > 10 else ''} (total {len(only_csv)})")
    return match


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="validation/tracking.db")
    ap.add_argument("--audit-csv", default="validation/audit/audit_keep_drop.csv")
    args = ap.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Base introuvable : {db_path}", file=sys.stderr)
        return 1

    n_before = count_total(db_path)

    bak_path = backup_db(db_path)
    print(f"Sauvegarde effectuée : {bak_path}")
    print(f"(restauration manuelle si besoin : cp {bak_path} {db_path})\n")

    td.init_db(db_path)   # ajoute la colonne daily_duplicate si absente (idempotent)
    try:
        n_flagged_ret = td.flag_daily_duplicates(db_path=db_path)
    except RuntimeError as exc:
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        print(f"Aucune modification n'a été persistée (rollback interne). "
              f"La base {db_path} est inchangée.", file=sys.stderr)
        return 1

    print(f"flag_daily_duplicates() : {n_flagged_ret} lignes oos passées à daily_duplicate=1\n")

    print("── Contrôles de non-régression (§6) " + "─" * 40)
    checks = run_regression_checks(db_path, n_before)
    print(f"Total lignes avant / après       : {checks['n_before']} / {checks['n_after']} "
          f"{'OK' if checks['n_after'] == checks['n_before'] else 'MISMATCH'}")
    print(f"daily_duplicate = 1 (flaggées)   : {checks['n_flagged']}")
    print(f"daily_duplicate = 0 (survivants) : {checks['n_kept']}")
    print(f"  dont live                      : {checks['n_live_total']} "
          f"(live flaggées : {checks['n_live_flagged']}, attendu 0)")
    print(f"  dont oos                       : {checks['n_oos_total']}")
    print(f"Groupes oos avec un nombre de survivants != 1 : {checks['n_bad_groups']} (attendu 0)")

    # rejouabilité : un second appel ne doit rien changer
    n_flagged_replay = td.flag_daily_duplicates(db_path=db_path)
    checks_replay = run_regression_checks(db_path, n_before)
    idempotent = (
        n_flagged_replay == n_flagged_ret
        and checks_replay["n_flagged"] == checks["n_flagged"]
        and checks_replay["n_kept"] == checks["n_kept"]
        and checks_replay["n_after"] == checks["n_after"]
    )
    print(f"Idempotence (2e appel identique) : {'OK' if idempotent else 'MISMATCH'} "
          f"({n_flagged_replay} lignes flaggées au 2e appel)")

    cross_check_ok = cross_check_with_audit_csv(db_path, args.audit_csv)

    all_ok = checks["ok"] and checks_replay["ok"] and idempotent and cross_check_ok is not False
    print("\n" + ("TOUS LES CONTRÔLES SONT PASSÉS." if all_ok else "AU MOINS UN CONTRÔLE A ÉCHOUÉ."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
