"""apply_sim_trades_dedup.py — script one-shot (BRIEF_correction_sim_trades.md).

Propage le flag `daily_duplicate` (posé par le brief précédent, cf.
validation/correction/apply_daily_duplicate_flag.py) jusqu'à `sim_trades.py` :

  1. sauvegarde `.bak_YYYYMMDD` avant toute écriture ;
  2. `sim_trades.init_db()` (volet 1 : recrée la vue `all_predictions` filtrée
     `daily_duplicate = 0`, le prochain appel suffit -- aucune migration nécessaire) ;
  3. `sim_trades.reconcile_oos_sim_trades()` (volet 2 : suppression ciblée §4.A des
     sim_trades OOS issus de prédictions flaguées, puis régénération idempotente) ;
  4. affiche les contrôles de non-régression (§6 du brief).

Le live n'est jamais supprimé ni régénéré. Idempotent (rejouable sans effet de bord).

Usage :
    python validation/correction/apply_sim_trades_dedup.py --db validation/tracking.db
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo, pour `import validation`
from validation import sim_trades as st

DUPLICATE_SIGNAL_QUERY = """
    SELECT COUNT(*) FROM sim_trades s
    JOIN predictions p
      ON p.source='oos' AND p.run_id=s.run_id AND p.model=s.model AND p.asset=s.asset
     AND p.horizon=s.horizon AND p.cutoff_date=s.d_date
    WHERE s.source='oos' AND p.daily_duplicate=1
"""


def backup_db(db_path: str) -> str:
    bak_path = f"{db_path}.bak_{date.today().strftime('%Y%m%d')}"
    shutil.copy2(db_path, bak_path)
    return bak_path


def legacy_all_predictions_count(db_path: str, source: str) -> int:
    """Équivalent SQL de l'ANCIENNE vue all_predictions (horizon=1 seul, sans filtre
    daily_duplicate), calculé directement sur `predictions` -- utilisé UNIQUEMENT pour
    le contrôle §6 'avant/après'. On ne peut pas obtenir ce nombre en appelant une
    fonction de sim_trades.py : dès le premier appel, le self-init (init_db()) recrée
    la vue déjà corrigée par ce module, écrasant l'ancienne définition en base."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE horizon=1 AND source=?", (source,)
        ).fetchone()[0]
    finally:
        conn.close()


def run_regression_checks(db_path: str, n_sim_trades_live_before: int) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        n_live_after = conn.execute(
            "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
        ).fetchone()[0]
        n_dup_signals = conn.execute(DUPLICATE_SIGNAL_QUERY).fetchone()[0]
        n_business_key_conflicts = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT rule_version, model, asset, horizon, d_date, COUNT(DISTINCT run_id) AS n_runs
                FROM sim_trades
                WHERE source='oos'
                GROUP BY rule_version, model, asset, horizon, d_date
                HAVING n_runs > 1
            )
        """).fetchone()[0]
        return {
            "n_live_after": n_live_after,
            "live_intact": n_live_after == n_sim_trades_live_before,
            "n_dup_signals": n_dup_signals,
            "n_business_key_conflicts": n_business_key_conflicts,
            "ok": (
                n_live_after == n_sim_trades_live_before
                and n_dup_signals == 0
                and n_business_key_conflicts == 0
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

    # "Avant" capturé AVANT tout appel à sim_trades.* : dès le premier appel, le
    # self-init redéploie la vue déjà corrigée par ce module (donc plus moyen
    # d'observer le comportement pré-fix une fois le code chargé).
    n_total_oos_before = legacy_all_predictions_count(db_path, "oos")
    conn = sqlite3.connect(db_path)
    try:
        n_sim_trades_live_before = conn.execute(
            "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
        ).fetchone()[0]
        n_sim_trades_oos_before = conn.execute(
            "SELECT COUNT(*) FROM sim_trades WHERE source='oos'"
        ).fetchone()[0]
    finally:
        conn.close()

    bak_path = backup_db(db_path)
    print(f"Sauvegarde effectuée : {bak_path}")
    print(f"(restauration manuelle si besoin : cp {bak_path} {db_path})\n")

    try:
        result = st.reconcile_oos_sim_trades(db_path=db_path)
    except RuntimeError as exc:
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        print(f"Aucune modification n'a été persistée sur sim_trades (rollback interne). "
              f"La base {db_path} est inchangée.", file=sys.stderr)
        return 1

    print(f"reconcile_oos_sim_trades() : rule_versions={result['rule_versions']}, "
          f"{result['n_deleted']} sim_trade(s) OOS supprimé(s) (doublons), "
          f"{result['n_regenerated']} régénéré(s)\n")

    n_total_oos_after = st.kpi_report(db_path=db_path, source="oos", group_by=())[0]["n_total"]

    print("── Contrôles de non-régression (§6) " + "─" * 40)
    print(f"n_total OOS (KPI global) avant / après : {n_total_oos_before} / {n_total_oos_after} "
          f"{'(baisse attendue -- ~4 099 survivants horizon=1)' if n_total_oos_after < n_total_oos_before else 'MISMATCH'}")

    checks = run_regression_checks(db_path, n_sim_trades_live_before)
    print(f"sim_trades live avant / après          : {n_sim_trades_live_before} / {checks['n_live_after']} "
          f"{'OK' if checks['live_intact'] else 'MISMATCH'}")
    print(f"sim_trades oos avant réconciliation     : {n_sim_trades_oos_before}")
    print(f"Signaux OOS issus d'un doublon          : {checks['n_dup_signals']} (attendu 0)")
    print(f"Groupes métier oos avec run_id multiples : {checks['n_business_key_conflicts']} (attendu 0)")

    # rejouabilité : un second appel ne doit rien changer
    result_replay = st.reconcile_oos_sim_trades(db_path=db_path)
    checks_replay = run_regression_checks(db_path, n_sim_trades_live_before)
    idempotent = (
        result_replay["n_deleted"] == 0
        and result_replay["n_regenerated"] == 0
        and checks_replay["n_dup_signals"] == 0
        and checks_replay["n_business_key_conflicts"] == 0
        and checks_replay["live_intact"]
    )
    print(f"Idempotence (2e appel : 0 supprimé, 0 régénéré) : {'OK' if idempotent else 'MISMATCH'} "
          f"(deleted={result_replay['n_deleted']}, regenerated={result_replay['n_regenerated']})")

    all_ok = checks["ok"] and checks_replay["ok"] and idempotent and n_total_oos_after < n_total_oos_before
    print("\n" + ("TOUS LES CONTRÔLES SONT PASSÉS." if all_ok else "AU MOINS UN CONTRÔLE A ÉCHOUÉ."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
