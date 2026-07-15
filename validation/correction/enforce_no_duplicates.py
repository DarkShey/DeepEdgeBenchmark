"""enforce_no_duplicates.py — script one-shot (BRIEF_prevention_doublons.md).

Passe la base d'une garantie *réversible* (flag `daily_duplicate`, brief précédent) à
une garantie *dure* (suppression physique + contrainte SQL) : plus aucun doublon OOS,
ni dans l'historique, ni à l'avenir.

Ordre d'exécution (§10 du brief -- **l'ordre est impératif**, ne pas réordonner) :
  1. sauvegarde `.bak_YYYYMMDD` avant toute écriture ;
  2. DELETE des lignes OOS flaguées (`daily_duplicate=1`) + contrôle bloquant : AUCUNE
     fonction `tracking_db.init_db`/`sim_trades.init_db` n'est appelée avant cette
     étape -- le code déployé pose désormais un index UNIQUE strict sur la clé métier
     OOS, qui refuserait de se (re)créer tant que des doublons physiques subsistent ;
  3. maintenant la table est propre : `sim_trades.init_db()` peut tourner en sécurité,
     ce qui (re)pose l'index (`tracking_db.init_db`, DROP+CREATE, cf. §4) et recrée la
     vue `all_predictions` ;
  4. reconstruction complète de `sim_trades` (source='oos') pour repartir propre ;
  5. contrôle anti-régression : rejoue `ingest_oos` sur les `Run/*-D1` réels (upsert
     idempotent, ne doit ajouter aucune ligne ni créer de doublon) ;
  6. affiche tous les contrôles de non-régression (§8).

Le live n'est jamais touché (toutes les opérations sont `WHERE source='oos'`).

Usage :
    python validation/correction/enforce_no_duplicates.py --db validation/tracking.db
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du repo, pour `import validation`

BUSINESS_KEY = "source, model, asset, horizon, cutoff_date"


def backup_db(db_path: str) -> str:
    bak_path = f"{db_path}.bak_{date.today().strftime('%Y%m%d')}"
    shutil.copy2(db_path, bak_path)
    return bak_path


def delete_flagged_duplicates(db_path: str) -> dict:
    """§3 du brief. Connexion sqlite3 brute UNIQUEMENT (surtout pas `td.init_db`/
    `st.init_db` avant que les doublons soient partis : le nouvel index empêcherait sa
    propre (re)création tant que des lignes dupliquées existent encore). Une seule
    transaction : rollback si le contrôle bloquant échoue. Retourne les compteurs."""
    conn = sqlite3.connect(db_path)
    try:
        n_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_to_delete = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='oos' AND daily_duplicate=1"
        ).fetchone()[0]

        conn.execute("DELETE FROM predictions WHERE source='oos' AND daily_duplicate=1")

        n_deleted = n_before - conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_dup_groups = conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM predictions WHERE source='oos'
                GROUP BY {BUSINESS_KEY}
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        if n_dup_groups != 0:
            conn.rollback()
            raise RuntimeError(
                f"enforce_no_duplicates : contrôle bloquant échoué -- {n_dup_groups} "
                "groupe(s) métier OOS encore en double après DELETE, rollback effectué. "
                "La base est inchangée."
            )

        conn.commit()
        return {
            "n_before": n_before,
            "n_to_delete": n_to_delete,
            "n_deleted": n_deleted,
            "n_dup_groups_after": n_dup_groups,
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="validation/tracking.db")
    ap.add_argument("--run-root", default="Run")
    ap.add_argument("--audit-csv", default="validation/audit/audit_keep_drop.csv")
    args = ap.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Base introuvable : {db_path}", file=sys.stderr)
        return 1

    # ── Étape 1 : sauvegarde ────────────────────────────────────────────────────
    bak_path = backup_db(db_path)
    print(f"Sauvegarde effectuée : {bak_path}")
    print(f"(restauration manuelle si besoin : cp {bak_path} {db_path})\n")

    # ── Étape 2 : DELETE des flaggés + contrôle bloquant (sqlite3 brut, cf. docstring) ──
    conn = sqlite3.connect(db_path)
    n_live_before = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='live'").fetchone()[0]
    n_sim_trades_live_before = conn.execute(
        "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
    ).fetchone()[0] if conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sim_trades'"
    ).fetchone()[0] else 0
    conn.close()

    try:
        delete_stats = delete_flagged_duplicates(db_path)
    except RuntimeError as exc:
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        return 1

    print(f"DELETE : {delete_stats['n_deleted']} ligne(s) OOS flaguée(s) supprimée(s) "
          f"(attendu 9875) -- total {delete_stats['n_before']} -> "
          f"{delete_stats['n_before'] - delete_stats['n_deleted']}")
    print(f"Contrôle bloquant (§3.2) : {delete_stats['n_dup_groups_after']} groupe(s) "
          f"métier OOS en double (attendu 0) -- OK\n")

    # ── Étape 3 : pose l'index dur + recrée la vue (sûr maintenant, table propre) ──
    from validation import tracking_db as td
    from validation import sim_trades as st

    st.init_db(db_path)   # td.init_db(db_path) + (re)création de l'index + de la vue
    print("Index OOS remplacé : (source, model, asset, horizon, cutoff_date), sans run_id.")

    # ── Étape 4 : reconstruction complète de sim_trades (source='oos') ──────────
    rebuild_stats = st.rebuild_oos_sim_trades(db_path=db_path)
    print(f"rebuild_oos_sim_trades() : {rebuild_stats['n_deleted']} sim_trade(s) OOS "
          f"supprimé(s), {rebuild_stats['n_regenerated']} régénéré(s)\n")

    # ── Étape 5 : contrôle anti-régression -- rejeu de l'ingestion réelle ───────
    # Deux appels consécutifs : le 1er peut légitimement ajouter des lignes s'il
    # existe des dossiers Run/*-D1 jamais encore ingérés (nouveaux backtests) -- ce
    # n'est pas un doublon, juste de la donnée neuve. Le vrai test d'idempotence est
    # le 2e appel, immédiatement après : à ce stade tout est déjà en base, il ne doit
    # plus rien ajouter ni modifier. Dans les deux cas, 0 groupe en double est
    # l'invariant qui compte vraiment.
    def _dup_groups(conn) -> int:
        return conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM predictions WHERE source='oos'
                GROUP BY {BUSINESS_KEY}
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

    conn = sqlite3.connect(db_path)
    n_predictions_before_replay = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    conn.close()

    replay_stats = st.ingest_oos(run_root=args.run_root, db_path=db_path)

    conn = sqlite3.connect(db_path)
    n_predictions_after_replay = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    n_dup_groups_after_replay = _dup_groups(conn)
    conn.close()

    replay_stats_2 = st.ingest_oos(run_root=args.run_root, db_path=db_path)

    conn = sqlite3.connect(db_path)
    n_predictions_after_replay_2 = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE source='oos'"
    ).fetchone()[0]
    n_dup_groups_after_replay_2 = _dup_groups(conn)
    conn.close()

    n_new_from_first_replay = n_predictions_after_replay - n_predictions_before_replay
    idempotent_on_2nd_call = n_predictions_after_replay_2 == n_predictions_after_replay
    replay_ok = (
        n_dup_groups_after_replay == 0
        and n_dup_groups_after_replay_2 == 0
        and idempotent_on_2nd_call
    )
    print(f"Rejeu ingest_oos sur {args.run_root}/*-D1 ({replay_stats['combos']} combos) : "
          f"predictions oos avant/après 1er appel = {n_predictions_before_replay} / "
          f"{n_predictions_after_replay} "
          f"({'+' + str(n_new_from_first_replay) + ' ligne(s) neuve(s), backtests jamais ingérés' if n_new_from_first_replay else 'inchangé'})")
    print(f"2e appel consécutif (vrai test idempotence) : "
          f"{n_predictions_after_replay} / {n_predictions_after_replay_2} "
          f"{'OK, rien ajouté' if idempotent_on_2nd_call else 'MISMATCH'}, "
          f"doublons créés = {n_dup_groups_after_replay}/{n_dup_groups_after_replay_2} (attendu 0/0)\n")

    # ── Étape 6 : contrôles de non-régression (§8) ──────────────────────────────
    conn = sqlite3.connect(db_path)
    n_total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_live_after = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='live'").fetchone()[0]
    n_oos_total = conn.execute("SELECT COUNT(*) FROM predictions WHERE source='oos'").fetchone()[0]
    n_sim_trades_live_after = conn.execute(
        "SELECT COUNT(*) FROM sim_trades WHERE source='live'"
    ).fetchone()[0]
    n_orphan_sim_trades = conn.execute("""
        SELECT COUNT(*) FROM sim_trades s
        WHERE s.source='oos' AND NOT EXISTS (
            SELECT 1 FROM predictions p
            WHERE p.source='oos' AND p.run_id=s.run_id AND p.model=s.model
              AND p.asset=s.asset AND p.horizon=s.horizon AND p.cutoff_date=s.d_date
        )
    """).fetchone()[0]
    index_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='idx_predictions_oos_unique'"
    ).fetchone()[0]
    conn.close()

    n_total_oos_kpi = st.kpi_report(db_path=db_path, source="oos", group_by=())[0]["n_total"]

    print("── Contrôles de non-régression (§8) " + "─" * 40)
    print(f"Total predictions : {n_total} "
          f"(= {delete_stats['n_before']} - {delete_stats['n_deleted']} avant réingestion, "
          f"+ {n_new_from_first_replay} ligne(s) neuve(s) issue(s) de backtests jamais ingérés)")
    print(f"  dont live : {n_live_after} (avant/après DELETE : {n_live_before} -> {n_live_after}, "
          f"{'OK, live jamais touché' if n_live_after == n_live_before else 'MISMATCH'})")
    print(f"  dont oos  : {n_oos_total}")
    print(f"Index idx_predictions_oos_unique porte bien run_id ? "
          f"{'NON (OK)' if 'run_id' not in index_sql else 'OUI (MISMATCH)'}")
    print(f"  -> bloque un doublon à l'insertion brute / garde le dernier via upsert : "
          f"voir test_oos_unique_index_rejects_raw_duplicate_business_key et "
          f"test_oos_unique_index_supports_upsert_keep_latest (validation/test_tracking_db.py)")
    print(f"sim_trades live : {n_sim_trades_live_before} -> {n_sim_trades_live_after} "
          f"{'OK' if n_sim_trades_live_before == n_sim_trades_live_after else 'MISMATCH'}")
    print(f"sim_trades OOS orphelins (sans predictions correspondante) : {n_orphan_sim_trades} (attendu 0)")
    print(f"kpi_report(source='oos', group_by=())[0]['n_total'] : {n_total_oos_kpi} "
          f"(cohérent avec ~4 099 survivants horizon=1)")

    all_ok = (
        delete_stats["n_deleted"] == 9875
        and delete_stats["n_dup_groups_after"] == 0
        and n_live_after == n_live_before
        and "run_id" not in index_sql
        and n_sim_trades_live_before == n_sim_trades_live_after
        and n_orphan_sim_trades == 0
        and replay_ok
    )
    print("\n" + ("TOUS LES CONTRÔLES SONT PASSÉS." if all_ok else "AU MOINS UN CONTRÔLE A ÉCHOUÉ."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
