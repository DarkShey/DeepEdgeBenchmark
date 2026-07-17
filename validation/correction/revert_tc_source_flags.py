"""revert_tc_source_flags.py — annule la réécriture du flag `source` faite par
apply_tc_dedup_and_source_flags.py (étape 2 uniquement).

Cette réécriture (oos/live -> live si vraie prédiction, -> oos si fausse) cassait
la jointure de `validation/sim_trades.py::daily_detail()` avec la table brute
`predictions` : cette jointure clé sur (source, run_id, model, asset, d_date), et
`source` y porte un sens de PROVENANCE technique (live = tourné en prod, oos =
reconstruction backtest) distinct du sens "vraie/fausse prédiction" qu'on lui a
fait porter en plus dans `sim_trades`. Quand les deux ne coïncidaient pas pour un
même run_id (une "vraie" prédiction backfillée en oos, cf. panne/backfill des 8,
11, 13, 14/07), la ligne perdait sa clé de jointure et ses signaux TC devenaient
invisibles dans le dashboard (cf. conversation -- ARIMA/BTC-USD n'affichait plus
que 2 jours avec signal au lieu de 10).

Ne touche PAS à la dédup (suppressions de lignes) faite par le même script, qui
reste valide et n'est pas concernée par ce problème -- seule la colonne `source`
des survivants est restaurée à sa valeur d'origine (`source_original`).

Usage :
    python validation/correction/revert_tc_source_flags.py --db validation/tracking.db          # dry-run
    python validation/correction/revert_tc_source_flags.py --db validation/tracking.db --apply   # applique
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def backup_db(db_path: str) -> str:
    bak_path = f"{db_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, bak_path)
    return bak_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="validation/tracking.db")
    ap.add_argument("--apply", action="store_true", help="Applique le revert (sinon dry-run)")
    args = ap.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Base introuvable : {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sim_trades)")]
        if "source_original" not in cols:
            print("Colonne source_original absente -- rien à annuler.", file=sys.stderr)
            return 1

        rows = conn.execute(
            "SELECT id, source, source_original FROM sim_trades WHERE source != source_original"
        ).fetchall()
        print(f"Lignes dont source sera restaurée à source_original : {len(rows)}")

        if args.apply and rows:
            bak_path = backup_db(db_path)
            print(f"Sauvegarde effectuée : {bak_path}")
            conn.execute("UPDATE sim_trades SET source = source_original WHERE source != source_original")
            conn.commit()
            n_remaining = conn.execute(
                "SELECT COUNT(*) FROM sim_trades WHERE source != source_original"
            ).fetchone()[0]
            print(f"Appliqué. Lignes encore désynchronisées (attendu 0) : {n_remaining}")
        elif not args.apply:
            print("Dry-run : aucune modification écrite. Relancer avec --apply pour appliquer.")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
