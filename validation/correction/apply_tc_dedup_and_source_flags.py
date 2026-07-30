"""apply_tc_dedup_and_source_flags.py — script one-shot de nettoyage `sim_trades`
(données des Test Cases TC1.1-TC1.5, dashboard).

Corrige deux problèmes distincts observés dans `sim_trades` :

1. Doublons par `d_date` (même journée D avec plusieurs prédictions) :
   a) Lignes invalides : le `business_lag` du pipeline (model_artifacts/pipeline.py
      _save_business_predictions) ajoute un jour CALENDAIRE (pas un jour de bourse),
      donc sur un vendredi (D) avec business_lag=1 la ligne `live` vise parfois un
      dimanche -- jour où le marché est fermé pour les actifs non-cryptos (SPY/TLT/
      ZN=F). Ces lignes sont supprimées inconditionnellement (détecté dynamiquement :
      jour de semaine du target_date jamais observé comme d_date pour cet actif).
   b) Chaîne D+1/D+2 redondante : un même D a une ligne visant D+1 (généralement
      `oos`, reconstruction backtest) et une ligne visant D+2 (généralement `live`,
      artefact business_lag). La ligne D+2 est supprimée SEULEMENT si le D+1 qu'elle
      duplique est déjà couvert par sa propre ligne D'=D+1 (sinon, la garder évite un
      trou de couverture sur ce target_date).
   c) Doublons exacts (même d_date ET même target_date, typiquement `live`+`oos` le
      jour de bascule 2026-07-06) : on ne garde que la ligne la plus récente
      (`created_at` puis `id` en cas d'égalité).

2. Réécriture du flag `source` sur les lignes survivantes : 'live' si la prédiction
   est réelle (d_date >= date de démarrage réel du modèle, cf.
   model_artifacts/generate_dashboard.py:945-949 isRealPrediction), 'oos' sinon --
   indépendamment de la valeur d'origine. La valeur d'origine est conservée dans la
   colonne `source_original` (ajoutée si absente) à des fins de traçabilité, en plus
   de la sauvegarde `.bak_YYYYMMDD` de la base entière.

Mode dry-run par défaut (aucune écriture) : rapporte ce qui serait fait. `--apply`
pour effectivement sauvegarder puis modifier la base. Idempotent (un second
`--apply` ne doit plus rien supprimer ni changer).

Usage :
    python validation/correction/apply_tc_dedup_and_source_flags.py --db validation/tracking.db          # dry-run
    python validation/correction/apply_tc_dedup_and_source_flags.py --db validation/tracking.db --apply  # applique
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

REAL_START_BY_MODEL = {"TSDiff": "2026-07-08"}
REAL_START_DEFAULT = "2026-07-06"


def is_real_prediction(model: str, d_date: str) -> bool:
    threshold = REAL_START_BY_MODEL.get(model, REAL_START_DEFAULT)
    return d_date >= threshold


def backup_db(db_path: str) -> str:
    bak_path = f"{db_path}.bak_{date.today().strftime('%Y%m%d')}"
    shutil.copy2(db_path, bak_path)
    return bak_path


def ensure_source_original_column(conn: sqlite3.Connection) -> bool:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sim_trades)")]
    if "source_original" in cols:
        return False
    conn.execute("ALTER TABLE sim_trades ADD COLUMN source_original TEXT")
    conn.execute(
        "UPDATE sim_trades SET source_original = source WHERE source_original IS NULL"
    )
    return True


def find_invalid_weekend_live_rows(conn: sqlite3.Connection) -> list:
    """Lignes `live` dont le target_date tombe un jour de semaine jamais observé
    comme d_date pour cet actif (marché fermé ce jour-là -- cible non atteignable)."""
    assets = [r[0] for r in conn.execute("SELECT DISTINCT asset FROM sim_trades")]
    observed_weekdays = {}
    for asset in assets:
        d_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT d_date FROM sim_trades WHERE asset = ?", (asset,)
        )]
        observed_weekdays[asset] = {datetime.strptime(d, "%Y-%m-%d").weekday() for d in d_dates}

    bad_ids = []
    rows = conn.execute(
        "SELECT id, asset, target_date FROM sim_trades WHERE source = 'live'"
    ).fetchall()
    for row_id, asset, target_date in rows:
        wd = datetime.strptime(target_date, "%Y-%m-%d").weekday()
        if wd not in observed_weekdays[asset]:
            bad_ids.append(row_id)
    return bad_ids


def find_exact_duplicate_groups(conn: sqlite3.Connection) -> dict:
    """Groupes (rule_version, model, asset, d_date, target_date) avec plusieurs
    lignes -- retourne {group_key: [(id, created_at), ...]} triés du plus récent
    au plus ancien (created_at DESC, id DESC)."""
    rows = conn.execute("""
        SELECT rule_version, model, asset, d_date, target_date, id, created_at
        FROM sim_trades
        ORDER BY rule_version, model, asset, d_date, target_date, created_at DESC, id DESC
    """).fetchall()
    groups = defaultdict(list)
    for rule_version, model, asset, d_date, target_date, row_id, created_at in rows:
        key = (rule_version, model, asset, d_date, target_date)
        groups[key].append((row_id, created_at))
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_chain_redundant_rows(conn: sqlite3.Connection) -> list:
    """Pour chaque (rule_version, model, asset, d_date) restant avec deux
    target_date distincts, supprime la ligne visant le target_date le plus
    éloigné SAUF si aucune autre ligne de la série ne porte d_date == target_date
    le plus proche (auquel cas la garder évite un trou de couverture)."""
    rows = conn.execute("""
        SELECT rule_version, model, asset, d_date, target_date, id
        FROM sim_trades
    """).fetchall()

    by_d = defaultdict(list)
    d_dates_present = defaultdict(set)
    for rule_version, model, asset, d_date, target_date, row_id in rows:
        series_key = (rule_version, model, asset)
        by_d[(series_key, d_date)].append((target_date, row_id))
        d_dates_present[series_key].add(d_date)

    to_delete = []
    for (series_key, d_date), entries in by_d.items():
        targets = sorted(set(t for t, _ in entries))
        if len(targets) <= 1:
            continue
        t_small, t_large = targets[0], targets[-1]
        if t_small in d_dates_present[series_key]:
            to_delete += [row_id for t, row_id in entries if t == t_large]
    return to_delete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="validation/tracking.db")
    ap.add_argument("--apply", action="store_true", help="Applique les changements (sinon dry-run)")
    args = ap.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Base introuvable : {db_path}", file=sys.stderr)
        return 1

    if args.apply:
        bak_path = backup_db(db_path)
        print(f"Sauvegarde effectuée : {bak_path}")
        print(f"(restauration manuelle si besoin : cp {bak_path} {db_path})\n")

    conn = sqlite3.connect(db_path)
    try:
        n_before = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]

        # ALTER TABLE n'est pas transactionnel sous SQLite (survit à un rollback) --
        # on ne l'exécute donc que si --apply. En dry-run on se contente de savoir si
        # elle serait ajoutée ; les colonnes lues plus bas restent celles déjà en base.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sim_trades)")]
        would_add_col = "source_original" not in cols
        if args.apply and would_add_col:
            ensure_source_original_column(conn)
        print(f"Colonne source_original {'ajoutée' if (args.apply and would_add_col) else ('déjà présente' if not would_add_col else 'serait ajoutée (dry-run)')}.")

        # Tout le reste (DELETE/UPDATE) EST transactionnel : on exécute toujours la
        # même séquence, et on ne décide qu'à la fin de commit() ou rollback(). Ainsi
        # le dry-run voit exactement l'état intermédiaire que verrait un --apply réel
        # (chaque détection tient compte des suppressions déjà faites juste avant),
        # au lieu de recalculer chaque étape sur la table pristine (source de
        # doublons de comptage entre étapes qui se chevauchent).

        # 1a. Lignes live invalides (target sur un jour de marché fermé pour l'actif)
        invalid_ids = find_invalid_weekend_live_rows(conn)
        if invalid_ids:
            conn.executemany("DELETE FROM sim_trades WHERE id = ?", [(i,) for i in invalid_ids])
        print(f"\n1a. Lignes 'live' à target invalide (jour de marché fermé) supprimées : {len(invalid_ids)}")

        # 1c. Doublons exacts (même d_date + même target_date) -- fait AVANT 1b pour
        # que la détection de chaîne D+1/D+2 ne voie plus qu'une ligne par target_date.
        exact_groups = find_exact_duplicate_groups(conn)
        exact_delete_ids = []
        for key, entries in exact_groups.items():
            exact_delete_ids += [row_id for row_id, _ in entries[1:]]  # garde entries[0] (le plus récent)
        if exact_delete_ids:
            conn.executemany("DELETE FROM sim_trades WHERE id = ?", [(i,) for i in exact_delete_ids])
        print(f"1c. Doublons exacts (même D + même target) : {len(exact_groups)} groupe(s), "
              f"{len(exact_delete_ids)} ligne(s) supprimée(s)")

        # 1b. Chaîne D+1/D+2 redondante -- calculé APRÈS 1a/1c, sur la table déjà réduite
        chain_delete_ids = find_chain_redundant_rows(conn)
        if chain_delete_ids:
            conn.executemany("DELETE FROM sim_trades WHERE id = ?", [(i,) for i in chain_delete_ids])
        print(f"1b. Lignes D+2 redondantes (D+1 déjà couvert ailleurs) supprimées : {len(chain_delete_ids)}")

        n_deleted = len(invalid_ids) + len(exact_delete_ids) + len(chain_delete_ids)

        # 2. Réécriture du flag source (vraie -> live, fausse -> oos), sur les survivants
        rows = conn.execute("SELECT id, model, d_date, source FROM sim_trades").fetchall()
        n_flag_changed = 0
        updates = []
        for row_id, model, d_date, source in rows:
            new_source = "live" if is_real_prediction(model, d_date) else "oos"
            if new_source != source:
                n_flag_changed += 1
                updates.append((new_source, row_id))
        if updates:
            conn.executemany("UPDATE sim_trades SET source = ? WHERE id = ?", updates)
        print(f"\n2. Lignes dont le flag source change : {n_flag_changed} / {len(rows)}")

        if args.apply:
            conn.commit()
        else:
            conn.rollback()

        n_after_prospective = n_before - n_deleted
        print(f"\nTotal sim_trades avant / après (réel si --apply, prospectif sinon) : "
              f"{n_before} / {n_after_prospective} "
              f"({'appliqué' if args.apply else 'DRY-RUN, base inchangée'})")

        if not args.apply:
            print("\nDry-run : aucune modification écrite. Relancer avec --apply pour appliquer.")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
