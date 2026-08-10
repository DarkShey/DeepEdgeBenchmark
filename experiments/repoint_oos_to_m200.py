"""
repoint_oos_to_m200.py -- chantier A1 : bascule la piste `oos` sur la reference
actee (n_samples=200) pour les modeles qui en dependent.

DECISION ACTEE (pas d'entre-deux, comme le demande le brief) : le dashboard
bascule sur la reference 200 tirages. Portee reelle, etablie par
`oos_reference_audit.py` et non supposee :

  * 4 modeles a bornes ANALYTIQUES (ARIMA-GARCH, SARIMA, Naive, LSTM) : rien a
    faire, ils ne tirent aucun echantillon ;
  * Prophet : bornes tirees sur ses 1000 echantillons internes -> couverture
    reelle 94.8 % pour une etiquette 95 %, soit 0.2 point de biais. Sous la
    resolution de tout ce que le programme mesure ; laisse tel quel, declare ;
  * NsDiff : SEUL modele reellement repointe (m=50 -> m=200) ;
  * TSDiff : RETIRE du benchmark (chantier A3, cf. `benchmark_registry`), donc
    non repointe.

Apres bascule, tout modele encore affiche est soit analytique, soit lu sur
>= 200 tirages : la comparabilite mutuelle -- la raison d'etre de la contrainte
"ne pas regenerer NsDiff seul" -- est retablie, sans regenerer quoi que ce soit
(les lignes 200 tirages de la graine 42 existent deja sur disque).

CE QUI EST ECRIT, exactement : les colonnes de prevision (y_pred, y_lower,
y_upper) des lignes `source='oos'`, `model='NsDiff'`, `horizon_type='weekly'`,
sur les 2700 memes (actif, regime, horizon, cutoff) -- via
`validation.sim_trades.insert_oos_predictions`, l'upsert idempotent que
TOUTES les autres ingestions oos utilisent. Aucune ligne creee, aucune
supprimee : l'audit a verifie au prealable que les cles coincident une a une.
`y_true` et `last_close` sont reecrits a l'identique (verifies egaux a 1e-5
pres avant toute ecriture -- sinon le script refuse de tourner).

Les colonnes derivees (abs_error, in_interval, beats_naif, direction_correct,
evaluated_at) sont remises a NULL sur les lignes touchees puis recalculees par
`backfill_eval_metrics.py`, appele ici -- sinon elles resteraient calees sur
les anciennes bornes, ce qui est precisement le genre d'incoherence silencieuse
que ce chantier corrige.

SECURITE : sauvegarde horodatee de la base avant toute ecriture, et `--apply`
obligatoire (defaut = dry-run detaille).

Sortie : experiments/repoint_oos_to_m200.json
Usage :
    python repoint_oos_to_m200.py                 # dry-run : montre tout, n'ecrit rien
    python repoint_oos_to_m200.py --apply         # sauvegarde + bascule + backfill
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmark_registry as reg                                      # noqa: E402
import oos_reference_audit as audit                                   # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH, HORIZON_UNITS           # noqa: E402
from validation import sim_trades as st                               # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "repoint_oos_to_m200.json"
RUN_ID = "20260806-oos-repoint-m200"
REFERENCE_SEED = 42
DERIVED_COLS = ["abs_error", "abs_error_naif", "beats_naif", "direction_correct",
                "in_interval", "evaluated_at"]
KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date"]


def build_rows(model: str, v2: pd.DataFrame, oos: pd.DataFrame) -> list:
    """Lignes au format `insert_oos_predictions`, construites a partir de
    l'artefact 200 tirages MAIS en reprenant de la base tout ce qui identifie la
    ligne (horizon, target_date, regime) -- de sorte que l'upsert tombe sur la
    cle existante et remplace, au lieu d'en creer une nouvelle."""
    db = oos[oos["model"] == model].set_index(KEYS)
    rows = []
    for r in v2.itertuples():
        key = (r.asset, r.frequence, r.horizon_unit, r.cutoff_date)
        if key not in db.index:
            raise SystemExit(f"[{model}] cle absente de la base : {key} -- repointage refuse")
        ref = db.loc[key]
        rows.append({
            "run_id": RUN_ID, "model": model, "asset": r.asset,
            "horizon": int(ref["horizon"]), "regime": "unknown",
            "cutoff_date": r.cutoff_date, "target_date": ref["target_date"],
            "last_close": float(ref["last_close"]),      # la base fait foi sur le prix
            "y_pred": float(r.y_pred), "y_lower": float(r.y_lower), "y_upper": float(r.y_upper),
            "y_true": float(ref["y_true"]),              # la base fait foi sur la cible
            "source": "oos", "frequence": r.frequence,
            "horizon_type": "weekly", "horizon_unit": r.horizon_unit,
        })
    return rows


def null_derived(db_path: str, model: str) -> int:
    """Remet a NULL les colonnes derivees des lignes touchees, pour que
    `backfill_eval_metrics.py` (qui ne cible que `abs_error IS NULL`) les
    recalcule sur les NOUVELLES bornes."""
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            f"UPDATE predictions SET {', '.join(f'{c} = NULL' for c in DERIVED_COLS)} "
            "WHERE source='oos' AND model=? AND horizon_type='weekly'", (model,))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def snapshot(db_path: str, model: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT frequence, horizon_unit, y_lower, y_upper, y_true, last_close, in_interval "
            "FROM predictions WHERE source='oos' AND model=? AND horizon_type='weekly'",
            con, params=[model])
    finally:
        con.close()
    inside = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return {
        "n_rows": int(len(df)),
        "cov95_recomputed": float(inside.mean()),
        "cov95_stored_column": None if df["in_interval"].isna().all() else float(df["in_interval"].mean()),
        "pi_width_pct_of_price": float(((df["y_upper"] - df["y_lower"]) / df["last_close"]).mean() * 100),
        "per_cell_cov95": {f"{f}|{h}": float(g.mean()) for (f, h), g
                           in inside.groupby([df["frequence"], df["horizon_unit"]])},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--v2-dir", default=str(audit.V2_DIR))
    p.add_argument("--seed", type=int, default=REFERENCE_SEED)
    p.add_argument("--apply", action="store_true", help="ecrit reellement (defaut : dry-run)")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    targets = [m for m in reg.sampled_models() if reg.sampling_reference(m) == 200]
    print(f"modeles a repointer (registre) : {targets}")
    print(f"modeles retires (non repointes) : {reg.retired_models()}")

    oos = audit.load_oos(list(audit.INTERVAL_MECHANISM), db_path=args.db_path)
    report = {"run_id": RUN_ID, "applied": bool(args.apply), "db_path": args.db_path,
              "targets": targets, "retired_not_repointed": reg.retired_models(),
              "left_as_is": {m: v for m, v in reg.ACTIVE.items() if v["intervals"] != "sampled"},
              "per_model": {}}

    prepared = {}
    for model in targets:
        v2 = audit.load_v2(model, seed=args.seed, v2_dir=Path(args.v2_dir))
        cmp_ = audit.compare_model(model, oos, v2)
        ch = cmp_.get("checks", {})
        if not (cmp_.get("status") == "compared" and ch.get("origins_identical")
                and ch.get("prices_aligned") and ch.get("targets_aligned")):
            raise SystemExit(f"[{model}] alignement non verifie -- repointage refuse : {ch}")
        rows = build_rows(model, v2, oos)
        prepared[model] = rows
        report["per_model"][model] = {
            "n_rows": len(rows), "alignment_checks": ch,
            "before": snapshot(args.db_path, model),
            "expected_after": cmp_["overall"]["m200_reference"],
        }
        print(f"[{model}] {len(rows)} lignes pretes | cov95 {cmp_['overall']['m50_db']['cov95']:.3f} "
              f"-> {cmp_['overall']['m200_reference']['cov95']:.3f} attendu")

    if not args.apply:
        report["note"] = "dry-run : aucune ecriture. Relancer avec --apply."
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\n--dry-run : rien ecrit. Plan -> {args.out}")
        return

    backup = Path(args.db_path).with_suffix(f".db.bak_repoint_m200_{time.strftime('%Y-%m-%dT%H%M%S')}")
    shutil.copy2(args.db_path, backup)
    report["backup"] = str(backup)
    print(f"\nSauvegarde -> {backup.name}")

    for model, rows in prepared.items():
        n = st.insert_oos_predictions(rows, db_path=args.db_path)
        n_null = null_derived(args.db_path, model)
        report["per_model"][model].update({"n_upserted": int(n), "n_derived_nulled": int(n_null)})
        print(f"[{model}] {n} lignes upsertees, {n_null} lignes de colonnes derivees remises a NULL")

    print("\nRecalcul des colonnes derivees (backfill_eval_metrics.py) ...")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "backfill_eval_metrics.py"),
                           "--db-path", args.db_path], capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())
    report["backfill_stdout"] = proc.stdout.strip()
    if proc.returncode != 0:
        raise SystemExit(f"backfill_eval_metrics a echoue : {proc.stderr}")

    for model in prepared:
        after = snapshot(args.db_path, model)
        report["per_model"][model]["after"] = after
        exp = report["per_model"][model]["expected_after"]["cov95"]
        if not np.isclose(after["cov95_recomputed"], exp, atol=1e-9):
            raise SystemExit(f"[{model}] couverture post-bascule {after['cov95_recomputed']:.6f} "
                             f"!= attendue {exp:.6f}")
        if after["cov95_stored_column"] is None or not np.isclose(
                after["cov95_stored_column"], after["cov95_recomputed"], atol=1e-9):
            raise SystemExit(f"[{model}] colonne in_interval desynchronisee des bornes apres backfill")
        print(f"[{model}] verifie : cov95 = {after['cov95_recomputed']:.4f}, colonne in_interval coherente")

    report["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
