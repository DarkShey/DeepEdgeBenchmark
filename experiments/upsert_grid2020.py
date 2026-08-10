"""
upsert_grid2020.py -- chantier R2 du BRIEF « regeneration oos et famille 3 » :
DEPOSER EN BASE la grille regeneree, une phase a la fois.

DISCIPLINE, reprise telle quelle du standard `repoint_oos_to_m200` : dry-run par
defaut, sauvegarde horodatee avant toute ecriture, `--apply` explicite,
verification 1:1 des cles, backfill des colonnes derivees, bandeau de config
ecrit a cote. `tracking.db` reste en lecture seule partout ailleurs.

UNE PISTE NEUVE, ET C'EST UNE DECISION, PAS UNE COMMODITE. Le brief interdit tout
melange ancien/nouveau : « les verdicts issus de la grille 90 origines restent
cites comme tels ». Or la nouvelle grille n'est pas un REPOINTAGE de l'ancienne --
elle a 340 origines au lieu de 90, 7 actifs au lieu de 5, et des prix regeles dont
les NIVEAUX different pour TLT (base d'ajustement de dividendes, ratio constant
0,99598782 ; cf. R1). Ecraser la piste `oos` rendrait les anciens verdicts
inverifiables, et un upsert partiel produirait exactement le melange interdit.

La grille regeneree est donc ecrite sous `source='oos2020'`, piste distincte :
  * la piste `oos` n'est ni lue en ecriture, ni modifiee, ni supprimee -- le script
    le VERIFIE avant et apres (comptage et somme de controle) et echoue sinon ;
  * l'index d'unicite porte `source`, donc les deux pistes coexistent sans
    collision de cle ;
  * TSDiff, retire du benchmark, n'a aucune ligne dans la nouvelle piste : ses
    lignes historiques restent ou elles sont, marquees comme telles.

« VERIFICATION 1:1 DES CLES » prend ici son sens de migration : chaque ligne
d'artefact doit atterrir exactement une fois, aucune ligne hors grille declaree ne
doit apparaitre, et le compte final doit valoir exactement
origines x actifs x horizons x modeles. Tout ecart est bloquant.

Sortie : experiments/upsert_grid2020.json
Usage :
    python upsert_grid2020.py                          # dry-run detaille
    python upsert_grid2020.py --apply                  # sauvegarde + ecriture + backfill
    python upsert_grid2020.py --regime daily --apply   # phase D, une fois qu'elle est justifiee
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

import grid2020_tests as g2t                                          # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH, HORIZON_UNITS           # noqa: E402
from validation import sim_trades as st                               # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "upsert_grid2020.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"
REFS_DIR = Path(__file__).resolve().parent / "grid2020_refs"
SOURCE = "oos2020"
PROTECTED_SOURCE = "oos"
REF_MODELS = ("SARIMA", "Prophet", "Naive", "LSTM")
HORIZON_BY_UNIT = {v: k for k, v in HORIZON_UNITS.items()}
KEYS = ["model", "asset", "frequence", "horizon_unit", "cutoff_date"]
OUT_COLS = ["model", "asset", "frequence", "horizon", "horizon_unit", "cutoff_date",
            "target_date", "last_close", "y_pred", "y_lower", "y_upper", "y_true"]


def _target_dates(garch: pd.DataFrame) -> pd.DataFrame:
    """Table de reference (asset, frequence, horizon_unit, cutoff_date) ->
    target_date. Les bras NsDiff et classiques la portent aussi ; on la prend une
    seule fois pour pouvoir VERIFIER que tous s'accordent."""
    cols = ["asset", "frequence", "horizon_unit", "cutoff_date", "target_date"]
    return garch[cols].drop_duplicates()


def collect(grid_dir: Path, refs_dir: Path, regime: str, garch_arm: str) -> pd.DataFrame:
    """Toutes les lignes a ecrire, tous modeles, pour la phase demandee."""
    rows, samples, _ = g2t.load_arms(grid_dir)
    garch = pd.read_parquet(grid_dir / garch_arm / "bands.parquet")

    frames = []
    ens = g2t.nsdiff_bands(rows, samples, 0.95)
    ens["model"] = "NsDiff"
    ens["horizon"] = ens["horizon_unit"].map(HORIZON_BY_UNIT)
    frames.append(ens[OUT_COLS])

    g = garch.copy()
    g["model"] = "ARIMA-GARCH"
    frames.append(g[OUT_COLS])

    tgt = _target_dates(garch)
    for model in REF_MODELS:
        path = refs_dir / model / f"bands_{regime}.parquet"
        if not path.exists():
            raise SystemExit(f"artefact manquant : {path} -- phase incomplete, ecriture refusee")
        df = pd.read_parquet(path)
        df["model"] = model
        frames.append(df[OUT_COLS])

    out = pd.concat(frames, ignore_index=True)
    out = out[out["frequence"] == regime].reset_index(drop=True)

    # Les bras doivent voir la MEME cible a la meme origine, sinon ils ne sont pas
    # comparables et la piste ne veut rien dire.
    merged = out.merge(tgt, on=["asset", "frequence", "horizon_unit", "cutoff_date"],
                       suffixes=("", "_ref"))
    bad = merged[merged["target_date"] != merged["target_date_ref"]]
    if len(bad):
        raise SystemExit(f"{len(bad)} lignes ou la cible differe du bras de reference -- "
                         f"ecriture refusee")
    return out


def check_grid(df: pd.DataFrame) -> dict:
    """Verification 1:1 : chaque cle une seule fois, grille pleine et rectangulaire."""
    models = sorted(df["model"].unique())
    assets = sorted(df["asset"].unique())
    hus = sorted(df["horizon_unit"].unique())
    cutoffs = df["cutoff_date"].nunique()
    dup = int(df.duplicated(KEYS).sum())
    per_model = df.groupby("model").size().to_dict()
    expected_hus = sorted(HORIZON_UNITS[h] for h in (1, 2, 3))
    checks = {
        "n_rows": int(len(df)), "n_models": len(models), "models": models,
        "n_assets": len(assets), "assets": assets, "n_horizons": len(hus),
        "horizon_units": hus, "horizon_units_expected": expected_hus,
        "horizon_units_ok": hus == expected_hus,
        "horizon_column_ok": bool(df["horizon"].notna().all()
                                  and set(df["horizon"].unique()) <= {1, 2, 3}),
        "n_cutoffs": int(cutoffs), "n_duplicated_keys": dup,
        "rows_per_model": {k: int(v) for k, v in per_model.items()},
        "rows_per_model_identical": len(set(per_model.values())) == 1,
        "no_null": bool(df[["y_pred", "y_lower", "y_upper", "y_true", "last_close"]]
                        .notna().all().all()),
        "bands_ordered": bool((df["y_lower"] <= df["y_upper"]).all()),
    }
    if dup:
        raise SystemExit(f"{dup} cles en double -- ecriture refusee")
    if not (checks["horizon_units_ok"] and checks["horizon_column_ok"]):
        raise SystemExit(f"etiquetage d'horizon non conforme : {hus} (attendu {expected_hus}) "
                         f"-- ecriture refusee. Les modeles de reference etiquettent « W1 » et "
                         f"la base « W+1 » : la conversion vit dans grid2020_refs.normalise_horizon")
    if not checks["rows_per_model_identical"]:
        raise SystemExit(f"grille non rectangulaire : {per_model} -- ecriture refusee")
    if not (checks["no_null"] and checks["bands_ordered"]):
        raise SystemExit(f"lignes invalides (NULL ou bornes inversees) -- ecriture refusee")
    return checks


def protected_fingerprint(db_path: str) -> dict:
    """Empreinte de la piste qu'on ne doit PAS toucher. Relue apres ecriture."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n, s = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(y_pred + y_lower + y_upper), 0) "
            "FROM predictions WHERE source=?", (PROTECTED_SOURCE,)).fetchone()
    finally:
        con.close()
    return {"source": PROTECTED_SOURCE, "n_rows": int(n), "checksum": round(float(s), 6)}


def track_snapshot(db_path: str, source: str) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT model, frequence, horizon_unit, y_lower, y_upper, y_true, last_close, "
            "in_interval FROM predictions WHERE source=?", con, params=[source])
    finally:
        con.close()
    if df.empty:
        return {"n_rows": 0}
    inside = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    return {
        "n_rows": int(len(df)),
        "cov95_recomputed": float(inside.mean()),
        "cov95_stored_column": (None if df["in_interval"].isna().all()
                                else float(df["in_interval"].mean())),
        "per_model_cov95": {m: float(inside[df["model"] == m].mean())
                            for m in sorted(df["model"].unique())},
    }


def build_rows(df: pd.DataFrame, run_id: str) -> list:
    return [{
        "run_id": run_id, "model": r.model, "asset": r.asset, "horizon": int(r.horizon),
        "regime": "unknown", "cutoff_date": r.cutoff_date, "target_date": r.target_date,
        "last_close": float(r.last_close), "y_pred": float(r.y_pred),
        "y_lower": float(r.y_lower), "y_upper": float(r.y_upper), "y_true": float(r.y_true),
        "source": SOURCE, "frequence": r.frequence, "horizon_type": "weekly",
        "horizon_unit": r.horizon_unit,
    } for r in df.itertuples()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--refs-dir", default=str(REFS_DIR))
    p.add_argument("--regime", default="weekly", choices=["weekly", "daily"])
    p.add_argument("--garch-arm", default="ARIMA-GARCH",
                   help="bras GARCH a deposer : la config championne actee par H1")
    p.add_argument("--apply", action="store_true", help="ecrit reellement (defaut : dry-run)")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    run_id = f"{time.strftime('%Y%m%d')}-grid2020-phase{'W' if args.regime == 'weekly' else 'D'}"
    df = collect(Path(args.grid_dir), Path(args.refs_dir), args.regime, args.garch_arm)
    checks = check_grid(df)
    before_protected = protected_fingerprint(args.db_path)

    print(f"phase {'W' if args.regime == 'weekly' else 'D'} ({args.regime}) : {checks['n_rows']} "
          f"lignes | {checks['n_models']} modeles x {checks['n_assets']} actifs x "
          f"{checks['n_horizons']} horizons x {checks['n_cutoffs']} origines")
    print(f"  lignes par modele : {checks['rows_per_model']}")
    print(f"  bras GARCH depose : {args.garch_arm}")
    print(f"  piste protegee '{PROTECTED_SOURCE}' : {before_protected['n_rows']} lignes "
          f"(empreinte {before_protected['checksum']})")

    payload = {
        "scope": f"chantier R2 -- upsert de la grille regeneree, phase "
                 f"{'W' if args.regime == 'weekly' else 'D'}",
        "run_id": run_id, "applied": bool(args.apply), "db_path": args.db_path,
        "source_written": SOURCE, "source_protected": PROTECTED_SOURCE,
        "why_new_track": "la nouvelle grille n'est pas un repointage (340 origines et 7 actifs "
                         "contre 90 et 5, prix regeles) ; ecraser la piste oos rendrait les "
                         "anciens verdicts inverifiables et produirait le melange que R3 interdit",
        "garch_arm": args.garch_arm, "regime": args.regime,
        "nsdiff_config": spec.PRODUCTION_SPEC,
        "grid_checks": checks,
        "protected_before": before_protected,
        "before": track_snapshot(args.db_path, SOURCE),
    }

    if not args.apply:
        payload["note"] = "dry-run : aucune ecriture. Relancer avec --apply."
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        print(f"\n--dry-run : rien ecrit. Plan -> {args.out}")
        return

    backup = Path(args.db_path).with_suffix(
        f".db.bak_grid2020_{args.regime}_{time.strftime('%Y-%m-%dT%H%M%S')}")
    shutil.copy2(args.db_path, backup)
    payload["backup"] = str(backup)
    print(f"\nSauvegarde -> {backup.name}")

    n = st.insert_oos_predictions(build_rows(df, run_id), db_path=args.db_path)
    print(f"{n} lignes upsertees sous source='{SOURCE}'")

    proc = subprocess.run([sys.executable,
                           str(Path(__file__).resolve().parent / "backfill_eval_metrics.py"),
                           "--db-path", args.db_path, "--source", SOURCE],
                          capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        raise SystemExit(f"backfill_eval_metrics a echoue : {proc.stderr}")

    after = track_snapshot(args.db_path, SOURCE)
    after_protected = protected_fingerprint(args.db_path)
    payload.update({"n_upserted": int(n), "after": after, "protected_after": after_protected})

    if after_protected != before_protected:
        raise SystemExit(f"la piste protegee a bouge : {before_protected} -> {after_protected}")
    expected = checks["n_rows"] + payload["before"].get("n_rows", 0)
    if after["n_rows"] != expected:
        raise SystemExit(f"compte final {after['n_rows']} != attendu {expected}")
    if after["cov95_stored_column"] is None or not np.isclose(
            after["cov95_stored_column"], after["cov95_recomputed"], atol=1e-9):
        raise SystemExit("colonne in_interval desynchronisee des bornes apres backfill")
    print(f"verifie : piste protegee intacte, {after['n_rows']} lignes deposees, "
          f"Cov95 = {after['cov95_recomputed']:.4f}, colonne in_interval coherente")
    for m, c in after["per_model_cov95"].items():
        print(f"    {m:<14} Cov95 {c:.4f}")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
