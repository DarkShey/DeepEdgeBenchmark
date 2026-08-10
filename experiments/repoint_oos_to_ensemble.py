"""
repoint_oos_to_ensemble.py -- chantier C du BRIEF "NsDiff : extension de
donnees, puissance, dashboard, re-jugement mensuel" : basculer la piste `oos` du
dashboard sur la CONFIGURATION PRODUCTION (ensemble 5 graines x 200 tirages).

POURQUOI, en une phrase : le balayage `nsamples_sweep` a montre que la piste
actuelle du dashboard -- graine 42, 200 tirages -- N'EST PAS CONVERGEE (le regime
weekly exige 800 tirages par graine aux trois horizons), alors que l'ensemble
5x200 = 1000 l'est sur 6 cellules sur 6. Le dashboard affiche donc aujourd'hui
une configuration que le programme sait sous-convergee, et qui n'est pas celle
qu'on deploierait.

CE QUI EST ECRIT : les colonnes de prevision (`y_pred`, `y_lower`, `y_upper`) des
2 700 lignes `source='oos'`, `model='NsDiff'`, `horizon_type='weekly'`, remplacees
par celles de l'ensemble. Aucune ligne creee, aucune supprimee : l'alignement des
cles est verifie une a une AVANT toute ecriture, et le script refuse de tourner
si une seule ne correspond pas. `y_true` et `last_close` sont reecrits a
l'identique depuis la base (elle fait foi sur le prix et la cible).

CE QUI CHANGE POUR LE LECTEUR DU DASHBOARD, et qui doit etre dit : la piste
`oos` cesse d'etre « un run », elle devient « la configuration production ». Les
deux ne repondent pas a la meme question. Un run a graine unique repond a
« qu'obtient-on en tirant une graine au hasard ? » ; l'ensemble repond a
« qu'obtient-on en deployant les cinq ? ». C'est la seconde question qui
interesse une decision, et c'est celle que le reste du programme teste depuis le
chantier A2. Le bandeau de configuration du dashboard porte desormais
`seeds: [42..46]` et `n_samples_effective: 1000` pour NsDiff.

CONSEQUENCE ATTENDUE ET DECLAREE : la couverture de NsDiff va MONTER (le melange
des 5 lois predictives est plus large que chacune), et le Winkler bouger dans les
deux sens selon les cellules. Ce n'est pas une amelioration du modele, c'est un
changement de configuration -- la note doit le presenter comme tel.

DISCIPLINE, identique a `repoint_oos_to_m200.py` : dry-run par defaut, sauvegarde
horodatee avant toute ecriture, `--apply` explicite, colonnes derivees remises a
NULL puis recalculees par `backfill_eval_metrics.py`.

Sortie : experiments/repoint_oos_to_ensemble.json
Usage :
    python repoint_oos_to_ensemble.py                 # dry-run detaille
    python repoint_oos_to_ensemble.py --apply         # sauvegarde + bascule + backfill
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

import dashboard_d7_w1 as dash                                        # noqa: E402
import nsdiff_production_spec as spec                                 # noqa: E402
import nsdiff_v2_data as v2                                           # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                          # noqa: E402
from nsdiff_seed_ensemble import build_ensemble_rows                  # noqa: E402
from repoint_oos_to_m200 import DERIVED_COLS, null_derived, snapshot   # noqa: E402
from validation import sim_trades as st                               # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "repoint_oos_to_ensemble.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2" / "NsDiff"
RUN_ID = "20260808-oos-repoint-ensemble"
MODEL = "NsDiff"
PRICE_TOL_REL = 1e-5
KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date"]


def load_oos(db_path: str = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT model, asset, frequence, horizon_type, horizon_unit, horizon,
                   cutoff_date, target_date, last_close, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE source='oos' AND model=? AND horizon_type='weekly' AND y_true IS NOT NULL
            """, con, params=[MODEL])
    finally:
        con.close()


def check_alignment(db: pd.DataFrame, ens: pd.DataFrame) -> dict:
    """Bloquant. Un repointage qui creerait ou perdrait une ligne changerait la
    grille d'evaluation, pas la configuration -- ce n'est pas ce qu'on fait ici."""
    a = db.sort_values(KEYS).reset_index(drop=True)
    b = ens.sort_values(KEYS).reset_index(drop=True)
    checks = {"n_db": int(len(a)), "n_ensemble": int(len(b))}
    checks["same_keys"] = bool(len(a) == len(b) and a[KEYS].equals(b[KEYS]))
    if not checks["same_keys"]:
        only_db = set(map(tuple, a[KEYS].values)) - set(map(tuple, b[KEYS].values))
        only_ens = set(map(tuple, b[KEYS].values)) - set(map(tuple, a[KEYS].values))
        checks["only_in_db"] = sorted(only_db)[:10]
        checks["only_in_ensemble"] = sorted(only_ens)[:10]
        raise SystemExit(f"cles desalignees : {len(only_db)} en base seule, "
                         f"{len(only_ens)} dans l'ensemble seul -- repointage refuse")
    checks["y_true_max_rel_diff"] = float(
        (np.abs(a["y_true"] - b["y_true"]) / np.abs(a["y_true"])).max())
    checks["last_close_max_rel_diff"] = float(
        (np.abs(a["last_close"] - b["last_close"]) / np.abs(a["last_close"])).max())
    checks["targets_aligned"] = bool(checks["y_true_max_rel_diff"] <= PRICE_TOL_REL)
    checks["prices_aligned"] = bool(checks["last_close_max_rel_diff"] <= PRICE_TOL_REL)
    if not (checks["targets_aligned"] and checks["prices_aligned"]):
        raise SystemExit(f"cibles ou prix desalignes : {checks} -- repointage refuse")

    # deplacement attendu, calcule avant ecriture pour pouvoir le verifier apres
    inside_a = ((a["y_true"] >= a["y_lower"]) & (a["y_true"] <= a["y_upper"])).astype(float)
    inside_b = ((b["y_true"] >= b["y_lower"]) & (b["y_true"] <= b["y_upper"])).astype(float)
    w_a = (a["y_upper"] - a["y_lower"]) / a["last_close"]
    w_b = (b["y_upper"] - b["y_lower"]) / b["last_close"]
    checks["cov95_before"] = float(inside_a.mean())
    checks["cov95_after_expected"] = float(inside_b.mean())
    checks["pi_width_pct_before"] = float(w_a.mean() * 100)
    checks["pi_width_pct_after_expected"] = float(w_b.mean() * 100)
    checks["winkler_before"] = float(dash.winkler_score(a["y_true"], a["y_lower"], a["y_upper"]).mean())
    checks["winkler_after_expected"] = float(dash.winkler_score(b["y_true"], b["y_lower"], b["y_upper"]).mean())
    return checks


def build_rows(db: pd.DataFrame, ens: pd.DataFrame) -> list:
    """Lignes au format `insert_oos_predictions`. Tout ce qui IDENTIFIE la ligne
    (horizon, target_date, last_close, y_true) vient de la base : l'upsert doit
    tomber sur la cle existante et remplacer, jamais en creer une."""
    ref = db.set_index(KEYS)
    rows = []
    for r in ens.itertuples():
        base = ref.loc[(r.asset, r.frequence, r.horizon_unit, r.cutoff_date)]
        rows.append({
            "run_id": RUN_ID, "model": MODEL, "asset": r.asset,
            "horizon": int(base["horizon"]), "regime": "unknown",
            "cutoff_date": r.cutoff_date, "target_date": base["target_date"],
            "last_close": float(base["last_close"]),
            "y_pred": float(r.y_pred), "y_lower": float(r.y_lower), "y_upper": float(r.y_upper),
            "y_true": float(base["y_true"]), "source": "oos", "frequence": r.frequence,
            "horizon_type": "weekly", "horizon_unit": r.horizon_unit,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--apply", action="store_true", help="écrit réellement (défaut : dry-run)")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    rows, samples = v2.load_rows(with_samples=True, data_dir=Path(args.v2_dir), model=MODEL)
    n_seeds, n_draws = rows["seed"].nunique(), samples.shape[1]
    if (n_seeds, n_draws) != (len(spec.SEEDS), spec.N_SAMPLES_PER_SEED):
        raise SystemExit(f"artefact hors spec production : {n_seeds}x{n_draws} au lieu de "
                         f"{len(spec.SEEDS)}x{spec.N_SAMPLES_PER_SEED}")
    ens = build_ensemble_rows(rows, samples)
    print(f"ensemble : {len(ens)} lignes, {int(ens['n_samples_total'].iloc[0])} tirages chacune "
          f"({n_seeds} graines x {n_draws})")

    db = load_oos(args.db_path)
    print(f"base : {len(db)} lignes oos {MODEL} (horizon_type='weekly')")
    checks = check_alignment(db, ens)
    print(f"alignement : clés identiques ({checks['n_db']} lignes), prix à "
          f"{checks['last_close_max_rel_diff']:.2e}, cibles à {checks['y_true_max_rel_diff']:.2e}")
    print(f"déplacement attendu : Cov95 {checks['cov95_before']:.4f} -> "
          f"{checks['cov95_after_expected']:.4f} | largeur {checks['pi_width_pct_before']:.2f} % -> "
          f"{checks['pi_width_pct_after_expected']:.2f} % | Winkler {checks['winkler_before']:.1f} -> "
          f"{checks['winkler_after_expected']:.1f}")

    payload = {
        "run_id": RUN_ID, "applied": bool(args.apply), "db_path": args.db_path,
        "from": "graine 42, 200 tirages (non convergée : le regime weekly exige 800/graine)",
        "to": {**spec.PRODUCTION_SPEC, "converged": "6/6 cellules (nsamples_sweep)"},
        "alignment_checks": checks,
        "before": snapshot(args.db_path, MODEL),
    }

    if not args.apply:
        payload["note"] = "dry-run : aucune écriture. Relancer avec --apply."
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n--dry-run : rien écrit. Plan -> {args.out}")
        return

    backup = Path(args.db_path).with_suffix(
        f".db.bak_repoint_ensemble_{time.strftime('%Y-%m-%dT%H%M%S')}")
    shutil.copy2(args.db_path, backup)
    payload["backup"] = str(backup)
    print(f"\nSauvegarde -> {backup.name}")

    n = st.insert_oos_predictions(build_rows(db, ens), db_path=args.db_path)
    n_null = null_derived(args.db_path, MODEL)
    payload.update({"n_upserted": int(n), "n_derived_nulled": int(n_null)})
    print(f"{n} lignes upsertées, {n_null} lignes de colonnes dérivées remises à NULL")

    print("\nRecalcul des colonnes dérivées (backfill_eval_metrics.py) ...")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "backfill_eval_metrics.py"),
                           "--db-path", args.db_path], capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        raise SystemExit(f"backfill_eval_metrics a échoué : {proc.stderr}")

    after = snapshot(args.db_path, MODEL)
    payload["after"] = after
    exp = checks["cov95_after_expected"]
    if not np.isclose(after["cov95_recomputed"], exp, atol=1e-9):
        raise SystemExit(f"couverture post-bascule {after['cov95_recomputed']:.6f} != attendue {exp:.6f}")
    if after["cov95_stored_column"] is None or not np.isclose(
            after["cov95_stored_column"], after["cov95_recomputed"], atol=1e-9):
        raise SystemExit("colonne in_interval désynchronisée des bornes après backfill")
    print(f"vérifié : Cov95 = {after['cov95_recomputed']:.4f}, colonne in_interval cohérente")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
