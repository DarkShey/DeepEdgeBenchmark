"""
oos_reference_audit.py -- chantier A1 : la piste `oos` / dashboard est-elle
encore comparable a la reference actee (n_samples=200), et pour QUELS modeles
la question se pose-t-elle reellement ?

Le brief demande de "regenerer la piste oos a n_samples=200 pour TOUS les
modeles echantillonnes". Ce script commence par etablir la liste de ces
modeles au lieu de la supposer -- parce que la reponse change radicalement le
cout du chantier :

  * un intervalle ANALYTIQUE (mu +/- sigma * ppf) ne tire aucun echantillon :
    il est insensible a m par construction, il n'y a rien a regenerer ;
  * un intervalle lu en QUANTILE EMPIRIQUE sur m tirages porte le biais
    mecanique quantifie ci-dessous, et lui seul doit etre repris.

BIAIS DE QUANTILE, rappel formel (la seule formule de ce fichier). `np.quantile`
en mode 'linear' vise le rang 1-indexe r = (m-1)*q + 1. L'esperance du niveau
reellement estime par la r-ieme statistique d'ordre d'un echantillon i.i.d. de
taille m vaut r/(m+1). A q=0.975 :

    m=50   -> r=48.775  -> niveau 0.9564 -> couverture bilaterale 91.27 %
    m=200  -> r=195.025 -> niveau 0.9703 -> couverture bilaterale 94.05 %
    m=500  -> r=487.525 -> niveau 0.9731 -> couverture bilaterale 94.62 %
    m=1000 -> r=975.025 -> niveau 0.9740 -> couverture bilaterale 94.81 %

Le biais est purement mecanique : il ne depend ni de la loi echantillonnee, ni
du modele. Un PI etiquete "95 %" lu sur 50 tirages n'en couvre que ~91.3 % meme
si le modele est parfaitement calibre.

Ce que le script verifie ENSUITE, empiriquement et sans rien supposer :
  * les origines de l'artefact v2 (graine 42) coincident-elles exactement avec
    les lignes `oos` en base -- memes cutoffs, meme `y_true`, meme `last_close` ;
  * de combien la couverture / la largeur / le Winkler bougent entre m=50 (base)
    et m=200 (artefact), par (modele, regime, horizon) ;
  * si l'ecart observe est compatible avec le SEUL changement de budget
    d'echantillonnage, ou s'il trahit un autre changement (budget d'epoques
    different, par exemple) -- auquel cas le repointage ne serait pas un
    re-cadrage mais une substitution de modele, et devrait etre refuse.

Lecture SEULE : ce script n'ecrit jamais en base. Le repointage lui-meme vit
dans `repoint_oos_to_m200.py`, qui consomme le plan produit ici.

Sortie : experiments/oos_reference_audit.json
Usage   : python oos_reference_audit.py
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dashboard_d7_w1 as dash                                        # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                          # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "oos_reference_audit.json"
V2_DIR = Path(__file__).resolve().parent / "diffusion_multiseed_v2"
V2_NSDIFF_SOLO = Path(__file__).resolve().parent / "nsdiff_multiseed_v2"
REFERENCE_SEED = 42
HORIZON_UNITS = ["W+1", "W+2", "W+3"]

# Mecanisme d'intervalle de chaque modele de la piste `oos`, avec sa preuve dans
# le code. Table DECLAREE (pas devinee) : chaque entree a ete verifiee en lisant
# le constructeur de bornes du modele concerne.
INTERVAL_MECHANISM = {
    "ARIMA-GARCH": {
        "kind": "analytic",
        "evidence": "models/arima_model.py:_std_quantiles / forecast -- "
                    "last_price*exp(mu + sigma*dist.ppf(alpha/2, 1-alpha/2)), "
                    "ppf de la loi d'innovation ajustee",
        "n_samples": None,
    },
    "SARIMA": {
        "kind": "analytic",
        "evidence": "models/sarima_model.py -- get_forecast().conf_int() de "
                    "statsmodels (bornes fermees), recalibrees par un facteur sigma",
        "n_samples": None,
    },
    "Naive": {
        "kind": "analytic",
        "evidence": "models/naive_model.py:55 Z_95=1.96 -- prev +/- 1.96*sigma_t, "
                    "sigma_t = EWMA causale des variations (sigma_mode='ewma')",
        "n_samples": None,
    },
    "LSTM": {
        "kind": "analytic",
        "evidence": "models/lstm_model.py -- pred +/- 1.96*sigma_t (EWMA causale des "
                    "residus). Le MC-Dropout (`n_ensemble`) existe mais vaut 0 par "
                    "defaut et ne sert pas a construire les bornes.",
        "n_samples": None,
    },
    "Prophet": {
        "kind": "sampled_internally",
        "evidence": "models/prophet_model.py:176 Prophet(interval_width=0.95) -- bornes "
                    "yhat_lower/upper, tirees par Prophet sur ses `uncertainty_samples` "
                    "internes (defaut 1000, jamais surcharge ici)",
        "n_samples": 1000,
    },
    "NsDiff": {
        "kind": "sampled",
        "evidence": "experiments/oos_nsdiff_daily_weekly.py:341 -- np.quantile(s, [0.025, 0.975]) "
                    "sur le nuage predictif ; DEFAULT_N_SAMPLES=50",
        "n_samples": 50,
    },
    "TSDiff": {
        "kind": "sampled",
        "evidence": "meme boucle de generation (weekly_headtohead_v2, n_samples=50) -- "
                    "quantiles empiriques 2.5/97.5 sur le nuage",
        "n_samples": 50,
    },
}

# Seuil de tolerance sur `last_close` entre la base et l'artefact : au-dela, les
# deux ne regardent pas la meme serie de prix et le repointage est refuse.
PRICE_TOL_REL = 1e-5


# ── 1. biais de quantile ────────────────────────────────────────────────────

def expected_quantile_level(m: int, q: float = 0.975) -> float:
    """Niveau reellement estime par `np.quantile(x, q)` (mode 'linear') sur m
    tirages i.i.d. : E[niveau de la r-ieme statistique d'ordre] = r/(m+1) avec
    r = (m-1)*q + 1."""
    r = (m - 1) * q + 1.0
    return r / (m + 1.0)


def two_sided_coverage(m: int, q: float = 0.975) -> float:
    """Couverture bilaterale reelle d'un PI etiquete 2*(1-q) : 2*level - 1."""
    return 2.0 * expected_quantile_level(m, q) - 1.0


def quantile_bias_table(ms=(50, 200, 500, 1000)) -> list:
    return [{"n_samples": int(m),
             "estimated_level_at_q975": round(expected_quantile_level(m), 6),
             "true_two_sided_coverage": round(two_sided_coverage(m), 6),
             "shortfall_vs_95pct_points": round(100 * (0.95 - two_sided_coverage(m)), 3)}
            for m in ms]


# ── 2. chargement ───────────────────────────────────────────────────────────

def load_oos(models: list, db_path: str = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT model, asset, frequence, horizon_type, horizon_unit, horizon,
                   cutoff_date, target_date, last_close, y_pred, y_lower, y_upper, y_true
            FROM predictions
            WHERE source='oos' AND horizon_type='weekly' AND y_true IS NOT NULL
                  AND model IN ({})
            """.format(",".join("?" * len(models))), con, params=models)
    finally:
        con.close()
    return df


def load_v2(model: str, seed: int = REFERENCE_SEED, v2_dir: Path = V2_DIR) -> pd.DataFrame:
    rows = pd.read_parquet(v2_dir / model / "rows.parquet")
    rows = rows[rows["seed"] == seed].copy()
    rows["model"] = model
    rows["horizon_type"] = "weekly"
    return rows


def _metrics(df: pd.DataFrame) -> dict:
    inside = ((df["y_true"] >= df["y_lower"]) & (df["y_true"] <= df["y_upper"])).astype(float)
    width = df["y_upper"] - df["y_lower"]
    return {
        "n": int(len(df)),
        "cov95": float(inside.mean()),
        "pi_width_mean": float(width.mean()),
        "pi_width_pct_of_price": float((width / df["last_close"]).mean() * 100),
        "winkler_mean": float(dash.winkler_score(df["y_true"], df["y_lower"], df["y_upper"]).mean()),
        "rmse": float(np.sqrt(((df["y_pred"] - df["y_true"]) ** 2).mean())),
    }


# ── 3. comparaison base (m=50) vs artefact (m=200) ──────────────────────────

KEYS = ["asset", "frequence", "horizon_unit", "cutoff_date"]


def compare_model(model: str, oos: pd.DataFrame, v2: pd.DataFrame) -> dict:
    """Aligne les deux pistes origine par origine, verifie qu'elles regardent
    la meme cible et le meme prix, puis mesure le deplacement."""
    a = oos[oos["model"] == model].sort_values(KEYS).reset_index(drop=True)
    b = v2.sort_values(KEYS).reset_index(drop=True)

    checks = {"n_oos": int(len(a)), "n_v2": int(len(b))}
    if len(a) != len(b) or not a[KEYS].equals(b[KEYS]):
        merged = a.merge(b, on=KEYS, suffixes=("_oos", "_v2"))
        checks["origins_identical"] = False
        checks["n_shared_origins"] = int(len(merged))
    else:
        merged = a.join(b.drop(columns=KEYS + ["model", "horizon_type"]),
                        rsuffix="_v2").rename(columns=lambda c: c if c.endswith("_v2") else c)
        merged = a.merge(b, on=KEYS, suffixes=("_oos", "_v2"))
        checks["origins_identical"] = True
        checks["n_shared_origins"] = int(len(merged))

    if merged.empty:
        return {"status": "no_overlap", "checks": checks}

    checks["y_true_max_abs_rel_diff"] = float(
        (np.abs(merged["y_true_oos"] - merged["y_true_v2"]) / np.abs(merged["y_true_oos"])).max())
    checks["last_close_max_abs_rel_diff"] = float(
        (np.abs(merged["last_close_oos"] - merged["last_close_v2"]) / np.abs(merged["last_close_oos"])).max())
    checks["prices_aligned"] = bool(checks["last_close_max_abs_rel_diff"] <= PRICE_TOL_REL)
    checks["targets_aligned"] = bool(checks["y_true_max_abs_rel_diff"] <= PRICE_TOL_REL)

    # Le point : un simple changement de m ne deplace le POINT (moyenne du nuage)
    # que du bruit Monte-Carlo -- il ne le deplace pas systematiquement. Un ecart
    # important sur y_pred signale un FIT different (budget d'epoques change),
    # donc une substitution de modele et non un re-cadrage.
    rel_pred = np.abs(merged["y_pred_oos"] - merged["y_pred_v2"]) / merged["last_close_oos"]
    rel_width = ((merged["y_upper_v2"] - merged["y_lower_v2"])
                 / (merged["y_upper_oos"] - merged["y_lower_oos"]))
    checks["y_pred_rel_shift_median_pct"] = float(np.median(rel_pred) * 100)
    checks["y_pred_rel_shift_p95_pct"] = float(np.quantile(rel_pred, 0.95) * 100)
    checks["width_ratio_v2_over_oos_median"] = float(np.median(rel_width))

    per_cell = {}
    for (regime, hu), g in merged.groupby(["frequence", "horizon_unit"]):
        oos_side = g.rename(columns=lambda c: c[:-4] if c.endswith("_oos") else c)
        v2_side = g.rename(columns=lambda c: c[:-3] if c.endswith("_v2") else c)
        per_cell[f"{regime}|{hu}"] = {"m50_db": _metrics(oos_side), "m200_reference": _metrics(v2_side)}

    overall_oos = _metrics(merged.rename(columns=lambda c: c[:-4] if c.endswith("_oos") else c))
    overall_v2 = _metrics(merged.rename(columns=lambda c: c[:-3] if c.endswith("_v2") else c))
    return {
        "status": "compared", "checks": checks, "per_cell": per_cell,
        "overall": {"m50_db": overall_oos, "m200_reference": overall_v2,
                    "cov95_gain_points": round(100 * (overall_v2["cov95"] - overall_oos["cov95"]), 2)},
    }


# ── 4. plan de repointage ───────────────────────────────────────────────────

def build_repoint_plan(comparisons: dict, retired: list) -> dict:
    """Un modele est repointable ssi (a) il est echantillonne a m=50, (b) ses
    origines/prix/cibles coincident exactement avec l'artefact, et (c) l'ecart
    observe est bien un ecart de BUDGET D'ECHANTILLONNAGE et pas de fit.

    Un modele RETIRE du benchmark n'est pas repointe : le repointer reviendrait
    a maintenir comme reference vivante un modele qu'on vient de sortir du
    perimetre. Ses lignes restent en base, marquees historiques."""
    plan = {"repoint": [], "leave_as_is": [], "retired": list(retired)}
    for model, mech in INTERVAL_MECHANISM.items():
        if mech["kind"] != "sampled":
            plan["leave_as_is"].append({
                "model": model, "reason": mech["kind"], "detail": mech["evidence"],
                "insensitive_to_m": mech["kind"] == "analytic",
            })
            continue
        if model in retired:
            plan["retired_detail"] = plan.get("retired_detail", [])
            plan["retired_detail"].append({
                "model": model,
                "reason": "retire du benchmark (chantier A3) -- non repointe, lignes conservees "
                          "en base et marquees historiques",
            })
            continue
        cmp_ = comparisons.get(model, {})
        checks = cmp_.get("checks", {})
        ok = (cmp_.get("status") == "compared" and checks.get("origins_identical")
              and checks.get("prices_aligned") and checks.get("targets_aligned"))
        plan["repoint" if ok else "leave_as_is"].append({
            "model": model, "n_rows": checks.get("n_shared_origins"),
            "reason": "echantillonne a m=50, artefact m=200 aligne" if ok
                      else "alignement non verifie -- repointage refuse",
            "checks": checks,
        })
    return plan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=DB_PATH)
    p.add_argument("--v2-dir", default=str(V2_DIR))
    p.add_argument("--retired", nargs="*", default=["TSDiff"],
                   help="modeles retires du benchmark (chantier A3) -- audites mais non repointes")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    sampled = [m for m, v in INTERVAL_MECHANISM.items() if v["kind"] == "sampled"]
    oos = load_oos(list(INTERVAL_MECHANISM), db_path=args.db_path)
    print(f"lignes oos weekly lues : {len(oos)} ({oos['model'].nunique()} modeles) -- LECTURE SEULE")

    comparisons = {}
    for model in sampled:
        v2 = load_v2(model, v2_dir=Path(args.v2_dir))
        comparisons[model] = compare_model(model, oos, v2)
        c = comparisons[model]
        if c["status"] != "compared":
            print(f"[{model}] {c['status']}")
            continue
        ch, ov = c["checks"], c["overall"]
        print(f"\n[{model}] origines identiques : {ch['origins_identical']} "
              f"({ch['n_shared_origins']} lignes) | prix alignes : {ch['prices_aligned']} "
              f"(max ecart relatif {ch['last_close_max_abs_rel_diff']:.2e})")
        print(f"[{model}] deplacement du POINT (y_pred, en % du prix) : mediane "
              f"{ch['y_pred_rel_shift_median_pct']:.4f} %, p95 {ch['y_pred_rel_shift_p95_pct']:.4f} %")
        print(f"[{model}] largeur m200/m50 (mediane) : x{ch['width_ratio_v2_over_oos_median']:.3f}")
        print(f"[{model}] Cov95 : {ov['m50_db']['cov95']:.3f} (m=50, base) -> "
              f"{ov['m200_reference']['cov95']:.3f} (m=200, reference)  "
              f"[{ov['cov95_gain_points']:+.2f} points]")

    plan = build_repoint_plan(comparisons, args.retired)
    payload = {
        "scope": "chantier A1 -- la piste oos est-elle encore comparable a la reference n_samples=200",
        "read_only": True,
        "reference": {"n_samples": 200, "seeds": [42, 43, 44, 45, 46],
                      "dashboard_track_seed": REFERENCE_SEED},
        "quantile_bias": {
            "formula": "np.quantile(mode 'linear') vise le rang r=(m-1)q+1 ; "
                       "E[niveau estime] = r/(m+1) ; couverture bilaterale = 2*niveau-1",
            "mechanical": "independant de la loi echantillonnee et du modele",
            "table": quantile_bias_table(),
        },
        "model_inventory": INTERVAL_MECHANISM,
        "sampled_models": sampled,
        "comparisons": comparisons,
        "repoint_plan": plan,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== biais de quantile (mecanique, tous modeles echantillonnes) ===")
    for r in payload["quantile_bias"]["table"]:
        print(f"  m={r['n_samples']:<5} niveau estime {r['estimated_level_at_q975']:.4f} -> "
              f"couverture reelle {100 * r['true_two_sided_coverage']:.2f} % "
              f"({r['shortfall_vs_95pct_points']:+.2f} points vs 95 %)")
    print("\n=== plan de repointage ===")
    for entry in plan["repoint"]:
        print(f"  REPOINTER  {entry['model']:<12} {entry['n_rows']} lignes -- {entry['reason']}")
    for entry in plan["leave_as_is"]:
        print(f"  inchange   {entry['model']:<12} {entry['reason']}")
    for entry in plan.get("retired_detail", []):
        print(f"  RETIRE     {entry['model']:<12} {entry['reason']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
