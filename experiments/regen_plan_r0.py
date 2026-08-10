"""
regen_plan_r0.py -- chantier R0 du BRIEF « regeneration oos et famille 3 » :
INVENTAIRE ET CHIFFRAGE de la regeneration complete de la grille `oos`, en
DRY-RUN STRICT (aucune ecriture, ni sur disque de production, ni en base).

Le brief traite R comme une migration de base, pas comme une experience :
« phase, chiffre avant execution, reversible a chaque etape ». R0 est l'etape
qui rend cela possible -- elle produit le plan que R1/R2 consomment.

CE QUE R0 ETABLIT, dans l'ordre du brief :

  1. RECENSEMENT des modeles de reference actifs et de leur mode de bornes.
     Le socle est `oos_reference_audit.INTERVAL_MECHANISM`, table DECLAREE et
     verifiee ligne a ligne dans le code des modeles. On y superpose une seule
     information neuve : ce que la grille 2020 contient DEJA. Un bras deja
     genere n'est pas rechiffre comme s'il restait a produire -- c'est la
     difference entre le cout affiche et le cout reel du chantier.

  2. CHIFFRAGE par modele : temps de refit x origines x actifs x regimes,
     mesure et non devine. La brique de chronometrage est celle du chantier
     precedent (`cost_grid_2020.probe_per_origin`), reutilisee telle quelle ;
     ce fichier n'ajoute que le perimetre neuf -- le bras GARCH skew-t exige
     par H1, et la separation phase W / phase D exigee par R2.

  3. SEUIL DECLARE AVANT LECTURE : 48 h de compute pour la regeneration
     complete (valeur proposee par le brief lui-meme). Au-dela, decoupage
     obligatoire. La phase W est de toute facon executee en premier -- pas par
     economie, mais parce que l'hypothese primaire pre-declaree est weekly et
     que la phase D est conditionnee a son verdict.

  4. PERIMETRE D'ECRITURE chiffre lui aussi : combien de lignes `oos` la base
     porte aujourd'hui, combien la nouvelle grille en produirait, et quel est
     le facteur d'expansion. Une migration dont on ne connait pas le volume
     n'est pas reversible.

LECTURE SEULE. Ce script ouvre `tracking.db` en mode `file:...?mode=ro` : toute
tentative d'ecriture leverait. Aucun artefact de run n'est touche.

Sortie : experiments/regen_plan_r0.json
Usage   : python regen_plan_r0.py [--n-probe 3] [--skip-lstm]
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmarks.multi_horizon as mh                                   # noqa: E402
from cost_grid_2020 import load_grid, probe_per_origin                  # noqa: E402
from oos_reference_audit import INTERVAL_MECHANISM                      # noqa: E402
from backtest_rolling_tsdiffw import DB_PATH                            # noqa: E402
from prices_v3 import ORIGIN_START, OUT_DIR as PRICES_V3, PANEL, slug   # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "regen_plan_r0.json"
GRID_DIR = Path(__file__).resolve().parent / "grid2020"

BUDGET_HOURS = 48.0          # seuil declare AVANT lecture du chiffrage (brief R0)
REGIMES = ("weekly", "daily")
HORIZONS_PER_ORIGIN = 3      # W+1, W+2, W+3

# Bras a produire pour completer la grille, avec la fonction qui les produit.
# ARIMA-GARCH apparait deux fois : le bras gaussien est deja genere (c'est la
# variante que portent les lignes `oos` actuelles, etablie par mesure au
# chantier precedent), le bras skew-t est le perimetre neuf de H1.
ARMS = {
    "ARIMA-GARCH[normal]": lambda tr, hs: mh.forecast_horizons_arima(tr, hs, dist="normal"),
    "ARIMA-GARCH[skewt]": lambda tr, hs: mh.forecast_horizons_arima(tr, hs, dist="skewt"),
    "SARIMA": mh.forecast_horizons_sarima,
    "Prophet": mh.forecast_horizons_prophet,
    "Naive": mh.forecast_horizons_naive,
    "LSTM": mh.forecast_horizons_lstm,
}


# ── 1. ce que la grille 2020 contient deja ──────────────────────────────────

def inventory_existing(grid_dir: Path) -> dict:
    """Lit les artefacts deja produits. Un bras present n'est pas a refaire ;
    le dire explicitement evite de rechiffrer un cout deja paye."""
    done = {}
    ns = grid_dir / "NsDiff" / "rows.parquet"
    if ns.exists():
        r = pd.read_parquet(ns, columns=["seed", "asset", "frequence", "cutoff_date"])
        done["NsDiff"] = {
            "artefact": str(ns.relative_to(grid_dir.parent)),
            "n_rows": int(len(r)), "n_origins": int(r["cutoff_date"].nunique()),
            "assets": sorted(r["asset"].unique()), "seeds": sorted(int(s) for s in r["seed"].unique()),
            "regimes": sorted(r["frequence"].unique()),
        }
    gc = grid_dir / "ARIMA-GARCH" / "bands.parquet"
    if gc.exists():
        r = pd.read_parquet(gc, columns=["asset", "frequence", "cutoff_date"])
        done["ARIMA-GARCH[normal]"] = {
            "artefact": str(gc.relative_to(grid_dir.parent)),
            "n_rows": int(len(r)), "n_origins": int(r["cutoff_date"].nunique()),
            "assets": sorted(r["asset"].unique()), "regimes": sorted(r["frequence"].unique()),
        }
    return done


# ── 2. dimensions reelles de la grille ──────────────────────────────────────

def grid_dimensions(assets) -> dict:
    per_asset = {}
    for a in assets:
        daily, weekly, weekly_dates, test_pos = load_grid(a)
        per_asset[a] = {"daily_obs": int(len(daily)), "weekly_obs": int(len(weekly)),
                        "n_origins": int(len(test_pos)),
                        "history_start": str(daily.index[0].date())}
    n_origins = max(v["n_origins"] for v in per_asset.values())
    return {"origin_start": ORIGIN_START, "n_origins": n_origins, "assets": list(assets),
            "per_asset": per_asset, "prices": str(PRICES_V3),
            "n_origin_cells_per_model": n_origins * len(assets) * len(REGIMES),
            "n_rows_per_model": n_origins * len(assets) * len(REGIMES) * HORIZONS_PER_ORIGIN}


# ── 3. etat de la base, en lecture seule ────────────────────────────────────

def db_state(db_path: str) -> dict:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT model, frequence, COUNT(*) n, COUNT(DISTINCT cutoff_date) n_cutoffs, "
            "MIN(cutoff_date) first_cutoff, MAX(cutoff_date) last_cutoff, "
            "COUNT(DISTINCT asset) n_assets "
            "FROM predictions WHERE source='oos' AND horizon_type='weekly' "
            "GROUP BY model, frequence ORDER BY model, frequence", con)
    finally:
        con.close()
    return {"db_path": db_path, "mode": "read-only (file:...?mode=ro)",
            "oos_weekly_track": df.to_dict(orient="records"),
            "n_rows_total": int(df["n"].sum())}


# ── 4. chiffrage ────────────────────────────────────────────────────────────

def cost_arms(arms: dict, probe_asset: str, n_probe: int, dims: dict) -> dict:
    n_o, n_a = dims["n_origins"], len(dims["assets"])
    out = {}
    for name, fn in arms.items():
        print(f"[{name}] chronometrage sur {n_probe} origines reelles ...")
        probe = probe_per_origin(name, fn, probe_asset, n_probe)
        per_regime = {}
        for regime, r in probe["per_regime"].items():
            s = r["median_s_per_origin"]
            per_regime[regime] = {**r, "extrapolated_h": (s * n_o * n_a / 3600.0) if s else None}
        out[name] = {"protocol": "refit par origine", "per_regime": per_regime,
                     "failures": probe["failures"],
                     "weekly_h": per_regime["weekly"]["extrapolated_h"] or 0.0,
                     "daily_h": per_regime["daily"]["extrapolated_h"] or 0.0}
        out[name]["total_h"] = out[name]["weekly_h"] + out[name]["daily_h"]
        print(f"   -> W {out[name]['weekly_h']:.2f} h + D {out[name]['daily_h']:.2f} h"
              + (f"  [ECHECS: {probe['failures'][:1]}]" if probe["failures"] else ""))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-asset", default="SPY")
    p.add_argument("--n-probe", type=int, default=3)
    p.add_argument("--assets", nargs="+", default=list(PANEL))
    p.add_argument("--skip-lstm", action="store_true",
                   help="saute le chronometrage LSTM (le plus lent) et reprend celui de "
                        "cost_grid_2020.json ; le plan le declare alors comme repris")
    p.add_argument("--grid-dir", default=str(GRID_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    t0 = time.time()
    dims = grid_dimensions(args.assets)
    print(f"grille : {dims['n_origins']} origines depuis {ORIGIN_START}, {len(args.assets)} actifs, "
          f"{len(REGIMES)} regimes = {dims['n_rows_per_model']} lignes par modele\n")

    existing = inventory_existing(Path(args.grid_dir))
    todo = {k: v for k, v in ARMS.items() if k not in existing}
    print("deja genere : " + (", ".join(existing) or "rien"))
    print("a generer   : " + ", ".join(todo) + "\n")

    if args.skip_lstm and "LSTM" in todo:
        todo.pop("LSTM")
    cost = cost_arms(todo, args.probe_asset, args.n_probe, dims)

    if args.skip_lstm:
        prev = json.loads((Path(args.out).parent / "cost_grid_2020.json").read_text())["per_model"]["LSTM"]
        n_o, n_a = dims["n_origins"], len(dims["assets"])
        cost["LSTM"] = {
            "protocol": "refit par origine",
            "per_regime": {r: {**v, "extrapolated_h": v["median_s_per_origin"] * n_o * n_a / 3600.0}
                           for r, v in prev["per_regime"].items()},
            "failures": prev.get("failures", []), "source": "repris de cost_grid_2020.json",
        }
        cost["LSTM"]["weekly_h"] = cost["LSTM"]["per_regime"]["weekly"]["extrapolated_h"]
        cost["LSTM"]["daily_h"] = cost["LSTM"]["per_regime"]["daily"]["extrapolated_h"]
        cost["LSTM"]["total_h"] = cost["LSTM"]["weekly_h"] + cost["LSTM"]["daily_h"]

    phase_w = sum(c["weekly_h"] for c in cost.values())
    phase_d = sum(c["daily_h"] for c in cost.values())
    full = phase_w + phase_d

    # perimetre d'ecriture : ce que la migration deposerait en base
    dbs = db_state(DB_PATH)
    n_models_after = len(existing) + len(cost) - 1      # les deux bras GARCH n'en font qu'un en base
    write_scope = {
        "rows_now_oos_weekly_track": dbs["n_rows_total"],
        "rows_per_model_per_phase": {
            "weekly": dims["n_origins"] * len(args.assets) * HORIZONS_PER_ORIGIN,
            "daily": dims["n_origins"] * len(args.assets) * HORIZONS_PER_ORIGIN,
        },
        "rows_after_phase_W": dims["n_origins"] * len(args.assets) * HORIZONS_PER_ORIGIN * n_models_after,
        "rows_after_both_phases": dims["n_rows_per_model"] * n_models_after,
        "expansion_factor": round(dims["n_rows_per_model"] * n_models_after / max(dbs["n_rows_total"], 1), 2),
        "n_models_written": n_models_after,
        "regle": "un seul script d'upsert par phase, standard repoint_oos_to_m200 : dry-run par "
                 "defaut, sauvegarde horodatee, --apply explicite, verification 1:1 des cles, "
                 "backfill des colonnes derivees, bandeau de config. tracking.db en lecture seule "
                 "partout ailleurs.",
    }

    order = [
        {"step": "R1", "what": "gel des donnees prices_v3 + recouvrement avec diffusion_multiseed_v2/prices",
         "cost_h": 0.0, "writes": "aucune (verification bloquante)"},
        {"step": "R2/W", "what": "phase W -- grille weekly 340 origines, 7 actifs, tous modeles "
                                 "(H1 integre : bras GARCH skew-t genere dans la meme passe)",
         "cost_h": round(phase_w, 3), "writes": "artefacts disque uniquement"},
        {"step": "R2/W-upsert", "what": "upsert phase W en base (script dedie, dry-run puis --apply)",
         "cost_h": 0.0, "writes": f"{write_scope['rows_after_phase_W']} lignes"},
        {"step": "F3", "what": "famille 3 reposee -- exploratoire sur grille courante, confirmatoire "
                               "sur la nouvelle grille weekly",
         "cost_h": 0.0, "writes": "artefacts disque uniquement"},
        {"step": "R2/D", "what": "phase D -- idem daily, CONDITIONNEE au verdict de la phase W "
                                 "(hypothese primaire survivante OU signal de calibration nouveau)",
         "cost_h": round(phase_d, 3), "writes": "artefacts + upsert phase D"},
        {"step": "H2/H3", "what": "realisme d'execution des futures ; monitoring de couverture en ligne",
         "cost_h": 0.0, "writes": "aucune"},
    ]

    payload = {
        "scope": "chantier R0 -- inventaire et chiffrage de la regeneration complete de la grille oos",
        "read_only": True,
        "brief": "BRIEF_nsdiff_regeneration_oos_et_famille3.md",
        "grid": dims,
        "model_inventory": {
            m: {**v, "regeneration": ("retire -- aucune ligne regeneree" if m == "TSDiff"
                                      else "deja genere sur la grille 2020" if m in existing
                                      else "a generer")}
            for m, v in INTERVAL_MECHANISM.items()
        },
        "already_generated": existing,
        "cost": cost,
        "totals": {
            "phase_W_h": phase_w, "phase_D_h": phase_d, "full_regeneration_h": full,
            "budget_threshold_h": BUDGET_HOURS,
            "declared_before_reading": True,
            "fits_in_budget": bool(full <= BUDGET_HOURS),
            "already_paid_h": "NsDiff (5 graines x 200) et ARIMA-GARCH gaussien sont deja produits "
                              "sur cette grille : leur cout n'est pas recompte ici",
            "decoupage": "phase W d'abord dans tous les cas -- l'hypothese primaire pre-declaree est "
                         "weekly, et la phase D lui est conditionnee (R2.2). Le decoupage n'est donc "
                         "pas une consequence du budget mais du protocole.",
        },
        "db_state": dbs,
        "write_scope": write_scope,
        "execution_order": order,
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))

    print(f"\n=== chiffrage (heures extrapolees, {dims['n_origins']} origines x "
          f"{len(args.assets)} actifs) ===")
    for name, c in sorted(cost.items(), key=lambda kv: -kv[1]["total_h"]):
        print(f"  {name:<22}{c['weekly_h']:>7.2f} h (W) +{c['daily_h']:>7.2f} h (D) "
              f"= {c['total_h']:>7.2f} h")
    print(f"  {'PHASE W':<22}{phase_w:>7.2f} h")
    print(f"  {'PHASE D':<22}{phase_d:>7.2f} h")
    print(f"  {'TOTAL':<22}{full:>7.2f} h   -> "
          f"{'dans le budget' if full <= BUDGET_HOURS else 'HORS BUDGET, decoupage obligatoire'} "
          f"(seuil declare {BUDGET_HOURS:.0f} h)")
    print(f"\n=== perimetre d'ecriture (aucune ecriture faite ici) ===")
    print(f"  base aujourd'hui        {write_scope['rows_now_oos_weekly_track']} lignes oos")
    print(f"  apres phase W           {write_scope['rows_after_phase_W']} lignes")
    print(f"  apres les deux phases   {write_scope['rows_after_both_phases']} lignes "
          f"(x{write_scope['expansion_factor']})")
    print(f"\n=== ordre d'execution ===")
    for s in order:
        print(f"  {s['step']:<12}{s['cost_h']:>7.2f} h  {s['what'][:70]}")
    print(f"\n-> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
