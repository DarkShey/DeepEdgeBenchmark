"""
nsdiff_v2_data.py -- chargement et mise en forme de l'artefact multi-graines v2
(`experiments/nsdiff_multiseed_v2/`), partages par les quatre scripts de
consolidation du BRIEF "NsDiff : consolider le verdict daily vs weekly"
(taches 1/2/3, 5, 6, 7).

Ne calcule AUCUNE statistique : ce module se contente de rendre les lignes v2
indiscernables, pour les briques existantes, des lignes `source='oos'` de
`tracking.db` -- de sorte que `matrice_paired_tests.*` et
`dashboard_d7_w1.build_enriched_pairs` s'appliquent tels quels, sans
adaptation ni recopie.

Convention de POOLING DES GRAINES, utilisee partout dans le chantier et
declaree une fois ici : on moyenne la METRIQUE par origine a travers les 5
graines (sq_error, Winkler, indicateur de couverture), puis on teste cette
serie chronologique d'origines. L'unite d'inference reste donc l'ORIGINE
(n inchange, `effective_n` inchange) -- moyenner sur les graines reduit le
bruit Monte-Carlo du modele, ca ne cree pas de points de donnees. Ce qui est
teste ainsi est la performance ATTENDUE d'un run a graine tiree au hasard,
pas celle d'un ensemble des 5 graines (ca, c'est la tache 6, ou les nuages
eux-memes sont fusionnes -- resultat different, script separe).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "models", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dashboard_d7_w1 as dash                                        # noqa: E402
import matrice_paired_tests as mpt                                    # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "nsdiff_multiseed_v2"
HORIZON_UNITS = ["W+1", "W+2", "W+3"]
MODEL = "NsDiff"


def load_config(data_dir: Path = DATA_DIR) -> dict:
    return json.loads((Path(data_dir) / "config.json").read_text())


def load_rows(with_samples: bool = False, data_dir: Path = DATA_DIR, model: str = MODEL):
    """Lignes v2 + colonnes derivees, au format attendu par les briques
    existantes (`model`/`horizon_type` ajoutes, `sq_error`/`in_interval`
    calcules exactement comme `matrice_paired_tests.load_predictions`).

    `data_dir`/`model` par defaut = l'artefact NsDiff de la consolidation
    (`nsdiff_multiseed_v2/`), pour que les JSON de ce chantier restent
    reproductibles a l'identique. Le match diffusion-vs-diffusion pointe, lui,
    sur `diffusion_multiseed_v2/<Modele>/` -- artefact distinct, prix geles
    partages entre les deux bras."""
    rows = pd.read_parquet(Path(data_dir) / "rows.parquet")
    rows["model"] = model
    rows["horizon_type"] = "weekly"
    rows["sq_error"] = (rows["y_pred"] - rows["y_true"]) ** 2
    rows["in_interval"] = ((rows["y_true"] >= rows["y_lower"])
                           & (rows["y_true"] <= rows["y_upper"])).astype(float)
    if not with_samples:
        return rows
    samples = np.load(Path(data_dir) / "samples.npy")
    if len(samples) != len(rows):
        raise SystemExit(f"rows/samples misaligned: {len(rows)} vs {len(samples)}")
    return rows, samples


def seeds(rows: pd.DataFrame) -> list:
    return sorted(int(s) for s in rows["seed"].unique())


def assets(rows: pd.DataFrame) -> list:
    return sorted(rows["asset"].unique())


def price_cache(rows: pd.DataFrame, refresh: bool = False) -> dict:
    """Cache prix local de `dashboard_d7_w1` (reutilise tel quel, meme
    repertoire `.price_cache_d7_w1/`) -- necessaire aux bandes RW du
    skill-score, et garant de la reproductibilite d'un run a l'autre."""
    end_date = str(pd.to_datetime(rows["target_date"]).max().date())
    return dash.load_price_history_cache(assets(rows), end_date, refresh=refresh)


def enriched_pairs(rows: pd.DataFrame, cache: dict, horizon_unit: str,
                   seed: int | None = None) -> pd.DataFrame:
    """Paires regime B / regime C enrichies (Winkler, bandes RW, skill-scores)
    par `dashboard_d7_w1.build_enriched_pairs`, APPELE TEL QUEL -- seul son
    kwarg `horizon_unit` (opt-in, defaut W+1 = comportement d'origine) est
    utilise pour couvrir aussi W+2/W+3 (tache 2)."""
    sub = rows if seed is None else rows[rows["seed"] == seed]
    return dash.build_enriched_pairs(sub, cache, horizon_unit=horizon_unit)


SEED_AVERAGED_COLS = [
    "sq_error_daily", "sq_error_weekly", "winkler_daily", "winkler_weekly",
    "in_interval_daily", "in_interval_weekly", "pi_width_daily", "pi_width_weekly",
    "skill_sqerror_daily", "skill_sqerror_weekly", "skill_winkler_daily",
    "skill_winkler_weekly", "skill_diff_sqerror", "skill_diff_winkler",
]


def seed_average(pairs_by_seed: dict) -> pd.DataFrame:
    """Moyenne des metriques par (asset, cutoff_date) a travers les graines --
    voir la convention declaree dans le docstring du module. Les colonnes
    non-metriques (y_true, bandes RW, classe d'actif) sont identiques d'une
    graine a l'autre par construction : on prend celles de la premiere graine
    apres avoir VERIFIE que les origines coincident."""
    frames = list(pairs_by_seed.values())
    keys = ["asset", "cutoff_date"]
    ref = frames[0].sort_values(keys).reset_index(drop=True)
    for other in frames[1:]:
        o = other.sort_values(keys).reset_index(drop=True)
        if not ref[keys].equals(o[keys]):
            raise SystemExit("seed_average: les graines ne partagent pas les memes origines")
        if not np.allclose(ref["y_true_daily"], o["y_true_daily"]):
            raise SystemExit("seed_average: y_true differe entre graines -- artefact incoherent")

    stacked = pd.concat([f.sort_values(keys).reset_index(drop=True) for f in frames])
    averaged = stacked.groupby(level=0)[SEED_AVERAGED_COLS].mean()
    out = ref.drop(columns=SEED_AVERAGED_COLS).join(averaged)
    out["n_seeds"] = len(frames)
    return out


def pooled_skill_test(pairs: pd.DataFrame, seed: int = 42) -> dict:
    """`dashboard_d7_w1.build_pooled_series` + `run_pooled_test` appeles tels
    quels (dedoublonnage ZN=F/TLT en une contribution "taux" inclus), pour le
    global et chaque classe d'actifs."""
    pooled = dash.build_pooled_series(pairs)
    out = {"global": dash.run_pooled_test(pooled, None, seed)}
    for cls in ("crypto", "index", "bond"):
        out[cls] = dash.run_pooled_test(pooled, cls, seed)
    return out


def cell_rmse_test(rows: pd.DataFrame, horizon_unit: str) -> dict:
    """Test RMSE par cellule via `matrice_paired_tests.comparison_3_daily_vs_
    weekly` (reutilise tel quel, graine interne 0 comme le dashboard),
    filtre a l'horizon demande."""
    return {r["asset"]: r for r in mpt.comparison_3_daily_vs_weekly(rows)
            if r.get("horizon_unit") == horizon_unit}


ASSET_CLASS = mpt.ASSET_CLASS
