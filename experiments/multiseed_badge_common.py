"""
multiseed_badge_common.py -- badge de robustesse inter-graines + label de config
honnête (BRIEF_dashboard_multiseed_200.md), factorisé hors de dashboard_d7_w1.py
pour être importable SANS ses dépendances lourdes (arima_model -> yfinance,
matrice_paired_tests/paired_test -> statsmodels/arch, cf. leurs imports).

Pourquoi ce module existe (bug constaté en prod) : model_artifacts/
generate_dashboard.py est déployé par .github/workflows/deploy-pages.yml, dont
l'étape "Install minimal deps" installe UNIQUEMENT `pandas pyarrow numpy scipy`
(volontairement minimal, cf. BRIEF_dashboard_externalisation.md -- pas question
d'alourdir ce job juste pour le badge). `import dashboard_d7_w1` y échouait
silencieusement (ImportError sur une dépendance de arima_model/matrice_paired_
tests, absente de cet environnement), capturé par le try/except défensif de
generate_dashboard.collect_weekly_multiseed() -> badge/bandeau vides en
production alors que tout fonctionnait en local (venv complet). Ce module n'a
besoin que de la stdlib (json, pathlib, collections) : aucune dépendance
supplémentaire à installer, aucun risque de ce genre de panne silencieuse.

Réutilisé par :
  - experiments/dashboard_d7_w1.py (réexporte ces noms pour compatibilité --
    nsdiff_daily_weekly_multiseed.py/tsdiff_daily_weekly_multiseed.py importent
    `dashboard_d7_w1 as dash` et n'utilisent que `dash.winkler_score`, donc pas
    d'impact sur eux).
  - model_artifacts/generate_dashboard.py (collect_weekly_multiseed()).
"""

import json
from collections import Counter
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Audit §3 de BRIEF_dashboard_multiseed_200.md (grep + lecture de benchmarks/
# multi_horizon.py + models/{tsdiff,nsdiff,prophet}_model.py, pas supposé) :
# seuls NsDiff et TSDiff ont des bandes issues d'un nuage fini de n_samples
# tirages (seed + n_samples qu'on contrôle, np.mean/np.quantile sur le nuage).
# Naive/ARIMA-GARCH/SARIMA : bandes fermées (aucune graine). LSTM : graine
# d'ENTRAÎNEMENT mais bandes = formule fermée (point +/- 1.96*std*sqrt(h)), pas
# un nuage relu en quantiles -- pas concerné. Prophet échantillonne en interne
# (Facebook Prophet, MC non graine) mais ce dépôt n'expose ni seed ni n_samples
# pour lui -- dette déclarée, non régénéré (cf. NOTE_dashboard_multiseed_200.md).
MULTISEED_MODELS = ["NsDiff", "TSDiff"]
TARGET_ENSEMBLE_LABEL = "ensemble 5 graines (42-46) x 200 tirages"
DEFAULT_MULTISEED_JSON = {m: EXPERIMENTS_DIR / f"{m.lower()}_daily_weekly_multiseed.json"
                          for m in MULTISEED_MODELS}


def load_multiseed_artifacts(overrides: dict = None) -> dict:
    """{model: {"path": str, "data": dict}} pour chaque modèle échantillonné
    (MULTISEED_MODELS) dont l'artefact JSON multiseed existe et se lit. Jamais
    bloquant (brief §5.2, dégradation gracieuse) : un artefact absent ou
    illisible => juste pas d'entrée pour ce modèle, pas de badge sur ses
    cellules, aucune exception ne remonte."""
    overrides = overrides or {}
    out = {}
    for model in MULTISEED_MODELS:
        path = Path(overrides.get(model, DEFAULT_MULTISEED_JSON[model]))
        if not path.exists():
            print(f"  [multiseed] {model}: artefact absent ({path}) -- pas de badge pour ce modèle.")
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"  [multiseed] {model}: illisible ({path}): {exc} -- pas de badge pour ce modèle.")
            continue
        out[model] = {"path": str(path), "data": data}
        n_samples = data.get("config", {}).get("n_samples")
        n_seeds = len(data.get("seeds", []))
        print(f"  [multiseed] {model}: {path.name} chargé (n_samples={n_samples}, {n_seeds} graines).")
    return out


def cell_robustness_badge(multiseed_artifacts: dict, model: str, asset: str):
    """Badge de robustesse inter-graines pour la cellule (model, asset), lu
    depuis l'artefact JSON multiseed (jamais depuis `oos`, brief §1 point 2) --
    None si le modèle n'a pas d'artefact chargé ou si cet actif n'y figure pas
    (dégradation gracieuse, pas d'erreur)."""
    entry = multiseed_artifacts.get(model)
    if entry is None:
        return None
    cv = entry["data"].get("cv_table", {}).get(asset)
    if cv is None:
        return None
    verdicts_by_seed = cv.get("verdicts_by_seed") or {}
    n_seeds = len(verdicts_by_seed)
    majority_count = Counter(verdicts_by_seed.values()).most_common(1)[0][1] if n_seeds else None
    return {
        "model": model,
        "verdict_stable": cv.get("verdict_stable"),
        "n_seeds": n_seeds,
        "majority_count": majority_count,
        "cv_winkler_daily": cv.get("cv_winkler_daily"),
        "cv_winkler_weekly": cv.get("cv_winkler_weekly"),
        "n_samples": entry["data"].get("config", {}).get("n_samples"),
        "source_path": entry["path"],
    }


def build_data_config(multiseed_artifacts: dict, models_in_df: list) -> dict:
    """Traçabilité honnête (brief §5.1/§5.3) : décrit l'état RÉEL des artefacts
    (jamais une affirmation figée de "200/ensemble") -- se met à jour tout
    seul, sans toucher au code, dès que le tuteur régénère les JSON à
    n_samples=200/5 graines (cf. RUNBOOK_regeneration_multiseed_200.md)."""
    per_model = {}
    for model in MULTISEED_MODELS:
        entry = multiseed_artifacts.get(model)
        if entry is None:
            per_model[model] = {
                "artifact_found": False, "n_samples": None, "n_seeds": None, "source_path": None,
                "is_target_config": False,
                "status_label": "artefact multiseed absent -- badge indisponible pour ce modèle",
            }
            continue
        n_samples = entry["data"].get("config", {}).get("n_samples")
        n_seeds = len(entry["data"].get("seeds", []))
        is_target = (n_samples == 200 and n_seeds == 5)
        per_model[model] = {
            "artifact_found": True, "n_samples": n_samples, "n_seeds": n_seeds,
            "source_path": entry["path"], "is_target_config": is_target,
            # Juste le suffixe : les templates JS préfixent déjà "{n_seeds} graine(s) x
            # {n_samples} tirages -- " avant d'insérer status_label (dashboard_d7_w1_
            # template.py et model_artifacts/generate_dashboard.py) -- le préfixer ICI
            # AUSSI dupliquait le texte à l'affichage (bug signalé, corrigé).
            "status_label": "config cible atteinte" if is_target else "pas encore la config cible",
        }
    analytic_models = sorted(m for m in models_in_df if m not in MULTISEED_MODELS)
    all_target = bool(per_model) and all(v["is_target_config"] for v in per_model.values())
    if all_target:
        headline = (f"Données (modèles échantillonnés {', '.join(MULTISEED_MODELS)}) : "
                   f"{TARGET_ENSEMBLE_LABEL} -- config de production (tâche 6).")
    else:
        parts = []
        for m, v in per_model.items():
            if v["artifact_found"]:
                parts.append(f"{m} : {v['n_seeds']} graine(s) x {v['n_samples']} tirages")
            else:
                parts.append(f"{m} : artefact absent")
        headline = ("Données actuelles (" + "; ".join(parts) + f") -- cible : {TARGET_ENSEMBLE_LABEL}, "
                   "régénération en attente côté tuteur (voir RUNBOOK_regeneration_multiseed_200.md).")
    return {
        "target": TARGET_ENSEMBLE_LABEL,
        "sampled_models": MULTISEED_MODELS,
        "analytic_models": analytic_models,
        "analytic_note": ("Bandes fermées/déterministes (aucun nuage de tirages, non concernés par ce "
                          "budget) : Naive/ARIMA-GARCH/SARIMA (formule fermée) ; LSTM (formule fermée malgré "
                          "une graine d'entraînement). Prophet échantillonne en interne (librairie Facebook "
                          "Prophet) mais sans seed ni n_samples exposés dans ce dépôt -- dette déclarée, non "
                          "régénéré ici."),
        "multiseed_artifacts": per_model,
        "all_target_config": all_target,
        "headline": headline,
        "oos_rows_note": ("Le badge (artefact JSON multiseed) et les lignes `oos` affichées dans le tableau "
                          "sont DEUX écritures indépendantes (jamais la même) -- l'artefact peut être à 200 "
                          "tirages sans garantir que les lignes `oos` aient déjà été réécrites en ensemble, et "
                          "inversement. Se fier au run_id/n_samples affichés ici pour la vraie provenance, "
                          "jamais à une supposition."),
    }
