"""
benchmark_registry.py -- QUI est une reference vivante du benchmark, et a quelle
reference d'echantillonnage chaque modele est cale. Une seule source de verite,
importable, au lieu d'une convention repetee dans chaque script.

Ne contient que des declarations : aucun calcul, aucun acces base. Les scripts
qui produisent des verdicts (dashboard, matchs, backtests) importent ACTIVE /
RETIRED d'ici plutot que de lister les modeles a la main.

REFERENCE D'ECHANTILLONNAGE ACTEE (chantier A1, 2026-08-06) : n_samples=200
pour tout modele dont les bornes sont lues en quantile empirique. Motif : a
m=50 un PI etiquete 95 % n'en couvre mecaniquement que 91.3 %
(cf. `oos_reference_audit.expected_quantile_level`) ; a m=200, 94.1 %. Les
modeles a bornes ANALYTIQUES sont insensibles a m par construction -- ils ne
sont ni concernes ni regeneres.

RETRAITS. Un modele retire n'est plus une reference du benchmark : ses lignes
restent en base (rien n'est efface, l'historique est verifiable) mais aucune
conclusion du programme ne s'y appuie et le dashboard ne l'affiche plus par
defaut. Le retrait est une decision documentee, pas un effet de bord.
"""

# ── modeles actifs, avec la nature de leurs bornes ──────────────────────────
# `sampling_reference` : m effectif de lecture des quantiles ; None = analytique.
ACTIVE = {
    "ARIMA-GARCH": {"intervals": "analytic", "sampling_reference": None},
    "SARIMA":      {"intervals": "analytic", "sampling_reference": None},
    "Naive":       {"intervals": "analytic", "sampling_reference": None},
    "LSTM":        {"intervals": "analytic", "sampling_reference": None},
    "Prophet":     {"intervals": "sampled_internally", "sampling_reference": 1000},
    # NsDiff : la piste oos porte desormais la CONFIGURATION PRODUCTION
    # (ensemble 5 graines x 200 tirages = 1000 lus sur un nuage concatene), et
    # non plus un run a graine unique. Bascule du 2026-08-08, chantier C du
    # brief extension/puissance : `nsamples_sweep` a montre qu'a graine unique
    # 200 tirages n'est PAS converge (le regime weekly en exige 800), alors que
    # l'ensemble l'est sur 6 cellules sur 6. Cf. `repoint_oos_to_ensemble.py`.
    "NsDiff":      {"intervals": "sampled", "sampling_reference": 1000,
                    "configuration": "ensemble 5x200 (production)",
                    "seeds": [42, 43, 44, 45, 46]},
}

# ── modeles retires du benchmark ────────────────────────────────────────────
RETIRED = {
    "TSDiff": {
        "date": "2026-08-06",
        "chantier": "A3 du BRIEF 'NsDiff : valeur economique, re-cadrage 200 tirages, cadrage monthly'",
        "reason": (
            "Deux motifs, cumulatifs. (1) FUITE DE SELECTION non corrigee : les epoques "
            "hebdomadaires (40/60/80) viennent d'une validation dont le bloc "
            "(SPY : 2025-09-19 -> 2025-12-05) tombe A L'INTERIEUR de la grille de test oos "
            "(2024-10-18 -> 2026-07-02) -- ~13 % des origines evaluees ont servi a choisir "
            "l'hyperparametre. (2) CALIBRATION EFFONDREE : malgre cette fuite en sa faveur, "
            "TSDiff perd les 6 tests pooles weekly du duel a budget egal, et sa couverture "
            "observee sur la grille oos va de 0.53 a 0.85 pour un PI etiquete 95 %. "
            "Re-sweeper le bras hebdomadaire coutait un chantier complet pour reparer un "
            "modele deja battu ; le retrait est la decision economique."
        ),
        "rows_kept": True,
        "note": "Ses lignes oos restent en base a m=50 et ne sont PAS repointees : le passage "
                "a l'artefact 200 tirages changerait aussi son budget d'epoques cote daily "
                "(selectionne par validation depuis, largeur du PI 5.58 % -> 3.87 % du prix), "
                "ce qui serait une substitution de modele et non un re-cadrage.",
    },
}


def active_models() -> list:
    return sorted(ACTIVE)


def retired_models() -> list:
    return sorted(RETIRED)


def sampled_models() -> list:
    """Modeles dont les bornes sont lues en quantile empirique sur un nuage --
    les seuls concernes par la reference n_samples."""
    return sorted(m for m, v in ACTIVE.items() if v["intervals"] == "sampled")


def sampling_reference(model: str):
    if model in ACTIVE:
        return ACTIVE[model]["sampling_reference"]
    if model in RETIRED:
        return None
    raise KeyError(f"modele inconnu du registre : {model!r}")
