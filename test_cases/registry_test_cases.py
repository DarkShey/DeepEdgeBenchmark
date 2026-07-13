"""
test_cases/registry_test_cases.py — Tableau des test cases (transitions de régime)
===================================================================================
Chaque test case est ancré sur une transition de régime de marché (colonne `Regime`
déjà calculée dans test_cases/data/<ticker>.csv, sortie de calibration/regime) :
`regime_from` (jour D-1) -> `regime_to` (jour D, 1er jour du nouveau régime).

`extra_filter` référence par nom une fonction de test_cases/transitions.py appliquée
en plus du changement de régime brut, pour ne garder que les transitions "confirmées"
(cf. TC1.2 ci-dessous). None = aucun filtre additionnel, toute transition de régime
brute compte.

Pour ajouter un test case, voir test_cases/README.md.
"""

TEST_CASES = [
    {
        "id": "TC1.2",
        "name": "bull_stress",
        "label": "Bull → Stress",
        "description": (
            "Transition bull→stress confirmée : en plus du changement de régime, le "
            "prix réel au jour D casse déjà la borne basse à 95% qu'une bande de "
            "volatilité GARCH (colonne Sigma_t_pct) aurait projetée la veille (D-1) pour le "
            "jour D — filtre P-market(D) < PI-low(D), cf. transitions.py::tc12_stress_filter."
        ),
        "regime_from": "bull",
        "regime_to": "stress",
        "extra_filter": "tc12_stress_filter",
        "horizons_days": [1, 7],
    },
    {
        "id": "TC1.3",
        "name": "bear_calm",
        "label": "Bear → Calme",
        "description": "Transition bear→calm brute (changement de régime, sans filtre additionnel).",
        "regime_from": "bear",
        "regime_to": "calm",
        "extra_filter": None,
        "horizons_days": [1, 7],
    },
    {
        "id": "TC1.4",
        "name": "bear_stress",
        "label": "Bear → Stress",
        "description": "Transition bear→stress brute (changement de régime, sans filtre additionnel).",
        "regime_from": "bear",
        "regime_to": "stress",
        "extra_filter": None,
        "horizons_days": [1, 7],
    },
]

TEST_CASE_BY_ID = {tc["id"]: tc for tc in TEST_CASES}

# Nombre d'occurrences historiques (les plus récentes) retenues par (actif x test case) -
# borne le coût de calcul (LSTM notamment). Configurable via --max-occurrences en CLI.
MAX_OCCURRENCES_PER_ASSET = 3
