# Run 20260707 — rapports et validation business (holdout 7 jours)

Ce dossier a changé de rôle : il ne contient plus le bundle auto-généré par
`export_run_bundle()` (supprimé, cf. `validation/generate_test_cases.py`). Les
résultats par combinaison (modèle x actif x horizon) vivent désormais dans
`Run/20260707-<modèle>-<asset>-<horizon>/business_validation.json`, aux côtés
des artefacts du pipeline ML (`metrics.json`, `unit_tests.json`, ...),
conformément à `Run/readme.md`.

Ce qui reste ici, ce sont les livrables transverses (pas rattachables à une
seule combinaison) produits pendant la session du 2026-07-07 :

- `results.xlsx` — les 50 test cases (prédiction, réel si déjà connu, erreur %,
  statut) + un onglet `synthese_mape` avec la MAPE par modèle (formules Excel).
  Donnée brute équivalente désormais dans chaque `business_validation.json`.
- `tests_smoke.xlsx` — tests effectués par Claude pendant la session sur la
  mécanique de `validation/tracking_db.py` (persistance, dédoublonnage, règles
  d'intégrité/plausibilité, holdout, export) — ne teste PAS les modèles de
  prévision eux-mêmes (voir `unit_tests.json` dans chaque dossier combo pour
  ça). Les "Test Cases" décrits par l'utilisateur (2e type de test mentionné
  dans `Run/readme.md`) restent à ajouter séparément.
- `chart_d1.png`, `rapport_validation.pdf`, `rapport_comparaison.pdf` — les
  livrables déjà produits pendant la session.
- `meta_data.json` — paramètres de ce run (assets, modèles, horizons, holdout,
  cutoff) — contexte global, en plus du détail par combo.

## Résumé des résultats

50 test cases (5 actifs x 5 modèles x 2 horizons), 0 échec technique. 35
résolus (comparés à la vraie valeur), 15 encore en attente (D+7 sur
SPY/ZN=F/TLT — le holdout de 7 jours calendaires ne cache que ~4 jours de
trading pour ces actifs, il en faut 5 pour résoudre D+7 ; voir
`rapport_comparaison.pdf`).

MAPE par modèle : Naive et les modèles statistiques (ARIMA-GARCH, SARIMA,
LSTM) restent sous 1,6 % à D+1 et sous ~7 % à D+7. Prophet décroche fortement
sur BTC-USD/ETH-USD (20 % à D+1, 36 % à D+7) — diagnostic détaillé dans
`rapport_comparaison.pdf`.
