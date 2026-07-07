# Run 20260707 — validation business (holdout 7 jours)

Ce dossier a été assemblé manuellement (rétroactivement) pour ce run précis, car il a
été lancé avant l'ajout de `export_run_bundle()` dans `generate_test_cases.py`. Les
runs suivants génèreront ce dossier automatiquement (voir `scripts/generate_test_cases.py`,
fonction `export_run_bundle`).

## Contenu

- `meta_data.json` — paramètres du run (assets, modèles, horizons, holdout, cutoff).
- `results.xlsx` — les 50 test cases (prédiction, réel si déjà connu, erreur %, statut),
  + un onglet `synthese_mape` avec la MAPE par modèle (formules Excel, pas de valeurs figées).
- `tests_smoke.xlsx` — les tests effectués par Claude pendant la session (persistance,
  dédoublonnage, règles d'intégrité/plausibilité, mécanisme de holdout, export du bundle).
  Les "Test Cases" décrits par l'utilisateur (2e type de test mentionné dans `Run/readme.md`)
  restent à ajouter séparément.
- `scripts/` — copie de `tracking_db.py`, `verdict_rules.py`, `generate_test_cases.py`
  tels qu'utilisés pour ce run.
- `chart_d1.png`, `rapport_validation.pdf`, `rapport_comparaison.pdf` — les livrables
  déjà produits pendant la session.
- `training_data/` — **vide pour ce run** : les séries de cours utilisées à l'entraînement
  n'ont pas été sauvegardées à l'époque (fonctionnalité ajoutée après ce run). Les prochains
  runs y déposeront automatiquement un CSV par actif.

## Résumé des résultats

50 test cases (5 actifs x 5 modèles x 2 horizons), 0 échec technique. 35 résolus
(comparés à la vraie valeur), 15 encore en attente (D+7 sur SPY/ZN=F/TLT — le holdout
de 7 jours calendaires ne cache que ~4 jours de trading pour ces actifs, il en faut 5
pour résoudre D+7 ; voir `rapport_comparaison.pdf`).

MAPE par modèle : Naive et les modèles statistiques (ARIMA-GARCH, SARIMA, LSTM) restent
sous 1,6 % à D+1 et sous ~7 % à D+7. Prophet décroche fortement sur BTC-USD/ETH-USD
(20 % à D+1, 36 % à D+7) — diagnostic détaillé dans `rapport_comparaison.pdf`.
