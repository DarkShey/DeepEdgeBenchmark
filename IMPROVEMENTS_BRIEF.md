# Brief d'amélioration — DeepEdgeBenchmark

**Date :** 2026-07-08
**Contexte :** L'analyse du dashboard (run 20260707) montre que les prévisions J+1 de tous les modèles suivent le marché avec un retard d'un jour : corr(pred_t, prix_{t-1}) ≈ 1.000, corrélation des variations prédites/réalisées ≈ 0, précision directionnelle 43–54 % (pile ou face). Les modèles prédisent en pratique « le prix d'hier ± ε ». Ce n'est pas un biais corrigible mais un artefact d'évaluation : le walk-forward 1-pas ré-ancre chaque prévision sur le dernier prix observé, et l'affichage en niveaux masque l'absence de pouvoir prédictif. Ce brief définit 4 chantiers pour rendre le benchmark honnête et discriminant, plus un correctif préalable.

---

## Point 0 (préalable) — Corriger la baseline Naïve du pipeline dashboard

**Problème.** La baseline « Naive » du pipeline qui génère le dashboard n'est pas le dernier prix observé : elle contient un bruit injecté (~3 % du prix, identique sur les 5 actifs). Son RMSE est gonflé ×1,5 à ×10 (SPY : 22,0 affiché vs 6,28 réel ; ZN=F : 3,36 vs 0,31). Toutes les colonnes « vs Naïf » et les tests DM du dashboard sont invalidés.

**Statut : corrigé.** Après fusion du pipeline externe dans le repo, la source a été identifiée : `models/naive_model.py` appliquait un tirage uniforme ±5 % « by design » (écart-type 5 %/√3 ≈ 2,9 %, exactement le bruit mesuré). Le fichier est désormais une persistence stricte (prédiction = clôture de la veille, PI gaussien marche aléatoire), l'adaptateur multi-horizon (`benchmarks/multi_horizon.py`) suit la même convention (point = dernier prix, PI ∝ σ√h), et `models/test_naive_model.py` + `honest_eval.naive.verify_naive()` verrouillent le critère d'acceptation.

**Actions.**
- Localiser et supprimer l'injection de bruit dans la génération de la baseline (pipeline dashboard).
- Régénérer les 50 records et vérifier : RMSE naïf BTC D+1 ≈ 1 864, SPY ≈ 6,3, ZN=F ≈ 0,31.
- Compléter les runs BTC-USD manquants dans dashboard (1).html (seul LSTM est présent).

**Critère d'acceptation.** Pour chaque actif : |RMSE_naïf_dashboard − RMSE_naïf_recalculé| < 0,1 %. Les prédictions naïves coïncident exactement avec la clôture de la veille.

---

## Point 1 — Évaluer les variations, pas les niveaux

**Objectif.** Rendre visible le (non-)pouvoir prédictif en scorant Δpred_t = pred_t − prix_{t-1} contre Δréel_t = prix_t − prix_{t-1}.

**Actions.**
- Ajouter les métriques : **MASE** (MAE / MAE naïf), **Theil's U** (RMSE / RMSE naïf), corrélation des variations, précision directionnelle avec IC binomial 95 %, **Diebold-Mariano vs naïf corrigé**.
- Ajouter au dashboard un panneau « variations » : nuage de points Δpred vs Δréel + série temporelle des variations. Supprimer ou reléguer le graphe en niveaux (il crée l'illusion de skill).
- Règle de lecture : U ≈ 1 et DM non significatif ⇒ le modèle n'apporte rien vs naïf ; l'afficher explicitement.

**Critère d'acceptation.** Le tableau KPI affiche MASE, U, DirAcc±IC et DM pour chaque (modèle, actif, horizon). Résultat attendu sur les données actuelles : U ≈ 1,00 pour ARIMA/SARIMA, > 1 pour LSTM/Prophet.

---

## Point 2 — Fiabiliser le schéma de validation (walk-forward reste le bon cadre)

**Objectif.** Garder le walk-forward (rolling origin) comme schéma principal, mais tester sa robustesse et éliminer tout risque de fuite lors du tuning.

**Actions.**
- Comparer **fenêtre expansive vs fenêtre glissante fixe** (ex. 250 / 500 / 750 jours) : instabilité des paramètres ⇒ écart entre les deux.
- Pour tout tuning d'hyperparamètres (LSTM, Prophet, ordres ARIMA) : **validation croisée bloquée avec purge et embargo** (type López de Prado) — jamais de K-fold aléatoire.
- Reporter les métriques par **sous-périodes / régimes** (ex. trimestres, ou régimes de volatilité) pour vérifier la stabilité, pas seulement la moyenne globale sur 165 jours.

**Critère d'acceptation.** Script de CV réutilisable (`validation.py`) avec fenêtres paramétrables + purge/embargo ; rapport comparatif expansif vs glissant sur au moins BTC et SPY.

---

## Point 3 — Évaluation multi-pas sans ré-ancrage (D+7 / D+30)

**Objectif.** À horizon > 1 jour, l'astuce « copier le dernier prix » se dégrade : c'est là que les modèles peuvent se différencier. Le D+7 actuel (n=10) est trop pauvre — les couvertures à 100 % n'ont aucun sens statistique.

**Actions.**
- Remplacer l'échantillonnage clairsemé (10 origines espacées de ~17 j) par un **rolling origin quotidien avec chevauchement** : une prévision à 7 jours émise chaque jour ⇒ ~158 évaluations sur la fenêtre de validation.
- Ajouter l'horizon **D+30** avec le même protocole.
- Corriger l'inférence pour le chevauchement : erreurs autocorrélées ⇒ écarts-types **Newey-West** (troncature ≥ h−1) dans les tests DM.
- Tracer RMSE/MASE **en fonction de l'horizon** (h = 1…30) : courbe de dégradation par modèle vs naïf.

**Critère d'acceptation.** n ≥ 150 évaluations par actif pour D+7 ; courbes erreur-vs-horizon dans le dashboard ; DM avec correction Newey-West.

---

## Point 4 — Reformuler les cibles là où il y a du signal

**Objectif.** Le niveau de prix à J+1 est essentiellement imprévisible. Réorienter le benchmark vers des cibles où un modèle peut battre le hasard.

**Actions.**
- **Volatilité** : faire de la prévision de vol la cible principale du couple GARCH (les PI à 95 % sont déjà bien calibrés : couverture 94,55 % sur BTC D+1). Métriques : QLIKE, MSE sur variance réalisée, calibration des quantiles (PIT), Winkler.
- **Direction** : cadrer la hausse/baisse comme classification binaire ; métriques AUC, Brier score, test binomial vs 50 % ; comparer à une baseline « toujours hausse ».
- **Rendements avec features exogènes** : enrichir LSTM/modèles stat de variables (vol réalisée, volume, taux, dollar index, funding rates crypto) et tester si U < 1 devient atteignable.
- Prioriser dans cet ordre : vol → direction → rendements (du plus au moins prometteur).

**Critère d'acceptation.** Un onglet dashboard par cible (vol / direction) avec baselines dédiées ; conclusion explicite par actif : « bat / ne bat pas la baseline » avec p-value.

---

## Séquencement proposé

| Ordre | Chantier | Dépendance | Effort estimé |
|---|---|---|---|
| 1 | Point 0 — fix baseline | — | faible |
| 2 | Point 1 — métriques variations | Point 0 | faible |
| 3 | Point 3 — multi-pas dense | Point 0 | moyen |
| 4 | Point 2 — CV robuste | — | moyen |
| 5 | Point 4 — nouvelles cibles | Points 1–3 | élevé |

**Livrable final :** dashboard régénéré où chaque affirmation de performance est mesurée contre la baseline correcte, avec incertitude statistique explicite.
