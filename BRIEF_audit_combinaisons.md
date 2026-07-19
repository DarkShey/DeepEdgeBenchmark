# BRIEF — Audit de couverture de la matrice (modèle × fréquence × horizon × actif)

## 0. Contexte

Le duel TSDiff weekly vs daily a répondu à une question **étroite** (une seule famille de
modèle, 2 actifs). Entre-temps, la base `validation/tracking.db` a été enrichie de trois
attributs qui décrivent précisément **comment** une prédiction a été produite :

- **`frequence`** — fréquence d'**entraînement** du modèle : `daily` ou `weekly`.
- **`horizon_type`** — nature de l'horizon visé : `daily` ou `weekly`.
- **`horizon_unit`** — le pas précis : `D+1`, `D+7`, `W+1`, `W+2`, `W+3`, …

Ces trois champs distinguent des cas qui étaient jusqu'ici confondus, notamment le cas
**croisé** : un modèle **entraîné en daily** mais qui **vise un horizon weekly** (= le
multi-pas, ex. TSDiff-D). C'est cette granularité qui rend un vrai audit possible.

## 1. Objectif

Auditer **systématiquement** l'espace des configurations : cartographier ce qui a été testé,
ce qui **manque**, et **classer les combinaisons par performance réelle** pour faire émerger
celles qui marchent (globalement et par classe d'actif). On sort de l'hypothèse unique
« weekly vs daily sur TSDiff » pour regarder toute la matrice.

## 2. La matrice complète

Dimensions à croiser :

- **Modèle** (6) : ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive, TSDiff.
- **Fréquence d'entraînement** (2) : daily, weekly.
- **Type d'horizon** (2) : daily, weekly.
- **Unité d'horizon** : D+1, D+7 (daily) ; W+1, W+2, W+3 (weekly).
- **Actif** (5) : BTC-USD, ETH-USD, SPY, ZN=F, TLT.

Toutes les paires (fréquence, horizon_type) ne sont pas également pertinentes — les trois
régimes qui ont un sens sont :

| Régime | frequence | horizon_type | Exemple | Testé aujourd'hui ? |
|---|---|---|---|---|
| **A. Daily natif** | daily | daily | D+1, D+7 | ✅ les 6 modèles |
| **B. Daily → weekly (multi-pas)** | daily | weekly | daily poussé à W+1..3 | ⚠️ TSDiff seulement |
| **C. Weekly natif** | weekly | weekly | W+1, W+2, W+3 | ⚠️ TSDiff seulement |

## 3. État des lieux (hypothèse à confirmer par requête)

D'après le code (`model_artifacts/pipeline.py`, `experiments/`), la couverture actuelle est
très **déséquilibrée** :

- **Régime A (daily natif)** : couvert pour les 6 modèles, horizons D+1 et D+7, sur les 5
  actifs. C'est le socle historique.
- **Régimes B et C (weekly)** : couverts **uniquement pour TSDiff**. Les 5 autres modèles
  (ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive) **n'ont jamais été testés en weekly**, ni natif
  ni multi-pas.

**Le trou principal est donc clair** : tout le pan weekly × {ARIMA, SARIMA, Prophet, LSTM,
Naive}. C'est précisément l'extension « étendre le prototype weekly à tous les modèles »
évoquée dès le départ.

> La première tâche de l'audit est de **confirmer cet état par une vraie requête** sur
> `tracking.db` (matrice remplie vs vide), pas de le supposer.

## 4. Comment juger qu'une combinaison est « bonne »

La base trace déjà les métriques nécessaires par prédiction (`predictions`) :

- `abs_error` → **RMSE / MAE** agrégés (précision du point).
- `in_interval` → **couverture** de l'intervalle (calibration ; cible ≈ 0.95).
- `direction_correct` → **exactitude directionnelle** (utile pour un usage trading).
- `beats_naif` / `abs_error_naif` → **skill vs naïf** (un signal parmi d'autres, pas la
  finalité — l'objectif est de trouver les meilleures combinaisons, pas de refaire le procès
  de la marche aléatoire).
- `regime` → permet de segmenter la performance par régime de marché.

**Définition d'une bonne combinaison** (à agréger par modèle × régime d'entraînement × horizon
× actif) : bonne précision (RMSE bas) **et** intervalle bien calibré (couverture proche de
0.95, ni sur- ni sous-couvrant), idéalement avec une exactitude directionnelle > 50 %. Une
combinaison précise mais sur-confiante (couverture ≪ 0.95) n'est pas « bonne ».

## 5. Plan d'implémentation

1. **Requête d'audit de couverture** : produire la matrice (modèle × frequence × horizon_type
   × horizon_unit × actif) avec le nombre de prédictions **et** de prédictions évaluées
   (`y_true` non nul) par cellule → identifie précisément les cellules vides.
2. **Tableau de performance** : pour chaque cellule non vide, agréger RMSE, couverture,
   exactitude directionnelle, skill vs naïf, segmentables par régime. Trier pour faire
   ressortir le top par actif et par classe d'actif.
3. **Combler les trous prioritaires** : lancer les 5 modèles manquants en weekly (régimes B et
   C), mêmes actifs/horizons/protocole que l'existant, pour compléter la matrice.
4. **Synthèse comparative** : quelles combinaisons dominent, globalement et par classe d'actif
   (crypto / actions / obligations) — en gardant à l'esprit que ce qui marche sur crypto peut
   échouer sur taux (cf. résultats TSDiff-W).
5. **Vérification** : (a) cohérence des nouveaux champs (`horizon_unit` cohérent avec
   `frequence`/`horizon_type`, pas de `W+2` étiqueté `daily`) ; (b) mêmes fenêtres/dates-cibles
   entre modèles pour une combinaison donnée (comparaison équitable) ; (c) pas de doublons
   (l'index d'unicité OOS existe déjà, le vérifier) ; (d) lire les classements en gardant en
   tête la puissance limitée par actif.

## 6. Livrables

- `experiments/audit_coverage.json` (ou vue SQL) — la matrice testé/manquant.
- Tableau de performance trié par combinaison (modèle × frequence × horizon × actif).
- Liste des cellules manquantes prioritaires + résultats une fois comblées.
- Synthèse : les combinaisons gagnantes, globales et par classe d'actif.

## 7. Ce que ce brief ne fait PAS

- Pas de nouveau procès « bat-il la marche aléatoire ? » comme finalité — `beats_naif` n'est
  qu'une colonne parmi d'autres. L'objectif est de **trouver les bonnes combinaisons**.
- Pas de migration DB ni d'extension dashboard tant que l'audit n'a pas montré des combinaisons
  qui valent la peine d'être exposées.
- Pas de recalibration des modèles existants — on audite d'abord ce qui existe, on complète la
  matrice, on décide ensuite.
