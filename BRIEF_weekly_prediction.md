# BRIEF — Prédiction hebdomadaire : TSDiff-W vs TSDiff-D (head-to-head train-once-forward)

## 0. Contexte

Le tuteur a esquissé (en local, non poussé sur le repo) un prototype **TSDiff-W** :
resample de la série en hebdomadaire + génération des semaines W+1/W+2/W+3 + backtest
hebdo. Un premier mini-test (Phase 0) a été lancé et donne un signal **bruité et non
concluant** :

| Actif | Horizon | Modèle | RMSE | Cov95 | CRPS |
|---|---|---|---|---|---|
| SPY | W1 | TSDiff-W | 17.19 | 1.00 | 9.00 |
| SPY | W1 | TSDiff-D | 14.08 | 0.17 | 11.49 |
| SPY | W2 | TSDiff-W | 16.20 | 1.00 | 10.15 |
| SPY | W2 | TSDiff-D | 14.86 | 0.00 | 12.94 |
| SPY | W3 | TSDiff-W | 20.14 | 1.00 | 11.77 |
| SPY | W3 | TSDiff-D | 12.16 | 0.33 | 9.23 |
| BTC | W1 | TSDiff-W | 7901 | 0.83 | 4448 |
| BTC | W1 | TSDiff-D | 5866 | 0.00 | 4069 |
| BTC | W2 | TSDiff-W | 12219 | 1.00 | 6121 |
| BTC | W2 | TSDiff-D | 8378 | 0.00 | 6360 |
| BTC | W3 | TSDiff-W | 16818 | 1.00 | 7048 |
| BTC | W3 | TSDiff-D | 7532 | 0.00 | 5560 |

**Lecture honnête de la Phase 0** : le daily gagne le RMSE point partout (contre-intuitif),
mais **sous-couvre catastrophiquement** (Cov95 ≈ 0.00 : la vraie valeur ne tombe jamais dans
son IC → sur-confiance). Le weekly est bien calibré (0.83–1.0, peut-être un peu large). Le
CRPS est un match nul (3/6 chacun). **Mais** : (a) 6 origines = puissance statistique quasi
nulle (une couverture « 0.17 » = 1/6) ; (b) la sur-confiance du daily est **confondue** avec
un sous-entraînement (40 epochs). Résultat *suggestif*, pas *probant*.

**État du repo au moment d'écrire ce brief** : ni le brief du tuteur, ni le code du prototype
weekly, ni `experiments/weekly_headtohead_results.json` ne sont présents (vérifié sur l'arbre
de travail, toutes les branches et tout l'historique git). On (ré)implémente donc **à partir
du TSDiff quotidien existant** (`models/tsdiff_model.py`), qui supporte déjà le mécanisme
nécessaire (voir §3).

## 1. Objectif

Trancher proprement l'hypothèse (§2) avec une **puissance statistique suffisante** et **les
deux modèles correctement entraînés**, sans exploser le budget de calcul. Concrètement :
refactorer le prototype en protocole **train-once-forward**, lancer le head-to-head à
**30 origines walk-forward × 300 epochs** sur **SPY + BTC**, et produire un verdict exploitable
(RMSE / Cov95 / CRPS par actif × horizon).

Ce brief **s'arrête à la décision**. L'extension à tous les modèles + bascule weekly/daily du
Dashboard + pipeline daily (§9) n'est déclenchée **que si** le verdict est net.

## 2. Hypothèse à tester

> Pour un horizon **hebdomadaire** (W+1, W+2, W+3), un modèle diffusion **entraîné nativement
> en hebdo** (TSDiff-W) produit-il de **meilleures prévisions** — en particulier une **meilleure
> calibration de l'incertitude** — qu'un modèle **quotidien poussé en multi-pas** (TSDiff-D) ?

Le point d'intérêt principal est la **calibration** (Cov95, CRPS), pas seulement la précision
ponctuelle (RMSE). Le pattern à confirmer/infirmer de la Phase 0 : *le daily-multistep est-il
fondamentalement sur-confiant à horizon pluri-hebdo, ou est-ce juste un artefact de
sous-entraînement ?*

## 3. Le cœur : protocole train-once-forward + le helper « from fitted »

**Problème de calcul.** La config visée (30 origines × 300 epochs × 3 actifs) avec
ré-entraînement à chaque origine ≈ **180 entraînements ≈ 15–20 h**. Infaisable en une fois.

**Décision de conception assumée** : on passe en **train-once-forward**.

- On entraîne chaque modèle **une seule fois**, sur les données **≤ première origine**.
- On prévoit ensuite aux **30 origines** en ne faisant que **ré-échantillonner** : TSDiff est
  conditionné sur la **fenêtre d'historique** passée en entrée, donc il forecast depuis
  n'importe quelle origine **sans réentraînement**.
- → **6 entraînements** au lieu de 180 (2 actifs × 2 modèles + marge) → **~1 h** avec epochs élevés.

**Validité confirmée par le code** (`models/tsdiff_model.py`) :

- `TSDiff.train()` est le **seul** endroit où les poids bougent (`opt.step` sur `net`+`decomp`+`hist_embed`).
- `TSDiff.sample_paths(hist_window, ...)` est décoré `@torch.no_grad()` : uniquement des
  forward passes conditionnés sur `hist_embed(h)` de la fenêtre fournie, **aucun** pas
  d'optimisation. → un modèle fitté forecast depuis n'importe quel historique par simple
  ré-échantillonnage.

C'est **OOS et sans lookahead** (on ne conditionne que sur du passé réalisé ≤ origine), et
**strictement identique** pour les deux modèles → comparaison équitable.

**Le helper à écrire — `forecast_from_fitted(...)`** : à partir d'un modèle **déjà entraîné**
+ les stats de standardisation figées (`mu`, `sd`) + le dernier prix + une fenêtre
d'historique jusqu'à une origine donnée, il retourne les échantillons de prévision (chemins)
aux horizons W1/W2/W3. C'est essentiellement ce que fait déjà la boucle interne de
`run_tsdiff()`, extrait en fonction réutilisable appelée à chaque origine **sans** rappeler
`model.train()`.

## 4. Les deux modèles comparés

Même architecture TSDiff, deux régimes de données :

- **TSDiff-W (natif hebdo)** : série resamplée en **hebdomadaire** (clôture de fin de semaine,
  ex. `W-FRI`), entraîné sur les **rendements hebdo**, `horizon = 3` (→ W1/W2/W3 directement).
- **TSDiff-D (quotidien multi-pas)** : série **quotidienne**, `horizon` étendu à
  **15 jours ouvrés** (3 semaines × 5 jours de bourse), puis **agrégé en hebdo** pour
  produire W1/W2/W3.

> ⚠️ Le `HORIZON = 7` actuel du TSDiff quotidien ne couvre que ~1,4 semaine → **insuffisant**
> pour W2/W3. Le passer à **15** est indispensable pour une comparaison honnête sur les trois
> horizons. C'est un paramètre du run daily, pas une modif du modèle.

**Alignement obligatoire** : les deux modèles sont évalués sur **exactement les mêmes
dates-cibles** (mêmes fins de semaines W1/W2/W3), sinon la comparaison n'a pas de sens.

## 5. Configuration expérimentale

- **Origines** : **30**, walk-forward (puissance statistique — corrige le principal défaut de
  la Phase 0).
- **Epochs** : **300** (les deux modèles doivent être correctement entraînés — corrige le
  confounding sous-entraînement de la Phase 0).
- **Actifs** : **SPY + BTC** (reprise directe de la Phase 0 pour comparer au signal bruité
  existant).
- **Horizons** : **W1, W2, W3**.
- **n_samples** : ≥ 50 par prévision (nuage d'échantillons → point = moyenne, IC = quantiles
  2.5 / 97.5).
- **Budget** : ~1 h en mode train-once-forward (le « mode caféine » tient).

## 6. Métriques

Par combinaison **(actif × horizon × modèle)** :

- **RMSE** — précision du point (moyenne du nuage d'échantillons vs réalisé).
- **Cov95** — couverture empirique de l'IC à 95 % (fraction des réalisés dans [q2.5, q97.5]).
  Cible ≈ 0.95. **Métrique centrale** de l'hypothèse.
- **CRPS** — score point+distribution. À calculer en **CRPS empirique sur le nuage
  d'échantillons** (pas l'approximation gaussienne de `archives/`), TSDiff produisant
  nativement des échantillons.

## 7. Garde-fous (ne pas les ignorer)

- **`mu`/`sd` figés** aux stats de la **1ère origine** (calculés une fois sur le train, jamais
  recalculés aux origines suivantes → sinon micro-lookahead).
- **Historique ≤ origine uniquement** : à chaque origine, la fenêtre ne contient que du réalisé
  passé (aucun point futur).
- **Équité** : même graine, même `n_samples`, même `k_denoise`, mêmes dates-cibles pour les
  deux modèles.
- **Persistance** : le TSDiff n'a pas de `save`/`load` (hors EMA). En un seul process en
  mémoire c'est suffisant ; ajouter une sérialisation seulement si le run doit être scindé en
  plusieurs process.

## 8. Livrables

1. **`forecast_from_fitted(...)`** — le helper d'inférence multi-horizon depuis modèle fitté.
2. **Script head-to-head refactoré** (train-once-forward, 30 origines, 2 modèles, 2 actifs).
3. **`experiments/weekly_headtohead_results.json`** — résultats bruts (RMSE/Cov95/CRPS par
   combinaison).
4. **Tableau de synthèse** + lecture honnête (comme §0), avec le verdict : hypothèse
   **confirmée / infirmée / toujours non concluante**.

## 9. Critère de décision & suite (conditionnel)

**Décision** : au vu des 30 origines × 300 epochs, on regarde si le pattern de calibration
persiste une fois les deux modèles bien entraînés.

- **Si le verdict est net** (ex. le weekly reste nettement mieux calibré même à 300 epochs) →
  déclencher l'**Étape 3** : étendre le prototype weekly à **tous les modèles** (ARIMA, SARIMA,
  Prophet, LSTM, Naive, TSDiff) pour permettre au **Dashboard** de basculer **weekly ↔ daily**.
  **Bonus** : ajouter un **pipeline daily** dédié. *(Fera l'objet d'un brief séparé.)*
- **Sinon** → on documente le résultat non concluant et on n'engage pas l'extension.

## 10. Plan d'implémentation

1. **Étendre `HORIZON` du daily à 15** (paramètre de run, pas du modèle).
2. **Écrire `forecast_from_fitted(...)`** — extraire la logique d'inférence de `run_tsdiff()`
   en fonction réutilisable (échantillonnage → dé-standardisation → prix → agrégation hebdo),
   **sans** rappeler `train()`.
3. **Ajouter un resample hebdo** (`W-FRI`) pour construire la série TSDiff-W.
4. **Écrire le CRPS empirique** sur le nuage d'échantillons.
5. **Assembler le protocole head-to-head** train-once-forward (entraîner une fois ≤ origine 1,
   boucler sur 30 origines, mêmes dates-cibles pour W et D).
6. **Lancer** sur SPY + BTC (300 epochs) et sauver `experiments/weekly_headtohead_results.json`.
7. **Vérification** : contrôler l'absence de lookahead (fenêtres ≤ origine), l'alignement des
   dates-cibles W/D, et la stabilité des métriques (relancer avec une 2ᵉ graine pour estimer le
   bruit résiduel).
8. **Rédiger la lecture honnête** + le verdict (§8.4).
