# BRIEF — Construire le « duel » diffusion vs classiques qui n'existe pas encore

> Origine : rapports du tuteur `rapport_methodologie.pdf` (27/07) et `rapport_audit_code2.pdf`
> (28/07). Ce brief traite **la recommandation #2 de l'audit** (« construire le duel qui n'existe
> pas ») — le verrou qui débloque tout classement légitime. Il couvre les écarts **É2** (asymétrie
> de protocole), **N1** (aucun CRPS diffusion-vs-classiques dans le code) et une partie de **É3/É6**
> (tests et agrégation appliqués à la bonne comparaison), plus l'ajout du **MCS** (absent du dépôt).
>
> Hors périmètre de CE brief (briefs suivants) : baselines fortes GARCH-Student (#3), itération de
> la recalibration conformale (#5), multi-graines / entraînement global (#6), portage aux pipelines
> de production A & B (#7), repeuplement du régime C (#8).

## 0. Le problème, en une phrase

Le duel « la diffusion est-elle meilleure que les classiques ? » **n'est calculé nulle part dans le
code**. Le seul CRPS empirique correct du dépôt (`experiments/crps_metrics.py`) compare la diffusion
**à elle-même** (TSDiff-W vs TSDiff-D). Les pipelines qui comparent réellement diffusion et
classiques (A et B) ne calculent **aucun CRPS** et souffrent d'une asymétrie de protocole (TSDiff
gelé à T0, les 5 autres refit à chaque origine) qu'**aucun test ne corrige**. `pooled_analysis.py`
refuse d'ailleurs explicitement de classer les modèles entre eux pour cette raison.

**Objectif du brief :** un backtest commun, apparié, où les 6 modèles sont évalués dans des
conditions **strictement identiques**, scoré en CRPS équitable, testé statistiquement par paire,
résumé par un Model Confidence Set. C'est la seule voie vers un verdict interprétable.

## 1. Ce qui existe déjà et qu'on réutilise (ne pas réécrire)

- `experiments/crps_metrics.py::crps_empirical` — CRPS forme énergie, testé contre `properscoring`.
- `honest_eval/metrics.py::dm_hac_test` — DM avec HAC Bartlett tronqué à `max(h-1, ⌊T^(1/3)⌋)` +
  correction Harvey-Leybourne-Newbold + quantiles de Student. **Correctement implémenté**, il suffit
  de l'appliquer aux bonnes paires.
- `experiments/paired_test.py::paired_block_bootstrap_test` — bootstrap apparié par blocs
  (`block_length=3`, aligné sur le chevauchement W1–W3), avec `effective_n` rapporté.
- `experiments/pooled_analysis.py` — `compute_asset_scales` (échelle MASE par actif), `dual_test`
  (DM-HAC + bootstrap sous règle de concordance), `holm_correction`, fusion en classes d'actifs.
- `experiments/epoch_sweep.py::three_way_split` — split imbriqué train / 12 origines validation /
  30 test, disjoints et chronologiques (déjà utilisé par `weekly_headtohead_v2.py`).
- `models/tsdiff_model.py::fit_tsdiff` / `forecast_from_fitted` — produit un **nuage de
  trajectoires** ; `experiments/weekly_headtohead_v2.py::random_walk_samples` — tirage du naïf.

L'infrastructure de test et de scoring est là. **Ce qui manque, c'est l'assemblage d'un backtest
commun et les adaptateurs d'échantillonnage des 5 modèles classiques.**

## 2. Les cinq briques à construire

### 2.1 Un générateur d'origines commun (résout É2 — cœur du verrou)

Un module unique produit la liste des origines de backtest et l'impose aux 6 modèles :

- **Mêmes origines, mêmes dates-cibles** pour tous les modèles (backtest apparié, principe P2).
- **Même règle de ré-estimation** pour tous : soit tout le monde refit à chaque origine, soit
  tout le monde gelé à T0. L'asymétrie actuelle (TSDiff gelé, les autres refit) est **interdite** ;
  si une asymétrie est inévitable pour raison de coût, elle est **déclarée et son biais quantifié**.
- **Origines glissantes** (rolling origin, Tashman 2000), fenêtre glissante ou extensible déclarée.
- **Purge + embargo** pour les étiquettes chevauchantes (horizons > 1 jour) : aucun exemple
  d'entraînement dont la fenêtre d'étiquette chevauche le test (López de Prado). Point d'audit
  prioritaire, aujourd'hui absent partout.
- Splits ancrés sur `three_way_split` : sélection d'hyperparamètres uniquement sur le bloc de
  validation antérieur, **jamais sur le test** (verrou É1, déjà respecté côté weekly — le préserver).

### 2.2 Des adaptateurs d'échantillonnage pour les 5 modèles classiques (résout N1)

Chaque modèle classique doit produire, à chaque origine et horizon, **`m` trajectoires simulées**
(pas seulement point + IC analytique), avec **`m` identique** à TSDiff (cible `m = 500`) :

- **ARIMA-GARCH** : simuler des trajectoires de rendements via le processus GARCH ajusté (au lieu
  de la borne analytique `1.96·σ`), puis agréger.
- **SARIMA / Prophet** : tirer des trajectoires de la loi prédictive du modèle (simulation du
  state-space / échantillonnage de la loi Prophet), pas seulement `conf_int()`.
- **LSTM** : nuage par MC-Dropout ou bootstrap des résidus (le réseau ne sort qu'un point).
- **Naive** : `random_walk_samples` existe déjà — le réutiliser.

Interdit : la **reconstruction paramétrique** (gaussienne/log-normale des bornes d'IC stockées)
utilisée aujourd'hui pour 5 modèles sur 6 — elle détruit l'information distributionnelle que le
duel est censé mesurer (réserve N1 de l'audit). Chaque adaptateur échantillonne le **modèle réel**.

### 2.3 Un CRPS équitable, identique pour tous (résout la moitié de É3)

- Scorer avec `crps_empirical` sur les nuages de `m = 500` trajectoires, **`m` strictement
  identique** entre tous les modèles (le biais `E|X1−X2|/2m` devient commun, donc neutre).
- **Recommandé** : ajouter la variante **fair CRPS** (Ferro 2014) pour supprimer le biais résiduel,
  utile si un modèle garde un CRPS analytique exact face à un modèle échantillonné.
- **Même grille de quantiles** si une approximation pinball est utilisée quelque part.

### 2.4 Les tests, appliqués à la bonne comparaison (résout É3)

- Pour chaque paire (diffusion, classique) × actif × horizon : `dm_hac_test` sur le différentiel
  de CRPS, **HAC tronqué ≥ h−1** + HLN (déjà géré par la fonction), doublé du bootstrap par blocs.
- **Clark-West** pour toute comparaison contre un benchmark emboîté (marche aléatoire).
- **Correction de multiplicité** sur la grille complète (5 actifs × 3 horizons × paires) via
  `holm_correction` déjà présent (ou Romano-Wolf).
- Normalisation d'échelle par actif (MASE, `compute_asset_scales`) avant tout pooling, et fusion
  des actifs corrélés en classes — la machinerie de `pooled_analysis.py` est réutilisée.

### 2.5 Un Model Confidence Set par case (comble un manque — MCS absent du dépôt)

- Implémenter le **MCS de Hansen-Lunde-Nason (2011)** par actif × horizon : il livre l'ensemble des
  modèles statistiquement indiscernables du meilleur — c'est la formulation correcte de « jeu égal »,
  bien supérieure à « on ne rejette pas H0 ».
- Optionnel mais recommandé : test **SPA de Hansen** contre GARCH(1,1) comme benchmark privilégié.
- `grep` confirme qu'aucun MCS/SPA n'existe aujourd'hui dans le dépôt : c'est du code neuf, testé.

## 3. Le livrable de sortie

Un artefact reproductible (JSON + une table) qui, par actif × horizon, donne : CRPS de chaque
modèle (fair CRPS, `m=500`), les p-valeurs DM-HLN corrigées par paire, la taille effective
d'échantillon, et **l'appartenance au MCS**. Plus une **conclusion honnête** :

> À remplacer partout : « jeu égal » / « globalement moins bon » → *« au niveau de confiance usuel
> et avec les échantillons disponibles, tel modèle appartient / n'appartient pas au Model Confidence
> Set du meilleur ; aucune famille ne démontre de skill contre le naïf »* (rappel N3 : 1 verdict
> favorable sur 440 en production). Le verdict ne se lit qu'après unification du protocole.

## 4. Critères de conformité (grille d'audit — à vérifier avant de clore)

| Point | Critère (niveau 1) | Statut visé |
|---|---|---|
| Estimateur CRPS | fair CRPS ou naïf à `m` identique ; même grille de quantiles | conforme |
| Nombre de tirages `m` | identique (≥ 500) entre diffusion et classiques | conforme |
| Splits temporels | rolling origin + purge/embargo pour h > 1 j | conforme |
| Ré-estimation | **identique entre familles**, ou asymétrie déclarée + quantifiée | conforme |
| Échantillonnage classiques | vraies trajectoires du modèle, pas de reconstruction gaussienne | conforme |
| Tests | DM-HAC(≥h−1)-HLN par paire + CW vs naïf + Holm sur la grille | conforme |
| Verdict multi-modèles | MCS par case (+ SPA vs GARCH(1,1)) | conforme |
| Agrégation inter-actifs | échelle MASE + classes d'actifs, jamais de CRPS bruts moyennés | conforme |

## 5. Garde-fous d'exécution

- Travail **directement sur `main`** (choix acté) → **commits atomiques**, un chantier par commit,
  **tests verts à chaque étape** (`pytest`), rien de cassé dans les pipelines existants.
- Ne pas toucher aux pipelines de production A & B dans ce brief (chantier #7, plus tard).
- Ne rien recalculer sur les artefacts contaminés (`weekly_headtohead_results*.json`) : le nouveau
  duel repart d'origines propres via `three_way_split`.
- Toute asymétrie résiduelle non résolue est **écrite noir sur blanc** dans le livrable, jamais
  masquée (comme le fait déjà `pooled_analysis.py`).

## 6. Suite (briefs ultérieurs, pour mémoire)

#3 baselines GARCH-Student/GJR · #5 itération recalibration conformale · #6 multi-graines +
entraînement global · #7 portage aux pipelines de production A & B + ablation overlay régime ·
#8 repeuplement du régime C · #9–#10 requalification du bilan + annexe statistique.
