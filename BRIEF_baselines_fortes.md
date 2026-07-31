# BRIEF — Mettre les baselines classiques dans leur configuration forte

> Origine : rapports du tuteur `rapport_methodologie.pdf` (27/07) et `rapport_audit_code2.pdf`
> (28/07). Ce brief traite **le chantier #3** (« baselines fortes »), qui prolonge la
> recommandation #5 de l'audit (« renforcer les deux bras ») et lève la réserve sur l'écart **É4 /
> baselines**. Il vient APRÈS le duel unifié (`BRIEF_unification_protocole_duel.md`, livré) dont il
> réutilise tout le protocole.
>
> Hors périmètre de CE brief (briefs suivants) : multi-graines / entraînement global (#6),
> itération de la recalibration conformale (#5), portage aux pipelines de production A & B (#7),
> repeuplement du régime C (#8), requalification du bilan + annexe statistique (#9–#10).

## 0. Le problème, en une phrase

Le duel est équitable côté protocole, mais **le camp classique n'est pas dans sa configuration
forte** : le GARCH tourne en innovations **gaussiennes uniquement** (`dist="normal"`,
`models/arima_model.py:141`), sans variante à queues lourdes ni asymétrique. Or la référence à
réfuter de la littérature, c'est le GARCH *bien réglé* (Hansen-Lunde 2005, « does anything beat a
GARCH(1,1)? »). Tant que le GARCH est bridé, le verdict « rien ne bat les classiques / le naïf »
est attaquable — un duel ne vaut que si **chaque camp est au meilleur de sa forme**.

**Objectif du brief :** renforcer les baselines classiques (GARCH-Student + GJR), rejouer le duel
et documenter si le verdict bouge — sans rien changer d'autre au protocole déjà unifié.

## 1. Ce qui existe déjà et qu'on réutilise (ne pas réécrire)

- Tout le protocole du duel : `duel_origins.py` (origines communes), `duel_backtest.py`
  (assemblage + run), `crps_metrics.py::crps_fair`, `duel_pairwise_tests.py`, `mcs.py`,
  `experiments/epoch_sweep.py::three_way_split`.
- `experiments/duel_sampling_adapters.py` — les adaptateurs qui produisent les `m = 500`
  trajectoires par modèle. **C'est le seul point d'entrée à modifier côté GARCH** : l'adaptateur
  ARIMA-GARCH doit simuler avec la nouvelle loi d'innovation.
- `models/arima_model.py` — l'ajustement GARCH (aujourd'hui `dist="normal"`, l. 141).
- `experiments/NOTE_duel_diffusion_vs_classiques.md` — la NOTE de référence à mettre à jour.

L'infrastructure est là. **Ce qui manque, c'est la loi d'innovation forte du GARCH et son câblage
dans l'adaptateur d'échantillonnage.**

## 2. Les briques à construire

### 2.1 GARCH à innovations de Student (queues lourdes)

- Passer l'ajustement de `dist="normal"` à `dist="t"` (Student), qui capte les queues lourdes des
  rendements financiers — en particulier crypto (kurtosis extrême, cf. Zhang et al. 2018 cité par
  l'audit).
- Le degré de liberté `nu` est estimé par le modèle (pas fixé à la main).

### 2.2 Variante asymétrique GJR-GARCH (effet de levier)

- Ajouter une variante **GJR-GARCH** (ou EGARCH) qui modélise l'asymétrie hausse/baisse de la
  volatilité — standard sur actions et indices.
- Innovations Student ici aussi.

### 2.3 Câblage dans l'échantillonnage réel du duel (le point critique)

- L'adaptateur ARIMA-GARCH de `duel_sampling_adapters.py` doit **simuler des trajectoires** avec les
  innovations Student/GJR ajustées (tirage des chocs dans la loi de Student, propagation de la
  variance conditionnelle), puis agréger sur l'horizon.
- **Interdit** : revenir à une borne analytique `1.96·σ` ou à une reconstruction gaussienne. On
  échantillonne le **modèle réel**. `m = 500` inchangé.

### 2.4 Sélection propre de la spécification

- Le choix de la loi d'innovation / variante (normal vs t vs GJR-t) se fait **uniquement sur le
  bloc de validation** de `three_way_split`, **jamais sur le test** (verrou É1 préservé) — ou est
  fixé a priori et déclaré. Aucune optimisation sur le jeu de test.

## 3. Le livrable de sortie

- `experiments/duel_backtest.json` et `NOTE_duel_diffusion_vs_classiques.md` régénérés avec les
  baselines fortes.
- **Un bloc « avant / après » explicite** dans la NOTE : le verdict bouge-t-il ? Un GARCH-Student
  ou GJR entre/sort-il du MCS quelque part ? Bat-il maintenant TSDiff ? Quelque chose bat-il enfin
  le naïf ? Réponse franche dans les deux sens.
- Ligne « Baselines classiques » de la grille §4 : de « partiel » à **conforme**.

## 4. Critères de conformité (à vérifier avant de clore)

| Point | Critère (niveau 1) | Statut visé |
|---|---|---|
| GARCH innovations | Student (`dist="t"`), `nu` estimé | conforme |
| Variante asymétrique | GJR-GARCH (ou EGARCH) présente et évaluée | conforme |
| Échantillonnage | vraies trajectoires simulées, `m=500`, pas de borne analytique ni gaussienne | conforme |
| Sélection de spéc. | sur validation uniquement, jamais sur le test | conforme |
| Protocole du duel | origines / gel T0 / CRPS / tests / MCS strictement inchangés | conforme |
| Traçabilité | verdict avant/après documenté, dans les deux sens | conforme |

## 5. Garde-fous d'exécution

- Travail **directement sur `main`** → **commits atomiques**, **`pytest` vert à chaque étape**,
  rien de cassé dans l'existant.
- **Ne toucher QUE la famille classique** (GARCH) et son adaptateur : SARIMA, Prophet, LSTM, Naive,
  TSDiff et tout le reste du protocole restent identiques.
- **Ne pas** toucher aux pipelines de production A & B (chantier #7).
- **Ne pas** lancer la multi-graine ici : ce run reste sur `seed = 42` pour rester comparable à la
  NOTE actuelle ; la multi-graine (#6) englobera ce #3 juste après.
- Toute réserve résiduelle (ex. une spéc. qui ne converge pas sur un actif) est **écrite noir sur
  blanc**, jamais masquée.

## 6. Suite (briefs ultérieurs, pour mémoire)

#6 multi-graines + entraînement global · #5 itération recalibration conformale · #7 portage aux
pipelines de production A & B + ablation overlay régime · #8 repeuplement du régime C · #9–#10
requalification du bilan + annexe statistique.
