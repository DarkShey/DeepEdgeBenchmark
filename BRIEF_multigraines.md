# BRIEF — Robustesse multi-graines (+ entraînement global) du duel

> Origine : rapports de l'audit `rapport_methodologie.pdf` (27/07) et `rapport_audit_code2.pdf`
> (28/07). Ce brief traite **la seconde moitié de la recommandation #5** de l'audit (« renforcer
> les deux bras » — côté diffusion) : la moitié « baselines fortes » est déjà livrée
> (`BRIEF_baselines_fortes.md`). Il lève les lignes **« Graines et variance »** et **« Entraînement
> global vs par actif »** de la grille d'audit, et le reste de l'écart **É4**.
>
> Une fois ce brief terminé, **la reco #5 est complète** et le verdict du duel devient présentable
> comme définitif.
>
> Hors périmètre (chantiers suivants) : diagnostic + itération de la calibration (reco #4),
> emballage de l'annexe statistique (reco #6), portage aux pipelines de production A & B,
> repeuplement du régime C.

## 0. Le problème, en une phrase

Tout le duel tourne sur **une seule graine (`seed = 42`)**. Or la variance inter-graines est le
facteur dominant des benchmarks ML (Bouthillier et al. 2021, cité par l'audit) : un verdict de
modèle génératif sur une seule graine n'est pas défendable. Il faut montrer que le résultat
(TSDiff dans le MCS 15/15, rien ne bat le naïf, skill seulement contre Prophet/LSTM) est **stable
d'une graine à l'autre**, pas un artefact de la 42. Second point de l'audit : TSDiff est entraîné
**par actif**, configuration sous-optimale pour un modèle de diffusion — à tester contre un
entraînement **global** (multi-actifs).

## 1. Ce qui existe déjà et qu'on réutilise (ne pas réécrire)

- Tout le duel déjà livré et renforcé : `duel_backtest.py` (assemblage + run), `duel_origins.py`,
  `duel_sampling_adapters.py` (baselines fortes incluses), `crps_metrics.py::crps_fair`,
  `duel_pairwise_tests.py`, `mcs.py`, `epoch_sweep.py::three_way_split`.
- `models/tsdiff_model.py` — entraînement / échantillonnage TSDiff (gère la graine, `N_SAMPLES`,
  `EPOCHS`).
- `experiments/NOTE_duel_diffusion_vs_classiques.md` — la NOTE à compléter (elle a déjà les grilles
  duel + baselines fortes).

**Ce qui manque : une boucle multi-graines autour du duel, l'agrégation de la dispersion, et un
mode d'entraînement global de TSDiff.**

## 2. Les briques à construire

### 2.1 Boucle multi-graines autour du duel (brique obligatoire)

- Rejouer le duel **complet** (déjà renforcé par les baselines fortes) sur **S graines** — viser
  **S ≥ 5** (déclarer S ; si le coût l'impose, S = 5 minimum, jamais 1).
- Chaque graine ré-arme **toute** la stochastique du run : init + échantillonnage TSDiff, MC-Dropout
  LSTM, tirages des trajectoires classiques (GARCH/SARIMA/Prophet). Les bootstraps de test dérivent
  de façon déterministe de la graine du run (on mesure la variance des **modèles**, pas du test).
- **Rien d'autre ne change** : mêmes origines, gel à T0, `three_way_split` (sélection hors test),
  fair CRPS, DM-HAC-HLN + bootstrap + Clark-West + Holm, MCS + SPA.

### 2.2 Agrégation de la dispersion (le vrai livrable de robustesse)

Par actif × horizon et par modèle, rapporter **sur les S graines** :

- **CRPS : moyenne ± écart-type** (ou IQR) inter-graines — la barre d'incertitude qui manquait.
- **Stabilité du MCS** : dans quelle fraction des S graines chaque modèle appartient au MCS
  (ex. « TSDiff dans le MCS dans 15/15 cases sur 5/5 graines »).
- **Stabilité des verdicts significatifs** : les cases Holm significatives et les verdicts poolés
  sont-ils les **mêmes** d'une graine à l'autre ? Signaler toute case qui bascule.
- **Le rejet SPA vs GARCH(1,1)** tient-il sur toutes les graines ?

### 2.3 Entraînement global vs par actif (brique design de la reco #5)

- Ajouter un mode où TSDiff est entraîné **globalement** (une seule fois sur les 5 actifs
  conjointement) au lieu de par actif, puis évalué sur les mêmes origines/cibles.
- Comparer global vs par-actif **à armes égales** (mêmes graines, même budget), et dire si le
  standing de TSDiff (CRPS, MCS) change.
- Si le coût est prohibitif, le faire sur un **sous-ensemble de graines déclaré** (≥ 2) et
  documenter l'approximation — ne jamais le sauter en silence.

### 2.4 Budgets déclarés (exigé par la reco #5)

- Documenter noir sur blanc, par modèle : époques, données (fenêtre), `m`, HPO éventuel, nb de
  graines. C'est la transparence de budget que l'audit réclame.

## 3. Le livrable de sortie

- `duel_backtest.json` régénéré avec l'axe « graine » (résultats par graine + agrégats).
- `NOTE_duel_diffusion_vs_classiques.md` complétée : **une section « robustesse multi-graines »**
  (CRPS moyen ± dispersion, stabilité MCS/verdicts sur les S graines) + le **comparatif global vs
  par-actif** + le tableau des budgets déclarés.
- **Une phrase de conclusion franche** : le verdict est-il stable sur les graines, oui ou non ? Si
  une case bascule, la nommer.

## 4. Critères de conformité (à vérifier avant de clore)

| Point | Critère (niveau 1) | Statut visé |
|---|---|---|
| Nombre de graines | S ≥ 5 déclaré, jamais 1 | conforme |
| Portée de la graine | ré-arme toute la stochastique modèle à chaque run | conforme |
| Dispersion rapportée | CRPS moyen ± écart-type/IQR par case et par modèle | conforme |
| Stabilité du verdict | fraction de graines où MCS / cases Holm / poolé tiennent | conforme |
| Entraînement global | mode global testé vs par-actif, à graines/budget égaux | conforme |
| Budgets déclarés | époques / données / m / HPO / graines documentés par modèle | conforme |
| Protocole du duel | origines / gel T0 / CRPS / tests / MCS strictement inchangés | conforme |

## 5. Garde-fous d'exécution

- Travail **directement sur `main`** → **commits atomiques**, **`pytest` vert à chaque étape**.
- **Checkpoint par (graine, actif)** : le run doit être **reprenable** (ne pas tout perdre si ça
  s'interrompt). Lancer sous `caffeinate` (les lenteurs venaient des mises en veille du Mac, pas du
  calcul).
- **Ne toucher QUE l'axe graine + le mode d'entraînement TSDiff** : tout le reste du protocole
  reste identique et vérifié.
- **Ne pas** toucher aux pipelines de production A & B (chantier séparé).
- Toute réduction de périmètre pour raison de coût (S plus petit, global sur 2 graines) est
  **écrite noir sur blanc**, jamais masquée.

## 6. Suite (chantiers ultérieurs, pour mémoire)

Reco #4 (diagnostic + itération de la calibration : PIT de rang, Kupiec/Christoffersen,
recalibration conformale) · reco #6 (emballer l'annexe statistique) · portage aux pipelines de
production A & B + ablation overlay régime · repeuplement du régime C · synthèse finale.
