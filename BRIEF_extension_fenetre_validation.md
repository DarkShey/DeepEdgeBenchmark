# BRIEF — Élargir la fenêtre de validation weekly pour gagner en puissance (dashboard D7/W1)

## 0. Contexte et cadrage honnête

Le dashboard `experiments/dashboard_d7_w1.html` compare, à horizon **W+1**, le **régime B**
(modèle daily-natif, agrégé à la cible hebdo) contre le **régime C** (weekly natif). Les deux
côtés partagent **exactement** le même `cutoff_date` et `target_date` par construction (vérifié
100 % des paires) — ce n'est PAS le vieux join D+7 « 7 jours calendaires » du régime A
(`comparison_4`), qui souffrait d'une confusion d'horizon (le D+7 crypto ciblait réellement +5
jours de bourse). Le dashboard recoupe donc **`comparison_3_daily_vs_weekly_per_model`
(filtré `horizon_unit=W+1`)** de `experiments/matrice_paired_tests.json`.

**Point de départ mesuré (base actuelle, `validation/tracking.db`) :**

- **32 origines** appariées régime B / **34** régime C, de **2025-12-05** (un vendredi, grille
  W-FRI) à **2026-06-26**. C'est piloté par `--n-test 30` (défaut `DEFAULT_N_TEST=30`,
  `experiments/epoch_sweep.py`).
- **effective_n ≈ 10** par cellule (bootstrap par blocs, `block_length=3`).
- Verdicts actuels : sur 30 cellules, **20 indistinguables**, 8 pro-daily, 2 pro-weekly. Poolé :
  Winkler crypto pro-weekly (p<0,001), sq-error pro-daily sur index/bond (significatif),
  **crypto sq-error borderline (p=0,057, non significatif)**.

La prémisse « pas assez de rigueur » n'est donc que **partiellement** vraie : la rigueur existe
déjà au niveau **poolé / par classe**, mais reste fragile pour les cas borderline et hors de
portée cellule-par-cellule. Ce brief va chercher le gain **là où il est atteignable**, sans
survendre.

---

## 1. Analyse de puissance — combien d'origines faut-il (réponse au point 5)

Calcul rétrospectif sur les CI bootstrap existantes de `experiments/dashboard_d7_w1_data.json`
(on récupère le SE depuis `ci95`, puis le nombre d'origines requis pour **80 % de puissance à
α=0,05** ; `block_length=3`, donc `origines ≈ 3 × n_eff_requis`) :

| Niveau | Métrique | État | z actuel | **Origines requises (80 % power)** |
|---|---|---|---|---|
| Poolé global | Winkler | déjà SIG | 5,29 | ~9 |
| Poolé global | sq-error | déjà SIG | 2,65 | ~34 |
| Crypto | Winkler | déjà SIG | 7,08 | ~5 |
| **Crypto** | **sq-error** | **borderline (p=0,057)** | 1,84 | **~70** |
| Index | sq-error | déjà SIG | 2,56 | ~36 |
| Bond | sq-error | déjà SIG | 3,26 | ~23 |
| Index / Bond | Winkler | vrais nuls | 0,77 | ~400 |
| **Cellule (médiane)** | RMSE | indistinguable | — | **~216 (≈ 4 ans de vendredis)** |

**Lecture de quant, sans complaisance :**

1. **Le gain réel est le poolé/par-classe borderline.** Passer de 32 à **~70 origines** fait
   franchir le seuil à la crypto sq-error. C'est la cible qui justifie l'effort.
2. **Les « indistinguables » index/bond Winkler sont de vrais nuls** (effet quasi nul, z≈0,77 →
   ~400 origines). Les élargir ne changera rien : ne pas les poursuivre, ne pas les présenter
   comme un manque de données.
3. **La significativité cellule-par-cellule est hors de portée** (médiane ~216 origines ≈ 4 ans
   de cadence hebdo). **On ne la vise pas.** Les verdicts vivent au niveau poolé/par-classe ; les
   cellules restent **descriptives**, avec `n` et `effective_n` affichés — jamais un verdict de
   cellule titré sur cette base.

**Décision (cible retenue) : `--n-test = 90`.** Justification :
- couvre les ~70 requis pour la crypto sq-error **avec marge** ;
- porte l'**effective_n de ~10 à ~30**, seuil conventionnel pour faire confiance à un CI de
  bootstrap par blocs ;
- fait remonter la première origine de 2025-12-05 vers **~octobre 2024** sur la grille W-FRI —
  donc **du backfill vers le passé** (pas d'attente hebdomadaire), ce qui répond à « récupérer
  des données d'avant ».

**Plancher acceptable si le retraining est trop long : `--n-test = 70`** (franchit la crypto
sq-error sans marge, effective_n ~23). En dessous de 70, l'effort n'a pas de sens statistique.

---

## 2. Le point « commencer le 8 janvier / lundi ou mardi » — recadrage

Il faut le dire clairement, car la consigne telle quelle est incohérente avec le protocole :

- Le **8 janvier 2026 est un jeudi** ; lundi = 5 jan, mardi = 6 jan.
- Les origines de validation **ne se choisissent pas par date calendaire ni par jour de semaine**.
  Elles sont les points de la **série resamplée en fin de semaine (W-FRI)** — donc des **vendredis
  par construction** — et le split walk-forward (`three_way_split`) prend les **`n_test` derniers**
  points en remontant depuis la fin de série. On choisit **un nombre d'origines**, pas une date de
  début : la fenêtre s'étend automatiquement vers le passé.
- **Démarrer « au 8 janvier »** reviendrait à **couper** des données (déc. 2025 existe déjà) — le
  contraire de l'objectif « plus de données ». **Changer l'ancrage au lundi/mardi** casserait
  l'appariement B/C (défini vendredi=vendredi, `W-FRI`), désalignerait le resampling weekly et
  ferait perdre le recoupement avec `comparison_3`. **Rejeté.**

**Sur l'intuition « lundi/mardi » (elle n'est pas absurde, mais elle ne touche pas l'origine).**
Un modèle weekly qui se place au **close du vendredi** pour prédire le vendredi suivant suppose
qu'on peut agir au close du vendredi ; en pratique on tradera peut-être au lundi/mardi. Mais cette
latence d'exécution est **identique des deux côtés (B et C)** → elle **s'annule dans la différence
appariée**. Ce n'est donc pas un paramètre de la comparaison de validation ; c'est une question de
**simulation de trading** (table `sim_trades`), à traiter séparément si besoin.

**Décision : on garde l'ancrage vendredi (W-FRI) et on étend vers le passé par `n_test`.**

---

## 3. Ce qu'il faut faire (exécution)

### 3.1 Régénérer les origines weekly (régimes B et C) sur une fenêtre plus large

Relancer `experiments/weekly_multimodel.py` avec **`--n-test 90`** (fallback 70) pour les
**régimes B et C**, sur les **5 actifs** (SPY, BTC, ETH, ZN, TLT) et l'horizon W+1 (les W+2/W+3
suivent si déjà dans le pipeline).

- `n_val` reste **12** ; `WEEK_MARGIN` reste **3**. Augmenter `n_test` recule d'autant le bloc de
  validation ET T0 (position de fin d'entraînement de référence) : c'est attendu et correct.
- **Coût de retraining — clarification importante.** Seuls **TSDiff et LSTM** ont un vrai coût de
  training (réseaux neuronaux ré-entraînés à chaque origine). Les 4 paramétriques (ARIMA-GARCH,
  SARIMA, Prophet, Naive) **reconstruisent leurs intervalles analytiquement** → ils couvrent les
  nouvelles origines **gratuitement**, sans GPU, rien à « forcer ». Or **les deux modèles les plus
  lents SONT justement TSDiff et LSTM** — et TSDiff est **de très loin** le plus lent (diffusion,
  échantillonnage par origine ; cf. la note de deadlock TFE_Execute dans le docstring de
  `weekly_multimodel.py` et le coût signalé dans `METHODOLOGIE_weekly_vs_daily.md`).
- **Décision : les 6 modèles** (matrice complète, 30 cellules pleines — aucune couverture inégale,
  plus propre qu'un sous-ensemble). Séquencer pour ne pas bloquer sur le plus lent :
  1. **4 paramétriques** : étendre sur toutes les nouvelles origines (gratuit, analytique).
  2. **LSTM** : retrainer en walk-forward (neuronal léger).
  3. **TSDiff** : lancer en **run par batch avec checkpoint** (`--models TSDiff`, mécanisme
     `weekly_multimodel_checkpoint.json` déjà en place) → **reprenable** si interrompu, donc sa
     longue durée n'est plus un blocage. C'est aussi le modèle **clé** de l'histoire weekly
     (calibration corrigée en natif hebdo, seul résultat significatif de
     `METHODOLOGIE_weekly_vs_daily.md`) : à faire tourner jusqu'au bout.
  → Ordre conseillé : paramétriques + LSTM d'abord (matrice exploitable à 5/6 rapidement), TSDiff
     complété par batch ensuite.
- **LSTM : re-lancer le sweep SEQ_LEN** (`experiments/lstm_weekly_sweep.py`) **avant** la
  régénération, car le bloc de validation (12 origines) se déplace avec `n_test` : le
  `SEQ_LEN*` par actif doit être resélectionné sur le **nouveau** bloc de validation (règle 1-SE,
  régime C, jamais le test — cf. `BRIEF_lstm_weekly_retune.md`). Ne pas réutiliser
  `lstm_weekly_sweep.json` tel quel.

### 3.2 Garde-fou anti-fuite (NON négociable)

- Le walk-forward de `weekly_multimodel.py` ré-entraîne **à chaque origine** sur
  `weekly.iloc[:m+1]` (régime C) / `daily.iloc[:daily_pos+1]` (régime B) : **chaque origine
  n'utilise que son propre passé**. Augmenter `n_test` ne fait qu'**ajouter des origines plus
  anciennes** — aucune fuite introduite.
- **Interdit** : tout raccourci « train-once-forward » (entraîner une fois puis réutiliser le
  modèle sur des origines postérieures) pour TSDiff/LSTM. Le coût du retraining par origine est
  le prix de l'honnêteté ; ne pas le contourner.
- La sélection LSTM (SEQ_LEN) se fait sur le **bloc de validation**, jamais sur le test.

### 3.3 Injecter en base puis régénérer les analyses

1. **Backfill DB** : `experiments/backfill_multimodel_predictions.py` (upsert des nouvelles
   lignes régime B/C, `source='oos'`, `horizon_type='weekly'`), puis
   `experiments/backfill_eval_metrics.py` pour renseigner `y_true`, `in_interval`, `abs_error`,
   etc. Vérifier qu'on n'introduit pas de doublons (garde-fous doublons déjà en place, cf.
   `BRIEF_prevention_doublons.md`).
2. **Re-tests appariés** : relancer `experiments/matrice_paired_tests.py` (seed fixe) →
   `comparison_3` doit maintenant reposer sur ~90 origines.
3. **Régénérer le dashboard** :
   `python -m experiments.dashboard_d7_w1 --db-path validation/tracking.db --out experiments/dashboard_d7_w1.html --seed 42`.
   Ne rien changer d'autre au script (appariement, Winkler, skill-score, tokens CSS, page
   autonome `file://`, pas de synchro JS — cf. `BRIEF_dashboard_D7_vs_W1.md`).

---

## 4. Ce que le brief ne fait PAS

- Ne change **pas** l'ancrage hebdo (reste vendredi/W-FRI), ne démarre pas à une date calendaire.
- Ne vise **pas** la significativité cellule-par-cellule (hors de portée, ~216 origines).
- Ne relance **pas** le régime A « D+7 calendaire » (abandonné pour confusion d'horizon) — sauf
  demande explicite ultérieure.
- Ne retouche **pas** `Run/dashboard.html`, ni le schéma DB, ni la logique d'appariement du
  dashboard.
- Ne modifie **pas** le régime B des modèles paramétriques au-delà de la couverture des nouvelles
  origines (asymétrie B/C déjà documentée, `METHODOLOGIE_weekly_vs_daily.md`).

---

## 5. Vérification (à faire à la fin, point d'étape court)

1. **Comptage** : régimes B et C ont bien ~90 origines (ou 70 en fallback), première origine
   ~oct. 2024, toutes des **vendredis** ; aucune origine post-fin-de-série (WEEK_MARGIN respecté).
2. **effective_n** ≈ 30 par cellule et au poolé (au lieu de ~10) ; affiché partout.
3. **Verdicts recoupés** : `comparison_3` de `matrice_paired_tests.json` = table du dashboard
   (mêmes diffs, même seed). Noter les bascules attendues : **crypto sq-error doit devenir
   significative** ; les vrais nuls (index/bond Winkler) doivent **rester** indistinguables — si
   l'un d'eux bascule, suspecter une fuite ou un bug, pas un « vrai » gain.
4. **Pas de fuite** : pour 2-3 origines anciennes tirées au hasard, vérifier que `T0`/train ne
   dépasse pas la date d'origine (contrôle sur les logs `weekly_multimodel.py`).
5. **Reproductibilité** : deux runs à seed=42 → mêmes p-values.
6. **Firefox `file://`** : la page ouvre sans serveur, sans gel (rejouer un clic de zoom), tout
   inline.

---

## 6bis. Mini-rapport développé à côté de chaque verdict (demande tuteur)

**Indépendant du retraining** — se fait sur la base actuelle, ne nécessite ni nouvelle origine ni
GPU. Objectif : partout où un **verdict** est affiché (table par cellule ET tuiles d'agrégat),
ajouter un **petit rapport plus développé** qui explique **sur quelle base** le verdict est rendu
et **ce que dit chaque KPI**.

### Où
- `experiments/dashboard_d7_w1.py` (générateur) + `experiments/dashboard_d7_w1_template.py` (rendu).
- Le rapport est un **champ calculé** du payload JSON, pas du texte en dur : fonctions **pures** des
  métriques déjà présentes dans chaque dict de cellule / d'agrégat (aucun recalcul du pipeline
  lourd, aucun accès prix/DB supplémentaire). Ainsi il se régénère automatiquement après le
  backfill/retraining, avec les nouveaux chiffres.

### Contenu par cellule (model × asset)
1. **En-tête = le verdict + sa base statistique.** Rappeler que le badge est décidé par le **test
   RMSE** (erreur quadratique), bootstrap par blocs : `p`, `IC95` de la différence moyenne (0
   exclu ⇒ significatif / 0 inclus ⇒ indistinguable), `n` origines, `effective_n` (puissance
   réelle). Expliquer en une phrase ce que « significatif » veut dire ici (l'IC95 bootstrap de la
   différence d'erreur par origine, rééchantillonnée par blocs pour l'autocorrélation, exclut 0).
2. **Lecture KPI par KPI** (petit tableau) : pour chacun des 5 KPI — **RMSE** (précision du point,
   celui qui décide), **Winkler/Interval Score** (note probabiliste), **Cov95** (calibration, vs
   cible 0,95), **largeur PI** (finesse, *à couverture comparable seulement*), **direction**
   (diagnostic, *balisé très bruité à n≈30*) — afficher valeur daily, valeur weekly, **de quel côté
   ça penche**, et une note (sens de lecture + écart relatif).
3. **Concordance / arbitrage** : combien de KPI penchent daily vs weekly, et surtout **signaler
   quand le point (RMSE) et l'incertitude (Winkler/Cov95) désignent des gagnants différents** —
   c'est le cas le plus important à expliciter (« un modèle peut être moins précis mais avoir une
   incertitude plus fiable », cf. TSDiff dans `METHODOLOGIE_weekly_vs_daily.md`).

### Contenu par tuile d'agrégat (global / crypto / actions / obligations)
- Distinguer **deux axes** : précision ponctuelle (**skill RMSE**) et fiabilité de l'incertitude
  (**skill Winkler**), chacun avec verdict + `p` + `IC95` + base (`n_origins`, `effective_n`,
  définition du skill sans échelle). **Synthèse** en clair : concordant, ou **arbitrage** si les
  deux axes ne désignent pas le même gagnant.

### Contraintes de rendu (identiques au reste du dashboard)
- **Autonome `file://`** (tout inline, aucun CDN/fetch), **aucune synchro JS multi-graphiques**
  (leçon `CORRECTIF_dashboard_v4_boucle_infinie.md`). Réutiliser les tokens CSS existants.
- UX : verdict **cliquable** → dépliage d'un panneau détaillé sous la ligne (table par cellule) ;
  pour les tuiles d'agrégat, le rapport peut être affiché directement (peu de tuiles).
- **Aucune affirmation non étayée** : le texte doit être **généré depuis les chiffres**, jamais
  écrit à la main ; un verdict indistinguable dit explicitement « pas de gagnant » même si des KPI
  penchent.

> Note : ce mini-rapport est **indépendant** de la fenêtre (§2bis) — faisable tout de suite sur la
> base actuelle, et se re-remplira automatiquement avec les nouveaux chiffres après le backfill.

---

## 2bis. Fenêtre — décision : remonter AVANT décembre 2025

Le « 8 janvier 2026 » demandé au départ est **abandonné** : la validation weekly commence **déjà le
5 décembre 2025** dans `validation/tracking.db`, donc démarrer au 8 janvier *retirerait* ~1 mois de
données — l'inverse du but. L'objectif réel est **plus de rigueur = plus d'origines**, et
décembre 2025 ne suffit pas (effective_n ≈ 10, un seul cas poolé borderline).

**Décision : backfill vers le passé, avant décembre 2025**, jusqu'à atteindre la cible d'origines
du §1. Ancrage vendredi (W-FRI) conservé ; pas de date de début imposée — on choisit un **nombre
d'origines** (`n_test`), la fenêtre remonte automatiquement.

- **Cible : `n_test = 90`** (1ʳᵉ origine ~oct. 2024). Franchit le cas borderline (crypto sq-error,
  ~70 requis) **avec marge** et porte l'effective_n de ~10 à ~30.
- **Plancher : 70** si le temps de calcul (TSDiff) serre.
- **Ne pas sur-remonter au-delà de ~90-120.** Rendements décroissants : la significativité
  **cellule-par-cellule** exige ~216 origines (≈ 4 ans, cf. §1) pour un gain marginal — ce n'est
  **pas** l'objectif. Au-delà de ~120 on paie surtout du temps TSDiff sans verdict nouveau. Si tu
  veux de la marge, 120 est un maximum raisonnable ; 90 est la cible.

---

## 6. Résumé exécutif (à retenir)

- **Fenêtre tranchée (§2bis)** : le « 8 janvier » est abandonné (la validation commence déjà au
  5 déc. 2025) ; on **remonte AVANT décembre**. Cible **`--n-test` 30 → 90** (plancher 70, max
  raisonnable 120), backfill W-FRI vers ~oct. 2024, ancrage vendredi conservé. Ne pas viser le
  cellule-par-cellule (~216 origines, rendements décroissants).
- **Les 6 modèles** (matrice complète) : paramétriques + LSTM d'abord, **TSDiff en batch
  checkpointé** (reprenable, donc pas bloquant malgré sa durée).
- **Indépendant de la fenêtre — faisable tout de suite : §6bis** (mini-rapport développé à côté de
  chaque verdict, base statistique + KPI par KPI). Ne nécessite ni retraining ni GPU.
- **Honnêteté des gains** : élargir solidifie le poolé/par-classe (crypto sq-error → significatif,
  effective_n ~30) mais **ne tranche pas** cellule-par-cellule (~216 origines) et ne ressuscite pas
  les vrais nuls (index/bond Winkler).
