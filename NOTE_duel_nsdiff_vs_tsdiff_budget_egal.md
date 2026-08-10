# NOTE — NsDiff vs TSDiff à budget d'échantillonnage égal, régimes weekly ET daily

*2026-08-06. Complète `NOTE_nsdiff_consolidation_daily_vs_weekly.md` (chantier
intra-NsDiff) et `NOTE_compare_weekly_tsdiff_nsdiff.md` (TSDiff-W vs NsDiff-W à
m=500, 30 origines, régime weekly seulement). Artefacts :
`experiments/diffusion_multiseed_v2.py` → `diffusion_multiseed_v2/`,
`experiments/tsdiff_daily_epoch_sweep.py` → `tsdiff_daily_epochs.json`,
`experiments/nsdiff_vs_tsdiff_v2.py` → `nsdiff_vs_tsdiff_v2.json`,
machinerie de test partagée dans `experiments/diffusion_headtohead.py`.*

## 0. Ce que cette note ajoute

`NOTE_compare_weekly_tsdiff_nsdiff.md` comparait déjà les deux diffusions, à
budget d'échantillonnage égal (m=500 des deux côtés) et sur 5 graines — ce
n'était donc pas un match biaisé. Mais elle est limitée au **régime
weekly-natif** et aux **30 origines du duel**. Deux manques :

1. le **régime daily (B)** n'avait jamais été comparé entre les deux
   diffusions — c'est précisément l'axe du chantier de consolidation ;
2. la **grille des 90 origines `oos`**, sur laquelle tout le reste de la
   consolidation est mesuré, n'était pas couverte.

Cette note comble les deux. Les lignes `oos` de TSDiff en base n'étaient pas
réutilisables : elles sont à `n_samples=50`, où un intervalle étiqueté 95 %
n'en couvre réellement que ~91,3 % (biais du quantile empirique, cf.
`NOTE_nsdiff_consolidation_daily_vs_weekly.md` §1) — soit ~2,8 points offerts
à NsDiff. Les deux bras ont donc été régénérés à 200 tirages, **dans le même
run, sur une seule série de prix gelée, par la même boucle de génération**
(`generate_nsdiff_asset` avec son kwarg `engine`).

## 0.1 Ce que `p` désigne ici

Toutes les p-values de ce chantier viennent du **bootstrap par blocs par
percentile** (`paired_test.paired_block_bootstrap_test`), jamais d'un test
paramétrique. Concrètement : on rééchantillonne 10 000 fois la série
chronologique des différences par origine, en blocs de 3 origines consécutives
(pour respecter le chevauchement des cibles W+2/W+3), et

    p = 2 × (fraction des 10 000 moyennes rééchantillonnées tombant du côté
             de zéro OPPOSÉ à la moyenne observée),  plafonnée à 1.

C'est donc une p-value **bilatérale**, qui répond à « si le vrai écart moyen
était nul, quelle fraction des rééchantillonnages produirait un écart au moins
aussi extrême que celui observé ? ». Elle ne suppose ni normalité, ni
indépendance des origines.

**Résolution** : avec 10 000 réplicats, le plus petit p non nul vaut
2/10 000 = **2×10⁻⁴**. Un `p = 0` affiché signifie donc *zéro réplicat sur
10 000 du mauvais côté* — soit `p < 2×10⁻⁴`, et rien de plus fin ne peut être
affirmé sans augmenter `n_boot`.

**Seuil de décision.** La convention du repo est α = 0,05
(`significant_at_05`), mais elle ne suffit pas ici pour deux raisons :

- **tests multiples** — 30 cellules et 12 tests poolés sans correction. À 5 %,
  on attend ~1,5 faux positif par famille de 30. Une correction de Holm sur les
  12 tests poolés donnerait un seuil effectif de 0,05/12 ≈ **0,004** ;
- **puissance faible** — `effective_n` = 30. Un `p` juste au-dessus de 0,05
  ne démontre rien du tout.

D'où la lecture pratique appliquée dans cette note : un `p` de l'ordre de
**2×10⁻⁴** survit à n'importe quelle correction raisonnable et se lit comme un
résultat ; un `p` entre 0,01 et 0,05 (ex. crypto/Winkler à W+3 dans la note de
consolidation, p = 0,0378) ne survit pas à Holm et se lit comme un signal à
confirmer ; un `p` au-dessus de 0,05 ne se lit **pas** comme une absence
d'effet — c'est ce que le TOST, lui, permet d'affirmer.

## 1. Verdict

### 1.1 Régime weekly-natif — NsDiff domine sur les deux axes

| Horizon | Actif | RMSE NsDiff / TSDiff | verdict RMSE | graines sig. | Cov95 NsDiff / TSDiff | verdict Winkler |
|---|---|---|---|---|---|---|
| W+1 | BTC-USD | 5255 / 5789 | **NsDiff** | 4/5 | 0.967 / 0.911 | indistinguable |
| W+1 | ETH-USD | 274.0 / 349.4 | **NsDiff** | 4/5 | 0.947 / 0.927 | **NsDiff** |
| W+1 | SPY | 12.95 / 13.19 | indistinguable | 0/5 | 0.944 / 0.802 | **NsDiff** |
| W+1 | TLT | 1.402 / 1.790 | **NsDiff** | 5/5 | 0.964 / 0.956 | **NsDiff** |
| W+1 | ZN=F | 0.7725 / 0.7712 | indistinguable | 0/5 | 0.933 / 0.827 | **NsDiff** |
| W+2 | BTC-USD | 8118 / 9567 | **NsDiff** | 4/5 | 0.949 / 0.898 | **NsDiff** |
| W+2 | ETH-USD | 413.6 / 632.8 | **NsDiff** | 4/5 | 0.962 / 0.884 | **NsDiff** |
| W+2 | SPY | 17.07 / 17.61 | indistinguable | 0/5 | 0.951 / 0.811 | **NsDiff** |
| W+2 | TLT | 1.832 / 2.820 | **NsDiff** | 4/5 | 0.976 / 0.924 | **NsDiff** |
| W+2 | ZN=F | 0.9722 / 1.017 | indistinguable | 1/5 | 0.962 / 0.793 | **NsDiff** |
| W+3 | BTC-USD | 10 610 / 12 760 | **NsDiff** | 4/5 | 0.942 / 0.891 | **NsDiff** |
| W+3 | ETH-USD | 525.8 / 849.4 | **NsDiff** | 4/5 | 0.936 / 0.880 | **NsDiff** |
| W+3 | SPY | 20.82 / 21.76 | indistinguable | 1/5 | 0.920 / 0.818 | **NsDiff** |
| W+3 | TLT | 2.073 / 3.538 | **NsDiff** | 5/5 | 0.987 / 0.947 | **NsDiff** |
| W+3 | ZN=F | 1.143 / 1.237 | **NsDiff** | 0/5 | 0.964 / 0.838 | **NsDiff** |

**Poolé tous actifs : NsDiff significativement meilleur sur les 6 tests**
(3 horizons × RMSE/Winkler), **p = 0 partout** (0 réplicat bootstrap sur 10 000 du mauvais côté). TSDiff ne gagne aucune
cellule sur aucun axe.

### 1.2 Régime daily — égalité sur le point, écart massif sur l'incertitude

| Horizon | Poolé RMSE | Poolé Winkler | Cov95 NsDiff | Cov95 TSDiff |
|---|---|---|---|---|
| W+1 | indistinguable (p=0.152) | **NsDiff** (p = 0) | 0.882 – 0.933 | 0.598 – 0.738 |
| W+2 | indistinguable (p=0.813) | **NsDiff** (p = 0) | 0.889 – 0.951 | 0.611 – 0.747 |
| W+3 | indistinguable (p=0.280) | **NsDiff** (p = 0) | 0.816 – 0.976 | 0.567 – 0.724 |

Sur le **point**, les deux diffusions sont **interchangeables en daily** :
aucun test poolé ne les sépare, et les verdicts par cellule sont
majoritairement indistinguables (TSDiff prend BTC-USD à W+1/W+2 sur 1 graine
sur 5 ; NsDiff prend SPY/TLT/ZN=F à W+3). Sur l'**incertitude**, l'écart est
sans exception : **20 à 30 points de couverture** d'avance pour NsDiff, et le
Winkler en sa faveur sur les **15 cellules sur 15**.

### 1.3 Synthèse sur les 30 cellules (2 régimes × 3 horizons × 5 actifs)

| Axe | NsDiff gagne | TSDiff gagne | indistinguable |
|---|---|---|---|
| **Winkler** | **29** | **0** | 1 |
| RMSE | 12 | 2 | 16 |

Le motif est net et il recoupe le finding de fond du programme : **le gain de
NsDiff se loge dans la fiabilité de l'incertitude, pas dans le point**. En
weekly il gagne les deux ; en daily, seulement la calibration.

## 2. Deux erreurs de protocole rencontrées en route

### 2.1 Le premier bras daily était invalide (corrigé)

La première génération réutilisait, côté daily, le budget d'époques
hebdomadaire de chaque modèle — convention héritée de
`oos_nsdiff_daily_weekly.py`. Elle est sûre pour NsDiff (40 plat) et **fausse
pour TSDiff**. La série quotidienne compte ~2465 observations contre ~511 en
hebdomadaire : à époques égales, ~5× plus de pas de gradient. Mesuré sur SPY,
largeur médiane du PI 95 % en % du prix :

| epochs | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| largeur PI | 12.39 % | 3.41 % | 0.46 % | **0.12 %** |

À 80 époques (ce qu'utilisait le premier run pour 4 actifs sur 5), la
couverture observée tombait à **1–6 %** : plus un modèle sous-calibré, un
modèle effondré, contre lequel toute victoire au Winkler est mécanique. Le
repo connaissait la fragilité à demi-mot — `weekly_headtohead_v2.run_pair_v2`
désactive TSDiff-D par défaut en le qualifiant de *structurally
under-calibrated*, et `weekly_headtohead_results.json` (300 époques)
enregistre `coverage_95: 0.0`.

Correction : `generate_nsdiff_asset` accepte désormais un `epochs_daily`
distinct (opt-in, défaut = `epochs`, comportement inchangé), et
`tsdiff_daily_epoch_sweep.py` sélectionne le budget de TSDiff-D par validation
— argmin CRPS, critère maison réutilisé tel quel (`epoch_sweep.select_epochs`),
candidats 10/20/40, entraînement incrémental (`epoch_sweep.fit_checkpoints`).
Retenu : **20 époques dans 19 cellules sur 25** (40 dans 5, 10 dans 1), ce qui
reproduit l'ordre de grandeur des lignes `oos` en base (3.41 % contre 3.65 %).

**Vérification du correctif** : sur `seed42|SPY`, le bras weekly se reproduit
**bit-à-bit** (largeur 5.930 % → 5.930 %, échantillons identiques) tandis que
le bras daily passe de 0.177 % à 4.573 %. Les tirages hebdomadaires précèdent
le fit daily dans la boucle et n'en dépendent pas — vérifié, pas supposé. Les
chiffres weekly du §1.1 sont donc les mêmes avant et après correction.

### 2.2 Une fuite dans la sélection d'époques hebdomadaire de TSDiff (déclarée, non corrigée)

Les époques hebdomadaires de TSDiff (40/60/80) sont relues verbatim de
`compare_weekly_diffusion.json`, où elles avaient été sélectionnées sur
validation **dans le protocole du duel**. Or ce découpage place la validation
(SPY : 2025-09-19 → 2025-12-05, 12 origines) **à l'intérieur** de la grille de
test `oos` (2024-10-18 → 2026-07-02). ~13 % des origines évaluées ont donc
servi à choisir l'hyperparamètre de TSDiff.

Non corrigé (il faudrait re-sweeper le bras hebdomadaire), mais **la fuite
joue en faveur de TSDiff** — qui perd quand même les 6 tests poolés weekly.
Le verdict du §1.1 en est d'autant plus conservateur. Le sweep daily du §2.1,
lui, est ancré **strictement avant** la première origine de test : aucune des
90 origines n'entre dans la sélection.

## 3. Non-négociables — statut

- **Budget d'échantillonnage strictement égal** : 200 tirages des deux côtés,
  lus par la même formule (quantiles empiriques 2.5/97.5) dans la même boucle.
- **Prix identiques** : téléchargés une fois, gelés dans
  `diffusion_multiseed_v2/prices/*.parquet`, partagés par les deux bras.
  Nécessaire : yfinance resert les mêmes cotes avec ~2e-7 de bruit relatif d'un
  appel à l'autre, qu'un fit de 40-80 époques amplifie à ~6e-6 sur les
  échantillons.
- **Même code de génération** : `generate_nsdiff_asset` paramétré par un
  `DiffusionEngine`. Le chemin par défaut (NsDiff) a été vérifié **bit-à-bit**
  équivalent à l'appel d'origine sur les trois points touchés (fit,
  standardisation, forecast).
- **Même code de test** : `diffusion_headtohead.py`, partagé avec le match vs
  ARIMA-GARCH. Le refactor qui l'a extrait a été vérifié à **0 écart
  numérique** sur ce match.
- **Multi-graines** : 42-46, chaque test rejoué par graine et sur les graines
  poolées (métrique moyennée par origine, l'unité d'inférence reste
  l'origine — `n`=90, `effective_n`=30).
- **`tracking.db` jamais écrit** : artefact isolé. La piste `oos`/dashboard
  reste single-seed 42 et `n_samples=50`.
- **pytest** : **365 passed, 1 skipped** (+14 tests neufs sur les conventions
  de signe du duel — la couverture se maximise quand tout le reste se
  minimise, c'est là qu'un head-to-head s'inverse en silence).

## 4. Limites déclarées

- **Budgets d'époques non égalisés, et c'est volontaire** : TSDiff a 40/60/80
  par (actif, graine) côté weekly, sélectionnés sur validation ; NsDiff a 40
  plat. Les égaliser reviendrait à jeter la sélection de TSDiff. L'asymétrie
  va dans le sens de TSDiff (§2.2).
- **TSDiff-D reste fragile** : sa calibration varie d'un facteur 100 entre 10
  et 80 époques. Le budget retenu est l'argmin CRPS, **pas** un optimum de
  calibration — même à son meilleur CRPS, il plafonne à 0.60–0.76 de
  couverture en validation. La supériorité de calibration de NsDiff en daily
  se lit donc face à un adversaire structurellement mal calibré, pas face à
  son meilleur réglage possible en couverture.
- **Puissance** : `effective_n`=30. Aucun « indistinguable » de cette note
  n'est une absence d'effet démontrée — et aucun TOST n'a été fait ici
  (contrairement au chantier intra-NsDiff), donc l'« interchangeabilité » du
  point en daily (§1.2) est une non-réfutation, pas une équivalence établie.
- **Tests multiples non corrigés** sur les 30 cellules.
- **Protocole d'entraînement** : *train-once-forward* pour les deux bras, donc
  symétrique ici — contrairement au match vs ARIMA-GARCH.
