# NOTE — NsDiff : régime daily (B) vs régime mensuel-natif (C), cibles M+1/M+2/M+3

*2026-08-05. Suite de `BRIEF_nsdiff_mensuel_M1M2M3.md`, en miroir de
`NOTE_compare_daily_vs_weekly_nsdiff.md` (même question, un cran plus haut sur
l'échelle des horizons : daily→semaine devient daily→mois, et
weekly-natif devient mensuel-natif). Source : `experiments/oos_nsdiff_monthly.py`
(génération + insertion oos, seed=42) + `experiments/nsdiff_monthly_multiseed.py`
(graines 42-46, artefact isolé `nsdiff_monthly_multiseed.json`, jamais dans
`tracking.db`) + `experiments/analyze_nsdiff_monthly.py` (extraction,
`nsdiff_daily_vs_monthly_extract.json`). Contrairement au weekly, **pas de
dashboard** : `dashboard_d7_w1.py` reste strictement W+1/weekly (brief §5) ;
cette note est l'unique livrable côté mensuel.*

---

## 0. AVERTISSEMENT PUISSANCE — à lire avant toute ligne de résultat

L'historique disponible (~2015 → aujourd'hui, BTC/SPY/ZN=F/TLT ; ~2017-11 →
aujourd'hui pour ETH-USD) ne fait que **~105 à 139 fins de mois**, contre
~570 semaines côté weekly. Après réservation d'un bloc train (~55 à 90 mois)
et d'un petit bloc validation réservé (6 mois, non consommé — pas de sweep
d'epochs mensuel ici, budget déclaré §3), il ne reste que **40 origines de
test** par actif — et les cibles M+1/M+2/M+3 se chevauchent (bootstrap par
blocs, `block_length=3`) → **`effective_n=13`** par cellule (17 sur
l'agrégat global poolé), **pas 30** comme au weekly (`n_test=90`,
`effective_n=30`). C'est la réalité annoncée par le brief : à cette échelle,
une bonne partie des cellules — en particulier Actions et Obligations,
cf. §2 — ressortent **indistinguables**, non par absence d'effet réel mais
par manque de puissance statistique. **Un verdict mensuel « indistinguable »
n'est PAS une équivalence démontrée** : c'est l'absence de preuve du
contraire à `effective_n≈13`, point.

À l'inverse, **quand un effet mensuel ressort significatif malgré ce
`effective_n` réduit** (crypto, §2 et §5) — et surtout quand il ne tient
QUE sur 4 graines sur 5 (§5bis, BTC-USD/ETH-USD) — la lecture honnête est
« un effet assez fort pour percer malgré la faible puissance, mais dont
la significativité elle-même n'est pas garantie robuste graine à graine ».
Aucun verdict de cette note n'est présenté sans son statut multi-graines
(non-négociable brief §3).

---

## 1. Cadrage — ce qui a été généré, et comment

**Régime B (daily → fin de mois) vs régime C (mensuel-natif), à cible
identique** (même `target_date` = dernier jour de bourse du mois-cible, même
`cutoff_date` = dernier jour de bourse du mois d'origine — vérifié 100%
identiques par construction, `assert` dans `build_enriched_pairs_monthly`).

- **Régime B (daily)** : NsDiff fit sur rendements **quotidiens**, prévision
  lue à la **distance réelle en jours de bourse** jusqu'au dernier jour de
  bourse du mois-cible (`month_targets`, miroir exact de
  `epoch_sweep.week_targets`). `frequence='daily'`, `horizon_type='monthly'`.
- **Régime C (mensuel natif)** : NsDiff fit **directement sur les rendements
  mensuels** (`nsdiff_model.fit_nsdiff`/`forecast_from_fitted` — vérifiés
  agnostiques à la fréquence : ils ne voient qu'une `pd.Series` + un horizon
  entier, rien de daily/weekly en dur), horizon=3, M+1/M+2/M+3 en un seul
  tir. `frequence='monthly'`, `horizon_type='monthly'`.

**Origines — générées, pas réutilisées** (contrairement au weekly, aucun des
6 autres modèles n'a de ligne `oos` mensuelle à réutiliser verbatim, brief
§2) : `three_way_split_monthly` (miroir exact de `epoch_sweep.
three_way_split`) sur la série mensuelle de chaque actif — train (le reste,
55 à 90 mois selon l'actif), validation (6 mois, réservés, non consommés),
test (**40 origines**, 2023-01-31 → 2026-04-30, identiques pour les 5
actifs). Point-in-time strict : `mu`/`sd` gelés au train (`fit_nsdiff` une
seule fois par actif, train-once-forward, jamais de refit dans la boucle),
fenêtre de conditionnement `[:m]`/`[:daily_pos]` qui grandit sans jamais
dépasser l'origine, resample fin-de-mois `dropna` avec **abandon explicite
du mois en cours** (`build_monthly`, jamais un mois partiel).

**Budgets déclarés** (brief §6) :

| Paramètre | Valeur | Justification |
|---|---|---|
| `epochs` | 40 | `weekly_nsdiff_production.NSDIFF_EPOCHS_W`, même constante que le daily/weekly (aucun budget mensuel séparé inventé) |
| `k_denoise` | 20 | `nsdiff_model.K_DENOISE`, inchangé |
| `n_samples` | 50 | identique au daily/weekly |
| `seq_len` régime B (daily) | 30 | défaut `nsdiff_model.SEQ_LEN`, non modifié (même convention que le daily/weekly) |
| `seq_len` régime C (mensuel) | **18** (~1.5 an) | **nouveau, déclaré** — reprendre 30 (comme au weekly) consommerait 30 des ~130 points mensuels totaux avant la moindre origine de test ; le weekly peut se le permettre (30 sur ~570 semaines), le mensuel non. 18 laisse `sigma_kernel=8` avec marge et préserve l'essentiel de l'historique pour train+test. |
| `model_d.horizon` (régime B) | calculé, PAS deviné (63-65 actions/obligations, 92 crypto) | distance réelle en jours de bourse jusqu'à M+3, calculée sur les vraies origines de chaque actif (le crypto trade 7j/7, les actions/obligations 5j/7 → un horizon unique aurait tronqué le crypto ou gaspillé l'entraînement actions/obligations) |
| Ancre cible | dernier jour de bourse du mois (`build_monthly`, `resample("ME").last().dropna()`, mois en cours toujours exclu) | jamais un mois partiel, brief §0/§9 |

**Isolation `tracking.db`** : `frequence='monthly'`/`daily`, `horizon_type=
'monthly'` — nouvelles combinaisons, table `predictions` non modifiée.
Comptage avant/après (source='oos') :

| | avant | après | delta |
|---|---|---|---|
| `daily`/`daily` | 6224 | 6224 | 0 |
| `daily`/`weekly` | 9450 | 9450 | 0 |
| `weekly`/`weekly` | 9450 | 9450 | 0 |
| `daily`/`monthly` | 0 | 600 | **+600** |
| `monthly`/`monthly` | 0 | 600 | **+600** |

Lignes daily/weekly des 7 modèles **intactes**, confirmé. Insertion réexécutée
une seconde fois (SPY seul) : compte total inchangé (upsert idempotent,
`ON CONFLICT` sur `(source, model, asset, horizon, frequence, horizon_type,
cutoff_date)`). `horizon_unit` toujours passé explicitement ('M+1'/'M+2'/'M+3')
dans chaque ligne insérée — jamais laissé au calcul de repli de
`insert_oos_predictions` (qui n'a pas de cas 'M', aurait mal étiqueté
'D+1'/'D+2'/'D+3', cf. docstring `oos_nsdiff_monthly.py`). Multi-graines
strictement dans un artefact JSON isolé (`nsdiff_monthly_multiseed.json`) —
`nsdiff_monthly_multiseed.py` n'appelle jamais `insert_oos_predictions`.

---

## 2. Tableau par actif — M+1 / M+2 / M+3

Seul le **RMSE** est testé statistiquement par cellule (bootstrap par blocs,
seed interne 0, `block_length=3`, n=40 origines/actif → `effective_n=13`) ;
Cov95/largeur PI/Winkler/direction sont des lectures **descriptives**, non
testées à ce niveau (même convention que le weekly).

| Horizon | Actif | Classe | RMSE (daily → mensuel) | Cov95 (cible 95%) | Largeur PI (daily → mensuel) | Winkler (daily → mensuel) | Direction (daily → mensuel) | Verdict (RMSE) | p |
|---|---|---|---|---|---|---|---|---|---|
| M+1 | BTC-USD | Crypto | 9675 → 12 160 | 95% → 100% | 38 610 → 117 800 | 42 210 → 117 800 | 60% → 53% | **daily significativement meilleur** | 0.0004 |
| M+1 | ETH-USD | Crypto | 513.4 → 755.2 | 95% → 100% | 2252 → 8325 | 2472 → 8325 | 53% → 50% | **daily significativement meilleur** | 0.0012 |
| M+1 | SPY | Actions | 20.55 → 20.37 | 95% → 100% | 81.77 → 187.9 | 87.53 → 187.9 | 47% → 60% | indistinguable | 0.8854 |
| M+1 | TLT | Obligations | 3.323 → 3.315 | 90% → 100% | 12.07 → 24.25 | 15.43 → 24.25 | 47% → 57% | indistinguable | 0.6788 |
| M+1 | ZN=F | Obligations | 2.005 → 1.962 | 90% → 100% | 6.198 → 11.92 | 9.331 → 11.92 | 50% → 45% | indistinguable | 0.4930 |
| M+2 | BTC-USD | Crypto | 17 110 → 22 660 | 95% → 100% | 63 770 → 193 800 | 67 770 → 193 800 | 57% → 55% | **daily significativement meilleur** | <0.0001 |
| M+2 | ETH-USD | Crypto | 829.6 → 1574 | 100% → 100% | 3627 → 14 610 | 3627 → 14 610 | 53% → 50% | **daily significativement meilleur** | <0.0001 |
| M+2 | SPY | Actions | 29.77 → 34.07 | 97% → 100% | 114.9 → 272.8 | 128.2 → 272.8 | 62% → 50% | indistinguable | 0.1182 |
| M+2 | TLT | Obligations | 4.478 → 4.689 | 95% → 97% | 16.62 → 34.68 | 24.66 → 36.54 | 60% → 57% | indistinguable | 0.4172 |
| M+2 | ZN=F | Obligations | 2.598 → 2.599 | 93% → 97% | 8.567 → 17.08 | 13.54 → 17.79 | 53% → 55% | indistinguable | 1.0000 |
| M+3 | BTC-USD | Crypto | 25 580 → 32 160 | 93% → 100% | 85 170 → 269 500 | 87 890 → 269 500 | 57% → 57% | **daily significativement meilleur** | 0.0104 |
| M+3 | ETH-USD | Crypto | 1151 → 2512 | 95% → 100% | 4596 → 22 320 | 5095 → 22 320 | 53% → 53% | **daily significativement meilleur** | <0.0001 |
| M+3 | SPY | Actions | 34.25 → 40.03 | 97% → 100% | 150.2 → 335.7 | 152.4 → 335.7 | 68% → 62% | indistinguable | 0.1448 |
| M+3 | TLT | Obligations | 5.187 → 5.473 | 93% → 100% | 21.04 → 42.16 | 28.68 → 42.16 | 47% → 47% | indistinguable | 0.6180 |
| M+3 | ZN=F | Obligations | 3.319 → 3.192 | 88% → 100% | 10.81 → 20.79 | 15.77 → 20.79 | 53% → 55% | indistinguable | 0.4972 |

**Lecture immédiate** : motif net et **cohérent sur les 3 horizons** — sur les
2 actifs crypto, le régime daily bat significativement le régime mensuel-natif
sur le RMSE, à chaque fois (6/6 cellules crypto). Sur les 3 actifs
actions/obligations, **toutes les cellules sont indistinguables** (9/9), exactement
la lecture attendue par l'avertissement §0. **Mais** ce motif crypto n'est
**pas garanti stable graine à graine** — voir §5bis avant de le lire comme
acquis.

Régularité frappante indépendante du RMSE : **Cov95 du régime mensuel-natif
est quasi toujours 100%** (contre 88-97% côté daily, plus proche de la cible
95%), avec une **largeur de PI 2 à 4× plus grande**. Ce n'est pas une
meilleure fiabilité — c'est une intervalle systématiquement trop large,
symptomatique d'un réseau de variance (`g_psi`) mal calibré faute de
données d'entraînement mensuelles (34 à 70 fenêtres d'entraînement selon
l'actif, contre des milliers côté daily) — hypothèse plausible et cohérente
avec §0, non démontrée mécaniquement ici.

---

## 3. Agrégat par classe + global — skill-score RW (M+1, NsDiff seul)

Baseline RW réutilisée telle quelle (`dashboard_d7_w1.historical_h_day_returns`/
`rw_pi_bounds`, paramétrées en jours calendaires — fonctionnent sans
modification pour une cible mensuelle). Scope M+1 uniquement (mirroir du
scope W+1 du dashboard weekly — M+2/M+3 restent au niveau cellule §2, pas
d'agrégat poolé dédié, le brief ne l'exige pas). `skill = 1 - score/
médiane(score RW)` par (actif, horizon_unit) ; obligations dédoublonnées
(ZN=F+TLT → une seule contribution "taux").

| Groupe | n origines (eff.) | Skill RMSE (daily vs mensuel) | p | Skill Winkler (daily vs mensuel) | p |
|---|---|---|---|---|---|
| **Global** | 52 (eff. 17) | **daily significativement meilleur** | 0.0002 | **daily significativement meilleur** | <0.0001 |
| Crypto | 40 (eff. 13) | **daily significativement meilleur** | 0.0006 | **daily significativement meilleur** | <0.0001 |
| Actions | 40 (eff. 13) | indistinguable | 0.8888 | **daily significativement meilleur** | <0.0001 |
| Obligations | 40 (eff. 13) | indistinguable | 0.5746 | **daily significativement meilleur** | 0.0056 |

**Le résultat le plus robuste de toute cette note n'est pas le RMSE crypto —
c'est le skill Winkler** : le régime daily bat significativement le régime
mensuel-natif sur les 4 lignes (global + les 3 classes), y compris Actions et
Obligations où le RMSE est une pure égalité. Cohérent avec le motif Cov95/
largeur-PI du §2 : le mensuel-natif n'est pas moins précis en pointe sur ces
2 classes, mais son incertitude est nettement moins bien calibrée
(sur-couverture, PI trop larges) — un arbitrage point/incertitude net en
faveur du daily, à cette taille d'échantillon.

---

## 4. Lecture par métrique

- **RMSE** (testé) : daily gagne sur crypto (6/6 cellules significatives),
  match nul partout ailleurs (9/9 indistinguables). Aucune cellule où le
  mensuel-natif gagne significativement.
- **Cov95** : régime mensuel-natif quasi toujours à 100% (sur-couvert),
  régime daily plus proche de la cible 95% (88-97%) — lecture descriptive,
  mais très régulière (15/15 cellules).
- **Largeur PI** : mensuel-natif systématiquement 2 à 4× plus large que
  daily — cohérent avec la sur-couverture ci-dessus, pas une largeur "gratuite"
  bien calibrée.
- **Winkler** (skill testé §3, brut descriptif §2) : daily gagne partout,
  y compris là où le RMSE ne tranche pas — c'est le signal le plus cohérent
  de cette note.
- **Direction** : bruitée, 45-68%, proche du hasard — à `n=40`, lecture
  purement diagnostique, aucune conclusion à en tirer (identique à la
  posture du weekly).

---

## 5bis. Robustesse multi-graines (brief §3, non-négociable)

Graines 42-46, mêmes origines, mêmes budgets (`nsdiff_monthly_multiseed.py`,
artefact isolé `nsdiff_monthly_multiseed.json`, **jamais dans `tracking.db`**).
Scope M+1 (comme §3).

| Actif | Verdicts par graine (42→46) | Stable ? | CV Winkler daily / mensuel |
|---|---|---|---|
| BTC-USD | signif., signif., signif., signif., **indistinguable (seed 43)** | **NON** (4/5) | 0.087 / 0.173 |
| ETH-USD | signif., **indistinguable (43)**, **indistinguable (44)**, signif., signif. | **NON** (3/5) | 0.118 / 0.192 |
| SPY | indistinguable ×5 | **OUI** | 0.127 / 0.162 |
| ZN=F | indistinguable ×5 | **OUI** | 0.114 / 0.159 |
| TLT | indistinguable ×5 | **OUI** | 0.116 / 0.146 |

*(« signif. » = daily significativement meilleur sur RMSE M+1.)*

**Lecture honnête** : Actions et Obligations sont **stables** — indistinguable
sur les 5 graines, cohérent avec un effet vraiment nul ou trop petit pour
`effective_n=13`, comme annoncé §0. Crypto est **le motif le plus fort de la
note (§2/§3) mais pas robuste à 100%** : BTC-USD et ETH-USD basculent en
"indistinguable" respectivement pour la graine 43 et pour les graines 43/44 —
sur BTC-USD, 4 graines sur 5 restent significatives (p allant de 0.0004 à
0.044), donc la lecture "daily plutôt meilleur sur crypto en régime mensuel"
tient globalement, mais **n'est démontrée sur AUCUNE graine isolée avec une
garantie absolue** — c'est exactement l'avertissement du brief §3/§9 (un
p<0.05 sur une seule graine à `effective_n≈13` est à prendre avec prudence).
Le CV Winkler (10-19%) est modéré-à-large des deux côtés, plus élevé côté
mensuel-natif partout (cohérent avec un modèle moins bien calibré, §2/§4) —
mais pas assez pour être le facteur dominant du basculement de verdict RMSE
(qui vient du point, pas de l'incertitude).

**Aucun verdict de cette note n'est présenté comme acquis sans ce tableau.**

---

## 6. Non-négociables — statut

- [x] **Isolation `tracking.db`** : `frequence='monthly'`+`horizon_type='monthly'`
      nouvelles, upsert idempotent vérifié (réinsertion SPY : compte inchangé),
      lignes daily/daily/9450+9450 des 7 modèles existants **intactes**
      (comptage avant/après, §1). Multi-graines **jamais** en DB, artefact
      JSON isolé uniquement.
- [x] **Point-in-time** partout : mu/sd gelés au train-once, fenêtre `[:m]`
      strictement croissante sans dépasser l'origine, `dropna` + mois en
      cours toujours exclu (`build_monthly`).
- [x] **Budgets déclarés** : `seq_len` mensuel=18 (justifié §1), `epochs=40`,
      `n_samples=50`, `k_denoise=20`, ancre fin-de-mois — tous écrits §1.
- [x] **Puissance annoncée** : `n`/`effective_n` affichés à chaque tableau
      (§0, §2, §3, §5bis) ; caveat "indistinguable ≠ équivalence" explicite §0.
- [x] Aucun fichier source existant modifié — 4 fichiers **nouveaux**
      (`oos_nsdiff_monthly.py`, `nsdiff_monthly_multiseed.py`,
      `build_monthly_pairs.py`, `analyze_nsdiff_monthly.py`), aucune édition.
- [x] **pytest vert avant ET après** : `experiments/` 145 passed, `validation/`
      188 passed, `models/` 88 passed — identiques avant et après la génération
      + l'insertion (aucune régression).

---

## 7. Pièges évités (brief §9)

- Aucune sur-interprétation d'un "significatif" isolé sur une seule graine :
  le motif crypto est présenté avec sa réserve multi-graines explicite (§5bis),
  jamais comme acquis.
- Aucun mois partiel en cours régénéré : `build_monthly` exclut explicitement
  le mois calendaire courant à chaque exécution (vérifié : la borne test la
  plus récente est 2026-04-30, pas le mois en cours au moment du run).
- SARIMA/Prophet mensuels **non inclus** (saisonnalités calées daily/weekly,
  brief §4) — hors-scope déclaré, pas un oubli.
- Le mensuel n'a touché ni les tables/agrégats de `dashboard_d7_w1.py`, ni
  les lignes weekly/daily des 7 modèles existants (vérifié §1/§6).
- Aucun verdict mensuel présenté comme un résultat fort et définitif : c'est
  un complément de la carte des horizons (daily/weekly/mensuel), sous
  réserve de puissance assumée du début à la fin.

---

## 8. Verdict pour le tuteur

**Le mensuel complète la carte des horizons sans la trancher — comme
annoncé.** À `effective_n≈13` (contre ~30 au weekly), la majorité des
cellules (Actions, Obligations — 9/9) sont **indistinguables entre régime
daily-poussé et régime mensuel-natif**, exactement l'issue attendue d'un
historique de ~130 mois. **Le seul signal net** est sur les 2 actifs
crypto : le régime daily bat significativement le régime mensuel-natif sur
le RMSE aux 3 horizons M+1/M+2/M+3 (6/6 cellules) et sur le skill-score
Winkler à toutes les classes (4/4, y compris là où le RMSE ne tranche pas) —
mais ce motif crypto **n'est pas garanti robuste graine à graine** (BTC-USD
4/5, ETH-USD 3/5), donc à présenter comme "penche vers le daily, pas prouvé
au sens d'un verdict univoque".

**Découverte accessoire mais cohérente** : le régime mensuel-natif produit
systématiquement des PI 2-4× plus larges avec une couverture quasi-100%
(sur-couvert) contre une couverture plus proche de la cible côté daily —
plausiblement un effet de la rareté des données d'entraînement mensuelles
(34-70 fenêtres vs des milliers côté daily), pas démontré mécaniquement mais
cohérent avec l'avertissement de puissance du brief.

**Recommandation pratique** : à horizon 1-3 mois, pousser un modèle NsDiff
**daily** vers la cible plutôt que d'entraîner NsDiff nativement sur des
rendements mensuels — l'historique mensuel disponible ici (~130 points) est
simplement trop court pour que la variante mensuelle-native calibre
correctement son incertitude, et sur crypto (le seul cas où l'effet perce
malgré la faible puissance), elle perd aussi en précision de point.
