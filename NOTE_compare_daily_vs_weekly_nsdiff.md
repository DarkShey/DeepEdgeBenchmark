# NOTE — NsDiff : régime daily (B) vs régime weekly-natif (C), cible W+1

*2026-08-04, mise à jour suite à `BRIEF_nsdiff_ameliorer_limites.md` (Fix 1 :
récupération de TLT, 5/5 actifs ; Fix 2 : robustesse multi-graines 42-46).
Suite de `BRIEF_synthese_daily_vs_weekly_nsdiff.md`, de
`NOTE_nsdiff_dashboard_daily_oos.md` (première passe daily-vs-weekly, à
approfondir ici) et de `BRIEF_nsdiff_dashboard_daily_oos.md`. Source :
`experiments/dashboard_d7_w1_data.json` (régénéré, 35 cellules, `model==
'NsDiff'`) + `experiments/extract_nsdiff_daily_vs_weekly.py` (extraction/
recoupement, aucun modèle relancé côté agrégat) + `experiments/oos_nsdiff_tlt.py`
(Fix 1, insertion TLT en oos) + `experiments/nsdiff_daily_weekly_multiseed.py`
(Fix 2, artefact isolé, jamais dans `tracking.db`). Ne pas confondre avec
`NOTE_compare_weekly_tsdiff_nsdiff.md` (TSDiff vs NsDiff, question distincte).*

---

## 1. Cadrage — ce que compare cette note (et ce qu'elle ne compare pas)

**Régime daily (B) vs régime weekly-natif (C), à cible identique W+1** (même
`target_date`-vendredi, même `cutoff_date` — vérifié 100% identiques par le
dashboard). Les deux régimes prédisent **la même chose** (le prix à 1
semaine), par deux chemins différents :

- **Régime B (daily)** : NsDiff entraîné/fit en fréquence quotidienne, sa
  prévision est lue à la distance en jours de bourse réelle jusqu'au
  vendredi-cible (pas de règle "+7j calendaires" en dur).
- **Régime C (weekly natif)** : NsDiff entraîné/fit en fréquence
  hebdomadaire, sur les mêmes origines.

**Ce n'est PAS** une comparaison d'horizon (1 jour vs 1 semaine) — c'est une
comparaison de **régime d'entraînement/inférence** à horizon de cible fixe.

---

## 2. Tableau par actif (5/5 actifs — TLT récupéré, Fix 1 §7)

Source : `dashboard_d7_w1_data.json` régénéré après insertion de TLT
(`experiments/oos_nsdiff_tlt.py`, 35 cellules = 7 modèles × 5 actifs),
cellules `model=='NsDiff'`. Seul le **RMSE** est testé statistiquement par
cellule (bootstrap par blocs, seed interne 0, `block_length=3`, n=90
origines/actif → `effective_n=30` blocs quasi indépendants) ; Cov95/largeur
PI/Winkler/direction sont des lectures **descriptives**, non testées à ce
niveau.

| Actif | Classe | RMSE (daily → weekly) | Cov95 (cible 95%) | Largeur PI (daily → weekly) | Winkler (daily → weekly) | Direction (daily → weekly) | Verdict cellule (RMSE) | p | n / eff_n |
|---|---|---|---|---|---|---|---|---|---|
| BTC-USD | Crypto | 5917.0 → 5452.0 | 90.0% → 96.7% | 21 247 → 25 904 | 37 237 → 29 605 | 51.1% → 47.8% | **weekly natif significativement meilleur** | 0.0048 | 90 / 30 |
| ETH-USD | Crypto | 293.8 → 285.6 | 86.7% → 93.3% | 990.7 → 1113.0 | 1942.7 → 1440.5 | 48.9% → 43.3% | indistinguable | 0.3956 | 90 / 30 |
| SPY | Actions | 13.61 → 13.11 | 82.2% → 92.2% | 42.22 → 47.43 | 98.13 → 72.16 | 47.8% → 52.2% | indistinguable | 0.2804 | 90 / 30 |
| ZN=F | Obligations | 0.7956 → 0.7956 | 85.6% → 87.8% | 2.585 → 2.839 | 4.476 → 4.189 | 43.3% → 52.2% | indistinguable | 0.9256 | 90 / 30 |
| **TLT** | **Obligations** | 1.4270 → 1.4290 | 88.9% → 95.6% | 4.765 → 6.117 | 8.530 → 7.931 | 47.8% → 43.3% | indistinguable | 0.8804 | 90 / 30 |

Lecture immédiate : **une seule cellule est statistiquement tranchée** —
BTC-USD, en faveur du weekly natif. Les 4 autres sont **indistinguables** sur
le point (RMSE) au niveau cellule, même si Cov95/Winkler penchent
systématiquement du côté weekly (cf. §4) — TLT confirme exactement le même
motif que ZN=F : indistinguable sur RMSE, Cov95 sensiblement plus proche de
la cible côté weekly (88.9%→95.6%, quasi pile sur la cible), Winkler
légèrement meilleur côté weekly malgré une PI plus large.

---

## 3. Agrégat par classe + global — **NsDiff seul** (skill-score RMSE et Winkler vs RW)

### 3.0 Point méthodologique important : l'`aggregate` du JSON n'est PAS filtrable par modèle

Le champ top-level `aggregate` de `dashboard_d7_w1_data.json` (et le tableau
correspondant en fin de `NOTE_nsdiff_dashboard_daily_oos.md` §6) est un
**pooling des 7 modèles ensemble** par origine (`build_pooled_series` moyenne
les diffs de skill de *tous* les `(model, asset)` contribuant à une origine
donnée, avant le bootstrap) — ce n'est pas un agrégat par modèle, et il n'y a
donc **aucun agrégat NsDiff-seul déjà présent dans le JSON**. Vérifié en
retraçant `dashboard_d7_w1.py::build_pooled_series`/`run_pooled_test` : les
p-values de la table §6 de `NOTE_nsdiff_dashboard_daily_oos.md` (RMSE bond
p=0.0034, Winkler bond p=0.165, etc.) correspondent **exactement** à
`dashboard['aggregate']['bond']` — c'est-à-dire au pooling des 7 modèles, pas
de NsDiff seul. **Cette table de la note précédente est donc mal étiquetée**
(elle dit "NsDiff" mais chiffre le pool des 7 modèles) — à corriger, c'est
fait ici.

`experiments/extract_nsdiff_daily_vs_weekly.py` réimporte directement
`build_enriched_pairs`/`build_pooled_series`/`run_pooled_test` de
`dashboard_d7_w1.py` (mêmes fonctions, **aucune réimplémentation**), filtre
les paires à `model=='NsDiff'` avant le pooling, même seed (42), mêmes
origines (même DB, même cache prix local `.price_cache_d7_w1/`, aucun accès
réseau). **Recoupement vérifié bit-à-bit** : le RMSE daily/weekly recalculé
par cellule (moyenne des `sq_error`) est identique à 1e-6 près aux **5**
cellules NsDiff du JSON dashboard (TLT inclus depuis Fix 1).

### 3.1 Résultat — agrégat NsDiff seul, poolé par classe et global (5/5 actifs)

| Groupe | n_origines (n_contrib.) | Skill RMSE — verdict | p | eff_n | Skill Winkler — verdict | p | eff_n |
|---|---|---|---|---|---|---|---|
| **Global** (5 actifs) | 95 (361) | indistinguable | 0.1578 | 31 | **weekly natif significativement meilleur** | 0.0100 | 31 |
| **Crypto** (BTC-USD, ETH-USD) | 90 (180) | **weekly natif significativement meilleur** | 0.0284 | 30 | **weekly natif significativement meilleur** | 0.0154 | 30 |
| **Actions** (SPY seul) | 90 (90) | indistinguable | 0.2874 | 30 | indistinguable | 0.1012 | 30 |
| **Obligations** (ZN=F + TLT, dédoublonnés — réserve « ZN=F seul » levée, Fix 1) | 91 (91) | indistinguable | 0.9594 | 30 | indistinguable | 0.4352 | 30 |

`effective_n` toujours affiché (puissance faible à assumer, convention
maison) : **30-31 blocs quasi indépendants**, pas 90-95 — à garder en tête
pour toute lecture de significativité. Le verdict "Obligations" repose
désormais sur les **deux** actifs de taux (ZN=F + TLT, moyennés par origine
avant le bootstrap — même dédoublonnage que pour les 6 autres modèles) et
reste **indistinguable** sur les deux axes, un chiffre légèrement plus près
de zéro qu'avec ZN=F seul (p=0.96 vs 0.93 sur RMSE ; p=0.44 vs 0.61 sur
Winkler) mais la conclusion ne change pas.

---

## 4. Lecture par métrique

### 4.1 Précision (RMSE)
Sur la **seule graine 42** (single-seed, celle du dashboard `oos`), le
weekly-natif semble aider sur la crypto (skill RMSE poolé p=0.0284, porté
presque entièrement par BTC-USD, verdict cellule significatif p=0.0048).
**Mais le passage multi-graines (§6bis) montre que ce résultat ne se
reproduit PAS** : sur BTC-USD, 4 des 5 graines (43-46) sont indistinguables,
et aucune autre cellule (SPY, ZN=F) n'est significative à plus d'une graine
sur cinq non plus. **La lecture honnête est donc : à ce niveau de
puissance (n=90/actif, effective_n=30), aucune cellule ne montre un avantage
RMSE robuste du weekly-natif sur le daily** — le "weekly significativement
meilleur sur BTC-USD" de la graine 42 est un effet de tirage, pas un effet
de régime stable. Ailleurs (actions, obligations), toujours indistinguable
sur toutes les graines testées. Aucune cellule ne montre "daily
significativement meilleur" sur aucune graine — contrairement à ce que
suggérait (à tort, cf. §3.0) la table pool-7-modèles de la note précédente.

### 4.2 Calibration (Cov95) et finesse (largeur PI)
**Signal directionnel très cohérent, mais pas partout significatif.** Sur les
4 cellules, sans exception, le weekly-natif est **mieux couvert** (Cov95 plus
proche de 95%, le daily sous-couvre systématiquement : 82–90% contre 88–97%
côté weekly) — et sur les 4 cellules aussi, le **Winkler descend** côté
weekly, malgré une PI **plus large** dans 3 cas sur 4 (BTC, ETH, SPY — ZN=F
fait exception, PI un peu plus large aussi mais écart minime). Le gain de
couverture n'est donc pas un simple élargissement gratuit des bandes : il
compense (et au-delà) le coût de largeur dans le score composite. **Mais**
seul le pooling **crypto** (p=0.0154) et **global** (p=0.0190) atteint la
significativité sur le skill Winkler — SPY (p=0.10) et ZN=F (p=0.61) restent
indistinguables malgré la même direction descriptive, à `effective_n=30`
sur un seul actif chacun : puissance insuffisante pour trancher, pas une
absence d'effet démontrée.

### 4.3 Où ça bascule par classe
Sur la **précision (RMSE)**, une fois la robustesse multi-graines prise en
compte (§6bis), **aucune classe ne bascule de façon fiable** — le "crypto
gagne" de la graine 42 ne survit pas au passage à 5 graines. Sur la
**calibration (Winkler)**, en revanche, la crypto reste la classe où le
pooling seed-42 est significatif (p=0.0154, §3.1) ET où la direction
descriptive est la plus consistante à travers les graines (§6bis, BTC et ETH
tous deux nettement plus bas côté weekly sur les 5 graines) — cohérent avec
l'hypothèse (non testée formellement, un seul actif de chaque autre classe
ne permet pas de trancher) qu'un actif 24/7 sans vraie clôture pénalise
davantage un modèle daily poussé multi-step qu'un modèle weekly-natif. Sur
**actions (SPY)** et **obligations (ZN=F, TLT)**, où le daily voit de vraies
clôtures de bourse consécutives, aucun des deux régimes ne l'emporte de
façon significative ou stable — daily et weekly-natif restent
interchangeables au niveau de puissance actuel.

### 4.4 Croisement avec le finding de fond (« la diffusion perd en précision, gagne en calibration »)
Ce finding (`methodologie_diffusion_vs_classiques.md` §3) porte sur un autre
axe : **diffusion (daily) vs modèles classiques (daily)**, où TSDiff perd
nettement en précision (MAE 460 vs 407-413) pour ne gagner qu'en calibration.
Ici, la comparaison est **intra-NsDiff, daily vs weekly**, un axe différent.
Le motif se retrouve **partiellement** : le passage daily → weekly-natif de
NsDiff **n'améliore pas la précision de façon défendable** (indistinguable
partout, y compris en crypto une fois la robustesse multi-graines prise en
compte, §6bis) tout en améliorant **la calibration en lecture descriptive**,
de façon plus stable à travers les graines (significatif en crypto/global
sur Winkler à la graine 42, direction cohérente sur 4 actifs/5 en moyenne
multi-graines, §6bis) —
c'est bien le même déséquilibre qualitatif (le gain se loge côté fiabilité de
l'incertitude, pas côté point). **Mais NsDiff ne "perd" pas en précision en
passant au weekly** comme TSDiff perdait face aux classiques : au pire
indistinguable, et même significativement meilleur sur crypto. Ce n'est donc
pas un miroir exact du finding TSDiff-vs-classiques — c'est une version plus
favorable : là où la diffusion payait un vrai coût de précision pour gagner
en calibration (face aux classiques), le passage daily→weekly de NsDiff
**gagne en calibration sans perdre en précision** (dans les limites de la
puissance disponible).

---

## 5. Bonus — W+2/W+3 (si disponible, §4 du brief)

Les lignes `oos` NsDiff couvrent aussi W+2/W+3 (`NOTE_nsdiff_dashboard_daily_oos.md`
§1, 1080+1080 lignes insérées, 3 horizons × 2 régimes). Extrait via
`comparison_3_daily_vs_weekly` (test par cellule déjà utilisé par le
dashboard, réutilisé tel quel) :

| Horizon | Actif | Verdict RMSE | p | Cov95 (daily → weekly) | Winkler (daily → weekly) |
|---|---|---|---|---|---|
| W+2 | BTC-USD | **weekly natif significativement meilleur** | 0.0006 | 87.8% → 93.3% | 51 950 → 44 790 |
| W+2 | ETH-USD | indistinguable | 0.4186 | 83.3% → 91.1% | 2472 → 2022 |
| W+2 | SPY | indistinguable | 0.2414 | 90.0% → 93.3% | 93.0 → 100.8 |
| W+2 | TLT | indistinguable | 0.5584 | 94.4% → 95.6% | 10.18 → 10.42 |
| W+2 | ZN=F | indistinguable | 0.4104 | 90.0% → 90.0% | 5.22 → 5.19 |
| W+3 | BTC-USD | indistinguable (p proche du seuil) | 0.0824 | 86.7% → 93.3% | 76 640 → 62 490 |
| W+3 | ETH-USD | indistinguable | 0.9288 | 77.8% → 90.0% | 3691 → 2645 |
| W+3 | SPY | indistinguable | 0.3596 | 92.2% → 93.3% | 112.6 → 131.2 |
| W+3 | TLT | indistinguable | 0.7494 | 96.7% → 98.9% | 10.58 → 11.10 |
| W+3 | ZN=F | indistinguable | 0.9152 | 93.3% → 96.7% | 6.05 → 5.41 |

**Cohérent avec W+1** : BTC-USD reste le seul cas où le weekly-natif domine
nettement sur le point (significatif à W+1 et W+2, p=0.08 — proche du seuil,
pas franchi — à W+3), et la direction Cov95/Winkler favorable au weekly
persiste sur quasiment toutes les cellules et tous les horizons, sans jamais
devenir significative hors BTC. **Lecture bonus, non testée en pooling par
classe** (pas de skill-score RW recalculé à ces horizons, hors scope de
cette extraction) — à prendre comme confirmation qualitative du motif W+1,
pas comme un résultat indépendant à part entière.

---

## 6bis. Robustesse multi-graines (Fix 2, BRIEF_nsdiff_ameliorer_limites.md)

**Cadrage** : la piste `oos`/dashboard reste single-seed (42), comparable aux
6 autres modèles — **rien n'y a été modifié**. Cette section vient d'un
artefact **isolé**, `experiments/nsdiff_daily_weekly_multiseed.json`, produit
par `experiments/nsdiff_daily_weekly_multiseed.py` (jamais écrit dans
`tracking.db`) : NsDiff daily(B)+weekly(C) relancé sur les graines **42-46**,
mêmes origines (lues verbatim), mêmes budgets (epochs=40, n_samples=50,
k_denoise=20), verdict par cellule au niveau W+1 obtenu via
`comparison_3_daily_vs_weekly` (réutilisé tel quel, seed interne 0 comme le
dashboard). 3,3 min pour les 5 graines × 5 actifs.

### Table verdict × graine (test RMSE, W+1, n=90/eff_n=30 à chaque graine)

| Actif | seed 42 (p) | seed 43 (p) | seed 44 (p) | seed 45 (p) | seed 46 (p) | Stable sur 5 graines ? |
|---|---|---|---|---|---|---|
| BTC-USD | **weekly** (0.005) | indist. (0.141) | indist. (0.071) | indist. (0.664) | indist. (0.836) | **NON** — 1/5 |
| ETH-USD | indist. (0.396) | indist. (0.804) | indist. (0.521) | indist. (0.672) | indist. (0.225) | **oui** — indist. 5/5 |
| SPY | indist. (0.280) | indist. (0.988) | indist. (0.673) | **weekly** (0.004) | indist. (0.261) | **NON** — 1/5 |
| ZN=F | indist. (0.926) | indist. (0.482) | indist. (0.976) | indist. (0.738) | **weekly** (0.050, limite) | **NON** — 1/5 |
| TLT | indist. (0.880) | indist. (0.788) | indist. (0.292) | indist. (0.784) | indist. (0.050, limite) | **oui** — indist. 5/5 |

### CV inter-graines (std/mean sur les 5 graines, RMSE et Winkler moyens par cellule)

| Actif | CV RMSE daily | CV RMSE weekly | CV Winkler daily | CV Winkler weekly |
|---|---|---|---|---|
| BTC-USD | 7.3% | 2.8% | 11.8% | 2.9% |
| ETH-USD | 4.3% | 2.4% | 11.6% | 5.1% |
| SPY | 2.6% | 1.6% | 11.3% | 4.2% |
| ZN=F | 1.7% | 1.7% | 4.0% | 5.9% |
| TLT | 0.8% | 2.4% | 5.8% | 1.4% |

### Réponse à la question centrale : le verdict « weekly meilleur sur crypto » tient-il sur 5 graines ?

**Non, pas au niveau où la note le disait initialement.** Le verdict cellule
« weekly natif significativement meilleur » sur BTC-USD à la graine 42
(p=0.0048, celle du dashboard `oos` single-seed) **ne se reproduit à
AUCUNE des 4 autres graines** (p allant de 0.07 à 0.84) — c'est un effet de
graine, pas un effet de régime stable, exactement le type de renversement
que le brief demandait de vérifier honnêtement (cf. l'expérience équivalente
côté TSDiff dans le duel). Sur 25 tests (5 actifs × 5 graines), seuls 3
franchissent p<0.05, chacun sur une graine différente (BTC@42, SPY@45,
ZN=F@46 à la limite) — un taux (~12%) pas très éloigné de ce qu'on
attendrait par chance sans correction pour tests multiples (~5% par test,
plus si les tests sont légèrement corrélés) : **aucune cellule ne montre un
verdict RMSE robuste à travers les graines**, ETH-USD et TLT restant
indistinguables sur les 5.

**Ce que ça change dans la lecture de la note** : la conclusion « weekly
natif significativement meilleur sur RMSE en crypto » (§3.1, §4.1, §7,
issue du dashboard single-seed 42) **doit être dégradée** — au niveau
cellule/robustesse graine, ce n'est PAS un effet établi, c'est un résultat
qui tient à un seul tirage de graine sur cinq. **Ce qui, en revanche, tient
mieux au niveau descriptif** (moyenne sur les 5 graines, pas de test dessus,
à lire comme tendance et non comme un verdict statistique) : le Winkler
moyen reste plus bas côté weekly sur 4 actifs sur 5 (BTC 31490→29025, ETH
1611→1356, SPY 87.5→69.9, TLT 8.10→7.92 — seul ZN=F s'inverse légèrement,
4.365→4.404) — la direction "calibration côté weekly" est plus stable dans
les moyennes que la significativité RMSE crypto ne l'est dans les tests,
mais ceci reste une lecture descriptive multi-graines, pas un test poolé par
graine (non fait ici, faute de temps — cf. limites §6).

---

## 6. Non-négociables — statut

- **Aucun modèle relancé, aucun test réimplémenté** : `extract_nsdiff_daily_vs_weekly.py`
  importe et appelle `build_enriched_pairs`/`build_pooled_series`/
  `run_pooled_test`/`comparison_3_daily_vs_weekly`/`winkler_score` de
  `dashboard_d7_w1.py`/`matrice_paired_tests.py` **tels quels** — aucune
  fonction de test ou de fit n'est recopiée. `tracking.db` lu en lecture
  seule ; cache prix local réutilisé (aucun accès réseau).
- **p-values recoupées** : mêmes seed (42 poolé, 0 par cellule), mêmes
  origines (mêmes 90/91/95 origines que le dashboard) — recoupement
  bit-à-bit vérifié sur les 4 RMSE de cellule (§3.0) ; le nouvel agrégat
  NsDiff-seul (§3.1) diffère volontairement de l'agrégat pool-7-modèles du
  JSON, pour la raison documentée en §3.0 (pas la même question).
- **Aucune conclusion sur une cellule/agrégat non significatif** : ETH-USD,
  SPY, ZN=F au niveau cellule, et Actions/Obligations au niveau agrégat sont
  rapportés « indistinguable » avec `n`/`effective_n`, jamais interprétés
  comme un verdict.
- **Limites déclarées (allégées depuis `BRIEF_nsdiff_ameliorer_limites.md`)** :
  - **TLT** : **levée** — récupéré via une source de prix hybride vérifiée
    (offline jusqu'au 2026-07-02 + close brut live pour la queue
    2026-07-03→07-24, recollée aux 180 `last_close`/`y_true` stockés à
    1e-6 près, cf. `experiments/oos_nsdiff_tlt.py`). 5/5 actifs, 35 cellules,
    réserve "obligations = ZN=F seul" levée (§3.1).
  - **Single-seed** : **partiellement levée** — table de robustesse
    multi-graines ajoutée (§6bis, graines 42-46, artefact isolé). Elle
    **change la lecture** : le "weekly significativement meilleur" sur BTC-USD/
    crypto n'est valable qu'à la graine 42 (dashboard `oos`), pas un effet
    stable. Limite résiduelle : robustesse testée seulement au niveau cellule
    W+1 (RMSE) ; pas de re-test du skill-score poolé par classe à chaque
    graine (aurait demandé de rejouer `build_enriched_pairs`/RW-scale par
    graine, hors budget de cette passe) — direction Winkler multi-graines
    lue en descriptif seulement (moyennes, pas de test), pas en verdict.
  - **Scope W+1** pour le corps de la note (§1-4, §6bis) ; W+2/W+3 en bonus
    non poolé, single-seed (§5).
- **Aucun fichier source modifié hors ajout** : fichiers nouveaux,
  `experiments/extract_nsdiff_daily_vs_weekly.py`, `experiments/oos_nsdiff_tlt.py`,
  `experiments/nsdiff_daily_weekly_multiseed.py` (+ artefacts JSON associés).
  `oos_nsdiff_daily_weekly.py`/`dashboard_d7_w1.py`/`offline_prices.py`
  importés et réutilisés tels quels, **non modifiés**. pytest vert avant ET
  après (`python -m pytest experiments validation -q` depuis la racine →
  **333 passed**, cf. §6ter).

### 6ter. Intégrité des lignes `oos` (contrôle avant/après)
- `tracking.db` avant Fix 1 : 27174 lignes totales, 2160 `source='oos'
  model='NsDiff'`. Après insertion TLT (540 lignes, `oos_nsdiff_tlt.py`) :
  **27714** (+540 ✓) et **2700** NsDiff `oos` (+540 ✓, dont 540 `asset='TLT'`).
  Aucune ligne des 6 autres modèles ni des lignes NsDiff `live`/
  `backtest_rolling_nsdiff` touchée (upsert idempotent scoping sur
  `model='NsDiff', asset='TLT'` uniquement).
- Fix 2 (multi-graines) : **aucune écriture dans `tracking.db`** — seul
  artefact touché, `experiments/nsdiff_daily_weekly_multiseed.json` +
  `experiments/checkpoints_nsdiff_multiseed/*.json`.
- Dashboard régénéré : **35 cellules** (7 modèles × 5 actifs).

---

## 7. Verdict pour le tuteur

Pour NsDiff à cible W+1, sur les **5 actifs** (TLT récupéré, Fix 1) et
**vérifié sur 5 graines** (Fix 2) : **sur la précision (RMSE), aucun
avantage robuste du weekly-natif nulle part** — le résultat "weekly
significativement meilleur sur BTC-USD/crypto" affiché par le dashboard
single-seed (graine 42) **ne se reproduit pas** sur les graines 43-46
(§6bis) ; à ce niveau de puissance (`effective_n=30`), daily poussé et
weekly-natif sont **interchangeables sur le point, sur toutes les classes**.
**Sur la calibration (Winkler)**, le signal est plus robuste sans être
généralisé : significatif en crypto et au global à la graine 42 (p=0.015 et
p=0.010), et la direction "Winkler plus bas côté weekly" se retrouve dans
les moyennes multi-graines sur 4 actifs sur 5 (tout sauf ZN=F) — mais ceci
reste une lecture descriptive (pas de test poolé par graine ici, §6). Sur
**actions (SPY) et obligations (ZN=F + TLT, réserve levée)**, aucun des deux
axes n'est jamais tranché de façon stable. Le motif de fond « la diffusion
perd en précision, gagne en calibration » se retrouve donc version encore
plus allégée qu'initialement estimé : le passage daily→weekly de NsDiff
**ne perd pas en précision** (jamais "daily significativement meilleur", sur
aucune graine) mais son **gain de précision en crypto n'était pas
établi non plus** — seul le déplacement vers une meilleure calibration
(Winkler) tient, en lecture descriptive, à travers les graines. **Message
principal pour le tuteur : la version single-seed de cette synthèse
surestimait la solidité de l'avantage crypto du weekly-natif ; la version
robuste (5 graines) ramène le résultat à quelque chose de plus modeste mais
plus défendable — pas de perte de précision daily→weekly, gain de
calibration plausible mais non testé formellement en multi-graines.**
