# BRIEF v2 — Chaque modèle à son propre sweet spot d'epochs (choisi équitablement)

> Suite du run 30 origines × 300 epochs (`experiments/weekly_headtohead_results.json`),
> verdict **non concluant** à cause d'un collapse de l'incertitude. Ce brief re-cadre la mission.

## 0. Rappel de la situation (ce que l'audit a établi)

À 300 epochs, TSDiff **sur-apprend** : ses échantillons s'effondrent vers une trajectoire quasi
unique (`dernier_prix × dérive` ≈ une marche aléatoire). Conséquences mesurées :

- **Incertitude effondrée** : dispersion des échantillons 3.29 % à 3 semaines (40 ep) → 0.09 %
  (150 ep) → 0.07 % (300 ep). Les intervalles deviennent ~40× trop étroits → **Cov95 0.00–0.10**
  au lieu de 0.95.
- **Les deux modèles deviennent identiques** : en collapse, W et D convergent vers la même RW →
  RMSE au dollar près (ex. BTC W1 : 4650.28 vs 4650.88). Comparaison non informative.

Le protocole (train-once-forward, mu/sd figés, no-lookahead, dates-cibles partagées) est **sain
et testé** — le problème est le **régime d'entraînement**, pas l'expérience.

## 1. Idée directrice

Le collapse vient du **sur-entraînement**. On teste donc l'hypothèse la plus simple d'abord :
**il existe, pour chaque modèle, un nombre d'epochs plus bas où il est bien entraîné SANS
s'effondrer.** Si oui, on n'a besoin d'aucune machinerie de calibration — juste du bon réglage.

**Point clé (et correction du v1)** : les deux modèles n'ont PAS à utiliser le même nombre
d'epochs. Weekly (données hebdo, horizon 3) et daily (données quotidiennes, horizon 15) sont
deux modèles différents ; leur bon nombre d'epochs peut différer. Forcer un chiffre identique
peut en handicaper un.

Ce qui rend la comparaison **juste**, ce n'est pas un chiffre d'epochs identique — c'est que
**chaque modèle soit choisi à son meilleur, par la même règle**. (Analogie : dans une course
coureur vs cycliste, on ne donne pas « le même nombre de coups de pédale » ; chacun va à son
meilleur rythme et on compare le temps d'arrivée.)

## 2. Décision de conception assumée

Le nombre d'epochs est un **hyperparamètre**, sélectionné pour **chaque modèle séparément** en
minimisant le **CRPS sur un jeu de validation** (le CRPS est une *proper scoring rule* : il
pénalise à la fois un mauvais point ET une mauvaise incertitude — un modèle effondré/sur-confiant
est puni). Interdiction absolue de choisir les epochs en regardant le jeu de test → sinon on
ajuste un réglage sur la métrique qu'on cherche à mesurer (triche).

Le nombre d'epochs final peut donc différer entre W et D (ex. W=60, D=90) : c'est normal et
équitable tant que la règle de sélection est la même.

## 3. Le découpage validation / test (sans fuite temporelle)

Trois blocs chronologiques stricts, du plus ancien au plus récent :

```
[ ——— entraînement (≤ T0) ——— | — validation (V origines) — | — test (30 origines) — ]
                              T0                            T1
```

- **Entraînement** : données ≤ T0.
- **Validation** : ~10–15 origines juste après T0 → sert **uniquement** à choisir les epochs de
  chaque modèle (par CRPS de validation).
- **Test** : les 30 origines après T1 → sert **uniquement** au verdict final, jamais vu pendant
  la sélection.

Validation strictement **avant** le test → aucune fuite. Même découpage pour W, D et le baseline.

## 4. Le balayage d'epochs

- **Candidats** : {40, 60, 80, 100, 120} (zone sous la falaise 40→150 ; extensible si besoin).
- Pour chaque (modèle × actif × candidat) : entraîner ≤ T0, prévoir sur les origines de
  validation, mesurer **CRPS de validation** + Cov95 de validation (contrôle).
- **Sélection** : pour chaque modèle, l'epoch* qui minimise le CRPS de validation.
- **Diagnostic à garder** : la dispersion `rel_std%` à chaque candidat (voir si/où le collapse
  commence pour ce nouveau daily horizon 15, qu'on n'a jamais mesuré à bas epochs).

## 5. Baseline marche aléatoire (référence, léger)

Aux mêmes origines/dates-cibles : point = `dernier_prix`, intervalle = quantiles des rendements
historiques cumulés à h semaines (fenêtre ≤ origine). C'est le **plancher** : si aucun TSDiff ne
le bat, le modèle n'apporte rien et la question weekly/daily est prématurée. (Une prévision, ça
doit au minimum battre « demain = aujourd'hui ».)

## 6. Métriques & significativité (sur le jeu de test, aux epochs sélectionnés)

Par (actif × horizon × modèle ∈ {TSDiff-W, TSDiff-D, RW}) :
- **RMSE** — précision du point.
- **Cov95** — couverture réelle (doit approcher 0.95 si le sweet spot existe vraiment).
- **Largeur moyenne d'intervalle** — finesse (départage à couverture égale).
- **CRPS** — note globale.
- **Test apparié obligatoire** : différence par origine de CRPS entre modèles → bootstrap
  apparié (ou Diebold-Mariano). On ne conclut « W ≠ D » (ni « bat la RW ») que si l'écart est
  **significatif**. Fini les « 4650.28 vs 4650.88 » commentés comme des différences.

## 7. Critères de décision

- **G1 — TSDiff sert-il à quelque chose ?** Aux epochs sélectionnés, TSDiff-W ou TSDiff-D
  bat-il significativement la RW (CRPS) ? Si **non** → STOP, on documente que TSDiff n'apporte
  rien à ce régime (reco : filet de secours §9). Si **oui** → on passe à G2.
- **G2 — Weekly vs Daily.** Écart W vs D significatif sur CRPS/finesse à couverture égale ?
  - Significatif → **verdict net** → déclenche l'extension multi-modèles (ancienne §9 du v1).
  - Non significatif → weekly ≈ daily une fois chacun bien réglé ; on documente, pas d'extension.

## 8. Plan d'implémentation

1. **Découpage** train / validation / test (§3) dans le head-to-head.
2. **Boucle de balayage** : entraîner chaque (modèle × actif) aux 5 candidats, scorer sur
   validation (CRPS + Cov95 + rel_std%). Sauver dans `experiments/epoch_sweep_results.json`.
3. **Sélection** epoch* par modèle (min CRPS validation).
4. **Baseline RW** aux mêmes origines/dates-cibles.
5. **Head-to-head final** aux epochs sélectionnés sur les 30 origines de test + baseline.
6. **Test apparié** (`experiments/paired_test.py`) sur les différences de CRPS par origine.
7. **Résultats** → `experiments/weekly_headtohead_v2_results.json` (epochs choisis, métriques
   test, p-values, verdict G1 puis G2).
8. **Vérification** : (a) les epochs sont bien choisis sur validation, jamais sur test ;
   (b) validation strictement antérieure au test (pas de fuite) ; (c) mêmes dates-cibles W/D/RW ;
   (d) relire les p-values avec l'œil « 30 origines = puissance limitée ».
9. **Rédaction** du verdict honnête.

## 9. Filet de secours (si le balayage échoue)

Si aucun candidat ne donne à la fois une couverture décente ET une précision correcte (piste
probable : la Phase 0 montrait le daily déjà sous-calibré même à 40 epochs) → alors le réglage
d'epochs seul ne suffit pas, et **là seulement** on active la calibration a posteriori
(conformal) pour réparer les intervalles sans toucher aux points. À ne PAS faire avant d'avoir
la preuve que le balayage échoue.

## 10. Ce que ce brief ne fait PAS

- Pas de re-run à 300 epochs avec 2ᵉ graine (collapse déterministe en epochs, pas en graine).
- Pas de nombre d'epochs identique imposé aux deux modèles (chacun son sweet spot, même règle).
- Pas de calibration a posteriori tant que le balayage n'a pas prouvé qu'elle est nécessaire.
- Pas d'extension multi-modèles / Dashboard tant que G2 n'a pas donné un verdict net.
