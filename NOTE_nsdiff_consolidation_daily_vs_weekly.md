# NOTE — NsDiff daily vs weekly : consolidation du verdict (tâches 1 à 7)

*2026-08-05. Réponse au brief « NsDiff : consolider le verdict daily vs
weekly ». **Supersède les conclusions de `NOTE_compare_daily_vs_weekly_nsdiff.md`
§4.2/§6bis/§7** (qui restent valables comme description de l'artefact
`n_samples=50`, mais dont le verdict « signal réel de calibration en faveur du
weekly » ne survit pas tel quel — voir §1 et §3). Ne remplace pas
`NOTE_compare_weekly_tsdiff_nsdiff.md` (TSDiff vs NsDiff, question distincte).*

**Artefacts produits** (tous isolés, `tracking.db` jamais écrit — la piste
`oos`/dashboard reste single-seed 42, comparable aux 6 autres modèles) :

| Fichier | Contenu |
|---|---|
| `experiments/nsdiff_multiseed_v2.py` + `nsdiff_multiseed_v2/` | tâche 4 : régénération 5 graines × 5 actifs × 3 horizons × 2 régimes à `n_samples=200`, nuages complets conservés (13 500 lignes × 200 tirages, 8 min) |
| `experiments/calibration_tests.py` (+ `test_calibration_tests.py`) | briques neuves : Kupiec, Christoffersen, TOST, test d'écart de couverture par blocs — 19 tests unitaires |
| `experiments/nsdiff_consolidation_tests.py` → `.json` | tâches 1, 2, 3 |
| `experiments/nsdiff_conformal_daily.py` → `.json` | tâche 5 |
| `experiments/nsdiff_seed_ensemble.py` → `.json` | tâche 6 |
| `experiments/nsdiff_vs_garch_w23.py` → `.json` | tâche 7 |
| `experiments/nsdiff_v2_data.py` | chargement/mise en forme partagés, aucune statistique |

**Suite** : `NOTE_duel_nsdiff_vs_tsdiff_budget_egal.md` étend le match
inter-modèles à TSDiff, aux deux régimes et à budget d'échantillonnage égal
(200 tirages des deux côtés, prix gelés partagés). Les deux notes partagent la
machinerie de test (`experiments/diffusion_headtohead.py`).

**Convention de lecture**, demandée par le brief et appliquée partout :

- **descriptif** = un chiffre, jamais testé ;
- **significatif à graine fixe** = test sur une seule graine (ce que faisait le
  dashboard `oos`) ;
- **significatif poolé multi-graines** = métrique moyennée sur les 5 graines
  origine par origine, puis testée. L'unité d'inférence reste **l'origine**
  (`n`=90, `effective_n`=30 inchangés) : moyenner sur les graines réduit le
  bruit Monte-Carlo, ça ne crée pas de points de donnée. Ce qui est testé
  ainsi est la performance **attendue d'un run à graine tirée au hasard** —
  pas celle d'un ensemble des 5 graines, qui est la tâche 6.

---

## 0. Le résultat principal, d'emblée

**Le « signal réel de calibration en faveur du weekly » de la synthèse
précédente était, pour l'essentiel, un artefact de `n_samples=50` — et il se
scinde en deux fois testé proprement :**

- **la couverture** : le weekly est bel et bien mieux calibré, le daily
  sous-couvre, et ça **résiste** au passage à 200 tirages, au pooling
  multi-graines et à deux familles de tests indépendantes → **établi** ;
- **le Winkler** (score composite largeur + pénalité) : l'avantage weekly
  **disparaît**. Il valait p=0.0100 en global et p=0.0154 en crypto à la
  graine 42 avec 50 tirages ; à 200 tirages, **la même graine 42** donne
  p=0.0500 et p=0.0510, et le poolé 5 graines donne **p=0.3180** (global,
  W+1). → **non établi**.

Autrement dit : le weekly **couvre mieux, mais il paie cette couverture en
largeur**, et une fois les bandes du daily correctement estimées, le score
composite ne départage plus les deux régimes. C'est plus modeste que
« weekly mieux calibré », et c'est ce que les données supportent.

Sur la **précision**, le brief demandait de conclure *positivement* : c'est
fait. Le TOST établit l'**équivalence à ±5 %** sur 4 actifs / 5 à W+1 et à
W+2 — ce n'est plus « indistinguable », c'est « interchangeable », avec une
marge déclarée a priori.

---

## 1. Tâche 4 (faite en premier, comme conseillé) — 50 → 200 tirages

`experiments/nsdiff_multiseed_v2.py`. Aucun réentraînement : mêmes fits, mêmes
graines (42-46), mêmes origines (lues verbatim), mêmes budgets
(epochs=40, seq_len=30, k_denoise=20) ; seul le **forecast** est refait avec
200 tirages au lieu de 50. 8 minutes pour 5 graines × 5 actifs × 3 horizons ×
2 régimes. Les nuages complets sont conservés sur disque, ce qui rend les
tâches 5 et 6 possibles sans refit.

**Pourquoi c'était le bon ordre.** À 50 tirages, le quantile empirique à
97,5 % est la ~49ᵉ statistique d'ordre sur 50 : il est **biaisé vers
l'intérieur**, donc les bandes sont trop étroites, donc il y a trop de
violations, donc le Winkler est pénalisé. L'effet est mesurable, et il n'a
**pas frappé les deux régimes également** :

Couverture 95 %, **graine 42 seule**, W+1 (colonne gauche : artefact
`n_samples=50` de la note précédente §2 ; colonne droite : artefact v2) :

| Actif | daily 50 → 200 | weekly 50 → 200 |
|---|---|---|
| BTC-USD | 0.900 → **0.922** | 0.967 → 0.967 |
| ETH-USD | 0.867 → **0.911** | 0.933 → 0.944 |
| SPY | 0.822 → **0.867** | 0.922 → 0.944 |
| ZN=F | 0.856 → **0.911** | 0.878 → 0.956 |
| TLT | 0.889 → **0.933** | 0.956 → 0.967 |

Le daily gagne **+4,2 points** de couverture en moyenne (médiane +4,4), le
weekly **+2,4** (médiane +1,1 — sa moyenne est tirée vers le haut par le seul
ZN=F, +7,8).
Hypothèse (cohérente, **non testée formellement** — elle demanderait un plan
d'expérience sur `n_samples` que ce brief ne demande pas) : le nuage du
régime daily est propagé sur ~5 pas quotidiens contre 1 pas hebdomadaire,
donc plus dispersé et à queues plus lourdes, et le biais du quantile
empirique à 50 tirages y coûte davantage. **Ce qui est certain, c'est la
conséquence** : une partie du déficit de couverture attribué au « régime
daily » était en fait un déficit d'échantillonnage.

**Conséquence directe sur la note précédente** : sa §4.2 (« le Winkler
descend côté weekly … malgré une PI plus large ») décrivait un artefact
autant qu'un effet de régime. Sur les moyennes 5 graines à 200 tirages, le
Winkler est **plus bas côté daily sur 2 actifs / 5** à W+1 (TLT 7.80 vs
8.02 ; ZN=F 4.15 vs 4.32) et quasi ex æquo sur BTC-USD (30 540 vs 30 430,
0,4 % d'écart) — à 50 tirages, le daily ne gagnait que sur ZN=F et l'écart
BTC était de 8 %.

---

## 2. Tâche 2 — W+2 et W+3 passés en multi-graines

Tout ce qui suit (§3, §4) est calculé aux **trois** horizons, 5 graines,
90 origines/actif, `effective_n`=30. Les W+2/W+3 ne sont plus un « bonus
qualitatif single-seed » (note précédente §5) : même protocole complet que
W+1. C'est utile, parce que **c'est là que le motif est le plus net** — la
sous-couverture du daily se creuse avec l'horizon :

| Actif | Cov95 daily W+1 → W+2 → W+3 | Cov95 weekly W+1 → W+2 → W+3 |
|---|---|---|
| BTC-USD | 0.933 → 0.911 → 0.884 | 0.967 → 0.949 → 0.942 |
| ETH-USD | 0.898 → 0.889 → **0.816** | 0.947 → 0.962 → 0.936 |
| SPY | 0.880 → 0.927 → 0.938 | 0.944 → 0.951 → 0.920 |
| TLT | 0.933 → 0.951 → 0.976 | 0.964 → 0.976 → 0.987 |
| ZN=F | 0.922 → 0.942 → 0.976 | 0.933 → 0.962 → 0.964 |

(graines moyennées.) La crypto se dégrade franchement côté daily ; les taux
partent au contraire en **sur**-couverture aux horizons longs, des deux côtés.

---

## 3. Tâche 1 — le signal calibration, testé formellement

### 3.1 Skill Winkler poolé (le test que la note précédente déclarait hors budget)

`dashboard_d7_w1.build_pooled_series` / `run_pooled_test` réutilisés tels
quels (dédoublonnage ZN=F+TLT en une contribution « taux » inclus), rejoués
**par graine** puis **sur les 5 graines poolées**.

| Horizon | Groupe | graines significatives (weekly) | p poolé 5 graines | verdict poolé |
|---|---|---|---|---|
| W+1 | global | **0/5** | 0.3180 | indistinguable |
| W+1 | crypto | 0/5 | 0.2946 | indistinguable |
| W+1 | actions | 0/5 | 0.4344 | indistinguable |
| W+1 | obligations | 0/5 | 0.5968 | indistinguable |
| W+2 | global | 0/5 | 0.8082 | indistinguable |
| W+2 | crypto | 1/5 | 0.1138 | indistinguable |
| W+2 | actions | 0/5 | 0.2980 | indistinguable |
| W+2 | obligations | 0/5 | 0.9538 | indistinguable |
| W+3 | global | 0/5 | 0.4314 | indistinguable |
| W+3 | crypto | **2/5** | **0.0378** | **weekly significativement meilleur** |
| W+3 | actions | 0/5 | 0.2204 | indistinguable |
| W+3 | obligations | 0/5 | 0.3268 | indistinguable |

Une seule case survit : **crypto à W+3** (p=0.0378, 2 graines sur 5 —
42 : p=0.0124 et 44 : p=0.0422, 46 à 0.0524 juste au-dessus du seuil). C'est
un signal, pas un verdict : **une case sur les douze** (4 groupes × 3
horizons) à p<0.05, sans correction pour tests multiples, portée par 2
graines sur 5.

**Le résultat de la note précédente ne se reproduit donc pas** : à la graine
42, W+1, global, on passe de p=0.0100 (50 tirages) à **p=0.0500** (200
tirages) — pile au seuil, du mauvais côté. Le skill Winkler du weekly
n'était pas robuste au budget d'échantillonnage.

### 3.2 Couverture — là, le signal tient

Deux familles de tests, volontairement redondantes.

**(a) Test de référence** — écart (indicateur − 0.95) bootstrappé **par
blocs** (`paired_test.paired_block_bootstrap_test`, block_length=3), sans
hypothèse d'indépendance des violations. Indicateur moyenné sur les 5
graines par origine.

**(b) Complément de manuel** — **Kupiec** (POF, couverture inconditionnelle)
et **Christoffersen** (indépendance + conditionnelle), calculés **par
graine**, χ². Leurs p-values sont **optimistes** (elles supposent des
violations i.i.d., or les cibles W+2/W+3 se chevauchent d'une origine à
l'autre) ; `LR_ind` est en outre structurellement biaisé vers le rejet à
W+2/W+3, ce chevauchement créant une dépendance **mécanique**. C'est déclaré
dans le module et répété ici.

W+1, graines moyennées (`p` = test (a) ; `Kupiec` = nombre de graines
rejetant à 5 %) :

| Actif | daily : écart / p / Kupiec | weekly : écart / p / Kupiec | tête-à-tête (weekly − daily) |
|---|---|---|---|
| BTC-USD | −0.017 / 0.434 / 0/5 | +0.017 / 0.333 / 0/5 | **weekly couvre plus** (p=0.009) |
| ETH-USD | **−0.052 / 0.024 / 2/5** | −0.003 / 0.864 / 0/5 | **weekly couvre plus** (p<0.001) |
| SPY | **−0.070 / 0.009 / 4/5** | −0.006 / 0.865 / 0/5 | indistinguable (p=0.052) |
| TLT | −0.017 / 0.498 / 0/5 | +0.014 / 0.448 / 0/5 | **weekly couvre plus** (p=0.006) |
| ZN=F | −0.028 / 0.229 / 1/5 | −0.017 / 0.468 / 0/5 | indistinguable (p=0.599) |

Aux trois horizons réunis : le **daily** est significativement mal couvert
sur **4 cellules/15** (ETH W+1/W+2/W+3, SPY W+1), **toujours par
sous-couverture** ; le **weekly** sur **1 cellule/15** (TLT W+3, et c'est une
**sur**-couverture, +0.037). Kupiec par graine raconte la même chose en plus
marqué (ETH daily : 2/5 puis 4/5 puis **5/5** graines rejettent quand
l'horizon s'allonge ; weekly : 0/5 partout sauf TLT W+3). Le tête-à-tête
apparié donne « weekly couvre significativement plus » sur **8 cellules/15**,
et jamais l'inverse.

**Verdict tâche 1 :** le weekly est **mieux calibré en couverture** — établi,
au sens poolé multi-graines, par deux tests indépendants, aux trois horizons.
Il **n'est pas mieux calibré au sens du Winkler** — l'avantage composite
était un artefact de `n_samples=50`. La différence entre les deux énoncés est
la **largeur** : le weekly achète sa couverture en élargissant (PI plus large
sur 15 cellules/15), ce que le Winkler facture.

---

## 4. Tâche 3 — TOST : « indistinguable » devient « équivalent »

Marge **±5 % de RMSE relatif**, fixée **a priori**, identique pour les 5
actifs et les 3 horizons. Justification déclarée : (i) c'est l'ordre de
grandeur de la variabilité **inter-graines** déjà mesurée sur ce modèle
(CV du RMSE 0,8 %–7,3 %) — une différence de régime plus petite que le bruit
de graine n'est pas exploitable en production ; (ii) elle est plus stricte
que les écarts observés sur les cellules déjà déclarées indistinguables ;
(iii) aucune marge n'a été choisie après coup.

| Actif | W+1 (ratio, p_TOST) | W+2 | W+3 |
|---|---|---|---|
| BTC-USD | 1.016 — **équivalent** (0.012) | 0.995 — **équivalent** (0.000) | 0.947 — **différence établie** (daily meilleur) |
| ETH-USD | 0.992 — **équivalent** (0.020) | 0.972 — non conclusif (0.198) | 0.955 — non conclusif (0.427) |
| SPY | 1.024 — non conclusif (0.098) | 1.015 — **équivalent** (0.047) | 1.017 — non conclusif (0.074) |
| TLT | 1.002 — **équivalent** (0.000) | 1.006 — **équivalent** (0.000) | 0.998 — **équivalent** (0.000) |
| ZN=F | 1.009 — **équivalent** (0.003) | 1.013 — **équivalent** (0.002) | 1.002 — **équivalent** (0.000) |

**4 actifs/5 à W+1 et W+2** : l'interchangeabilité daily/weekly sur la
précision est **positivement établie**, pas seulement non réfutée. Les cases
« non conclusif » sont un vrai résultat : ni différence ni équivalence — à
`effective_n`=30, il ne faut pas les lire comme « interchangeables ».

Une case va dans l'autre sens : **BTC-USD à W+3**, où le **daily** est
significativement meilleur (ratio 0.947, soit −5,3 % de RMSE) — l'exact
opposé du « weekly meilleur sur BTC » de la version single-seed.

**Réserve à déclarer** : le TOST poolé conclut à l'équivalence là où le TOST
graine-par-graine ne le fait que sur 0 à 4 graines selon la case. Ce n'est
pas une contradiction — moyenner sur les graines supprime la variance
Monte-Carlo, et c'est bien le **ratio de RMSE attendu** qu'on veut borner,
pas celui d'un tirage particulier. Mais le résultat porte sur la performance
attendue, pas sur la garantie d'un run individuel.

---

## 5. Tâche 5 — recalibration conformale du daily : corrigeable, mais pas comme ça

Brique existante (`tsdiff_recalibrate.py`) importée telle quelle : split
conformal, score de non-conformité côté par côté, k par (actif, graine,
horizon, régime), **jamais** de facteur global. Découpe 50/50 par
`cutoff_date` unique, calibration strictement avant le test.
**`effective_n` ≈ 15 sur le bloc de test** — deux fois moins puissant que le
corps de l'étude ; tout « indistinguable » y est encore moins informatif.

W+1, bloc de test :

| Actif | cov daily avant → après | largeur daily avant → après | k daily (médiane graines) |
|---|---|---|---|
| BTC-USD | 0.911 → 0.920 | 19 860 → 21 110 | 1.043 |
| ETH-USD | 0.916 → 0.960 | 940 → 1 351 | 1.462 |
| SPY | 0.929 → **0.996** | 43.84 → **88.34** | 1.915 |
| TLT | 0.947 → **1.000** | 4.504 → 6.471 | 1.498 |
| ZN=F | 0.947 → **0.996** | 2.312 → 3.545 | 1.554 |

**Réponse : oui, l'écart de couverture se comble — et il se comble beaucoup
trop.** Le conformal par split à `n_calib`=45 est franchement conservateur
(le quantile fini-échantillon retenu est la ~44ᵉ statistique d'ordre sur 45,
soit le 97,8ᵉ percentile des scores) : la couverture monte à 0,99–1,00 et la
largeur de SPY **double**. Ce n'est plus une correction, c'est une
sur-correction.

Le re-test demandé par le brief, sur le bloc de test, skill Winkler global :

| Scénario | verdict | p |
|---|---|---|
| avant recalibration | indistinguable | 0.719 |
| **A** — daily recalibré seul (la question du brief) | **weekly significativement meilleur** | **p = 0** |
| **B** — daily **et** weekly recalibrés (contrôle d'équité) | indistinguable | 0.114 |

Le scénario A est **trompeur si on le lit seul** : le weekly n'y gagne pas
parce qu'il est meilleur, mais parce qu'on a infligé au daily une couche de
sur-élargissement qu'on n'a pas infligée au weekly — et le Winkler facture la
largeur. Le scénario B, symétrique, ramène à « indistinguable » (p=0.114,
et même direction qu'avant recalibration). C'est pour ça qu'il a été ajouté.

**Conclusion pratique** : la sous-couverture du daily *est* corrigeable, mais
le split conformal à ce volume de calibration n'est pas le bon outil — il
coûte plus en largeur qu'il ne rapporte en couverture. La tâche 6 fait mieux,
gratuitement.

---

## 6. Tâche 6 — ensemble multi-graines : le meilleur rapport qualité/prix du lot

Les 5 nuages de 200 tirages sont **concaténés** en un nuage de 1000 (mélange
des 5 prédictives — élargit quand les graines sont en désaccord, ce qui est
l'effet recherché ; moyenner les bornes l'aurait au contraire lissé). Point
et bandes lus dessus avec **la même formule** que pour un run simple.
Référence de comparaison : la performance **attendue d'un run à graine
unique** (métrique moyennée sur les 5 graines, origine par origine) — pas la
meilleure graine, pas la pire. **Coût : aucun refit.**

W+1 :

| Actif | régime | RMSE 1 graine → ens. | Cov95 1 graine → ens. | Winkler 1 graine → ens. | test Winkler |
|---|---|---|---|---|---|
| BTC-USD | daily | 5338 → 5150 | 0.933 → 0.956 | 30 540 → 28 750 | indistinguable |
| BTC-USD | weekly | 5255 → 5221 | 0.967 → 0.967 | 30 430 → 30 360 | indistinguable |
| ETH-USD | daily | 271.9 → 266.4 | 0.898 → 0.911 | 1557 → 1432 | **ensemble meilleur** |
| ETH-USD | weekly | 274.0 → 272.5 | 0.947 → 0.956 | 1366 → 1357 | indistinguable |
| SPY | daily | 13.26 → 13.03 | 0.880 → 0.900 | 81.18 → 73.73 | **ensemble meilleur** |
| SPY | weekly | 12.95 → 12.89 | 0.944 → 0.956 | 70.71 → 68.16 | **ensemble meilleur** |
| TLT | daily | 1.405 → 1.399 | 0.933 → 0.944 | 7.80 → 7.47 | **ensemble meilleur** |
| TLT | weekly | 1.402 → 1.395 | 0.964 → 0.967 | 8.02 → 8.04 | indistinguable |
| ZN=F | daily | 0.779 → 0.775 | 0.922 → 0.933 | 4.150 → 3.981 | **ensemble meilleur** |
| ZN=F | weekly | 0.773 → 0.769 | 0.933 → 0.933 | 4.323 → 4.210 | **ensemble meilleur** |

Sur les 3 horizons : le Winkler s'améliore significativement sur **7 cellules
daily / 15** contre **2 cellules weekly / 15** — exactement ce que le brief
anticipait (« en particulier sur le daily dont la dispersion est la plus
forte »). La couverture monte des deux côtés, davantage côté daily. Le RMSE
s'améliore partout, de façon modeste et non testée individuellement.

**Et le verdict daily-vs-weekly rejoué sur l'ensemble** : skill Winkler
**indistinguable** aux trois horizons (p=0.865 / 0.325 / 0.715) ; skill RMSE
indistinguable à W+1/W+2, et à **W+3 le daily devient significativement
meilleur** (p=0.0216). L'ensemble mange une bonne part de ce qui restait de
l'écart daily/weekly — et il ne coûte rien.

---

## 7. Tâche 7 — NsDiff contre le modèle volatilité (ARIMA-GARCH) à W+2/W+3

Appariement par (actif, horizon, `target_date`), **à l'intérieur d'un même
régime**. **Asymétrie de protocole à citer avec tout chiffre de cette
section** : ARIMA-GARCH est refit à **chaque** origine sur fenêtre glissante
et est déterministe (pas de graine) ; NsDiff est *train-once-forward* et
dépend d'une graine. L'écart mesuré mélange donc « modèle » et « protocole
d'entraînement » — la réserve que porte déjà tout classement inter-modèles de
ce repo.

Poolé tous actifs (skill vs marche aléatoire, dédoublonnage taux inclus) :

| Horizon | Régime | Skill RMSE | Skill Winkler |
|---|---|---|---|
| W+1 | weekly | **GARCH meilleur** (p=0.0298) | indistinguable (0.116) |
| W+1 | daily | **GARCH meilleur** (p = 0, cf. §note sur p) | **GARCH meilleur** (p=0.0194) |
| **W+2** | weekly | indistinguable (0.0660) | indistinguable (0.365) |
| **W+2** | daily | **GARCH meilleur** (p=0.0080) | indistinguable (0.169) |
| **W+3** | weekly | **GARCH meilleur** (p=0.0300) | indistinguable (1.000) |
| **W+3** | daily | indistinguable (0.233) | **GARCH meilleur** (p=0.0484) |

Par cellule à W+2/W+3 (20 cellules = 5 actifs × 2 horizons × 2 régimes) :
**NsDiff ne gagne aucune cellule**, ni sur le RMSE ni sur le Winkler, dans
aucun des deux régimes. GARCH en gagne **8 sur le RMSE** et **3 sur le
Winkler** ; le reste est indistinguable. Côté couverture, GARCH
est plus proche de la cible que NsDiff-daily sur presque toutes les cellules
(ex. W+2 SPY : 0.989 vs 0.927 ; W+3 ETH : 0.933 vs 0.816).

**Réponse à la question du programme : non, NsDiff ne referme pas l'écart vs
le modèle volatilité à W+2/W+3.** Il ne s'y effondre pas non plus (aucune
cellule où GARCH gagne sur les deux axes à la fois hors ETH-daily W+3), mais
il n'y a **aucun test poolé où NsDiff l'emporte**, à aucun horizon, dans
aucun régime. C'est cohérent avec le duel CRPS
(`NOTE_duel_nsdiff.md` : « ns 1/5 » contre ARIMA-GARCH aux trois horizons,
rejet SPA vs GARCH(1,1) toujours à 0/75) — la conclusion la plus stable du
programme reste inchangée.

---

## 8. Non-négociables — statut

- **Protocole multi-graines conservé** : toute conclusion de cette note
  repose sur les 5 graines (42-46), soit en poolé, soit en comptage
  graine-par-graine. Aucune conclusion n'est tirée d'une graine unique — et
  §1/§3.1 montrent une fois de plus qu'on aurait conclu à tort en le faisant.
- **Briques réutilisées, pas réimplémentées** : bootstrap par blocs
  (`paired_test`), Winkler + skill-score RW + pooling par classe
  (`dashboard_d7_w1`), test RMSE par cellule (`matrice_paired_tests`),
  recalibration conformale (`tsdiff_recalibrate`), fit/forecast NsDiff
  (`oos_nsdiff_daily_weekly`) — tous **importés et appelés**. Seuls Kupiec,
  Christoffersen et le TOST sont du code neuf (rien d'équivalent n'existait) :
  ils vivent dans `calibration_tests.py` et sont couverts par 19 tests
  unitaires (cas calculés à la main, entrées dégénérées, invariants).
- **Trois modifications de fichiers partagés, toutes additives et
  opt-in**, comportement par défaut strictement inchangé :
  `generate_nsdiff_asset(..., collect_samples=False)`,
  `build_enriched_pairs(..., horizon_unit="W+1")`,
  `paired_block_bootstrap_test(..., return_boot_means=False)`. Le dernier
  existe parce que le `p_value` bilatéral renvoyé est plafonné à 1.0 et ne
  peut donc pas être redivisé en p unilatérale fiable pour le TOST.
- **`tracking.db` jamais écrit** : lecture seule (origines + lignes
  ARIMA-GARCH). La piste `oos`/dashboard reste single-seed 42 et
  `n_samples=50`, comparable aux 6 autres modèles — **rien n'y a été
  touché**. Les chiffres de cette note ne sont donc **pas** ceux du
  dashboard, par construction (§1).
- **pytest vert** : `python -m pytest experiments validation -q` →
  **351 passed, 1 skipped**. Le skip est pré-existant et sans rapport
  (`test_crps_metrics.py:61 — properscoring not installed`). La note
  précédente rapportait 333 passed avant ce chantier ; +19 tests neufs ici.
- **Dépendance ajoutée** : `openpyxl` (absent du venv, requis par
  `experiments/offline_prices.py` pour la source de prix TLT). Il n'est pas
  dans `requirements.txt` — à ajouter, hors périmètre de ce brief.

## 9. Limites déclarées

- **Puissance** : `effective_n`=30 (90 origines, blocs de 3) pour §1-§4 et
  §7 ; **≈15** pour §5 (bloc de test seulement). Aucun « indistinguable » de
  cette note ne doit être lu comme une absence d'effet démontrée — sauf là où
  le TOST conclut explicitement à l'équivalence (§4), qui est précisément
  l'énoncé que le reste ne permet pas.
- **Tests multiples non corrigés** : les tableaux comptent beaucoup de
  cellules (5 actifs × 3 horizons × 2 régimes). Aucune correction de type
  Holm n'est appliquée — les cases isolées à p≈0.04 (crypto/Winkler à W+3,
  §3.1) doivent être lues comme telles.
- **Kupiec/Christoffersen** : p-values χ² optimistes (violations supposées
  i.i.d., cibles W+2/W+3 chevauchantes) ; `LR_ind` mécaniquement biaisé vers
  le rejet à ces horizons. Rapportés en complément, jamais comme verdict
  principal — le test de référence reste le bootstrap par blocs.
- **§7** : asymétrie de protocole (refit par origine vs train-once-forward),
  déclarée plus haut ; GARCH n'ayant pas de graine, la question posée est
  « un run NsDiff à graine tirée au hasard bat-il GARCH ? », pas « le
  meilleur NsDiff bat-il GARCH ? ».
- **§1, mécanisme** : l'explication du biais de quantile à 50 tirages est une
  hypothèse cohérente avec les chiffres, **non testée** (il faudrait balayer
  `n_samples`) ; l'effet, lui, est mesuré.
- **Non fait, volontairement** : aucun réentraînement, aucun rejeu du
  dashboard `oos`, aucune reprise des 6 autres modèles à 200 tirages (leur
  comparabilité mutuelle en dépend). Si le dashboard doit un jour refléter
  §1, c'est un chantier à part — il faudrait régénérer **tous** les modèles
  au même budget.
