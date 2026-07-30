# Weekly vs Daily — ce qui a été testé, comment, et ce qu'on peut honnêtement en dire

Ce document explique, étape par étape et sans code, le test **régime B (daily→hebdo)
vs régime C (hebdo natif)** relancé sur la base à jour, et surtout **ce qu'on a le
droit d'affirmer** au vu des résultats. Objectif : que ce ne soit pas une boîte noire,
et que la conclusion tienne devant une lecture critique.

Résultats bruts reproductibles : `experiments/weekly_vs_daily_pooled.json`
Script : `experiments/weekly_vs_daily_pooled.py`

---

## 0. La question, et pourquoi elle se pose

Le tuteur pense qu'**en hebdomadaire les modèles font mieux qu'en quotidien**, et veut
intégrer une pipeline weekly. Il faut trancher proprement, pas « se débrouiller pour
que ça passe ». Deux pièges à éviter d'emblée :

1. **Ne pas confondre deux questions.** « Le weekly bat-il le daily ? » (B vs C) n'est
   PAS « le weekly bat-il une marche aléatoire ? » (le head-to-head v2, qui avait
   échoué). Ici on teste uniquement B vs C.
2. **Ne pas confondre deux régimes weekly.** Il y en a deux, comparés à cibles
   identiques (W+1, W+2, W+3) :
   - **Régime B** — modèle entraîné en **quotidien**, puis les pas journaliers sont
     agrégés en cibles hebdomadaires.
   - **Régime C** — modèle entraîné **nativement en hebdomadaire** (série resamplée en
     fin de semaine).

La question du tuteur, formellement : **à cibles hebdo identiques, le régime C est-il
meilleur que le régime B ?**

---

## 1. Les données : le weekly est déjà en base

Contrairement à ce que laissait entendre la méthodo KPI de la veille (où le régime C
était quasi-vide), la base `validation/tracking.db` a depuis été **backfillée**. État
réel vérifié directement :

- **5 400 lignes weekly** (`horizon_type='weekly'`, `source='oos'`), soit
  **2 700 régime B + 2 700 régime C**.
- Couvertes pour **les 6 modèles × 5 actifs × 3 horizons** (W+1/W+2/W+3), toutes
  évaluées (`y_true` renseigné).
- **2 698 paires** B↔C appariables (même modèle, actif, horizon, date-cible).

Conséquence importante : « intégrer le weekly pour de vrai » au niveau **pipeline de
données** est déjà largement fait. Ce qui manquait, c'est un **verdict** appuyé sur
assez de puissance — le head-to-head v2 ne s'appuyait que sur TSDiff × 2 actifs ×
30 origines, soit ~10× moins de données que ce qui est disponible ici.

---

## 2. Les métriques calculées sur chaque ligne

À partir des 4 nombres stockés par prédiction (`y_pred`, `y_lower`, `y_upper`,
`y_true`) :

- **CRPS gaussien normalisé** — score qui note la distribution entière (précision +
  incertitude) : plus bas = meilleur. On récupère l'écart-type implicite de l'intervalle
  stocké (`sigma = (y_upper − y_lower) / (2 × 1,96)`) puis on applique la forme fermée
  gaussienne du CRPS (`honest_eval.metrics.crps_gaussian`, la même fonction que
  `pooled_analysis.py`). Normalisé par une **échelle d'actif** (voir §3) pour pouvoir
  agréger BTC (~1 300 $/jour de mouvement) et TLT (~0,31 $) sans que BTC écrase tout.
- **Couverture 95 %** (`in_interval`) — la vraie valeur tombe-t-elle dans l'intervalle
  à 95 % ? Moyennée, ça doit approcher 0,95 si le modèle est bien calibré.
- **MASE** — erreur ponctuelle sans échelle, secondaire, pour référence.

> **Limite honnête et centrale.** Le CRPS gaussien est **exact** pour les 5 modèles
> paramétriques (ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive) : leur intervalle EST
> construit comme `centre ± 1,96·σ`, on ne fait que retrouver le σ qu'ils utilisaient
> déjà. Pour **TSDiff**, qui génère un nuage non gaussien, c'est une **approximation**
> (on force son nuage dans une gaussienne de même intervalle). Pour un CRPS empirique
> exact sur TSDiff, il faudrait régénérer les nuages d'échantillons (coûteux :
> réentraînement). La **couverture**, elle, ne dépend d'aucune hypothèse de forme —
> c'est pourquoi, pour TSDiff, le résultat de calibration (§4) est plus solide que
> celui de CRPS.

---

## 3. La méthode statistique (réutilisée du dépôt, pas réinventée)

On reprend la machinerie **déjà validée** dans `experiments/pooled_analysis.py` /
`paired_test.py` — mêmes fonctions, mêmes garde-fous. Seule différence pratique : les
échelles d'actif sont dérivées **depuis la base** (moyenne des `|variation quotidienne
du dernier prix|`) au lieu d'être téléchargées depuis Yahoo, faute d'accès réseau dans
l'environnement — substitution assumée, définition identique.

Quatre garde-fous, chacun pour une raison précise :

1. **Appariement par (origine, horizon).** On teste des **différences** B−C ligne à
   ligne sur exactement la même origine et le même horizon → la comparaison contrôle
   l'échelle et le régime de marché. La clé d'appariement est composite
   (`cutoff_date | horizon_unit`) car la cible W+2 d'une origine tombe le même jour
   calendaire que la W+1 de l'origine suivante — une clé « date » nue les
   confondrait.
2. **Clustering par classe d'actif.** ZN=F et TLT (deux paris sur les taux) sont
   moyennés en une série « obligations » ; BTC-USD et ETH-USD en « crypto » ; SPY reste
   seul. Sinon on compterait deux fois la même information (actifs corrélés ≠ deux
   preuves indépendantes).
3. **Double test DM-HAC + bootstrap par blocs.** Les origines se **chevauchent** dans
   le temps → un test qui suppose l'indépendance sous-estime l'incertitude (c'est
   exactement ce qui gonflait les p-values d'un premier test rapide, avant correction).
   Le bootstrap par blocs (longueur 3) et le Diebold-Mariano à variance HAC ne
   supposent pas l'indépendance ; on rapporte le **n effectif** (~90, très en dessous
   des 270 lignes brutes) qui reflète la vraie puissance. Les deux tests doivent être
   concordants.
4. **Correction de Holm** sur les 6 modèles. On teste la même question 6 fois → on
   resserre le seuil pour ne pas déclarer « significatif » par pur hasard de
   multiplicité. **Toutes les p-values « significatives » citées ci-dessous sont
   post-Holm.**

---

## 4. Résultats

### 4.1 Précision distributionnelle — CRPS, régime B vs C

Différence `CRPS_B − CRPS_C` : **> 0 ⇒ le weekly (C) est meilleur** ; < 0 ⇒ le daily
(B) est meilleur. n = 270, n_eff = 90 par modèle.

| Modèle | diff (B−C) | IC 95 % | p brut | p (Holm) | Verdict |
|---|---|---|---|---|---|
| ARIMA-GARCH | +0,015 | [−0,033 ; +0,068] | 0,537 | 1,000 | égalité |
| SARIMA | −0,030 | [−0,052 ; −0,010] | 0,004 | **0,018** | **daily meilleur** |
| Prophet | −0,006 | [−0,211 ; +0,234] | 1,000 | 1,000 | égalité (très bruité) |
| LSTM † | −0,732 | [−1,221 ; −0,268] | 0,002 | **0,014** | **daily meilleur** |
| Naive | +0,005 | [−0,004 ; +0,014] | 0,282 | 0,847 | égalité |
| TSDiff | +0,289 | [−0,009 ; +0,629] | 0,059 | 0,236 | penche weekly, **non significatif** |

† LSTM re-testé après re-réglage équitable des hyperparamètres (SEQ_LEN par actif,
BRIEF_lstm_weekly_retune.md, cf. §5bis) — valeur avant re-réglage : −0,745. Le
re-réglage ne change ni le signe ni la significativité.

**Lecture.** Le weekly ne bat significativement le daily sur le CRPS pour **aucun
modèle**. TSDiff penche pour le weekly mais ne franchit pas le seuil (p=0,236 corrigé).
Deux modèles sont significativement **moins bons** en weekly : SARIMA (léger) et surtout
**LSTM** (énorme, −0,732, re-testé à réglage équitable — voir §5bis).

### 4.2 Calibration — couverture 95 %, régime B vs C

Différence de couverture `cov_C − cov_B` : **> 0 ⇒ le weekly est mieux calibré**.

| Modèle | Cov95 B (daily) | Cov95 C (weekly) | diff (C−B) | IC 95 % | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| ARIMA-GARCH | 0,953 | 0,940 | −0,015 | [−0,035 ; +0,002] | 0,328 | ns |
| SARIMA | 0,900 | 0,898 | −0,006 | [−0,020 ; +0,009] | 0,554 | ns |
| Prophet | 0,622 | 0,738 | +0,115 | [+0,080 ; +0,156] | **<0,001** | **weekly mieux calibré** |
| LSTM † | 0,911 | 0,923 | −0,011 | [−0,065 ; +0,041] | 1,000 | ns |
| Naive | 0,913 | 0,902 | −0,013 | [−0,032 ; +0,002] | 0,328 | ns |
| **TSDiff** | **0,553** | **0,784** | **+0,211** | **[+0,135 ; +0,285]** | **<0,001** | **weekly mieux calibré** |

† LSTM re-testé après re-réglage équitable (cf. §5bis) — C=0,889 avant re-réglage.

**Lecture.** C'est ici le résultat fort. **TSDiff en daily est pathologiquement
sur-confiant** (couverture 0,553 au lieu de 0,95 : ses intervalles sont beaucoup trop
étroits). L'entraînement **natif hebdomadaire corrige significativement** cette
sur-confiance — la couverture remonte à 0,784, soit **+21 points**, avec un IC qui ne
touche pas 0 et une p < 0,001 même après correction. Prophet bénéficie aussi du weekly
sur la calibration, mais partait de très bas (0,62) et reste mal calibré.

---

## 5. Verdict honnête — ce qu'on peut / ne peut pas dire au tuteur

**Ce qu'on NE peut PAS affirmer :**
- « Le weekly bat le daily. » Faux en général : sur le CRPS, aucun modèle ne gagne
  significativement, et deux perdent.
- « Pour TSDiff le weekly est plus précis. » Le CRPS penche dans ce sens mais **n'est
  pas significatif** (p=0,236). L'affirmer serait du sur-vente démontable.

**Ce qu'on PEUT affirmer, preuve à l'appui :**
- Pour **TSDiff**, l'entraînement natif hebdomadaire **corrige significativement la
  sur-confiance** du modèle : la couverture à 95 % passe de 0,55 à 0,78
  (diff +0,21, IC [+0,14 ; +0,29], p<0,001 après Holm), sur 6 modèles × 5 actifs et un
  test qui tient compte du chevauchement des origines et de la multiplicité.
- C'est un argument **légitime et ciblé** pour intégrer une pipeline weekly **pour
  TSDiff** — mais sur le terrain de la **calibration de l'incertitude**, pas de la
  précision globale. Formulation défendable : *« le weekly ne rend pas TSDiff plus
  précis, mais il rend son incertitude nettement plus fiable, ce qui compte autant pour
  un usage en risque/trading que le point lui-même. »*

**Proposition d'intégration.** Activer le régime weekly natif **spécifiquement pour
TSDiff** dans le dashboard/matrice, justifié par le gain de calibration. Ne pas le
généraliser aux 5 autres modèles sur la base de ces données (au mieux neutre,
significativement pire pour SARIMA/LSTM).

---

## 5bis. LSTM weekly re-réglé (BRIEF_lstm_weekly_retune.md) — verdict mis à jour

Le diagnostic préalable (avant re-réglage) avait établi qu'il n'y a **PAS de bug** de
données côté LSTM weekly (900 lignes complètes, 0 valeur manquante) — seulement des
hyperparamètres (`SEQ_LEN=30`, `EPOCHS`) copiés tels quels du daily, jamais adaptés à
une série ~5x plus courte en hebdo. Le brief a rendu la comparaison **équitable** :

- **Sélection** : `experiments/lstm_weekly_sweep.py`, grille `SEQ_LEN ∈ {8, 16, 26}`
  (EPOCHS fixé à la valeur daily), **régime C uniquement**, sur les 12 origines de
  validation (jamais le test), CRPS gaussien fermé (exact pour LSTM, son IC est
  construit `point ± 1,96·σ`), sélection par **règle 1-SE** (candidat le plus
  parcimonieux dans 1 erreur-type du minimum, pas l'argmin brut — n_val=12 est trop
  peu pour trancher finement). Résultats bruts :
  `experiments/lstm_weekly_sweep.json`.

  | Actif | SEQ_LEN* retenu | CRPS validation (C, retenu) | CRPS validation (B, défaut inchangé) |
  |---|---|---|---|
  | SPY | 16 | 6,49 ± 0,35 | 8,90 ± 0,85 |
  | BTC-USD | 16 | 5126 ± 678 | 6631 ± 1035 |
  | ETH-USD | 8 | 174,7 ± 17,9 | 414,4 ± 59,2 |
  | ZN=F | 8 | 0,535 ± 0,019 | **0,505 ± 0,018** (meilleur que C) |
  | TLT | 8 | 1,292 ± 0,054 | **1,182 ± 0,069** (meilleur que C) |

- **Asymétrie B/C assumée et vérifiée.** Seul le régime C est re-réglé (le brief ne
  demande pas de retoucher B). Le contrôle de parité ci-dessus (même règle de
  sélection — CRPS validation — appliquée aussi à B, sans agir dessus) montre que le
  défaut daily n'est **pas systématiquement** loin de son propre optimum : sur les
  obligations (ZN=F, TLT), le défaut B bat même la meilleure config C trouvée sur ce
  bloc de validation. Sur les actions/crypto (SPY, BTC, ETH), B est plus loin de son
  optimum que C ne l'est du sien, mais rien n'indique que B soit mal réglé au sens du
  brief — même traitement méthodologique par régime, conclusion différente selon
  l'actif, ce n'est pas un deux-poids-deux-mesures.

- **Re-génération + re-test.** LSTM régime C régénéré sur les 30 origines de test avec
  le `SEQ_LEN*` par actif ci-dessus (`experiments/weekly_multimodel.py --models LSTM
  --regimes C`), upserté (`experiments/backfill_multimodel_predictions.py`),
  `experiments/weekly_vs_daily_pooled.py` relancé.

  **Verdict CRPS re-testé** : diff(B−C) = **−0,7317** IC [−1,2206 ; −0,2678],
  p_Holm = **0,0144** — quasiment inchangé par rapport à l'ancien réglage
  (−0,745 avant re-réglage équitable). **Le signe et la significativité ne changent
  pas** : le LSTM weekly reste significativement moins bon que le daily en CRPS, même
  à hyperparamètres choisis équitablement sur validation. Ce n'est donc pas un artefact
  de réglage aveugle — c'est un vrai désavantage du LSTM en régime hebdo natif pour ce
  jeu de données, à documenter tel quel (aucun nouveau balayage tenté pour inverser ce
  signe, conformément au garde-fou du brief).

  **Couverture 95%** : B=0,911, C=0,923 (contre C=0,889 avant re-réglage) — légère
  amélioration, toujours **non significative** (diff(C−B), p_Holm=1,000, ns).

- **Conclusion pour l'intégration (BRIEF_weekly_pipeline_integration.md)** : le LSTM
  weekly n'est plus à traiter comme « suspect / possible bug » — c'est un résultat
  re-testé et confirmé. Il reste correctement **absent** de la génération récurrente
  weekly (seul TSDiff y est justifié) et s'affiche désormais dans le dashboard comme un
  résultat normal (non masqué), avec ce verdict documenté au lieu d'un badge « suspect ».

## 6. Limites et points à traiter

- **CRPS TSDiff = approximation gaussienne.** Le seul chiffre fragile est le CRPS de
  TSDiff (nuage non gaussien forcé en gaussienne). Le résultat de **calibration**, lui,
  ne souffre pas de cette limite (la couverture est indépendante de la forme). Pour un
  CRPS empirique exact sur TSDiff → régénérer les nuages d'échantillons (réentraînement).
- **LSTM weekly : re-réglage effectué, verdict confirmé (cf. §5bis).** Diagnostiqué :
  pas de bug de données, hyperparamètres non adaptés au régime hebdo. Re-réglé
  équitablement (SEQ_LEN par actif, sélection sur validation, régime C uniquement) —
  le désavantage CRPS persiste quasi identique (−0,745 → −0,7317, toujours significatif
  après Holm) : un vrai effet, pas un artefact. Le pool global reste tiré vers le bas
  par LSTM et Prophet, mais ce n'est plus une hypothèse de bug non tranchée.
- **Asymétrie de protocole non poolée.** Le classement *inter-modèles* reste biaisé
  (TSDiff réentraîné vs 5 autres ré-échantillonnés depuis leur loi) — on ne compare donc
  ici que **B vs C au sein d'un même modèle**, jamais un modèle contre un autre.
- **Échelles d'actif dérivées de la base** (pas de réseau Yahoo) : même définition que
  le dépôt, mais calculée sur les `last_close` en base plutôt que sur l'historique
  téléchargé.
- **Puissance.** n_eff ≈ 90 : bien mieux que les 30 origines du v2, mais les effets
  faibles (±0,01–0,03 de CRPS) restent hors de portée. Les deux effets significatifs de
  calibration (TSDiff +0,21, Prophet +0,11) sont assez gros pour tenir.

---

## 7. Où retrouver chaque chose

| Quoi | Où |
|---|---|
| Script du test (reproductible, sans réseau/torch) | `experiments/weekly_vs_daily_pooled.py` |
| Résultats bruts (CRPS + couverture, par modèle, Holm) | `experiments/weekly_vs_daily_pooled.json` |
| Machinerie stat réutilisée | `experiments/pooled_analysis.py`, `paired_test.py`, `honest_eval/metrics.py` |
| Données sources | `validation/tracking.db` (table `predictions`, `source='oos'`, `horizon_type='weekly'`) |
| Ce document | `experiments/METHODOLOGIE_weekly_vs_daily.md` |
