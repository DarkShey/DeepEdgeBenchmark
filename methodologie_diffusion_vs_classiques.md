# Méthodologie — Les modèles de diffusion battent-ils les modèles classiques en prévision financière ?

*Document de cadrage — dernière mise à jour : 2026-07-27*

---

## 1. Le problème (formulation en une phrase)

**Hypothèse de départ :** les modèles de diffusion (TSDiff) devraient surpasser les modèles classiques (ARIMA-GARCH, SARIMA, Naive) et ML (LSTM, Prophet) en prévision de séries financières.

**Constat actuel :** sur 23 098 prédictions évaluées, ce n'est pas le cas. TSDiff perd sur MAE et sur CRPS au daily, et s'effondre au weekly. Son seul avantage net est la **calibration daily**.

**La vraie question à trancher :** ce résultat est-il
- **(A) un vrai résultat** — la diffusion ne convient pas (ou pas mieux) à la finance, pour des raisons de fond, ou
- **(B) un artefact** — notre protocole d'entraînement/évaluation désavantage la diffusion et fausse la comparaison ?

Tant qu'on n'a pas éliminé (B), on ne peut rien conclure sur (A). C'est ça le cœur de ce que le tuteur demande.

---

## 2. Décomposition du problème en sous-problèmes isolables

On ne peut pas répondre en bloc. On isole 5 sources possibles d'écart, de la plus "artefact" à la plus "de fond" :

| # | Sous-problème | Nature | Statut actuel |
|---|---------------|--------|---------------|
| P1 | **Protocole weekly** : le TSDiff weekly n'est pas natif, c'est le daily poussé en multi-pas → bruit mal propagé → intervalles trop étroits | Artefact confirmé | TSDiff-W natif codé et en prod depuis 2026-07-21, **pas encore évalué** |
| P2 | **Équité de la comparaison CRPS** : les baselines échantillonnent depuis une PI gaussienne stockée (biais en leur faveur) ; réglages du sampler TSDiff (nb d'échantillons, pas de débruitage) potentiellement sous-optimaux | Artefact possible | Biais identifié, non corrigé |
| P3 | **Puissance statistique** : conclure sur du live weekly = poignée de points sur des mois → insuffisant | Méthodo | À corriger (voir P1 bis) |
| P4 | **Conformité à la littérature** : notre CRPS et notre split train/val/test sont-ils calculés comme dans le papier TSDiff ? | Méthodo | **Non vérifié** (revue littérature pas faite) |
| P5 | **Cause de fond** : la finance est un cas à faible signal/bruit, rendements quasi-martingale → il se peut que la diffusion n'ait structurellement rien de plus à capturer qu'un GARCH | Vrai résultat possible | À investiguer une fois P1–P4 écartés |

**Principe directeur :** on ne conclut sur P5 (le vrai résultat) qu'**après** avoir neutralisé P1–P4. Tant qu'un artefact subsiste, "la diffusion perd" n'est pas défendable.

---

## 3. Ce qu'on sait déjà (résultats à ne pas rejouer)

- **Daily, précision :** TSDiff MAE 460 vs SARIMA 407 / ARIMA 413 → **perd**. Direction 54,8 % (meilleur de tous, mais à peine au-dessus du hasard — **à tester en significativité**).
- **Daily, CRPS :** ARIMA/SARIMA/Naive ~1,65–1,87 vs TSDiff 2,55 → **perd**, même sur le terrain distributionnel censé lui être favorable, et **malgré un biais CRPS en faveur des baselines**.
- **Daily, calibration :** TSDiff = seul modèle non significativement mal calibré à 50 % (gap 1,3 %, p=0,49), mieux calibré à 95 % (97,7 %) → **seul vrai point fort**.
- **Weekly :** calibration effondrée (16 % observé pour cible 50 %) → **artefact P1 documenté**, à ne pas interpréter comme un échec de la diffusion.

**Point qui coince (à assumer devant le tuteur) :** le daily est déjà une comparaison native et propre, et la diffusion y perd sur la précision. Aucune version TSDiff-W ne corrigera ça. Donc l'excuse "protocole" ne vaut que pour le weekly, pas pour le daily.

---

## 4. Plan de résolution — actions, livrables, critères

Ordre = du plus déterminant au moins déterminant.

### Étape 1 — Backtest rolling-origin de TSDiff-W natif (priorité haute)
- **But :** trancher P1 avec de la puissance statistique, sans attendre des mois de live.
- **Action :** backtester TSDiff-W sur l'historique en rolling-origin → des centaines de points d'évaluation immédiatement.
- **Livrable :** CRPS + calibration weekly recalculés sur ce backtest.
- **Critère :** si la calibration weekly reste mauvaise sur backtest → P1 n'était pas la cause, c'est plus profond. Si elle se corrige → l'artefact weekly est confirmé et levé.
- *(Le script de suivi live reste utile, mais en complément — pas comme plan principal.)*

### Étape 2 — Comparaison CRPS équitable
- **But :** neutraliser P2.
- **Action :** (a) rendre tous les modèles sample-based (ou faire produire à TSDiff une PI paramétrique ajustée), (b) vérifier/optimiser le sampler TSDiff (nb d'échantillons, nb de pas de débruitage).
- **Livrable :** tableau CRPS/MASE recalculé à armes égales.
- **Critère :** TSDiff bat-il les baselines une fois le biais retiré et le sampler réglé ?

### Étape 3 — Revue de littérature + alignement méthodo (ce que le tuteur demande explicitement, point 2)
- **But :** neutraliser P4 et cadrer P5.
- **Action :** lire le papier TSDiff (Kollovieh et al., *Predict, Refine, Synthesize*) + littérature diffusion hors finance. Vérifier : (a) sur quels datasets/horizons la diffusion gagne (electricity, traffic, solar = forte structure), (b) comment ils calculent le CRPS (empirique depuis échantillons vs pinball), (c) leur split train/val/test (rolling-origin ? scoring rules ?).
- **Livrable :** note comparant leur méthodo à la nôtre + liste des écarts à corriger.
- **Critère :** notre protocole est-il conforme ? Le prior "la diffusion gagne" vient-il de domaines qui ne se transfèrent pas à la finance ?

### Étape 4 — Voir avec Kyrio ce qui est fait côté modèles classiques (point 1 du tuteur)
- **But :** s'assurer que les baselines sont au meilleur de leur forme (sinon la comparaison est faussée dans l'autre sens).
- **Action :** récupérer le détail de l'implémentation ML/classiques de Kyrio.
- **Livrable :** synthèse d'une page.

### Étape 5 — Conclusion différenciée (une fois P1–P4 levés)
- **But :** répondre à P5 honnêtement.
- **Action :** séparer explicitement deux verdicts :
  - **Précision point** : la diffusion perd — potentiellement structurel en finance (faible SNR).
  - **Qualité des intervalles / calibration** : la diffusion gagne au daily → *c'est* le finding défendable et publiable.
- **Critère :** on peut dire au tuteur "voici ce que la diffusion apporte réellement, et voici ce qu'elle n'apporte pas, avec le protocole validé."

---

## 5. Réponse type "où en es-tu ?" (à ressortir en réunion)

> « L'hypothèse "diffusion > classiques" ne se vérifie pas dans les résultats actuels, mais avant de conclure je nettoie le protocole, parce qu'une partie du désavantage vient de nous, pas du modèle.
> Concrètement : (1) le weekly testé jusqu'ici était un artefact — daily poussé en multi-pas ; la version native est en prod et je la backteste en rolling-origin pour avoir de la puissance statistique tout de suite. (2) Je corrige le biais de la comparaison CRPS, qui favorisait les baselines. (3) Je fais la revue littérature pour aligner notre méthodo CRPS/split sur le papier TSDiff et voir sur quels types de données la diffusion gagne vraiment.
> Ce qui est déjà solide et ne bougera pas : au daily, la diffusion perd sur la précision mais est **le seul modèle bien calibré** — donc son apport réel est la qualité des intervalles, pas la moyenne. Le vrai sujet n'est pas "diffusion perd", c'est "diffusion apporte de la calibration, pas de la précision — et il reste à voir si un protocole propre change ce verdict au weekly." »

---

## 6. Risque méthodologique à ne pas commettre

Ne pas attendre TSDiff-W en live pour "espérer" qu'il valide l'hypothèse : c'est du raisonnement motivé (on attend la version qu'on veut voir gagner). Le backtest rolling-origin donne la réponse maintenant, sur données historiques, avec puissance statistique. Le live ne sert qu'à confirmer.
