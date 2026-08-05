# NOTE — TSDiff vs NsDiff à l'horizon mensuel (M+1/M+2/M+3), même régime

*2026-08-05. Extension de `NOTE_compare_daily_vs_monthly_nsdiff.md` — item
"optionnel/next" de `BRIEF_nsdiff_mensuel_M1M2M3.md` §4 ("ajouter TSDiff
mensuel pour la comparaison diffusion-vs-diffusion"), demandée directement
pour compléter le Run mensuel (NsDiff + TSDiff). Source :
`experiments/oos_tsdiff_monthly.py` (génération + insertion oos, seed=42) +
`experiments/tsdiff_monthly_multiseed.py` (graines 42-46, artefact isolé) +
`experiments/compare_tsdiff_nsdiff_monthly.py` (comparaison modèle-vs-modèle,
seed=42) + `experiments/compare_tsdiff_nsdiff_monthly_multiseed.py`
(stabilité inter-graines de cette comparaison, post-hoc depuis les
checkpoints déjà calculés, aucun réentraînement).*

---

## 0. Cadrage — ce que cette note compare, et sa limite délibérée

Question **orthogonale** à `NOTE_compare_daily_vs_monthly_nsdiff.md`
(daily-vs-mensuel, un seul modèle) : ici, **TSDiff vs NsDiff, à régime
identique** (régime B daily-poussé entre eux, régime C mensuel-natif entre
eux), même origines, mêmes cibles M+1/M+2/M+3.

**Limite assumée, pas contournée** : contrairement à
`NOTE_compare_weekly_tsdiff_nsdiff.md` (CRPS/PIT complets côté weekly), ici
**pas de CRPS ni de PIT** — ces métriques ont besoin du nuage d'échantillons
complet par prévision (`keep_samples=True`), que ni `oos_nsdiff_monthly.py`
ni `oos_tsdiff_monthly.py` ne stockent (seulement point + quantiles
2.5/97.5, même limite déjà documentée pour le multi-graines weekly). Cette
note compare donc sur **RMSE (testé), Cov95, Winkler, direction** —
descriptif/testé comme dans le reste de ce corpus, pas de CRPS/PIT.

Même avertissement puissance que `NOTE_compare_daily_vs_monthly_nsdiff.md`
§0 : `n_test=40`, `effective_n=13` par cellule, 17 sur l'agrégat global
poolé M+1.

**Budgets** : identiques entre les deux modèles où c'est comparable
(`n_test=40`, `n_val=6`, `seq_len_monthly=18`, `n_samples=50`, mêmes
origines) ; `epochs=40` pour les deux (défaut propre à chaque modèle,
`tsdiff_model.EPOCHS`/`weekly_nsdiff_production.NSDIFF_EPOCHS_W` — même
valeur numérique, déclarée indépendamment pour chacun, pas copiée). TSDiff
utilise en plus `k_denoise=20` (pas d'entraînement) + `ddim_eta=1.0`
("nécessaire pour un vrai PI", commentaire de `tsdiff_model.py` lui-même) —
paramètres que NsDiff n'a pas (échantillonnage ancestral complet, pas de
DDIM, cf. `oos_nsdiff_monthly.py`).

**Coût mesuré** (à retenir pour la suite) : TSDiff est **beaucoup plus
lourd** que NsDiff sur ce run — régime B (daily) seul : 176-409s par
(actif, graine) contre <15s côté NsDiff (UNet + 1000 pas de diffusion à
l'entraînement vs les petits MLP de NsDiff). Run complet mesuré : seed
unique ≈ 30 min, 5 graines ≈ **152 minutes**.

---

## 1. Résultat principal — asymétrique entre les deux régimes

| Régime | Skill RMSE (vs RW, poolé global M+1) | Skill Winkler (vs RW, poolé global M+1) |
|---|---|---|
| **NsDiff** (rappel `NOTE_compare_daily_vs_monthly_nsdiff.md` §3) | daily significativement meilleur (p=0.0002) | daily significativement meilleur (p<0.0001) |
| **TSDiff** | daily significativement meilleur (p<0.0001), **et sur les 4 classes** (crypto/actions/obligations, pas seulement crypto comme NsDiff) | indistinguable (global/crypto/actions, p>0.35) ; régime C significativement meilleur sur obligations (p=0.0056) |

*(Le label brut produit par `dashboard_d7_w1.run_pooled_test`, réutilisé
tel quel, écrit littéralement "weekly_native_significantly_better" même
pour une comparaison mensuelle — texte générique de la fonction partagée,
pas un bug de cette note ; traduit ici en "régime C significativement
meilleur".)*

**TSDiff daily-vs-mensuel est un contraste ENCORE plus net que NsDiff
daily-vs-mensuel** sur le RMSE (4/4 classes significatives contre 2/4 pour
NsDiff) — cohérent avec §3 : TSDiff (plus gros modèle) souffre davantage du
manque de données mensuelles (34-70 fenêtres d'entraînement) que NsDiff.

---

## 2. TSDiff vs NsDiff, même régime — tableau M+1

RMSE testé par cellule (bootstrap par blocs, `effective_n=13`).

| Régime | Actif | RMSE TSDiff | RMSE NsDiff | Cov95 TSDiff | Cov95 NsDiff | Winkler TSDiff | Winkler NsDiff | Verdict | p |
|---|---|---|---|---|---|---|---|---|---|
| B (daily) | BTC-USD | 9281 | 9675 | **7.5%** | 95% | 228 327 | 42 213 | indistinguable | 0.2424 |
| B (daily) | ETH-USD | 493.7 | 513.4 | **72.5%** | 95% | 4740 | 2472 | indistinguable | 0.2402 |
| B (daily) | SPY | 19.89 | 20.55 | **30%** | 95% | 403.0 | 87.53 | indistinguable | 0.6244 |
| B (daily) | TLT | 3.344 | 3.323 | **22.5%** | 90% | 66.94 | 15.43 | indistinguable | 1.0000 |
| B (daily) | ZN=F | 1.843 | 2.005 | **25%** | 90% | 41.28 | 9.331 | **TSDiff significativement meilleur** | 0.0120 |
| C (mensuel) | BTC-USD | 28 566 | 12 156 | 100% | 100% | 154 194 | 117 763 | **NsDiff significativement meilleur** | <0.0001 |
| C (mensuel) | ETH-USD | 1132 | 755.2 | 100% | 100% | 10 184 | 8325 | **NsDiff significativement meilleur** | 0.0030 |
| C (mensuel) | SPY | 88.26 | 20.37 | 97.5% | 100% | 347.0 | 187.9 | **NsDiff significativement meilleur** | <0.0001 |
| C (mensuel) | TLT | 10.60 | 3.315 | 95% | 100% | 43.61 | 24.25 | **NsDiff significativement meilleur** | <0.0001 |
| C (mensuel) | ZN=F | 5.876 | 1.962 | 97.5% | 100% | 21.64 | 11.92 | **NsDiff significativement meilleur** | <0.0001 |

**Lecture immédiate, deux motifs opposés :**

- **Régime C (mensuel-natif) : NsDiff écrase TSDiff, 5/5 cellules
  significatives, RMSE 1.4× à 4.3× meilleur.** TSDiff est un modèle
  beaucoup plus gros (UNet) que NsDiff (petits MLP) — avec seulement 34-70
  fenêtres d'entraînement mensuelles, le plus gros modèle sur-apprend/
  sous-généralise nettement plus.
- **Régime B (daily) : ⚠️ probable COLLAPSE de TSDiff, pas une comparaison
  propre.** Cov95 de TSDiff s'effondre à 7.5-30% (BTC-USD/SPY/TLT/ZN=F,
  cible 95%) — exactement la signature déjà diagnostiquée dans
  `BRIEF_weekly_prediction_v2.md` §0 pour ce même modèle : "incertitude
  effondrée... intervalles ~40× trop étroits → Cov95 0.00-0.10", causée à
  l'époque par un **sur-entraînement (300 epochs)**. Ici, `epochs=40` était
  déjà utilisé (la valeur "safe" issue de ce précédent) — donc ce n'est
  **pas** une répétition de l'erreur connue par epoch count. Hypothèse la
  plus probable : le régime B mensuel force un horizon de génération
  beaucoup plus long (63-92 pas quotidiens pour viser 3 mois, contre 15 pas
  pour viser 3 semaines côté weekly) — un chemin diffusé aussi long peut
  réactiver le même effondrement des échantillons par un autre levier que
  le nombre d'epochs. **Non vérifié mécaniquement ici** (pas de nuage
  d'échantillons conservé, §0) — signalé comme hypothèse plausible, pas
  démontré.
  **Conséquence pour la lecture de ce tableau** : la cellule "ZN=F,
  régime B, TSDiff significativement meilleur" (p=0.012) et les 4 cellules
  "indistinguable" du régime B ne doivent **pas** être lues comme une
  comparaison de précision propre entre les deux modèles — le Winkler très
  dégradé de TSDiff (403 vs 87.5 sur SPY, malgré un RMSE comparable)
  confirme que l'écart vient de l'incertitude effondrée, pas du point.
  **À netttoyer avant toute conclusion** : rejouer le régime B avec
  `keep_samples=True` pour vérifier directement la dispersion des
  échantillons (comme le diagnostic weekly l'a fait), ou tester un
  `ddim_eta`/horizon différent — non fait ici, prochain chantier.

---

## 3. Robustesse multi-graines (non-négociable, comme partout dans ce corpus)

Reconstruit **post-hoc** depuis les checkpoints déjà calculés par les deux
scripts multiseed (`experiments/checkpoints_{ns,ts}diff_monthly_multiseed/`)
— aucun réentraînement, juste la même comparaison rejouée graine par graine.

| Régime | Actif | Stable sur les 5 graines ? |
|---|---|---|
| C (mensuel) | BTC-USD, ETH-USD, SPY, TLT, ZN=F | **OUI — 5/5, NsDiff gagne à chaque fois** |
| B (daily) | BTC-USD | OUI (indistinguable ×5) |
| B (daily) | SPY, ETH-USD, ZN=F, TLT | **NON** (verdict change de graine en graine) |

**Le résultat le plus robuste de toute cette comparaison — et probablement
de tout le corpus mensuel — est le régime C : NsDiff bat significativement
TSDiff sur les 5 actifs × 5 graines, 25/25, sans exception.** C'est
nettement plus stable que n'importe quel verdict de
`NOTE_compare_daily_vs_monthly_nsdiff.md` (où même le motif crypto le plus
net n'était que 4/5 et 3/5 sur les graines). Le régime B, à l'inverse, est
bruité graine à graine — cohérent avec des cellules déjà indistinguables à
seed=42 (proche de 0, donc sensible au bruit d'échantillonnage).

**Confirmation indépendante côté daily-vs-mensuel intra-TSDiff** (rappel,
`tsdiff_monthly_multiseed.json`) : TSDiff daily bat TSDiff mensuel-natif sur
**5/5 actifs × 5/5 graines** — encore plus stable que le motif équivalent
côté NsDiff (qui n'était stable que sur 3/5 actifs, cf. `NOTE_compare_
daily_vs_monthly_nsdiff.md` §5bis). Les deux angles (intra-modèle et
inter-modèle) pointent donc dans la même direction avec la même robustesse.

⚠️ **Cette stabilité 5/5×5/5 porte sur le RMSE (point), pas sur la
fiabilité de l'intervalle** — et le régime B (daily) de TSDiff est
précisément celui suspecté de collapse (§2). Un point moyen peut rester
correct même quand l'incertitude autour est effondrée ; la robustesse
multi-graines ici confirme donc que "TSDiff-daily bat TSDiff-mensuel en
précision de point", **pas** que TSDiff-daily est un régime fiable dans
l'absolu — nuance à garder pour ne pas sur-vendre ce résultat.

---

## 4. Non-négociables — statut

- [x] **Isolation `tracking.db`** : `model='TSDiff'`+`frequence='monthly'`/
      `daily`+`horizon_type='monthly'`, +1200 lignes (600+600), toutes les
      lignes préexistantes (daily/daily 6224, daily/weekly 9450,
      weekly/weekly 9450, NsDiff monthly 600+600) **intactes**, vérifié par
      comptage avant/après à chaque étape.
- [x] **Multi-graines jamais en DB** : les deux scripts multiseed
      n'appellent jamais `insert_oos_predictions` — artefacts JSON isolés
      uniquement.
- [x] **Puissance annoncée** : `n`/`effective_n` affichés partout (§0, §2).
- [x] **Aucun verdict présenté sans sa stabilité multi-graines** (§3),
      y compris pour la comparaison modèle-vs-modèle (pas seulement
      régime-vs-régime).
- [x] Aucun fichier source existant modifié — 4 fichiers nouveaux
      (`oos_tsdiff_monthly.py`, `tsdiff_monthly_multiseed.py`,
      `compare_tsdiff_nsdiff_monthly.py`,
      `compare_tsdiff_nsdiff_monthly_multiseed.py`).
- [x] **pytest vert avant ET après** : `experiments/` 145 passed,
      `validation/` 188 passed, `models/` 88 passed, aucune régression.

---

## 5. Verdict pour le tuteur

**Le régime compte plus que le modèle, et un seul résultat est vraiment
solide ici : en mensuel-natif (régime C), NsDiff bat TSDiff de façon nette
et parfaitement stable (25/25 graines×actifs)** — un modèle plus petit et
plus simple généralise mieux sur seulement 34-70 fenêtres d'entraînement
mensuelles qu'un modèle plus gros (UNet) qui, lui, sur-apprend visiblement.
En régime daily-poussé (B), les deux modèles sont globalement
indistinguables sur le point (RMSE), mais **TSDiff montre la signature d'un
collapse de diffusion déjà documenté dans ce repo** (`BRIEF_weekly_
prediction_v2.md` : incertitude effondrée, intervalles ~40× trop étroits,
Cov95 0.00-0.10 — même symptôme mesuré ici, Cov95 7.5-30% sur 4/5 actifs).
Historiquement causé par un sur-entraînement (300 epochs) ; ici `epochs=40`
était déjà la valeur "safe" du précédent, donc le déclencheur cette fois est
probablement l'horizon de génération très long qu'exige le régime B mensuel
(63-92 pas quotidiens), pas le nombre d'epochs — hypothèse posée, non
vérifiée mécaniquement (§2). **Les cellules régime B de TSDiff ne doivent
pas être lues comme une comparaison fiable en l'état** — à revérifier avec
le nuage d'échantillons complet avant toute conclusion dessus.

**Recommandation pratique, cohérente avec `NOTE_compare_daily_vs_monthly_
nsdiff.md`** : à horizon mensuel, pousser un modèle **daily** vers la cible
plutôt qu'entraîner nativement sur les rendements mensuels — et si le choix
se limite au régime mensuel-natif, **préférer NsDiff à TSDiff** : c'est le
seul verdict de cette note robuste sans réserve.
