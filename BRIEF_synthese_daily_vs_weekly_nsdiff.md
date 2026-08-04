# BRIEF — Synthèse comparative NsDiff : Daily (régime B) vs Weekly natif (régime C)

*Créé le 2026-08-04. Suite de `BRIEF_nsdiff_dashboard_daily_oos.md` (NsDiff désormais présent des
deux côtés en `source='oos'`) et de `NOTE_compare_weekly_tsdiff_nsdiff.md`. Répond au 2ᵉ objectif
du tuteur : « Daily vs Weekly », appliqué à NsDiff.*

---

## 0. Objectif (une phrase)

Produire **une synthèse claire** répondant à : **pour prévoir NsDiff à 1 semaine, vaut-il mieux
le régime daily (modèle daily poussé à la cible-vendredi) ou le régime weekly natif ?** — par
métrique, par actif, par classe, avec verdict statistique et limites déclarées.

**Point de méthode central : la comparaison est DÉJÀ CALCULÉE.** Le dashboard
`dashboard_d7_w1.py` produit déjà, pour chaque modèle (NsDiff inclus depuis le câblage), les
métriques daily vs weekly appariées + les tests. **Cette synthèse EXTRAIT et INTERPRÈTE ces
chiffres — elle ne relance pas les modèles et ne réimplémente aucun test.**

---

## 1. Sources de données (à lire, pas à recalculer)

- **`experiments/dashboard_d7_w1_data.json`** (déjà généré) : par cellule (model, asset), les
  champs `rmse_daily`/`rmse_weekly`, `winkler_daily`/`winkler_weekly`,
  `cov95_daily`/`cov95_weekly`, `pi_width_daily`/`pi_width_weekly`,
  `direction_daily`/`direction_weekly`, plus `verdict`/`p_value`/`effective_n`/`n` du test par
  cellule, et l'`aggregate` poolé (skill-score par classe crypto/index/bond + global, avec
  verdict et significativité). **Filtrer `model == "NsDiff"`.**
- **`validation/tracking.db`** (`source='oos'`, `model='NsDiff'`) : source brute si un chiffre
  doit être re-vérifié ou si on étend à W+2/W+3 (voir §4, secondaire).
- **`NOTE_compare_weekly_tsdiff_nsdiff.md`** : détail calibration/sharpness/Winkler/PIT du côté
  weekly (pour recouper).
- **`NOTE_nsdiff_dashboard_daily_oos.md`** : origines réutilisées, seed, premier verdict
  daily-vs-weekly (point de départ à approfondir ici).

---

## 2. Contenu de la synthèse (livrable `NOTE_compare_daily_vs_weekly_nsdiff.md`)

### 2.1 Cadrage (à écrire noir sur blanc, pour éviter la confusion horizon × régime)
La comparaison est **régime daily (B) vs régime weekly-natif (C)** à **cible identique (W+1,
même target_date-vendredi)** — PAS « horizon 1 jour vs horizon 1 semaine ». Les deux côtés
prédisent la MÊME chose (le prix à 1 semaine), par deux chemins différents. Reprendre le
libellé de l'en-tête du dashboard.

### 2.2 Tableau récapitulatif par actif (les 5 actifs, ou 4 si TLT manque)
Pour NsDiff, par actif : RMSE (daily / weekly), Cov95 (daily / weekly, cible 0.95),
largeur PI (daily / weekly), Winkler (daily / weekly), direction (daily / weekly),
**verdict de cellule** (daily meilleur / weekly meilleur / indistinguable) avec `p_value`,
`n` et `effective_n`.

### 2.3 Agrégat par classe + global
Reprendre l'`aggregate` du dashboard pour NsDiff : skill-score daily-vs-weekly poolé par classe
(crypto / index / obligations, ZN=F+TLT dédoublonnés) et global, sur **RMSE** ET **Winkler**,
avec verdict et `n_origins`/`effective_n`. **Toujours afficher `effective_n`** (puissance faible
à assumer, pas à cacher — convention maison).

### 2.4 Lecture (le cœur de la synthèse)
Répondre explicitement, métrique par métrique :
- **Précision (RMSE)** : le weekly-natif aide-t-il, ou le daily poussé est-il aussi bon /
  meilleur ?
- **Calibration (Cov95) et finesse (largeur PI)** : le weekly-natif est-il mieux calibré (comme
  côté TSDiff-W vs TSDiff-D, où le natif corrigeait l'effondrement de couverture) ? Gagne-t-il
  la couverture sans juste élargir (regarder Winkler, pas Cov95 seul) ?
- **Où ça bascule par classe** : crypto (24/7, pas de vraie clôture) vs actions/obligations
  (vraie clôture quotidienne) — le régime daily a-t-il un sens différent selon la classe ?
- **Croisement avec le finding de fond** (`methodologie_diffusion_vs_classiques.md` §3-5) : la
  diffusion « perd en précision, gagne en calibration ». Le passage daily → weekly-natif de
  **NsDiff** suit-il le même motif, ou NsDiff se comporte-t-il différemment de TSDiff sur cet
  axe ?

### 2.5 Verdict pour le tuteur (2-3 phrases franches)
Une conclusion différenciée : *sur quelle métrique et quelle classe le weekly-natif de NsDiff
apporte quelque chose, et où il n'apporte rien / le daily suffit* — avec la significativité
(jamais un verdict sur une différence non significative ou `effective_n` trop faible).

---

## 3. Non-négociables
- [ ] **Ne relance aucun modèle, ne réimplémente aucun test** : extraction depuis
      `dashboard_d7_w1_data.json` (et `tracking.db` en lecture seule). Les p-values doivent
      **recouper** celles du dashboard (même seed, mêmes origines).
- [ ] **Aucune conclusion sur une cellule non significative** ou à `effective_n` trop faible —
      dire « indistinguable » et donner le `n`.
- [ ] **Limites déclarées** : passe **single-seed** (seed=42, pas multi-graines) ; scope
      **W+1** (voir §4) ; **TLT** absent le cas échéant → ne pas conclure sur la classe
      obligations si seul ZN=F survit.
- [ ] Aucun fichier source modifié hors ajouts (note + éventuel petit script d'extraction) ;
      pytest vert si un script est ajouté.

## 4. Extension optionnelle (si le temps, secondaire)
Le dashboard est scopé **W+1**. Si les lignes `oos` de NsDiff couvrent aussi **W+2/W+3**
(vérifier dans `tracking.db`), ajouter un tableau daily-vs-weekly aux horizons plus longs, en
réutilisant `prob_kpi_common.row_kpis` sur ces lignes — utile car l'écart daily/weekly se creuse
souvent avec l'horizon. **Ne pas bloquer la synthèse W+1 dessus.**

## 5. Livrables
1. `NOTE_compare_daily_vs_weekly_nsdiff.md` : cadrage + tableau par actif + agrégat par classe +
   lecture + verdict + limites.
2. (Optionnel) petit script d'extraction `experiments/extract_nsdiff_daily_vs_weekly.py` si ça
   clarifie / rend reproductible la lecture du JSON.

## 6. Pièges à éviter
- **Ne pas confondre** « daily vs weekly » (les deux régimes visant la même cible W+1) avec
  « horizon 1j vs horizon 1 semaine ».
- **Ne pas recalculer** ce que le dashboard a déjà produit — extraire, recouper, interpréter.
- **Ne pas surinterpréter** : puissance faible (crypto surtout), single-seed, W+1 — verdict
  différencié et prudent, jamais un « NsDiff est meilleur en weekly » global sans le détail.
- **Ne pas mélanger** cette comparaison **intra-NsDiff** (daily vs weekly) avec la comparaison
  **TSDiff vs NsDiff** (`NOTE_compare_weekly_tsdiff_nsdiff.md`) — deux questions distinctes.
