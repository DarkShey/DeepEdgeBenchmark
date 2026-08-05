# BRIEF — NsDiff à l'horizon mensuel (M+1 / M+2 / M+3)

*Créé le 2026-08-04. Étend la comparaison daily-vs-weekly de NsDiff
(`NOTE_compare_daily_vs_weekly_nsdiff.md`) à l'horizon **mensuel**. Objectif : produire NsDiff en
M+1/M+2/M+3 en deux régimes (daily poussé vers la fin de mois, et mensuel-natif), sur origines
glissantes, **multi-graines dès le départ**, et en tirer une synthèse daily-vs-mensuel.*

---

## 0. Avertissement à lire avant tout (puissance statistique)

L'historique disponible (~2015 → 2026, ~11 ans) fait **~130 fins de mois** — contre ~570 semaines
côté weekly. Après réservation train/validation, le nombre d'**origines de test mensuelles sera
faible (quelques dizaines au mieux)**, et les cibles M+1/M+2/M+3 se chevauchent → **`effective_n`
de l'ordre de ~10, pas 30**. Conséquence à **assumer et écrire noir sur blanc** : **la plupart des
verdicts seront « indistinguable »**, non par absence d'effet mais par manque de puissance. Le
mensuel sert à **compléter le tableau des horizons** (daily / weekly / mensuel), pas à produire des
verdicts tranchés. **Ne jamais** présenter un « indistinguable » mensuel comme une équivalence
démontrée.

---

## 1. Ce qu'on produit — deux régimes, cible fin de mois

Analogue exact du weekly, à l'échelle mensuelle. Cible = **dernier jour de bourse du mois**
(ancre à déclarer : `BM`/business month-end, ou `resample("M").last().dropna()` — **jamais** un
mois partiel en cours, point-in-time).

- **Régime B (daily → fin de mois)** : NsDiff **daily**, prévision lue à la **distance en jours de
  bourse réelle** jusqu'à la fin de mois-cible (nouvelle fonction `month_targets`, miroir de
  `epoch_sweep.week_targets`). `frequence='daily'`, `horizon_type='monthly'`.
- **Régime C (mensuel natif)** : NsDiff **fit sur les rendements mensuels** (mêmes fonctions
  `nsdiff_model.fit_nsdiff`/`forecast_from_fitted`, qui sont agnostiques à la fréquence — on ne
  change que la série d'entrée), horizon=3 mois, génère M+1/M+2/M+3 **en un seul tir**.
  `frequence='monthly'`, `horizon_type='monthly'`.

Les deux visent **exactement la même cible** (même fin de mois) — c'est la comparaison de régime à
cible fixe, comme pour le weekly.

---

## 2. Origines & données — à générer (pas de triplets préexistants)

Contrairement au weekly, **aucun modèle n'a de lignes `oos` mensuelles** dans `tracking.db` → pas
d'alignement gratuit. Donc :

- **Générer les origines mensuelles** par un `three_way_split` sur la série mensuelle (train /
  validation / test), documenté. Viser le **max d'origines de test possible** tout en gardant un
  train décent (cf. §0 : ce sera peu).
- **Point-in-time strict** : `mu`/`sd` gelés au train ; fenêtre de conditionnement qui grandit mais
  ne dépasse jamais l'origine ; resample fin-de-mois avec `dropna` (jamais le mois courant partiel).
- **Baseline RW pour le skill-score** : réutiliser `dashboard_d7_w1.rw_pi_bounds` /
  `historical_h_day_returns` (agnostiques à la fréquence, paramétrés en jours calendaires `h_days`)
  — fonctionne tel quel pour des cibles mensuelles, **aucune réimplémentation**.

---

## 3. Multi-graines dès le départ (leçon du weekly)

Le weekly a montré qu'un verdict à **graine unique** peut être un artefact (l'avantage crypto de
la graine 42 ne tenait sur aucune autre). **Ne pas répéter l'erreur** : lancer NsDiff daily(B) +
mensuel(C) sur les **graines 42-46 d'emblée**, mêmes origines, mêmes budgets. Rapporter par
cellule et par horizon : verdict RMSE × graine (stable ou non), et CV inter-graines
(RMSE/Winkler), comme la §6bis du weekly. **Aucun verdict mensuel ne sera présenté sans sa
stabilité inter-graines.**

---

## 4. Métriques & synthèse

Mêmes métriques que le weekly (réutiliser l'outillage existant, **rien de réimplémenté**) :
RMSE (testé par cellule via `comparison_3_daily_vs_weekly` — vérifier qu'il est agnostique à
l'unité d'horizon, sinon adapter a minima), Cov95, largeur PI, **Winkler**
(`dashboard_d7_w1.winkler_score`), direction ; skill-score vs RW poolé par classe (crypto / actions
/ obligations, ZN=F+TLT dédoublonnés) à graine fixe **et** stabilité multi-graines.

Périmètre modèles : **NsDiff d'abord** (comparaison intra-modèle daily-vs-mensuel, cohérente avec le
travail récent). *Optionnel / next* : ajouter TSDiff mensuel pour la comparaison diffusion-vs-diffusion,
et Naive/ARIMA-GARCH (frequency-agnostiques) comme repères — **mais SARIMA/Prophet ont des
saisonnalités calées daily/weekly** qui ne transfèrent pas au mensuel, donc à ne PAS inclure sans
adaptation déclarée (les traiter comme hors-scope ici, pas comme un oubli).

---

## 5. Où ça s'affiche

Le dashboard `dashboard_d7_w1.py` est **spécifiquement D7/W1** (weekly) — le mensuel **n'y entre
pas**. Livrer le mensuel comme **analyse autonome** (extraction + note), à la manière de la synthèse
daily-vs-weekly. (Un mini-dashboard mensuel dédié serait un chantier séparé, hors de ce brief.)

---

## 6. Non-négociables
- [ ] **Isolation `tracking.db`** : nouvelle valeur `frequence='monthly'` + `horizon_type='monthly'`,
      insertion `oos` idempotente (index unique propre) ; **lignes daily/weekly des modèles
      existants INTACTES** (comptage/`git` avant-après). Le multi-graines dans un **artefact isolé**,
      jamais dans `tracking.db`.
- [ ] **Point-in-time** partout (mu/sd gelés, resample fin-de-mois `dropna`, distance jours-de-bourse
      réelle côté B).
- [ ] **Budgets déclarés** : `seq_len` mensuel (proposer ~12-24 mois, justifier vu la taille de
      l'historique), `epochs=40`, `n_samples`, `k_denoise=20`, ancre fin-de-mois — tous écrits.
- [ ] **Puissance annoncée** : `n`/`effective_n` affichés partout ; « indistinguable » ≠ équivalence.
- [ ] Aucun fichier source existant modifié hors ajouts ; **pytest vert** avant ET après.

## 7. Livrables
1. `experiments/oos_nsdiff_monthly.py` (mirror de `oos_nsdiff_daily_weekly.py`) : régimes B + C
   mensuels, walk-forward multi-graines, insertion `oos` (`frequence='monthly'`).
2. `month_targets` (helper, miroir de `week_targets`) + éventuel `build_monthly` (miroir de
   `build_weekly`).
3. Extraction + `NOTE_compare_daily_vs_monthly_nsdiff.md` : tableau par actif (M+1/M+2/M+3),
   agrégat par classe, table de robustesse multi-graines, **caveat puissance en tête**, verdict
   honnête.
4. Artefact multi-graines isolé (JSON).

## 8. Vérification finale
- [ ] Lignes NsDiff `oos` mensuelles présentes (frequence='monthly', M+1/M+2/M+3, régimes B et C),
      origines documentées, daily/weekly intactes.
- [ ] Table verdict × graine mensuelle + CV ; une phrase claire sur ce qui est stable (probablement
      « rien de significatif de robuste », vu §0 — et c'est un résultat honnête, pas un échec).
- [ ] Note autonome, `n`/`effective_n` partout, caveat puissance explicite. Point d'étape court.

## 9. Pièges à éviter
- **Ne pas** sur-interpréter un « significatif » mensuel isolé : à `effective_n≈10`, un p<0.05 sur
  une graine est très probablement du bruit — d'où le multi-graines obligatoire (§3).
- **Ne pas** régénérer un mois partiel en cours (fuite) ; toujours la dernière fin de mois **résolue**.
- **Ne pas** inclure SARIMA/Prophet mensuels sans adaptation de saisonnalité (§4).
- **Ne pas** écrire le mensuel dans les tables/agrégats du dashboard D7/W1 (hors scope) ni toucher
  aux lignes weekly/daily.
- **Ne pas** présenter le mensuel comme un verdict fort : c'est un complément de la carte des
  horizons, sous forte réserve de puissance.

---

### Livrable final (côté Cowork, après la note)
Une fois `NOTE_compare_daily_vs_monthly_nsdiff.md` prête, la synthèse PDF pourra être **étendue en
une vue multi-horizons** (daily→weekly→mensuel) ou complétée d'une section mensuelle — régénérée
côté Cowork depuis la note, avec le caveat de puissance mis en avant.
