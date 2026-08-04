# BRIEF — NsDiff « en entier » sur le dashboard D7/W1 : côté daily (régime B) + weekly (régime C) en `source='oos'`

*Créé le 2026-08-04. Suite directe de `BRIEF_nsdiff_weekly_parite_et_compa.md` (axe 2, dette §7)
et de `NOTE_compare_weekly_tsdiff_nsdiff.md`. Ferme d'un coup deux sujets : (a) NsDiff visible
sur le dashboard, (b) 2ᵉ objectif du tuteur « Daily vs Weekly » pour NsDiff.*

---

## 0. Objectif (une phrase)

Faire apparaître **NsDiff comme 7ᵉ modèle** sur `experiments/dashboard_d7_w1.py`, en produisant
ses prédictions **`source='oos'`** des **deux côtés** de la comparaison — **daily (régime B)** ET
**weekly natif (régime C)** — à l'horizon **W+1**, sur **exactement les mêmes origines**
(actif, cutoff, target) que les 6 modèles déjà présents.

---

## 1. Pourquoi ça bloque aujourd'hui (mécanisme, vérifié dans le code)

`dashboard_d7_w1.py` **n'est pas** un simple affichage de modèles : c'est par construction une
comparaison **Daily vs Weekly par modèle** à W+1. Chaîne exacte (relue) :

- `main()` → `df = mpt.load_predictions(db_path)` → **ne charge que les lignes `source='oos'`**.
- `build_enriched_pairs` → `mpt.build_daily_weekly_pairs(df, horizon_units=["W+1"])` : pour
  **chaque modèle**, apparie son **côté daily** (régime B : `frequence='daily'`,
  `horizon_type='weekly'`, `horizon_unit='W+1'`) avec son **côté weekly** (régime C :
  `frequence='weekly'`), les deux devant partager **le même `target_date`** (assert dans le code).
- Point d'étape : « **30 attendues = 6 modèles × 5 actifs** ».

**Conséquence :** un modèle n'apparaît que s'il a **les deux côtés en `oos`**. NsDiff n'a
aujourd'hui **ni côté daily** (le vrai trou), **ni lignes `oos`** (son weekly existe seulement
sous `source='live'` et `source='backtest_rolling_nsdiff'`, que le dashboard ne lit pas). Ce
n'est ni une liste de modèles en dur (confirmé au grep), ni le bug `aggregate_per_cell` déjà
corrigé : il n'y a **littéralement aucune donnée NsDiff à apparier**.

---

## 2. Ce qu'il faut produire — deux briques, mêmes origines

Cible : dans `validation/tracking.db`, `source='oos'`, `horizon_unit='W+1'` (au moins ;
idéalement W+1/W+2/W+3 pour matcher le stockage des autres modèles), pour les 5 actifs :

- **Brique A — NsDiff daily, régime B (LE nouveau travail).** Walk-forward du **NsDiff daily**
  sur les origines OOS, prévoyant à la **distance en jours de bourse** jusqu'à la cible-vendredi
  W+1 — exactement comme TSDiff-D et les classiques côté régime B. Adaptateur **déjà câblé** :
  `benchmarks/multi_horizon.forecast_horizons_nsdiff` (dans `MODEL_ADAPTERS`).
  → `frequence='daily'`, `horizon_type='weekly'`, `horizon_unit='W+1'`.
- **Brique B — NsDiff weekly, régime C, en `oos`.** Walk-forward **weekly-natif** de NsDiff sur
  **les mêmes origines**, écrit en `source='oos'`. Le weekly existe déjà (module
  `models/nsdiff_weekly.py` / `nsdiff_model.fit_nsdiff` sur données weekly), mais **pas sous
  `oos`** — il faut le (re)produire sur les origines du dashboard et l'insérer en `oos`.
  → `frequence='weekly'`, `horizon_type='weekly'`, `horizon_unit='W+1'`.

**Contrainte d'appariement (non négociable) :** les deux briques doivent tomber sur des
**(asset, cutoff_date, target_date) strictement identiques** à ceux des 6 modèles déjà en `oos`
— sinon `build_daily_weekly_pairs` ne trouve pas la jointure et NsDiff est écarté, ou pire, il
pollue le pooling avec des origines décalées.

---

## 3. Comment — reproduire l'existant, ne rien réinventer

1. **Découvrir (grep) comment TSDiff-D et les baselines ont peuplé leurs lignes `oos`
   daily(B) + weekly(C) W+1** : quel script, quelles origines (`n_test`, `three_way_split`,
   `week_targets`), quel chemin d'insertion (probables : `experiments/weekly_multimodel.py`
   pour les régimes B/C des classiques ; insertion `oos` via
   `validation/…insert_oos_predictions` / `experiments/backfill_weekly_predictions.py` —
   **à confirmer, ne pas supposer**). Le côté diffusion daily/weekly de TSDiff est le miroir
   le plus proche à suivre.
2. **Réutiliser le MÊME jeu d'origines OOS** que ces lignes (le lire depuis la DB / le script
   d'origine, pas le régénérer « à peu près »). Vérifier après coup que les `target_date` de
   NsDiff coïncident avec ceux d'au moins un modèle existant sur chaque actif.
3. **Insérer en `source='oos'`** avec le chemin idempotent existant (index unique partiel
   `WHERE source='oos'`). La colonne `model='NsDiff'` distingue les lignes des 6 autres modèles
   → **aucune collision**, et surtout **on vise bien `oos`** (un `source_tag` isolé comme au
   backtest rolling **ne marcherait pas** : le dashboard ne le lirait pas).

---

## 4. Décision seed (à ne pas confondre avec le duel)

La piste `oos`/dashboard est un **walk-forward déterministe single-seed** (les 6 modèles y sont
en une passe, `seed=42`). Produire NsDiff `oos` **au même seed canonique**, **pas** en
multi-graines. Le multi-graines, c'est le **duel** (chantier séparé). Une comparaison daily
rigoureuse multi-graines TSDiff-vs-NsDiff reste possible plus tard, mais **hors scope de ce
brief** : ici on veut juste NsDiff **complet et apparié** sur le dashboard.

---

## 5. Non-négociables (repris des briefs précédents)

- [ ] **Point-in-time** : `mu`/`sd` sur le train seul ; côté C, resample `W-FRI` + `dropna()` ;
      côté B, forecast à la distance jours-de-bourse réelle jusqu'à la cible (jamais +7j
      calendaires en dur — cf. le recâblage régime A→B documenté dans l'en-tête du dashboard).
- [ ] **Origines identiques** aux 6 modèles (bloquant — sinon pas d'appariement).
- [ ] **Insertion `oos` idempotente** ; **lignes des 6 modèles et `backtest_rolling*` intactes**
      (`git`/comptage de contrôle avant/après).
- [ ] **Aucun fichier source existant modifié** hors ajouts ; **pytest vert** avant ET après.
- [ ] Époques / seq_len / k_denoise NsDiff **déclarés** ; cohérents avec la définition NsDiff-W
      déjà utilisée (cf. `BRIEF_nsdiff_weekly_parite_et_compa.md` §2 — expliciter seq_len et
      budget d'époques, ne pas les masquer).

---

## 6. Vérification finale (le test « c'est fini »)

- [ ] `python experiments/dashboard_d7_w1.py` affiche « **35 attendues = 7 modèles × 5 actifs** »
      (au lieu de 30), NsDiff présent dans les **cellules** ET dans l'**agrégat poolé**.
- [ ] Page ouvrable en `file://` **sans gel** (rejouer un clic), `n`/`effective_n` affichés par
      cellule NsDiff.
- [ ] **Recoupement** : les métriques weekly de NsDiff sur le dashboard recoupent
      `NOTE_compare_weekly_tsdiff_nsdiff.md` (même modèle, mêmes origines → mêmes chiffres à
      l'échantillonnage près).
- [ ] Verdict Daily-vs-Weekly de NsDiff **lisible** (par cellule + poolé) — c'est le livrable
      « Daily vs Weekly » pour NsDiff.

---

## 7. Piège connu — TLT / yfinance

La dérive `yfinance` qui a bloqué TLT au backtest rolling peut se reproduire. Si c'est le cas :
utiliser le cache prix (`experiments/offline_prices.py` / `.price_cache_*`) ou **documenter
l'actif manquant** dans la note — **ne pas** laisser un actif faire échouer tout le run, et
**ne pas** conclure sur la classe *obligations* si seul ZN=F survit.

---

## 8. Livrables

1. Le(s) script(s) producteur(s) NsDiff régime B + régime C `oos` (miroir de l'existant TSDiff/
   baselines), paramétrables (actifs, origines, seed).
2. Lignes NsDiff `oos` (daily + weekly, W+1 min) dans `tracking.db`, sur les origines des 6 modèles.
3. `dashboard_d7_w1.html` régénéré, **NsDiff visible** (7ᵉ modèle), page autonome sans gel.
4. Courte note (`NOTE_nsdiff_dashboard_daily_oos.md`) : origines réutilisées, seed, verdict
   Daily-vs-Weekly de NsDiff, limites/dettes déclarées (TLT le cas échéant, single-seed).

---

## 9. Pièges à éviter

- **Ne pas** écrire NsDiff sous un `source_tag` isolé (backtest_rolling_nsdiff, live…) : le
  dashboard **ne lit que `source='oos'`**.
- **Ne pas** inventer de nouvelles origines : réutiliser **exactement** celles des 6 modèles,
  sinon l'appariement daily/weekly échoue et le pooling est faussé.
- **Ne pas** confondre cette passe `oos` single-seed (dashboard) avec le duel multi-graines.
- **Ne pas** oublier le côté **daily** : c'est LE manque. Le weekly seul en `oos` ne suffit pas —
  sans côté daily, NsDiff reste invisible (rien à apparier).
