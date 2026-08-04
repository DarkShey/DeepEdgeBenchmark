# BRIEF — Réduire deux limites de la synthèse NsDiff : récupérer TLT + robustesse multi-graines

*Créé le 2026-08-04. Suite de `NOTE_compare_daily_vs_weekly_nsdiff.md` et
`NOTE_nsdiff_dashboard_daily_oos.md`. But : lever (ou documenter précisément l'échec de) les deux
limites principales de la synthèse daily-vs-weekly de NsDiff — TLT absent, et single-seed.*

---

## Fix 1 — Récupérer TLT pour NsDiff (passer de 4/5 à 5/5 actifs)

### Cause exacte (déjà diagnostiquée, `NOTE_nsdiff_dashboard_daily_oos.md` §3)
`yfinance` (`auto_adjust=True`) retraite rétroactivement les dividendes de TLT → le `last_close`
récupéré ne matche plus les triplets déjà stockés en `oos` pour les 6 autres modèles (86.59 vs
86.94 attendu, ~0.4%) → le garde-fou anti-fuite refuse → bascule sur le cache offline
(`offline_prices.py` / `DONNEE~1.XLS`) qui **s'arrête au 2026-07-02** → les `target_date` W+2/W+3
des dernières origines (~2026-07-23) tombent hors cache → `IndexError` → TLT sauté.

### Approche (dans l'ordre, s'arrêter à la première qui passe le garde-fou)
1. **Prix bruts, bonne convention** : refetch TLT en `auto_adjust=False` (close brut, la
   convention des lignes `oos` déjà en base), couvrant jusqu'à la dernière `target_date` requise
   (~2026-07-25). Vérifier que `last_close` matche les triplets stockés (`load_baseline_triplets`
   / `_daily`) à **chaque cutoff réutilisé**, tolérance ~1e-6 relatif — le MÊME garde-fou qui a
   échoué avant doit maintenant passer.
2. **Sinon, étendre le cache offline** : compléter `offline_prices` avec les closes TLT du
   2026-07-03 → ~07-25, dans la même convention que les lignes stockées, puis relancer.
3. **Ne jamais bricoler un prix pour passer le check.** Si aucune source ne recolle aux prix en
   base, **garder TLT exclu** et documenter précisément pourquoi (quel cutoff, quel écart) —
   une limite honnête vaut mieux qu'un chiffre faux.

### Livrable Fix 1
- NsDiff daily(B) + weekly(C) **TLT** insérés en `source='oos'` sur les mêmes origines que les 6
  modèles (mirror exact de `oos_nsdiff_daily_weekly.py`, juste TLT ajouté).
- Dashboard régénéré : **35 cellules** (7 modèles × 5 actifs), verdict « Obligations » désormais
  sur **ZN=F + TLT** (plus la réserve « ZN=F seul »).
- Note mise à jour (`NOTE_compare_daily_vs_weekly_nsdiff.md`) : ligne TLT remplie, réserve
  obligations levée si le pooling bond change de verdict.

---

## Fix 2 — Robustesse multi-graines du verdict daily-vs-weekly

### Cadrage (important, à ne pas casser)
La piste `oos`/dashboard est **single-seed par design** : les 6 modèles y sont stockés en une
ligne par origine (seed 42). **Ne PAS multi-seeder les lignes `oos` du dashboard** — elles
doivent rester comparables aux 6 autres modèles. Le multi-graines est une **table de robustesse
séparée**, en complément, pas un remplacement.

Rappel : le côté **weekly** de NsDiff est déjà multi-graines dans le duel — c'est le côté
**daily** et la comparaison daily-vs-weekly qui ne le sont pas encore.

### Approche
- Relancer NsDiff daily(B) + weekly(C) sur les **graines 42-46** (comme le duel), mêmes origines,
  mêmes budgets (epochs=40, seq_len, k_denoise déclarés), dans un **artefact isolé**
  (JSON/checkpoints dédiés, à la manière du duel — surtout pas dans `tracking.db` `oos`).
- Rapporter, par actif et par classe : **stabilité du verdict** daily-vs-weekly (le signe et la
  significativité RMSE/Winkler changent-ils d'une graine à l'autre ?) et **CV inter-graines** du
  CRPS/Winkler de NsDiff daily et weekly (comme `NOTE_duel_nsdiff.md` §Q3 le fait côté weekly).
- Réutiliser l'outillage de test existant (`comparison_3_daily_vs_weekly`, `paired_test`,
  `winkler_score`) — **aucune réimplémentation**.

### Question à laquelle la table doit répondre
Le verdict « **weekly meilleur sur la crypto, indistinguable ailleurs** » tient-il sur les 5
graines, ou était-ce un effet de la graine 42 ? (C'est exactement le type de renversement que le
duel a déjà trouvé côté TSDiff — donc à vérifier honnêtement.)

### Coût
NsDiff est très bon marché (~0.7 min les 4 actifs single-seed) → 5 graines × 2 régimes ≈ quelques
minutes. Pas de raison de s'en priver.

### Livrable Fix 2
- `experiments/nsdiff_daily_weekly_multiseed.py` (+ artefact JSON isolé) et une section
  « robustesse multi-graines » ajoutée à `NOTE_compare_daily_vs_weekly_nsdiff.md` :
  table verdict × graine + CV inter-graines, conclusion sur la stabilité.

---

---

## Livrable final — synthèse PDF améliorée (régénérée après les deux fix)

Une fois Fix 1 et Fix 2 intégrés et `NOTE_compare_daily_vs_weekly_nsdiff.md` à jour, **produire
une meilleure version de la synthèse PDF** (`synthese_nsdiff_daily_vs_weekly.pdf`) à partir de la
note mise à jour. Elle doit :

- **Intégrer les nouveaux résultats** : ligne **TLT** remplie (ou son exclusion documentée
  proprement) ; verdict « Obligations » sur **ZN=F + TLT** ; **nouvelle section / table de
  robustesse multi-graines** (verdict × graine + CV inter-graines) ; **section limites allégée**
  en conséquence (TLT et single-seed ne sont plus des réserves ouvertes si les fix aboutissent).
- **Être meilleure sur le fond ET la forme**, pas juste des chiffres réactualisés :
  - un **résumé exécutif de 3-4 lignes en tête** (le verdict d'un coup d'œil) ;
  - au moins **un visuel** clair — p.ex. couverture (Cov95) daily vs weekly par actif, ou
    skill-score par classe — lisible en N&B, cohérent avec la charte des autres figures du dépôt ;
  - tables propres, **`n`/`effective_n` visibles**, direction descriptive distinguée du
    significatif.
- **Rester envoyable telle quelle** : aucune mention de « tuteur » ni de destinataire interne ;
  ton neutre de note de résultats.

Note d'exécution : ce PDF est **régénéré côté Cowork** à partir de la note mise à jour (ce n'est
pas un artefact que Claude Code doit produire) — le rôle de Claude Code s'arrête à la note `.md`
correcte et complète ; le PDF amélioré en découle.

---

## Non-négociables (communs)
- [ ] **Lignes `oos` des 6 modèles + single-seed NsDiff INTACTES** : Fix 2 écrit dans un artefact
      isolé, jamais dans `oos`. Contrôle comptage/`git` avant-après.
- [ ] **Origines identiques** aux lignes existantes (lues verbatim, pas régénérées).
- [ ] **Point-in-time** partout (mu/sd gelés au train ; W-FRI dropna côté C ; distance
      jours-de-bourse réelle côté B).
- [ ] **Aucun prix inventé** pour TLT (Fix 1 point 3).
- [ ] Aucun fichier source existant modifié hors ajouts ; **pytest vert** avant ET après.
- [ ] Toute conclusion accompagnée de `n`/`effective_n` ; rien de significatif ≠ « équivalent ».

## Vérification finale
- [ ] Fix 1 : dashboard = 35 cellules, TLT présent pour NsDiff, `last_close` TLT recollé aux
      triplets (ou TLT documenté comme irrécupérable, avec le détail).
- [ ] Fix 2 : table verdict × graine + CV, et une phrase claire — le verdict crypto tient-il sur
      5 graines ?
- [ ] `NOTE_compare_daily_vs_weekly_nsdiff.md` mise à jour ; section limites allégée en
      conséquence. Point d'étape court à la fin.

## Pièges
- Ne pas multi-seeder les lignes `oos` du dashboard (elles doivent rester single-seed, comparables
  aux 6 modèles) — le multi-graines est une table à part.
- Ne pas forcer un prix TLT pour passer le garde-fou : soit ça recolle honnêtement, soit TLT
  reste exclu et documenté.
- Ne pas relancer les 6 autres modèles ni le duel : seul NsDiff est concerné ici.
