# Message à coller dans Claude Code

Lis `BRIEF_extension_fenetre_validation.md` à la racine. Deux chantiers sur le dashboard
`experiments/dashboard_d7_w1.html` (régime B daily-natif vs régime C weekly natif, horizon W+1,
recoupe `comparison_3` de `matrice_paired_tests.json`).

## Chantier 1 — FAISABLE TOUT DE SUITE : mini-rapport développé à côté de chaque verdict (§6bis)

Partout où un **verdict** s'affiche (table par cellule ET tuiles d'agrégat), ajouter un petit
rapport plus développé, **calculé** (fonctions pures des métriques déjà dans le payload, pas de
texte en dur, aucun recalcul du pipeline) :

- **Base statistique du verdict** : le badge est décidé par le test **RMSE** (bootstrap par blocs) —
  afficher `p`, `IC95` de la différence moyenne (0 exclu/inclus), `n`, `effective_n`, et une phrase
  expliquant ce que « significatif » signifie ici.
- **Lecture KPI par KPI** (RMSE, Winkler, Cov95 vs 0,95, largeur PI *à couverture comparable*,
  direction *balisée bruitée*) : valeur daily, valeur weekly, de quel côté ça penche, note.
- **Concordance / arbitrage** : signaler surtout quand le point (RMSE) et l'incertitude
  (Winkler/Cov95) désignent des gagnants différents.
- Contraintes inchangées : autonome `file://`, tout inline, **aucune synchro JS multi-graphiques**,
  tokens CSS existants. Verdict cliquable → dépliage sous la ligne.

## Chantier 2 — Élargir la fenêtre en remontant AVANT décembre 2025 (§2bis + §3)

Fenêtre tranchée : le « 8 janvier » est abandonné (la validation commence déjà au 5 déc. 2025, plus
tôt — y démarrer *retirerait* des données). On **remonte vers le passé** pour plus de rigueur :

- `--n-test` **30 → 90** (plancher 70, max raisonnable 120) dans `experiments/weekly_multimodel.py`,
  régimes **B et C**, 5 actifs. `n_val`=12, `WEEK_MARGIN`=3 inchangés. 1ʳᵉ origine ~oct. 2024,
  ancrage vendredi (W-FRI). Ne PAS viser le cellule-par-cellule (~216 origines, inutile).
- **Les 6 modèles** (matrice complète). Coût : seuls TSDiff/LSTM ont un vrai training ; les 4
  paramétriques s'étendent gratuitement. Ordre : **4 paramétriques + LSTM d'abord** (LSTM :
  re-sweep SEQ_LEN sur le nouveau bloc de validation d'abord), puis **TSDiff en batch checkpointé**
  (`weekly_multimodel_checkpoint.json`, reprenable).
- **Anti-fuite (§3.2)** : walk-forward par origine obligatoire, aucun train-once-forward.
- Puis backfill DB (`backfill_multimodel_predictions.py` + `backfill_eval_metrics.py`, sans
  doublons) → `matrice_paired_tests.py` (seed fixe) → régénérer le dashboard.

Tu peux faire les chantiers 1 et 2 ; le chantier 1 (mini-rapport) est indépendant et rapide. Si tu
bloques sur une ambiguïté réelle, signale-la — ne devine pas.
