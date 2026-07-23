# Message à coller dans Claude Code

Lis `BRIEF_dashboard_D7_vs_W1.md` à la racine et suis-le. Résumé et non-négociables :

Je veux un **mini-dashboard autonome** sur la seule comparaison **D+7 vs W+1** (origines-vendredi),
**externalisé** dans un fichier séparé ouvrable directement dans Firefox en `file://`. Il ne doit
**pas** toucher `Run/dashboard.html`, ni le pipeline, ni la DB — il lit `validation/tracking.db` et
produit `experiments/dashboard_d7_w1.html` via un nouveau script `experiments/dashboard_d7_w1.py`.

Contraintes que je ne négocie pas :
- Appariement **identique à `comparison_4_d7_vs_w1`** (importe la logique de
  `experiments/matrice_paired_tests.py`, ne la recopie pas) ; le résultat doit **recouper**
  `matrice_paired_tests.json`.
- Tests par **bootstrap par blocs via `experiments/paired_test.py`**, seed fixe, verdict seulement si
  p < 0,05 sinon « indistinguable ». **Affiche `n` ET `effective_n` partout** (~3-4 par cellule :
  puissance faible, à montrer, pas à cacher).
- Métrique probabiliste = **Winkler / Interval Score @95** (la DB n'a que les bornes, pas d'ensemble
  → pas de faux CRPS). Plus RMSE, Cov95, largeur de PI, et direction en diagnostic secondaire balisé.
- **Agrégation inter-actifs** via **skill-score sans échelle** vs baseline RW, **groupée par classe**
  (crypto / actions / obligations), en **dédoublonnant ZN=F & TLT** (corrélés) — pour gagner en
  puissance sans tricher. Caveat de corrélation affiché.
- En-tête : libeller la comparaison **« régime A (daily→7j) vs régime C (weekly natif) sur
  cible-vendredi »**, pas « daily vs weekly » (confusion horizon × régime).
- **Page 100 % autonome** (tout inline, aucun CDN/fetch requis) et **aucune synchro JS
  multi-graphiques** (cf. `CORRECTIF_dashboard_v4_boucle_infinie.md` — ça a figé Firefox). Réutilise
  les tokens CSS de `Run/dashboard.html`.

Termine par la **vérification du §6 du brief** : recoupement avec `comparison_4`, ouverture Firefox
`file://` sans gel (rejoue un clic de zoom), reproductibilité à seed égal, Winkler validé sur un cas
jouet. Fais un point d'étape court à la fin.

Décisions déjà tranchées (ne me les redemande pas) : générateur lisant la DB ; on pousse
l'agrégation pour la puissance ; Winkler et non CRPS. Si tu bloques sur une ambiguïté réelle,
signale-la — ne devine pas.
