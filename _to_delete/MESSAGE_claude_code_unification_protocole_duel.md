# Message à coller dans Claude Code

Lis `BRIEF_unification_protocole_duel.md` à la racine et suis-le. Ce brief vient des deux
rapports du tuteur (`rapport_methodologie.pdf` 27/07 + `rapport_audit_code2.pdf` 28/07) et
traite **la reco #2 de l'audit** — construire le « duel » diffusion vs classiques qui n'existe
nulle part dans le code. C'est le verrou qui débloque tout classement légitime.

## Le problème en une phrase

Le seul CRPS empirique correct du dépôt (`experiments/crps_metrics.py`) compare la diffusion
**à elle-même** (TSDiff-W vs TSDiff-D). Les pipelines qui comparent vraiment diffusion et
classiques (A et B) ne calculent **aucun CRPS** et souffrent d'une asymétrie de protocole
(TSDiff gelé à T0, les 5 autres refit à chaque origine) qu'**aucun test ne corrige** —
`pooled_analysis.py` refuse d'ailleurs de classer les modèles pour ça. Objectif : un backtest
commun, apparié, 6 modèles à conditions **strictement identiques**, CRPS équitable, tests par
paire, résumé par MCS.

## Non-négociables (détaillés au §2 du brief)

- **Réutiliser l'existant, ne pas réécrire** : `crps_empirical`, `honest_eval::dm_hac_test`
  (DM-HAC-HLN déjà correct), `paired_test.py`, `pooled_analysis.py` (échelle MASE, Holm,
  classes d'actifs), `epoch_sweep.py::three_way_split`. L'infra est là — il manque **l'assemblage**.
- **Générateur d'origines commun** (résout É2) : mêmes origines + mêmes dates-cibles pour les 6
  modèles ; **même règle de ré-estimation pour tous** (l'asymétrie actuelle est interdite ; si
  inévitable pour raison de coût → déclarée ET biais quantifié) ; rolling origin ; **purge +
  embargo** pour les horizons > 1 jour (absent partout aujourd'hui) ; splits ancrés sur
  `three_way_split`, sélection d'hyperparamètres **jamais sur le test** (verrou É1, déjà respecté
  côté weekly — le préserver).
- **Adaptateurs d'échantillonnage des 5 classiques** (résout N1) : chacun produit `m = 500`
  **vraies trajectoires simulées** du modèle réel (GARCH simulé, state-space SARIMA, loi Prophet,
  LSTM par MC-Dropout/bootstrap résidus, Naive via `random_walk_samples`). **Interdit** : la
  reconstruction paramétrique gaussienne/log-normale des bornes d'IC (détruit l'info
  distributionnelle — réserve N1).
- **CRPS équitable** : `crps_empirical`, `m = 500` **strictement identique** pour tous ; ajouter
  le **fair CRPS** (Ferro 2014) recommandé.
- **Tests à la bonne comparaison** (résout É3) : `dm_hac_test` par paire (diffusion, classique) ×
  actif × horizon, **HAC ≥ h−1** + HLN + bootstrap par blocs ; **Clark-West** vs le naïf emboîté ;
  **correction Holm** sur toute la grille (5 actifs × 3 horizons × paires).
- **MCS neuf** (absent du dépôt) : Model Confidence Set de Hansen-Lunde-Nason (2011) par
  actif × horizon ; SPA vs GARCH(1,1) optionnel. Code neuf, **testé**.

## Livrable

Un artefact reproductible (JSON + table) par actif × horizon : fair CRPS (`m=500`), p-valeurs
DM-HLN corrigées par paire, `effective_n`, et **appartenance au MCS**. Conclusion à formuler en
MCS, pas en « jeu égal » / « globalement moins bon » (rappel N3 : 1 verdict favorable sur 440).

## Décisions déjà tranchées (ne me les redemande pas)

- Travail **directement sur `main`**, **commits atomiques** (un chantier par commit), **`pytest`
  vert à chaque étape**, rien de cassé dans les pipelines existants.
- **Ne PAS toucher aux pipelines de production A & B** — c'est le chantier #7, plus tard.
- **Ne rien recalculer** sur les artefacts contaminés (`weekly_headtohead_results*.json`) : le
  duel repart d'origines propres via `three_way_split`.
- Toute asymétrie résiduelle non résolue → **écrite noir sur blanc** dans le livrable, jamais
  masquée.

Termine par la **grille de conformité du §4** (les 8 lignes doivent passer « conforme ») et les
garde-fous du §5. Fais un point d'étape court à la fin. Si tu bloques sur une ambiguïté réelle,
signale-la — ne devine pas.
