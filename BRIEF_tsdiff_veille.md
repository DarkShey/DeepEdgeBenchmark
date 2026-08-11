# BRIEF — Mettre TSDiff en veille (sortir des runs courants + archiver un run de référence)

> Origine : consigne (07/08, 15h00) — « Mettre TsDiff en veille : le sortir des runs
> courants mais archiver un run comme point de comparaison de référence ; ça tu peux faire asap ».
> Fait suite au verdict du duel (NsDiff domine la calibration, Winkler 29/30 cellules weekly+daily ;
> TSDiff au mieux à égalité sur le point en daily — et au prix d'un intervalle effondré ; s'effondre
> en régime peu doté) et à la note de décision (retrait de production, borne haute « point »
> conservée).
>
> Hors périmètre : **aucun réentraînement**, **aucun nouveau run TSDiff**, **aucune suppression**
> de ligne d'historique ou d'artefact existant. La mise en veille est réversible par construction.

## 0. Le problème, en une phrase

TSDiff est encore un **contendant actif** : il est enregistré dans le registre de prévision
(`benchmarks/multi_horizon.py`, dict des forecasters → `"TSDiff": forecast_horizons_tsdiff`,
l. 443, builder l. 354–377) et listé dans les runs (`experiments/audit_coverage.py:25`,
`MODELS = (…, "TSDiff")`). **Objectif** : qu'aucun *nouveau* run ne le régénère, tout en gelant
**un** run existant comme point de comparaison de référence.

## 1. Ce qui existe déjà (à réutiliser, ne pas réécrire)

- **Registre des prévisionnistes** : `benchmarks/multi_horizon.py` — le builder TSDiff (l. 354–377)
  et l'entrée `"TSDiff": forecast_horizons_tsdiff` (l. 443).
- **Listes de modèles des runs courants** : `experiments/audit_coverage.py:25`
  (`MODELS = (…, "TSDiff")`) et tout script d'orchestration qui itère sur cette liste.
- **Artefacts TSDiff existants** (candidats à la référence) : `experiments/duel_backtest_nsdiff_swept.json`
  — le duel **équitable, budget d'époques sweepé** (cellules TSDiff par actif × graine × horizon).
  C'est le run le plus défendable comme référence. Les `backtest_rolling_tsdiffw*.json` et
  `tsdiff_monthly_multiseed.json` sont des runs secondaires (à laisser en place, pas la référence).
- **Lignes `oos` TSDiff dans `tracking.db`** (historique lu par le dashboard) — **à CONSERVER**
  telles quelles.

L'infrastructure est là. Ce qu'il faut, c'est **débrancher la génération** et **figer une
référence** — rien de plus.

## 2. Les changements à faire

### 2.1 Sortir TSDiff des runs courants (génération)

- Dans `benchmarks/multi_horizon.py`, **retirer `"TSDiff"` du dict des forecasters** (le commenter,
  **ne pas supprimer** le builder `forecast_horizons_tsdiff` — on veut pouvoir réactiver). Ajouter
  un commentaire d'ancrage :
  `# TSDiff EN VEILLE (2026-08-07) — voir archives/tsdiff_reference/. Décommenter pour réactiver.`
- Dans `experiments/audit_coverage.py:25` et toute autre liste `MODELS = (…)` d'un **script de run
  courant**, retirer `"TSDiff"` de la partie active (commenter + même note).
- **Ne PAS toucher** aux scripts d'analyse/comparaison historiques (`compare_*`, `duel_*`,
  `oos_tsdiff_*`, `weekly_*_tsdiff*`) : ils doivent rester exécutables pour rejouer la référence si
  besoin. On débranche la *génération courante*, pas la capacité de rejouer.

### 2.2 Archiver un run de référence (figé)

- Créer `archives/tsdiff_reference/`.
- Y **copier** (copie, jamais déplacement) : `experiments/duel_backtest_nsdiff_swept.json` +, s'ils
  existent séparément, les méta d'époques sélectionnées associées.
- Écrire `archives/tsdiff_reference/MANIFEST.md` consignant :
  - date de gel + commit git (`git rev-parse HEAD`) ;
  - run choisi et **pourquoi** (duel équitable, budget d'époques sweepé, 5 actifs × 5 graines ×
    3 horizons) ;
  - budgets figés (`seq_len`, epochs par actif, `n_samples`) tels qu'ils sont dans le JSON ;
  - le **verdict associé** (calibration dominée par NsDiff — Winkler 29/30 ; égalité au mieux sur le
    point en daily ; collapse en régime peu doté) ;
  - renvois vers la note de décision et le glossaire.
- **Ne pas régénérer** ce run : on archive l'existant tel quel.

### 2.3 Dashboard (option, à confirmer)

Le dashboard lit les lignes `oos` de `tracking.db`. Deux choix :

- **(a) recommandé** — laisser la colonne TSDiff s'afficher, mais l'**étiqueter « référence
  archivée / hors compétition (gelée) »**, pour garder la référence lisible sans la réintroduire
  comme contendant ;
- (b) la masquer de l'affichage courant.

Défaut : (a). Dans les deux cas, **aucune ligne `oos` n'est supprimée**.

## 3. Non-négociables

- **Aucune ligne supprimée** dans `tracking.db` (contrôle avant/après : compte des lignes
  `source='oos'` par modèle **identique** ; TSDiff `oos` conservé).
- **Aucun artefact existant supprimé ou déplacé** — on **copie** vers `archives/`.
- **Aucun réentraînement**, aucun nouveau run TSDiff, aucune écriture nouvelle de ligne TSDiff.
- **pytest vert avant ET après** : `python -m pytest experiments validation models -q`.
- **Réversibilité** : la veille = commenter des entrées de registre, pas supprimer du code. Un simple
  décommentage réactive TSDiff.

## 4. Critères d'acceptation

- `grep -n "TSDiff" benchmarks/multi_horizon.py` : le builder est présent, l'entrée du registre est
  **commentée** avec la note « EN VEILLE ».
- Un run courant (dashboard / benchmark) ne produit **aucune** nouvelle ligne TSDiff — compte `oos`
  TSDiff **inchangé** avant/après.
- `archives/tsdiff_reference/` contient le JSON de référence + `MANIFEST.md` (commit, budgets,
  verdict, renvois).
- pytest vert, `tracking.db` intègre (compte par modèle inchangé).

## 5. Ce que la veille ne fait PAS (à garder en tête)

- Elle ne supprime pas TSDiff du code ni de l'historique — elle le sort de la compétition courante.
- Elle ne préjuge pas d'un retour : si une architecture TSDiff dotée d'un **backbone de dispersion
  dédié** apparaît un jour, on réactive via le registre (§2.1) et on rejoue le duel équitable.
