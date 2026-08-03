# BRIEF — Égaliser le budget d'époques de NsDiff (sweep validation) dans le duel

*Cible : Claude Code. Rédigé le 2026-08-03. Fait suite à `BRIEF_integration_nsdiff.md` +
`NOTE_duel_nsdiff.md`.*

---

## 0. Objectif en une phrase

Donner à **NsDiff-W** exactement le **même sweep d'époques sélectionné sur la validation**
que TSDiff-W possède déjà (candidats, sélection val-only, jamais le test), rejouer le duel
multi-graines `--include-nsdiff`, et régénérer la note — pour que la comparaison de
**stabilité inter-graines** (le CV) soit *toutes choses égales par ailleurs* et non biaisée
par le fait que NsDiff était figé à `NSDIFF_EPOCHS_W=40` pendant que TSDiff sweepait 40/60/80.

**Ce n'est PAS une optimisation pour “faire gagner” NsDiff.** C'est une correction
d'**équité méthodologique** : on neutralise la seule réserve de `NOTE_duel_nsdiff.md`
(§Limites : « le CV 4× plus faible de NsDiff n'est pas toutes choses égales par ailleurs »).
Le verdict CRPS (NsDiff bat TSDiff 15/15) et le mur GARCH (0/75) ne sont pas censés bouger —
et il faut le **rapporter honnêtement quel que soit le résultat**, y compris si le sweep
réduit l'écart de stabilité ou si les époques sélectionnées finissent toutes à 40.

---

## 1. Règle d'or : additif, jamais destructif (identique au brief précédent)

1. **Ne rien casser.** Le sweep de **TSDiff-W** existant (`epoch_sweep._sweep_one_model` /
   `select_epochs`, flags `--tsdiff-epoch-candidates` / `--tsdiff-hp-samples`) reste
   **strictement inchangé** dans son comportement. Idem pour les 5 classiques, le duel
   6-modèles, et le run NsDiff-fixe déjà produit.
2. **Artefacts isolés.** Ne jamais écraser `experiments/duel_backtest.json` (6-modèles),
   `experiments/duel_backtest_nsdiff.json` (NsDiff époques fixes, déjà scoré), ni les
   checkpoints existants. Le run swept écrit dans un **nouveau** fichier
   (`experiments/duel_backtest_nsdiff_swept.json`) et un **nouveau** sous-dossier de
   checkpoints.
3. **Reproductibilité préservée.** Sans les nouveaux flags, tout run antérieur reste
   byte-identique. Le sweep NsDiff est **opt-in** (voir §3), défaut = comportement actuel
   (époques fixes) inchangé.
4. **Non-régression prouvée.** Suite `pytest` verte avant/après chaque étape (491 tests
   aujourd'hui), relancée après chaque fichier touché.
5. **Branche dédiée** (`feat/nsdiff-epoch-sweep`, à partir de `feat/nsdiff-model`), commits
   atomiques.

---

## 2. Le contrat de sélection à respecter (verrou E1, non négociable)

C'est le point le plus sensible : la sélection d'époques doit être **conforme au protocole
du duel**, exactement comme TSDiff-W.

- **Sélection sur le bloc de validation UNIQUEMENT.** Le bloc de test n'est **jamais** lu
  pour choisir les époques. C'est le verrou E1 documenté dans `duel_origins.py` et déjà
  respecté par `epoch_sweep.select_epochs` pour TSDiff-W. NsDiff doit passer par la même
  logique `val_pos`-only.
- **Frozen-at-T0.** Une fois les époques sélectionnées, NsDiff-W est fit **une seule fois**
  sur les données ≤ T0 avec ce nombre d'époques, puis gelé (mêmes origines, même règle que
  tous les modèles du duel).
- **`m` et `hp_samples`.** Le nombre d'échantillons pour la **sélection** (`--nsdiff-hp-
  samples`, bloc val) est séparé du `m=500` du **scoring** final (identique pour tous les
  modèles). Copier la séparation que TSDiff-W fait déjà (`--tsdiff-hp-samples` vs
  `--m-samples`).
- **Traçabilité.** Enregistrer les époques effectivement sélectionnées **par (graine, actif)**,
  comme le tableau TSDiff-W de `NOTE_duel_diffusion_vs_classiques.md`. Aucune moyenne ne doit
  masquer la variabilité du choix d'une graine à l'autre — c'est une donnée du résultat.

---

## 3. Plan d'implémentation (additif, en réutilisant la machinerie existante)

### 3.1 Lire d'abord
- `experiments/epoch_sweep.py` — surtout `_sweep_one_model` et `select_epochs`. **Déterminer
  s'ils sont paramétrables par modèle** ou codés en dur pour TSDiff-W (le label
  `"TSDiff-W"` est passé en argument dans `duel_backtest.py` lignes ~146-150, ce qui suggère
  une certaine généricité — à vérifier).
- `experiments/duel_backtest.py` — le bloc « TSDiff-W epoch selection » (~142-159) : c'est
  le **patron exact** à répliquer pour NsDiff-W (sweep val-only → `select_epochs` →
  `fit(..., epochs=epochs*)`).
- `experiments/duel_multiseed.py` — flags `--tsdiff-epoch-candidates` / `--tsdiff-hp-samples`,
  threading de `args.seed`, `--include-nsdiff`.
- `models/nsdiff_weekly.py` — comment NsDiff-W s'entraîne (pour savoir s'il supporte des
  checkpoints incrémentaux, cf. §3.3).

### 3.2 Router NsDiff dans le sweep — chemin le moins invasif
Choisir, par ordre de préférence :
1. **Si `_sweep_one_model` est déjà model-agnostique** (prend un label / un builder de
   modèle) : l'appeler pour `"NsDiff-W"` avec `nsdiff_weekly.fit_weekly` comme entraîneur,
   `--nsdiff-epoch-candidates`, `--nsdiff-hp-samples`, `val_pos`. Zéro modif de la fonction.
2. **Sinon** (codé en dur TSDiff) : ajouter dans `epoch_sweep.py` une fonction sœur
   `_sweep_one_model_nsdiff` (ou paramétrer `_sweep_one_model` par un `model_builder`
   **avec valeur par défaut = TSDiff**, pour ne rien changer aux appels existants). La
   nouvelle fonction **réutilise `select_epochs` tel quel** (le critère de sélection
   val-only est identique et ne doit pas être dupliqué/réécrit).
   → **Interdiction** de modifier la branche TSDiff de `_sweep_one_model` autrement que par
   ajout d'un paramètre optionnel rétro-compatible.

### 3.3 Sweep incrémental vs fits indépendants (décision à déclarer)
Le sweep TSDiff-W est efficace parce qu'il entraîne **une fois** et snapshote la perte de
validation aux paliers 40/60/80 (checkpoints incrémentaux). Pour NsDiff :
- **Si `nsdiff_weekly` permet un entraînement incrémental avec snapshots** aux paliers →
  le faire (même schéma que TSDiff, un seul entraînement par graine×actif).
- **Sinon** → fits indépendants (un fit complet par candidat d'époques). C'est plus coûteux
  mais **acceptable** (NsDiff est ~23× plus rapide que TSDiff : même ×3 candidats, le sweep
  complet 5 graines × 5 actifs reste de l'ordre de quelques minutes). **Déclarer ce choix**
  en tête du code et dans la note — ne pas le masquer.

### 3.4 Nouveaux flags (duel_backtest.py ET duel_multiseed.py)
- `--nsdiff-epoch-candidates` (défaut `[40, 60, 80]`, comme TSDiff-W).
- `--nsdiff-hp-samples` (défaut `100`, comme TSDiff-W).
- Ces flags ne s'activent que sous `--include-nsdiff` ; sans lui, aucun effet.
- **Quand `--include-nsdiff` est actif mais que les nouveaux flags ne sont pas fournis**,
  garder le comportement actuel (époques fixes `NSDIFF_EPOCHS_W`) comme défaut de repli
  **OU** basculer par défaut sur le sweep — **au choix, mais documenté** ; ma recommandation :
  activer le sweep par défaut sous `--include-nsdiff` et garder une échappatoire
  `--nsdiff-fixed-epochs N` pour reproduire l'ancien run figé.

### 3.5 Artefacts de sortie
- `--out experiments/duel_backtest_nsdiff_swept.json` (nouveau fichier).
- Checkpoints NsDiff-swept dans un **nouveau** sous-dossier (ex.
  `experiments/checkpoints_nsdiff_swept/`), jamais les dossiers existants.
- `experiments/duel_backtest.json`, `duel_backtest_nsdiff.json` et `checkpoints/` vérifiés
  **inchangés** après le run (`git diff` vide dessus).

---

## 4. Rejouer + régénérer la note

1. Lancer le duel multi-graines swept (commande §7).
2. Régénérer une note **`NOTE_duel_nsdiff_swept.md`** (nouveau fichier, ne pas écraser
   `NOTE_duel_nsdiff.md` — on garde la trace du run figé pour la transparence), en miroir de
   `NOTE_duel_nsdiff.md`, qui doit :
   - Publier le **tableau des époques NsDiff-W sélectionnées par (graine, actif)**
     (traçabilité §2).
   - **Recalculer le tableau de CV inter-graines** NsDiff (swept) vs TSDiff, et dire
     explicitement si l'écart de stabilité **survit** à l'égalisation du budget d'époques.
     C'est LE livrable de ce chantier.
   - **Ré-répondre aux 3 questions** (SPA vs GARCH ; NsDiff vs TSDiff CRPS ; MCS) et indiquer
     si quoi que ce soit a changé vs le run figé — attendu : CRPS-win et 0/75 GARCH stables,
     mais le confirmer, pas le supposer.
   - Comparer explicitement, case par case si utile, **swept vs fixe** (le delta CRPS moyen
     et le delta de CV), et **déclarer honnêtement** le verdict même s'il affaiblit l'avantage
     de stabilité de NsDiff.
   - Conserver la section **Limites déclarées**, en retirant celle du budget d'époques (résolue)
     et en ajoutant toute nouvelle limite introduite par ce chantier (ex. sweep incrémental vs
     fits indépendants).

---

## 5. Tests

- Ajouter/étendre un test (dans `experiments/`, façon `test_duel_*.py`) qui vérifie que la
  sélection d'époques NsDiff **ne lit jamais le bloc de test** (verrou E1) — p. ex. en
  vérifiant que `select_epochs` pour NsDiff n'est alimenté que par des scores calculés sur
  `val_pos`. S'inspirer des tests d'`epoch_sweep`/`duel_origins` existants.
- Test additif : les nouveaux flags par défaut ne changent pas les artefacts d'un run **sans**
  `--include-nsdiff`.
- Suite complète verte (491 + nouveaux).

---

## 6. Critères d'acceptation (definition of done)

- [ ] `--nsdiff-epoch-candidates` / `--nsdiff-hp-samples` (+ échappatoire `--nsdiff-fixed-epochs`)
      ajoutés à `duel_backtest.py` et `duel_multiseed.py`.
- [ ] NsDiff-W passe par un sweep **val-only** (verrou E1 prouvé par test), `select_epochs`
      réutilisé tel quel, sweep TSDiff-W **inchangé**.
- [ ] Tableau des époques sélectionnées par (graine, actif) produit.
- [ ] Run swept écrit dans `duel_backtest_nsdiff_swept.json` + checkpoints isolés ; artefacts
      existants **inchangés** (`git diff` vide dessus).
- [ ] `NOTE_duel_nsdiff_swept.md` : CV recalculé, comparaison swept-vs-fixe explicite,
      3 questions ré-répondues, verdict honnête.
- [ ] Décision sweep incrémental vs fits indépendants **déclarée**.
- [ ] `pytest` entièrement vert ; branche `feat/nsdiff-epoch-sweep` prête pour PR.
- [ ] Résumé de 5 lignes en fin de run avant la note complète.

---

## 7. Ordre d'exécution + commande

1. Lire `epoch_sweep.py` (décider chemin §3.2), `duel_backtest.py` (~142-159),
   `nsdiff_weekly.py` (incrémental ? §3.3).
2. Câbler le sweep NsDiff (additif) + flags ; `pytest`.
3. Lancer :
   ```
   python experiments/duel_multiseed.py --seeds 42 43 44 45 46 --m-samples 500 \
       --skip-global --include-nsdiff \
       --nsdiff-epoch-candidates 40 60 80 --nsdiff-hp-samples 100 \
       --end 2026-07-29 --out experiments/duel_backtest_nsdiff_swept.json
   ```
   (job long → checkpoints isolés, reprenable ; réutilise les checkpoints TSDiff/classiques
   existants en lecture seule s'ils sont présents.)
4. Régénérer `NOTE_duel_nsdiff_swept.md` (§4) ; résumé 5 lignes ; PR.

---

## 8. Pièges à éviter (spécifiques à ce repo)

- **Verrou E1** : ne jamais laisser la sélection d'époques toucher `test_pos`. C'est
  l'erreur qui invaliderait tout le chantier.
- **Ne pas écraser** `duel_backtest.json` / `duel_backtest_nsdiff.json` / `checkpoints/`.
- **Ne pas modifier** la branche TSDiff de `_sweep_one_model` (au plus un paramètre optionnel
  rétro-compatible) — le sweep TSDiff doit rester bit-identique.
- **Honnêteté du verdict** : si le sweep n'améliore pas (ou dégrade) la stabilité de NsDiff,
  ou si les époques sélectionnées convergent vers 40 partout, **le dire** — le but est
  d'égaliser la comparaison, pas de fabriquer un avantage.

*Fichiers compagnons : `NOTE_duel_nsdiff.md` (run à époques fixes), `SYNTHESE_duel_nsdiff`
(2 pages), `BRIEF_integration_nsdiff.md` (intégration initiale).*
