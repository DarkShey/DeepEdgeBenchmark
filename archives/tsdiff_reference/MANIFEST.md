# MANIFEST — TSDiff, référence gelée (mise en veille)

## Contexte

Consigne tuteur (2026-08-07, 15h00) : « Mettre TsDiff en veille : le sortir des runs
courants mais archiver un run comme point de comparaison de référence ». Exécuté via
`BRIEF_tsdiff_veille.md` (racine du dépôt). Fait suite au duel de comparaison
`20260806_Detailed_and_Extended_Comparison_TsDiff_vs_NsDiff.pdf` (2026-08-06) et à la
note de décision qui en découle (retrait de production, borne haute « point »
conservée).

- **Date de gel** : 2026-08-07
- **Commit `git rev-parse HEAD` au moment du gel** : `334775ae4b0c95e8ebc8885259272d8aa4487df9`
- **Ce répertoire n'a été alimenté que par copie** — aucun fichier n'a été déplacé ni
  supprimé ailleurs dans le dépôt ; aucun nouveau run TSDiff n'a été exécuté pour
  produire ce gel.

## Fichier archivé

`duel_backtest_nsdiff_swept.json` — copie bit-identique de
`experiments/duel_backtest_nsdiff_swept.json` (conservé en place, non déplacé).

- **SHA-256** : `0796bc05a617e90d4c167bd846a2dbe385603f0f4ec282650153fa5e0f675045`
  (identique source/copie, vérifié à l'archivage)
- **Origine** : commit `d49a8bd` (2026-08-03), « Add NsDiff epoch-swept multiseed duel
  results + comparative note » — cf. `NOTE_duel_nsdiff_swept.md` pour le rapport complet
  d'accompagnement (déjà versionné, non dupliqué ici).
- **Méta d'époques sélectionnées** : entièrement embarquées dans le JSON lui-même
  (`per_seed.<seed>.meta_by_asset.<asset>.epochs_tsdiff_w` /
  `epochs_nsdiff_w`, par graine × actif) — aucun fichier séparé n'existe pour ces
  méta, donc rien d'autre à copier.

## Pourquoi ce run et pas un autre

C'est le duel **le plus défendable comme référence** parmi les artefacts TSDiff
existants :

- **Équitable** : budget d'époques *sweepé* pour les deux modèles (TSDiff-W et
  NsDiff-W partagent le même protocole de sélection — candidats [40, 60, 80],
  sélection sur validation uniquement, verrou E1 — cf. `NOTE_duel_nsdiff_swept.md`).
  Élimine la réserve méthodologique « pas toutes choses égales par ailleurs » des
  runs à époques figées précédents (`duel_backtest_nsdiff.json`).
- **Couverture** : 5 actifs (SPY, BTC-USD, ETH-USD, ZN=F, TLT) × 5 graines
  (42, 43, 44, 45, 46) × 3 horizons hebdomadaires (W+1, W+2, W+3) = 75 cellules
  scorées, plus 2 graines (42, 43) en variante *global* (sélection pooled).
- Les autres artefacts TSDiff (`backtest_rolling_tsdiffw*.json`,
  `tsdiff_monthly_multiseed.json`) restent des runs **secondaires**, laissés en place
  dans `experiments/` mais non retenus comme référence.

## Budgets figés (tels qu'ils apparaissent dans le JSON, `config`)

| Paramètre | Valeur |
|---|---|
| Actifs | SPY, BTC-USD, ETH-USD, ZN=F, TLT |
| Graines (per-asset) | 42, 43, 44, 45, 46 |
| Graines (global, pooled) | 42, 43 |
| `n_val` (origines de validation) | 12 |
| `n_test` (origines de test) | 30 |
| `m_samples` (échantillons de scoring) | 500 |
| `k_denoise` | 20 |
| `n_boot` (bootstrap) | 2000 |
| Fenêtre de données | 2015-01-01 → 2026-07-29 |
| Temps d'exécution du run | 144.5 s |
| **TSDiff (par actif)** — époques | candidats 40/60/80, sélection sur validation par actif (verrou E1) |
| **NsDiff (par actif)** — époques | candidats [40, 60, 80], sélection sur validation par graine × actif (verrou E1), `hp_samples=100` |
| **TSDiff (global)** — époques | candidats 40/60/80, sélection *pooled* sur validation de tous les actifs (2 graines déclarées) |

Architecture TSDiff-W sous-jacente (`models/tsdiff_weekly.py`, non modifiée par ce
gel) : `SEQ_LEN_W=26` semaines de lookback, `HORIZON_W=3` (W+1/W+2/W+3),
`N_SAMPLES_W=100` échantillons par prévision — la sélection d'époques par
actif × graine (embarquée dans le JSON) prime sur `EPOCHS_W=60` par défaut.

Époques TSDiff-W effectivement sélectionnées par le sweep (extrait, cf. JSON
`per_seed.<seed>.meta_by_asset.<asset>.epochs_tsdiff_w` pour le détail complet
5 graines × 5 actifs) : majoritairement 40, avec des paliers à 60/80 selon
l'actif et la graine — cf. tableau équivalent côté NsDiff dans
`NOTE_duel_nsdiff_swept.md` (même sweep, même protocole).

## Verdict associé (gelé avec ce run)

- **Calibration dominée par NsDiff** : Winkler favorable à NsDiff sur 29/30 cellules
  (weekly + daily confondus).
- **TSDiff au mieux à égalité** sur le point en daily — et seulement au prix d'un
  intervalle de confiance effondré (largeur anormalement faible, pas un vrai gain de
  précision).
- **TSDiff s'effondre en régime peu doté** (peu de données/peu d'itérations
  d'entraînement) — instabilité inter-graines nettement supérieure à NsDiff
  (CV moyen TSDiff = 9.7% vs NsDiff-swept = 2.7% sur les 15 cases actif × horizon
  de ce run, cf. `NOTE_duel_nsdiff_swept.md`).
- **Décision** : retrait de production ; la borne haute « point » de TSDiff reste
  documentée/conservée comme référence historique, pas comme contendant actif.

## Renvois

- Note de décision / origine : `20260806_Detailed_and_Extended_Comparison_TsDiff_vs_NsDiff.pdf`
  (racine du dépôt).
- Brief d'exécution de cette mise en veille : `BRIEF_tsdiff_veille.md` (racine du dépôt).
- Rapport détaillé du run archivé : `NOTE_duel_nsdiff_swept.md` (racine du dépôt).
- Glossaire des métriques (CRPS, Winkler, MCS, SPA, …) : `documentation/Glossaire.pdf`.

## Réactivation

Cette mise en veille est réversible par construction — rien n'a été supprimé :

- `benchmarks/multi_horizon.py` : décommenter l'entrée `"TSDiff": forecast_horizons_tsdiff`
  dans `MODEL_ADAPTERS` (le builder `forecast_horizons_tsdiff` n'a pas été touché).
- `experiments/audit_coverage.py` : réintégrer `"TSDiff"` dans `MODELS`.
- `model_artifacts/pipeline.py` : réintégrer `"TSDiff"` dans `MODELS`.

Aucune ligne `oos` TSDiff n'a été supprimée de `validation/tracking.db` — l'historique
reste intact et consultable indépendamment de ce gel.
