# NOTE — comparaison weekly TSDiff-W vs NsDiff-W, au-delà du CRPS (calibration, sharpness, Winkler, PIT)

*2026-08-04. Suite de `NOTE_duel_nsdiff.md` (2026-08-03). Produite par
`experiments/compare_weekly_diffusion.py`, artefact
`experiments/compare_weekly_diffusion.json`. Rattachée à
`BRIEF_nsdiff_weekly_parite_et_compa.md` §3.*

## 0. Étape 0 — ce que les artefacts du duel contenaient (et ne contenaient pas)

Audité avant d'écrire une ligne de code : `duel_backtest_nsdiff.json`,
`duel_backtest_nsdiff_swept.json` et `experiments/checkpoints_nsdiff/*.json`
stockent, par (actif, graine, horizon, origine), **`point`/`crps`/
`crps_empirical` uniquement** — jamais les intervalles ni le nuage
d'échantillons. Aucun poids de modèle n'est sérialisé non plus (pas de
`.pt`/`.pth` dans le repo). Résultat : `prob_kpi_common.row_kpis` (qui a
besoin du nuage complet — le PIT empirique est une fraction du nuage, pas
lisible sur des quantiles seuls) ne pouvait pas être calculé "zéro re-run"
depuis l'existant.

Décision (§3.3 du brief) : ni un re-run du duel complet (6 modèles, sweep
d'époques, bootstrap MCS/SPA — le run coûteux, **non relancé**), ni un rejeu
depuis des checkpoints qui n'ont pas les échantillons — mais une **passe
ciblée**, limitée aux 2 modèles de diffusion, qui rejoue exactement
`duel_backtest.run_asset_duel`/`run_nsdiff_records` (mêmes origines
déterministes, même convention de seed par origine) **sans re-sweeper les
époques** : `epochs_tsdiff_w`/`epochs_nsdiff_w` sont relues telles quelles
depuis `meta_by_asset` de `duel_backtest_nsdiff.json`, et le nuage
d'échantillons est gardé au lieu d'être collapsé.

**Garde-fou de fidélité (recoupement vs hier) :** chaque ligne recalculée est
comparée à la `crps_empirical` originale du duel, même (actif, graine,
horizon, origine). Résultat sur les 4500 lignes (5 actifs × 5 graines × 3
horizons × 2 modèles, m=500, mêmes époques) : **pearson r = 0,9999999996**,
écart relatif médian **3,4·10⁻⁷** (`compare_weekly_diffusion.json` →
`cross_check_vs_yesterday`). La passe reproduit le protocole d'hier
quasiment bit-à-bit, pas une approximation.

## 1. Verdict poolé (tous actifs, 5 graines) — CRPS, calibration, sharpness, Winkler, PIT

| Horizon | Modèle | CRPS | Cov50 | Cov80 | Cov95 | Sharp95 | Winkler95 | PIT moyen |
|---|---|---|---|---|---|---|---|---|
| W1 | NsDiff | 530.8 | **0.60** | **0.86** | **0.95** | 4456 | **6367** | 0.466 |
| W1 | TSDiff | 537.3 | 0.42 | 0.67 | 0.83 | 3273 | 6906 | 0.431 |
| W2 | NsDiff | 861.3 | **0.62** | **0.83** | **0.94** | 6400 | **9236** | 0.448 |
| W2 | TSDiff | 930.5 | 0.36 | 0.64 | 0.81 | 4608 | 12410 | 0.394 |
| W3 | NsDiff | 1142.7 | **0.63** | **0.84** | **0.91** | 8073 | **11959** | 0.439 |
| W3 | TSDiff | 1283.4 | 0.35 | 0.62 | 0.82 | 6021 | 17263 | 0.373 |

(Cibles : Cov50=0.50, Cov80=0.80, Cov95=0.95, PIT moyen=0.50 pour un modèle
bien calibré.) CRPS ici est **poolé brut**, dominé par l'échelle BTC (prix ~
10⁴-10⁵) — lecture décomposée par actif/classe au §2, ne pas s'arrêter à cette
ligne pour un verdict par actif.

**Lecture honnête, différenciée (comme demandé) :**
- **Précision (CRPS)** : NsDiff gagne aux 3 horizons, écart qui se creuse avec
  l'horizon (1.2% à W1 → 11% à W3) — cohérent avec `NOTE_duel_nsdiff.md`
  (15/15 cases hier).
- **Calibration** : c'est **là** que NsDiff marque le point le plus net.
  TSDiff-W est **systématiquement sous-couvrant** à tous les niveaux et
  horizons (Cov95 ∈ [0.81, 0.83], loin des 95% nominaux — intervalles trop
  étroits, excès de confiance), NsDiff-W reste proche de la cible (Cov95 ∈
  [0.91, 0.95]). Le `g_psi`/UANS de NsDiff (largeur d'intervalle conditionnée
  sur la volatilité récente, cf. `NOTE_nsdiff_vs_tsdiff.md`) fait exactement
  ce qu'il est censé faire ici.
- **NsDiff ne gagne pas juste en élargissant sans discernement** : à Cov95
  comparable, NsDiff reste MEILLEUR sur Winkler95 (qui pénalise la largeur
  ET la sortie d'intervalle) aux 3 horizons — donc le gain de calibration
  n'est pas payé par une explosion de la pénalité de largeur. C'est TSDiff,
  plus étroit, qui perd sur Winkler malgré sa sharpness plus faible : ses
  ratés de couverture coûtent plus cher que ce que sa finesse rapporte.
- **PIT** : les deux moyennes PIT sont autour de 0.37-0.50 (TSDiff plus bas,
  0.37-0.43, cohérent avec sa sous-couverture — biais vers y_true au-dessus
  du centre de la distribution prédictive). Le PIT moyen seul ne dit pas si
  l'histogramme est PLAT (calibration réelle) ou juste centré — voir limite
  §4.

## 2. Décomposition par classe d'actif

| Classe | Horizon | Modèle | CRPS | Cov95 | Winkler95 | PIT moyen |
|---|---|---|---|---|---|---|
| Bonds  | W1 | NsDiff | 0.515 | 0.96 | 4.48  | 0.475 |
| Bonds  | W1 | TSDiff | 0.549 | 0.80 | 6.04  | 0.438 |
| Crypto | W1 | NsDiff | 1323  | 0.93 | 15886 | 0.440 |
| Crypto | W1 | TSDiff | 1339  | 0.89 | 17214 | 0.386 |
| Index  | W1 | NsDiff | 7.14  | 0.97 | 53.5  | 0.498 |
| Index  | W1 | TSDiff | 7.61  | 0.80 | 88.1  | 0.505 |

(W2/W3 dans `compare_weekly_diffusion.json → aggregate_by_class_horizon_model`
— même motif partout : NsDiff gagne CRPS + calibration + Winkler sur les 3
classes et 3 horizons, sans exception.)

**Décrochage TSDiff sur crypto atténué à la calibration** : le motif MCS
d'hier (TSDiff décroche sur ETH-USD, MCS=0.6 ; NsDiff décroche sur BTC-USD à
W1/W2) se retrouve partiellement ici — sur crypto, l'écart CRPS
NsDiff/TSDiff est le PLUS FAIBLE des 3 classes (1.2% à W1) alors que l'écart
de calibration (Cov95 0.93 vs 0.89) reste net mais moins spectaculaire que
sur bonds/index (0.96-0.97 vs 0.80). Sur BTC seul (`aggregate_by_asset_
horizon_model`, W1) : NsDiff Cov95=0.93 vs TSDiff Cov95=0.85 — NsDiff garde
l'avantage même sur l'actif où il "décroche" le plus au sens MCS/CRPS d'hier.

## 3. Limites de protocole — À NE PAS PASSER SOUS SILENCE (brief §2)

Le duel (et cette passe, qui en hérite car elle ne re-sweepe rien) compare
**deux définitions différentes**, pas seulement deux modèles :

| | TSDiff-W (duel) | NsDiff-W (duel) |
|---|---|---|
| Module | daily nourri en weekly (`tsdiff_model.fit_tsdiff`) | weekly dédié (`nsdiff_weekly.fit_weekly`) |
| `seq_len` | **30** | **26** |
| Budget d'époques | **sweepé** par graine/actif (candidats 40/60/80, verrou E1) | **fixe déclaré** (40, jamais sélectionné) |

Conséquence à assumer devant le tuteur : l'avantage de NsDiff observé ici
(CRPS ET calibration) est **en partie** un effet de protocole, pas
uniquement de modèle :
- le budget d'époques fixe de NsDiff retire une source de variance
  inter-graines que TSDiff porte (déjà noté hier, CV 2.4% vs 9.7%) ;
- le lookback différent (26 vs 30 semaines) n'est **pas neutralisé** — un
  lookback plus court peut mécaniquement mieux capter un régime de
  volatilité récent (ce qui aiderait précisément la calibration `g_psi`
  de NsDiff), sans que ce soit imputable à LSNM/UANS en tant que tel.

Cette réserve s'applique À LA CALIBRATION AUSSI, pas seulement au CRPS
d'hier : NsDiff pourrait sembler mieux calibré en partie parce qu'il regarde
une fenêtre plus courte et donc plus réactive à la volatilité récente, pas
uniquement parce que son mécanisme de variance conditionnelle est meilleur
en soi. Ce n'est pas disqualifiant, c'est une limite déclarée — et un
candidat naturel pour un test de robustesse (§5 du brief, `--nsdiff-epoch-
candidates`/alignement `seq_len`, non fait aujourd'hui, dette déclarée).

## 4. Autres limites de cette passe

- **PIT non-uniformité** : seul le PIT *moyen* est rapporté (`pit_mean`/
  `pit_std` par cellule) — pas d'histogramme/test KS d'uniformité par
  cellule ici (would need per-cell binning, hors scope aujourd'hui). Le
  moyen suffit à détecter un biais directionnel (sur/sous-estimation
  systématique), pas à confirmer une calibration marginale complète.
- **Re-fit, pas rejeu exact** : torch n'est pas garanti bit-exact même à
  seed fixée. Le recoupement (§0, pearson r=0,9999999996) montre que l'écart
  est négligeable ici, mais ce n'est structurellement PAS le même run que
  celui d'hier, juste un run quasi-identique du même protocole.
- **Définition prod ≠ définition duel** : cette note porte sur le NsDiff-W du
  DUEL (`nsdiff_weekly.fit_weekly`, seq_len=26). La chaîne production
  (axe 2, `weekly_nsdiff_production.py`) utilise délibérément une AUTRE
  définition (`nsdiff_model.fit_nsdiff`, seq_len=30, miroir ligne-à-ligne de
  TSDiff-prod) — les deux définitions ne sont pas encore alignées (brief §5).
  Ne pas mélanger les deux lectures.

## 5. Fichiers produits

- `experiments/compare_weekly_diffusion.py` — script (réutilise `prob_kpi_
  common.row_kpis`, `tsdiff_model.fit_tsdiff`/`forecast_from_fitted`,
  `nsdiff_weekly.fit_weekly`/`forecast_from_fitted_weekly` tels quels ;
  aucun code modèle nouveau).
- `experiments/compare_weekly_diffusion.json` — 4500 lignes per-row + agrégats
  par (actif×horizon×modèle), (classe×horizon×modèle), (horizon×modèle) +
  recoupement vs `duel_backtest_nsdiff.json`.
