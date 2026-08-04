# NOTE — NsDiff en entier (daily + weekly) sur le dashboard D7/W1, `source='oos'`

*2026-08-04 — `experiments/oos_nsdiff_daily_weekly.py`, run réel, 4/5 actifs (TLT
exclu, voir §3), 0.7 min. Suite de `BRIEF_nsdiff_dashboard_daily_oos.md`, ferme
le trou identifié dans `NOTE_backtest_rolling_nsdiffw.md` §6 (« NsDiff ne va pas
apparaître dans ce dashboard aujourd'hui »).*

## 1. Ce qui a été produit

Deux briques, `validation/tracking.db`, `source='oos'`, `model='NsDiff'`,
`horizon_unit` ∈ {W+1, W+2, W+3} :

- **Brique A (le vrai nouveau travail) — NsDiff daily, régime B** :
  `frequence='daily'`, `horizon_type='weekly'`. Walk-forward du NsDiff daily
  (`nsdiff_model.fit_nsdiff`/`forecast_from_fitted`, horizon=15 jours de bourse,
  comme TSDiff-D), prévision lue à la **distance en jours de bourse réelle**
  jusqu'à la cible-vendredi (`epoch_sweep.week_targets`, jamais +7j calendaires
  en dur).
- **Brique B — NsDiff weekly, régime C, en `oos`** : `frequence='weekly'`.
  Même mécanisme, horizon=3 semaines, sur les mêmes origines. Le weekly
  existait déjà en `source='live'`/`'backtest_rolling_nsdiff'` — jamais en
  `'oos'`, donc invisible du dashboard (`matrice_paired_tests.load_predictions`
  ne lit que `source='oos'`) ; c'est corrigé ici.

**1080 + 1080 = 2160 lignes insérées** (4 actifs × 90 origines × 3 horizons ×
2 régimes). `backfill_eval_metrics.py` (existant, réutilisé tel quel) a ensuite
peuplé `abs_error`/`in_interval`/etc. sur exactement ces 2160 lignes.

## 2. Origines : réutilisées verbatim, pas régénérées

Non-négociable du brief : mêmes `(asset, cutoff_date, target_date)` que les 6
modèles déjà en `oos`. Concrètement :

- **Régime C** : `backtest_rolling_tsdiffw.load_baseline_triplets` (fonction
  existante, importée telle quelle — déjà utilisée par
  `backtest_rolling_nsdiffw.py` pour la même raison) — lit directement les
  triplets des 5 baselines paramétriques dans la DB, aucun recalcul de split.
- **Régime B** : `load_baseline_triplets_daily` (nouvelle fonction, miroir de
  la précédente pour `frequence='daily'`) — même lecture directe.

**Bug pré-existant découvert en creusant pourquoi le régime B affichait 60
cutoffs au lieu de 90 en filtrant `daily_duplicate=0`** :
`validation.tracking_db.flag_daily_duplicates()` partitionne par `(source,
model, asset, horizon, cutoff_date, target_date)` — **sans** `frequence`/
`horizon_type`. Or à W+1/2/3, le régime B et le régime C d'un même modèle
partagent exactement ce tuple (`horizon`=numéro de semaine dans les deux cas).
La fonction traite donc à tort ces deux lignes distinctes comme des doublons
l'une de l'autre et flague `daily_duplicate=1` sur celle des deux qui a le plus
petit `id`. Vérifié directement : aucun vrai doublon n'existe (`GROUP BY
(model,asset,frequence,horizon_type,horizon_unit,cutoff_date) HAVING
count(*)>1` → 0 ligne, sur tout `source='oos'` à W+1/2/3). `load_baseline_
triplets_daily` n'applique donc **pas** ce filtre (documenté dans le script) —
fichier existant `tracking_db.py` **non modifié** (hors scope, bug pré-existant
sans rapport avec cette tâche). À signaler séparément si une correction du
`PARTITION BY` est un jour souhaitée.

## 3. TLT exclu (4/5 actifs) — limite pré-existante, pas un bug introduit ici

`yfinance` (auto_adjust=True) retraite rétroactivement les dividendes de TLT :
le fetch live échoue dès le tout premier cutoff (`last_close mismatch`,
86.59 vs 86.94 attendu, ~0.4% relatif — même symptôme que
`NOTE_backtest_rolling_nsdiffw.md` §0). Bascule automatique vers le cache
offline (`experiments/offline_prices.py`, `DONNEE~1.XLS`) : les cutoffs
passent la vérification, **mais** ce cache se termine pile au dernier cutoff
réutilisé (2026-07-02) — aucune marge pour les target_date W+2/W+3 des
dernières origines (~2026-07-23), d'où un `IndexError` capturé proprement :
TLT est **skip **(logué, documenté dans
`experiments/oos_nsdiff_daily_weekly_summary.json`), le run continue sur les
4 autres actifs (non-négociable du brief §7 : un actif ne doit pas faire
échouer tout le run). **Ne pas conclure sur la classe obligations** à partir
de ZN=F seul (brief §7) — le verdict "Obligations (taux)" du dashboard ci-
dessous ne repose que sur ZN=F pour NsDiff (dédoublonné avec TLT en amont pour
les 6 autres modèles, donc son poids dans le pooling classe "bond" est réduit
d'moitié pour NsDiff spécifiquement — lecture à prendre avec cette réserve).

## 4. Méthode : train-once-forward, miroir de TSDiff — pas réinventé

Mécanisme identique à celui qui a produit les lignes `oos` actuelles de
TSDiff (`weekly_headtohead_v2.run_pair_v2`, vérifié en traçant
`weekly_headtohead_v2_n90.json` → `backfill_weekly_predictions.py` →
`insert_oos_predictions`) : NsDiff-D et NsDiff-W sont chacun **fit UNE SEULE
FOIS** par actif, sur les données strictement antérieures à la première
origine réutilisée (`mu`/`sd` gelés à cet instant), puis chaque origine est
prévue à partir d'une fenêtre d'historique qui grandit mais ne dépasse jamais
l'origine elle-même (`weekly_z[:m]` / `daily_z[:daily_pos]`) — aucune fuite,
aucun ré-entraînement dans la boucle. C'est **délibérément différent** du
mécanisme "refit périodique tous les 15 origines" de `backtest_rolling_
nsdiffw.py` (qui cible une source isolée `backtest_rolling_nsdiff`, un autre
chantier) : ici, la consigne était de reproduire **comment TSDiff a
effectivement peuplé ses lignes `oos`**, qui est train-once, pas refit
périodique.

**Seed = 42** (déterministe, single-seed, comme les 6 modèles — pas le
multi-graines du duel, décision déjà tranchée par le brief §4).

## 5. Époques / seq_len / k_denoise — déclarés

`NSDIFF_EPOCHS = 40` pour **les deux régimes** (=
`weekly_nsdiff_production.NSDIFF_EPOCHS_W`, déjà le budget déclaré côté
weekly ; **aucun budget daily séparé n'existe nulle part dans ce dépôt** — en
inventer un nouveau aurait été un chiffre non déclaré, donc le budget weekly
existant est réutilisé tel quel pour le daily aussi, décision explicite,
cohérente avec la formulation du brief §5). `seq_len=30`, `k_denoise=20`
(défauts `nsdiff_model.py`, identiques à `weekly_nsdiff_production.py` et
`backtest_rolling_nsdiffw.py`). `n_samples=50` (aligné sur la config réelle de
`weekly_headtohead_v2_n90.json`, PAS les 500 de `backtest_rolling_nsdiffw.py`
qui sert un backtest KPI différent, plus gourmand en précision).

Coût réel mesuré : **quelques secondes de fit par régime et par actif** (le
mean-backbone de NsDiff est un petit MLP, cf. `models/nsdiff_model.py` — bien
moins cher que TSDiff). Run complet (4 actifs, 2 régimes, 90 origines
chacun) : **0.7 min**.

## 6. Verdict Daily-vs-Weekly de NsDiff

### Par cellule (test RMSE, seed interne 0, n=90/actif)

| Actif | Verdict | p |
|---|---|---|
| BTC-USD | **weekly natif significativement meilleur** | 0.0048 |
| ETH-USD | indistinguable | 0.3956 |
| SPY | indistinguable | 0.2804 |
| ZN=F | indistinguable | 0.9256 |
| TLT | — (absent, §3) | — |

### Agrégat poolé (skill sans échelle vs RW, seed=42)

| Groupe | n_origines | Skill RMSE | Skill Winkler |
|---|---|---|---|
| Global | 95 | indistinguable (p=0.169) | **weekly natif significativement meilleur** (p<0.0001) |
| Crypto | 90 | indistinguable (p=0.828) | **weekly natif significativement meilleur** (p<0.0001) |
| Actions | 90 | **daily significativement meilleur** (p=0.0002) | indistinguable (p=0.068) |
| Obligations (taux) | 91 | **daily significativement meilleur** (p=0.0034) | indistinguable (p=0.165) — voir réserve §3 (ZN=F seul) |

**Lecture** : sur le point (RMSE), NsDiff est soit indistinguable (crypto,
global) soit légèrement en faveur du daily (actions, obligations — réserve
TLT). Sur la fiabilité de l'incertitude (Winkler), NsDiff weekly-natif est
significativement meilleur globalement et sur la crypto, indistinguable
ailleurs — cohérent avec `NOTE_backtest_rolling_nsdiffw.md` §3 (NsDiff-W très
bien calibré, proche du nominal 95%) : le weekly natif produit des bandes plus
fiables que le daily poussé multi-step, même quand le point seul ne tranche
pas. Aucun arbitrage global net daily-vs-weekly ne se dégage à travers toutes
les classes — la comparaison intra-modèle est plus nuancée que pour les
classiques.

## 7. Recoupement avec les notes précédentes

**Calibration standalone de NsDiff-W** (régime C, sur les 2160 lignes
insérées) : Cov95 pooled ≈ **92.5%/91.9%/93.3%** (W+1/W+2/W+3) — proche du
nominal 95%, cohérent avec `NOTE_backtest_rolling_nsdiffw.md` §3 (pooled
95%/95%/95%, protocole prod identique — refit périodique au lieu de
train-once, `n_samples=500` au lieu de 50, ce qui explique l'écart de
quelques points, pas une divergence de fond) **et** avec
`NOTE_compare_weekly_tsdiff_nsdiff.md` §1 (Cov95 duel : 95%/94%/91% — même
histoire qualitative de bonne calibration, mais définition différente, cf.
`NOTE_backtest_rolling_nsdiffw.md` §5 : `nsdiff_model.fit_nsdiff` seq_len=30
ici et en prod, `nsdiff_weekly.fit_weekly` seq_len=26 dans le duel — dette
déjà déclarée ailleurs, pas réintroduite ici). Recoupement **directionnel**,
pas chiffre-à-chiffre, pour cette raison déjà documentée avant ce brief.

## 8. Non-négociables — statut

- Origines identiques aux 6 modèles : **oui**, lues verbatim (§2).
- Point-in-time : `mu`/`sd` gelés au train, régime C resample W-FRI + dropna
  (`build_weekly`, réutilisé), régime B à la vraie distance jours-de-bourse
  (`week_targets`, réutilisé) : **oui**.
- Insertion `oos` idempotente : **oui** (`insert_oos_predictions`, upsert
  existant, réutilisé tel quel).
- Lignes des 6 modèles + `backtest_rolling*`/`live` NsDiff intactes : **vérifié
  par comptage avant/après** (diff exacte : uniquement 6 nouveaux groupes
  `NsDiff`/`oos`, tout le reste byte-identique).
- Aucun fichier source existant modifié hors ajouts : **oui** (1 seul fichier
  nouveau, `experiments/oos_nsdiff_daily_weekly.py`).
- pytest vert avant ET après : **oui** (models, honest_eval, experiments,
  model_artifacts, validation, benchmarks — tous verts après le run).

## 9. Vérification finale

`python experiments/dashboard_d7_w1.py` : **34 cellules** (6 modèles × 5
actifs + NsDiff × 4 actifs, TLT exclu pour NsDiff seulement — 35 n'est
atteignable que si TLT est un jour réintégré, §3). NsDiff présent dans les
cellules (`n=90` chacune) **et** dans l'agrégat poolé (contribue à
`n_contributions` de chaque classe). Page `dashboard_d7_w1.html` régénérée,
autonome, JSON embarqué validé (parse + comptage des modèles ok).

## 10. Limites déclarées

- TLT absent pour NsDiff (4/5 actifs) — §3.
- Single-seed (42), déterministe — décision du brief, pas le protocole
  multi-graines du duel.
- `n_samples=50` (vs 500 pour `backtest_rolling_nsdiffw.py`) — quantiles
  95% un peu plus bruités, cohérent avec l'écart de calibration observé au §7.
- Époques daily (40) non sélectionnées sur une grille propre au daily —
  réutilise le budget weekly déjà déclaré, pas un nouveau sweep (§5).
- Bug `flag_daily_duplicates` (partition sans `frequence`/`horizon_type`,
  §2) découvert mais non corrigé — hors scope, fichier partagé non modifié.
