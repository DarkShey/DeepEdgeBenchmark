# HANDOFF — Suivi calibration σ : robustesse multi-fenêtres, Prophet, LSTM, adoptions prod

**Date : 2026-07-30. Statut : terminé (adoptions branchées dans `models/*.py`, tests verts).**
Suite directe de `HANDOFF_dist_options_comparison.md` (comparaison options 1/2/3 du stagiaire,
fenêtre unique 2020-2024). Ce document traite les six questions restées ouvertes : robustesse
multi-fenêtres, décision d'adoption, investigation Prophet, σ variable LSTM, incohérence CRPS
gaussien, et le branchement réel en production.

---

## 1. Méthode

- **Données hors-ligne** : `experiments/offline_prices.py` lit `DONNEE~1.XLS` (Close quotidien
  des 5 actifs, 2018-02 → 2026-07), vérifié **identique à yfinance** (écart max 4e-6 % sur 736
  dates communes vs `Run/*/prices.parquet`). Tous les runs sont donc reproductibles sans réseau.
- **Fenêtres** : W1 = 2020-01-01→2024-12-31 (référence du stagiaire), W2 = 2018-03-01→2022-12-31,
  W3 = 2022-01-01→2026-06-30 (l'éval de W3 n'a jamais été vue par aucune comparaison antérieure).
  Même protocole partout : test 15 %, split calib/éval chronologique 40 %, seed 42.
- **Exécution par tranches** : les backtests sont découpés en unités reprables avec checkpoint
  JSON (`robustness_windows.py`, `prophet_sigma_investigation.py --window/--configs`,
  `lstm_sigma_variants.py --window`) — le découpage SARIMA/Prophet est **exact** (refit à chaque
  pas sur l'historique complet, donc trancher ne change rien au résultat).
- **Réconciliation** : le re-run W1 reproduit le run du stagiaire à **179/180 couvertures
  identiques** (seul écart : 0,89 pt sur une variante CQR). L'infrastructure est validée.
- **Métrique** : MACE = moyenne des |couverture − cible| sur 50/80/95 %. ⚠️ Deux agrégations
  existent : `mace_strict` (moyenne inter-actifs des erreurs absolues) et `mace_loose`
  (erreur absolue de la couverture moyenne inter-actifs — c'est celle de
  `summarize_dist_options.py`). **La loose laisse les erreurs de signe opposé s'annuler entre
  actifs** (BTC sur-couvre, SPY sous-couvre → « calibré » en moyenne) : les chiffres du rapport
  du stagiaire (ex. student-t 1,32) sont donc flatteurs. Tout ce qui suit est en **strict**.

## 2. Robustesse : les gagnants statiques de W1 ne généralisent PAS

MACE strict, moyenne 5 actifs (`experiments/robustness_W{1,2,3}_results.json`) :

| Modèle | Variante | W1 | W2 | W3 |
|---|---|---|---|---|
| ARIMA-GARCH | normal | 4,49 | 6,79 | 4,50 |
| ARIMA-GARCH | student_t (post-hoc) | 3,27 | 6,28 | 4,58 |
| ARIMA-GARCH | **native_skewt** | **3,0*** | **4,04** | **3,02** |
| ARIMA-GARCH | native_ged | 3,1* | 4,42 | 2,85 |
| SARIMA | normal | 8,07 | 11,66 | 10,09 |
| SARIMA | cqr (gagnant W1) | 4,43 | 11,49 | 8,92 |
| SARIMA | ged | 7,45 | 12,03 | 5,92 |
| Naive | normal | 9,26 | 12,83 | 10,54 |
| Naive | ged (gagnant W1) | 7,81 | 12,78 | 5,97 |
| Naive | cqr | 3,70 | 10,90 | 8,02 |

\* natives W1 recalculées du run du stagiaire, même protocole.

Lecture : **chaque fenêtre a un « gagnant » statique différent** pour SARIMA/Naive (CQR en W1,
rien en W2, GED en W3). Adopter GED ou CQR sur la foi de W1 aurait été une erreur. La seule
option robuste sur les 3 fenêtres est **ARIMA-GARCH natif skew-t/GED** — logique : c'est le seul
modèle dont le σ est déjà dynamique (GARCH). Le « natif ≈ post-hoc pour 800× le coût » de W1 ne
tient pas non plus (le natif est nettement meilleur en W2/W3) ; en prod le surcoût réel est ~1-2 s
par backtest complet, négligeable au rythme d'un refit/jour.

## 3. La vraie cause commune : le NIVEAU de σ ne suit pas les régimes

Le mécalibrage dominant de SARIMA/Naive/LSTM/Prophet est une erreur de **niveau** de σ qui bouge
avec les régimes de volatilité — aucun échange de forme de queue (option 1) ni décalage additif
figé (CQR) ne peut le corriger durablement. Le correctif validé partout : **σ dynamique par EWMA
causale** (RiskMetrics, λ=0,94), en niveau (résidus) ou en correction multiplicative
(σ'ₜ = σₜ·√EWMA(z²), z = résidu standardisé — préserve la dynamique propre du modèle).

MACE strict cross-fenêtres (`experiments/dynamic_sigma_variants_results.json`,
`lstm_sigma_variants_results*.json`) :

| Modèle | Meilleur statique (cross-W) | σ dynamique (cross-W) |
|---|---|---|
| SARIMA | cqr 8,28 | **scaled_ewma+student_t 3,49** (normal 3,96) |
| Naive | cqr 7,54 | **ewma+student_t 3,45** (normal 4,27) |
| LSTM | frozen+cqr ~7-9 | **ewma 3,1-6,1 selon fenêtre** (meilleure famille sur les 3) |

- **LSTM** : l'EWMA règle le « σ figé » **sans MDN** : MACE frozen→ewma = 10,3→3,9 (W1),
  15,6→6,1 (W2), 10,3→3,1 (W3). À comparer au MDN raté (9,23, instable, +23 % de coût). La tête
  de distribution (normal/student-t/GED/CQR) n'a pas de gagnant stable → normal conservé en prod.
- La même conclusion vaut pour Naive/SARIMA : la famille EWMA est robuste, la forme ne l'est pas.

## 4. Prophet : cause racine identifiée et corrigée

`experiments/prophet_sigma_investigation*.json` (phases A/B/C, 3 fenêtres × SPY/BTC) :

1. **Phase A** : à h=1 l'intervalle Prophet ≈ ±z·σ_obs, et σ_obs ≡ std des résidus **in-sample**
   (identiques à 4 décimales). Ce n'est PAS une constante trop petite dans l'absolu (1,5-3× le
   bruit journalier) — mais l'erreur 1-pas **out-of-sample** est bien plus grosse que les résidus
   in-sample d'un trend flexible.
2. **Phase B** : le ratio σ/erreur réalisée est piloté par `changepoint_prior_scale` (0,43-0,72
   au défaut ; >1 à cps=0,001 mais RMSE 4× pire ; pire encore à cps=0,5). Le sur-ajustement
   in-sample du trend piecewise-linéaire est la cause racine. Deuxième cause : σ homoscédastique
   en espace **prix** (ignore que l'erreur suit le niveau de prix — flagrant sur BTC).
3. **Correctifs validés** : `log(price)` améliore les 6 cas fenêtre×actif (MACE 13-40 → 4-19,
   RMSE aussi) ; la **correction EWMA multiplicative** (même mécanisme que SARIMA) ferme le
   reste : **log+EWMA = MACE 1,1-7,1 partout** (vs 13-40 en prod). Le rescale scalaire fixe
   (phase C, ~1,08) ne suffit pas : le déficit de σ varie dans le temps.

## 5. Adoptions branchées dans `models/*.py` (défauts changés, legacy conservé)

| Modèle | Adoption (défaut) | Retour arrière |
|---|---|---|
| `arima_model.py` | `GARCH_DIST="skewt"` (natif, backtest **et** `next_step` qui utilisait encore Z_95 normal ; CLI `--dist`) | `dist="normal"` |
| `sarima_model.py` | `calibrate_sigma="ewma"` (correction multiplicative causale ; `next_step` via les innovations standardisées du filtre de Kalman) | `calibrate_sigma="off"` |
| `naive_model.py` | `sigma_mode="ewma"` (EWMA des variations 1 jour observées) | `sigma_mode="frozen"` |
| `lstm_model.py` | `sigma_mode="ewma"` (EWMA des résidus walk-forward ; réseau et prévision ponctuelle inchangés) | `sigma_mode="frozen"` |
| `prophet_model.py` | `log_space=True` + `calibrate_sigma="ewma"` (backtest) ; `next_step` : `log_space=True` + param `sigma_scale` à alimenter par l'appelant | `log_space=False`, `calibrate_sigma="off"` |

⚠️ `next_step_prophet` ne peut PAS s'auto-corriger (les résidus in-sample sous-estiment l'erreur
OOS — c'est la cause racine). Le paramètre `sigma_scale` attend √EWMA(z²) calculé sur les
prédictions trackées (tracking.db) : **intégration pipeline à faire** (cf. §8).

## 6. Incohérence CRPS gaussien : corrigée côté reporting

- `honest_eval/metrics.py` : nouvelles fonctions `crps_parametric` (normal / student_t forme
  fermée / GED / lognormal forme fermée Baran & Lerch / ppf custom), `crps_student_t`,
  `crps_lognormal`, `pit_parametric`. `crps_gaussian`/`pit_values` inchangés (compat).
- `experiments/prob_kpi_common.py` : `sample_parametric` reconstruit désormais en **two-piece
  normal** (σ par côté depuis les bornes stockées — exact si bornes symétriques, respecte
  l'asymétrie sinon) et traite **Prophet en espace log** comme ARIMA-GARCH (lognormal en prix).
  Plus aucune loi asymétrique écrasée sur une gaussienne symétrique.
- `model_artifacts/generate_dashboard.py` : libellés « (gaussien) » → « (paramétrique) » avec
  description de la reconstruction.

## 7. Tests

- Nouveaux : `models/test_sigma_calibration.py` (9 tests : défauts adoptés, switches legacy,
  causalité EWMA, asymétrie skew-t, bornes log positives), `honest_eval/test_metrics_dist.py`
  (8 tests : convergences des formes fermées, conventions unit-variance, two-piece).
- Corrigé : `models/test_metrics.py` (cassé depuis le commit du stagiaire — la signature
  `pi_bounds` d'arima n'était pas prise en compte) ; `models/test_naive_model.py` (test du band
  figé → `sigma_mode="frozen"` + 2 tests EWMA).
- Verts dans le sandbox : `models/` (56), `honest_eval/test_metrics_dist.py` (8),
  `model_artifacts/test_{crps_kpis,generate_dashboard,generate_taux_utilisation}.py`,
  `experiments/test_crps_metrics.py`. **Non exécutable dans le sandbox** :
  `model_artifacts/test_pipeline.py` (`pandas_ta` indisponible via le proxy) — à lancer en local.

## 8. Reste à faire (connu, non traité ici)

1. **Intégration pipeline** : alimenter `next_step_prophet(sigma_scale=...)` depuis tracking.db
   (√EWMA(z²) des dernières prédictions réalisées) ; vérifier que les Gates KPI de
   `model_artifacts/pipeline.py` réagissent bien aux nouvelles couvertures ; lancer
   `test_pipeline.py` en local.
2. `benchmarks/multi_horizon.py` (D+7) instancie ses propres Prophet/LSTM : non migré.
3. L'agrégation `mean_abs_calibration_error` de `summarize_dist_options.py` (loose) mérite
   d'être doublée d'une version stricte pour ne plus flatter les comparaisons.
4. LSTM/Prophet σ-EWMA validés sur 3 fenêtres mais 2 actifs seulement pour Prophet (SPY/BTC) —
   étendre aux 5 si besoin avant de graver dans le marbre.
5. Le commit est **local uniquement** — push après revue.

## 9. Reproduction

```bash
# comparaison robustesse (par fenêtre, reprenable, ~7 min de calcul par fenêtre)
python experiments/robustness_windows.py --window W2 --budget-s 3600
# investigation Prophet (sweep 6 configs W1, ou base+log pour W2/W3)
python experiments/prophet_sigma_investigation.py --budget-s 3600 [--window W3 --configs base log]
# LSTM sigma variants (par fenêtre)
python experiments/lstm_sigma_variants.py --budget-s 3600 [--window W2]
# variantes sigma dynamique Naive/SARIMA (depuis les états des runs robustesse)
python experiments/dynamic_sigma_variants.py
```
Les états bruts (`*_state*.json`, `lstm_sigma_ckpt*/`) sont gitignorés ; seuls les
`*_results.json` agrégés sont versionnés.
