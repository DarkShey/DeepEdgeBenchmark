# Audit de `validation/tracking.db` — table `predictions`

Audit en lecture seule (aucune suppression, aucune correction). Effectué sur une **copie** de la base. Branche `maeva/audit-tracking-db`.

## Résumé chiffré

- **Total lignes `predictions`** : 14174
- **Doublons (lignes excédentaires) — live** : 0
- **Doublons (lignes excédentaires) — oos** : 9875
- **Total lignes qui seraient supprimées (dry-run)** : 9875
- **Lignes Prophet OOS évaluées (y_true renseigné)** : 2496 (sur 2496 lignes Prophet OOS au total)
- **Erreur moyenne Prophet OOS** : +7.75% vs **Prophet live** : +7.78%

## A. Inventaire

### Par `source`

| source | n lignes | run_id distincts | cutoff min | cutoff max | évaluées | en attente |
|---|---:|---:|---|---|---:|---:|
| live | 200 | 5 | 2026-07-06 | 2026-07-11 | 89 | 111 |
| oos | 13974 | 106 | 2026-01-21 | 2026-07-10 | 13974 | 0 |

### Par `(source, model, asset, horizon)`

| source | model | asset | horizon | n lignes | run_id distincts | cutoff min | cutoff max | évaluées | en attente |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| live | ARIMA-GARCH | BTC-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | ARIMA-GARCH | BTC-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | ARIMA-GARCH | ETH-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | ARIMA-GARCH | ETH-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | ARIMA-GARCH | SPY | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | ARIMA-GARCH | SPY | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | ARIMA-GARCH | TLT | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | ARIMA-GARCH | TLT | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | ARIMA-GARCH | ZN=F | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | ARIMA-GARCH | ZN=F | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | LSTM | BTC-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | LSTM | BTC-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | LSTM | ETH-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | LSTM | ETH-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | LSTM | SPY | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | LSTM | SPY | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | LSTM | TLT | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | LSTM | TLT | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | LSTM | ZN=F | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | LSTM | ZN=F | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Naive | BTC-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | Naive | BTC-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | Naive | ETH-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | Naive | ETH-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | Naive | SPY | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Naive | SPY | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Naive | TLT | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Naive | TLT | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Naive | ZN=F | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Naive | ZN=F | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Prophet | BTC-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | Prophet | BTC-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | Prophet | ETH-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | Prophet | ETH-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | Prophet | SPY | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Prophet | SPY | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Prophet | TLT | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Prophet | TLT | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | Prophet | ZN=F | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | Prophet | ZN=F | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | SARIMA | BTC-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | SARIMA | BTC-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | SARIMA | ETH-USD | 1 | 3 | 3 | 2026-07-06 | 2026-07-11 | 3 | 0 |
| live | SARIMA | ETH-USD | 7 | 3 | 3 | 2026-07-06 | 2026-07-11 | 1 | 2 |
| live | SARIMA | SPY | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | SARIMA | SPY | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | SARIMA | TLT | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | SARIMA | TLT | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | SARIMA | ZN=F | 1 | 4 | 4 | 2026-07-06 | 2026-07-10 | 2 | 2 |
| live | SARIMA | ZN=F | 7 | 4 | 4 | 2026-07-06 | 2026-07-10 | 1 | 3 |
| live | TSDiff | BTC-USD | 1 | 2 | 2 | 2026-07-09 | 2026-07-11 | 2 | 0 |
| live | TSDiff | BTC-USD | 7 | 2 | 2 | 2026-07-09 | 2026-07-11 | 0 | 2 |
| live | TSDiff | ETH-USD | 1 | 2 | 2 | 2026-07-09 | 2026-07-11 | 2 | 0 |
| live | TSDiff | ETH-USD | 7 | 2 | 2 | 2026-07-09 | 2026-07-11 | 0 | 2 |
| live | TSDiff | SPY | 1 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| live | TSDiff | SPY | 7 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| live | TSDiff | TLT | 1 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| live | TSDiff | TLT | 7 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| live | TSDiff | ZN=F | 1 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| live | TSDiff | ZN=F | 7 | 2 | 2 | 2026-07-09 | 2026-07-10 | 0 | 2 |
| oos | ARIMA-GARCH | BTC-USD | 1 | 492 | 3 | 2026-01-25 | 2026-07-10 | 492 | 0 |
| oos | ARIMA-GARCH | ETH-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | ARIMA-GARCH | SPY | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | ARIMA-GARCH | TLT | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | ARIMA-GARCH | ZN=F | 1 | 452 | 4 | 2026-01-21 | 2026-07-09 | 452 | 0 |
| oos | LSTM | BTC-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | LSTM | ETH-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | LSTM | SPY | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | LSTM | TLT | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | LSTM | ZN=F | 1 | 452 | 4 | 2026-01-21 | 2026-07-09 | 452 | 0 |
| oos | Naive | BTC-USD | 1 | 492 | 3 | 2026-01-25 | 2026-07-10 | 492 | 0 |
| oos | Naive | ETH-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | Naive | SPY | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | Naive | TLT | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | Naive | ZN=F | 1 | 452 | 4 | 2026-01-21 | 2026-07-09 | 452 | 0 |
| oos | Prophet | BTC-USD | 1 | 492 | 3 | 2026-01-25 | 2026-07-10 | 492 | 0 |
| oos | Prophet | ETH-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | Prophet | SPY | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | Prophet | TLT | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | Prophet | ZN=F | 1 | 452 | 4 | 2026-01-21 | 2026-07-09 | 452 | 0 |
| oos | SARIMA | BTC-USD | 1 | 492 | 3 | 2026-01-25 | 2026-07-10 | 492 | 0 |
| oos | SARIMA | ETH-USD | 1 | 656 | 4 | 2026-01-23 | 2026-07-10 | 656 | 0 |
| oos | SARIMA | SPY | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | SARIMA | TLT | 1 | 448 | 4 | 2026-01-22 | 2026-07-09 | 448 | 0 |
| oos | SARIMA | ZN=F | 1 | 452 | 4 | 2026-01-21 | 2026-07-09 | 452 | 0 |
| oos | TSDiff | BTC-USD | 1 | 328 | 2 | 2026-01-26 | 2026-07-10 | 328 | 0 |
| oos | TSDiff | ETH-USD | 1 | 328 | 2 | 2026-01-26 | 2026-07-10 | 328 | 0 |
| oos | TSDiff | SPY | 1 | 224 | 2 | 2026-01-27 | 2026-07-09 | 224 | 0 |
| oos | TSDiff | TLT | 1 | 224 | 2 | 2026-01-27 | 2026-07-09 | 224 | 0 |
| oos | TSDiff | ZN=F | 1 | 226 | 2 | 2026-01-26 | 2026-07-09 | 226 | 0 |

### Conformité des formats de `run_id` (§5)

Tous les `run_id` respectent l'un des deux formats attendus (`run_YYYYMMDDThhmmss` pour live, `YYYYMMDD-MODEL-ASSET-D<h>` pour oos). La règle de récence du §5 est donc **applicable telle quelle**.

## B. Doublons détectés

- Nombre de **groupes** en doublon (clé métier avec >1 ligne) : live=0, oos=3998
- Nombre de **lignes excédentaires** (celles qui seraient supprimées) : live=0, oos=9875, **total=9875**

Extrait des 25 groupes de doublons les plus copiés (clé métier = source, model, asset, horizon, cutoff_date, target_date) :

| source | model | asset | h | cutoff_date | target_date | n_copies | run_ids | y_pred (par copie) | spread | stdev |
|---|---|---|---:|---|---|---:|---|---|---:|---:|
| oos | ARIMA-GARCH | ZN=F | 1 | 2026-01-27 | 2026-01-28 | 4 | 20260707-ARIMA-ZN=F-D1, 20260709-ARIMA-ZN=F-D1, 20260710-ARIMA-ZN=F-D1, 20260712-ARIMA-ZN=F-D1 | 111.79, 111.79, 111.79, 111.78 | 0.0123 | 0.0049 |
| oos | LSTM | ZN=F | 1 | 2026-01-27 | 2026-01-28 | 4 | 20260707-LSTM-ZN=F-D1, 20260709-LSTM-ZN=F-D1, 20260710-LSTM-ZN=F-D1, 20260712-LSTM-ZN=F-D1 | 111.83, 111.81, 111.77, 111.81 | 0.064 | 0.0231 |
| oos | Naive | ZN=F | 1 | 2026-01-27 | 2026-01-28 | 4 | 20260707-Naive-ZN=F-D1, 20260709-Naive-ZN=F-D1, 20260710-Naive-ZN=F-D1, 20260712-Naive-ZN=F-D1 | 107.95, 111.80, 111.80, 111.80 | 3.8459 | 1.6653 |
| oos | Prophet | ZN=F | 1 | 2026-01-27 | 2026-01-28 | 4 | 20260707-Prophet-ZN=F-D1, 20260709-Prophet-ZN=F-D1, 20260710-Prophet-ZN=F-D1, 20260712-Prophet-ZN=F-D1 | 112.38, 111.98, 111.81, 111.74 | 0.6466 | 0.2507 |
| oos | SARIMA | ZN=F | 1 | 2026-01-27 | 2026-01-28 | 4 | 20260707-SARIMA-ZN=F-D1, 20260709-SARIMA-ZN=F-D1, 20260710-SARIMA-ZN=F-D1, 20260712-SARIMA-ZN=F-D1 | 111.79, 111.80, 111.80, 111.79 | 0.0072 | 0.0033 |
| oos | ARIMA-GARCH | ETH-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-ARIMA-ETH-USD-D1, 20260709-ARIMA-ETH-USD-D1, 20260710-ARIMA-ETH-USD-D1, 20260712-ARIMA-ETH-USD-D1 | 3009.46, 3009.32, 3009.34, 3009.40 | 0.1336 | 0.0532 |
| oos | ARIMA-GARCH | SPY | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-ARIMA-SPY-D1, 20260709-ARIMA-SPY-D1, 20260710-ARIMA-SPY-D1, 20260712-ARIMA-SPY-D1 | 692.79, 692.80, 692.82, 692.79 | 0.0335 | 0.0129 |
| oos | ARIMA-GARCH | TLT | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-ARIMA-TLT-D1, 20260709-ARIMA-TLT-D1, 20260710-ARIMA-TLT-D1, 20260712-ARIMA-TLT-D1 | 85.96, 85.96, 85.96, 85.63 | 0.3318 | 0.1433 |
| oos | ARIMA-GARCH | ZN=F | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-ARIMA-ZN=F-D1, 20260709-ARIMA-ZN=F-D1, 20260710-ARIMA-ZN=F-D1, 20260712-ARIMA-ZN=F-D1 | 111.64, 111.65, 111.65, 111.65 | 0.0118 | 0.0043 |
| oos | LSTM | BTC-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-LSTM-BTC-USD-D1, 20260709-LSTM-BTC-USD-D1, 20260710-LSTM-BTC-USD-D1, 20260712-LSTM-BTC-USD-D1 | 89570.20, 89569.67, 90053.73, 90503.28 | 933.6094 | 388.3046 |
| oos | LSTM | ETH-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-LSTM-ETH-USD-D1, 20260709-LSTM-ETH-USD-D1, 20260710-LSTM-ETH-USD-D1, 20260712-LSTM-ETH-USD-D1 | 2985.60, 2978.25, 2989.43, 3009.12 | 30.8694 | 11.4218 |
| oos | LSTM | SPY | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-LSTM-SPY-D1, 20260709-LSTM-SPY-D1, 20260710-LSTM-SPY-D1, 20260712-LSTM-SPY-D1 | 689.92, 687.32, 691.60, 688.61 | 4.2876 | 1.5886 |
| oos | LSTM | TLT | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-LSTM-TLT-D1, 20260709-LSTM-TLT-D1, 20260710-LSTM-TLT-D1, 20260712-LSTM-TLT-D1 | 86.01, 86.00, 86.11, 85.55 | 0.5623 | 0.2172 |
| oos | LSTM | ZN=F | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-LSTM-ZN=F-D1, 20260709-LSTM-ZN=F-D1, 20260710-LSTM-ZN=F-D1, 20260712-LSTM-ZN=F-D1 | 111.80, 111.77, 111.73, 111.78 | 0.0637 | 0.0232 |
| oos | Naive | ETH-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Naive-ETH-USD-D1, 20260709-Naive-ETH-USD-D1, 20260710-Naive-ETH-USD-D1, 20260712-Naive-ETH-USD-D1 | 2873.74, 3006.61, 3006.61, 3006.61 | 132.8669 | 57.5331 |
| oos | Naive | SPY | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Naive-SPY-D1, 20260709-Naive-SPY-D1, 20260710-Naive-SPY-D1, 20260712-Naive-SPY-D1 | 667.95, 691.74, 691.74, 691.74 | 23.7964 | 10.3041 |
| oos | Naive | TLT | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Naive-TLT-D1, 20260709-Naive-TLT-D1, 20260710-Naive-TLT-D1, 20260712-Naive-TLT-D1 | 83.04, 85.99, 85.99, 85.66 | 2.9583 | 1.24 |
| oos | Naive | ZN=F | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Naive-ZN=F-D1, 20260709-Naive-ZN=F-D1, 20260710-Naive-ZN=F-D1, 20260712-Naive-ZN=F-D1 | 106.71, 111.64, 111.64, 111.64 | 4.9336 | 2.1363 |
| oos | Prophet | ETH-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Prophet-ETH-USD-D1, 20260709-Prophet-ETH-USD-D1, 20260710-Prophet-ETH-USD-D1, 20260712-Prophet-ETH-USD-D1 | 2720.87, 2742.07, 2676.74, 2698.46 | 65.3364 | 24.4223 |
| oos | Prophet | SPY | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Prophet-SPY-D1, 20260709-Prophet-SPY-D1, 20260710-Prophet-SPY-D1, 20260712-Prophet-SPY-D1 | 689.75, 688.30, 688.38, 688.88 | 1.4564 | 0.5794 |
| oos | Prophet | TLT | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Prophet-TLT-D1, 20260709-Prophet-TLT-D1, 20260710-Prophet-TLT-D1, 20260712-Prophet-TLT-D1 | 85.71, 85.41, 85.58, 85.35 | 0.3636 | 0.1437 |
| oos | Prophet | ZN=F | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-Prophet-ZN=F-D1, 20260709-Prophet-ZN=F-D1, 20260710-Prophet-ZN=F-D1, 20260712-Prophet-ZN=F-D1 | 112.35, 111.94, 111.76, 111.69 | 0.6556 | 0.2545 |
| oos | SARIMA | ETH-USD | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-SARIMA-ETH-USD-D1, 20260709-SARIMA-ETH-USD-D1, 20260710-SARIMA-ETH-USD-D1, 20260712-SARIMA-ETH-USD-D1 | 3006.41, 3006.16, 3006.17, 3006.17 | 0.2526 | 0.1071 |
| oos | SARIMA | SPY | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-SARIMA-SPY-D1, 20260709-SARIMA-SPY-D1, 20260710-SARIMA-SPY-D1, 20260712-SARIMA-SPY-D1 | 691.92, 691.91, 691.91, 691.92 | 0.0168 | 0.0069 |
| oos | SARIMA | TLT | 1 | 2026-01-28 | 2026-01-29 | 4 | 20260707-SARIMA-TLT-D1, 20260709-SARIMA-TLT-D1, 20260710-SARIMA-TLT-D1, 20260712-SARIMA-TLT-D1 | 85.97, 85.98, 85.98, 85.62 | 0.3608 | 0.1558 |

*(table complète : 3998 groupes en doublon — voir `validation/audit/audit_keep_drop.csv` pour le détail ligne par ligne)*

Sur les 3998 groupes en doublon : **176** ont des `y_pred` strictement identiques entre copies (vrais re-runs stables), **3822** présentent une divergence de valeurs entre copies (le modèle a produit un résultat différent selon le run — cohérent avec des runs Prophet/LSTM/TSDiff non déterministes ou des refits à des instants différents).

### Cause racine des doublons

Confirmée par l'inventaire : la table live n'a **aucun** doublon détecté (protégée par `UNIQUE (tc_id, model, cutoff_date)`). Tous les doublons proviennent de `source='oos'`. L'index partiel `idx_predictions_oos_unique` inclut `run_id` dans sa clé d'unicité — or chaque backtest (re-run de `sim_trades.py`/pipeline sur les mêmes dates historiques) génère un `run_id` différent (préfixe date du jour d'exécution du backtest, pas de la donnée). Deux backtests exécutés à des dates différentes mais portant sur les **mêmes** `cutoff_date`/`target_date` historiques ne sont donc jamais reconnus comme doublons par la contrainte SQL, et s'empilent. C'est l'hypothèse du brief (§3), **confirmée** par les données observées.

## C. Règle « garder le dernier run » — keep/drop (dry-run)

Règle appliquée (§5) : par clé métier `(source, model, asset, horizon, cutoff_date, target_date)`, on garde la ligne dont le `run_id` est le plus récent (préfixe date extrait pour live et oos), départage par `created_at` puis `id` max. **Aucune suppression n'a été effectuée** — voir `validation/audit/audit_keep_drop.csv` pour le détail exhaustif (une ligne par enregistrement, colonne `decision`).

- Lignes `keep` : 4299
- Lignes `drop` : 9875

## D. Diagnostic Prophet OOS

### Constat chiffré (depuis la base)

| source | n évaluées | erreur moyenne | erreur médiane | écart-type | min | max | |err|>10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| live | 17 | 7.78% | 1.33% | 15.85% | -11.67% | 38.85% | 5 |
| oos | 2496 | 7.75% | 0.49% | 20.4% | -32.47% | 93.7% | 795 |

Comparaison avec les autres modèles OOS (même colonne d'erreur), pour confirmer que l'anomalie est **propre à Prophet** :

| model (oos) | n évaluées | erreur moyenne |
|---|---:|---:|
| ARIMA-GARCH | 2496 | 0.19% |
| LSTM | 2660 | 2.81% |
| Naive | 2496 | 0.08% |
| Prophet | 2496 | 7.75% |
| SARIMA | 2496 | 0.17% |
| TSDiff | 1330 | -0.05% |

**Ventilation par actif (Prophet OOS)** — l'erreur moyenne globale (+7.75%) masque une forte hétérogénéité : elle **n'est pas uniforme sur tous les actifs**, contrairement à l'hypothèse §6.4 du brief (« systématique... toutes les lignes Prophet OOS »). L'anomalie est concentrée sur les cryptos :

| asset | n évaluées | erreur moyenne | min | max |
|---|---:|---:|---:|---:|
| ETH-USD | 656 | 21.77% | -32.47% | 93.7% |
| BTC-USD | 492 | 11.75% | -8.25% | 38.62% |
| ZN=F | 452 | -0.04% | -2.37% | 2.3% |
| SPY | 448 | -0.46% | -6.84% | 8.88% |
| TLT | 448 | -1.09% | -4.23% | 4.19% |

**ETH-USD** (+21.8% en moyenne) et **BTC-USD** (+11.8%) portent la quasi-totalité du biais ; **SPY, TLT, ZN=F** sont quasi neutres (entre -1.1% et -0.04% en moyenne). Le même calcul restreint aux lignes gagnantes après dédoublonnage (`decision=keep`) donne des chiffres très proches (ETH +22.4%, BTC +10.2%, autres ≈0), donc les doublons ne créent pas cet écart. Ce constat renforce le diagnostic « fit unique + extrapolation de tendance batch » : BTC-USD et ETH-USD ont connu une tendance haussière marquée sur la période d'entraînement de chaque backtest, que Prophet (`growth='linear'` par défaut) extrapole indéfiniment sur toute la fenêtre de test batch ; SPY/TLT/ZN=F, plus proches d'un régime stationnaire/latéral sur la même période, ne présentent pas ce biais de tendance — cohérent avec une erreur de **dérive de tendance non réactualisée**, pas avec un bug de colonne ou d'échelle qui affecterait tous les actifs uniformément.

### Cause racine (preuve code + données)

**Ce n'est pas un problème de colonne ni d'échelle.** La colonne lue par `sim_trades.py` (`predicted` → `y_pred`) est bien `forecast["yhat"]`, la même colonne brute utilisée côté live (`model_artifacts/pipeline.py` → `benchmarks/multi_horizon.py:143`, `results[h] = (float(row["yhat"]), ...)`). Aucune transformation log/cap n'est appliquée dans un chemin et pas l'autre — `Prophet(...)` est instancié avec les **mêmes hyperparamètres** dans les deux chemins (`growth` par défaut = `'linear'`, pas de `cap`/`floor`, `weekly_seasonality=True`, `yearly_seasonality=True`, `daily_seasonality=False`).

La vraie cause est **architecturale** :

- **live** (`model_artifacts/pipeline.py:795`) : `cutoff_date = full_series.index[-1].date()` — le pipeline live **réentraîne Prophet chaque jour** sur toutes les données connues jusqu'à hier, et ne prédit que l'horizon D1 (1 jour). C'est un walk-forward implicite au rythme du cron quotidien.

- **oos** (`models/prophet_model.py:108-128`, fonction `run_prophet`) : le docstring du fichier l'indique explicitement (`models/prophet_model.py:9`) : *"fitted once on the training prices, then predicting the test dates in one batch"*. Le modèle est **entraîné une seule fois** sur `train`, puis `model.predict(df_future)` est appelé **une seule fois** sur la totalité de la fenêtre de test (165 lignes dans l'exemple `Run/20260712-Prophet-BTC-USD-D1/`), soit plusieurs mois d'horizon réel.

- `validation/sim_trades.py:build_oos_prediction_rows` (lignes ~363-390) prend ensuite chaque ligne `t` du parquet et lui assigne `cutoff_date = date[t-1]`, `target_date = date[t]`, **comme si** chaque prédiction était un walk-forward 1-jour frais — alors qu'en réalité, pour Prophet, toutes ces lignes proviennent du **même** fit unique réalisé au début de la fenêtre de test. Le modèle n'est jamais reconditionné sur les prix réellement observés entre-temps.

- Preuve dans les données : tous les autres modèles OOS (`ARIMA-GARCH`, `SARIMA`, `LSTM`, `TSDiff`, `Naive`) implémentent explicitement un **walk-forward 1-step** (cf. docstrings `models/*.py` : *"Forecasting is walk-forward (rolling 1-step-ahead)"*, et code type `models/sarima_model.py:112` / `models/lstm_model.py:154` / `models/arima_model.py:164` qui ajoutent la valeur réalisée à l'historique avant l'étape suivante). **`prophet_model.run_prophet` est le seul à ne pas le faire** (`models/prophet_model.py:108-128`). C'est la seule différence structurelle entre Prophet et les modèles sains, et elle correspond exactement au modèle en anomalie.

- Extrait `Run/20260712-Prophet-BTC-USD-D1/predictions.parquet` (165 lignes) : l'erreur oscille entre -4% et +22% selon que le prix réel repasse sous ou au-dessus de la tendance Prophet extrapolée depuis le fit unique — ce n'est pas une dérive monotone avec l'index temporel (corrélation ≈ 0.06), ce qui est cohérent avec une **courbe de tendance figée** (issue d'un seul fit) comparée à un prix réel volatile, plutôt qu'avec un bug d'échelle constant.


**Conclusion** : l'anomalie Prophet OOS vient du fait que `run_prophet()` (`models/prophet_model.py`) fait un **fit unique + predict batch** sur toute la fenêtre de test, alors que l'ingestion `sim_trades.py` **étiquette** ces prédictions comme des cutoff→target walk-forward 1-jour (même sémantique que le live). Le décalage entre l'étiquetage (implicitement 1-day-ahead) et la réalité du calcul (extrapolation à plusieurs mois depuis un fit figé) explique le biais, **et non un bug de colonne ou d'échelle** : ce biais se manifeste surtout sur les actifs qui avaient une tendance haussière marquée pendant la fenêtre d'entraînement (BTC-USD, ETH-USD), et est quasi nul sur les actifs plus stationnaires (SPY, TLT, ZN=F) — signature typique d'une extrapolation de tendance non réactualisée, pas d'une erreur de transformation systématique qui toucherait tous les actifs de façon identique.

**Recommandation pour le brief de correction** : soit (a) faire de `run_prophet` un walk-forward réel (refit ou reconditionnement à chaque pas, comme les autres modèles), soit (b) si le batch est conservé pour des raisons de coût de calcul, corriger l'étiquetage OOS pour refléter le véritable horizon (date de fit → date cible) plutôt que `date[t-1] → date[t]`, et documenter que la comparaison Prophet OOS vs live n'est alors pas à iso-horizon. Option (a) est recommandée pour rester cohérent avec les autres modèles et avec le live.

## Hors périmètre de cet audit

Conformément au brief, aucune suppression, aucun correctif de code, aucune modification de contrainte d'unicité n'a été effectué ici. Ces actions feront l'objet d'un brief de correction séparé, à partir des conclusions ci-dessus.
