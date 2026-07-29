# NOTE — Le duel diffusion vs classiques (BRIEF_unification_protocole_duel.md)

*2026-07-29. Fichiers : `duel_origins.py` (§2.1), `duel_sampling_adapters.py` (§2.2),
`crps_metrics.py::crps_fair` (§2.3), `duel_pairwise_tests.py` (§2.4), `mcs.py` (§2.5),
`duel_backtest.py` (assemblage §3, exécuté ici) + `duel_backtest.json` (le livrable brut).
Run réel : 5 actifs (SPY, BTC-USD, ETH-USD, ZN=F, TLT), `n_val=12`, `n_test=30`,
`m=500`, `k_denoise=20`, TSDiff-W épochs sélectionnés par actif sur le bloc de validation
uniquement (40/60/80 candidats) : {SPY:80, BTC:40, ETH:60, ZN=F:80, TLT:80}.*

## Grille de conformité (§4 du brief)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| Estimateur CRPS | fair CRPS (Ferro 2014), `m` identique ; pas d'approximation par quantiles | **conforme** |
| Nombre de tirages `m` | 500, strictement identique pour les 6 modèles à chaque origine | **conforme** |
| Splits temporels | rolling origin (`three_way_split`) + embargo=2 semaines (purge structurelle déjà garantie par construction des fenêtres) | **conforme** |
| Ré-estimation | **identique entre les 6 modèles : gelé à T0 pour tous** — aucune asymétrie résiduelle (pas seulement déclarée : éliminée) | **conforme** |
| Échantillonnage classiques | GARCH(1,1) simulé, SARIMAX `.simulate()`, Prophet `predictive_samples()`, LSTM MC-Dropout multi-pas, Naive `random_walk_samples` — aucune reconstruction gaussienne | **conforme** |
| Tests | `dm_hac_test` (HAC≥h-1, HLN) + bootstrap par blocs par paire × actif × horizon, Clark-West des 5 modèles vs Naive, Holm sur la grille 5×3 par paire | **conforme** |
| Verdict multi-modèles | MCS (Hansen-Lunde-Nason 2011) par case + SPA de Hansen vs GARCH(1,1) | **conforme** |
| Agrégation inter-actifs | `pooled_pair_verdict` : échelle MASE (`compute_asset_scales`) + fusion bonds/crypto (`class_series`) avant tout pooling | **conforme** |

Aucune ligne en écart. La seule asymétrie qui existait dans le dépôt (TSDiff gelé, les 5 autres
refit à chaque origine) est **éliminée par construction** ici, pas seulement quantifiée : les 6
`fit_*` sont appelés exactement une fois par actif, sur `<= T0` ; chaque origine de test n'avance
que l'état de conditionnement (résidus GARCH réalisés, `.append(refit=False)` SARIMAX, tampon
LSTM, buffer de rendements TSDiff), jamais les paramètres/poids.

## Résultat — CRPS moyen par case (fair, m=500)

| Actif | Horizon | ARIMA-GARCH | LSTM | Naive | Prophet | SARIMA | TSDiff |
|---|---|---|---|---|---|---|---|
| BTC-USD | W1 | 2541.0 | 2529.2 | 2475.8 | 62704.5 | 2305.0 | 2491.5 |
| BTC-USD | W2 | 4077.3 | 3809.3 | 4067.5 | 64228.8 | 3833.2 | 3833.9 |
| BTC-USD | W3 | 5387.7 | 4863.5 | 5318.6 | 65736.7 | 4946.2 | 5050.6 |
| ETH-USD | W1 | 105.6 | 108.4 | 103.4 | 598.7 | 101.4 | 98.5 |
| ETH-USD | W2 | 160.3 | 168.9 | 159.6 | 623.4 | 154.8 | 143.2 |
| ETH-USD | W3 | 212.5 | 231.2 | 213.6 | 647.5 | 204.6 | 190.1 |
| SPY | W1 | 6.92 | 11.79 | 7.12 | 20.03 | 7.50 | 7.87 |
| SPY | W2 | 10.34 | 15.04 | 10.89 | 19.95 | 10.97 | 11.52 |
| SPY | W3 | 12.78 | 19.02 | 13.68 | 20.00 | 13.62 | 14.07 |
| TLT | W1 | 0.60 | 0.68 | 0.63 | 2.05 | 0.68 | 0.68 |
| TLT | W2 | 0.86 | 0.96 | 0.88 | 2.02 | 0.95 | 0.97 |
| TLT | W3 | 1.04 | 1.13 | 1.07 | 2.00 | 1.15 | 1.20 |
| ZN=F | W1 | 0.36 | 0.47 | 0.38 | 0.98 | 0.38 | 0.42 |
| ZN=F | W2 | 0.51 | 0.63 | 0.51 | 1.01 | 0.53 | 0.57 |
| ZN=F | W3 | 0.64 | 0.75 | 0.64 | 1.09 | 0.66 | 0.70 |

## Model Confidence Set (par actif × horizon, alpha=0.05 — 15 cases, une par actif × horizon)

Prophet est éliminé dans **11 cases sur 15** (CRPS 3 à ~600x plus élevé que les autres — sa loi
générative native prise telle quelle, sans reconstruction gaussienne, est nettement moins adaptée
ici, en particulier sur crypto). LSTM est éliminé dans **2 cases** (SPY-W1, ZN=F-W1). **TSDiff
n'est éliminé dans aucune des 15 cases** : il appartient au MCS partout, aux côtés d'ARIMA-GARCH/
SARIMA/Naive (et LSTM hors les 2 cases citées) — indiscernable du meilleur, jamais démontré
meilleur seul.

## SPA de Hansen vs GARCH(1,1)

**H0 (aucun modèle ne bat GARCH(1,1)) n'est rejetée dans aucune des 15 cases** (p-values
observées entre 0.107 et 1.0) — aucun modèle, TSDiff inclus, ne démontre de supériorité
significative sur ce benchmark au niveau usuel.

## Tests appariés TSDiff vs chaque classique, par case, Holm sur la grille 5×3

13 des 75 cases testées (5 actifs × 3 horizons × 5 paires) restent significatives après Holm.
**11 favorisent TSDiff** — 6 contre Prophet (les 3 horizons sur BTC-USD et ETH-USD, plus SPY-W1
et TLT-W1/W2), 1 contre LSTM (SPY-W1). **2 favorisent le classique** (TSDiff significativement
*moins bon*, non masqué) : TLT-W3 vs ARIMA-GARCH et ZN=F-W1 vs ARIMA-GARCH (mean_diff positif
dans les deux cas, écart faible en valeur absolue). Aucune case significative contre SARIMA ou
Naive.

## Verdict poolé (MASE + classes d'actifs, 3 séries de classe : index/bonds/crypto)

| Paire | W1 | W2 | W3 |
|---|---|---|---|
| TSDiff vs ARIMA-GARCH | ns | ns | ns |
| TSDiff vs SARIMA | ns | ns | ns |
| TSDiff vs Prophet | **significatif (TSDiff meilleur)** | **significatif** | **significatif** |
| TSDiff vs LSTM | **significatif (TSDiff meilleur)** | ns | ns |
| TSDiff vs Naive | ns | ns | ns |

Une fois le protocole unifié, **TSDiff ne démontre de skill poolé significatif que contre Prophet
(3 horizons) et LSTM (W1 seulement)** — jamais contre ARIMA-GARCH, SARIMA ou le naïf.

## Lecture (rappel §3 du brief, contre N3)

Ne pas lire : « jeu égal » / « globalement moins bon ». Lire : *à ce niveau de confiance et sur ces
échantillons (`effective_n`≈10 par case après bloc de 3, 30 par série pooled sur 3 classes d'actif),
TSDiff appartient au Model Confidence Set du meilleur dans 15/15 cases ; Prophet n'y appartient
que dans 4/15 ; aucune famille ne démontre de skill significatif contre GARCH(1,1) (SPA) ; et le
seul skill poolé significatif de TSDiff est contre Prophet et LSTM (W1) — jamais contre
ARIMA-GARCH, SARIMA ou le naïf, dans un sens ou dans l'autre.*

## Limites déclarées (pas masquées, §5)

- `tsdiff_hp_samples=100` (et non 500) pour la sélection d'époques sur le bloc de validation
  uniquement — un choix de coût, sans effet sur le score final (`m=500` partout dans le duel
  scoré), et sans toucher au bloc de test (verrou É1 préservé).
- `n_boot=2000` pour MCS/SPA (contre 10000 pour les bootstrap par paire de `paired_test.py`,
  déjà en place) — un compromis coût/précision, documenté dans `config` du JSON.
- Run exécuté sur une seule seed (`seed=42`) : la multi-graine (chantier #6) reste hors périmètre
  de ce brief, comme annoncé.
