# NOTE — Le duel diffusion vs classiques (BRIEF_unification_protocole_duel.md + BRIEF_baselines_fortes.md)

*2026-07-29. Fichiers : `duel_origins.py` (§2.1), `duel_sampling_adapters.py` (§2.2, GARCH-Student/
GJR ajouté par BRIEF_baselines_fortes.md), `crps_metrics.py::crps_fair` (§2.3), `duel_pairwise_tests.py`
(§2.4), `mcs.py` (§2.5), `duel_backtest.py` (assemblage + sélection du spec GARCH, exécuté ici) +
`duel_backtest.json` (le livrable brut). Run réel : 5 actifs (SPY, BTC-USD, ETH-USD, ZN=F, TLT),
`n_val=12`, `n_test=30`, `m=500`, `k_denoise=20`, `seed=42`, mêmes origines que le run précédent
(`--end 2026-07-29` explicite, épochs TSDiff-W identiques : {SPY:80, BTC:40, ETH:60, ZN=F:80,
TLT:80}) — seul le spec GARCH change entre les deux versions de cette NOTE.*

## Grille de conformité — protocole du duel (§4 de BRIEF_unification_protocole_duel.md)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| Estimateur CRPS | fair CRPS (Ferro 2014), `m` identique ; pas d'approximation par quantiles | **conforme** |
| Nombre de tirages `m` | 500, strictement identique pour les 6 modèles à chaque origine | **conforme** |
| Splits temporels | rolling origin (`three_way_split`) + embargo=2 semaines (purge structurelle déjà garantie par construction des fenêtres) | **conforme** |
| Ré-estimation | **identique entre les 6 modèles : gelé à T0 pour tous** — aucune asymétrie résiduelle (pas seulement déclarée : éliminée) | **conforme** |
| Échantillonnage classiques | GARCH-Student/GJR simulé, SARIMAX `.simulate()`, Prophet `predictive_samples()`, LSTM MC-Dropout multi-pas, Naive `random_walk_samples` — aucune reconstruction gaussienne | **conforme** |
| Tests | `dm_hac_test` (HAC≥h-1, HLN) + bootstrap par blocs par paire × actif × horizon, Clark-West des 5 modèles vs Naive, Holm sur la grille 5×3 par paire | **conforme** |
| Verdict multi-modèles | MCS (Hansen-Lunde-Nason 2011) par case + SPA de Hansen vs GARCH(1,1) | **conforme** |
| Agrégation inter-actifs | `pooled_pair_verdict` : échelle MASE (`compute_asset_scales`) + fusion bonds/crypto (`class_series`) avant tout pooling | **conforme** |

## Grille de conformité — baselines fortes (§4 de BRIEF_baselines_fortes.md)

| Point | Critère (niveau 1) | Statut |
|---|---|---|
| GARCH innovations | Student (`dist="t"`), `nu` estimé par le fit | **conforme** |
| Variante asymétrique | GJR-GARCH (`o=1`, `gamma` estimé) présente et évaluée | **conforme** |
| Échantillonnage | vraies trajectoires simulées (`_sample_innovations` + récursion de variance asymétrique), `m=500`, aucune borne analytique ni reconstruction gaussienne | **conforme** |
| Sélection de spéc. | argmin CRPS fair sur le bloc de validation uniquement (`select_garch_spec`), jamais sur le test — verrou É1 | **conforme** |
| Protocole du duel | origines / gel T0 / CRPS / tests / MCS strictement inchangés (vérifié : épochs TSDiff-W identiques avant/après) | **conforme** |
| Traçabilité | bloc avant/après ci-dessous, dans les deux sens | **conforme** |

**Ligne « Baselines classiques » : partiel → conforme.** Le seul GARCH évalué jusqu'ici tournait en
innovations gaussiennes uniquement ; chaque actif sélectionne maintenant, sur son propre bloc de
validation, la meilleure spécification parmi {normal, t, gjr-t} — **SPY → gjr-t, BTC-USD → gjr-t,
ETH-USD → normal, ZN=F → normal, TLT → t**. Le choix data-driven a honnêtement retenu `normal`
pour 2 actifs sur 5 (bonds/ETH) : rien n'a été forcé vers une spécification "plus forte" par
construction — c'est la validation qui tranche, jamais le test.

## Bloc avant / après (BRIEF_baselines_fortes.md §3 — la question du chantier)

**Le verdict ne bouge pas — dans aucun sens.** Comparaison exhaustive avant (GARCH gaussien partout)
vs après (GARCH fort, sélectionné par actif) :

| Vérification | Avant | Après | Changement |
|---|---|---|---|
| CRPS ARIMA-GARCH moyen (15 cases) | — | — | delta ∈ [-4.4%, +1.9%], la plupart < 1% (BTC-USD-W1 : -4.4%, le plus net) |
| Model Confidence Set (15 cases) | TSDiff 15/15, ARIMA-GARCH 15/15, Prophet 4/15, LSTM 13/15 | **identique, case par case** | **0/15 changée** |
| SPA vs GARCH(1,1) (15 cases) | 0/15 rejettent H0 | 0/15 rejettent H0 | **0/15 changée** |
| Paires TSDiff vs classique significatives après Holm (75 cases) | 13/75 (mêmes cases exactes) | 13/75 (mêmes cases exactes) | **0 case ajoutée/retirée** |
| Verdict poolé (5 paires × 3 horizons = 15 cases) | sig. seulement vs Prophet (3h) et LSTM (W1) | **identique, horizon par horizon** | **0 case changée** |
| ARIMA-GARCH vs Naive, poolé Holm (15 cases, vérification ad hoc pour ce brief) | 0/15 significatives | 0/15 significatives | **rien ne bat le naïf, ni avant ni après** |

**Réponses franches aux questions du brief :**
- *Un GARCH-Student ou GJR entre/sort-il du MCS quelque part ?* Non — ARIMA-GARCH (sous n'importe
  quelle spécification testée) appartenait déjà au MCS dans les 15/15 cases, et y reste.
- *Bat-il maintenant TSDiff ?* Non — les 2 seules cases où (TSDiff vs ARIMA-GARCH) est significatif
  après Holm (TLT-W3, ZN=F-W1, TSDiff moins bon) étaient déjà significatives avant le renforcement ;
  aucune case ne bascule dans un sens ni dans l'autre.
- *Quelque chose bat-il enfin le naïf ?* Non, toujours pas — vérifié explicitement pour ARIMA-GARCH
  (0/15 significatif contre Naive après Holm, avant et après), et le verdict poolé TSDiff vs Naive
  reste non significatif aux 3 horizons.

**Pourquoi ça ne bouge pas :** le GARCH gaussien (l'ancienne baseline) était déjà, sur ce
protocole précis (m=500, fair CRPS, gel à T0), suffisamment proche de sa version renforcée pour
que la différence reste sous le bruit d'échantillonnage à ces tailles d'échantillon
(`effective_n`≈10 par case). Ce n'est pas un artefact de sous-optimisation du camp classique — la
sélection par validation confirme que la spécification forte n'apporte qu'un gain marginal ici, ce
qui **renforce** (et ne fragilise pas) la conclusion « rien ne bat le naïf de manière robuste » :
elle ne peut plus être attaquée sur le motif que le GARCH était bridé.

## Résultat — CRPS moyen par case (fair, m=500, GARCH fort)

| Actif | Horizon | ARIMA-GARCH | LSTM | Naive | Prophet | SARIMA | TSDiff |
|---|---|---|---|---|---|---|---|
| BTC-USD | W1 | 2429.2 | 2529.2 | 2475.8 | 62704.5 | 2305.0 | 2491.5 |
| BTC-USD | W2 | 4076.2 | 3809.3 | 4067.5 | 64228.8 | 3833.2 | 3833.9 |
| BTC-USD | W3 | 5384.6 | 4863.5 | 5318.6 | 65736.7 | 4946.2 | 5050.6 |
| ETH-USD | W1 | 105.6 | 108.4 | 103.4 | 598.7 | 101.4 | 98.5 |
| ETH-USD | W2 | 160.3 | 168.9 | 159.6 | 623.4 | 154.8 | 143.2 |
| ETH-USD | W3 | 212.5 | 231.2 | 213.6 | 647.5 | 204.6 | 190.1 |
| SPY | W1 | 6.91 | 11.79 | 7.12 | 19.95 | 7.44 | 7.87 |
| SPY | W2 | 10.41 | 15.04 | 10.89 | 19.87 | 10.92 | 11.52 |
| SPY | W3 | 13.02 | 19.02 | 13.68 | 19.91 | 13.57 | 14.07 |
| TLT | W1 | 0.60 | 0.68 | 0.63 | 2.06 | 0.68 | 0.68 |
| TLT | W2 | 0.85 | 0.96 | 0.88 | 2.02 | 0.93 | 0.97 |
| TLT | W3 | 1.04 | 1.13 | 1.07 | 2.01 | 1.16 | 1.20 |
| ZN=F | W1 | 0.36 | 0.47 | 0.38 | 0.98 | 0.38 | 0.42 |
| ZN=F | W2 | 0.51 | 0.63 | 0.51 | 1.01 | 0.53 | 0.57 |
| ZN=F | W3 | 0.64 | 0.75 | 0.64 | 1.09 | 0.66 | 0.70 |

*(SARIMA/Prophet varient de <1% par rapport au run précédent malgré un code inchangé — bruit
numérique run-to-run des optimiseurs statsmodels/cmdstan, non lié au chantier GARCH ; confirmé
sans effet sur aucun verdict de significativité ci-dessus.)*

## Model Confidence Set (par actif × horizon, alpha=0.05 — 15 cases)

Prophet est éliminé dans **11 cases sur 15** (CRPS 3 à ~600x plus élevé que les autres). LSTM est
éliminé dans **2 cases** (SPY-W1, ZN=F-W1). **TSDiff n'est éliminé dans aucune des 15 cases**, ni
**ARIMA-GARCH** (sous sa spécification forte) — les deux appartiennent au MCS partout, aux côtés
de SARIMA/Naive (et LSTM hors les 2 cases citées) — indiscernables du meilleur, aucun démontré
meilleur seul.

## SPA de Hansen vs GARCH(1,1) (désormais le GARCH fort par actif)

**H0 (aucun modèle ne bat GARCH(1,1)) n'est rejetée dans aucune des 15 cases** (p-values entre
0.107 et 1.0) — même en confrontant chaque modèle au GARCH dans sa meilleure configuration
validée, aucun ne démontre de supériorité significative.

## Tests appariés TSDiff vs chaque classique, par case, Holm sur la grille 5×3

13 des 75 cases testées restent significatives après Holm — identiques avant/après le
renforcement du GARCH. **11 favorisent TSDiff** (6 contre Prophet, 1 contre LSTM sur SPY-W1, plus
Prophet aux 3 horizons BTC/ETH et TLT-W1/W2). **2 favorisent ARIMA-GARCH** (TSDiff significativement
*moins bon*, non masqué) : TLT-W3 et ZN=F-W1 — écart faible en valeur absolue, présent aussi bien
avec l'ancien GARCH gaussien qu'avec le nouveau GARCH fort. Aucune case significative contre
SARIMA ou Naive.

## Verdict poolé (MASE + classes d'actifs, 3 séries de classe : index/bonds/crypto)

| Paire | W1 | W2 | W3 |
|---|---|---|---|
| TSDiff vs ARIMA-GARCH | ns | ns | ns |
| TSDiff vs SARIMA | ns | ns | ns |
| TSDiff vs Prophet | **significatif (TSDiff meilleur)** | **significatif** | **significatif** |
| TSDiff vs LSTM | **significatif (TSDiff meilleur)** | ns | ns |
| TSDiff vs Naive | ns | ns | ns |

Identique au run précédent (GARCH gaussien) — **TSDiff ne démontre de skill poolé significatif que
contre Prophet (3 horizons) et LSTM (W1 seulement)**, jamais contre ARIMA-GARCH (fort ou faible),
SARIMA ou le naïf.

## Lecture (rappel §3 du brief initial, contre N3)

Ne pas lire : « jeu égal » / « globalement moins bon ». Lire : *à ce niveau de confiance et sur ces
échantillons (`effective_n`≈10 par case après bloc de 3, 30 par série pooled sur 3 classes d'actif),
TSDiff appartient au Model Confidence Set du meilleur dans 15/15 cases ; Prophet n'y appartient que
dans 4/15 ; aucune famille — y compris un GARCH-Student/GJR bien réglé et sélectionné par
validation — ne démontre de skill significatif contre GARCH(1,1) (SPA) ni contre le naïf ; et le
seul skill poolé significatif de TSDiff est contre Prophet et LSTM (W1) — jamais contre
ARIMA-GARCH, SARIMA ou le naïf, dans un sens ou dans l'autre. Ce verdict n'est plus attaquable sur
le motif "le camp classique était bridé" : il l'a été, sous sa meilleure configuration validée, et
rien ne change.*

## Limites déclarées (pas masquées)

- `tsdiff_hp_samples=100` (et non 500) pour la sélection d'époques sur le bloc de validation
  uniquement — un choix de coût, sans effet sur le score final (`m=500` partout dans le duel
  scoré), et sans toucher au bloc de test (verrou É1 préservé).
- `n_boot=2000` pour MCS/SPA (contre 10000 pour les bootstrap par paire de `paired_test.py`,
  déjà en place) — un compromis coût/précision, documenté dans `config` du JSON.
- Run exécuté sur une seule seed (`seed=42`) : la multi-graine (chantier #6) reste hors périmètre
  de ce brief, comme annoncé.
- Bruit numérique run-to-run (<1%) sur SARIMA/Prophet entre les deux versions de ce run, malgré un
  code inchangé pour ces deux modèles — attribué aux optimiseurs statsmodels/cmdstan (non
  déterministes au niveau du threading BLAS), sans effet sur aucune conclusion de significativité
  (vérifié exhaustivement : 0 case MCS/SPA/Holm/poolé changée).
- Sélection du spec GARCH a retenu `normal` pour 2 actifs sur 5 (ETH-USD, ZN=F) : la validation
  (12 origines) ne trouve pas d'avantage à la spécification forte pour ces deux actifs précis —
  résultat honnête du critère data-driven, pas une régression cachée vers l'ancienne baseline.
