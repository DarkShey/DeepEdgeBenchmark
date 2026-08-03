# NOTE — NsDiff dans le duel multi-graines (miroir de `experiments/NOTE_duel_diffusion_vs_classiques.md`)

*2026-08-03. Fichiers : `experiments/duel_backtest.py::run_nsdiff_records`/
`build_grid_analysis_with_nsdiff` (§4.4 de `BRIEF_integration_nsdiff.md`),
`experiments/duel_multiseed.py --include-nsdiff` (agrégation inter-graines
inchangée : `aggregate_crps_dispersion`, `aggregate_mcs_stability`,
`aggregate_holm_stability`, `aggregate_pooled_stability`, `aggregate_spa_stability`).
Run réel : **5 actifs (SPY, BTC-USD, ETH-USD, ZN=F, TLT), S = 5 graines
(42, 43, 44, 45, 46)**, `n_val=12`, `n_test=30`, `m=500`, `k_denoise=20`,
`--end 2026-07-29` (identique au run 6-modèles), `--skip-global` (comparaison
entraînement global vs par-actif hors périmètre pour NsDiff, cf. brief).*

## Protocole et isolation des artefacts (garde-fous respectés)

- **Le duel 6-modèles TSDiff/ARIMA-GARCH/SARIMA/Prophet/LSTM/Naive n'a pas été
  rejoué** : les 25 checkpoints (graine × actif) déjà scorés dans
  `experiments/checkpoints/seed{N}_{actif}.json` ont été **relus tels quels**
  (`duel_multiseed.load_or_run_asset`, inchangé, qui n'écrit jamais si le
  fichier existe déjà). Vérifié avant lancement : avec les mêmes `n_val=12,
  n_test=30, start=2015-01-01, end=2026-07-29, embargo=défaut`,
  `duel_origins.build_common_origins` reproduit EXACTEMENT les mêmes
  `train_end`/`val_origins`/`test_origins` que ceux enregistrés dans les
  checkpoints existants (vérifié sur SPY) — les CRPS de TSDiff/classiques dans
  ce nouveau fichier sont donc **bit-identiques** à ceux de
  `experiments/duel_backtest.json` (vérifié : CV TSDiff sur BTC-USD =
  4.4/10.3/12.0%, identique à la NOTE existante).
- **NsDiff a été calculé à neuf**, checkpointé dans un sous-dossier dédié
  (`experiments/checkpoints_nsdiff/seed{N}_{actif}_nsdiff.json`, jamais le même
  fichier que les checkpoints existants), via un monkeypatch runtime de
  `duel_multiseed.checkpoint_path_nsdiff` dans le script de lancement (pas une
  modification du code source : par défaut cette fonction écrit déjà dans
  `experiments/checkpoints/seed{N}_{actif}_nsdiff.json` -- même dossier que les
  classiques mais nom de fichier distinct, donc déjà sans collision possible ;
  le sous-dossier séparé est une précaution supplémentaire prise pour ce run,
  cf. §Reproduire). Résultat écrit dans un fichier distinct
  (`experiments/duel_backtest_nsdiff.json`), jamais
  `experiments/duel_backtest.json`.
- `experiments/duel_backtest.json` et `experiments/checkpoints/` (25 fichiers)
  vérifiés **inchangés** après le run (`git diff` vide, comptage de fichiers
  identique). Aucun fichier source modifié (`git status` ne montre que les deux
  nouveaux artefacts + cette note). Suite pytest inchangée et verte
  (491 tests : models 88, honest_eval 14, experiments 139, model_artifacts 44,
  validation 188, benchmarks 18).
- Budget NsDiff : époques fixes déclarées (`NSDIFF_EPOCHS_W=40`, non
  sélectionnées sur validation/test, brief §4.4), `m=500` identique à tous les
  modèles. Coût CPU de la partie NsDiff (25 combinaisons graine×actif, sur une
  machine où le run 6-modèles original avait pris ~9h24) : **95 secondes**
  (cf. `config.elapsed_s` de `duel_backtest_nsdiff.json`) — cohérent avec le
  ×23 de vitesse déjà mesuré sur 1 actif dans `NOTE_nsdiff_vs_tsdiff.md`.

## Réponses aux 3 questions (dans l'ordre demandé)

### 1) NsDiff vs GARCH(1,1) sous SPA : premier modèle à battre GARCH significativement ?

**Non — toujours 0 rejet sur 75 tests** (15 cases × 5 graines), strictement
identique au chiffre déjà établi sans NsDiff. Le test SPA (Hansen 2005) est un
test **joint** sur l'ensemble des challengers (SARIMA, Prophet, LSTM, Naive,
TSDiff, NsDiff) : ajouter NsDiff au groupe ne fait basculer aucune des 75
cases de "non-rejet" à "rejet".

Signal directionnel (non significatif, à ne pas confondre avec le verdict
SPA) : sur le **gain moyen de CRPS par modèle vs GARCH** (`per_model_mean_gain
_vs_benchmark`), NsDiff est le modèle qui gagne le plus souvent (à égalité
avec SARIMA) — **gain positif sur 28/75 cases**, et il est le
**meilleur candidat de la case** (gain le plus élevé parmi les 6 challengers)
sur **16/75 cases**, contre 17 pour SARIMA, 7 pour LSTM, 3 pour TSDiff, 0 pour
Prophet, et 32 pour Naive (qui reste le candidat le plus fréquemment
"meilleur", cohérent avec le rejet SPA jamais atteint : le candidat le plus
souvent gagnant n'est pas celui qui gagne fort et systématiquement). Les gains
NsDiff les plus marqués sont concentrés sur **BTC-USD/ETH-USD, surtout aux
graines 45/46 et aux horizons W2/W3** (ex. BTC-USD W2/W3, graine 45 :
gain = 198/254 ; graine 46 : gain = 70/116) — mais ce sont des graines/cases
isolées, pas une tendance stable sur les 5 graines (cf. §Q2/Q3 ci-dessous et
le tableau de dispersion) : **NsDiff n'est pas le premier modèle à battre
GARCH(1,1) significativement, le rejet SPA reste à 0/75 dans les deux runs.**

### 2) NsDiff vs TSDiff en CRPS (la paire diffusion-vs-diffusion) : signe et significativité stables ?

**Le signe est stable sur 14/15 combinaisons graine×horizon (5 graines × 3
horizons), toujours en faveur de NsDiff** (`mean_diff` = TSDiff − NsDiff,
échelle MASE, positif = TSDiff moins bon) :

| Horizon | Graine 42 | Graine 43 | Graine 44 | Graine 45 | Graine 46 |
|---|---|---|---|---|---|
| W1 | +0.103 (ns) | **+0.344 (sig)** | +0.115 (ns) | +0.075 (ns) | +0.138 (ns) |
| W2 | +0.057 (ns) | **+1.076 (sig)** | **+0.278 (sig)** | **+0.368 (sig)** | **+0.304 (sig)** |
| W3 | −0.022 (ns) | **+1.409 (sig)** | **+0.522 (sig)** | **+0.676 (sig)** | **+0.357 (sig)** |

**Verdict poolé (`aggregate_pooled_stability["TSDiff vs NsDiff"]`)** : W1 sig
1/5 (43), **W2 sig 4/5 (43,44,45,46)**, **W3 sig 4/5 (43,44,45,46)** — le même
"decrochage à W2/W3, robuste sur 4 graines/5" que celui déjà observé pour
`TSDiff vs ARIMA-GARCH`, sauf qu'ici c'est **NsDiff qui gagne**, pas un
classique. Seule la graine 42 (la graine historique de
`BRIEF_baselines_fortes.md`) donne un signe quasi-nul/inversé et non
significatif partout — **exactement le même schéma que le "graine 42 la plus
favorable à TSDiff" documenté dans `NOTE_duel_diffusion_vs_classiques.md`,
ici transposé : la graine 42 est aussi celle qui minimise l'écart NsDiff-
TSDiff.**

En CRPS brut moyenné sur les 5 graines, NsDiff a un CRPS moyen **strictement
inférieur à TSDiff sur les 15 cases sur 15**, sans exception (BTC-USD/ETH-USD/
SPY/TLT/ZN=F × W1/W2/W3) — un balayage complet, cohérent avec le contrôle 1-
actif/1-graine de `NOTE_nsdiff_vs_tsdiff.md`.

**Conclusion Q2 : oui, robuste.** Le signe (NsDiff meilleur que TSDiff en
CRPS) est stable sur 14 des 15 combinaisons graine×horizon (la seule exception
est minuscule et non significative, graine 42/W3), et la significativité
poolée est acquise sur 4 des 5 graines à W2 et W3 (jamais à W1, où l'écart
existe en signe mais reste trop petit pour être significatif hors graine 43).

### 3) MCS : NsDiff plus souvent que TSDiff dans le MCS (crypto, horizons longs) ? CV inter-graines de NsDiff ?

**Mitigé, pas un "NsDiff partout meilleur" — la réponse dépend de l'actif crypto :**

| Actif | Horizon | TSDiff (fraction MCS) | NsDiff (fraction MCS) |
|---|---|---|---|
| BTC-USD | W1 | **1.0** | 0.6 |
| BTC-USD | W2 | **1.0** | 0.8 |
| BTC-USD | W3 | 0.8 | 0.8 |
| ETH-USD | W1 | 0.8 | **1.0** |
| ETH-USD | W2 | 0.6 | **1.0** |
| ETH-USD | W3 | 0.6 | **1.0** |

- **Sur ETH-USD, l'hypothèse se confirme nettement** : c'est précisément là où
  TSDiff décrochait déjà dans le duel original (0.6-0.8) que NsDiff reste
  **parfaitement stable (1.0 sur les 3 horizons, y compris W2/W3)** — NsDiff
  n'est jamais exclu du MCS sur ETH-USD, sur aucune des 5 graines.
- **Sur BTC-USD, c'est l'inverse** : TSDiff est plus stable (1.0/1.0/0.8) que
  NsDiff (0.6/0.8/0.8) à W1/W2 — NsDiff y est exclu du MCS sur 1 à 2 graines
  sur 5 aux horizons courts (W1 surtout).
- Sur SPY/TLT/ZN=F, les deux sont à 1.0 partout (aucune différenciation) sauf
  quelques cases déjà connues pour Prophet/LSTM (non liées à NsDiff/TSDiff).
- **Moyenne sur les 15 cases** : NsDiff = 0.947, TSDiff = 0.920 — NsDiff est
  légèrement plus stable en moyenne, mais **entièrement porté par ETH-USD** ;
  sans cet actif, NsDiff serait légèrement en retrait (BTC-USD).

**CV inter-graines de NsDiff (coefficient de variation du CRPS moyen, %) :**

| Actif | W1 | W2 | W3 |
|---|---|---|---|
| BTC-USD | 3.5 | 5.0 | **5.4** |
| ETH-USD | 0.8 | 1.5 | 2.0 |
| SPY | 1.8 | 2.5 | 2.2 |
| TLT | 1.5 | 2.3 | 2.0 |
| ZN=F | 1.4 | 2.0 | 2.0 |

**Moyenne des 15 cases : NsDiff = 2.4%, TSDiff = 9.7%** — NsDiff est environ
**4× plus stable d'une graine à l'autre que TSDiff** (dont la pire case,
ETH-USD/W2, atteignait 33.9%). Fait notable et un peu contre-intuitif : la
case la moins stable de NsDiff (BTC-USD, CV jusqu'à 5.4%) est précisément
celle où TSDiff est LA PLUS stable (CV 0.5-4.4%) — et inversement, la case la
moins stable de TSDiff (ETH-USD, jusqu'à 33.9%) est celle où NsDiff est la
PLUS stable (CV ≤ 2.0%). Deux modèles avec des points faibles de
reproductibilité sur des actifs différents, pas les mêmes.

**Prudence sur l'interprétation du CV** : une partie de l'écart de stabilité
vient probablement du **budget d'époques fixe et déclaré de NsDiff**
(`NSDIFF_EPOCHS_W=40`, jamais sélectionné par graine) contre le **sweep
d'époques par graine de TSDiff-W** (candidats 40/60/80, sélection validation
par graine × actif, cf. tableau de `NOTE_duel_diffusion_vs_classiques.md`) —
une source de variance inter-graines en moins pour NsDiff, pas seulement un
modèle intrinsèquement plus stable. Ce n'est pas masqué : c'est une limite
déclarée de cette comparaison (cf. §Limites).

## Stabilité des tests appariés (Holm) impliquant NsDiff — 6 paires × 15 cases = 90

| Paire | Cases stables significatives (5/5) | Cases stables non-sig. (0/5) | Cases instables |
|---|---|---|---|
| NsDiff vs ARIMA-GARCH | 0 | 7 | **8** (seed 44 isolée sur BTC-USD, seeds 42/46 isolées sur TLT/ZN=F) |
| NsDiff vs SARIMA | 0 | 15 | 0 |
| NsDiff vs Prophet | 11 | 3 | 1 (SPY-W2) |
| NsDiff vs LSTM | 1 | 8 | 6 |
| NsDiff vs Naive | 0 | 14 | 1 |
| TSDiff vs NsDiff | 0 | 12 | 3 (toutes sur ETH-USD, cf. Q2) |

**19 des 90 cases (21%) sont instables** (proche du 28% déjà observé sur les
75 cases TSDiff-vs-classiques du duel original) — jamais masqué, le détail
graine par graine est dans `experiments/duel_backtest_nsdiff.json ->
aggregate.holm_stability`. Point notable : **`NsDiff vs Prophet` est
significatif sur 11 des 15 cases (73%), pas 15/15 comme `TSDiff vs Prophet`**
dans le duel original — sur SPY-W3 et ZN=F-W2/W3, la différence n'est pas
significative (Prophet n'y est pas aussi massivement décalibré que sur
BTC-USD/ETH-USD/TLT). **`NsDiff vs SARIMA` n'est JAMAIS significatif, sur
aucune case, aucune graine** (15/15 stables non-significatives) — NsDiff et
SARIMA sont statistiquement indiscernables en CRPS sur ce duel.
**`NsDiff vs LSTM` a un signe constant en faveur de NsDiff sur les 15
combinaisons graine×horizon** (toujours `mean_diff < 0`), mais n'atteint la
significativité pleine (5/5) que sur une seule case (SPY-W1) — direction
robuste, ampleur pas toujours suffisante.

## Verdict poolé (MASE) — paires impliquant NsDiff

| Paire | W1 | W2 | W3 |
|---|---|---|---|
| NsDiff vs ARIMA-GARCH | ns 1/5 (44) | ns 1/5 (44) | ns 1/5 (44) |
| NsDiff vs SARIMA | ns 0/5 | ns 0/5 | ns 0/5 |
| NsDiff vs Prophet | **sig 5/5** | **sig 5/5** | **sig 5/5** |
| NsDiff vs LSTM | **sig 5/5** | **sig 5/5** | sig 4/5 (43,44,45,46) |
| NsDiff vs Naive | ns 1/5 (44) | ns 0/5 | ns 1/5 (45) |
| TSDiff vs NsDiff | ns 1/5 (43) | **sig 4/5 (43,44,45,46)** | **sig 4/5 (43,44,45,46)** |

Signes (rappel, mean_diff = ligne_gauche − ligne_droite) : `NsDiff vs
ARIMA-GARCH`/`SARIMA` ont un signe **qui change de graine à graine** (positif
sur 42-44, négatif sur 45-46 — NsDiff n'est ni systématiquement meilleur ni
systématiquement pire que ces deux classiques, contrairement à TSDiff qui
était systématiquement pire que ARIMA-GARCH à W2/W3 sur 4 graines/5 dans le
duel original). `NsDiff vs Prophet`/`LSTM` : signe négatif (NsDiff meilleur)
sur toutes les graines, toutes les cases — les deux seules paires **robustes
sans exception** avec NsDiff comme gagnant, comme TSDiff vs Prophet l'était
déjà.

## Le rejet SPA vs GARCH(1,1) — confirmé inchangé

**0 rejet sur 75 tests (15 cases × 5 graines), NsDiff inclus dans le groupe de
challengers** : strictement le même chiffre que le duel original sans NsDiff.
Résultat identique, la conclusion la plus robuste du chantier duel reste
inchangée par l'ajout de NsDiff.

## Résumé factuel (à ne pas sur-interpréter)

- NsDiff bat TSDiff en CRPS sur 15/15 cases (moyenne 5 graines), signe stable
  sur 14/15 combinaisons graine×horizon, significatif poolé sur 4/5 graines à
  W2 et W3 (jamais à W1).
- NsDiff bat significativement Prophet et LSTM de façon robuste (poolé
  significatif sur 4 ou 5 graines/5, à tous les horizons) ; contre ARIMA-
  GARCH/SARIMA/Naive, le signe change de graine à graine et le verdict poolé
  n'est jamais stable-significatif (au mieux 1 graine/5 isolée, jamais 4 ou
  5/5).
- NsDiff est ~4× plus stable inter-graines que TSDiff en moyenne (CV 2.4% vs
  9.7%), mais ce n'est pas garanti actif par actif : NsDiff est le modèle le
  MOINS stable des deux sur BTC-USD précisément.
- NsDiff est plus souvent dans le MCS que TSDiff sur ETH-USD (là où TSDiff
  décrochait), mais moins souvent sur BTC-USD à W1/W2 — pas un gain uniforme
  sur "toute la crypto".
- Le rejet SPA vs GARCH(1,1) reste à 0/75, inchangé : NsDiff ne devient pas le
  premier modèle à battre GARCH significativement, malgré un signal
  directionnel notable (meilleur gain moyen ex-aequo avec SARIMA sur 28/75
  cases, concentré sur BTC-USD/ETH-USD à W2/W3 sur certaines graines).

## Limites déclarées (pas masquées)

- **Époques NsDiff fixes et déclarées** (`NSDIFF_EPOCHS_W=40`), jamais
  sélectionnées par graine/actif, contrairement au sweep 40/60/80 de TSDiff-W
  — brief §4.4 sanctionne ce choix pour le coût, mais il retire une source de
  variance inter-graines à NsDiff que TSDiff porte ; la comparaison de CV
  (§Q3) n'est donc pas parfaitement "toutes choses égales par ailleurs".
- **`--skip-global`** : la comparaison entraînement global vs par-actif
  (brief `BRIEF_multigraines.md` §2.3) n'a pas été reconduite pour NsDiff —
  hors périmètre demandé pour ce chantier.
- Le rejet SPA reste basé sur `n_boot=2000` (comme le duel original) ; les
  gains directionnels de NsDiff (§Q1) ne sont PAS des résultats significatifs,
  seulement une lecture qualitative du candidat le plus proche de battre le
  benchmark.
- **19 des 90 cases Holm impliquant NsDiff sont instables** — nommées ci-
  dessus, ne pas les lire comme "significatif" ou "non significatif" sans
  préciser la graine.
- Comparaison strictement à 5 graines (42-46), comme le duel original — pas
  de garantie hors de cet échantillon de graines.

## Reproduire

```
python experiments/duel_multiseed.py --seeds 42 43 44 45 46 --m-samples 500 \
    --skip-global --include-nsdiff --end 2026-07-29 \
    --out experiments/duel_backtest_nsdiff.json
```

Cette commande seule reproduit tout, avec le code tel qu'il est actuellement
sur `feat/nsdiff-model` : les checkpoints TSDiff/classiques existants dans
`experiments/checkpoints/` sont relus tels quels si présents (jamais
réécrits, `duel_multiseed.load_or_run_asset` inchangé) ; les checkpoints
NsDiff sont écrits dans `experiments/checkpoints/seed{N}_{actif}_nsdiff.json`
par défaut (`duel_multiseed.checkpoint_path_nsdiff`, nom de fichier distinct
des checkpoints classiques, donc sans collision même dans le même dossier).

Le run de cette note est allé un cran plus loin par précaution : un petit
script de lancement a monkeypatché `checkpoint_path_nsdiff` en runtime (sans
toucher au fichier source) pour écrire dans le sous-dossier séparé
`experiments/checkpoints_nsdiff/` à la place -- une isolation supplémentaire,
pas requise pour la non-collision mais plus lisible/auditable. Pour
l'obtenir avec la commande ci-dessus telle quelle, ajouter avant l'appel à
`main()` :
```python
import duel_multiseed as dm
from pathlib import Path
d = Path("experiments/checkpoints_nsdiff"); d.mkdir(parents=True, exist_ok=True)
dm.checkpoint_path_nsdiff = lambda seed, asset: d / f"seed{seed}_{asset}_nsdiff.json"
```

Reprise sur interruption : dans les deux cas, relancer la même commande
saute automatiquement toute combinaison (graine, actif) déjà checkpointée,
classique ou NsDiff.
