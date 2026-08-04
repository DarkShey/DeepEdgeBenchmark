# NOTE — Backtest rolling-origin NsDiff-W natif (parité prod avec TSDiff-W)

*2026-08-04 — `experiments/backtest_rolling_nsdiffw.py`, run réel, 4/5 actifs
(TLT exclu — voir §0), 43s. Miroir de
`experiments/NOTE_backtest_rolling_tsdiffw.md`, per
`BRIEF_nsdiff_weekly_parite_et_compa.md` §4.4.*

## 0. TLT exclu — limite préexistante, pas un bug introduit ici

Le run réel s'arrête net sur TLT avec un `last_close mismatch` (contrôle
strict hérité de `backtest_rolling_tsdiffw.py`, tolérance 1e-6 relative).
**Vérifié avant de conclure quoi que ce soit** : le script **original**
`backtest_rolling_tsdiffw.py`, non modifié, échoue **exactement pareil**
aujourd'hui sur TLT (même écart : ~86.59 vs ~86.94, ~0.4% relatif). Cause :
`yfinance auto_adjust=True` retraite rétroactivement l'historique des
dividendes de TLT (ETF obligataire, distributions mensuelles) — le close
d'une date passée change légèrement à mesure que le temps passe depuis la
capture des baselines (`FETCH_END="2026-07-24"`, figé), pour TOUT script qui
retélécharge aujourd'hui. ZN=F (futures, pas de dividende) n'a pas ce
problème — cohérent avec l'hypothèse. Ce n'est pas quelque chose que ce
script (NsDiff) doit corriger seul, ni quelque chose à réparer en modifiant
`backtest_rolling_tsdiffw.py` (fichier existant, hors scope). **Dette
déclarée**, pas une régression NsDiff : le run réel couvre **SPY, BTC-USD,
ETH-USD, ZN=F** (4/5 actifs, 1080 triplets).

## 1. Protocole (résumé — détails dans le script)

- **1080 triplets appariés** (actif, cutoff, horizon) : 90 origines hebdo ×
  4 actifs × 3 horizons (W+1/W+2/W+3). Origines reprises **verbatim** de
  `backtest_rolling_tsdiffw.load_baseline_triplets` (mêmes cutoffs que les 5
  baselines ET que TSDiff-W lui-même — les deux backtests de diffusion sont
  directement superposables sur la même grille).
- **NsDiff-W** : réentraînement expansif périodique (tous les 15 origines, 6
  blocs/actif), identique au mécanisme TSDiff-W. `nsdiff_model.fit_nsdiff`
  (seq_len=30, module daily nourri en weekly — **même définition que
  `weekly_nsdiff_production.py`**, PAS `nsdiff_weekly.fit_weekly` seq_len=26
  utilisé dans le duel, cf. `NOTE_compare_weekly_tsdiff_nsdiff.md` §4).
  Époques **fixes déclarées = 40** (`weekly_nsdiff_production.NSDIFF_EPOCHS_W`).
  N=500 échantillons natifs de diffusion par cellule.
- **Pas de garde-fou de fuite d'epoch-selection** (différence avec TSDiff-W,
  documentée, pas oubliée) : TSDiff-W exclut ~12 dates/actif de son
  backtest parce que ses époques* sont *sélectionnées* sur un bloc de
  validation (risque de fuite si ces dates servaient aussi de test ici).
  NsDiff-W utilise un budget d'époques **fixe**, jamais sélectionné sur
  aucun split — il n'y a donc rien à guarder contre, et toutes les origines
  partagées avec les baselines sont utilisées (aucune exclusion).
- **Baselines** : ré-échantillonnage gaussien (N=500) depuis leur PI 95%
  déjà stockée — identique au protocole TSDiff-W, caveat repris mot pour
  mot : NsDiff-W tire de vrais échantillons de diffusion, les baselines des
  échantillons gaussiens paramétriques. Si NsDiff-W gagne, vérifier que ce
  n'est pas juste le handicap gaussien des baselines ; s'il perd malgré ce
  handicap potentiel, le résultat n'en est que plus solide.
- Lignes stockées `source='backtest_rolling_nsdiff'` (1080 lignes, isolées
  de `'live'`/`'oos'`/`'backtest_rolling'` — voir §4).
- Tests appariés : bootstrap par blocs par (actif, horizon) + test poolé
  (bootstrap par blocs, cluster par classe d'actif, CRPS normalisé par
  l'échelle de chaque actif), Holm sur les 5 comparaisons, séparément par
  horizon — `paired_test.py`/`pooled_analysis.py`, réutilisés tels quels.

## 2. Bug découvert et corrigé en cours de route (pas de fichier existant modifié)

`build_kpi_probabilistes.aggregate_per_cell` (réutilisée par
`backtest_rolling_tsdiffw.py`) caste la colonne `model` du résultat agrégé en
`pd.Categorical(categories=MODEL_ORDER)` où `MODEL_ORDER` liste les 6
modèles du duel (`ARIMA-GARCH, SARIMA, Prophet, Naive, LSTM, TSDiff`) —
**"NsDiff" n'y figure pas**, donc pandas silencieusement transforme toutes
les étiquettes de modèle NsDiff en `null` dans l'agrégat par cellule (les
KPI par ligne, eux, sont corrects — seul le label de l'agrégat est perdu).
Vérifié directement (pas supposé) : avant fix, `per_cell` contenait bien
72 lignes NsDiff avec les bons chiffres mais `"model": null`.

Fichier existant `build_kpi_probabilistes.py` **non modifié** (d'autres
appelants dépendent de sa liste à 6 modèles). Fix local : une fonction
miroir `aggregate_per_cell_nsdiff` dans `backtest_rolling_nsdiffw.py`,
identique sauf `categories=MODEL_ORDER + ["NsDiff"]`. Documenté dans le
script (docstring de la fonction), pas juste corrigé en silence.

## 3. Résultat 1 — Calibration NsDiff-W (standalone, indépendante de la méthode d'échantillonnage des baselines)

| Classe | Cov 50% | Cov 80% | Cov 95% |
|---|---|---|---|
| Bonds (ZN=F) | 0.54 | 0.81 | 0.95 |
| Crypto (BTC/ETH) | 0.61 | 0.84 | 0.95 |
| Index (SPY) | 0.56 | 0.82 | 0.94 |
| **Pooled (4 actifs)** | **0.58** | **0.83** | **0.95** |

Cibles : 50/80/95%. **Calibration très proche du nominal partout**, contraste
net avec TSDiff-W sur les mêmes origines/actifs (cf.
`NOTE_backtest_rolling_tsdiffw.md` : pooled 41/71/87% — sous-couvrant aux 3
niveaux). Cohérent avec `NOTE_compare_weekly_tsdiff_nsdiff.md` §1 (même
verdict de calibration, sur les origines du DUEL cette fois) : deux
protocoles indépendants (duel vs backtest rolling prod), même lecture.

## 4. Résultat 2 — CRPS vs les 5 baselines, pooled et Holm-corrigé

NsDiff-W vs chaque baseline (positif = NsDiff-W a un CRPS **plus élevé**,
donc pire ; `p_holm` = p-value bootstrap poolée, Holm-corrigée sur les 5
comparaisons, par horizon) :

| Horizon | vs ARIMA-GARCH | vs SARIMA | vs Naive | vs LSTM | vs Prophet |
|---|---|---|---|---|---|
| W+1 | +0.002 (n.s.) | +0.015 (n.s.) | +0.030 (n.s.) | **-3.334 (sig., NsDiff meilleur)** | **-8.551 (sig., NsDiff meilleur)** |
| W+2 | +0.051 (n.s.) | +0.126 (n.s.) | +0.146 (n.s.) | **-2.455 (sig., NsDiff meilleur)** | **-7.918 (sig., NsDiff meilleur)** |
| W+3 | +0.009 (n.s.) | +0.090 (n.s.) | +0.146 (n.s.) | **-2.050 (sig., NsDiff meilleur)** | **-7.528 (sig., NsDiff meilleur)** |

**NsDiff-W est statistiquement indistinguable des 3 baselines fortes**
(ARIMA-GARCH, SARIMA, Naive) **aux 3 horizons** — écarts quasi nuls, jamais
significatifs après Holm. Il **bat significativement Prophet et LSTM aux 3
horizons**. C'est un résultat sensiblement meilleur que celui de TSDiff-W
sur ces mêmes origines : TSDiff-W **perdait significativement** contre
ARIMA-GARCH/SARIMA/Naive aux 3 horizons (`NOTE_backtest_rolling_tsdiffw.md`
§Résultat 2) alors que NsDiff-W tient tête à ces mêmes baselines. À lire
avec la réserve de protocole du §5 : le budget d'époques fixe/le seq_len de
NsDiff-W ne sont pas garantis "toutes choses égales" face à TSDiff-W.

## 5. Réserves de protocole (reprises, pas nouvelles)

- **Asymétrie d'échantillonnage** (§1, caveat repris mot pour mot) : NsDiff
  natif vs baselines gaussiennes.
- **Définition prod ≠ définition duel** : ce backtest utilise
  `nsdiff_model.fit_nsdiff` (seq_len=30), PAS `nsdiff_weekly.fit_weekly`
  (seq_len=26, utilisé dans `duel_backtest_nsdiff.json`) — cf. §0 et
  `NOTE_compare_weekly_tsdiff_nsdiff.md` §4. Les deux définitions ne sont
  pas encore alignées (dette déclarée, brief §5).
- **Époques fixes, jamais sélectionnées** : contrairement à TSDiff-W (sweep
  40/60/80 par graine/actif dans le duel, et sélection validée dans
  `epoch_sweep_results.json` pour la prod), NsDiff-W tourne à 40 époques
  fixes partout. Un budget aligné (sweep NsDiff comme TSDiff) pourrait
  changer ces chiffres dans un sens ou l'autre — non fait aujourd'hui.
- **TLT exclu** (§0) — le tableau §3/§4 porte sur 4/5 actifs, pas 5. À
  recalculer une fois le problème de dérive `yfinance` sur TLT contourné
  (hors scope de ce brief).

## 6. Dashboard (`experiments/dashboard_d7_w1*.py`) — pas touché, blocage structurel documenté

Brief §4.3 : "ajouter NsDiff à la liste des modèles que le dashboard itère/
affiche en weekly, vérifier qu'aucune liste n'est codée en dur qui
exclurait NsDiff." Vérifié directement (`grep` sur les 4 fichiers
`dashboard_d7_w1*.py/.html` : zéro occurrence d'un nom de modèle en dur,
ni dans `dashboard_d7_w1.py` ni dans `dashboard_d7_w1_template.py`) : la
liste de modèles affichée est **100% dynamique**, dérivée du contenu de
`tracking.db` via `matrice_paired_tests.load_predictions`/
`build_daily_weekly_pairs`. Il n'y a donc **rien à ajouter dans le code**.

Mais NsDiff ne va pas apparaître dans ce dashboard aujourd'hui, pour deux
raisons structurelles indépendantes de ce brief, découvertes en creusant
(pas devinées) :

1. `matrice_paired_tests.load_predictions` filtre **`WHERE source='oos'`
   uniquement**. Nos lignes NsDiff-W sont en `source='live'`
   (`weekly_nsdiff_production.py`) et `source='backtest_rolling_nsdiff'`
   (`backtest_rolling_nsdiffw.py`) — ni l'une ni l'autre n'est `'oos'`,
   donc ce dashboard ne les verra JAMAIS, quel que soit leur volume.
2. Même si elles l'étaient : `build_daily_weekly_pairs` n'affiche une
   cellule (modèle, actif) que si CE MODÈLE a À LA FOIS un côté "régime B"
   (`frequence='daily'`, modèle quotidien projeté à l'horizon hebdo) ET un
   côté "régime C" (`frequence='weekly'`, natif) — vérifié sur TSDiff, qui a
   les deux (1350 lignes chacun, `source='oos'`). NsDiff n'a **aucune**
   production daily (brief §7 : le daily NsDiff est le vrai trou, cadré
   pour plus tard, hors scope aujourd'hui) — donc même avec des lignes
   `source='oos'`, NsDiff resterait invisible ici faute de côté "régime B".

**Décision** : ne pas modifier `dashboard_d7_w1.py`/`matrice_paired_tests.py`
(fichiers existants) pour contourner ces deux points — élargir le filtre de
source casserait potentiellement d'autres usages de `load_predictions`, et
la vraie fenêtre à ouvrir (un daily NsDiff à l'échelle) est explicitement
planifiée après aujourd'hui (brief §7). Documenté ici comme dette déclarée,
pas silencieusement laissé de côté.

## 7. Fichiers produits

- `experiments/backtest_rolling_nsdiffw.py` — script (réutilise
  `load_baseline_triplets`/`generate_baselines`/`ensure_backtest_rolling_
  index`/`_weekly_position` de `backtest_rolling_tsdiffw.py` tels quels,
  `prob_kpi_common`/`compute_prob_kpi_pilot`/`paired_test`/`pooled_analysis`
  tels quels ; seule fonction locale non triviale : `aggregate_per_cell_
  nsdiff`, fix du bug §2).
- `experiments/backtest_rolling_nsdiffw.json` / `_paired_tests.json`.
- `validation/tracking.db` : 1080 lignes `source='backtest_rolling_nsdiff'`,
  `model='NsDiff'` — isolation vérifiée (§ contrôle isolation, rapport de
  fin de tâche).
