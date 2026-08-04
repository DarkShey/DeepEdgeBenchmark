# BRIEF — NsDiff-W : approfondir la comparaison weekly + parité chaîne production/dashboard

*Créé le 2026-08-04. Rattaché à `methodologie_diffusion_vs_classiques.md` (étape 1/5) et à
`BRIEF_integration_nsdiff.md`. Fait suite à `NOTE_duel_nsdiff.md` (2026-08-03).*
*Objectif de fin de semaine (rappel du tuteur) : (1) comparer **TSDiff vs NsDiff**, puis
(2) comparer **Daily vs Weekly**. Ce brief traite le weekly ; l'axe Daily est cadré au §7.*

---

## 0. TL;DR — la question du tuteur, réponse honnête

> « TSDiff est le seul modèle de diffusion en weekly ; est-ce que NsDiff l'est aussi ? »

**Oui — mais pas partout, et c'est ça qu'il faut clarifier devant le tuteur.** Il existe
**deux chaînes weekly distinctes** dans le repo, et NsDiff-W n'est présent que dans l'une :

| Chaîne weekly | Rôle | TSDiff-W | NsDiff-W |
|---|---|---|---|
| **Duel** (`duel_backtest.py` / `duel_multiseed.py`) | comparaison rigoureuse offline (fair CRPS, MCS/SPA, multi-graines) | ✅ | ✅ **déjà fait hier** (`--include-nsdiff`, 5 actifs × 5 graines) |
| **Production / dashboard** (`weekly_tsdiff_production.py`, `backtest_rolling_tsdiffw.py`, `tracking.db`, `dashboard_d7_w1`) | génération live + backtest stocké + visualisation | ✅ | ❌ **absent** |

Donc :
- Le **weekly-natif de NsDiff est déjà codé** (`models/nsdiff_weekly.py`, miroir fidèle de
  `models/tsdiff_weekly.py`) **et déjà évalué** dans le duel : il bat TSDiff en CRPS **15/15
  cases** (moyenne 5 graines), significatif poolé **4/5 graines à W2 et W3**
  (cf. `NOTE_duel_nsdiff.md`). **Rien à réimplémenter côté duel.**
- Là où « TSDiff est le seul en weekly » est **littéralement vrai**, c'est la chaîne
  **production/dashboard** (celle qui écrit dans `tracking.db` et alimente le dashboard) :
  NsDiff-W n'y a jamais été câblé.

**Ce brief fait donc deux choses, dans cet ordre :**
- **Axe 1 (analyse, pas de nouveau code modèle)** — approfondir la comparaison weekly
  TSDiff vs NsDiff à partir des artefacts du duel **déjà calculés hier**, au-delà du
  CRPS/MCS/CV (calibration 50/80/95, sharpness, Winkler, PIT, décomposition par actif/classe).
- **Axe 2 (câblage)** — mettre NsDiff-W à **parité** avec TSDiff-W dans la chaîne
  production/dashboard, pour que les deux modèles de diffusion soient visibles côte à côte
  en weekly (et backtestés sur les mêmes triplets `tracking.db`).

---

## 1. État des lieux — la grille complète {modèle} × {horizon} × {chaîne}

Avant de coder quoi que ce soit, la carte exacte de ce qui existe (vérifiée dans le code) :

| Modèle | Daily | Weekly (duel) | Weekly (prod/dashboard) |
|---|---|---|---|
| **TSDiff** | ✅ complet (`kpi_probabilistes.json`, `multi_horizon.forecast_horizons_tsdiff`) | ✅ complet (`duel_backtest.json`, multi-graines) | ✅ complet (`weekly_tsdiff_production.py` → `tracking.db` `source='live'` ; `backtest_rolling_tsdiffw.py` → `source='backtest_rolling'`) |
| **NsDiff** | ⚠️ **contrôle 1-actif/1-graine seulement** (`NOTE_nsdiff_vs_tsdiff.md`) + adapter câblé (`multi_horizon.forecast_horizons_nsdiff`) | ✅ complet (`duel_backtest_nsdiff.json`, `duel_backtest_nsdiff_swept.json`, multi-graines) | ❌ **rien** |

**Deux enseignements contre-intuitifs à assumer devant le tuteur :**
1. **Le weekly de NsDiff n'est PAS le manque** — il est déjà fait, et plutôt mieux couvert que
   le daily.
2. **Le vrai déséquilibre est sur le daily** : NsDiff daily n'a qu'un contrôle ponctuel
   (SPY, 1 graine), alors que son weekly est multi-graines complet. Pour l'objectif
   « Daily vs Weekly » de fin de semaine, c'est le côté daily de NsDiff qui devra être monté
   en puissance (cadré §7, hors focus d'aujourd'hui par choix).

---

## 2. Point technique clé — TSDiff-W et NsDiff-W ne sont pas définis *identiquement* dans le duel

À scruter dans l'axe 1, parce que ça conditionne l'équité de la comparaison d'hier :

| | TSDiff-W (duel) | NsDiff-W (duel) |
|---|---|---|
| Fonction de fit | `td.fit_tsdiff(train_weekly, …)` — **module daily nourri en weekly** | `nw.fit_weekly(train_weekly, …)` — **module weekly dédié** |
| Lookback `seq_len` | **30** (défaut daily `tsdiff_model.SEQ_LEN`) | **26** (`nsdiff_weekly.SEQ_LEN_W`, ~6 mois) |
| Horizon généré | 3 (W1/W2/W3), un seul tir | 3 (W1/W2/W3), un seul tir |
| Budget d'époques | **sweep validation 40/60/80 par graine × actif** (verrou E1) | **fixe déclaré 40** (`NSDIFF_EPOCHS_W`, jamais sélectionné par graine) |
| Pas de diffusion `k_denoise` | 20, fixé à l'échantillonnage | 20, **fixé au fit** (`T=k_denoise`) |

Références : `duel_backtest.py` L171 (TSDiff) et L315 (NsDiff) ; `NOTE_duel_nsdiff.md` §Limites.

**Conséquence à écrire noir sur blanc dans la note de comparaison :** l'avantage CRPS de NsDiff
sur TSDiff (15/15) et sa reproductibilité ~4× supérieure (CV 2.4% vs 9.7%) sont **en partie**
attribuables à deux différences de protocole, pas seulement au modèle :
- le **budget d'époques fixe** de NsDiff retire une source de variance inter-graines que TSDiff
  porte (déjà noté hier) ;
- le **lookback différent** (26 vs 30) n'est *pas* neutralisé.

Ce n'est pas disqualifiant — c'est une **limite déclarée** à mettre en avant, et un candidat
naturel pour un test de robustesse (« et si on aligne seq_len et le budget d'époques ? »).

---

## 3. Axe 1 — Approfondir la comparaison weekly TSDiff vs NsDiff (depuis le duel)

**But :** répondre à « compare les deux plus en détail qu'hier » **sans relancer le duel** —
tout part des artefacts déjà produits.

### 3.1 Ce qu'on a déjà (hier, `NOTE_duel_nsdiff.md`)
CRPS (15/15 en faveur de NsDiff), stabilité du signe (14/15), verdict poolé Holm (W2/W3 sig
4/5), MCS par actif, CV inter-graines, SPA (0/75 inchangé).

### 3.2 Ce qu'on ajoute (le « plus en détail »)
Toutes ces métriques existent déjà dans l'outillage — **ne rien réimplémenter** :
`prob_kpi_common.py::row_kpis` (coverage 50/80/95, sharpness, Winkler, PIT) — le même que
`backtest_rolling_tsdiffw.py`.

1. **Calibration comparée** : couverture empirique 50/80/95 % de NsDiff-W **vs** TSDiff-W,
   par horizon et par classe d'actif (crypto / index / bonds). Le point fort reconnu de la
   diffusion étant la calibration (cf. `methodologie…md` §3), c'est **la** métrique où NsDiff
   doit prouver qu'il apporte quelque chose — son `g_psi`/UANS conditionne explicitement la
   largeur d'intervalle sur l'historique récent (`NOTE_nsdiff_vs_tsdiff.md`).
2. **Sharpness vs couverture** : NsDiff est-il mieux calibré *à finesse égale*, ou seulement
   plus large ? Tracer (sharpness, coverage) pour les deux — un modèle qui couvre mieux juste
   en élargissant ne « gagne » pas vraiment.
3. **Winkler score** (récompense couverture ET finesse) et **PIT** (histogramme
   d'uniformité) par modèle × horizon.
4. **Décomposition par actif** croisée avec le décrochage connu : TSDiff décroche sur ETH-USD
   (MCS 0.6) là où NsDiff reste à 1.0 ; l'inverse sur BTC-USD (W1/W2). Étendre cette lecture
   au-delà du MCS/CRPS : la calibration suit-elle le même motif ?
5. **Test de robustesse (optionnel, si le temps)** : relancer NsDiff-W dans le duel avec
   **époques swept 40/60/80** (comme TSDiff, via `--nsdiff-epoch-candidates`, déjà disponible)
   et/ou `seq_len` aligné, pour voir si l'avantage CRPS survit à l'égalisation du budget HPO
   — exactement le raisonnement qui a été fait pour TSDiff dans l'epoch-sweep. NB : un run
   swept existe peut-être déjà (`duel_backtest_nsdiff_swept.json`) — **vérifier son contenu
   avant de relancer**.

### 3.3 Étape 0 obligatoire — auditer ce que les artefacts du duel contiennent
`duel_backtest_nsdiff.json` / `_swept.json` stockent-ils, par origine, les **intervalles /
échantillons** (nécessaires pour couverture/sharpness/Winkler/PIT), ou seulement le **CRPS**
agrégé ? Regarder aussi `experiments/checkpoints_nsdiff/seed{N}_{actif}_nsdiff.json`.
- Si les intervalles/échantillons y sont → calcul direct des KPI, zéro re-run.
- Sinon → soit relancer le duel avec persistance des échantillons, soit rejouer uniquement
  la passe de scoring KPI depuis les checkpoints. **Décider à cette étape**, pas avant.

### 3.4 Livrable axe 1
- `experiments/compare_weekly_diffusion.py` : lit les artefacts du duel, produit la matrice
  KPI TSDiff-W vs NsDiff-W (CRPS + calibration + sharpness + Winkler + PIT) par actif × horizon.
- `NOTE_compare_weekly_tsdiff_nsdiff.md` : 1–2 pages, tableaux + verdict différencié
  (précision / calibration / robustesse), **limites de protocole du §2 explicitées**.

---

## 4. Axe 2 — Parité NsDiff-W dans la chaîne production/dashboard

**But :** que le dashboard et `tracking.db` montrent NsDiff-W au même titre que TSDiff-W —
c'est là que « TSDiff est le seul en weekly » est vrai aujourd'hui.

**Bonne nouvelle (faisabilité) :** `nsdiff_model.py` expose déjà, au niveau module,
`fit_nsdiff(train, …, horizon=, epochs=) -> (model, mu, sd)` et
`forecast_from_fitted(model, hist_window, mu, sd, last_price, horizons=, n_samples=, **_ignored)`
— **signatures identiques** à `tsdiff_model.fit_tsdiff` / `forecast_from_fitted` (le `k_denoise`
est absorbé par `**_ignored`, les pas de diffusion NsDiff étant fixés au fit). Les deux scripts
TSDiff de cette chaîne appellent justement ces fonctions **du module daily nourri en weekly**
(pas `tsdiff_weekly.py`). Donc le miroir NsDiff est un **quasi-swap `td` → `nm`**, ligne pour
ligne, avec les nuances ci-dessous.

> **Décision de définition à trancher (importante) :** pour rester cohérent avec le NsDiff-W
> **déjà évalué dans le duel**, utiliser `nsdiff_weekly.fit_weekly` (module weekly, seq_len=26).
> Pour un miroir strictement parallèle à TSDiff-prod (seq_len=30), utiliser `nm.fit_nsdiff`.
> **Recommandation : `nm.fit_nsdiff`** (miroir ligne-à-ligne, même conditionnement que
> TSDiff-prod) **et déclarer explicitement** que la définition prod diffère de la définition
> duel sur seq_len — ou, mieux, aligner les deux définitions une bonne fois (voir §5).

### 4.1 `experiments/weekly_nsdiff_production.py` (miroir de `weekly_tsdiff_production.py`)
Changements par rapport à l'original :
- `import nsdiff_model as nm` ; `MODEL_NAME = "NsDiff"`.
- Fit : `model, mu, sd = nm.fit_nsdiff(weekly, horizon=HORIZON_WEEKLY, epochs=ep)` ;
  forecast : `nm.forecast_from_fitted(model, weekly_z, mu, sd, last_price, horizons=[1,2,3],
  n_samples=…)`.
- **Époques** : NsDiff n'a **pas** d'entrée dans `epoch_sweep_results.json` (`…|TSDiff-W`
  seulement). Deux options : (a) budget fixe déclaré `NSDIFF_EPOCHS_W=40` (le plus simple,
  cohérent avec le duel) ; (b) ajouter une sélection NsDiff-W à `epoch_sweep.py` (plus propre,
  aligne avec TSDiff — cf. §5). **Choisir (a) pour aujourd'hui, (b) en dette déclarée.**
- **`tc_id` / idempotence** : garder `tc_id = f"TC_{ticker}_W{h}"`. L'unicité de
  `save_prediction` est `UNIQUE(tc_id, model, cutoff_date)` — la colonne `model` distingue
  `NsDiff` de `TSDiff`, donc **aucune collision** avec les lignes TSDiff-W existantes
  (à re-vérifier dans `validation/tracking_db.py`). `run_id = f"weekly-nsdiff-{origin.date()}"`.
- `verdict_rules` / `TRADING_DAYS_PER_WEEK` réutilisés tels quels.

### 4.2 `experiments/backtest_rolling_nsdiffw.py` (miroir de `backtest_rolling_tsdiffw.py`)
- **Réutiliser les MÊMES triplets baselines** déjà dans `tracking.db`
  (`load_baseline_triplets`, `frequence='weekly'`, `source='oos'`) → alignement exact
  (actif, cutoff, target) **gratuit**, comparaison appariée avec les 5 classiques garantie,
  et **superposable au backtest TSDiff-W** (mêmes origines).
- Génération : swap `td.fit_tsdiff`/`td.forecast_from_fitted` → `nm.fit_nsdiff`/
  `nm.forecast_from_fitted` dans `generate_*_asset` (refit expansif périodique inchangé).
- **Garde-fou de fuite d'epoch-selection** : chez TSDiff, on exclut ~12 dates/actif du bloc
  de validation du sweep d'époques. **Si NsDiff-W utilise des époques FIXES (option a), il n'y
  a pas de sélection d'epoch → pas de fuite à guarder → on garde toutes les origines**
  (à documenter). Si on passe au sweep (option b), reporter le même garde-fou.
- **Isolation `tracking.db`** : `--source-tag backtest_rolling_nsdiff` (valeur `source`
  distincte + son propre index unique partiel, exactement le mécanisme prévu par le script).
  **Ne jamais écrire sur `source` in {'live','oos','backtest_rolling'}** (lignes TSDiff/baselines).
- **Asymétrie d'échantillonnage** : identique à TSDiff-W — NsDiff tire de vrais échantillons de
  diffusion, les baselines des échantillons gaussiens depuis leur PI stockée. Reprendre le
  caveat mot pour mot.

### 4.3 Dashboard (`experiments/dashboard_d7_w1*.py`)
Une fois les lignes NsDiff-W présentes dans `tracking.db` (live + backtest_rolling), ajouter
`"NsDiff"` à la liste des modèles que le dashboard itère/affiche en weekly. Vérifier qu'aucune
liste de modèles n'est codée en dur d'une façon qui exclurait NsDiff.

### 4.4 Livrable axe 2
- `experiments/weekly_nsdiff_production.py` + `experiments/backtest_rolling_nsdiffw.py`
  (+ tests miroirs des tests TSDiff correspondants s'ils existent).
- Lignes NsDiff-W dans `tracking.db` (`source='live'` + `source='backtest_rolling_nsdiff'`),
  isolées et idempotentes.
- Dashboard affichant TSDiff-W **et** NsDiff-W en weekly.
- `NOTE_backtest_rolling_nsdiffw.md` (miroir de `NOTE_backtest_rolling_tsdiffw.md`) :
  calibration + CRPS de NsDiff-W sur les mêmes triplets que TSDiff-W et les 5 baselines.

---

## 5. Dette déclarée / alignement d'équité (à mentionner, pas forcément à faire aujourd'hui)
Pour que TSDiff-W vs NsDiff-W soit *toutes choses égales par ailleurs* :
- **Aligner `seq_len`** (26 vs 30) entre les deux modèles de diffusion en weekly.
- **Aligner le budget d'époques** (sweep 40/60/80 pour NsDiff aussi) — le flag existe déjà
  (`--nsdiff-epoch-candidates`) et un artefact swept existe peut-être déjà.
- Ajouter une entrée `…|NsDiff-W` à `epoch_sweep_results.json` pour que la prod NsDiff-W
  sélectionne ses époques comme TSDiff-W (option b du §4.1).

---

## 6. Contraintes non-négociables (reprises des briefs existants)
- [ ] **Point-in-time / anti-fuite** : `mu`/`sd` calculés sur le train seul, jamais recalculés
      sur données futures ; resample `W-FRI` avec `dropna()` (jamais de barre vendredi partielle).
- [ ] **Gel à T0** : un seul fit par origine-bloc (ou par origine), jamais de poids mis à jour
      sur des données postérieures à l'origine.
- [ ] **Isolation `tracking.db`** : nouveau `source_tag` + index unique partiel dédié ;
      `git diff` vide sur les lignes existantes ; upsert idempotent (re-run sans doublon).
- [ ] **Aucun fichier source existant modifié** hors ajouts (`git status` = nouveaux fichiers
      + notes). Suite **pytest verte** avant/après.
- [ ] **Époques / seq_len / m / k_denoise** déclarés par modèle dans chaque livrable.
- [ ] **Caveat d'asymétrie** (diffusion natif vs gaussien baselines) repris tel quel.

---

## 7. Cadrage de l'axe « Daily vs Weekly » (fin de semaine, hors focus d'aujourd'hui)
Pour mémoire, parce que c'est le 2ᵉ objectif du tuteur et que la grille du §1 montre où est le
trou : **le daily de NsDiff n'a qu'un contrôle 1-actif/1-graine.** Une comparaison
Daily-vs-Weekly *symétrique* entre TSDiff et NsDiff exigera de monter NsDiff daily à l'échelle
(5 actifs × multi-graines), via `multi_horizon.forecast_horizons_nsdiff` (déjà câblé dans
`MODEL_ADAPTERS`) sur le protocole daily de `kpi_probabilistes`. **À planifier après les deux
axes d'aujourd'hui** — signalé ici pour que ce ne soit pas une surprise en fin de semaine.

---

## 8. Ordre d'exécution suggéré aujourd'hui
1. **Axe 1 · étape 0** — auditer le contenu de `duel_backtest_nsdiff.json` / `_swept.json` /
   `checkpoints_nsdiff/*` (intervalles présents ou non ?). *(rapide, débloque tout le reste)*
2. **Axe 1** — `compare_weekly_diffusion.py` + `NOTE_compare_weekly_tsdiff_nsdiff.md`
   (calibration/sharpness/Winkler/PIT + limites protocole §2). *(livrable « compare en détail »)*
3. **Axe 2 · prod** — `weekly_nsdiff_production.py` (swap `td`→`nm`, époques fixes 40),
   smoke `--dry-run`, puis écriture `source='live'`.
4. **Axe 2 · backtest** — `backtest_rolling_nsdiffw.py` (`--smoke` d'abord, puis run complet,
   `--source-tag backtest_rolling_nsdiff`) + `NOTE_backtest_rolling_nsdiffw.md`.
5. **Axe 2 · dashboard** — ajouter NsDiff-W à l'affichage weekly.
6. **pytest** complet + `git status`/`git diff` de contrôle d'isolation.

### Commandes de référence
```bash
# Axe 1 — audit rapide
python -c "import json; d=json.load(open('experiments/duel_backtest_nsdiff.json')); print(list(d.keys()), list(d.get('config',{}).keys()))"

# Axe 2 — prod NsDiff-W (dry-run puis réel)
python experiments/weekly_nsdiff_production.py --assets SPY BTC --dry-run
python experiments/weekly_nsdiff_production.py

# Axe 2 — backtest rolling NsDiff-W (smoke puis complet, isolé)
python experiments/backtest_rolling_nsdiffw.py --smoke
python experiments/backtest_rolling_nsdiffw.py --source-tag backtest_rolling_nsdiff

# (optionnel §5) robustesse : NsDiff-W avec sweep d'époques comme TSDiff
python experiments/duel_multiseed.py --seeds 42 43 44 45 46 --m-samples 500 \
    --skip-global --include-nsdiff --nsdiff-epoch-candidates 40 60 80 --end 2026-07-29 \
    --out experiments/duel_backtest_nsdiff_swept.json   # vérifier s'il existe déjà avant
```

---

## 9. Pièges à éviter
- **Ne pas réimplémenter le weekly de NsDiff** : `nsdiff_weekly.py` existe et est déjà dans le
  duel. Le travail d'aujourd'hui est *analyse* (axe 1) + *câblage prod/dashboard* (axe 2), pas
  un nouveau modèle.
- **Ne pas conclure « NsDiff > TSDiff » sans mentionner les asymétries de protocole du §2**
  (époques fixes, seq_len 26 vs 30) — le tuteur verra la faille si elle est passée sous silence.
- **Ne pas polluer `tracking.db`** : lignes TSDiff-W / baselines / `oos` intactes ; NsDiff sous
  son propre `source_tag`.
- **Ne pas oublier que le vrai trou est le daily de NsDiff** (§7), pas le weekly — pour
  l'objectif « Daily vs Weekly », c'est là qu'il faudra investir ensuite.
- **Ne pas relancer un run coûteux qui existe déjà** : vérifier `duel_backtest_nsdiff_swept.json`
  avant tout re-sweep.
