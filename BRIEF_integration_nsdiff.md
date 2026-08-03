# BRIEF — Intégration d'un second modèle de diffusion : NsDiff

*Cible : Claude Code. Rédigé le 2026-08-03. Contexte validé (recommandation + réglage
par actif sur les actifs de test existants approuvés).*

---

## 0. Objectif en une phrase

Ajouter au benchmark un **second modèle de diffusion, NsDiff** (*Non-stationary
Diffusion for Probabilistic Time Series Forecasting*, ICML 2025), **en univarié /
par actif**, en respectant **à la lettre** le contrat des modèles existants, **sans
rien casser** de ce qui tourne aujourd'hui (TSDiff et les 5 baselines, le duel
multi-graines déjà scoré, le pipeline dashboard).

**Pourquoi NsDiff et pas un autre** (à garder en tête, ça guide les choix
d'implémentation) : c'est le seul diffuseur dont le mécanisme central modélise la
**variance conditionnelle dépendante de l'historique** dans le processus de diffusion
lui-même (modèle *location-scale* + *noise schedule* piloté par l'incertitude). C'est
l'analogue génératif de GARCH. En finance à faible SNR, la précision ponctuelle est
perdue pour tous les modèles ; le seul edge exploitable vit dans la **distribution
conditionnelle** (volatilité, queues, calibration). NsDiff attaque précisément cet axe
— celui où TSDiff a son unique point fort (calibration daily) et où GARCH reste
imbattable. **L'implémentation doit donc préserver religieusement la partie variance /
calibration : c'est la raison d'être du modèle, pas un détail.**

Référence officielle : papier arXiv:2505.04278 ; **code officiel** :
`https://github.com/wwy155/NsDiff`. Porter à partir de là, ne pas réinventer les
équations.

---

## 1. Règle d'or : additif, jamais destructif

1. **Ne jamais modifier** le comportement de `models/tsdiff_model.py`,
   `models/tsdiff_weekly.py`, `models/{arima,sarima,prophet,lstm,naive}_model.py`,
   ni des `experiments/duel_*.py`/`benchmarks/*`/`model_artifacts/*` existants au-delà
   d'**ajouts** (nouvelle entrée de dict, nouvelle branche `elif`, nouvelle fonction).
   Aucune signature existante ne change, aucune valeur par défaut existante ne bouge.
2. **Toute l'intégration passe par les points d'extension déjà prévus** (cf. §4). Le
   registre `benchmarks/multi_horizon.MODEL_ADAPTERS` est explicitement commenté
   « point d'extension » : c'est le patron à suivre partout.
3. **Le duel scoré (`experiments/duel_backtest.py` / `duel_multiseed.py`) doit rester
   reproductible à l'octet près sans NsDiff.** L'ajout de NsDiff au duel est **opt-in**
   (drapeau CLI / liste de modèles), défaut = comportement 6-modèles actuel inchangé.
   Voir §4.4.
4. **Non-régression prouvée** : la suite `pytest` complète doit être **verte avant et
   après** chaque étape. On lance les tests après chaque fichier touché, pas à la fin.
5. Travailler sur une **branche dédiée** (`feat/nsdiff-model`), commits petits et
   atomiques (un fichier / un point d'intégration par commit).

---

## 2. Le contrat à respecter (observé dans le code, non négociable)

Tout modèle du benchmark expose une interface commune. NsDiff doit la répliquer
exactement — c'est ce que testent `models/test_models_common.py` et
`models/test_tsdiff_model.py`.

### 2.1 API du module `models/nsdiff_model.py`

- `fetch_data(ticker, start, end) -> pd.Series` — mêmes conventions que
  `tsdiff_model.fetch_data` (yfinance, Close auto-adjust, tz-naive, dropna). **Réutiliser
  la même implémentation** (copier telle quelle : c'est identique pour tous les modèles).
- `compute_metrics(actual, predicted, pi_lower, pi_upper, train_time) -> dict` —
  **copier telle quelle** depuis `tsdiff_model.py` (contrat identique : clés `RMSE`,
  `MAE`, `MAPE (%)`, `SMAPE (%)`, `Dir. Acc (%)`, `PI Cov 95% (%)`, `Ljung-Box p`,
  `Train Time (s)`). Ne pas réinventer.
- `run_nsdiff(train, test, ..., keep_samples: bool = False) -> dict` — walk-forward
  1-pas quotidien. Retourne un dict contenant **au minimum** :
  `RMSE, MAE, "Dir. Acc (%)", "PI Cov 95% (%)"` (+ autres clés de `compute_metrics`),
  `predictions, lower, upper, index, actual`.
  - `predictions` = moyenne du nuage d'échantillons ; `lower`/`upper` = quantiles
    2.5 / 97.5 du nuage (la vraie distribution prédictive, **jamais** une bande
    gaussienne reconstruite).
  - `keep_samples=True` ⇒ ajoute `result["ensemble"]` : liste de longueur `len(test)`,
    un array `[n_samples]` de **prix** par pas. `keep_samples=False` (défaut) ⇒
    **aucune** clé `ensemble` (non-régression stricte, testée).
  - `len(predictions) == len(lower) == len(upper) == len(test)`, tout fini,
    `lower <= predictions <= upper`.
  - Série trop courte ⇒ `raise ValueError` (message clair mentionnant `seq_len`).
- `next_step_nsdiff(series, ...) -> (pred, lo, hi)` — un seul pas au-delà de la dernière
  observation, `lo <= pred <= hi`, finis.
- `fit_nsdiff(train, ...) -> (model, mu, sd)` + `forecast_from_fitted(model, hist_window,
  mu, sd, last_price, horizons=None, n_samples=..., ...) -> {h: price_samples}` —
  **séparer l'entraînement de la prévision** exactement comme `tsdiff_model.fit_tsdiff`
  / `forecast_from_fitted`. C'est indispensable au protocole *train-once-forward* du duel
  (fit une fois à T0, prévoir à N origines sans réentraîner). `mu`/`sd` sont les stats
  de standardisation du train, **gelées** — jamais recalculées sur des données postérieures
  (anti-lookahead).
- Bloc `main()` CLI avec les mêmes flags que les autres :
  `--ticker/--start/--end/--test-ratio/--epochs/--next-step/--plot` (+ hyperparamètres
  propres NsDiff). S'inspirer du `main()` de `tsdiff_model.py`.

### 2.2 API du module `models/nsdiff_weekly.py`

Mirror de `models/tsdiff_weekly.py` : resample daily→weekly (`W-FRI`, `.last().dropna()`,
point-in-time), entraîne sur rendements **hebdo** standardisés, génère **W+1/W+2/W+3 en
un seul tir** (non-autorégressif — c'est ce qui évite le compounding d'erreur, cf.
l'artefact P1 déjà corrigé pour TSDiff-W). Fonctions attendues : `to_weekly`,
`fit_weekly`, `forecast_from_fitted_weekly`, `forecast_weekly`, mêmes signatures/retours
que la version TSDiff (`{k: {"point","lo","hi","samples"}}`).

### 2.3 Invariants de rigueur (quant dev)

- **Reproductibilité** : une fonction `set_seed(seed)` seedant python/numpy/torch (copier
  celle de `tsdiff_model.py`). Résultat déterministe pour (seed, données) fixés. Le duel
  thread **un seul** `seed` partout — respecter ça.
- **Anti-lookahead / point-in-time** : `mu, sd` figés à T0 ; la standardisation du test
  réutilise ces stats. Le test `test_point_in_time_no_lookahead` (dans
  `test_models_common.py`) tronque la fenêtre de test et exige que le préfixe des
  prédictions coïncide — NsDiff doit passer l'équivalent.
- **Sécurité numérique** : variance prédite clampée `>= eps` (jamais 0/négatif — une
  variance nulle casse le sampling) ; `x0`/échantillons clampés à une plage raisonnable ;
  perte non finie ⇒ on saute le batch (comme TSDiff : `if not torch.isfinite(loss): continue`) ;
  `clip_grad_norm_`. Le CRPS et la couverture n'ont de sens que si aucun NaN/Inf ne fuit.
- **CPU-tractable** : c'est un benchmark qui tourne sur CPU. Backbone dégraissé (cf. §3),
  échantillonnage type DDIM avec **peu de pas** (`k_denoise ≈ 20`, comme TSDiff). Fournir
  des defaults « CPU-raisonnables » et documenter le budget temps mesuré sur 1 actif.
- **Style maison** : type hints, docstrings de module et de fonction denses (le repo est
  très documenté — s'aligner sur le ton de `tsdiff_model.py`/`duel_sampling_adapters.py`),
  auto-contenu, aucune dépendance nouvelle hors `torch` (déjà dans `requirements.txt`). Si
  une dépendance s'avère nécessaire, l'ajouter à `requirements.txt` avec un commentaire.

---

## 3. La science NsDiff — ce qu'il faut porter fidèlement

Porter depuis `wwy155/NsDiff` (+ équations du papier), **réduit à l'univarié**
(`n_features = 1`) sur des **log-rendements standardisés**. Composants :

1. **Modèle location-scale (LSNM)** — au lieu du bruit additif à variance constante
   `Y = f(X) + ε` des diffuseurs classiques (dont TSDiff), NsDiff pose
   `Y = f_φ(X) + √g_ψ(X) · ε`, où :
   - `f_φ(X)` = réseau de **moyenne conditionnelle** (l'officiel : Non-stationary
     Transformer). Sur nos fenêtres courtes (`seq_len ≈ 26–30`) et en univarié, un
     Transformer complet est surdimensionné : **le dégraisser** (peu de couches/têtes,
     dim cachée modeste) *ou* le remplacer par un backbone plus léger **en préservant
     strictement le LSNM + le schedule de bruit** (§ ci-dessous). Ce choix
     d'architecture est une **décision déclarée** à documenter en tête de fichier (pas un
     écart silencieux) : la contribution de NsDiff est la variance, pas le backbone de la
     moyenne.
   - `g_ψ(X)` = **prédicteur de variance conditionnelle** (l'officiel : MLP 3 couches).
     C'est le cœur du modèle — **le garder**. Cible d'entraînement = **variance réalisée
     glissante** des rendements standardisés sur l'horizon (régression supervisée, comme
     dans l'officiel). Sortie clampée `>= eps`.
2. **Uncertainty-aware noise schedule (UANS)** — la vraie innovation : la variance du
   forward process dépend de `g_ψ(X)` et transitionne vers l'endpoint appris
   (schématiquement `σ_t² = β_t²·g_ψ(X) + α_t·β_t·σ_{Y0}`, cf. papier — **reprendre la
   formule exacte du repo, ne pas l'approximer**). En espace de rendements standardisés,
   `σ_{Y0} ≈ 1`. C'est ce qui donne au modèle sa calibration hétéroscédastique de type
   GARCH.
3. **Échantillonnage** — reverse diffusion (type DDIM, peu de pas) produisant `m`
   trajectoires de l'horizon **en un tir** (non-autorégressif). Réduire à
   moyenne + quantiles 2.5/97.5 pour l'IC ; conserver le nuage pour le CRPS
   (`keep_samples`).
4. **Conditionnement** : sur la fenêtre de look-back (`seq_len` derniers rendements
   standardisés), comme le fait TSDiff avec son history-embedding. Pas de régime/macro
   (ils n'existent pas ici) — même simplification univariée que le port TSDiff (voir le
   docstring « Adaptation vs the DEITA original » de `tsdiff_model.py`).

**Validation de fidélité** (recommandée) : reproduire un chiffre du papier sur **un**
petit jeu standard (p. ex. ETTh1) avec le code porté, pour attester que le LSNM+UANS est
correctement transcrit avant de le lâcher sur nos rendements. À défaut, au minimum un
test de cohérence (cf. §5).

---

## 4. Plan d'intégration fichier par fichier (avec les points exacts)

### 4.1 Nouveaux fichiers (100 % additifs)
- `models/nsdiff_model.py` — cœur (cf. §2.1, §3).
- `models/nsdiff_weekly.py` — variante hebdo (cf. §2.2).
- `models/test_nsdiff_model.py` — miroir de `test_tsdiff_model.py` (cf. §5).

### 4.2 `benchmarks/multi_horizon.py` — le point d'extension principal (additif)
- Ajouter `forecast_horizons_nsdiff(train, horizons, epochs=None, seed=None) -> dict`,
  **calqué ligne à ligne** sur `forecast_horizons_tsdiff` (lignes ~355-398) : fit une
  fois, `sample_paths`, point = moyenne du nuage, IC = quantiles, plafonnement à
  l'horizon généré.
- Ajouter **une ligne** au registre `MODEL_ADAPTERS` (fin de fichier) :
  `"NsDiff": forecast_horizons_nsdiff,`.
- **Effet automatique** : `benchmarks/run_benchmark.py` itère `MODEL_ADAPTERS.keys()`
  (lignes 261-266) → NsDiff apparaît dans l'UI de sélection **sans toucher**
  `run_benchmark.py`. Ne rien y modifier.

### 4.3 `model_artifacts/pipeline.py` — dashboard (additif, plusieurs points)
Tous des **ajouts** (élément de liste / branche `elif` / entrée de dict), en miroir exact
des lignes TSDiff :
- `MODELS` (ligne 109) : ajouter `"NsDiff"`.
- Dict de labels (ligne ~112) : ajouter `"NsDiff": "NsDiff"`.
- Branche Gate-2 validation (ligne ~610) : `elif model_key == "NsDiff":` →
  `import nsdiff_model; nsdiff_model.set_seed(...); result = nsdiff_model.run_nsdiff(train, validation, keep_samples=True)`.
- Map de modules (lignes ~635, ~909) : ajouter `"NsDiff": "nsdiff_model"`.
- Branches prévision live (lignes ~661-662, ~687-688) :
  `if model_key == "NsDiff": return mh.forecast_horizons_nsdiff(...)`.
- Ensembles « pas d'artefact Gate 1 » (lignes ~1198, ~1239) : ajouter `"NsDiff"` à côté
  de `"Naive", "TSDiff"` (NsDiff, comme TSDiff, s'entraîne en Gate 2, pas d'artefact
  Gate 1 sérialisé).

### 4.4 Le duel — `experiments/duel_backtest.py` + `duel_multiseed.py` (**opt-in**, additif)
Objectif : pouvoir opposer NsDiff aux 5 classiques **et à TSDiff** dans le même protocole
rigoureux (rolling-origin, gel T0, `m=500`, fair CRPS, MCS/SPA), **sans altérer** le duel
6-modèles déjà scoré.
- **Gate derrière un drapeau** : ajouter `--include-nsdiff` (défaut `False`). Si absent,
  `MODELS`, `PAIRS` et les artefacts produits sont **identiques à l'actuel** (byte-for-byte).
- Quand activé :
  - Ajouter `"NsDiff"` à `MODELS` (ligne 91) et étendre `PAIRS` (ligne 96) avec
    `("NsDiff", c)` pour chaque classique **et** `("TSDiff", "NsDiff")` (la comparaison
    diffusion-vs-diffusion, celle qui a le plus d'intérêt scientifique ici).
  - Fit unique à T0 : mirror de `td.fit_tsdiff(train_weekly, horizon=HORIZON_WEEKLY, epochs=...)`
    (ligne 156) avec `nsdiff_weekly.fit_weekly(...)`.
  - Sélection d'époques sur **validation uniquement** (verrou E1) : réutiliser
    `epoch_sweep._sweep_one_model`/`select_epochs` comme TSDiff-W (lignes 146-150) — ou,
    a minima, un budget d'époques fixe **déclaré** si le sweep NsDiff est trop coûteux (à
    documenter, jamais choisi sur le test).
  - Échantillonnage à chaque origine : mirror de
    `td.forecast_from_fitted(model_w, weekly_z[:m_pos], mu_w, sd_w, last_price, horizons=..., n_samples=m)`
    (ligne 197). Même `m` que tout le monde (déjà passé partout via `--m-samples`).
  - `duel_multiseed.py` : NsDiff hérite du threading de graine existant (init/entraînement/
    échantillonnage re-armés par graine via `set_seed`) — vérifier que la stochasticité
    NsDiff est bien couverte par le `seed` unique (comme TSDiff via `td.set_seed`).
- **Ne pas** toucher la comparaison global-vs-par-actif de TSDiff (§2.3 du duel) : hors
  périmètre.

### 4.5 `requirements.txt`
N'ajouter une ligne **que si** une dépendance réellement nouvelle est requise (a priori
`torch` suffit). Le cas échéant, commenter comme les autres.

---

## 5. Tests (obligatoires, rapides, façon `test_tsdiff_model.py`)

Créer `models/test_nsdiff_model.py`, config **minuscule** (petit backbone, `epochs=1`,
`n_samples≈6`, `k_denoise≈3`) pour que le gate reste rapide. Couvrir :
1. **Contrat** `run_nsdiff` : présence de toutes les clés requises, longueurs = `len(test)`,
   finitude, `lower <= predictions <= upper`, **absence** de `ensemble` par défaut.
2. `keep_samples=True` : `ensemble` présent, longueur `len(test)`, chaque nuage de forme
   `(n_samples,)` et fini.
3. `next_step_nsdiff` : `lo <= pred <= hi`, fini.
4. `forecast_horizons_nsdiff` (via `benchmarks.multi_horizon`) : clés = horizons demandés,
   `lo <= point <= hi`.
5. **Série trop courte** ⇒ `ValueError` (`match="seq_len"`).
6. **Non-régression fit/forecast** : `run_nsdiff` doit reproduire **exactement** une
   référence walk-forward inline (mirror de `test_forecast_from_fitted_matches_reference_walk_forward`)
   → garantit que `fit_nsdiff`+`forecast_from_fitted` ne dérivent pas.
7. **Anti-lookahead** : équivalent de `test_point_in_time_no_lookahead` (préfixe stable
   quand on tronque le test).
8. **Sanity calibration (spécifique NsDiff, la valeur ajoutée)** : sur une série
   synthétique **hétéroscédastique** (variance qui varie dans le temps), vérifier que la
   couverture de l'IC 95 % reste dans une plage raisonnable et **meilleure** qu'un modèle
   à variance constante jouet — atteste que `g_ψ`/UANS font bien leur travail. Garder ce
   test tolérant (bornes larges) pour ne pas le rendre instable, mais présent.

**Ne pas** ajouter NsDiff à la paramétrisation de `test_models_common.py` (réservée aux 4
forecasters classiques via `conftest.py`) — TSDiff a son propre fichier de test dédié,
NsDiff suit la même convention.

Commande de non-régression après **chaque** étape :
`python -m pytest models/ benchmarks/ model_artifacts/ experiments/ -q` (doit rester vert).

---

## 6. Critères d'acceptation (definition of done)

- [ ] `models/nsdiff_model.py`, `models/nsdiff_weekly.py`, `models/test_nsdiff_model.py`
      créés ; **aucun fichier existant modifié autrement que par ajout**.
- [ ] `git diff` sur les fichiers touchés (multi_horizon/pipeline/duel) ne contient que des
      **ajouts** (nouvelles entrées/branches/fonctions), zéro modification de logique TSDiff
      ou baseline.
- [ ] Suite `pytest` **entièrement verte** (existants + nouveaux).
- [ ] Duel sans `--include-nsdiff` : sortie **inchangée** vs `main` (vérifier sur un run court
      / un test de reproductibilité).
- [ ] `python models/nsdiff_model.py --ticker SPY` tourne sur CPU, sort des métriques finies,
      IC bien ordonné ; budget temps mesuré et noté (comparé à TSDiff).
- [ ] Contrôle qualitatif : sur **1 actif × 1 graine**, comparer la couverture de l'IC
      (PI Cov 95 %) et le CRPS empirique de NsDiff vs TSDiff — attendu : calibration au
      moins comparable, idéalement meilleure (c'est la thèse). Consigner le résultat dans
      une `NOTE_*.md` courte (comme les autres briques du repo).
- [ ] Décisions d'archi (backbone de la moyenne dégraissé/remplacé, budget d'époques,
      defaults CPU) **déclarées** en tête de `nsdiff_model.py`.

---

## 7. Ordre d'exécution recommandé

1. Lire les fichiers d'ancrage : `models/tsdiff_model.py`, `models/tsdiff_weekly.py`,
   `models/test_tsdiff_model.py`, `benchmarks/multi_horizon.py` (surtout
   `forecast_horizons_tsdiff` + `MODEL_ADAPTERS`), `experiments/duel_backtest.py`
   (fit/sample TSDiff : lignes ~91, 96, 146-159, 197), `experiments/duel_sampling_adapters.py`
   (patron `fit_*_state`/`*_trajectory_samples`), `experiments/duel_origins.py`,
   `model_artifacts/pipeline.py` (points TSDiff du §4.3), `models/conftest.py`.
2. Inspecter/cloner `https://github.com/wwy155/NsDiff` pour porter LSNM + UANS + g_ψ
   fidèlement.
3. Écrire `models/nsdiff_model.py`, rendre `models/test_nsdiff_model.py` **vert en
   isolation** d'abord.
4. Ajouter `models/nsdiff_weekly.py` + `forecast_horizons_nsdiff` (+ registre) ; tester.
5. Câbler `model_artifacts/pipeline.py` (additif) ; relancer toute la suite.
6. Câbler le duel **derrière `--include-nsdiff`** ; vérifier la reproductibilité sans le flag.
7. Contrôle qualité 1 actif × 1 graine (calibration/CRPS vs TSDiff) + `NOTE_*.md`.
8. PR sur `feat/nsdiff-model`, commits atomiques, description reprenant les critères du §6.

---

## 8. Pièges à éviter (spécifiques à ce repo)

- **Ordre d'import TensorFlow/statsmodels** : `duel_backtest.py` documente un deadlock si
  `tensorflow` est importé après `statsmodels`/`arima` sous certains threads (import TF
  géré en tête). NsDiff est **PyTorch only** → ne rien changer à cet ordre, ne pas importer
  TF ; juste ne pas casser l'existant.
- **`HORIZON` généré ≠ horizon demandé** : comme TSDiff, le modèle génère un chemin de
  longueur fixe ; plafonner l'horizon demandé à la longueur générée (`min(h, HORIZON)`),
  lire le prix depuis le **retour cumulé** des `h` premiers pas.
- **`m` constant duel-wide** : ne jamais faire varier `n_samples` par origine dans le duel
  (le fair CRPS l'exige identique pour tous les modèles).
- **Standardisation par actif** : `mu/sd` calculés **par actif**, gelés à T0 (pas de
  pooling — le pooling multi-actifs dégrade le CRPS, c'est un résultat établi du repo).
- **Ne pas “améliorer” les baselines ou TSDiff au passage** : périmètre strict = ajouter
  NsDiff.

---

*Fichier compagnon : `NOTE_justification_NsDiff` (pourquoi ce modèle). Comparatif complet
des 9 candidats : `CHOIX_second_modele_diffusion.md`.*
