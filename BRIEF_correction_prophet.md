# BRIEF — Correction : `run_prophet` doit être walk-forward 1-pas

## 0. Contexte & cause racine (confirmée par lecture du code)

Prophet OOS sur-estime fortement sur les actifs en tendance (audit : **BTC +11,8 %**, **ETH
+21,8 %**), quasi nul sur les actifs plats (SPY, TLT, ZN=F). Ce n'est **pas** un bug de colonne
ou d'échelle. La cause est dans `models/prophet_model.py`, fonction `run_prophet()` (l.110-131) :

```python
model.fit(df_train)                       # UN seul fit sur le train
forecast = model.predict(df_future)       # predict de TOUTE la fenêtre de test d'un coup
preds = forecast["yhat"].values
```

Prophet est donc **le seul modèle du repo qui ne fait pas de walk-forward**. Tous les autres
(`run_sarima`, `run_arima_garch`, `run_lstm`, `run_tsdiff`) bouclent `for i in range(len(test))`,
prédisent **1 pas**, puis avancent avec la valeur réalisée. `run_prophet`, lui, produit une
prévision à *k pas* : `preds[t]` est extrapolé depuis la fin du train, pas depuis `t-1`. Sur un
actif qui monte, la tendance ajustée est projetée sur ~100 jours → dérive croissante → overshoot.

Le pipeline aggrave l'effet en toute bonne foi : `_run_model_d1` (`model_artifacts/pipeline.py`
l.413-423) appelle `run_prophet(train, validation)` **exactement comme les autres** et sa
docstring (l.600) qualifie la sortie D1 de « walk-forward 1-step ». `sim_trades.build_oos_prediction_rows`
l'ingère ensuite comme une prédiction `D → D+1` (`reference_price = actual[t-1]`). L'étiquette
« 1 jour » est donc fausse pour Prophet uniquement.

**Preuve que c'est bien ça :** le **live Prophet est correct** (±1,5 %) parce qu'il passe par
`next_step_prophet()` (l.134-148), qui fait un fit + un predict d'**une seule** date suivante —
une vraie prévision à 1 pas. Seul le chemin OOS (`run_prophet`, batch) est cassé.

## 1. Objectif

Réécrire `run_prophet()` en **walk-forward rolling 1-pas**, à l'identique du patron de
`run_sarima()`, **sans changer sa signature ni le dict retourné** (pour ne toucher ni au pipeline
ni au format `predictions.parquet`). `next_step_prophet()` est **déjà correct** → ne pas y toucher.

## 2. Garde-fous

- Branche dédiée : `maeva/correction-prophet` (sans accent).
- Ne modifier **que** `models/prophet_model.py` (fonction `run_prophet`) + son test.
- **Ne pas toucher** au pipeline, aux autres modèles, à `sim_trades.py`, `tracking_db.py`.
- Contrat de retour **inchangé** : dict avec `predictions`, `lower`, `upper`, `index`, `actual`
  + les clés de `compute_metrics`. Mêmes longueurs, alignées point à point sur `test.index`.

## 3. Le correctif — `run_prophet` walk-forward

Remplacer le corps de `run_prophet` par une boucle rolling 1-pas (miroir de `run_sarima`, l.90-119) :

```python
def run_prophet(train: pd.Series, test: pd.Series, refit_freq: int = 1) -> dict:
    """Rolling 1-step-ahead Prophet forecast over the test window (walk-forward),
    aligné sur run_sarima : à chaque pas, (ré)ajuste Prophet sur l'historique connu
    jusqu'à t-1 et ne prend QUE la date suivante (yhat/yhat_lower/yhat_upper).
    `refit_freq` : réajuste tous les N pas (1 = à chaque pas, le plus correct)."""
    t0 = time.time()
    history_ds = list(pd.to_datetime(train.index))
    history_y  = list(train.astype(float).values.flatten())
    preds, lower, upper = [], [], []

    model = None
    for i in range(len(test)):
        if model is None or (i % refit_freq == 0):
            df_train = pd.DataFrame({"ds": history_ds, "y": history_y})
            model = Prophet(interval_width=1 - PI_ALPHA, daily_seasonality=False,
                            weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df_train)

        next_ds = pd.to_datetime(test.index[i])
        fc = model.predict(pd.DataFrame({"ds": [next_ds]}))
        preds.append(float(fc["yhat"].iloc[0]))
        lower.append(float(fc["yhat_lower"].iloc[0]))
        upper.append(float(fc["yhat_upper"].iloc[0]))

        # walk forward : on révèle la valeur réalisée du pas i
        history_ds.append(next_ds)
        history_y.append(float(test.iloc[i]))

    train_time = time.time() - t0
    preds, lower, upper = map(np.array, (preds, lower, upper))
    metrics = compute_metrics(test.values, preds, pi_lower=lower, pi_upper=upper,
                              train_time=train_time)
    return {**metrics, "predictions": preds, "lower": lower, "upper": upper,
            "index": test.index, "actual": test.values}
```

Points d'attention :
- **`refit_freq=1` par défaut** = réajustement à chaque pas, c'est le comportement honnête (comme
  SARIMA). Un `refit_freq > N` accélère mais **réintroduit une dérive** proportionnelle à l'écart
  (entre deux réajustements, la prédiction redevient du k-pas) : à documenter, ne pas mettre >1 par
  défaut. Mentionner dans la docstring/CLI.
- **Coût :** Prophet réajusté à chaque pas est lent (backend Stan). Sur ~100 jours de test × N
  actifs, prévoir plusieurs minutes par combo. C'est le prix de la correction ; si c'est
  rédhibitoire, discuter d'une fenêtre glissante bornée (garder les `k` derniers points
  d'historique) plutôt que d'augmenter `refit_freq` — mais par défaut on reste sur l'expanding
  window exact.
- Mettre à jour l'entête du module (l.8-9 : « fitted once … predicting … in one batch ») et le
  titre du plot (l.162) qui décrivent l'ancien comportement.

## 4. Tests (`models/test_prophet_model.py`)

- **Non-régression de dérive** : sur une série **fortement tendancielle** synthétique (ex.
  croissance ~+0,5 %/jour + bruit), l'ancien `run_prophet` (batch) produit un biais moyen
  `mean(pred - actual)` largement positif et croissant ; le nouveau doit avoir un biais proche de
  0 et une MAE nettement plus faible. Vérifier `mean(pred - actual)` ≈ 0 (borne à fixer, p.ex.
  |biais| < 1 % du niveau moyen).
- **Contrat de retour** : longueurs `predictions/lower/upper/index/actual` == `len(test)`,
  `lower <= predictions <= upper` point à point, clés de metrics présentes.
- **Cohérence 1-pas** : le dernier point du walk-forward doit égaler (à tolérance numérique)
  `next_step_prophet(série jusqu'à l'avant-dernier point)` — les deux sont le même calcul.
- **Rapidité du test** : petite série + `yearly_seasonality=False` en test si besoin pour tenir
  le temps CI (le test valide la *logique* walk-forward, pas la précision de Prophet).

## 5. Régénération des données OOS Prophet (Partie 2 — à cadrer)

⚠️ **Le correctif de code ne répare pas les lignes déjà en base.** Les `Run/*-Prophet-*-D1/predictions.parquet`
et les lignes `predictions` (source='oos', model='Prophet') existantes contiennent toujours les
prévisions batch erronées. Pour en bénéficier il faut :

1. **Régénérer** les backtests Prophet D1 via le pipeline (⟹ réécrit `predictions.parquet`).
   Nécessite réseau (yfinance) + compute.
2. **Ré-ingérer** dans `predictions`. **Piège :** `sim_trades.insert_oos_predictions` fait
   `INSERT OR IGNORE` sur la clé `(source, run_id, model, asset, horizon, cutoff_date)` — si le
   `run_id` (= nom du dossier Run) est identique, **les anciennes lignes ne seront PAS écrasées**.
   Il faut donc, avant ré-ingestion, **supprimer les lignes OOS Prophet existantes** (et leurs
   `sim_trades` OOS Prophet), ou régénérer dans des dossiers `Run` datés du jour.
3. **Rejouer** ensuite `flag_daily_duplicates()` puis `reconcile_oos_sim_trades()` (briefs
   précédents) pour re-flaguer et réconcilier proprement.

Vu réseau + compute + suppression ciblée, **je recommande d'en faire un brief séparé** une fois le
code validé et testé. Ce brief-ci s'arrête à : **code corrigé + tests qui prouvent le
walk-forward**. Dire si tu veux que j'y ajoute directement la partie régénération.

## 6. Vérification finale (après §5, si régénéré)

- Biais Prophet OOS par actif **retombe au même ordre que les autres modèles** : ETH n'est plus
  à +21,8 %, BTC plus à +11,8 %.
- Couverture des IC 95 % Prophet OOS redevenue plausible (~95 %, pas écrasée par la dérive).
- Aucun autre modèle affecté (on n'a touché qu'à `run_prophet`).

## 7. Hors périmètre

- Retrait de `run_id` de l'index d'unicité OOS (prévention native des doublons) — décision
  ultérieure, non requise tant qu'on garde l'approche par flag.
