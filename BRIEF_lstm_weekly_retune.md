# BRIEF — LSTM weekly : re-réglage équitable des hyperparamètres (pas une chasse au bug)

> **Cadre honnête, à lire avant de coder.** Le diagnostic préalable a établi qu'il n'y a
> **PAS de bug** dans le LSTM weekly : 900 lignes complètes, 0 valeur manquante,
> couverture 95 % = 0,889 (≈ le daily 0,911), erreurs relatives raisonnables. Le LSTM
> weekly est seulement **modérément moins bon** que le daily→hebdo (CRPS_norm poolé
> +0,74, significatif après Holm — cf. `experiments/METHODOLOGIE_weekly_vs_daily.md`).
> Ce brief ne « répare » donc rien : il rend la **comparaison équitable** en donnant au
> LSTM des hyperparamètres adaptés au régime hebdo, choisis proprement. **Il est
> explicitement acceptable que le résultat ne change pas** : si, à réglage équitable, le
> LSTM reste moins bon en weekly, c'est un vrai verdict à documenter, pas un échec.

## 0. Cause identifiée (vérifiée dans le code)

`models/lstm_model.py` : `SEQ_LEN = 30`, `EPOCHS = 30`, **en dur**.
`benchmarks/multi_horizon.py::forecast_horizons_lstm` les utilise **à l'identique pour
daily ET weekly** (`weekly_multimodel.py` ligne ~31 assume explicitement « LSTM works the
same whether lags are days or weeks » et ne définit pas de variante weekly, contrairement
à SARIMA/Prophet qui en ont une).

Problème : en hebdo, `SEQ_LEN=30` = un lookback de **30 semaines (~7 mois)** sur une série
resamplée ~5× plus courte qu'en daily. Le lookback et le budget d'epochs ne sont pas
forcément adaptés → sous-apprentissage plausible. C'est un **réglage aveugle au régime**,
pas une erreur de calcul.

## 1. Objectif

Donner au LSTM, **pour le régime C (hebdo natif)**, un `SEQ_LEN` (et éventuellement un
`EPOCHS`) **sélectionné sur un jeu de validation**, selon la même discipline que le
balayage d'epochs de TSDiff (`experiments/epoch_sweep.py`). Puis re-générer les
prédictions LSTM weekly avec le réglage retenu et re-tester B vs C.

## 2. Décision de conception assumée

- On rend `SEQ_LEN`/`EPOCHS` **paramétrables par régime** dans le chemin LSTM, sans
  toucher au comportement daily (défauts inchangés → non-régression daily garantie).
- On sélectionne le réglage weekly par **CRPS de validation**, jamais sur le test.
  Interdiction absolue de choisir en regardant les origines de test (= p-hacking).
- Candidats raisonnables : `SEQ_LEN ∈ {8, 12, 16, 26}` (semaines), `EPOCHS ∈ {30, 60}`.
  Extensible, mais rester parcimonieux (peu de données weekly → peu de puissance).

## 3. Garde-fous (non négociables)

- **Sélection sur validation, jamais sur test.** Découpage chronologique strict
  entraînement < validation < test (mêmes principes que `BRIEF_weekly_prediction_v2.md`
  §3). Le bloc de validation ne sert QU'À choisir `SEQ_LEN*`/`EPOCHS*`.
- **Résultat non forcé.** On rapporte le verdict re-testé quel qu'il soit. Si le LSTM
  weekly reste significativement moins bon, on l'écrit — pas de re-balayage jusqu'à
  inverser le signe.
- **Non-régression daily.** Les défauts `SEQ_LEN=30`/`EPOCHS=30` du daily restent
  intacts ; les tests LSTM daily existants passent inchangés.
- **Même protocole que le reste du weekly** : resample `W-FRI`, refit par origine (comme
  `weekly_multimodel.py` régime C), pas de lookahead, seed fixe.
- **Insertion en base** via le chemin établi (`sim_trades.insert_oos_predictions`),
  `frequence='weekly'`, `horizon_type='weekly'`, `real_flag` recalculé — remplace les
  anciennes lignes LSTM weekly (upsert sur la clé de conflit).

## 4. Plan d'implémentation

1. **Paramétrer** `forecast_horizons_lstm` (et `fit_lstm`) pour accepter `seq_len` et
   `epochs` explicites, défauts = valeurs actuelles (aucun changement daily).
2. **Balayage validation** : pour chaque actif, sur le bloc de validation weekly,
   entraîner le LSTM régime C aux candidats `SEQ_LEN × EPOCHS`, mesurer le **CRPS de
   validation**. Sélectionner `(SEQ_LEN*, EPOCHS*)` = argmin CRPS validation, par actif.
   Sauver dans `experiments/lstm_weekly_sweep.json`.
3. **Re-génération** des prédictions LSTM régime C sur les origines de test avec le
   réglage retenu ; upsert en base.
4. **Re-test** B vs C pour le LSTM en relançant `experiments/weekly_vs_daily_pooled.py`
   (aucune modif du script : il relit la base). Comparer l'ancien (-0,74) et le nouveau.
5. **Documenter** le résultat re-testé dans `experiments/METHODOLOGIE_weekly_vs_daily.md`
   (section LSTM) : réglage retenu par actif + verdict, honnête quel qu'il soit.

## 5. Critère de succès

- Le LSTM weekly a un `SEQ_LEN`/`EPOCHS` **choisi sur validation**, tracé et reproductible.
- `weekly_vs_daily_pooled.py` relancé donne un verdict LSTM re-testé, **documenté tel
  quel** (amélioré, neutre ou toujours défavorable).
- Le comportement et les tests LSTM **daily** sont inchangés.

## 6. Ce que ce brief ne fait PAS

- Ne « répare » aucun bug (il n'y en a pas : données LSTM weekly complètes).
- Ne force pas un LSTM weekly gagnant ; n'itère pas jusqu'à inverser le signe.
- Ne touche pas au LSTM daily, ni aux 4 autres modèles, ni à TSDiff.
- Ne change pas la méthode statistique (déjà validée).

## 7. Références

| Quoi | Où |
|---|---|
| Constats du diagnostic + verdict poolé | `experiments/METHODOLOGIE_weekly_vs_daily.md` |
| Réglages LSTM en dur (à paramétrer) | `models/lstm_model.py` (`SEQ_LEN`, `EPOCHS`), `benchmarks/multi_horizon.py` (`forecast_horizons_lstm`, `fit_lstm`) |
| Génération weekly multi-modèles (régime C) | `experiments/weekly_multimodel.py` |
| Discipline de balayage (modèle à suivre) | `experiments/epoch_sweep.py`, `BRIEF_weekly_prediction_v2.md` §3-4 |
| Re-test (relancer tel quel) | `experiments/weekly_vs_daily_pooled.py` |
