# model_artifacts/ — Générer le dashboard HTML

Deux étapes : (1) faire tourner le pipeline pour produire les artefacts dans `Run/`,
(2) générer la page HTML à partir de ces artefacts.

## 1. Prérequis

Même environnement que le reste du repo (voir [`README.md`](../README.md) à la racine) :
Python 3.9+, dépendances installées via `pip install -r requirements.txt`, et une
connexion internet (téléchargement des prix via `yfinance`).

Toutes les commandes ci-dessous s'exécutent **depuis la racine du repo**
(`DeepEdgeBenchmark/`), pas depuis `model_artifacts/`.

## 2. Étape 1 — lancer le pipeline (entraînement + validation + prévision)

```bash
python -m model_artifacts.pipeline
```

Ça fait, pour chaque combinaison (modèle × actif × horizon D+1/D+7) :
- télécharge les prix (fenêtre glissante des 3 dernières années jusqu'à **aujourd'hui**,
  date d'exécution — pas une date figée) ;
- entraîne le modèle (Gate 1) et l'évalue en backtest sur la validation (Gate 2) ;
- calcule une **vraie prévision hors-échantillon** (au-delà de la dernière clôture
  connue) pour J+1 et J+7, avec intervalle de confiance à 95% ;
- écrit tout dans `Run/<YYYYMMDD>-<modèle>-<actif>-<horizon>/` (un sous-dossier par
  combinaison, `YYYYMMDD` = date d'exécution du run).

Options utiles :

```bash
python -m model_artifacts.pipeline --assets "BTC-USD,SPY" --models "ARIMA-GARCH,Naive"
python -m model_artifacts.pipeline --horizons D1
python -m model_artifacts.pipeline --epochs 20          # LSTM plus rapide (moins précis)
```

À savoir :
- Le run complet (sans `--models`) relance deux sous-processus séparés (LSTM isolé du
  reste) — c'est volontaire, voir le commentaire dans `pipeline.py` (deadlock TensorFlow
  constaté sinon).
- Ça peut prendre plusieurs minutes (5 modèles × 5 actifs × 2 horizons, LSTM compris).
- Relancer cette commande un autre jour régénère automatiquement un nouveau
  `Run/<date du jour>-...` avec des données et une prévision fraîches — rien à changer
  dans le code pour ça.

## 3. Étape 2 — générer la page HTML

```bash
python -m model_artifacts.generate_dashboard
```

Ça lit tous les dossiers sous `Run/` et écrit `Run/dashboard.html` (un seul fichier
HTML autonome, à ouvrir directement dans un navigateur — double-clic, ou
`start Run/dashboard.html` sur Windows / `open Run/dashboard.html` sur macOS).

Options :

```bash
python -m model_artifacts.generate_dashboard --run-root Run --out Run/dashboard.html   # défaut
```

Le dashboard agrège **tous** les runs présents dans `Run/` (sélecteur "Date de run" par
actif) — pas besoin de nettoyer les anciens dossiers avant de régénérer.

## 4. Dépannage

| Symptôme | Cause / fix |
|---|---|
| Onglet "Graphique" affiche "Aucune donnée de prix pour cette date de run" | `prices.parquet`/`predictions.parquet` absents du dossier `Run/<...>` correspondant — relancer `python -m model_artifacts.pipeline` (ces fichiers ne sont pas versionnés dans git). |
| Colonnes "Prévision" / PI 95% à "—" dans les tableaux | `forecast.json` absent (run généré avant l'ajout de la prévision hors-échantillon) — relancer le pipeline pour ce run. |
| KPI "Déphasage" à "—" | Moins de 4 points de backtest disponibles pour ce modèle/horizon (normal pour un D+7 avec peu d'origines de validation). |
| Pipeline lent / semble bloqué sur LSTM | Comportement connu documenté dans `pipeline.py` (isolation par sous-processus) — patienter, ou exclure LSTM avec `--models "ARIMA-GARCH,SARIMA,Prophet,Naive"` pour itérer plus vite. |
