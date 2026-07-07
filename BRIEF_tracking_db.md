# BRIEF — Partie B : Base de suivi des prédictions (`tracking_db.py`)

## 0. Contexte & périmètre

La couche de 4 modèles (ARIMA, SARIMA, Prophet, LSTM) est finalisée et testée.
On construit maintenant la **validation métier** : générer des prédictions sur plusieurs
actifs/horizons (les *test cases*), les exécuter, et **suivre leurs résultats dans une base**.

Répartition **producteur / consommateur** :
- **Partie A (Kyrio)** = le *benchmark* : `validation_runner.py` génère les prédictions et les verdicts.
- **Partie B (Maéva, ce brief)** = la *base de suivi* : stocke les résultats, les évalue a posteriori, produit le reporting de validation.

Le **seul point de contact** entre A et B est la fonction `save_prediction(record)` et le format d'enregistrement (« le contrat », §3). B ne touche à rien d'autre du code de A.

## 1. Objectif

Livrer un module `tracking_db.py` (SQLite, bibliothèque standard uniquement) qui :
1. crée et gère une base de suivi des prédictions ;
2. stocke sans doublon les enregistrements produits par la partie A ;
3. évalue chaque prédiction une fois la vraie valeur connue (métriques métier) ;
4. produit un reporting agrégé, **avec découpage par régime de marché**.

## 2. Organisation git

- Travailler sur une **branche dédiée** : `maeva/tracking-db` (éviter les accents dans le nom de branche).
- **Ne pas modifier** `run_benchmark.py`, les fichiers de modèles, ni `validation_runner.py` (partie A de Kyrio).
- Nouveau fichier isolé : `tracking_db.py` (+ son test). Aucun conflit possible avec la partie A.

## 3. Le contrat d'interface avec la partie A (« le record »)

La partie A appelle `save_prediction(record)` où `record` est un dict avec **exactement** ces champs :

| Champ | Type | Sens |
|-------|------|------|
| `run_id` | str | identifiant du run de benchmark |
| `tc_id` | str | identifiant du test case (TC1..TC8) |
| `model` | str | arima / sarima / prophet / lstm |
| `asset` | str | ticker (BTC-USD, GC=F, SPY, QQQ) |
| `horizon` | int | 1 ou 7 (jours de bourse) |
| `cutoff_date` | str (ISO) | date de coupure `T` (dernière donnée d'entraînement) |
| `target_date` | str (ISO) | date prédite `T+H` |
| `regime` | str | régime de marché actif à `T` (ou `"unknown"` si moteur non prêt) |
| `last_close` | float | prix à `T` (= prévision naïve) |
| `y_pred` | float | prévision ponctuelle |
| `y_lower` | float | borne basse IC 95 % |
| `y_upper` | float | borne haute IC 95 % |
| `verdict_integrite` | int (0/1) | verdict niveau 1 (calculé par A) |
| `verdict_plausibilite` | int (0/1) | verdict niveau 2 (calculé par A) |
| `created_at` | str (ISO) | horodatage de création |

Les champs d'évaluation (`y_true`, `in_interval`, etc.) sont **remplis par B**, pas par A.

## 4. Schéma de la base (SQLite)

Deux tables. `CREATE TABLE IF NOT EXISTS` (idempotent).

**Table `test_cases`** (les 8 cas de référence) : `tc_id` (PK), `asset`, `horizon`, `description`.

**Table `predictions`** :
- clé technique `id` INTEGER PK AUTOINCREMENT ;
- tous les champs du contrat (§3) ;
- champs d'évaluation : `y_true`, `in_interval`, `abs_error`, `abs_error_naif`, `beats_naif`, `direction_correct`, `evaluated_at` ;
- **contrainte d'unicité** `UNIQUE (tc_id, model, cutoff_date)` → garantit l'idempotence.

## 5. Fonctions à implémenter (API du module)

```python
init_db(db_path="tracking.db") -> None
    # crée les tables si absentes

register_test_case(tc_id, asset, horizon, description="", db_path=...) -> None
    # upsert d'un cas de référence (ON CONFLICT(tc_id) DO UPDATE)

save_prediction(record: dict, db_path=...) -> bool
    # valide la présence de tous les champs du contrat (sinon ValueError)
    # auto-enregistre le test_case correspondant
    # INSERT OR IGNORE (idempotent sur tc_id, model, cutoff_date)
    # retourne True si insertion, False si doublon ignoré

evaluate_pending(price_fetcher, db_path=..., today=None) -> int
    # pour chaque prédiction où y_true IS NULL ET target_date <= today :
    #   y_true = price_fetcher(asset, target_date)
    #   si None -> on saute (donnée pas encore dispo), on réessaiera
    #   sinon -> calcule les métriques (§6) et UPDATE
    # retourne le nombre de prédictions évaluées

report(group_by=("model",), db_path=...) -> list[dict]
    # agrégation ; group_by parmi {model, asset, horizon, regime}
    # par groupe : n_total, n_evalues, taux_integrite, taux_plausibilite,
    #              couverture_ic95, taux_bat_naif, exactitude_dir

export_csv(path, db_path=...) -> int
    # dump de la table predictions en CSV ; retourne le nb de lignes
```

## 6. Logique d'évaluation (définitions des métriques) — IMPORTANT

Pour une prédiction évaluée (avec `y_true` connu), calculer :

- `in_interval` = `1` si `y_lower <= y_true <= y_upper`, sinon `0` (calibration IC 95 %).
- `abs_error` = `|y_true - y_pred|` (erreur du modèle).
- `abs_error_naif` = `|y_true - last_close|` (erreur de la prévision naïve = marche aléatoire).
- `beats_naif` = `1` si `abs_error <= abs_error_naif`, sinon `0` (**skill** : le modèle bat-il le naïf ?).
- `direction_correct` = `1` si `signe(y_pred - last_close) == signe(y_true - last_close)`, sinon `0`.

> Rationale quant : sur un prix quasi-marche-aléatoire, un seuil d'erreur absolu ne veut rien dire. La validation repose sur (a) la **calibration** des intervalles et (b) le **skill vs naïf**, pas sur une erreur brute.

## 7. Dimension régime

- `regime` est stocké tel quel (fourni par A ; `"unknown"` tant que le moteur de détection n'est pas prêt).
- Le reporting **doit** pouvoir grouper par `regime` (ex. `report(group_by=("model", "regime"))`) : un modèle peut être bon en régime calme et mauvais en régime de stress. C'est un axe d'analyse clé de la validation métier.
- B ne calcule pas le régime : il ne fait que le stocker et le restituer.

## 8. Récupération des cours réels (`price_fetcher`)

`evaluate_pending` reçoit une fonction `price_fetcher(asset, target_date) -> float | None` :
- **découplage volontaire** : le module reste testable hors-ligne (on injecte un mock).
- Côté machine, fournir une implémentation basée sur **yfinance** qui renvoie la clôture ajustée à `target_date`, ou `None` si indisponible (week-end, férié, pas encore de donnée). La gérer dans un fichier d'usage séparé, pas en dur dans `tracking_db.py`.

## 9. Tests exigés (hors-ligne, déterministes)

Créer `test_tracking_db.py` (pytest) couvrant :
1. `init_db` crée les tables ; ré-appel sans erreur (idempotent).
2. `save_prediction` insère ; **doublon** (même tc_id/model/cutoff_date) → retourne `False`, pas de 2ᵉ ligne.
3. `save_prediction` sur record incomplet → `ValueError`.
4. `evaluate_pending` avec un **mock price_fetcher** : vérifier le calcul exact de `in_interval`, `abs_error`, `abs_error_naif`, `beats_naif`, `direction_correct` sur des cas connus.
5. `evaluate_pending` ignore les prédictions non échues (`target_date > today`) et celles dont le fetcher renvoie `None`.
6. `report` : agrégations correctes sur un petit jeu, y compris `group_by=("model","regime")`.
7. Base temporaire (`tmp_path` de pytest), **aucun accès réseau**.

## 10. Contraintes

- `tracking_db.py` : **bibliothèque standard uniquement** (`sqlite3`, `csv`) — pas de pandas, pas de réseau.
- Idempotence stricte : rejouer un run ne duplique rien.
- Ne jamais ré-évaluer une prédiction déjà évaluée (`y_true IS NULL` dans le filtre).
- Dates au format ISO `YYYY-MM-DD` (comparables comme chaînes).

## 11. Livrables

1. `tracking_db.py` (le module).
2. `test_tracking_db.py` (les tests, tous verts, hors-ligne).
3. Un court `usage_tracking_db.py` ou section README montrant : `init_db` → `save_prediction` (2-3 records d'exemple) → `evaluate_pending` (avec un fetcher yfinance) → `report`.
4. 2-3 **records d'exemple** conformes au contrat (utiles aussi à Kyrio pour tester son côté).

## 12. Checklist d'acceptation

- [ ] Branche `maeva/tracking-db`, aucun fichier de la partie A modifié.
- [ ] Les 2 tables créées, contrainte d'unicité en place.
- [ ] `save_prediction` idempotent (test doublon vert).
- [ ] Les 5 métriques du §6 calculées exactement (tests verts).
- [ ] `report` groupe par model / asset / horizon / regime.
- [ ] `test_tracking_db.py` passe à 100 %, hors-ligne.
- [ ] `price_fetcher` yfinance isolé hors du module cœur.
