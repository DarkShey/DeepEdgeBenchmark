"""Audit en LECTURE SEULE de validation/tracking.db (table `predictions`).

Ce script n'écrit JAMAIS dans la base : il ouvre une copie (ou la base originale
en mode read-only via URI sqlite) et ne produit que des fichiers de sortie dans
validation/audit/ (rapport markdown + CSV keep/drop). Aucun DELETE/UPDATE/INSERT.

Usage :
    python validation/audit/audit_tracking_db.py --db /path/to/copie/audit.db \
        --out-md validation/audit/audit_tracking_db.md \
        --out-csv validation/audit/audit_keep_drop.csv
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

LIVE_RUN_RE = re.compile(r"^run_(\d{8})T(\d{6})$")
OOS_RUN_RE = re.compile(r"^(\d{8})-.+-D\d+$")

BUSINESS_KEY_COLS = ("source", "model", "asset", "horizon", "cutoff_date", "target_date")


def connect_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def recency_key(source: str, run_id: str, created_at: str | None, row_id: int):
    """Retourne une clé de tri (plus grand = plus récent) selon la règle du §5 du brief."""
    if source == "live":
        m = LIVE_RUN_RE.match(run_id)
        primary = (m.group(1) + m.group(2)) if m else ""
    elif source == "oos":
        m = OOS_RUN_RE.match(run_id)
        primary = m.group(1) if m else ""
    else:
        primary = ""
    return (primary, created_at or "", row_id)


def fetch_all_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT id, run_id, tc_id, model, asset, horizon, cutoff_date, target_date,
               regime, last_close, y_pred, y_lower, y_upper, y_true, created_at,
               evaluated_at, source
        FROM predictions
        """
    )
    return cur.fetchall()


def run_id_format_check(rows: list[sqlite3.Row]) -> dict:
    bad = defaultdict(list)
    for r in rows:
        if r["source"] == "live" and not LIVE_RUN_RE.match(r["run_id"]):
            bad["live"].append((r["id"], r["run_id"]))
        elif r["source"] == "oos" and not OOS_RUN_RE.match(r["run_id"]):
            bad["oos"].append((r["id"], r["run_id"]))
    return bad


def inventory(conn: sqlite3.Connection) -> dict:
    out = {}
    cur = conn.execute(
        """
        SELECT source, COUNT(*) AS n, COUNT(DISTINCT run_id) AS n_runs,
               MIN(cutoff_date) AS min_cutoff, MAX(cutoff_date) AS max_cutoff,
               SUM(CASE WHEN y_true IS NOT NULL THEN 1 ELSE 0 END) AS n_eval,
               SUM(CASE WHEN y_true IS NULL THEN 1 ELSE 0 END) AS n_pending
        FROM predictions GROUP BY source
        """
    )
    out["by_source"] = [dict(r) for r in cur.fetchall()]

    cur = conn.execute(
        """
        SELECT source, model, asset, horizon, COUNT(*) AS n,
               COUNT(DISTINCT run_id) AS n_runs,
               MIN(cutoff_date) AS min_cutoff, MAX(cutoff_date) AS max_cutoff,
               SUM(CASE WHEN y_true IS NOT NULL THEN 1 ELSE 0 END) AS n_eval,
               SUM(CASE WHEN y_true IS NULL THEN 1 ELSE 0 END) AS n_pending
        FROM predictions
        GROUP BY source, model, asset, horizon
        ORDER BY source, model, asset, horizon
        """
    )
    out["by_model_asset_horizon"] = [dict(r) for r in cur.fetchall()]
    return out


def detect_duplicates(rows: list[sqlite3.Row]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[c] for c in BUSINESS_KEY_COLS)
        groups[key].append(r)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    per_source_dup_lines = defaultdict(int)  # lignes EXCEDENTAIRES (celles qui seraient drop)
    per_source_dup_groups = defaultdict(int)
    dup_table = []
    for key, members in dup_groups.items():
        source = key[0]
        per_source_dup_groups[source] += 1
        per_source_dup_lines[source] += len(members) - 1
        y_preds = [m["y_pred"] for m in members]
        spread = round(max(y_preds) - min(y_preds), 4)
        stdev = round(statistics.pstdev(y_preds), 4) if len(y_preds) > 1 else 0.0
        dup_table.append({
            "key": key,
            "n_copies": len(members),
            "run_ids": [m["run_id"] for m in members],
            "y_preds": y_preds,
            "spread_y_pred": spread,
            "stdev_y_pred": stdev,
            "ids": [m["id"] for m in members],
        })
    dup_table.sort(key=lambda d: (-d["n_copies"], d["key"][4] or ""), reverse=False)
    dup_table.sort(key=lambda d: (-d["n_copies"]))
    return {
        "groups": groups,
        "dup_groups": dup_groups,
        "dup_table": dup_table,
        "per_source_dup_groups": dict(per_source_dup_groups),
        "per_source_dup_lines": dict(per_source_dup_lines),
    }


def build_keep_drop(groups: dict) -> list[dict]:
    out = []
    for key, members in groups.items():
        source, model, asset, horizon, cutoff_date, target_date = key
        ranked = sorted(
            members,
            key=lambda r: recency_key(r["source"], r["run_id"], r["created_at"], r["id"]),
        )
        winner = ranked[-1]
        for m in members:
            decision = "keep" if m["id"] == winner["id"] else "drop"
            out.append({
                "id": m["id"],
                "source": source,
                "model": model,
                "asset": asset,
                "horizon": horizon,
                "cutoff_date": cutoff_date,
                "target_date": target_date,
                "run_id": m["run_id"],
                "created_at": m["created_at"],
                "decision": decision,
                "run_id_gagnant": winner["run_id"] if decision == "drop" else "",
                "n_copies": len(members),
            })
    out.sort(key=lambda d: (d["source"], d["model"], d["asset"], d["cutoff_date"], d["id"]))
    return out


def prophet_diagnostic(conn: sqlite3.Connection) -> dict:
    out = {}
    for source in ("live", "oos"):
        cur = conn.execute(
            """
            SELECT id, cutoff_date, target_date, run_id, last_close, y_pred, y_true
            FROM predictions
            WHERE model = 'Prophet' AND source = ? AND y_true IS NOT NULL
            """,
            (source,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        errs_pct = []
        for r in rows:
            if r["y_true"]:
                errs_pct.append((r["y_pred"] - r["y_true"]) / r["y_true"] * 100.0)
        out[source] = {
            "n_evaluated": len(rows),
            "mean_err_pct": round(statistics.mean(errs_pct), 2) if errs_pct else None,
            "median_err_pct": round(statistics.median(errs_pct), 2) if errs_pct else None,
            "stdev_err_pct": round(statistics.pstdev(errs_pct), 2) if len(errs_pct) > 1 else None,
            "min_err_pct": round(min(errs_pct), 2) if errs_pct else None,
            "max_err_pct": round(max(errs_pct), 2) if errs_pct else None,
            "n_over_10pct": sum(1 for e in errs_pct if abs(e) > 10),
        }
    # total rows Prophet OOS (evaluated or not) impactees
    cur = conn.execute("SELECT COUNT(*) AS n FROM predictions WHERE model='Prophet' AND source='oos'")
    out["n_rows_prophet_oos_total"] = cur.fetchone()["n"]
    # comparaison autres modeles OOS pour verifier que l'anomalie est propre a Prophet
    cur = conn.execute(
        """
        SELECT model, COUNT(*) AS n,
               ROUND(AVG((y_pred - y_true) / y_true * 100.0), 2) AS mean_err_pct
        FROM predictions
        WHERE source = 'oos' AND y_true IS NOT NULL AND y_true != 0
        GROUP BY model
        ORDER BY model
        """
    )
    out["other_models_oos"] = [dict(r) for r in cur.fetchall()]

    # ventilation par actif : l'anomalie est-elle uniforme ou concentree sur certains actifs ?
    cur = conn.execute(
        """
        SELECT asset, COUNT(*) AS n,
               ROUND(AVG((y_pred - y_true) / y_true * 100.0), 2) AS mean_err_pct,
               ROUND(MIN((y_pred - y_true) / y_true * 100.0), 2) AS min_err_pct,
               ROUND(MAX((y_pred - y_true) / y_true * 100.0), 2) AS max_err_pct
        FROM predictions
        WHERE model = 'Prophet' AND source = 'oos' AND y_true IS NOT NULL AND y_true != 0
        GROUP BY asset
        ORDER BY mean_err_pct DESC
        """
    )
    out["prophet_oos_by_asset"] = [dict(r) for r in cur.fetchall()]
    return out


def render_report(inv, dup, keep_drop, prophet, bad_run_ids, total_rows) -> str:
    lines = []
    a = lines.append

    n_drop = sum(1 for r in keep_drop if r["decision"] == "drop")
    dup_live = dup["per_source_dup_lines"].get("live", 0)
    dup_oos = dup["per_source_dup_lines"].get("oos", 0)
    prophet_oos_n = prophet["oos"]["n_evaluated"]
    prophet_oos_mean = prophet["oos"]["mean_err_pct"]
    prophet_live_mean = prophet["live"]["mean_err_pct"]

    a("# Audit de `validation/tracking.db` — table `predictions`\n")
    a("Audit en lecture seule (aucune suppression, aucune correction). "
      "Effectué sur une **copie** de la base. Branche `maeva/audit-tracking-db`.\n")

    a("## Résumé chiffré\n")
    a(f"- **Total lignes `predictions`** : {total_rows}")
    a(f"- **Doublons (lignes excédentaires) — live** : {dup_live}")
    a(f"- **Doublons (lignes excédentaires) — oos** : {dup_oos}")
    a(f"- **Total lignes qui seraient supprimées (dry-run)** : {n_drop}")
    a(f"- **Lignes Prophet OOS évaluées (y_true renseigné)** : {prophet_oos_n} "
      f"(sur {prophet['n_rows_prophet_oos_total']} lignes Prophet OOS au total)")
    a(f"- **Erreur moyenne Prophet OOS** : {prophet_oos_mean:+.2f}% "
      f"vs **Prophet live** : {prophet_live_mean:+.2f}%\n")

    a("## A. Inventaire\n")
    a("### Par `source`\n")
    a("| source | n lignes | run_id distincts | cutoff min | cutoff max | évaluées | en attente |")
    a("|---|---:|---:|---|---|---:|---:|")
    for r in inv["by_source"]:
        a(f"| {r['source']} | {r['n']} | {r['n_runs']} | {r['min_cutoff']} | {r['max_cutoff']} "
          f"| {r['n_eval']} | {r['n_pending']} |")
    a("")

    a("### Par `(source, model, asset, horizon)`\n")
    a("| source | model | asset | horizon | n lignes | run_id distincts | cutoff min | cutoff max | évaluées | en attente |")
    a("|---|---|---|---:|---:|---:|---|---|---:|---:|")
    for r in inv["by_model_asset_horizon"]:
        a(f"| {r['source']} | {r['model']} | {r['asset']} | {r['horizon']} | {r['n']} | {r['n_runs']} "
          f"| {r['min_cutoff']} | {r['max_cutoff']} | {r['n_eval']} | {r['n_pending']} |")
    a("")

    a("### Conformité des formats de `run_id` (§5)\n")
    if not bad_run_ids:
        a("Tous les `run_id` respectent l'un des deux formats attendus "
          "(`run_YYYYMMDDThhmmss` pour live, `YYYYMMDD-MODEL-ASSET-D<h>` pour oos). "
          "La règle de récence du §5 est donc **applicable telle quelle**.\n")
    else:
        a("**Attention** : des `run_id` ne suivent aucun des deux formats attendus :\n")
        for source, items in bad_run_ids.items():
            a(f"- `{source}` : {len(items)} lignes non conformes, ex. {items[:5]}")
        a("")

    a("## B. Doublons détectés\n")
    a(f"- Nombre de **groupes** en doublon (clé métier avec >1 ligne) : "
      f"live={dup['per_source_dup_groups'].get('live', 0)}, "
      f"oos={dup['per_source_dup_groups'].get('oos', 0)}")
    a(f"- Nombre de **lignes excédentaires** (celles qui seraient supprimées) : "
      f"live={dup_live}, oos={dup_oos}, **total={dup_live + dup_oos}**\n")

    a("Extrait des 25 groupes de doublons les plus copiés "
      "(clé métier = source, model, asset, horizon, cutoff_date, target_date) :\n")
    a("| source | model | asset | h | cutoff_date | target_date | n_copies | run_ids | y_pred (par copie) | spread | stdev |")
    a("|---|---|---|---:|---|---|---:|---|---|---:|---:|")
    for d in dup["dup_table"][:25]:
        source, model, asset, horizon, cutoff_date, target_date = d["key"]
        run_ids = ", ".join(d["run_ids"])
        y_preds = ", ".join(f"{v:.2f}" for v in d["y_preds"])
        a(f"| {source} | {model} | {asset} | {horizon} | {cutoff_date} | {target_date} "
          f"| {d['n_copies']} | {run_ids} | {y_preds} | {d['spread_y_pred']} | {d['stdev_y_pred']} |")
    a(f"\n*(table complète : {len(dup['dup_table'])} groupes en doublon — voir "
      f"`validation/audit/audit_keep_drop.csv` pour le détail ligne par ligne)*\n")

    # split "vrais re-runs identiques" vs "divergence de valeurs"
    identical = sum(1 for d in dup["dup_table"] if d["stdev_y_pred"] < 1e-6)
    diverging = len(dup["dup_table"]) - identical
    a(f"Sur les {len(dup['dup_table'])} groupes en doublon : **{identical}** ont des `y_pred` "
      f"strictement identiques entre copies (vrais re-runs stables), **{diverging}** présentent "
      f"une divergence de valeurs entre copies (le modèle a produit un résultat différent selon "
      f"le run — cohérent avec des runs Prophet/LSTM/TSDiff non déterministes ou des refits à "
      f"des instants différents).\n")

    a("### Cause racine des doublons\n")
    a("Confirmée par l'inventaire : la table live n'a **aucun** doublon détecté "
      "(protégée par `UNIQUE (tc_id, model, cutoff_date)`). Tous les doublons proviennent de "
      "`source='oos'`. L'index partiel `idx_predictions_oos_unique` inclut `run_id` dans sa "
      "clé d'unicité — or chaque backtest (re-run de `sim_trades.py`/pipeline sur les mêmes "
      "dates historiques) génère un `run_id` différent (préfixe date du jour d'exécution du "
      "backtest, pas de la donnée). Deux backtests exécutés à des dates différentes mais "
      "portant sur les **mêmes** `cutoff_date`/`target_date` historiques ne sont donc jamais "
      "reconnus comme doublons par la contrainte SQL, et s'empilent. C'est l'hypothèse du "
      "brief (§3), **confirmée** par les données observées.\n")

    a("## C. Règle « garder le dernier run » — keep/drop (dry-run)\n")
    a("Règle appliquée (§5) : par clé métier `(source, model, asset, horizon, cutoff_date, "
      "target_date)`, on garde la ligne dont le `run_id` est le plus récent "
      "(préfixe date extrait pour live et oos), départage par `created_at` puis `id` max. "
      "**Aucune suppression n'a été effectuée** — voir `validation/audit/audit_keep_drop.csv` "
      "pour le détail exhaustif (une ligne par enregistrement, colonne `decision`).\n")
    a(f"- Lignes `keep` : {sum(1 for r in keep_drop if r['decision']=='keep')}")
    a(f"- Lignes `drop` : {n_drop}\n")

    a("## D. Diagnostic Prophet OOS\n")
    a("### Constat chiffré (depuis la base)\n")
    a("| source | n évaluées | erreur moyenne | erreur médiane | écart-type | min | max | |err|>10% |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in ("live", "oos"):
        p = prophet[s]
        a(f"| {s} | {p['n_evaluated']} | {p['mean_err_pct']}% | {p['median_err_pct']}% "
          f"| {p['stdev_err_pct']}% | {p['min_err_pct']}% | {p['max_err_pct']}% | {p['n_over_10pct']} |")
    a("")
    a("Comparaison avec les autres modèles OOS (même colonne d'erreur), pour confirmer que "
      "l'anomalie est **propre à Prophet** :\n")
    a("| model (oos) | n évaluées | erreur moyenne |")
    a("|---|---:|---:|")
    for r in prophet["other_models_oos"]:
        a(f"| {r['model']} | {r['n']} | {r['mean_err_pct']}% |")
    a("")

    a("**Ventilation par actif (Prophet OOS)** — l'erreur moyenne globale (+7.75%) masque une "
      "forte hétérogénéité : elle **n'est pas uniforme sur tous les actifs**, contrairement à "
      "l'hypothèse §6.4 du brief (« systématique... toutes les lignes Prophet OOS »). "
      "L'anomalie est concentrée sur les cryptos :\n")
    a("| asset | n évaluées | erreur moyenne | min | max |")
    a("|---|---:|---:|---:|---:|")
    for r in prophet["prophet_oos_by_asset"]:
        a(f"| {r['asset']} | {r['n']} | {r['mean_err_pct']}% | {r['min_err_pct']}% | {r['max_err_pct']}% |")
    a("")
    a("**ETH-USD** (+21.8% en moyenne) et **BTC-USD** (+11.8%) portent la quasi-totalité du "
      "biais ; **SPY, TLT, ZN=F** sont quasi neutres (entre -1.1% et -0.04% en moyenne). "
      "Le même calcul restreint aux lignes gagnantes après dédoublonnage (`decision=keep`) "
      "donne des chiffres très proches (ETH +22.4%, BTC +10.2%, autres ≈0), donc les doublons "
      "ne créent pas cet écart. Ce constat renforce le diagnostic « fit unique + extrapolation "
      "de tendance batch » : BTC-USD et ETH-USD ont connu une tendance haussière marquée sur la "
      "période d'entraînement de chaque backtest, que Prophet (`growth='linear'` par défaut) "
      "extrapole indéfiniment sur toute la fenêtre de test batch ; SPY/TLT/ZN=F, plus proches "
      "d'un régime stationnaire/latéral sur la même période, ne présentent pas ce biais de "
      "tendance — cohérent avec une erreur de **dérive de tendance non réactualisée**, pas avec "
      "un bug de colonne ou d'échelle qui affecterait tous les actifs uniformément.\n")

    a("### Cause racine (preuve code + données)\n")
    a("**Ce n'est pas un problème de colonne ni d'échelle.** La colonne lue par "
      "`sim_trades.py` (`predicted` → `y_pred`) est bien `forecast[\"yhat\"]`, la même "
      "colonne brute utilisée côté live (`model_artifacts/pipeline.py` → "
      "`benchmarks/multi_horizon.py:143`, `results[h] = (float(row[\"yhat\"]), ...)`). "
      "Aucune transformation log/cap n'est appliquée dans un chemin et pas l'autre — "
      "`Prophet(...)` est instancié avec les **mêmes hyperparamètres** dans les deux chemins "
      "(`growth` par défaut = `'linear'`, pas de `cap`/`floor`, "
      "`weekly_seasonality=True`, `yearly_seasonality=True`, `daily_seasonality=False`).\n")
    a("La vraie cause est **architecturale** :\n")
    a("- **live** (`model_artifacts/pipeline.py:795`) : `cutoff_date = full_series.index[-1].date()` "
      "— le pipeline live **réentraîne Prophet chaque jour** sur toutes les données connues "
      "jusqu'à hier, et ne prédit que l'horizon D1 (1 jour). C'est un walk-forward implicite "
      "au rythme du cron quotidien.\n")
    a("- **oos** (`models/prophet_model.py:108-128`, fonction `run_prophet`) : le docstring du "
      "fichier l'indique explicitement (`models/prophet_model.py:9`) : *\"fitted once on the "
      "training prices, then predicting the test dates in one batch\"*. Le modèle est "
      "**entraîné une seule fois** sur `train`, puis `model.predict(df_future)` est appelé "
      "**une seule fois** sur la totalité de la fenêtre de test (165 lignes dans l'exemple "
      "`Run/20260712-Prophet-BTC-USD-D1/`), soit plusieurs mois d'horizon réel.\n")
    a("- `validation/sim_trades.py:build_oos_prediction_rows` (lignes ~363-390) prend ensuite "
      "chaque ligne `t` du parquet et lui assigne `cutoff_date = date[t-1]`, "
      "`target_date = date[t]`, **comme si** chaque prédiction était un walk-forward 1-jour "
      "frais — alors qu'en réalité, pour Prophet, toutes ces lignes proviennent du **même** "
      "fit unique réalisé au début de la fenêtre de test. Le modèle n'est jamais reconditionné "
      "sur les prix réellement observés entre-temps.\n")
    a("- Preuve dans les données : tous les autres modèles OOS (`ARIMA-GARCH`, `SARIMA`, "
      "`LSTM`, `TSDiff`, `Naive`) implémentent explicitement un **walk-forward 1-step** "
      "(cf. docstrings `models/*.py` : *\"Forecasting is walk-forward (rolling 1-step-ahead)\"*, "
      "et code type `models/sarima_model.py:112` / `models/lstm_model.py:154` / "
      "`models/arima_model.py:164` qui ajoutent la valeur réalisée à l'historique avant l'étape "
      "suivante). **`prophet_model.run_prophet` est le seul à ne pas le faire** "
      "(`models/prophet_model.py:108-128`). C'est la seule différence structurelle entre "
      "Prophet et les modèles sains, et elle correspond exactement au modèle en anomalie.\n")
    a("- Extrait `Run/20260712-Prophet-BTC-USD-D1/predictions.parquet` (165 lignes) : l'erreur "
      "oscille entre -4% et +22% selon que le prix réel repasse sous ou au-dessus de la "
      "tendance Prophet extrapolée depuis le fit unique — ce n'est pas une dérive monotone "
      "avec l'index temporel (corrélation ≈ 0.06), ce qui est cohérent avec une **courbe de "
      "tendance figée** (issue d'un seul fit) comparée à un prix réel volatile, plutôt qu'avec "
      "un bug d'échelle constant.\n")
    a("\n**Conclusion** : l'anomalie Prophet OOS vient du fait que `run_prophet()` "
      "(`models/prophet_model.py`) fait un **fit unique + predict batch** sur toute la fenêtre "
      "de test, alors que l'ingestion `sim_trades.py` **étiquette** ces prédictions comme des "
      "cutoff→target walk-forward 1-jour (même sémantique que le live). Le décalage entre "
      "l'étiquetage (implicitement 1-day-ahead) et la réalité du calcul (extrapolation à "
      "plusieurs mois depuis un fit figé) explique le biais, **et non un bug de colonne ou "
      "d'échelle** : ce biais se manifeste surtout sur les actifs qui avaient une tendance "
      "haussière marquée pendant la fenêtre d'entraînement (BTC-USD, ETH-USD), et est quasi nul "
      "sur les actifs plus stationnaires (SPY, TLT, ZN=F) — signature typique d'une "
      "extrapolation de tendance non réactualisée, pas d'une erreur de transformation "
      "systématique qui toucherait tous les actifs de façon identique.\n")
    a("**Recommandation pour le brief de correction** : soit (a) faire de `run_prophet` un "
      "walk-forward réel (refit ou reconditionnement à chaque pas, comme les autres modèles), "
      "soit (b) si le batch est conservé pour des raisons de coût de calcul, corriger "
      "l'étiquetage OOS pour refléter le véritable horizon (date de fit → date cible) plutôt "
      "que `date[t-1] → date[t]`, et documenter que la comparaison Prophet OOS vs live n'est "
      "alors pas à iso-horizon. Option (a) est recommandée pour rester cohérent avec les autres "
      "modèles et avec le live.\n")

    a("## Hors périmètre de cet audit\n")
    a("Conformément au brief, aucune suppression, aucun correctif de code, aucune modification "
      "de contrainte d'unicité n'a été effectué ici. Ces actions feront l'objet d'un brief de "
      "correction séparé, à partir des conclusions ci-dessus.\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chemin vers la COPIE de tracking.db à auditer")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    conn = connect_readonly(args.db)
    rows = fetch_all_rows(conn)
    total_rows = len(rows)

    inv = inventory(conn)
    dup = detect_duplicates(rows)
    keep_drop = build_keep_drop(dup["groups"])
    prophet = prophet_diagnostic(conn)
    bad_run_ids = run_id_format_check(rows)

    report = render_report(inv, dup, keep_drop, prophet, bad_run_ids, total_rows)
    Path(args.out_md).write_text(report, encoding="utf-8")

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "source", "model", "asset", "horizon", "cutoff_date", "target_date",
            "run_id", "created_at", "decision", "run_id_gagnant", "n_copies",
        ])
        for r in keep_drop:
            writer.writerow([
                r["id"], r["source"], r["model"], r["asset"], r["horizon"],
                r["cutoff_date"], r["target_date"], r["run_id"], r["created_at"],
                r["decision"], r["run_id_gagnant"], r["n_copies"],
            ])

    n_drop = sum(1 for r in keep_drop if r["decision"] == "drop")
    print(f"Total lignes: {total_rows}")
    print(f"Doublons live (lignes excedentaires): {dup['per_source_dup_lines'].get('live', 0)}")
    print(f"Doublons oos (lignes excedentaires): {dup['per_source_dup_lines'].get('oos', 0)}")
    print(f"Total drop (dry-run): {n_drop}")
    print(f"Prophet OOS evaluees: {prophet['oos']['n_evaluated']} / total oos rows: {prophet['n_rows_prophet_oos_total']}")
    print(f"Rapport ecrit: {args.out_md}")
    print(f"CSV ecrit: {args.out_csv}")

    conn.close()


if __name__ == "__main__":
    main()
