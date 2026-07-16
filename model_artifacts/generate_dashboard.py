"""
model_artifacts/generate_dashboard.py — Dashboard KPI à partir des artefacts de Run/
=====================================================================================
Lit tous les metrics.json (+ metadata.json, predictions.parquet, prices.parquet) produits
par model_artifacts/pipeline.py et génère une page HTML autonome affichant, par actif :
- un onglet KPIs (date de run / horizon / modèles sélectionnables, tuiles par modèle,
  breakdown modèle × horizon avec largeur du PI 95%) ;
- un onglet Graphique (courbe de prix réelle + prédictions de chaque modèle sélectionné,
  bande d'intervalle de confiance à 95%, séparation train/validation, zoom Plotly).
Plus un onglet Comparaison (tous actifs confondus, graphiques barres — fonctionnalité
préexistante conservée telle quelle).

En bas de chaque onglet actif où la donnée existe (BTC-USD pour l'instant) : un tableau
jour par jour des test cases TC1.1-TC1.5 (`validation/sim_trades.py`), le TC qui a généré
un signal ce jour-là et son counter — restreint à D+1 (seul horizon supporté par ces
règles, cf. docstring de validation/sim_trades.py).

Depuis 2026-07 : par défaut, les séries lourdes (predictions/prices) ne sont plus
embarquées dans le HTML mais écrites à côté dans data/<run_date>.json, chargées par le
navigateur au fil de la sélection de date (fetch), pour garder un poids de page borné
quel que soit le nombre de dates historiques. `--inline` retrouve l'ancien comportement
mono-fichier (tout embarqué), utile pour prévisualiser en file:// sans serveur.

Exécution (depuis DeepEdgeBenchmark/) :
    python -m model_artifacts.generate_dashboard --inline   # mono-fichier, file:// OK
    python -m model_artifacts.generate_dashboard             # mode CI : coquille + data/*.json
    python -m model_artifacts.generate_dashboard --run-root Run --out Run/dashboard.html
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT))
from validation import sim_trades as st

# Test cases TC1.1-TC1.5 : jamais Naive (hors périmètre des règles, cf. brief), et un
# seul actif pour l'instant (BTC-USD) -- étendre en ajoutant des tickers ici.
SIM_TRADES_MODELS = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "TSDiff"]
SIM_TRADES_ASSETS = ["BTC-USD"]
SIM_TRADES_DB_PATH = "validation/tracking.db"

# Bouton "pipeline" du tableau TC : famille de règles par horizon de résolution. Seul
# "daily" a des règles codées aujourd'hui (TC1.1-1.5, résolues à D+1, cf.
# validation/sim_trades.py) -- "weekly"/"monthly" sont des emplacements réservés pour de
# futures règles (D+7, mensuelles) pas encore écrites ; le bouton correspondant reste
# grisé côté dashboard tant qu'aucun tc_id n'y est rattaché ci-dessous.
SIM_TRADES_PIPELINES = ["daily", "weekly", "monthly"]
TC_PIPELINE = {
    "TC1.1": "daily", "TC1.2": "daily", "TC1.3": "daily", "TC1.4": "daily", "TC1.5": "daily",
}

MODEL_ORDER = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive", "TSDiff"]
# Palette catégorielle validée (skill dataviz) — slots 1..6 dans l'ordre fixe.
MODEL_COLORS_LIGHT = {
    "ARIMA-GARCH": "#2a78d6", "SARIMA": "#1baf7a", "Prophet": "#eda100",
    "LSTM": "#008300", "Naive": "#4a3aa7", "TSDiff": "#d64550",
}
MODEL_COLORS_DARK = {
    "ARIMA-GARCH": "#3987e5", "SARIMA": "#199e70", "Prophet": "#c98500",
    "LSTM": "#008300", "Naive": "#9085e9", "TSDiff": "#e5606b",
}


def _num(v):
    """float JSON-safe : NaN/inf -> None."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _load_assets_order():
    """Ordre/labels d'affichage des actifs (best-effort, fallback si import impossible)."""
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT / "models"))
        from calibration.regime.assets import ASSETS
        return [{"ticker": a["ticker"], "label": a["label"], "short": a.get("short", a["ticker"]),
                  "asset_class": a.get("asset_class", "")} for a in ASSETS]
    except Exception:
        return None


def collect_run_data(run_root: Path) -> dict:
    """Parcourt Run/ et construit : la liste des records (un par combinaison
    modèle x actif x horizon x date-de-run), les séries de prédictions (pour le
    graphe) et les séries de prix (historique complet, dédupliquées par
    (run_date, actif) — identiques pour tous les modèles/horizons d'un même actif+run,
    cf. write_prices_parquet)."""
    records = []
    predictions: dict = {}
    prices: dict = {}

    for combo_dir in sorted(run_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        metrics_path = combo_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        metadata_path = combo_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        # Prévision hors-échantillon repliée dans metrics.json (clé "forecast") plutôt que
        # dans un forecast.json séparé -- cf. model_artifacts/pipeline.py::process_asset_model.
        forecast = metrics.get("forecast") or {}
        # Le préfixe du nom de dossier (YYYYMMDD-...) fait foi pour regrouper par date de
        # run — indépendant du nombre de tirets dans le nom de l'actif (BTC-USD, ZN=F...).
        run_date = combo_dir.name.split("-", 1)[0]
        model, asset, horizon = metrics.get("model"), metrics.get("asset"), metrics.get("horizon")

        records.append({
            "model": model, "asset": asset, "asset_class": metadata.get("asset_class", ""),
            "rmse_vs_naive": None,
            "horizon": horizon,
            "RMSE": metrics.get("RMSE"), "MAE": metrics.get("MAE"), "MAPE": metrics.get("MAPE"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "pi_coverage_95": metrics.get("pi_coverage_95"),
            "pi_width_min": metrics.get("pi_width_min"),
            "pi_width_mean": metrics.get("pi_width_mean"),
            "pi_width_max": metrics.get("pi_width_max"),
            "n_val": metrics.get("n_val"),
            # Point 1 du brief — skill vs baseline persistence (peuvent être absents
            # des runs antérieurs au correctif : le dashboard affiche alors "—")
            "theil_u": metrics.get("theil_u"),
            "MASE": metrics.get("MASE"),
            "change_corr": metrics.get("change_corr"),
            "dir_acc_change": metrics.get("dir_acc_change"),
            "dir_acc_ci95": metrics.get("dir_acc_ci95"),
            "dir_acc_p_vs_coin": metrics.get("dir_acc_p_vs_coin"),
            "dm_stat": metrics.get("dm_stat"),
            "dm_p": metrics.get("dm_p"),
            "skill_vs_naive": metrics.get("skill_vs_naive"),
            "forecast_last_price": _num(forecast.get("last_price")),
            "forecast_last_date": forecast.get("last_date"),
            "forecast_predicted": _num(forecast.get("predicted")),
            "forecast_pi_lower": _num(forecast.get("pi_lower")),
            "forecast_pi_upper": _num(forecast.get("pi_upper")),
            "run_date": run_date,
            "dir": combo_dir.name,
        })

        preds_path = combo_dir / "predictions.parquet"
        if preds_path.exists() and model and asset and horizon:
            df = pd.read_parquet(preds_path)
            points = [
                {"date": d.strftime("%Y-%m-%d"), "actual": _num(a), "predicted": _num(p),
                 "pi_lower": _num(lo), "pi_upper": _num(hi)}
                for d, a, p, lo, hi in zip(df["date"], df["actual"], df["predicted"],
                                            df["pi_lower"], df["pi_upper"])
            ]
            (predictions.setdefault(run_date, {}).setdefault(asset, {})
                        .setdefault(model, {})[horizon]) = points

        asset_bucket = prices.get(run_date, {})
        if asset and asset not in asset_bucket:
            prices_path = combo_dir / "prices.parquet"
            if prices_path.exists():
                pdf = pd.read_parquet(prices_path)
                asset_bucket[asset] = {
                    "points": [{"date": d.strftime("%Y-%m-%d"), "close": _num(c)}
                               for d, c in zip(pdf["date"], pdf["close"])],
                    "train_end": metadata.get("train_end"),
                    "window_start": metadata.get("window_start"),
                    "window_end": metadata.get("window_end"),
                }
                prices[run_date] = asset_bucket

    context_map = {}
    for record in records:
        context_key = (record["asset"], record["horizon"], record["run_date"])
        context_map.setdefault(context_key, {})[record["model"]] = record

    for (asset, horizon, run_date), models in context_map.items():
        naive_rmse = None
        naive_record = models.get("Naive")
        if naive_record is not None:
            naive_rmse = _num(naive_record.get("RMSE"))
        if naive_rmse is None or naive_rmse == 0:
            continue
        for model_name, record in models.items():
            if model_name == "Naive":
                record["rmse_vs_naive"] = None
                continue
            model_rmse = _num(record.get("RMSE"))
            record["rmse_vs_naive"] = None if model_rmse is None else model_rmse / naive_rmse

    run_dates = sorted({r["run_date"] for r in records})
    return {"records": records, "predictions": predictions, "prices": prices, "run_dates": run_dates}


def build_asset_catalog(records: list) -> list:
    """Ordre des actifs : celui de calibration.regime.assets si dispo, sinon ordre
    d'apparition dans les artefacts (fallback qui ne dépend pas de l'environnement)."""
    known = _load_assets_order()
    seen_tickers = list(dict.fromkeys(r["asset"] for r in records if r["asset"]))
    if known:
        by_ticker = {a["ticker"]: a for a in known}
        ordered = [by_ticker[t] for t in [a["ticker"] for a in known] if t in seen_tickers]
        extra = [{"ticker": t, "label": t, "short": t, "asset_class": ""}
                 for t in seen_tickers if t not in by_ticker]
        return ordered + extra
    return [{"ticker": t, "label": t, "short": t, "asset_class": ""} for t in seen_tickers]


def collect_sim_trades_daily(db_path: str = SIM_TRADES_DB_PATH) -> dict:
    """Détail jour par jour des test cases TC1.1-TC1.5 (validation/sim_trades.py),
    un par actif de SIM_TRADES_ASSETS -- vide (silencieux) si tracking.db est absent
    ou si validation.sim_trades lève une erreur, pour ne jamais faire échouer la
    génération du reste du dashboard sur cette seule fonctionnalité annexe."""
    out = {}
    for asset in SIM_TRADES_ASSETS:
        try:
            rows = st.daily_detail(db_path=db_path, asset=asset, models=SIM_TRADES_MODELS)
        except Exception as exc:
            print(f"[generate_dashboard] sim_trades indisponible pour {asset} ({exc})")
            rows = []
        out[asset] = [
            {
                "source": r["source"], "d_date": r["d_date"], "target_date": r["target_date"],
                "model": r["model"],
                "reference_price": _num(r["reference_price"]), "predicted": _num(r["predicted"]),
                "pi_lower": _num(r["pi_lower"]), "pi_upper": _num(r["pi_upper"]),
                "realized_price": _num(r["realized_price"]),
                "signals": r["signals"],
            }
            for r in rows
        ]
    return out


def render_html(run_data: dict, run_root_label: str, external_series: bool = False) -> str:
    """external_series=False (mode --inline) : predictions/prices embarquées dans le
    payload comme avant, page autonome ouvrable en file://. external_series=True (mode
    par défaut) : predictions/prices exclues du payload -- le JS les récupère par
    fetch('data/<date>.json') à la demande (cf. main() pour l'écriture de ces fichiers)."""
    records = run_data["records"]
    asset_catalog = build_asset_catalog(records)
    models_present = [m for m in MODEL_ORDER if any(r["model"] == m for r in records)]
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z"),
        "run_root": run_root_label,
        "model_order": MODEL_ORDER,
        "models_present": models_present,
        "model_colors_light": MODEL_COLORS_LIGHT,
        "model_colors_dark": MODEL_COLORS_DARK,
        "assets": asset_catalog,
        "records": records,
        "run_dates": run_data["run_dates"],
        "sim_trades_daily": collect_sim_trades_daily(),
        "sim_trades_models": SIM_TRADES_MODELS,
        "sim_trades_pipelines": SIM_TRADES_PIPELINES,
        "tc_pipeline": TC_PIPELINE,
        "external_series": external_series,
    }
    if not external_series:
        payload["predictions"] = run_data["predictions"]
        payload["prices"] = run_data["prices"]
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


def write_date_series_files(run_data: dict, data_dir: Path) -> None:
    """Écrit data/<run_date>.json (predictions + prices de cette seule date) -- appelé
    uniquement en mode externe (cf. render_html external_series=True)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for run_date in run_data["run_dates"]:
        series = {
            "predictions": run_data["predictions"].get(run_date, {}),
            "prices": run_data["prices"].get(run_date, {}),
        }
        series_json = json.dumps(series, ensure_ascii=False)
        (data_dir / f"{run_date}.json").write_text(series_json, encoding="utf-8")


HTML_TEMPLATE = r"""<title>Dashboard KPI — Modèles de prévision</title>
<style>
:root {
  --surface-1:      #fcfcfb;
  --page-plane:     #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid-line:      #e1e0d9;
  --baseline:       #c3c2b7;
  --border-ring:    rgba(11,11,11,0.10);
  --card-shadow:    0 1px 2px rgba(11,11,11,0.06);
  --pos-color:      #1baf7a;
  --neg-color:      #d64550;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --grid-line:      #2c2c2a;
    --baseline:       #383835;
    --border-ring:    rgba(255,255,255,0.10);
    --card-shadow:    0 1px 3px rgba(0,0,0,0.4);
    --pos-color:      #2ecc9a;
    --neg-color:      #e5606b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 32px 64px;
  background: var(--page-plane);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 12px; color: var(--text-primary); }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border-ring);
  border-radius: 10px;
  box-shadow: var(--card-shadow);
  padding: 18px 20px;
  margin-bottom: 20px;
}
.controls-row {
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  margin-bottom: 20px;
}
.toggle-group {
  display: inline-flex; border: 1px solid var(--border-ring); border-radius: 8px; overflow: hidden;
}
.toggle-group button {
  font: inherit; font-size: 13px; padding: 7px 16px; border: none; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
}
.toggle-group button.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.search-box {
  font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-primary);
  min-width: 200px;
}
.select-box {
  font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-primary);
}
.btn-ghost {
  font: inherit; font-size: 13px; padding: 7px 16px; border-radius: 8px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-secondary);
  cursor: pointer;
}
.btn-ghost:hover { color: var(--text-primary); }
.stat-tiles { display: flex; gap: 16px; flex-wrap: wrap; }
.stat-tile { flex: 1 1 150px; }
.stat-tile .label { font-size: 12px; color: var(--text-secondary); }
.stat-tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.stat-tile .value.pos { color: var(--pos-color); }
.stat-tile .value.neg { color: var(--neg-color); }
.panel-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;
}
.panel { border: 1px solid var(--border-ring); border-radius: 8px; padding: 10px 12px; }
.panel-title { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; font-weight: 600; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-muted); }
svg .axis-line { stroke: var(--baseline); stroke-width: 1; }
svg .grid-line { stroke: var(--grid-line); stroke-width: 1; }
svg .ref-line { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: none; }
.bar { cursor: pointer; }
.bar:hover, .bar.hovered { filter: brightness(1.12); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
thead th {
  text-align: left; padding: 8px 10px; color: var(--text-secondary); font-weight: 600;
  border-bottom: 1px solid var(--grid-line); cursor: pointer; user-select: none; white-space: nowrap;
}
thead th:hover { color: var(--text-primary); }
tbody td { padding: 6px 10px; border-bottom: 1px solid var(--grid-line); font-variant-numeric: tabular-nums; }
tbody tr:hover { background: rgba(128,128,128,0.06); }
.no-data { color: var(--text-muted); font-size: 13px; padding: 12px 0; }
#tooltip {
  position: fixed; pointer-events: none; z-index: 50; display: none;
  background: var(--surface-1); border: 1px solid var(--border-ring); border-radius: 6px;
  padding: 8px 10px; font-size: 12px; box-shadow: var(--card-shadow); max-width: 240px;
}
#tooltip .tt-title { color: var(--text-secondary); margin-bottom: 4px; }
#tooltip .tt-row { display: flex; align-items: center; gap: 6px; }
#tooltip .tt-key { display: inline-block; width: 10px; height: 2px; }
#tooltip .tt-value { font-weight: 600; margin-left: auto; }

.tabbar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
.tab-btn {
  font: inherit; font-size: 13px; padding: 9px 18px; border-radius: 8px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-secondary);
  cursor: pointer;
}
.tab-btn:hover { color: var(--text-primary); }
.tab-btn.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.asset-panel { display: none; }
.asset-panel.active { display: block; }
.subtabbar {
  display: inline-flex; border: 1px solid var(--border-ring); border-radius: 8px; overflow: hidden;
}
.subtabbar button {
  font: inherit; font-size: 13px; padding: 7px 16px; border: none; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
}
.subtabbar button.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.sub-panel { display: none; }
.sub-panel.active { display: block; }
.field-label { font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 8px; }
.model-checks { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; }
.model-check { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.model-check .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.kpi-cards { display: flex; gap: 14px; flex-wrap: wrap; }
.kpi-card { flex: 1 1 220px; border: 1px solid var(--border-ring); border-radius: 8px; padding: 12px 14px; }
.kpi-card-title { font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.kpi-row { display: flex; justify-content: space-between; font-size: 12.5px; padding: 3px 0; color: var(--text-secondary); }
.kpi-row b { color: var(--text-primary); font-variant-numeric: tabular-nums; font-weight: 600; }
.kpi-row.warn { background: rgba(214,58,58,0.10); border-radius: 4px; margin: 1px -6px; padding: 3px 6px; }
.kpi-row.warn b { color: #d63a3a; }
td.warn-cell { background: rgba(214,58,58,0.10); color: #d63a3a; font-weight: 600; border-radius: 4px; }
.kpi-card.validated-ok { background: rgba(27,175,122,0.10); border-color: var(--pos-color); }
.kpi-card.validated-bad { background: rgba(214,69,80,0.10); border-color: var(--neg-color); }
@media (prefers-color-scheme: dark) {
  .kpi-card.validated-ok { background: rgba(46,204,154,0.14); }
  .kpi-card.validated-bad { background: rgba(229,96,107,0.14); }
}
.kpi-card.validation-global { border-width: 2px; }
.kpi-card.validation-global .kpi-card-title { font-size: 14px; }
.last-price-card {
  flex: 1 1 220px; border: 1px solid var(--text-primary); border-radius: 8px; padding: 12px 14px;
  background: rgba(128,128,128,0.05);
}
.last-price-card .kpi-card-title { color: var(--text-primary); }
.last-price-card .value { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.last-price-card .sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.info-dot {
  display: inline-flex; align-items: center; justify-content: center;
  width: 13px; height: 13px; border-radius: 50%;
  border: 1px solid var(--text-muted); color: var(--text-muted);
  font-size: 9.5px; line-height: 1; cursor: help; user-select: none; flex: none;
}
.info-dot:hover { border-color: var(--text-primary); color: var(--text-primary); }
.threshold-field { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary); }
.threshold-field input[type=range] { width: 110px; }
.threshold-field b { color: var(--text-primary); font-variant-numeric: tabular-nums; }
.chart-daterange { font-size: 12px; color: var(--text-secondary); margin: -4px 0 12px; }
.chart-checks { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; }
.chart-check { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.chart-wrap { min-height: 480px; }
</style>

<h1>Dashboard KPI — Modèles de prévision</h1>
<p class="subtitle" id="subtitle"></p>

<div class="tabbar" id="asset-tabbar"></div>
<div id="asset-panels"></div>

<div id="tooltip"></div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const DATA = __DATA_JSON__;

// ---- Séries par date (predictions/prices) : embarquées (mode --inline) ou chargées à
// la demande via fetch('data/<date>.json') (mode par défaut, cf. generate_dashboard.py).
const dataCache = new Map();
function seedDataCacheFromInline() {
  if (DATA.external_series) return;
  (DATA.run_dates || []).forEach(d => {
    dataCache.set(d, {
      predictions: (DATA.predictions || {})[d] || {},
      prices: (DATA.prices || {})[d] || {},
    });
  });
}
async function ensureDateData(date) {
  if (!date || dataCache.has(date)) return dataCache.get(date) || { predictions: {}, prices: {} };
  if (!DATA.external_series) return { predictions: {}, prices: {} };
  try {
    const res = await fetch(`data/${date}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    dataCache.set(date, d);
    return d;
  } catch (err) {
    console.error(`Échec du chargement de data/${date}.json`, err);
    return { predictions: {}, prices: {}, error: true };
  }
}
function predsBucket(date) { return (dataCache.get(date) || {}).predictions || {}; }
function pricesBucket(date) { return (dataCache.get(date) || {}).prices || {}; }

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const MODEL_COLORS = isDark ? DATA.model_colors_dark : DATA.model_colors_light;
const MODELS = DATA.model_order.filter(m => DATA.models_present.includes(m));
const ACTUAL_COLOR = isDark ? '#ffffff' : '#0b0b0b';
const TRAIN_COLOR = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(11,11,11,0.45)';
const GRID_COLOR = isDark ? '#2c2c2a' : '#e1e0d9';
const AXIS_TEXT_COLOR = isDark ? '#c3c2b7' : '#52514e';
const FORECAST_DAYS_OFFSET = { D1: 1, D7: 7 };

function addDays(dateStr, days) {
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function fmt(v, digits) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (Number(v) * 100).toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' %';
}

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16), g = parseInt(h.substring(2, 4), 16), b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---- Subtitle -------------------------------------------------------------
function renderSubtitle() {
  const el = document.getElementById('subtitle');
  const nCombos = DATA.records.length;
  const nAssets = new Set(DATA.records.map(r => r.asset)).size;
  el.textContent = `${nCombos} combinaisons (modèle × actif × horizon × date) — ${nAssets} actif(s), ${MODELS.length} modèle(s) — généré le ${DATA.generated_at} depuis ${DATA.run_root}`;
}

// ---- Tooltip helper (onglet Comparaison) -----------------------------------
const tooltipEl = document.getElementById('tooltip');
function showTooltip(evt, title, rows) {
  tooltipEl.innerHTML = '';
  const t = document.createElement('div');
  t.className = 'tt-title';
  t.textContent = title;
  tooltipEl.appendChild(t);
  rows.forEach(row => {
    const r = document.createElement('div');
    r.className = 'tt-row';
    const key = document.createElement('span');
    key.className = 'tt-key';
    key.style.background = row.color;
    const name = document.createElement('span');
    name.textContent = row.name;
    const value = document.createElement('span');
    value.className = 'tt-value';
    value.textContent = row.value;
    r.appendChild(key); r.appendChild(name); r.appendChild(value);
    tooltipEl.appendChild(r);
  });
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = (evt.clientX + 14) + 'px';
  tooltipEl.style.top = (evt.clientY + 14) + 'px';
}
function hideTooltip() { tooltipEl.style.display = 'none'; }

// ---- Définitions des KPI (bulle au survol du repère "ⓘ") -------------------
const KPI_DEFINITIONS = {
  rmse: "RMSE — racine de l'erreur quadratique moyenne entre prix réel et prédit sur la validation. Unité du prix ; plus bas = meilleur.",
  mae: "MAE — erreur absolue moyenne entre prix réel et prédit sur la validation. Unité du prix ; plus bas = meilleur.",
  mape: "MAPE — erreur absolue moyenne en % du prix réel. Comparable entre actifs de prix différents.",
  diracc: "Exactitude directionnelle — % de fois où le modèle a prédit le bon sens (hausse/baisse) par rapport à la veille.",
  picov: "Couverture du PI 95% — % des points réels de validation tombant dans la bande de prédiction à 95%. Cible ≈ 95% ; trop haut ou trop bas indique un intervalle mal calibré.",
  piwidth: "Largeur du PI 95% — écart entre borne haute et basse de l'intervalle de confiance sur la validation (min / moyenne / max). Une bande large traduit une forte incertitude du modèle.",
  forecast: "Prévision hors-échantillon — prix prédit par le modèle au-delà de la dernière clôture connue (pas du backtest), avec son intervalle de confiance à 95% (PI 95% [bas – haut]).",
  lastprice: "Dernier prix de marché utilisé — dernière clôture connue (veille de J+1), point de référence pour comparer chaque prévision de modèle.",
  warnthreshold: "Seuil d'alerte — si la prévision s'écarte de plus de ce pourcentage par rapport au dernier prix connu, la case est signalée en rouge comme a priori suspecte.",
  lag: "Déphasage (cross-corrélation) — corrèle prédit(t) avec réel(t−k) pour k=−5..5 sur le backtest de validation ; le k qui maximise la corrélation est le décalage effectif du modèle. k=0 : pas de déphasage. k=1 : le modèle reproduit en fait la valeur d'hier.",
  nval: "n (validation) — nombre de points de la période de validation utilisés pour calculer ces métriques.",
  theilu: "Theil's U — RMSE(modèle) / RMSE(persistence). U < 1 : bat la marche aléatoire ; U ≈ 1 : aucun apport ; U > 1 : pire. La métrique de skill de référence (Point 1 du brief).",
  masedef: "MASE — MAE(modèle) / MAE(persistence) sur le même jeu de validation. Même lecture que Theil's U, en erreur absolue.",
  changecorr: "Corr. des variations — corrélation entre variation prédite (préd − dernier prix connu) et variation réalisée. ≈ 0 : le modèle n'a aucune information sur le mouvement.",
  diraccchg: "Dir. Acc (variations) — % de bon sens sur la variation prédite vs le dernier prix connu, avec IC binomial de Wilson à 95%. Si l'IC contient 50%, indiscernable du pile-ou-face.",
  dmdef: "Diebold-Mariano vs persistence — test de différence de perte quadratique (variance Newey-West, correction HLN). p < 0.05 et DM < 0 : le modèle bat significativement le naïf.",
  skillverdict: "Verdict de skill — synthèse Theil's U + DM : 'beats naive' / 'no better than naive' / 'worse than naive'. Règle de lecture du Point 1 du brief.",
  tcsource: "Source (colonne technique) — 'oos' (rejouée hors-échantillon, backfill inclus) ou 'live' (production ce jour-là). Ne détermine PAS le filtre Vraies/Fausses prédictions ci-dessus : certains jours 'oos' (backfills des 8, 11, 13, 14/07) sont bien de vraies prédictions, cf. Vraies/Fausses.",
  tcsignal: "Test case(s) déclenché(s) — identifiant(s) TC1.1–TC1.5 dont les conditions étaient réunies ce jour pour ce modèle. '(ouvert)' : signal pas encore résolu, en attente du prix réalisé.",
  tccounter: "Counter — score de résolution du signal : +1/+2 si la trajectoire réalisée valide la branche gagnante du test case, -1/-2 si elle l'invalide. Vide si aucun signal ou signal encore ouvert.",
  tcusage: "Utilisation brute — nombre de lignes (jour × modèle) où au moins un test case s'est déclenché ET dont le counter résolu est positif (+1/+2). Une prédiction sans signal, ou dont le signal a été résolu négativement, n'est pas comptée.",
  tcperf: "Performance simulation (Σ counter) — somme de tous les counters résolus (positifs et négatifs) des signaux déclenchés sur la sélection courante. Positif : les signaux ont globalement été gagnants.",
  tcrate: "Taux d'utilisation — Utilisation brute rapportée au nombre total de lignes de la sélection (modèle(s) + pipeline choisis). Les signaux encore ouverts comptent comme non utilisables pour l'instant.",
  tcsourcefilter: "Vraies vs fausses prédictions — vraie : d_date à partir du 06/07/2026 (08/07/2026 pour TSDiff, arrivé plus tard dans la grille), qu'elle soit techniquement marquée 'live' ou 'oos' (des jours ont été rejoués en 'oos' faute d'avoir tourné le jour même : backfills des 8, 11, 13, 14/07, mais restent de vraies prédictions). Fausse : tout ce qui précède, reconstruction de backtest sur la période de validation.",
  tcvalidated: "Validation modèle × horizon (D+1) × actif — le modèle est considéré validé sur la sélection courante (pipeline, source(s), modèle) si son taux d'utilisation atteint le seuil choisi ci-dessous. Fond vert : validé ; fond rouge : non validé ; pas de couleur : aucune ligne pour ce modèle.",
  tcvalidatedglobal: "Validation agrégée — s'affiche quand plusieurs modèles sont cochés ci-dessus. Même règle que la validation par modèle, mais le taux d'utilisation est calculé sur les lignes de tous les modèles cochés confondus (pas une moyenne des taux individuels). Décochez tous les modèles sauf un pour voir le détail de ce seul modèle.",
};

function infoDot(defKey) {
  return `<span class="info-dot" data-def="${defKey}">i</span>`;
}

document.addEventListener('mouseover', (evt) => {
  const dot = evt.target.closest('.info-dot');
  if (!dot) return;
  const text = KPI_DEFINITIONS[dot.dataset.def];
  if (!text) return;
  tooltipEl.innerHTML = `<div style="max-width:220px;">${text}</div>`;
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = (evt.clientX + 14) + 'px';
  tooltipEl.style.top = (evt.clientY + 14) + 'px';
});
document.addEventListener('mousemove', (evt) => {
  if (!evt.target.closest('.info-dot')) return;
  tooltipEl.style.left = (evt.clientX + 14) + 'px';
  tooltipEl.style.top = (evt.clientY + 14) + 'px';
});
document.addEventListener('mouseout', (evt) => {
  if (evt.target.closest('.info-dot')) hideTooltip();
});

// =============================================================================
// Onglets par actif : état, squelette, KPIs, Graphique
// =============================================================================

const assetState = {};
DATA.assets.forEach(a => {
  assetState[a.ticker] = {
    date: DATA.run_dates[DATA.run_dates.length - 1] || null,
    horizon: 'D1',
    models: new Set(MODELS),
    showPI: true,
    showTrain: true,
    showVal: true,
    showPred: true,
    warnThreshold: 20,
    subtab: 'kpis',
  };
});

function buildTabBar() {
  const bar = document.getElementById('asset-tabbar');
  bar.innerHTML = '';
  DATA.assets.forEach((a, i) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.textContent = a.label;
    btn.dataset.asset = a.ticker;
    btn.addEventListener('click', () => switchAssetTab(a.ticker));
    bar.appendChild(btn);
  });
  const cmpBtn = document.createElement('button');
  cmpBtn.className = 'tab-btn';
  cmpBtn.textContent = 'Comparaison';
  cmpBtn.dataset.asset = 'COMPARISON';
  cmpBtn.addEventListener('click', () => switchAssetTab('COMPARISON'));
  bar.appendChild(cmpBtn);
}

function switchAssetTab(ticker) {
  document.querySelectorAll('.asset-panel').forEach(p => p.classList.toggle('active', p.dataset.asset === ticker));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.asset === ticker));
  if (ticker !== 'COMPARISON' && assetState[ticker].subtab === 'chart') renderAssetChart(ticker);
}

function assetPanelSkeleton(a) {
  const s = a.short;
  return `
    <div class="card controls-row">
      <div class="subtabbar" id="subtab-${s}">
        <button class="active" data-sub="kpis">KPIs</button>
        <button data-sub="chart">Graphique</button>
      </div>
      <label class="field-label">Date de run
        <select class="select-box" id="date-${s}"></select>
      </label>
      <div class="toggle-group" id="horizon-${s}"></div>
      <div class="model-checks" id="models-${s}"></div>
      <label class="threshold-field">Seuil d'alerte ${infoDot('warnthreshold')}
        <input type="range" id="warn-${s}" min="5" max="100" step="1" value="20">
        <span>±<b id="warn-value-${s}">20</b>%</span>
      </label>
    </div>

    <div class="sub-panel active" id="sub-kpis-${s}">
      <div class="card">
        <h2>KPIs par modèle — ${a.label}</h2>
        <div class="kpi-cards" id="kpi-cards-${s}"></div>
      </div>
      <div class="card">
        <h2>Breakdown modèle × horizon</h2>
        <div style="overflow-x:auto;" id="breakdown-wrap-${s}"></div>
      </div>
    </div>

    <div class="sub-panel" id="sub-chart-${s}">
      <div class="card">
        <div class="chart-daterange" id="chart-daterange-${s}"></div>
        <div class="controls-row" style="margin-bottom:12px;">
          <div class="chart-checks" id="chart-checks-${s}">
            <label class="chart-check"><input type="checkbox" id="showtrain-${s}" checked> Entraînement</label>
            <label class="chart-check"><input type="checkbox" id="showval-${s}" checked> Validation</label>
            <label class="chart-check"><input type="checkbox" id="showpred-${s}" checked> Prédiction</label>
          </div>
          <label class="field-label">
            <input type="checkbox" id="showpi-${s}" checked> Afficher les intervalles de confiance
          </label>
          <button class="btn-ghost" id="resetzoom-${s}">Réinitialiser le zoom</button>
        </div>
        <div class="chart-wrap" id="chart-${s}"></div>
      </div>
    </div>

    <div class="card" id="simtrades-card-${s}" style="display:none;">
      <h2>Test cases (TC1.1&ndash;TC1.5, D+1) &mdash; jour par jour</h2>
      <div class="controls-row" style="margin-bottom:12px;">
        <div class="toggle-group" id="simtrades-pipeline-${s}"></div>
        <div class="model-checks" id="simtrades-models-${s}"></div>
        <div class="chart-checks" id="simtrades-sources-${s}"></div>
        <label class="field-label">
          <input type="checkbox" id="simtrades-onlysignal-${s}" checked> Seulement les jours avec signal
        </label>
      </div>
      <div style="overflow:auto; max-height:480px;" id="simtrades-table-${s}"></div>
      <div class="stat-tiles" id="simtrades-usage-${s}" style="margin-top:16px;"></div>
      <div class="controls-row" style="margin-top:20px; margin-bottom:4px;">
        <h2 style="margin:0;">Validation par modèle ${infoDot('tcvalidated')}</h2>
        <label class="threshold-field">Seuil de validation
          <input type="range" id="simtrades-threshold-${s}" min="0" max="100" step="1" value="80">
          <span>&ge;<b id="simtrades-threshold-value-${s}">80</b>%</span>
        </label>
      </div>
      <div class="kpi-cards" id="simtrades-validation-${s}"></div>
    </div>
  `;
}

function comparisonSkeleton() {
  return `
    <div class="card controls-row">
      <div class="toggle-group" id="horizon-toggle"></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <div class="stat-tiles" id="stat-tiles"></div>
    </div>
    <div class="card">
      <h2>Exactitude directionnelle (%)</h2>
      <div id="chart-diracc"></div>
    </div>
    <div class="card">
      <h2>Couverture de l'intervalle à 95% (%)</h2>
      <div id="chart-picov"></div>
    </div>
    <div class="card">
      <h2>RMSE (racine de l'erreur quadratique moyenne, unité du prix de l'actif)</h2>
      <div class="panel-grid" id="chart-rmse"></div>
    </div>
    <div class="card">
      <h2>MAE (erreur absolue moyenne, unité du prix de l'actif)</h2>
      <div class="panel-grid" id="chart-mae"></div>
    </div>
    <div class="card">
      <div class="controls-row" style="margin-bottom:12px;">
        <h2 style="margin:0;">Table détaillée</h2>
        <input class="search-box" id="table-search" placeholder="Filtrer par actif ou modèle…">
      </div>
      <div style="overflow-x:auto;">
        <table id="data-table"></table>
      </div>
    </div>
  `;
}

function buildAssetPanels() {
  const root = document.getElementById('asset-panels');
  root.innerHTML = '';
  DATA.assets.forEach((a, i) => {
    const panel = document.createElement('div');
    panel.className = 'asset-panel' + (i === 0 ? ' active' : '');
    panel.dataset.asset = a.ticker;
    panel.innerHTML = assetPanelSkeleton(a);
    root.appendChild(panel);
  });
  const cmp = document.createElement('div');
  cmp.className = 'asset-panel';
  cmp.dataset.asset = 'COMPARISON';
  cmp.innerHTML = comparisonSkeleton();
  root.appendChild(cmp);

  DATA.assets.forEach(a => wireAssetPanel(a));
}

function wireAssetPanel(a) {
  const s = a.short, ticker = a.ticker, st = assetState[ticker];

  const dateSel = document.getElementById(`date-${s}`);
  DATA.run_dates.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    if (d === st.date) opt.selected = true;
    dateSel.appendChild(opt);
  });
  dateSel.addEventListener('change', () => { switchAssetDate(ticker, dateSel.value, dateSel); });

  const horizons = [...new Set(DATA.records.filter(r => r.asset === ticker).map(r => r.horizon))].sort();
  const hEl = document.getElementById(`horizon-${s}`);
  if (!horizons.includes(st.horizon)) st.horizon = horizons[0];
  horizons.forEach(h => {
    const btn = document.createElement('button');
    btn.textContent = h === 'D1' ? 'D+1' : (h === 'D7' ? 'D+7' : h);
    btn.className = h === st.horizon ? 'active' : '';
    btn.addEventListener('click', () => {
      st.horizon = h;
      hEl.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
      refreshAssetTab(ticker);
    });
    hEl.appendChild(btn);
  });

  const mEl = document.getElementById(`models-${s}`);
  MODELS.forEach(m => {
    const label = document.createElement('label');
    label.className = 'model-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = st.models.has(m);
    cb.addEventListener('change', () => {
      if (cb.checked) st.models.add(m); else st.models.delete(m);
      refreshAssetTab(ticker);
    });
    const sw = document.createElement('span');
    sw.className = 'swatch'; sw.style.background = MODEL_COLORS[m];
    const txt = document.createElement('span'); txt.textContent = m;
    label.appendChild(cb); label.appendChild(sw); label.appendChild(txt);
    mEl.appendChild(label);
  });

  document.getElementById(`subtab-${s}`).querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      st.subtab = btn.dataset.sub;
      document.getElementById(`subtab-${s}`).querySelectorAll('button')
        .forEach(b => b.classList.toggle('active', b === btn));
      document.getElementById(`sub-kpis-${s}`).classList.toggle('active', btn.dataset.sub === 'kpis');
      document.getElementById(`sub-chart-${s}`).classList.toggle('active', btn.dataset.sub === 'chart');
      if (btn.dataset.sub === 'chart') renderAssetChart(ticker);
    });
  });

  document.getElementById(`showpi-${s}`).addEventListener('change', (e) => {
    st.showPI = e.target.checked;
    if (st.subtab === 'chart') renderAssetChart(ticker);
  });
  document.getElementById(`resetzoom-${s}`).addEventListener('click', () => {
    Plotly.relayout(`chart-${s}`, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  });

  const warnEl = document.getElementById(`warn-${s}`);
  const warnValueEl = document.getElementById(`warn-value-${s}`);
  warnEl.addEventListener('input', () => {
    st.warnThreshold = Number(warnEl.value);
    warnValueEl.textContent = warnEl.value;
    renderAssetKpis(ticker);
  });

  document.getElementById(`showtrain-${s}`).addEventListener('change', (e) => {
    st.showTrain = e.target.checked;
    if (st.subtab === 'chart') renderAssetChart(ticker);
  });
  document.getElementById(`showval-${s}`).addEventListener('change', (e) => {
    st.showVal = e.target.checked;
    if (st.subtab === 'chart') renderAssetChart(ticker);
  });
  document.getElementById(`showpred-${s}`).addEventListener('change', (e) => {
    st.showPred = e.target.checked;
    if (st.subtab === 'chart') renderAssetChart(ticker);
  });

  renderAssetKpis(ticker);
  setupSimTradesControls(a);
  renderSimTradesTable(ticker);
}

function refreshAssetTab(ticker) {
  renderAssetKpis(ticker);
  if (assetState[ticker].subtab === 'chart') renderAssetChart(ticker);
}

// Changement de date de run pour un actif : si les séries de cette date ne sont pas
// encore en cache, les récupère (fetch, mode externe) avant de rafraîchir l'onglet --
// affiche un état de chargement le temps du fetch, un message d'erreur s'il échoue.
async function switchAssetDate(ticker, date, dateSel) {
  const st = assetState[ticker];
  st.date = date;
  if (dataCache.has(date) || !DATA.external_series) { refreshAssetTab(ticker); return; }

  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short;
  dateSel.disabled = true;
  document.getElementById(`kpi-cards-${s}`).innerHTML = '<div class="no-data">Chargement…</div>';
  const chartEl = document.getElementById(`chart-${s}`);
  if (chartEl) chartEl.innerHTML = '<div class="no-data">Chargement…</div>';

  const result = await ensureDateData(date);
  dateSel.disabled = false;
  if (st.date !== date) return; // l'utilisateur a changé de date entre-temps

  if (result.error) {
    document.getElementById(`kpi-cards-${s}`).innerHTML =
      '<div class="no-data">Erreur de chargement des données pour cette date. Réessayez.</div>';
    return;
  }
  refreshAssetTab(ticker);
}

// =============================================================================
// Test cases TC1.1-TC1.5 (validation/sim_trades.py) — tableau jour par jour, en bas
// de chaque onglet actif où la donnée existe (BTC-USD pour l'instant, cf.
// SIM_TRADES_ASSETS côté Python). Restreint à D+1 (seul horizon supporté par ces
// règles — l'alignement D->D+1 ne s'applique pas au backtest D+7 rolling-origin).
// =============================================================================

// Vraie vs fausse prédiction : NE dépend PAS de la colonne source (live/oos) -- des
// jours ont été rejoués en 'oos' faute d'avoir tourné en 'live' le jour même (panne/
// backfill des 8, 11, 13, 14/07), mais restent de vraies prédictions au sens où elles
// ont été produites comme telles. Règle métier : vraie <=> d_date >= date de démarrage
// réel du modèle (06/07/2026 pour les modèles historiques, 08/07/2026 pour TSDiff,
// arrivé plus tard dans la grille) ; tout ce qui précède est une fausse prédiction
// (reconstruction de backtest sur la période de validation).
const REAL_PREDICTION_START = { TSDiff: '2026-07-08' };
const REAL_PREDICTION_START_DEFAULT = '2026-07-06';
function isRealPrediction(row) {
  return row.d_date >= (REAL_PREDICTION_START[row.model] || REAL_PREDICTION_START_DEFAULT);
}

const simTradesState = {};

function setupSimTradesControls(a) {
  const s = a.short, ticker = a.ticker;
  const rows = (DATA.sim_trades_daily || {})[ticker];
  if (!rows || !rows.length) return;   // pas de données pour cet actif -> rien à câbler

  const models = DATA.sim_trades_models || [];
  const pipelines = DATA.sim_trades_pipelines || [];
  // Familles pipeline effectivement rattachées à au moins un TC (cf. TC_PIPELINE côté
  // Python) -- seule "daily" l'est aujourd'hui ; les autres restent grisées jusqu'à ce
  // que de nouvelles règles (D+7, mensuelles) leur soient rattachées.
  const activePipelines = new Set(Object.values(DATA.tc_pipeline || {}));
  simTradesState[ticker] = {
    models: new Set(models), pipeline: 'daily',
    predKinds: new Set(['real', 'fake']), validateThreshold: 80,
  };

  const pEl = document.getElementById(`simtrades-pipeline-${s}`);
  pEl.innerHTML = '';
  const pipelineLabel = { daily: 'Daily', weekly: 'Weekly', monthly: 'Monthly' };
  pipelines.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = pipelineLabel[p] || p;
    const enabled = activePipelines.has(p);
    btn.className = p === simTradesState[ticker].pipeline ? 'active' : '';
    if (!enabled) {
      btn.disabled = true;
      btn.title = 'Pas encore de règles pour cet horizon de résolution';
      btn.style.opacity = '0.4';
      btn.style.cursor = 'not-allowed';
    } else {
      btn.addEventListener('click', () => {
        simTradesState[ticker].pipeline = p;
        pEl.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
        renderSimTradesTable(ticker);
      });
    }
    pEl.appendChild(btn);
  });

  const mEl = document.getElementById(`simtrades-models-${s}`);
  mEl.innerHTML = '';
  models.forEach(m => {
    const label = document.createElement('label');
    label.className = 'model-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true;
    cb.addEventListener('change', () => {
      if (cb.checked) simTradesState[ticker].models.add(m); else simTradesState[ticker].models.delete(m);
      renderSimTradesTable(ticker);
    });
    const sw = document.createElement('span');
    sw.style.cssText = `width:9px;height:9px;border-radius:2px;display:inline-block;background:${MODEL_COLORS[m]};`;
    const txt = document.createElement('span'); txt.textContent = m;
    label.appendChild(cb); label.appendChild(sw); label.appendChild(txt);
    mEl.appendChild(label);
  });

  const sourceEl = document.getElementById(`simtrades-sources-${s}`);
  sourceEl.innerHTML = '';
  const predKindLabel = {
    real: `Vraies prédictions ${infoDot('tcsourcefilter')}`,
    fake: 'Fausses prédictions (backtest)',
  };
  ['real', 'fake'].forEach(kind => {
    const label = document.createElement('label');
    label.className = 'chart-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = simTradesState[ticker].predKinds.has(kind);
    cb.addEventListener('change', () => {
      if (cb.checked) simTradesState[ticker].predKinds.add(kind); else simTradesState[ticker].predKinds.delete(kind);
      renderSimTradesTable(ticker);
    });
    label.appendChild(cb);
    label.insertAdjacentHTML('beforeend', predKindLabel[kind]);
    sourceEl.appendChild(label);
  });

  const threshEl = document.getElementById(`simtrades-threshold-${s}`);
  const threshValueEl = document.getElementById(`simtrades-threshold-value-${s}`);
  threshEl.addEventListener('input', () => {
    simTradesState[ticker].validateThreshold = Number(threshEl.value);
    threshValueEl.textContent = threshEl.value;
    renderSimTradesTable(ticker);
  });

  document.getElementById(`simtrades-onlysignal-${s}`).addEventListener('change', () => renderSimTradesTable(ticker));
}

function renderSimTradesTable(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short;
  const card = document.getElementById(`simtrades-card-${s}`);
  const rows = (DATA.sim_trades_daily || {})[ticker];
  if (!rows || !rows.length) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  const state = simTradesState[ticker];
  const onlySignal = document.getElementById(`simtrades-onlysignal-${s}`).checked;
  const tcPipeline = DATA.tc_pipeline || {};

  // Filtre modèle(s) + vraie/fausse prédiction (cf. isRealPrediction, PAS r.source) +
  // pipeline sélectionné (daily pour l'instant), signaux restreints à la famille
  // choisie. Servie telle quelle à renderUsageStats/renderValidationCards -- calculée
  // sur TOUTES ces lignes, indépendamment de la case "Seulement les jours avec signal"
  // ci-dessous (sinon la cocher fausserait le taux d'utilisation en réduisant le
  // dénominateur).
  const base = rows
    .filter(r => state.models.has(r.model) && state.predKinds.has(isRealPrediction(r) ? 'real' : 'fake'))
    .map(r => ({ ...r, signals: r.signals.filter(sig => tcPipeline[sig.tc_id] === state.pipeline) }));

  renderUsageStats(s, base);
  const validationModels = (DATA.sim_trades_models || []).filter(m => state.models.has(m));
  renderValidationCards(s, base, validationModels, state.validateThreshold);

  let filtered = base;
  if (onlySignal) filtered = filtered.filter(r => r.signals.length > 0);

  const wrap = document.getElementById(`simtrades-table-${s}`);
  if (!filtered.length) {
    wrap.innerHTML = '<div class="no-data">Aucune ligne pour cette sélection.</div>';
    return;
  }

  let html = `<table><thead><tr><th>Date (D)</th><th>Cible (D+1)</th><th>Source ${infoDot('tcsource')}</th><th>Modèle</th>`
    + `<th>Réf. P(D)</th><th>Prévision</th><th>Réel</th><th>Test case(s) ${infoDot('tcsignal')}</th><th>Counter ${infoDot('tccounter')}</th></tr></thead><tbody>`;
  filtered.forEach(r => {
    const color = MODEL_COLORS[r.model] || '#888';
    const tcText = r.signals.length
      ? r.signals.map(sig => sig.tc_id + (sig.status === 'open' ? ' (ouvert)' : '')).join(', ')
      : '—';
    const counterText = r.signals.length
      ? r.signals.map(sig => sig.counter === null || sig.counter === undefined
          ? '—' : (sig.counter > 0 ? '+' + sig.counter : String(sig.counter))).join(', ')
      : '—';
    html += `<tr><td>${r.d_date}</td><td>${r.target_date}</td><td>${r.source}</td>`
      + `<td><span style="width:9px;height:9px;border-radius:2px;display:inline-block;background:${color};margin-right:6px;"></span>${r.model}</td>`
      + `<td>${fmt(r.reference_price, 2)}</td><td>${fmt(r.predicted, 2)}</td><td>${fmt(r.realized_price, 2)}</td>`
      + `<td>${tcText}</td><td>${counterText}</td></tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// Utilisation brute / performance simulation / taux d'utilisation : une ligne (jour ×
// modèle) est "utilisable" si au moins un TC s'y est déclenché ET que son counter résolu
// est positif (+1/+2) -- une prédiction qui ne déclenche rien n'a rien d'exploitable pour
// un trader, et un signal résolu négativement ne l'était pas non plus. La performance
// (Σ counter) somme tous les signaux résolus, positifs et négatifs. Le taux rapporte
// l'utilisation brute au nombre total de lignes de la sélection courante (modèle(s) +
// source(s) + pipeline) -- signaux ouverts (non encore résolus) comptés non-utilisables
// pour l'instant.
function computeUsage(rows) {
  const total = rows.length;
  let usableCount = 0;
  let counterSum = 0;
  rows.forEach(r => {
    let rowUsable = false;
    r.signals.forEach(sig => {
      if (sig.counter === null || sig.counter === undefined) return;
      counterSum += sig.counter;
      if (sig.counter > 0) rowUsable = true;
    });
    if (rowUsable) usableCount++;
  });
  return { total, usableCount, counterSum, taux: total ? usableCount / total : null };
}

function renderUsageStats(s, rows) {
  const el = document.getElementById(`simtrades-usage-${s}`);
  const { usableCount, counterSum, taux } = computeUsage(rows);

  const tiles = [
    { label: 'Utilisation brute', def: 'tcusage', value: String(usableCount), cls: '' },
    { label: 'Performance simulation (Σ counter)', def: 'tcperf', value: fmt(counterSum, 0),
      cls: counterSum > 0 ? 'pos' : (counterSum < 0 ? 'neg' : '') },
    { label: "Taux d'utilisation", def: 'tcrate', value: fmtPct(taux), cls: '' },
  ];
  el.innerHTML = tiles.map(t =>
    `<div class="stat-tile"><div class="label">${t.label} ${infoDot(t.def)}</div><div class="value ${t.cls}">${t.value}</div></div>`
  ).join('');
}

function validationCardHtml(titleHtml, rowsSubset, threshold, extraClass) {
  const { total, taux } = computeUsage(rowsSubset);
  const pct = taux === null ? null : taux * 100;
  const validated = pct !== null && pct >= threshold;
  const cls = pct === null ? '' : (validated ? 'validated-ok' : 'validated-bad');
  return `<div class="kpi-card ${cls} ${extraClass || ''}">`
    + `<div class="kpi-card-title">${titleHtml}</div>`
    + `<div class="kpi-row"><span>Taux d'utilisation ${infoDot('tcrate')}</span><b>${fmtPct(taux)}</b></div>`
    + `<div class="kpi-row"><span>n (lignes)</span><b>${total}</b></div>`
    + `<div class="kpi-row"><span>Verdict</span><b>${pct === null ? '—' : (validated ? 'Validé' : 'Non validé')}</b></div>`
    + `</div>`;
}

// Une seule carte de validation, qui s'adapte à la sélection de modèles (cases à cocher
// ci-dessus) : le détail du modèle si un seul est coché, l'agrégation de tous les
// modèles cochés confondus (pas une moyenne des taux individuels) s'ils sont plusieurs.
function renderValidationCards(s, rows, models, threshold) {
  const el = document.getElementById(`simtrades-validation-${s}`);
  if (!models.length) { el.innerHTML = '<div class="no-data">Sélectionnez au moins un modèle.</div>'; return; }
  if (models.length === 1) {
    const m = models[0];
    const swatch = `<span class="swatch" style="background:${MODEL_COLORS[m]};width:10px;height:10px;border-radius:2px;display:inline-block;"></span>${m}`;
    el.innerHTML = validationCardHtml(swatch, rows.filter(r => r.model === m), threshold);
    return;
  }
  const title = `${models.length} modèles sélectionnés (agrégé) ${infoDot('tcvalidatedglobal')}`;
  el.innerHTML = validationCardHtml(title, rows, threshold, 'validation-global');
}

// ---- Prévision : delta vs dernier prix, seuil d'alerte, déphasage ----------
function forecastPct(rec) {
  if (rec.forecast_predicted == null || !rec.forecast_last_price) return null;
  return (rec.forecast_predicted / rec.forecast_last_price - 1) * 100;
}
function isWarn(rec, threshold) {
  const pct = forecastPct(rec);
  return pct !== null && Math.abs(pct) > threshold;
}
function piRangeText(rec) {
  if (rec.forecast_pi_lower == null || rec.forecast_pi_upper == null) return '—';
  return `${fmt(rec.forecast_pi_lower, 2)} – ${fmt(rec.forecast_pi_upper, 2)}`;
}

function pearson(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? null : num / denom;
}

// Cross-corrélation à décalages -maxLag..+maxLag entre prédit(t) et réel(t-k) sur le
// backtest de validation — le k qui maximise |corr| est le déphasage effectif du modèle.
function lagCorrelation(points, maxLag = 5) {
  const actual = points.map(p => p.actual);
  const predicted = points.map(p => p.predicted);
  const n = points.length;
  let best = { lag: 0, corr: null };
  for (let k = -maxLag; k <= maxLag; k++) {
    const xs = [], ys = [];
    for (let t = 0; t < n; t++) {
      const tk = t - k;
      if (tk < 0 || tk >= n) continue;
      if (predicted[t] == null || actual[tk] == null) continue;
      xs.push(predicted[t]); ys.push(actual[tk]);
    }
    const corr = pearson(xs, ys);
    if (corr !== null && (best.corr === null || Math.abs(corr) > Math.abs(best.corr))) best = { lag: k, corr };
  }
  return best;
}
function lagLabel(lag) {
  if (lag === 0) return 'aucun déphasage';
  if (lag > 0) return `reproduit le réel d'il y a ${lag} jour(s) (retard)`;
  return `en avance de ${-lag} jour(s) sur le réel`;
}

// ---- KPIs par modèle (cartes) + breakdown modèle x horizon ------------------
const BREAKDOWN_COLS = [
  { key: 'model', label: 'Modèle' },
  { key: 'horizon', label: 'Horizon' },
  { key: 'RMSE', label: 'RMSE', digits: 4, def: 'rmse' },
  { key: 'rmse_vs_naive', label: 'RMSE / RMSE naïf', digits: 3, render: v => v == null ? '—' : `${fmt(v, 3)}×` },
  { key: 'MAE', label: 'MAE', digits: 4, def: 'mae' },
  { key: 'MAPE', label: 'MAPE (%)', digits: 2, def: 'mape' },
  { key: 'directional_accuracy', label: 'Exact. dir. (%)', digits: 2, def: 'diracc' },
  { key: 'pi_coverage_95', label: 'Couv. PI 95 (%)', digits: 2, def: 'picov' },
  { key: 'pi_width_min', label: 'Larg. PI min', digits: 4 },
  { key: 'pi_width_mean', label: 'Larg. PI moy.', digits: 4, def: 'piwidth' },
  { key: 'pi_width_max', label: 'Larg. PI max', digits: 4 },
  { key: 'theil_u', label: "Theil's U", digits: 3, def: 'theilu' },
  { key: 'MASE', label: 'MASE', digits: 3, def: 'masedef' },
  { key: 'dm_p', label: 'DM p', digits: 3, def: 'dmdef' },
  { key: 'skill_vs_naive', label: 'Skill vs naïf', def: 'skillverdict' },
  { key: 'n_val', label: 'n_val', digits: 0, def: 'nval' },
  { key: 'forecast_predicted', label: 'Prévision (prix)', digits: 2, def: 'forecast', warn: true },
  { key: '_pi_range', label: 'PI 95% [bas – haut]', render: piRangeText, warn: true },
];

function renderAssetKpis(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short, st = assetState[ticker];
  const checked = MODELS.filter(m => st.models.has(m));

  const cardsEl = document.getElementById(`kpi-cards-${s}`);
  cardsEl.innerHTML = '';

  const priceBucket = (pricesBucket(st.date) || {})[ticker];
  const anyRec = DATA.records.find(r => r.asset === ticker && r.horizon === st.horizon && r.run_date === st.date);
  const lastPrice = priceBucket ? priceBucket.points[priceBucket.points.length - 1] : null;
  const lastPriceCard = document.createElement('div');
  lastPriceCard.className = 'last-price-card';
  lastPriceCard.innerHTML = `<div class="kpi-card-title">Dernier prix de marché utilisé ${infoDot('lastprice')}</div>`
    + (lastPrice
        ? `<div class="value">${fmt(lastPrice.close, 2)}</div><div class="sub">${lastPrice.date}</div>`
        : (anyRec && anyRec.forecast_last_price != null
            ? `<div class="value">${fmt(anyRec.forecast_last_price, 2)}</div><div class="sub">${anyRec.forecast_last_date || ''}</div>`
            : `<div class="no-data">Pas de données</div>`));
  cardsEl.appendChild(lastPriceCard);

  if (!checked.length) {
    cardsEl.insertAdjacentHTML('beforeend', '<div class="no-data">Sélectionnez au moins un modèle.</div>');
  } else {
    checked.forEach(m => {
      const rec = DATA.records.find(r => r.asset === ticker && r.model === m
        && r.horizon === st.horizon && r.run_date === st.date);
      const card = document.createElement('div');
      card.className = 'kpi-card';
      let rowsHtml;
      if (!rec) {
        rowsHtml = '<div class="no-data">Pas de données</div>';
      } else {
        const pct = forecastPct(rec);
        const warn = isWarn(rec, st.warnThreshold);
        const pctText = pct === null ? '' : ` (${pct > 0 ? '+' : ''}${fmt(pct, 1)}%)`;
        const predSeries = ((predsBucket(st.date) || {})[ticker] || {})[m] || {};
        const backtestPoints = predSeries[st.horizon] || [];
        const lag = backtestPoints.length >= 4 ? lagCorrelation(backtestPoints, 5) : null;
        rowsHtml = [
          [`RMSE ${infoDot('rmse')}`, fmt(rec.RMSE, 4)],
          [`RMSE / RMSE naïf`, rec.rmse_vs_naive != null ? `${fmt(rec.rmse_vs_naive, 3)}×` : '—'],
          [`MAE ${infoDot('mae')}`, fmt(rec.MAE, 4)],
          [`MAPE ${infoDot('mape')}`, fmt(rec.MAPE, 2) + ' %'],
          [`Exact. directionnelle ${infoDot('diracc')}`, fmt(rec.directional_accuracy, 1) + ' %'],
          [`Couverture PI 95% ${infoDot('picov')}`, fmt(rec.pi_coverage_95, 1) + ' %'],
          [`Largeur PI min/moy/max ${infoDot('piwidth')}`, `${fmt(rec.pi_width_min, 2)} / ${fmt(rec.pi_width_mean, 2)} / ${fmt(rec.pi_width_max, 2)}`],
          [`Theil's U ${infoDot('theilu')}`, rec.theil_u != null ? fmt(rec.theil_u, 3) : '—',
           rec.theil_u != null && rec.theil_u >= 1],
          [`MASE ${infoDot('masedef')}`, rec.MASE != null ? fmt(rec.MASE, 3) : '—'],
          [`Corr. variations ${infoDot('changecorr')}`, rec.change_corr != null ? fmt(rec.change_corr, 3) : '—'],
          [`Dir. Acc variations ${infoDot('diraccchg')}`,
           rec.dir_acc_change != null
             ? `${fmt(rec.dir_acc_change, 1)} %` + (rec.dir_acc_ci95 ? ` [${fmt(rec.dir_acc_ci95[0], 1)}–${fmt(rec.dir_acc_ci95[1], 1)}]` : '')
             : '—'],
          [`DM vs naïf ${infoDot('dmdef')}`,
           rec.dm_p != null ? `p=${fmt(rec.dm_p, 3)}` : '—'],
          [`Skill vs naïf ${infoDot('skillverdict')}`, rec.skill_vs_naive ?? '—',
           rec.skill_vs_naive != null && rec.skill_vs_naive !== 'beats naive'],
          ['n (validation)', rec.n_val ?? '—'],
          [`Prévision ${infoDot('forecast')}`, fmt(rec.forecast_predicted, 2) + pctText, warn],
          ['PI 95% [bas – haut]', piRangeText(rec), warn],
          [`Déphasage ${infoDot('lag')}`, lag && lag.corr !== null ? `k=${lag.lag} (corr=${fmt(lag.corr, 2)}) — ${lagLabel(lag.lag)}` : '—'],
        ].map(([k, v, w]) => `<div class="kpi-row${w ? ' warn' : ''}"><span>${k}</span><b>${v}</b></div>`).join('');
      }
      card.innerHTML = `<div class="kpi-card-title">`
        + `<span class="swatch" style="background:${MODEL_COLORS[m]};width:10px;height:10px;border-radius:2px;display:inline-block;"></span>${m}</div>`
        + rowsHtml;
      cardsEl.appendChild(card);
    });
  }

  renderBreakdownTable(ticker);
}

function renderBreakdownTable(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short, st = assetState[ticker];
  const checked = MODELS.filter(m => st.models.has(m));
  let recs = DATA.records.filter(r => r.asset === ticker && r.run_date === st.date && checked.includes(r.model));
  recs = recs.slice().sort((x, y) =>
    MODELS.indexOf(x.model) - MODELS.indexOf(y.model) || String(x.horizon).localeCompare(String(y.horizon)));

  const wrap = document.getElementById(`breakdown-wrap-${s}`);
  wrap.innerHTML = '';
  if (!recs.length) {
    wrap.innerHTML = '<div class="no-data">Aucune donnée pour cette sélection.</div>';
    return;
  }

  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  BREAKDOWN_COLS.forEach(c => {
    const th = document.createElement('th');
    th.innerHTML = c.label + (c.def ? ' ' + infoDot(c.def) : '');
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  recs.forEach(r => {
    const tr = document.createElement('tr');
    const warnRow = isWarn(r, st.warnThreshold);
    BREAKDOWN_COLS.forEach(c => {
      const td = document.createElement('td');
      if (c.render) {
        td.textContent = c.render(r);
      } else {
        const v = r[c.key];
        td.textContent = c.digits !== undefined ? fmt(v, c.digits) : (v ?? '—');
      }
      if (c.warn && warnRow) td.classList.add('warn-cell');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
}

// ---- Graphique : prix réel + prédictions par modèle (Plotly, zoomable) ------
function renderAssetChart(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short, st = assetState[ticker];
  const container = document.getElementById(`chart-${s}`);
  const dateRangeEl = document.getElementById(`chart-daterange-${s}`);

  const priceBucket = (pricesBucket(st.date) || {})[ticker];
  if (!priceBucket) {
    container.innerHTML = '<div class="no-data">Aucune donnée de prix pour cette date de run.</div>';
    dateRangeEl.textContent = '';
    return;
  }

  const allPoints = priceBucket.points;
  const trainEnd = priceBucket.train_end;
  const trainPoints = trainEnd ? allPoints.filter(p => p.date <= trainEnd) : allPoints;
  const valPoints = trainEnd ? allPoints.filter(p => p.date >= trainEnd) : [];
  const windowStart = priceBucket.window_start || (trainPoints[0] || {}).date;
  const windowEnd = priceBucket.window_end || (allPoints[allPoints.length - 1] || {}).date;
  const valStart = (valPoints[0] || {}).date;
  const lastClose = (allPoints[allPoints.length - 1] || {}).close;

  dateRangeEl.textContent = trainEnd
    ? `Entraînement : ${windowStart || '—'} → ${trainEnd}   ·   Validation : ${valStart || trainEnd} → ${windowEnd || '—'}`
    : `Historique : ${windowStart || '—'} → ${windowEnd || '—'}`;

  const traces = [];
  if (!trainEnd) {
    if (st.showTrain || st.showVal) {
      traces.push({
        x: allPoints.map(p => p.date), y: allPoints.map(p => p.close),
        mode: 'lines', name: 'Réel',
        line: { color: ACTUAL_COLOR, width: 1.6 },
        hovertemplate: '%{x}<br>%{y:.2f}<extra>Réel</extra>',
      });
    }
  } else {
    if (st.showTrain && trainPoints.length) {
      traces.push({
        x: trainPoints.map(p => p.date), y: trainPoints.map(p => p.close),
        mode: 'lines', name: 'Réel (entraînement)',
        line: { color: TRAIN_COLOR, width: 1.6 },
        hovertemplate: '%{x}<br>%{y:.2f}<extra>Entraînement</extra>',
      });
    }
    if (st.showVal && valPoints.length) {
      traces.push({
        x: valPoints.map(p => p.date), y: valPoints.map(p => p.close),
        mode: 'lines', name: 'Réel (validation)',
        line: { color: ACTUAL_COLOR, width: 2.2 },
        hovertemplate: '%{x}<br>%{y:.2f}<extra>Validation</extra>',
      });
    }
  }

  const checked = MODELS.filter(m => st.models.has(m));
  const predBucket = (predsBucket(st.date) || {})[ticker] || {};
  const forecastAnchorDate = windowEnd;
  const forecastAnchorClose = (allPoints.find(p => p.date === forecastAnchorDate) || {}).close ?? lastClose;
  const forecastTargetDate = addDays(forecastAnchorDate, FORECAST_DAYS_OFFSET[st.horizon] || 1);
  let anyForecastPlotted = false;

  checked.forEach(m => {
    const color = MODEL_COLORS[m];
    let legendAdded = false;

    if (st.showVal) {
      const series = (predBucket[m] || {})[st.horizon];
      if (series && series.length) {
        if (st.showPI) {
          traces.push({
            x: series.map(p => p.date), y: series.map(p => p.pi_upper),
            mode: 'lines', line: { width: 0, color }, legendgroup: m,
            showlegend: false, hoverinfo: 'skip',
          });
          traces.push({
            x: series.map(p => p.date), y: series.map(p => p.pi_lower),
            mode: 'lines', line: { width: 0, color }, fill: 'tonexty',
            fillcolor: hexToRgba(color, 0.16), legendgroup: m,
            showlegend: false, hoverinfo: 'skip',
          });
        }
        traces.push({
          x: series.map(p => p.date), y: series.map(p => p.predicted),
          mode: 'lines+markers', name: m, legendgroup: m, showlegend: true,
          line: { color, width: 1.8, dash: 'dot' }, marker: { color, size: 4 },
          hovertemplate: '%{x}<br>%{y:.2f}<extra>' + m + ' (backtest)</extra>',
        });
        legendAdded = true;
      }
    }

    if (st.showPred) {
      const rec = DATA.records.find(r => r.asset === ticker && r.model === m
        && r.horizon === st.horizon && r.run_date === st.date);
      if (rec && rec.forecast_predicted != null && forecastAnchorClose != null) {
        traces.push({
          x: [forecastAnchorDate, forecastTargetDate], y: [forecastAnchorClose, rec.forecast_predicted],
          mode: 'lines', line: { color, width: 1.4, dash: 'dot' },
          legendgroup: m, showlegend: false, hoverinfo: 'skip',
        });
        const hi = rec.forecast_pi_upper != null ? rec.forecast_pi_upper - rec.forecast_predicted : 0;
        const lo = rec.forecast_pi_lower != null ? rec.forecast_predicted - rec.forecast_pi_lower : 0;
        traces.push({
          x: [forecastTargetDate], y: [rec.forecast_predicted],
          mode: 'markers', name: m, legendgroup: m, showlegend: !legendAdded,
          marker: { color, size: 10, symbol: 'diamond', line: { color: ACTUAL_COLOR, width: 1 } },
          error_y: { type: 'data', symmetric: false, array: [hi], arrayminus: [lo], color, thickness: 1.5, width: 4 },
          hovertemplate: `Prévision ${st.horizon}<br>%{x}<br>%{y:.2f}<extra>${m}</extra>`,
        });
        anyForecastPlotted = true;
      }
    }
  });

  const shapes = [];
  const annotations = [];
  if (trainEnd) {
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: trainEnd, x1: trainEnd, y0: 0, y1: 1,
      line: { color: AXIS_TEXT_COLOR, width: 1, dash: 'dash' },
    });
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: windowEnd, x1: windowEnd, y0: 0, y1: 1,
      line: { color: AXIS_TEXT_COLOR, width: 1, dash: 'dash' },
    });
    if (trainPoints.length) {
      annotations.push({
        x: trainPoints[Math.floor(trainPoints.length / 2)].date, y: 1, yref: 'paper', yanchor: 'bottom',
        text: 'Entraînement', showarrow: false, font: { size: 11, color: AXIS_TEXT_COLOR },
      });
    }
    if (valPoints.length) {
      annotations.push({
        x: valPoints[Math.floor(valPoints.length / 2)].date, y: 1, yref: 'paper', yanchor: 'bottom',
        text: 'Validation', showarrow: false, font: { size: 11, color: AXIS_TEXT_COLOR },
      });
    }
    if (anyForecastPlotted) {
      annotations.push({
        x: forecastTargetDate, y: 1, yref: 'paper', yanchor: 'bottom',
        text: 'Prévision', showarrow: false, font: { size: 11, color: AXIS_TEXT_COLOR },
      });
    }
  }
  if (lastClose != null) {
    shapes.push({
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: lastClose, y1: lastClose,
      line: { color: AXIS_TEXT_COLOR, width: 1, dash: 'dot' },
    });
  }

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: AXIS_TEXT_COLOR, family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12 },
    margin: { l: 55, r: 20, t: 30, b: 40 },
    xaxis: { gridcolor: GRID_COLOR, showgrid: true },
    yaxis: { gridcolor: GRID_COLOR, showgrid: true, title: 'Prix' },
    shapes,
    annotations,
    legend: { orientation: 'h', y: -0.15 },
    hovermode: 'x unified',
  };

  if (!traces.length) {
    container.innerHTML = '<div class="no-data">Sélectionnez au moins une partie (entraînement / validation / prédiction) et un modèle.</div>';
    return;
  }
  Plotly.newPlot(`chart-${s}`, traces, layout, { responsive: true, displaylogo: false });
}

// =============================================================================
// Onglet Comparaison (tous actifs confondus) — inchangé fonctionnellement
// =============================================================================

let comparisonState = { horizon: 'D1' };
let comparisonInitialized = false;

function comparisonFilteredRecords() {
  return DATA.records.filter(r => r.horizon === comparisonState.horizon);
}

function renderComparisonToggle() {
  const el = document.getElementById('horizon-toggle');
  el.innerHTML = '';
  const horizons = [...new Set(DATA.records.map(r => r.horizon))].sort();
  horizons.forEach(h => {
    const btn = document.createElement('button');
    btn.textContent = h === 'D1' ? 'D+1' : (h === 'D7' ? 'D+7' : h);
    btn.className = h === comparisonState.horizon ? 'active' : '';
    btn.addEventListener('click', () => { comparisonState.horizon = h; renderComparisonTab(); });
    el.appendChild(btn);
  });
}

function renderComparisonLegend() {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  MODELS.forEach(m => {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const sw = document.createElement('span');
    sw.className = 'legend-swatch';
    sw.style.background = MODEL_COLORS[m];
    const label = document.createElement('span');
    label.textContent = m;
    item.appendChild(sw); item.appendChild(label);
    el.appendChild(item);
  });
}

function renderComparisonStatTiles() {
  const recs = comparisonFilteredRecords();
  const el = document.getElementById('stat-tiles');
  el.innerHTML = '';
  const avg = (key) => {
    const vals = recs.map(r => r[key]).filter(v => v !== null && v !== undefined);
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };
  const tiles = [
    { label: 'Combinaisons (horizon sélectionné)', value: String(recs.length) },
    { label: 'Exact. directionnelle moyenne', value: fmt(avg('directional_accuracy'), 1) + ' %' },
    { label: 'Couverture PI 95% moyenne', value: fmt(avg('pi_coverage_95'), 1) + ' %' },
    { label: 'MAPE moyen', value: fmt(avg('MAPE'), 2) + ' %' },
  ];
  tiles.forEach(t => {
    const d = document.createElement('div');
    d.className = 'stat-tile';
    d.innerHTML = `<div class="label">${t.label}</div><div class="value">${t.value}</div>`;
    el.appendChild(d);
  });
}

// ---- Grouped bar chart (catégories = actifs, séries = modèles, échelle 0-100 partagée) ----
function niceMax(v) { return v <= 0 ? 1 : v; }

function renderGroupedChart(containerId, valueKey, opts) {
  const recs = comparisonFilteredRecords();
  const assets = DATA.assets.filter(a => recs.some(r => r.asset === a.ticker));
  const W = 900, H = 300, padL = 40, padR = 12, padT = 12, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const yMax = opts.yMax || 100;
  const groupW = plotW / Math.max(assets.length, 1);
  const barW = Math.min(22, (groupW - 12) / Math.max(MODELS.length, 1));

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', H);

  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const v = (yMax / nTicks) * i;
    const y = padT + plotH - (v / yMax) * plotH;
    const line = document.createElementNS(svg.namespaceURI, 'line');
    line.setAttribute('class', 'grid-line');
    line.setAttribute('x1', padL); line.setAttribute('x2', W - padR);
    line.setAttribute('y1', y); line.setAttribute('y2', y);
    svg.appendChild(line);
    const label = document.createElementNS(svg.namespaceURI, 'text');
    label.setAttribute('x', padL - 6); label.setAttribute('y', y + 4);
    label.setAttribute('font-size', '10'); label.setAttribute('text-anchor', 'end');
    label.textContent = Math.round(v);
    svg.appendChild(label);
  }

  if (opts.refValue !== undefined) {
    const y = padT + plotH - (opts.refValue / yMax) * plotH;
    const line = document.createElementNS(svg.namespaceURI, 'line');
    line.setAttribute('class', 'ref-line');
    line.setAttribute('x1', padL); line.setAttribute('x2', W - padR);
    line.setAttribute('y1', y); line.setAttribute('y2', y);
    svg.appendChild(line);
    const label = document.createElementNS(svg.namespaceURI, 'text');
    label.setAttribute('x', W - padR); label.setAttribute('y', y - 4);
    label.setAttribute('font-size', '10'); label.setAttribute('text-anchor', 'end');
    label.textContent = opts.refLabel || '';
    svg.appendChild(label);
  }

  const baseline = document.createElementNS(svg.namespaceURI, 'line');
  baseline.setAttribute('class', 'axis-line');
  baseline.setAttribute('x1', padL); baseline.setAttribute('x2', W - padR);
  baseline.setAttribute('y1', padT + plotH); baseline.setAttribute('y2', padT + plotH);
  svg.appendChild(baseline);

  assets.forEach((asset, gi) => {
    const groupX = padL + gi * groupW;
    const groupInnerW = barW * MODELS.length;
    const groupStart = groupX + (groupW - groupInnerW) / 2;

    MODELS.forEach((model, mi) => {
      const rec = recs.find(r => r.asset === asset.ticker && r.model === model);
      const label = document.createElementNS(svg.namespaceURI, 'text');
      if (mi === 0) {
        label.setAttribute('x', groupX + groupW / 2);
        label.setAttribute('y', H - padB + 16);
        label.setAttribute('font-size', '11'); label.setAttribute('text-anchor', 'middle');
        label.textContent = asset.short;
        svg.appendChild(label);
      }
      if (!rec || rec[valueKey] === null || rec[valueKey] === undefined) return;
      const v = rec[valueKey];
      const barH = (v / yMax) * plotH;
      const x = groupStart + mi * barW;
      const y = padT + plotH - barH;
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      rect.setAttribute('class', 'bar');
      rect.setAttribute('x', x + 1); rect.setAttribute('y', y);
      rect.setAttribute('width', Math.max(barW - 2, 1)); rect.setAttribute('height', Math.max(barH, 0));
      rect.setAttribute('rx', 2); rect.setAttribute('fill', MODEL_COLORS[model]);
      rect.addEventListener('pointermove', (evt) => showTooltip(evt, `${asset.label} — ${model}`,
        [{ color: MODEL_COLORS[model], name: opts.label, value: fmt(v, opts.digits) + (opts.suffix || '') }]));
      rect.addEventListener('pointerleave', hideTooltip);
      svg.appendChild(rect);
    });
  });

  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!assets.length) {
    container.innerHTML = '<div class="no-data">Aucune donnée pour cet horizon.</div>';
    return;
  }
  container.appendChild(svg);
}

// ---- Small-multiples (un panneau par actif, échelle propre, RMSE/MAE) ----
function renderSmallMultiples(containerId, valueKey, opts) {
  const recs = comparisonFilteredRecords();
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  const assets = DATA.assets.filter(a => recs.some(r => r.asset === a.ticker));
  if (!assets.length) {
    container.innerHTML = '<div class="no-data">Aucune donnée pour cet horizon.</div>';
    return;
  }

  assets.forEach(asset => {
    const panel = document.createElement('div');
    panel.className = 'panel';
    const title = document.createElement('div');
    title.className = 'panel-title';
    title.textContent = asset.label;
    panel.appendChild(title);

    const values = MODELS.map(m => {
      const rec = recs.find(r => r.asset === asset.ticker && r.model === m);
      return { model: m, value: rec ? rec[valueKey] : null };
    });
    const maxV = niceMax(Math.max(0, ...values.map(v => v.value || 0)));

    const W = 220, H = 160, padL = 34, padR = 8, padT = 8, padB = 22;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const barW = Math.min(22, plotW / values.length - 6);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%'); svg.setAttribute('height', H);

    [0, 0.5, 1].forEach(f => {
      const v = maxV * f;
      const y = padT + plotH - f * plotH;
      const line = document.createElementNS(svg.namespaceURI, 'line');
      line.setAttribute('class', 'grid-line');
      line.setAttribute('x1', padL); line.setAttribute('x2', W - padR);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      svg.appendChild(line);
      const label = document.createElementNS(svg.namespaceURI, 'text');
      label.setAttribute('x', padL - 5); label.setAttribute('y', y + 3);
      label.setAttribute('font-size', '9'); label.setAttribute('text-anchor', 'end');
      label.textContent = fmt(v, 0);
      svg.appendChild(label);
    });

    const baseline = document.createElementNS(svg.namespaceURI, 'line');
    baseline.setAttribute('class', 'axis-line');
    baseline.setAttribute('x1', padL); baseline.setAttribute('x2', W - padR);
    baseline.setAttribute('y1', padT + plotH); baseline.setAttribute('y2', padT + plotH);
    svg.appendChild(baseline);

    const slotW = plotW / values.length;
    values.forEach((v, i) => {
      const x = padL + i * slotW + (slotW - barW) / 2;
      if (v.value === null || v.value === undefined) return;
      const barH = (v.value / maxV) * plotH;
      const y = padT + plotH - barH;
      const rect = document.createElementNS(svg.namespaceURI, 'rect');
      rect.setAttribute('class', 'bar');
      rect.setAttribute('x', x); rect.setAttribute('y', y);
      rect.setAttribute('width', barW); rect.setAttribute('height', Math.max(barH, 0));
      rect.setAttribute('rx', 2); rect.setAttribute('fill', MODEL_COLORS[v.model]);
      rect.addEventListener('pointermove', (evt) => showTooltip(evt, `${asset.label} — ${v.model}`,
        [{ color: MODEL_COLORS[v.model], name: opts.label, value: fmt(v.value, opts.digits) + (opts.suffix || '') }]));
      rect.addEventListener('pointerleave', hideTooltip);
      svg.appendChild(rect);
    });

    panel.appendChild(svg);
    container.appendChild(panel);
  });
}

// ---- Table (onglet Comparaison) --------------------------------------------
const TABLE_COLS = [
  { key: 'asset', label: 'Actif' },
  { key: 'model', label: 'Modèle' },
  { key: 'RMSE', label: 'RMSE', digits: 4 },
  { key: 'MAE', label: 'MAE', digits: 4 },
  { key: 'MAPE', label: 'MAPE (%)', digits: 2 },
  { key: 'directional_accuracy', label: 'Exact. dir. (%)', digits: 2 },
  { key: 'pi_coverage_95', label: 'Couv. PI 95 (%)', digits: 2 },
  { key: 'theil_u', label: "Theil's U", digits: 3 },
  { key: 'MASE', label: 'MASE', digits: 3 },
  { key: 'dm_p', label: 'DM p', digits: 3 },
  { key: 'skill_vs_naive', label: 'Skill vs naïf' },
  { key: 'n_val', label: 'n_val', digits: 0 },
];
let sortState = { key: 'asset', dir: 1 };

function renderComparisonTable() {
  const search = (document.getElementById('table-search').value || '').toLowerCase();
  let recs = comparisonFilteredRecords().filter(r =>
    !search || r.asset.toLowerCase().includes(search) || r.model.toLowerCase().includes(search));
  recs = recs.slice().sort((a, b) => {
    const av = a[sortState.key], bv = b[sortState.key];
    if (av === bv) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return (av > bv ? 1 : -1) * sortState.dir;
  });

  const table = document.getElementById('data-table');
  table.innerHTML = '';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  TABLE_COLS.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label + (sortState.key === col.key ? (sortState.dir === 1 ? ' ▲' : ' ▼') : '');
    th.addEventListener('click', () => {
      sortState = { key: col.key, dir: sortState.key === col.key ? -sortState.dir : 1 };
      renderComparisonTable();
    });
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  recs.forEach(r => {
    const tr = document.createElement('tr');
    TABLE_COLS.forEach(col => {
      const td = document.createElement('td');
      const v = r[col.key];
      td.textContent = col.digits !== undefined ? fmt(v, col.digits) : (v ?? '—');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function renderComparisonTab() {
  renderComparisonToggle();
  renderComparisonLegend();
  renderComparisonStatTiles();
  renderGroupedChart('chart-diracc', 'directional_accuracy',
    { yMax: 100, refValue: 50, refLabel: 'hasard (50%)', label: 'Exact. directionnelle', digits: 1, suffix: ' %' });
  renderGroupedChart('chart-picov', 'pi_coverage_95',
    { yMax: 100, refValue: 95, refLabel: 'cible 95%', label: 'Couverture PI 95%', digits: 1, suffix: ' %' });
  renderSmallMultiples('chart-rmse', 'RMSE', { label: 'RMSE', digits: 4 });
  renderSmallMultiples('chart-mae', 'MAE', { label: 'MAE', digits: 4 });
  renderComparisonTable();
}

// =============================================================================
// Boot
// =============================================================================

async function boot() {
  seedDataCacheFromInline();
  renderSubtitle();
  buildTabBar();
  // Toutes les fiches actif démarrent sur la même date (la plus récente, cf. assetState
  // ci-dessus) -- un seul fetch couvre le rendu KPI initial de tous les actifs.
  const latestDate = DATA.run_dates[DATA.run_dates.length - 1];
  if (latestDate) await ensureDateData(latestDate);
  buildAssetPanels();
  renderComparisonTab();
  document.getElementById('table-search').addEventListener('input', renderComparisonTable);
}
boot();
</script>
"""


def main():
    p = argparse.ArgumentParser(description="Génère un dashboard HTML des KPI depuis Run/")
    p.add_argument("--run-root", default=str(REPO_ROOT / "Run"))
    p.add_argument("--out", default=None, help="défaut : <run-root>/dashboard.html")
    p.add_argument("--inline", action="store_true",
                    help="mono-fichier autonome (embarque toutes les séries) -- "
                         "ouvrable en file://, pratique en dev. Sans ce flag (défaut, "
                         "utilisé en CI) : coquille légère + data/<date>.json à côté de "
                         "--out, à servir via un serveur HTTP (fetch ne marche pas en file://).")
    args = p.parse_args()

    run_root = Path(args.run_root)
    out_path = Path(args.out) if args.out else run_root / "dashboard.html"

    run_data = collect_run_data(run_root)
    if not run_data["records"]:
        print(f"Aucun metrics.json trouvé sous {run_root}")
    html = render_html(run_data, str(run_root), external_series=not args.inline)
    out_path.write_text(html, encoding="utf-8")
    if args.inline:
        print(f"Dashboard généré (mono-fichier) : {out_path}  ({len(run_data['records'])} combinaisons)")
    else:
        data_dir = out_path.parent / "data"
        write_date_series_files(run_data, data_dir)
        print(f"Dashboard généré : {out_path} + {len(run_data['run_dates'])} fichier(s) dans {data_dir}"
              f"  ({len(run_data['records'])} combinaisons)")
        print("  Aperçu local : servez le dossier via `python -m http.server` "
              "(fetch() ne fonctionne pas en file://) ou utilisez --inline.")


if __name__ == "__main__":
    main()
