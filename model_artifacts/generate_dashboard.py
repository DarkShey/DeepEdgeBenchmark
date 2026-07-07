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

Exécution (depuis DeepEdgeBenchmark/) :
    python -m model_artifacts.generate_dashboard
    python -m model_artifacts.generate_dashboard --run-root Run --out Run/dashboard.html
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ORDER = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive"]
# Palette catégorielle validée (skill dataviz) — slots 1..5 dans l'ordre fixe.
MODEL_COLORS_LIGHT = {
    "ARIMA-GARCH": "#2a78d6", "SARIMA": "#1baf7a", "Prophet": "#eda100",
    "LSTM": "#008300", "Naive": "#4a3aa7",
}
MODEL_COLORS_DARK = {
    "ARIMA-GARCH": "#3987e5", "SARIMA": "#199e70", "Prophet": "#c98500",
    "LSTM": "#008300", "Naive": "#9085e9",
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
        # Le préfixe du nom de dossier (YYYYMMDD-...) fait foi pour regrouper par date de
        # run — indépendant du nombre de tirets dans le nom de l'actif (BTC-USD, ZN=F...).
        run_date = combo_dir.name.split("-", 1)[0]
        model, asset, horizon = metrics.get("model"), metrics.get("asset"), metrics.get("horizon")

        records.append({
            "model": model, "asset": asset, "asset_class": metadata.get("asset_class", ""),
            "horizon": horizon,
            "RMSE": metrics.get("RMSE"), "MAE": metrics.get("MAE"), "MAPE": metrics.get("MAPE"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "pi_coverage_95": metrics.get("pi_coverage_95"),
            "pi_width_min": metrics.get("pi_width_min"),
            "pi_width_mean": metrics.get("pi_width_mean"),
            "pi_width_max": metrics.get("pi_width_max"),
            "n_val": metrics.get("n_val"),
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
                }
                prices[run_date] = asset_bucket

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


def render_html(run_data: dict, run_root_label: str) -> str:
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
        "predictions": run_data["predictions"],
        "prices": run_data["prices"],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


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

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const MODEL_COLORS = isDark ? DATA.model_colors_dark : DATA.model_colors_light;
const MODELS = DATA.model_order.filter(m => DATA.models_present.includes(m));
const ACTUAL_COLOR = isDark ? '#ffffff' : '#0b0b0b';
const GRID_COLOR = isDark ? '#2c2c2a' : '#e1e0d9';
const AXIS_TEXT_COLOR = isDark ? '#c3c2b7' : '#52514e';

function fmt(v, digits) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
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
        <div class="controls-row" style="margin-bottom:12px;">
          <label class="field-label">
            <input type="checkbox" id="showpi-${s}" checked> Afficher les intervalles de confiance
          </label>
          <button class="btn-ghost" id="resetzoom-${s}">Réinitialiser le zoom</button>
        </div>
        <div class="chart-wrap" id="chart-${s}"></div>
      </div>
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
  dateSel.addEventListener('change', () => { st.date = dateSel.value; refreshAssetTab(ticker); });

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

  renderAssetKpis(ticker);
}

function refreshAssetTab(ticker) {
  renderAssetKpis(ticker);
  if (assetState[ticker].subtab === 'chart') renderAssetChart(ticker);
}

// ---- KPIs par modèle (cartes) + breakdown modèle x horizon ------------------
const BREAKDOWN_COLS = [
  { key: 'model', label: 'Modèle' },
  { key: 'horizon', label: 'Horizon' },
  { key: 'RMSE', label: 'RMSE', digits: 4 },
  { key: 'MAE', label: 'MAE', digits: 4 },
  { key: 'MAPE', label: 'MAPE (%)', digits: 2 },
  { key: 'directional_accuracy', label: 'Exact. dir. (%)', digits: 2 },
  { key: 'pi_coverage_95', label: 'Couv. PI 95 (%)', digits: 2 },
  { key: 'pi_width_min', label: 'Larg. PI min', digits: 4 },
  { key: 'pi_width_mean', label: 'Larg. PI moy.', digits: 4 },
  { key: 'pi_width_max', label: 'Larg. PI max', digits: 4 },
  { key: 'n_val', label: 'n_val', digits: 0 },
];

function renderAssetKpis(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short, st = assetState[ticker];
  const checked = MODELS.filter(m => st.models.has(m));

  const cardsEl = document.getElementById(`kpi-cards-${s}`);
  cardsEl.innerHTML = '';
  if (!checked.length) {
    cardsEl.innerHTML = '<div class="no-data">Sélectionnez au moins un modèle.</div>';
  } else {
    checked.forEach(m => {
      const rec = DATA.records.find(r => r.asset === ticker && r.model === m
        && r.horizon === st.horizon && r.run_date === st.date);
      const card = document.createElement('div');
      card.className = 'kpi-card';
      const rowsHtml = !rec ? '<div class="no-data">Pas de données</div>' : [
        ['RMSE', fmt(rec.RMSE, 4)],
        ['MAE', fmt(rec.MAE, 4)],
        ['MAPE', fmt(rec.MAPE, 2) + ' %'],
        ['Exact. directionnelle', fmt(rec.directional_accuracy, 1) + ' %'],
        ['Couverture PI 95%', fmt(rec.pi_coverage_95, 1) + ' %'],
        ['Largeur PI min/moy/max', `${fmt(rec.pi_width_min, 2)} / ${fmt(rec.pi_width_mean, 2)} / ${fmt(rec.pi_width_max, 2)}`],
        ['n (validation)', rec.n_val ?? '—'],
      ].map(([k, v]) => `<div class="kpi-row"><span>${k}</span><b>${v}</b></div>`).join('');
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
    th.textContent = c.label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  recs.forEach(r => {
    const tr = document.createElement('tr');
    BREAKDOWN_COLS.forEach(c => {
      const td = document.createElement('td');
      const v = r[c.key];
      td.textContent = c.digits !== undefined ? fmt(v, c.digits) : (v ?? '—');
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

  const priceBucket = (DATA.prices[st.date] || {})[ticker];
  if (!priceBucket) {
    container.innerHTML = '<div class="no-data">Aucune donnée de prix pour cette date de run.</div>';
    return;
  }

  const traces = [{
    x: priceBucket.points.map(p => p.date),
    y: priceBucket.points.map(p => p.close),
    mode: 'lines', name: 'Réel',
    line: { color: ACTUAL_COLOR, width: 1.6 },
    hovertemplate: '%{x}<br>%{y:.2f}<extra>Réel</extra>',
  }];

  const checked = MODELS.filter(m => st.models.has(m));
  const predBucket = (DATA.predictions[st.date] || {})[ticker] || {};
  checked.forEach(m => {
    const series = (predBucket[m] || {})[st.horizon];
    if (!series || !series.length) return;
    const color = MODEL_COLORS[m];

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
      mode: 'lines+markers', name: m, legendgroup: m,
      line: { color, width: 1.8, dash: 'dot' }, marker: { color, size: 4 },
      hovertemplate: '%{x}<br>%{y:.2f}<extra>' + m + '</extra>',
    });
  });

  const shapes = [];
  if (priceBucket.train_end) {
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: priceBucket.train_end, x1: priceBucket.train_end, y0: 0, y1: 1,
      line: { color: AXIS_TEXT_COLOR, width: 1, dash: 'dash' },
    });
  }

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: AXIS_TEXT_COLOR, family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12 },
    margin: { l: 55, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: GRID_COLOR, showgrid: true },
    yaxis: { gridcolor: GRID_COLOR, showgrid: true, title: 'Prix' },
    shapes,
    legend: { orientation: 'h', y: -0.15 },
    hovermode: 'x unified',
  };

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

renderSubtitle();
buildTabBar();
buildAssetPanels();
renderComparisonTab();
document.getElementById('table-search').addEventListener('input', renderComparisonTable);
</script>
"""


def main():
    p = argparse.ArgumentParser(description="Génère un dashboard HTML des KPI depuis Run/")
    p.add_argument("--run-root", default=str(REPO_ROOT / "Run"))
    p.add_argument("--out", default=None, help="défaut : <run-root>/dashboard.html")
    args = p.parse_args()

    run_root = Path(args.run_root)
    out_path = Path(args.out) if args.out else run_root / "dashboard.html"

    run_data = collect_run_data(run_root)
    if not run_data["records"]:
        print(f"Aucun metrics.json trouvé sous {run_root}")
    html = render_html(run_data, str(run_root))
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard généré : {out_path}  ({len(run_data['records'])} combinaisons)")


if __name__ == "__main__":
    main()
