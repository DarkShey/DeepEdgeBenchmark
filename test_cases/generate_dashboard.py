"""
test_cases/generate_dashboard.py — Dashboard HTML des test cases de transition de régime
==========================================================================================
Lit test_cases/results/<tc_id>/<ticker>/<transition_date>/<model>.json (écrits par
test_cases/run_test_cases.py) et génère une page HTML autonome, même charte visuelle que
model_artifacts/generate_dashboard.py (Run/dashboard.html) : un onglet par actif, un
sous-onglet par test case, un sélecteur d'occurrence (date de transition), un petit
graphe Plotly (prix + fond coloré par régime + prévisions D+1/D+7 par modèle) et un
tableau de données par modèle.

Exécution (depuis DeepEdgeBenchmark/) :
    python -m test_cases.generate_dashboard
    python -m test_cases.generate_dashboard --results-root test_cases/results --out test_cases/dashboard.html
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

from test_cases.registry_models import MODELS
from test_cases.registry_assets import ASSETS
from test_cases.registry_test_cases import TEST_CASES
from test_cases.transitions import load_series

MODEL_ORDER = [m["id"] for m in MODELS]
# Même palette que model_artifacts/generate_dashboard.py (Naive exclu : hors périmètre ici).
MODEL_COLORS_LIGHT = {
    "ARIMA-GARCH": "#2a78d6", "SARIMA": "#1baf7a", "Prophet": "#eda100",
    "LSTM": "#008300", "TSDiff": "#d64550",
}
MODEL_COLORS_DARK = {
    "ARIMA-GARCH": "#3987e5", "SARIMA": "#199e70", "Prophet": "#c98500",
    "LSTM": "#008300", "TSDiff": "#e5606b",
}
# Même palette que calibration/regime/assets.py (_REGIME_BG) — semi-transparente, lisible
# sur fond clair ou sombre.
REGIME_BG = {
    "calm": "rgba(46,204,113,0.24)", "bull": "rgba(241,196,15,0.22)",
    "bear": "rgba(74,105,189,0.24)", "stress": "rgba(231,76,60,0.26)",
}
REGIME_LABELS_FR = {"calm": "Calme", "bull": "Haussier", "bear": "Baissier", "stress": "Stress"}

WINDOW_BEFORE = 90   # jours de bourse de contexte avant le cutoff, pour le graphe
WINDOW_AFTER = 15    # jours de bourse de contexte après la cible la plus lointaine


def _num(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _build_window(df: pd.DataFrame, cutoff_idx: int, max_target_idx: int) -> list:
    start = max(0, cutoff_idx - WINDOW_BEFORE)
    end = min(len(df) - 1, max_target_idx + WINDOW_AFTER)
    sub = df.iloc[start : end + 1]
    return [
        {"date": d.strftime("%Y-%m-%d"), "close": _num(c), "regime": r}
        for d, c, r in zip(sub["Date"], sub["Close"], sub["Regime"])
    ]


def collect_results(results_root: Path) -> dict:
    """Parcourt results/<tc_id>/<ticker>/<transition_date>/<model>.json et construit :
    occurrences (dates disponibles par tc x actif), records (le detail par modèle) et
    windows (contexte de prix + régime autour de chaque occurrence, pour le graphe)."""
    occurrences: dict = {}
    records: dict = {}
    windows: dict = {}
    series_cache: dict = {}

    def get_series(ticker: str) -> pd.DataFrame:
        if ticker not in series_cache:
            series_cache[ticker] = load_series(ticker)
        return series_cache[ticker]

    if not results_root.exists():
        return {"occurrences": occurrences, "records": records, "windows": windows}

    for tc_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        tc_id = tc_dir.name
        for ticker_dir in sorted(p for p in tc_dir.iterdir() if p.is_dir()):
            ticker = ticker_dir.name
            for occ_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
                transition_date = occ_dir.name
                model_records = {}
                for model_file in sorted(occ_dir.glob("*.json")):
                    model_records[model_file.stem] = json.loads(model_file.read_text())
                if not model_records:
                    continue

                occurrences.setdefault(tc_id, {}).setdefault(ticker, [])
                if transition_date not in occurrences[tc_id][ticker]:
                    occurrences[tc_id][ticker].append(transition_date)
                records.setdefault(tc_id, {}).setdefault(ticker, {})[transition_date] = model_records

                any_rec = next(iter(model_records.values()))
                df = get_series(ticker)
                cutoff_ts = pd.Timestamp(any_rec["cutoff_date"])
                matches = df.index[df["Date"] == cutoff_ts]
                if len(matches) and any_rec["horizons"]:
                    cutoff_idx = int(matches[0])
                    max_h = max(int(h) for h in any_rec["horizons"].keys())
                    window_points = _build_window(df, cutoff_idx, cutoff_idx + max_h)
                else:
                    window_points = []
                windows.setdefault(tc_id, {}).setdefault(ticker, {})[transition_date] = window_points

    for tc_id in occurrences:
        for ticker in occurrences[tc_id]:
            occurrences[tc_id][ticker].sort()

    return {"occurrences": occurrences, "records": records, "windows": windows}


def render_html(data: dict) -> str:
    models_present = {m for tc in data["records"].values() for a in tc.values()
                      for occ in a.values() for m in occ}
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z"),
        "model_order": [m for m in MODEL_ORDER if m in models_present] or MODEL_ORDER,
        "model_colors_light": MODEL_COLORS_LIGHT,
        "model_colors_dark": MODEL_COLORS_DARK,
        "regime_bg": REGIME_BG,
        "regime_labels": REGIME_LABELS_FR,
        "test_cases": [{"id": tc["id"], "label": tc["label"], "description": tc["description"]}
                       for tc in TEST_CASES],
        "assets": [{"ticker": a["ticker"], "label": a["label"], "short": a["short"], "color": a["color"]}
                  for a in ASSETS],
        "occurrences": data["occurrences"],
        "records": data["records"],
        "windows": data["windows"],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


HTML_TEMPLATE = r"""<title>Test cases — Transitions de régime</title>
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
  --ok-color:       #1baf7a;
  --bad-color:      #d64550;
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
    --ok-color:       #2ecc9a;
    --bad-color:      #e5606b;
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
.controls-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
.select-box {
  font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-primary);
}
.field-label { font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
thead th { text-align: left; padding: 8px 10px; color: var(--text-secondary); font-weight: 600;
  border-bottom: 1px solid var(--grid-line); white-space: nowrap; }
tbody td { padding: 6px 10px; border-bottom: 1px solid var(--grid-line); font-variant-numeric: tabular-nums; }
tbody tr:hover { background: rgba(128,128,128,0.06); }
.no-data { color: var(--text-muted); font-size: 13px; padding: 12px 0; }
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
.subtabbar { display: inline-flex; border: 1px solid var(--border-ring); border-radius: 8px; overflow: hidden; }
.subtabbar button {
  font: inherit; font-size: 13px; padding: 7px 16px; border: none; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
}
.subtabbar button.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.sub-panel { display: none; }
.sub-panel.active { display: block; }
.tc-description { font-size: 12.5px; color: var(--text-secondary); margin: -8px 0 16px; max-width: 900px; }
.kpi-cards { display: flex; gap: 14px; flex-wrap: wrap; }
.kpi-card { flex: 1 1 240px; border: 1px solid var(--border-ring); border-radius: 8px; padding: 12px 14px; }
.kpi-card-title { font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.kpi-card-title .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.kpi-horizon { font-size: 11.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em;
  margin: 8px 0 2px; }
.kpi-row { display: flex; justify-content: space-between; font-size: 12.5px; padding: 3px 0; color: var(--text-secondary); }
.kpi-row b { color: var(--text-primary); font-variant-numeric: tabular-nums; font-weight: 600; }
.badge-ok { color: var(--ok-color); font-weight: 700; }
.badge-bad { color: var(--bad-color); font-weight: 700; }
.chart-wrap { min-height: 420px; }
.hp-note { font-size: 11.5px; color: var(--text-muted); margin-top: 8px; }
</style>

<h1>Test cases — Transitions de régime</h1>
<p class="subtitle" id="subtitle"></p>

<div class="tabbar" id="asset-tabbar"></div>
<div id="asset-panels"></div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const DATA = __DATA_JSON__;

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const MODEL_COLORS = isDark ? DATA.model_colors_dark : DATA.model_colors_light;
const MODELS = DATA.model_order;
const ACTUAL_COLOR = isDark ? '#ffffff' : '#0b0b0b';
const GRID_COLOR = isDark ? '#2c2c2a' : '#e1e0d9';
const AXIS_TEXT_COLOR = isDark ? '#c3c2b7' : '#52514e';

function fmt(v, digits) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function badge(ok) {
  if (ok === null || ok === undefined) return '—';
  return ok ? '<span class="badge-ok">✓</span>' : '<span class="badge-bad">✗</span>';
}
function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16), g = parseInt(h.substring(2, 4), 16), b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderSubtitle() {
  const nTC = DATA.test_cases.length;
  const nAssets = DATA.assets.length;
  document.getElementById('subtitle').textContent =
    `${nTC} test case(s) × ${nAssets} actif(s) — généré le ${DATA.generated_at}`;
}

// ---- État par actif : test case + occurrence sélectionnés -------------------
const assetState = {};
DATA.assets.forEach(a => { assetState[a.ticker] = { tc: DATA.test_cases[0].id, occIndex: 0 }; });

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
}

function switchAssetTab(ticker) {
  document.querySelectorAll('.asset-panel').forEach(p => p.classList.toggle('active', p.dataset.asset === ticker));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.asset === ticker));
  renderTestCase(ticker);
}

function assetPanelSkeleton(a) {
  const s = a.short;
  const subBtns = DATA.test_cases.map((tc, i) =>
    `<button class="${i === 0 ? 'active' : ''}" data-tc="${tc.id}">${tc.id}</button>`).join('');
  return `
    <div class="card controls-row">
      <div class="subtabbar" id="subtab-${s}">${subBtns}</div>
      <label class="field-label">Occurrence
        <select class="select-box" id="occ-${s}"></select>
      </label>
    </div>
    <div id="tc-description-${s}" class="tc-description"></div>
    <div id="tc-content-${s}"></div>
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
  DATA.assets.forEach(a => wireAssetPanel(a));
}

function wireAssetPanel(a) {
  const s = a.short, ticker = a.ticker;
  document.getElementById(`subtab-${s}`).querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      assetState[ticker].tc = btn.dataset.tc;
      assetState[ticker].occIndex = 0;
      document.getElementById(`subtab-${s}`).querySelectorAll('button')
        .forEach(b => b.classList.toggle('active', b === btn));
      renderTestCase(ticker);
    });
  });
  document.getElementById(`occ-${s}`).addEventListener('change', (e) => {
    assetState[ticker].occIndex = Number(e.target.value);
    renderOccurrence(ticker);
  });
}

function renderTestCase(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short, st = assetState[ticker];
  const tc = DATA.test_cases.find(t => t.id === st.tc);
  document.getElementById(`tc-description-${s}`).textContent = tc.description;

  const occDates = (DATA.occurrences[st.tc] || {})[ticker] || [];
  const occSel = document.getElementById(`occ-${s}`);
  occSel.innerHTML = '';
  occDates.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = d;
    occSel.appendChild(opt);
  });
  st.occIndex = 0;

  const content = document.getElementById(`tc-content-${s}`);
  if (!occDates.length) {
    content.innerHTML = `<div class="card"><div class="no-data">Aucune occurrence historique trouvée pour ${tc.label} sur ${a.label}.</div></div>`;
    return;
  }
  content.innerHTML = `
    <div class="card">
      <h2>Prix &amp; prévisions autour de la transition</h2>
      <div class="chart-wrap" id="chart-${s}"></div>
    </div>
    <div class="card">
      <h2>KPIs par modèle</h2>
      <div class="kpi-cards" id="kpi-cards-${s}"></div>
    </div>
    <div class="card">
      <h2>Tableau détaillé</h2>
      <div style="overflow-x:auto;" id="table-wrap-${s}"></div>
    </div>
  `;
  renderOccurrence(ticker);
}

function currentRecords(ticker) {
  const st = assetState[ticker];
  const occDates = (DATA.occurrences[st.tc] || {})[ticker] || [];
  const date = occDates[st.occIndex];
  const records = ((DATA.records[st.tc] || {})[ticker] || {})[date] || {};
  const window = ((DATA.windows[st.tc] || {})[ticker] || {})[date] || [];
  return { date, records, window };
}

function renderOccurrence(ticker) {
  renderChart(ticker);
  renderKpiCards(ticker);
  renderTable(ticker);
}

// ---- Graphique ----------------------------------------------------------------
function renderChart(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short;
  const { date, records, window } = currentRecords(ticker);
  const container = document.getElementById(`chart-${s}`);
  if (!window.length) {
    container.innerHTML = '<div class="no-data">Pas de contexte de prix disponible.</div>';
    return;
  }
  container.innerHTML = '';

  const traces = [{
    x: window.map(p => p.date), y: window.map(p => p.close),
    mode: 'lines', name: 'Réel', line: { color: ACTUAL_COLOR, width: 1.8 },
    hovertemplate: '%{x}<br>%{y:.2f}<extra>Réel</extra>',
  }];

  MODELS.forEach(m => {
    const rec = records[m];
    if (!rec) return;
    const color = MODEL_COLORS[m];
    let legendAdded = false;
    Object.keys(rec.horizons).forEach(h => {
      const hh = rec.horizons[h];
      traces.push({
        x: [hh.target_date], y: [hh.y_pred],
        mode: 'markers', name: m, legendgroup: m, showlegend: !legendAdded,
        marker: { color, size: 10, symbol: 'diamond', line: { color: ACTUAL_COLOR, width: 1 } },
        error_y: { type: 'data', symmetric: false, array: [hh.y_upper - hh.y_pred],
                   arrayminus: [hh.y_pred - hh.y_lower], color, thickness: 1.5, width: 4 },
        hovertemplate: `${m} D+${h}<br>%{x}<br>%{y:.2f}<extra></extra>`,
      });
      legendAdded = true;
    });
  });

  // Fond coloré par régime
  const shapes = [];
  let i = 0;
  while (i < window.length) {
    let j = i, r = window[i].regime;
    while (j < window.length && window[j].regime === r) j++;
    shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: window[i].date, x1: window[j - 1].date, y0: 0, y1: 1,
      fillcolor: DATA.regime_bg[r] || 'rgba(128,128,128,0.1)', line: { width: 0 }, layer: 'below',
    });
    i = j;
  }
  shapes.push({
    type: 'line', xref: 'x', yref: 'paper',
    x0: date, x1: date, y0: 0, y1: 1,
    line: { color: AXIS_TEXT_COLOR, width: 1.4, dash: 'dash' },
  });

  Plotly.newPlot(`chart-${s}`, traces, {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'system-ui,sans-serif', color: AXIS_TEXT_COLOR, size: 11 },
    margin: { l: 55, r: 18, t: 10, b: 40 },
    hovermode: 'x unified',
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
    xaxis: { gridcolor: GRID_COLOR, type: 'date' },
    yaxis: { title: 'Prix', gridcolor: GRID_COLOR, tickformat: ',.0f' },
    shapes,
    annotations: [{ x: date, y: 1, yref: 'paper', yanchor: 'bottom',
      text: 'Transition', showarrow: false, font: { size: 11, color: AXIS_TEXT_COLOR } }],
  }, { responsive: true, displayModeBar: false });
}

// ---- Cartes KPI par modèle ------------------------------------------------------
function renderKpiCards(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short;
  const { records } = currentRecords(ticker);
  const cardsEl = document.getElementById(`kpi-cards-${s}`);
  cardsEl.innerHTML = '';

  const present = MODELS.filter(m => records[m]);
  if (!present.length) {
    cardsEl.innerHTML = '<div class="no-data">Aucun résultat de modèle pour cette occurrence.</div>';
    return;
  }
  present.forEach(m => {
    const rec = records[m];
    const card = document.createElement('div');
    card.className = 'kpi-card';
    let rows = `<div class="kpi-card-title"><span class="swatch" style="background:${MODEL_COLORS[m]}"></span>${m}</div>`;
    rows += `<div class="kpi-row"><span>Dernier prix (cutoff ${rec.cutoff_date})</span><b>${fmt(rec.last_close, 2)}</b></div>`;
    Object.keys(rec.horizons).sort((x, y) => Number(x) - Number(y)).forEach(h => {
      const hh = rec.horizons[h];
      rows += `<div class="kpi-horizon">D+${h} — cible ${hh.target_date}</div>`;
      rows += `<div class="kpi-row"><span>Prévision [bas–haut]</span><b>${fmt(hh.y_pred, 2)} [${fmt(hh.y_lower, 2)} – ${fmt(hh.y_upper, 2)}]</b></div>`;
      rows += `<div class="kpi-row"><span>Réel</span><b>${fmt(hh.y_true, 2)}</b></div>`;
      rows += `<div class="kpi-row"><span>Dans l'intervalle 95%</span><b>${badge(hh.in_interval)}</b></div>`;
      rows += `<div class="kpi-row"><span>Sens correct</span><b>${badge(hh.direction_correct)}</b></div>`;
      rows += `<div class="kpi-row"><span>Bat le naïf</span><b>${badge(hh.beats_naif)}</b></div>`;
    });
    if (rec.hyperparams_source) {
      rows += `<div class="hp-note">Hyperparamètres : Run/${rec.hyperparams_source.source_run_dir} (${rec.hyperparams_source.source_run_date})</div>`;
    }
    card.innerHTML = rows;
    cardsEl.appendChild(card);
  });
}

// ---- Tableau détaillé ------------------------------------------------------------
function renderTable(ticker) {
  const a = DATA.assets.find(x => x.ticker === ticker);
  const s = a.short;
  const { records } = currentRecords(ticker);
  const wrap = document.getElementById(`table-wrap-${s}`);
  const present = MODELS.filter(m => records[m]);
  if (!present.length) {
    wrap.innerHTML = '<div class="no-data">Aucune donnée.</div>';
    return;
  }
  const rows = [];
  present.forEach(m => {
    const rec = records[m];
    Object.keys(rec.horizons).sort((x, y) => Number(x) - Number(y)).forEach(h => {
      const hh = rec.horizons[h];
      rows.push({ model: m, h, ...hh });
    });
  });
  let html = '<table><thead><tr><th>Modèle</th><th>Horizon</th><th>Cible</th><th>Prévision</th>'
    + '<th>PI 95% [bas–haut]</th><th>Réel</th><th>Dans PI</th><th>Sens</th><th>Bat naïf</th>'
    + '<th>Err. abs.</th><th>Err. abs. naïf</th></tr></thead><tbody>';
  rows.forEach(r => {
    html += `<tr><td>${r.model}</td><td>D+${r.h}</td><td>${r.target_date}</td>`
      + `<td>${fmt(r.y_pred, 2)}</td><td>${fmt(r.y_lower, 2)} – ${fmt(r.y_upper, 2)}</td>`
      + `<td>${fmt(r.y_true, 2)}</td><td>${badge(r.in_interval)}</td><td>${badge(r.direction_correct)}</td>`
      + `<td>${badge(r.beats_naif)}</td><td>${fmt(r.abs_error, 2)}</td><td>${fmt(r.abs_error_naif, 2)}</td></tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ---- Init -------------------------------------------------------------------------
renderSubtitle();
buildTabBar();
buildAssetPanels();
renderTestCase(DATA.assets[0].ticker);
</script>
"""


def main():
    p = argparse.ArgumentParser(description="Génère test_cases/dashboard.html")
    p.add_argument("--results-root", default=str(Path(__file__).parent / "results"))
    p.add_argument("--out", default=str(Path(__file__).parent / "dashboard.html"))
    args = p.parse_args()

    data = collect_results(Path(args.results_root))
    html = render_html(data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"[generate_dashboard] écrit -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
