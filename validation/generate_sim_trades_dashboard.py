"""
validation/generate_sim_trades_dashboard.py — Dashboard HTML des test cases TC1.1-TC1.5
==========================================================================================
Lit les KPIs de `validation/sim_trades.py` (`kpi_report`, `naive_always_long_report`,
`naive_always_short_report`) pour les 5 règles déjà codées (`bull_calm_d1`, `pi95_conf`,
`bear_calm_d1`, `bear_stress_d1`, `sideways_d1` — cf. BRIEF_bull_calm_d1.md §3bis et
BRIEF_sideways_d1.md) et génère une page HTML autonome, même charte visuelle que
`model_artifacts/generate_dashboard.py` (Run/dashboard.html) :
- un onglet par test case (TC1.1 Bull-Calm, TC1.2 Bull-Stress, TC1.3 Bear-Calm,
  TC1.4 Bear-Stress, TC1.5 Sideways) ;
- bascule OOS / Live (jamais mélangés, cf. brief) ;
- cartes KPI agrégées + tableau détaillé par actif × modèle, avec le counter (Σ et
  moyenne) et le ROI (sideways : justesse pure, pas de ROI) ;
- sélecteur d'actif (BTC-USD par défaut).

Exécution (depuis DeepEdgeBenchmark/) :
    python -m validation.generate_sim_trades_dashboard
    python -m validation.generate_sim_trades_dashboard --db-path validation/tracking.db --out validation/sim_trades_dashboard.html
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from validation import sim_trades as st

MODEL_ORDER = ["ARIMA-GARCH", "SARIMA", "Prophet", "LSTM", "Naive", "TSDiff"]
MODEL_COLORS_LIGHT = {
    "ARIMA-GARCH": "#2a78d6", "SARIMA": "#1baf7a", "Prophet": "#eda100",
    "LSTM": "#008300", "Naive": "#4a3aa7", "TSDiff": "#d64550",
}
MODEL_COLORS_DARK = {
    "ARIMA-GARCH": "#3987e5", "SARIMA": "#199e70", "Prophet": "#c98500",
    "LSTM": "#008300", "Naive": "#9085e9", "TSDiff": "#e5606b",
}

TEST_CASES = [
    {"id": "TC1.1", "rule_version": "bull_calm_d1", "label": "Bull-Calm", "kind": "directional",
     "naive_fn": "naive_always_long_report", "naive_label": "Naïf always-long",
     "description": "Long léger : PI_low ≤ P(D) < PI_mid (predicted > ref et ref ≥ PI_low)."},
    {"id": "TC1.2", "rule_version": "pi95_conf", "label": "Bull-Stress", "kind": "directional",
     "naive_fn": "naive_always_long_report", "naive_label": "Naïf always-long",
     "description": "Long forte conviction : P(D) < PI_low (toute la bande est au-dessus d'aujourd'hui)."},
    {"id": "TC1.3", "rule_version": "bear_calm_d1", "label": "Bear-Calm", "kind": "directional",
     "naive_fn": "naive_always_short_report", "naive_label": "Naïf always-short",
     "description": "Short léger : PI_mid < P(D) ≤ PI_high (predicted < ref et ref ≤ PI_high). Miroir de TC1.1."},
    {"id": "TC1.4", "rule_version": "bear_stress_d1", "label": "Bear-Stress", "kind": "directional",
     "naive_fn": "naive_always_short_report", "naive_label": "Naïf always-short",
     "description": "Short forte conviction : P(D) > PI_high (toute la bande est en-dessous d'aujourd'hui). Miroir de TC1.2."},
    {"id": "TC1.5", "rule_version": "sideways_d1", "label": "Sideways", "kind": "sideways",
     "naive_fn": None, "naive_label": None,
     "description": "Journée plate anticipée : P(D) dans la bande et |predicted-P(D)| ≤ k·W. Justesse pure, pas de ROI."},
]

DIRECTIONAL_COLS = [
    {"key": "n_signaux", "label": "N signaux", "digits": 0},
    {"key": "n_flat", "label": "N flat", "digits": 0},
    {"key": "n_open", "label": "N open", "digits": 0},
    {"key": "taux_signal", "label": "Taux signal", "pct": True},
    {"key": "precision_direction", "label": "Précision direction", "pct": True},
    {"key": "taux_realisation", "label": "Taux réalisation", "pct": True},
    {"key": "counter_sum", "label": "Counter Σ", "digits": 0},
    {"key": "counter_mean", "label": "Counter moy.", "digits": 3},
    {"key": "roi_sum", "label": "ROI Σ", "pct": True},
    {"key": "roi_compound", "label": "ROI composé", "pct": True},
    {"key": "roi_mean", "label": "ROI moyen", "pct": True},
    {"key": "roi_median", "label": "ROI médian", "pct": True},
    {"key": "roi_min", "label": "ROI min (pire trade)", "pct": True},
    {"key": "sharpe", "label": "Sharpe (indicatif)", "digits": 2},
    {"key": "pi_coverage_95", "label": "Couverture PI", "pct": True},
]

SIDEWAYS_COLS = [
    {"key": "n_signaux", "label": "N signaux", "digits": 0},
    {"key": "n_flat", "label": "N flat", "digits": 0},
    {"key": "n_open", "label": "N open", "digits": 0},
    {"key": "taux_signal", "label": "Taux signal", "pct": True},
    {"key": "taux_justesse", "label": "Taux justesse", "pct": True},
    {"key": "taux_immobile", "label": "Taux immobile", "pct": True},
    {"key": "taux_breakout", "label": "Taux breakout", "pct": True},
    {"key": "taux_breakout_haussier", "label": "Breakout haussier", "pct": True},
    {"key": "taux_breakout_baissier", "label": "Breakout baissier", "pct": True},
    {"key": "counter_sum", "label": "Counter Σ", "digits": 0},
    {"key": "counter_mean", "label": "Counter moy.", "digits": 3},
    {"key": "in_band_coverage", "label": "Couverture bande", "pct": True},
]


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _clean_row(row):
    return {k: _num(v) if not isinstance(v, dict) else v for k, v in row.items()}


def _clean(rows):
    return [_clean_row(row) for row in rows]


def collect_data(db_path: str) -> dict:
    st.init_db(db_path)
    payload = {}
    for tc in TEST_CASES:
        rv = tc["rule_version"]
        entry = {}
        for source in ("oos", "live"):
            breakdown = st.kpi_report(db_path=db_path, source=source, rule_version=rv,
                                      group_by=("asset", "model"))
            try:
                aggregate = st.kpi_report(db_path=db_path, source=source, rule_version=rv,
                                          group_by=())
            except Exception:
                aggregate = []
            naive = []
            if tc["naive_fn"]:
                naive_fn = getattr(st, tc["naive_fn"])
                naive = naive_fn(db_path=db_path, source=source, group_by=("asset",))
            entry[source] = {
                "breakdown": _clean(breakdown),
                "aggregate": _clean_row(aggregate[0]) if aggregate else None,
                "naive": _clean(naive),
            }
        payload[tc["id"]] = entry
    return payload


def render_html(data: dict, db_path: str) -> str:
    assets_seen = sorted({row["asset"] for tc in data.values() for src in tc.values()
                          for row in src["breakdown"]})
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z"),
        "db_path": db_path,
        "test_cases": [{"id": tc["id"], "label": tc["label"], "kind": tc["kind"],
                        "description": tc["description"], "naive_label": tc["naive_label"]}
                       for tc in TEST_CASES],
        "model_order": MODEL_ORDER,
        "model_colors_light": MODEL_COLORS_LIGHT,
        "model_colors_dark": MODEL_COLORS_DARK,
        "assets": assets_seen,
        "directional_cols": DIRECTIONAL_COLS,
        "sideways_cols": SIDEWAYS_COLS,
        "data": data,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


HTML_TEMPLATE = r"""<title>Test cases TC1.1-TC1.5 — Bull / Bear / Sideways à D+1</title>
<style>
:root {
  --surface-1:      #fcfcfb;
  --page-plane:     #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid-line:      #e1e0d9;
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
h2 { font-size: 15px; margin: 0 0 12px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border-ring); border-radius: 10px;
  box-shadow: var(--card-shadow); padding: 18px 20px; margin-bottom: 20px;
}
.controls-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
.tabbar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
.tab-btn {
  font: inherit; font-size: 13px; padding: 9px 18px; border-radius: 8px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-secondary);
  cursor: pointer;
}
.tab-btn:hover { color: var(--text-primary); }
.tab-btn.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.tc-panel { display: none; }
.tc-panel.active { display: block; }
.toggle-group { display: inline-flex; border: 1px solid var(--border-ring); border-radius: 8px; overflow: hidden; }
.toggle-group button {
  font: inherit; font-size: 13px; padding: 7px 16px; border: none; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
}
.toggle-group button.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.select-box {
  font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-primary);
}
.field-label { font-size: 13px; color: var(--text-secondary); display: inline-flex; align-items: center; gap: 8px; }
.tc-description { font-size: 12.5px; color: var(--text-secondary); margin: -8px 0 16px; max-width: 900px; }
.kpi-tiles { display: flex; gap: 14px; flex-wrap: wrap; }
.kpi-tile { flex: 1 1 150px; border: 1px solid var(--border-ring); border-radius: 8px; padding: 10px 14px; }
.kpi-tile .label { font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .03em; }
.kpi-tile .value { font-size: 22px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
.kpi-tile .value.pos { color: var(--pos-color); }
.kpi-tile .value.neg { color: var(--neg-color); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
thead th { text-align: left; padding: 7px 9px; color: var(--text-secondary); font-weight: 600;
  border-bottom: 1px solid var(--grid-line); white-space: nowrap; }
tbody td { padding: 5px 9px; border-bottom: 1px solid var(--grid-line); font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:hover { background: rgba(128,128,128,0.06); }
td.model-cell { display: flex; align-items: center; gap: 6px; }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; flex: none; }
.no-data { color: var(--text-muted); font-size: 13px; padding: 12px 0; }
.pos { color: var(--pos-color); }
.neg { color: var(--neg-color); }
</style>

<h1>Test cases TC1.1&ndash;TC1.5 &mdash; Bull / Bear / Sideways &agrave; D+1</h1>
<p class="subtitle" id="subtitle"></p>

<div class="tabbar" id="tc-tabbar"></div>
<div id="tc-panels"></div>

<script>
const DATA = __DATA_JSON__;
const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const MODEL_COLORS = isDark ? DATA.model_colors_dark : DATA.model_colors_light;

function fmt(v, digits) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (Number(v) * 100).toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' %';
}
function signClass(v) {
  if (v === null || v === undefined) return '';
  return Number(v) > 0 ? 'pos' : (Number(v) < 0 ? 'neg' : '');
}

function renderSubtitle() {
  document.getElementById('subtitle').textContent =
    `5 test cases × 2 sources (OOS / Live) — base ${DATA.db_path} — généré le ${DATA.generated_at}`;
}

const state = { tc: DATA.test_cases[0].id, source: 'oos', asset: 'BTC-USD' };

function buildTabBar() {
  const bar = document.getElementById('tc-tabbar');
  bar.innerHTML = '';
  DATA.test_cases.forEach((tc, i) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.textContent = `${tc.id} ${tc.label}`;
    btn.dataset.tc = tc.id;
    btn.addEventListener('click', () => switchTab(tc.id));
    bar.appendChild(btn);
  });
}

function switchTab(tcId) {
  state.tc = tcId;
  document.querySelectorAll('.tc-panel').forEach(p => p.classList.toggle('active', p.dataset.tc === tcId));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tc === tcId));
  renderPanel(tcId);
}

function panelSkeleton(tc) {
  return `
    <div class="card controls-row">
      <div class="toggle-group" id="source-${tc.id}">
        <button class="active" data-source="oos">OOS (backtest)</button>
        <button data-source="live">Live</button>
      </div>
      <label class="field-label">Actif
        <select class="select-box" id="asset-${tc.id}"></select>
      </label>
    </div>
    <div class="tc-description">${tc.description}</div>
    <div class="card">
      <h2>Agrégé (tous actifs × modèles)</h2>
      <div class="kpi-tiles" id="agg-${tc.id}"></div>
    </div>
    <div class="card">
      <h2>Détail par modèle — <span id="asset-label-${tc.id}"></span></h2>
      <div style="overflow-x:auto;" id="table-${tc.id}"></div>
    </div>
  `;
}

function buildPanels() {
  const root = document.getElementById('tc-panels');
  root.innerHTML = '';
  DATA.test_cases.forEach((tc, i) => {
    const panel = document.createElement('div');
    panel.className = 'tc-panel' + (i === 0 ? ' active' : '');
    panel.dataset.tc = tc.id;
    panel.innerHTML = panelSkeleton(tc);
    root.appendChild(panel);
  });
  DATA.test_cases.forEach(tc => wirePanel(tc));
}

function wirePanel(tc) {
  document.getElementById(`source-${tc.id}`).querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      state.source = btn.dataset.source;
      document.getElementById(`source-${tc.id}`).querySelectorAll('button')
        .forEach(b => b.classList.toggle('active', b === btn));
      renderPanel(tc.id);
    });
  });

  const assetSel = document.getElementById(`asset-${tc.id}`);
  const allOpt = document.createElement('option');
  allOpt.value = '__ALL__'; allOpt.textContent = 'Tous les actifs';
  assetSel.appendChild(allOpt);
  DATA.assets.forEach(a => {
    const opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    if (a === state.asset) opt.selected = true;
    assetSel.appendChild(opt);
  });
  if (!DATA.assets.includes(state.asset)) assetSel.value = '__ALL__';
  assetSel.addEventListener('change', () => {
    state.asset = assetSel.value;
    renderPanel(tc.id);
  });
}

function renderPanel(tcId) {
  const tc = DATA.test_cases.find(t => t.id === tcId);
  const bucket = DATA.data[tcId][state.source];
  renderAggregate(tc, bucket);
  renderTable(tc, bucket);
}

const AGG_TILES_DIRECTIONAL = [
  ['n_signaux', 'N signaux', 0, false],
  ['n_flat', 'N flat', 0, false],
  ['n_open', 'N open', 0, false],
  ['precision_direction', 'Précision direction', null, true],
  ['taux_realisation', 'Taux réalisation', null, true],
  ['counter_sum', 'Counter Σ', 0, false],
  ['counter_mean', 'Counter moy.', 3, false],
  ['roi_sum', 'ROI Σ', null, true],
  ['roi_compound', 'ROI composé', null, true],
  ['sharpe', 'Sharpe', 2, false],
];
const AGG_TILES_SIDEWAYS = [
  ['n_signaux', 'N signaux', 0, false],
  ['n_flat', 'N flat', 0, false],
  ['n_open', 'N open', 0, false],
  ['taux_justesse', 'Taux justesse', null, true],
  ['taux_immobile', 'Taux immobile', null, true],
  ['taux_breakout', 'Taux breakout', null, true],
  ['counter_sum', 'Counter Σ', 0, false],
  ['counter_mean', 'Counter moy.', 3, false],
  ['in_band_coverage', 'Couverture bande', null, true],
];

function renderAggregate(tc, bucket) {
  const el = document.getElementById(`agg-${tc.id}`);
  el.innerHTML = '';
  const agg = bucket.aggregate;
  if (!agg) { el.innerHTML = '<div class="no-data">Aucune donnée.</div>'; return; }
  const tiles = tc.kind === 'sideways' ? AGG_TILES_SIDEWAYS : AGG_TILES_DIRECTIONAL;
  tiles.forEach(([key, label, digits, pct]) => {
    const v = agg[key];
    const tile = document.createElement('div');
    tile.className = 'kpi-tile';
    const cls = (key.startsWith('counter') || key.startsWith('roi')) ? signClass(v) : '';
    tile.innerHTML = `<div class="label">${label}</div><div class="value ${cls}">${pct ? fmtPct(v) : fmt(v, digits)}</div>`;
    el.appendChild(tile);
  });
}

function renderTable(tc, bucket) {
  const el = document.getElementById(`table-${tc.id}`);
  document.getElementById(`asset-label-${tc.id}`).textContent =
    state.asset === '__ALL__' ? 'tous les actifs' : state.asset;
  let rows = bucket.breakdown;
  if (state.asset !== '__ALL__') rows = rows.filter(r => r.asset === state.asset);
  rows = rows.slice().sort((a, b) =>
    a.asset.localeCompare(b.asset) || DATA.model_order.indexOf(a.model) - DATA.model_order.indexOf(b.model));

  if (!rows.length) { el.innerHTML = '<div class="no-data">Aucune donnée pour cette sélection.</div>'; return; }

  const cols = tc.kind === 'sideways' ? DATA.sideways_cols : DATA.directional_cols;
  let html = '<table><thead><tr><th>Actif</th><th>Modèle</th>';
  cols.forEach(c => { html += `<th>${c.label}</th>`; });
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    const color = MODEL_COLORS[r.model] || '#888';
    html += `<tr><td>${r.asset}</td><td class="model-cell"><span class="swatch" style="background:${color}"></span>${r.model}</td>`;
    cols.forEach(c => {
      const v = r[c.key];
      const cls = (c.key.startsWith('counter') || c.key.startsWith('roi')) ? signClass(v) : '';
      html += `<td class="${cls}">${c.pct ? fmtPct(v) : fmt(v, c.digits)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  if (tc.naive_label && bucket.naive.length) {
    const naiveRows = state.asset === '__ALL__' ? bucket.naive : bucket.naive.filter(r => r.asset === state.asset);
    if (naiveRows.length) {
      html += `<h2 style="margin-top:18px;">${tc.naive_label} (benchmark, résolution appliquée à chaque jour, sans filtre de signal)</h2>`;
      html += '<table><thead><tr><th>Actif</th><th>N jours</th><th>ROI Σ</th><th>ROI composé</th><th>ROI moyen</th></tr></thead><tbody>';
      naiveRows.forEach(r => {
        html += `<tr><td>${r.asset}</td><td>${fmt(r.n_days, 0)}</td>`
          + `<td class="${signClass(r.roi_sum)}">${fmtPct(r.roi_sum)}</td>`
          + `<td class="${signClass(r.roi_compound)}">${fmtPct(r.roi_compound)}</td>`
          + `<td class="${signClass(r.roi_mean)}">${fmtPct(r.roi_mean)}</td></tr>`;
      });
      html += '</tbody></table>';
    }
  }
  el.innerHTML = html;
}

renderSubtitle();
buildTabBar();
buildPanels();
renderPanel(state.tc);
</script>
"""


def main():
    p = argparse.ArgumentParser(description="Génère validation/sim_trades_dashboard.html")
    p.add_argument("--db-path", default="validation/tracking.db")
    p.add_argument("--out", default=str(Path(__file__).parent / "sim_trades_dashboard.html"))
    args = p.parse_args()

    data = collect_data(args.db_path)
    html = render_html(data, args.db_path)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"[generate_sim_trades_dashboard] écrit -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
