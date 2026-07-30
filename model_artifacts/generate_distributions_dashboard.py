"""
model_artifacts/generate_distributions_dashboard.py — Gaussienne vs queues épaisses (D+1 / W+1)
==================================================================================================
Page HTML autonome, séparée du dashboard principal (model_artifacts/generate_dashboard.py),
qui compare deux hypothèses de loi prédictive pour la dernière prévision hors-échantillon
d'un modèle donné : une gaussienne (queues fines) et une loi de Student ν réglable (queues
épaisses), toutes deux reconstruites depuis le même (prédit, IC95%) déjà stocké dans les
artefacts — donc avec exactement la même prévision ponctuelle et la même largeur à 95%,
seule la forme de la queue diffère. RMSE (invariant par construction) et CRPS (sensible à
la forme) sont recalculés pour les deux, moyennés sur le backtest de validation walk-forward
(la dernière prévision elle-même n'a pas encore de valeur réalisée pour être notée).

Approximation assumée pour toute cette page (documentée dans l'UI, cf. info-dot) : la loi
de référence est reconstruite directement en ESPACE PRIX (mu=prédit, sigma tel que
mu ± z*sigma = IC95% stocké), y compris pour ARIMA-GARCH dont la loi native est log-normale
en rendement (cf. experiments/prob_kpi_common.py pour le traitement rigoureux en log-espace
utilisé ailleurs) — simplification volontaire pour que les deux moitiés de page restent
directement comparables, jamais un résultat de plus dans le dashboard principal.

CRPS gaussien : forme fermée déjà en place (honest_eval.metrics.crps_gaussian), réutilisée
telle quelle. CRPS Student-t : forme fermée de Gneiting & Raftery (2007) pour la loi de
Student standardisée (implémentée ici, vérifiée numériquement à la main -- converge vers le
CRPS gaussien quand ν -> l'infini, et recoupée contre un CRPS empirique Monte Carlo à ν=4,
cf. conversation de mise en œuvre) :

    crps(t_ν, z) = z*(2*F_ν(z) - 1) + 2*f_ν(z)*(ν + z²)/(ν - 1)
                   - (2*sqrt(ν)/(ν - 1)) * B(1/2, ν - 1/2) / B(1/2, ν/2)²

puis mise à l'échelle par `scale` (recouvré depuis l'IC95% stocké pour ce ν, comme sigma
pour la gaussienne). Calculé côté Python (scipy, exact) pour une grille entière de ν=2..30
et embarqué dans le payload -- le curseur ν du dashboard fait un simple accès table, aucune
fonction spéciale (gamma/bêta incomplète) n'est réimplémentée côté JS.

Horizons : D+1 (tous modèles ayant des artefacts Run/) et W+1 (TSDiff uniquement -- seul
modèle avec des données weekly en base, cf. WEEKLY_KPI_MODELS de generate_dashboard.py).

Exécution (depuis DeepEdgeBenchmark/) :
    python -m model_artifacts.generate_distributions_dashboard
    python -m model_artifacts.generate_distributions_dashboard --run-root Run --out Run/distributions_dashboard.html
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import beta as beta_fn
from scipy.stats import norm, t as student_t

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT))
from honest_eval import metrics as hm
from model_artifacts.generate_dashboard import (
    MODEL_COLORS_DARK, MODEL_COLORS_LIGHT, MODEL_ORDER,
    build_asset_catalog, collect_run_data,
)

DEFAULT_DB_PATH = "validation/tracking.db"
DEFAULT_OUT = REPO_ROOT / "Run" / "distributions_dashboard.html"

# Grille entière de degrés de liberté couverte par le curseur -- au-delà de ~30 la loi de
# Student est déjà quasi indiscernable d'une gaussienne (queue épaisse -> fine).
DF_GRID = list(range(2, 31))
DEFAULT_DF = 4  # choix standard en finance pour des rendements à queues épaisses.

Z95 = float(norm.ppf(0.975))  # 1.959963985 -- même constante que prob_kpi_common.Z95.

# Seul modèle avec des prévisions weekly en base (cf. generate_dashboard.WEEKLY_KPI_MODELS).
WEEKLY_MODELS = ["TSDiff"]


def _num(v):
    """float JSON-safe : NaN/inf -> None."""
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def crps_studentt_mean(pred, pi_lower, pi_upper, actual, df: int) -> float:
    """CRPS moyen (forme fermée) sous l'hypothèse Student-t(df), un `scale` par pas
    recouvré depuis l'IC95% stocké de CE pas (comme le sigma gaussien) -- même
    prévision ponctuelle et même largeur à 95% que la gaussienne, seule la queue change."""
    pred = np.asarray(pred, dtype=float)
    lo = np.asarray(pi_lower, dtype=float)
    hi = np.asarray(pi_upper, dtype=float)
    y = np.asarray(actual, dtype=float)
    t975 = float(student_t.ppf(0.975, df))
    scale = np.maximum((hi - lo) / (2.0 * t975), 1e-12)
    z = (y - pred) / scale
    Fz = student_t.cdf(z, df)
    fz = student_t.pdf(z, df)
    beta1 = beta_fn(0.5, df - 0.5)
    beta2 = beta_fn(0.5, df / 2.0)
    term = (z * (2 * Fz - 1) + 2 * fz * (df + z ** 2) / (df - 1)
            - (2 * np.sqrt(df) / (df - 1)) * beta1 / beta2 ** 2)
    return float(np.mean(scale * term))


def build_t_table(df_grid=DF_GRID) -> dict:
    """Table universelle (indépendante de l'actif/modèle) : pour chaque ν, la constante
    de densité au centre (c = pdf_t(0, ν)) et le quantile à 97.5% (pour recouvrer `scale`
    depuis une largeur d'IC95% donnée) -- tout calculé une fois via scipy (exact), le
    curseur JS ne fait qu'un accès table indexé par ν entier."""
    return {
        str(df): {"c": float(student_t.pdf(0, df)), "t975": float(student_t.ppf(0.975, df))}
        for df in df_grid
    }


def build_daily_payload(run_data: dict) -> dict:
    """Pour chaque (actif, modèle) D+1 : dernière prévision hors-échantillon (metrics.json
    "forecast", déjà collectée par collect_run_data) + RMSE/CRPS moyennés sur le backtest de
    validation walk-forward de ce même run (predictions.parquet du run le plus récent)."""
    latest = {}
    for r in run_data["records"]:
        if r["horizon"] != "D1" or r.get("forecast_predicted") is None:
            continue
        key = (r["asset"], r["model"])
        if key not in latest or r["run_date"] > latest[key]["run_date"]:
            latest[key] = r

    out = {}
    for (asset, model), rec in latest.items():
        points = (run_data["predictions"].get(rec["run_date"], {})
                  .get(asset, {}).get(model, {}).get("D1", []))
        rows = [p for p in points if None not in
                (p["predicted"], p["actual"], p["pi_lower"], p["pi_upper"])]
        if not rows:
            continue
        pred = np.array([p["predicted"] for p in rows], dtype=float)
        actual = np.array([p["actual"] for p in rows], dtype=float)
        lo = np.array([p["pi_lower"] for p in rows], dtype=float)
        hi = np.array([p["pi_upper"] for p in rows], dtype=float)
        sigma = np.maximum((hi - lo) / (2.0 * Z95), 1e-12)

        out.setdefault(asset, {})[model] = {
            "run_date": rec["run_date"],
            "n_val": len(rows),
            "rmse": _num(rec.get("RMSE")),
            "last_date": rec.get("forecast_last_date"),
            "last_price": _num(rec.get("forecast_last_price")),
            "predicted": _num(rec.get("forecast_predicted")),
            "pi_lower": _num(rec.get("forecast_pi_lower")),
            "pi_upper": _num(rec.get("forecast_pi_upper")),
            "crps_gaussian": _num(hm.crps_gaussian(pred, sigma, actual)),
            "crps_studentt": {str(df): _num(crps_studentt_mean(pred, lo, hi, actual, df))
                              for df in DF_GRID},
        }
    return out


def build_weekly_payload(db_path: str) -> dict:
    """Pour chaque actif, TSDiff W+1 uniquement (cf. docstring module) : dernière ligne en
    base (réalisée ou non -- "dernière prévision en date") + RMSE/CRPS moyennés sur les
    lignes déjà réalisées (y_true non nul)."""
    try:
        con = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(
                """
                SELECT asset, cutoff_date, target_date, last_close, y_pred, y_lower, y_upper, y_true
                FROM predictions
                WHERE frequence = 'weekly' AND horizon_type = 'weekly' AND horizon_unit = 'W+1'
                      AND model = 'TSDiff' AND source IN ('oos', 'live')
                """,
                con,
            )
        finally:
            con.close()
    except Exception as exc:
        print(f"[generate_distributions_dashboard] weekly indisponible ({exc})")
        return {}

    out = {}
    for asset, g in df.groupby("asset"):
        g = g.sort_values("cutoff_date")
        realized = g[g["y_true"].notna()]
        if realized.empty:
            continue
        pred = realized["y_pred"].to_numpy(dtype=float)
        lo = realized["y_lower"].to_numpy(dtype=float)
        hi = realized["y_upper"].to_numpy(dtype=float)
        actual = realized["y_true"].to_numpy(dtype=float)
        sigma = np.maximum((hi - lo) / (2.0 * Z95), 1e-12)
        last_row = g.iloc[-1]

        out[asset] = {
            "model": "TSDiff",
            "cutoff_date": last_row["cutoff_date"],
            "target_date": last_row["target_date"],
            "n_val": int(len(realized)),
            "rmse": _num(np.sqrt(np.mean((actual - pred) ** 2))),
            "last_date": last_row["cutoff_date"],
            "last_price": _num(last_row["last_close"]),
            "predicted": _num(last_row["y_pred"]),
            "pi_lower": _num(last_row["y_lower"]),
            "pi_upper": _num(last_row["y_upper"]),
            "crps_gaussian": _num(hm.crps_gaussian(pred, sigma, actual)),
            "crps_studentt": {str(d): _num(crps_studentt_mean(pred, lo, hi, actual, d))
                              for d in DF_GRID},
        }
    return out


def render_html(daily: dict, weekly: dict, records: list, run_root_label: str) -> str:
    relevant_tickers = set(daily) | set(weekly)
    asset_catalog = [a for a in build_asset_catalog(records) if a["ticker"] in relevant_tickers]
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %z"),
        "run_root": run_root_label,
        "assets": asset_catalog,
        "model_order": MODEL_ORDER,
        "model_colors_light": MODEL_COLORS_LIGHT,
        "model_colors_dark": MODEL_COLORS_DARK,
        "daily": daily,
        "weekly": weekly,
        "df_grid": DF_GRID,
        "default_df": DEFAULT_DF,
        "t_table": build_t_table(),
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


HTML_TEMPLATE = r"""<meta charset="utf-8">
<title>Gaussienne vs queues épaisses — Comparaison des distributions</title>
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
  --gaussian-color: #2a78d6;
  --heavy-color:    #d64550;
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
    --gaussian-color: #3987e5;
    --heavy-color:    #e5606b;
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
h2 { font-size: 15px; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; max-width: 76ch; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border-ring);
  border-radius: 10px;
  box-shadow: var(--card-shadow);
  padding: 18px 20px;
  margin-bottom: 20px;
}
.controls-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.tabbar { display: flex; gap: 6px; flex-wrap: wrap; }
.tab-btn {
  font: inherit; font-size: 13px; padding: 8px 16px; border-radius: 8px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-secondary);
  cursor: pointer;
}
.tab-btn:hover { color: var(--text-primary); }
.tab-btn.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.toggle-group { display: inline-flex; border: 1px solid var(--border-ring); border-radius: 8px; overflow: hidden; }
.toggle-group button {
  font: inherit; font-size: 13px; padding: 7px 16px; border: none; cursor: pointer;
  background: var(--surface-1); color: var(--text-secondary);
}
.toggle-group button.active { background: var(--text-primary); color: var(--surface-1); font-weight: 600; }
.toggle-group button:disabled { color: var(--text-muted); cursor: not-allowed; opacity: 0.5; }
.select-box {
  font: inherit; font-size: 13px; padding: 7px 12px; border-radius: 8px;
  border: 1px solid var(--border-ring); background: var(--surface-1); color: var(--text-primary);
}
.model-swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 6px; }
.info-line { font-size: 13px; color: var(--text-secondary); }
.info-line b { color: var(--text-primary); font-variant-numeric: tabular-nums; }
.info-dot {
  display: inline-flex; align-items: center; justify-content: center;
  width: 13px; height: 13px; border-radius: 50%;
  border: 1px solid var(--text-muted); color: var(--text-muted);
  font-size: 9.5px; line-height: 1; cursor: help; user-select: none; flex: none;
}
.info-dot:hover { border-color: var(--text-primary); color: var(--text-primary); }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
@media (max-width: 860px) { .compare-grid { grid-template-columns: 1fr; } }
.compare-card-title { font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.compare-card-sub { font-size: 12px; color: var(--text-secondary); margin: 2px 0 14px; }
.threshold-field { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary); }
.threshold-field input[type=range] { width: 160px; }
.threshold-field b { color: var(--text-primary); font-variant-numeric: tabular-nums; }
.curve-wrap { position: relative; height: 220px; }
.stat-tiles { display: flex; gap: 14px; margin-top: 14px; }
.stat-tile { flex: 1 1 0; border: 1px solid var(--border-ring); border-radius: 8px; padding: 10px 12px; }
.stat-tile .label { font-size: 12px; color: var(--text-secondary); display: flex; align-items: center; gap: 6px; }
.stat-tile .value { font-size: 22px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }
.note { font-size: 12px; color: var(--text-muted); margin-top: 18px; max-width: 90ch; line-height: 1.5; }
.no-data { color: var(--text-muted); font-size: 13px; padding: 24px 0; text-align: center; }
#tooltip {
  position: fixed; pointer-events: none; z-index: 50; display: none;
  background: var(--surface-1); border: 1px solid var(--border-ring); border-radius: 6px;
  padding: 6px 9px; font-size: 12px; box-shadow: var(--card-shadow);
}
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-muted); font-size: 10.5px; }
svg .axis-line { stroke: var(--baseline); stroke-width: 1; }
svg .ref-line { stroke-width: 1.5; stroke-dasharray: 3 3; }
</style>

<h1>Gaussienne vs queues épaisses</h1>
<p class="subtitle">
  Comparaison illustrative, pour la dernière prévision hors-échantillon d'un modèle, entre une
  loi gaussienne (queues fines) et une loi de Student ν réglable (queues épaisses) — les deux
  reconstruites à partir du même prédit et du même intervalle de confiance à 95% déjà stocké,
  donc directement comparables. RMSE et CRPS sont moyennés sur le backtest de validation
  walk-forward (la dernière prévision elle-même n'a pas encore de valeur réalisée à noter).
</p>

<div class="card controls-row">
  <div class="tabbar" id="asset-tabbar"></div>
  <div class="toggle-group" id="horizon-toggle">
    <button data-h="D1">D+1</button>
    <button data-h="W1">W+1</button>
  </div>
  <label class="info-line">Modèle
    <select class="select-box" id="model-select" style="margin-left:8px;"></select>
  </label>
</div>

<div class="card">
  <p class="info-line" id="info-line"></p>
</div>

<div id="content"></div>

<div id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;
const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const MODEL_COLORS = isDark ? DATA.model_colors_dark : DATA.model_colors_light;
const GAUSSIAN_COLOR = isDark ? '#3987e5' : '#2a78d6';
const HEAVY_COLOR = isDark ? '#e5606b' : '#d64550';
const GRID_COLOR = isDark ? '#2c2c2a' : '#e1e0d9';
const AXIS_TEXT_COLOR = isDark ? '#c3c2b7' : '#52514e';

const state = { asset: null, horizon: 'D1', model: null, df: DATA.default_df };

function fmt(v, digits) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function priceDigits(v) {
  return Math.abs(v) >= 1000 ? 0 : (Math.abs(v) >= 10 ? 2 : 4);
}
function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16), g = parseInt(h.substring(2, 4), 16), b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const tooltipEl = document.getElementById('tooltip');
function hideTooltip() { tooltipEl.style.display = 'none'; }

const KPI_DEFINITIONS = {
  rmse: "RMSE — identique des deux côtés par construction (ne dépend que de la prévision ponctuelle, jamais de la forme de la loi supposée). Calculé sur le backtest de validation walk-forward, pas sur la dernière prévision (pas encore réalisée).",
  crpsgauss: "CRPS sous hypothèse gaussienne — score probabiliste complet (précision + incertitude), forme fermée, moyenné sur le backtest de validation. Même prédit et même IC95% que la loi de Student en face ; seule la forme de la queue diffère.",
  crpsheavy: "CRPS sous hypothèse Student-t(ν) — même formule que le CRPS gaussien mais avec une loi à queues plus épaisses, même IC95% par construction. Change avec le curseur ν : plus ν est petit, plus les extrêmes pèsent lourd dans le score.",
  dfslider: "Degrés de liberté (ν) de la loi de Student — plus ν est petit, plus les queues sont épaisses (plus de probabilité affectée aux mouvements extrêmes au-delà de l'IC95%, qui reste identique à la gaussienne par construction). ν→∞ : converge vers la gaussienne.",
  approx: "Approximation assumée pour cette page : la loi de référence est reconstruite directement en espace prix (μ=prédit, μ±1,96σ=IC95% stocké) pour tous les modèles, y compris ARIMA-GARCH dont la loi native est log-normale en rendement — simplification volontaire pour que les deux moitiés restent directement comparables (cf. experiments/prob_kpi_common.py pour le traitement rigoureux en log-espace utilisé dans le dashboard principal).",
};
function infoDot(defKey) { return `<span class="info-dot" data-def="${defKey}">i</span>`; }
document.addEventListener('mouseover', (evt) => {
  const dot = evt.target.closest('.info-dot');
  if (!dot) return;
  const text = KPI_DEFINITIONS[dot.dataset.def];
  if (!text) return;
  tooltipEl.innerHTML = `<div style="max-width:240px;">${text}</div>`;
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = (evt.clientX + 14) + 'px';
  tooltipEl.style.top = (evt.clientY + 14) + 'px';
});
document.addEventListener('mousemove', (evt) => {
  if (!evt.target.closest('.info-dot')) return;
  tooltipEl.style.left = (evt.clientX + 14) + 'px';
  tooltipEl.style.top = (evt.clientY + 14) + 'px';
});
document.addEventListener('mouseout', (evt) => { if (evt.target.closest('.info-dot')) hideTooltip(); });

// ---- Data access ------------------------------------------------------------
function weeklyAvailable(asset) { return !!DATA.weekly[asset]; }
function availableModels(asset, horizon) {
  if (horizon === 'W1') return weeklyAvailable(asset) ? ['TSDiff'] : [];
  const models = (DATA.daily[asset] || {});
  return DATA.model_order.filter(m => models[m]);
}
function currentCombo() {
  if (!state.asset || !state.model) return null;
  if (state.horizon === 'W1') return DATA.weekly[state.asset] || null;
  return (DATA.daily[state.asset] || {})[state.model] || null;
}

// ---- PDF math ----------------------------------------------------------------
const Z95 = 1.9599639845400545;
function gaussianPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
function studentScale(piLower, piUpper, df) {
  const t975 = DATA.t_table[String(df)].t975;
  return Math.max((piUpper - piLower) / (2 * t975), 1e-12);
}
function studentPdf(x, mu, scale, df) {
  const c = DATA.t_table[String(df)].c;
  const z = (x - mu) / scale;
  return (c / scale) * Math.pow(1 + (z * z) / df, -(df + 1) / 2);
}

// ---- Rendering -----------------------------------------------------------------
function buildTabBar() {
  const bar = document.getElementById('asset-tabbar');
  bar.innerHTML = '';
  DATA.assets.forEach((a) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (a.ticker === state.asset ? ' active' : '');
    btn.textContent = a.label;
    btn.addEventListener('click', () => { state.asset = a.ticker; onAssetOrHorizonChange(); });
    bar.appendChild(btn);
  });
}

function onAssetOrHorizonChange() {
  if (state.horizon === 'W1' && !weeklyAvailable(state.asset)) state.horizon = 'D1';
  const models = availableModels(state.asset, state.horizon);
  if (!models.includes(state.model)) state.model = models[0] || null;
  renderControls();
  renderAll();
}

function renderControls() {
  document.querySelectorAll('#asset-tabbar .tab-btn').forEach((btn, i) => {
    btn.classList.toggle('active', DATA.assets[i].ticker === state.asset);
  });
  document.querySelectorAll('#horizon-toggle button').forEach(btn => {
    const h = btn.dataset.h;
    btn.classList.toggle('active', h === state.horizon);
    btn.disabled = h === 'W1' && !weeklyAvailable(state.asset);
  });
  const sel = document.getElementById('model-select');
  sel.innerHTML = '';
  availableModels(state.asset, state.horizon).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    if (m === state.model) opt.selected = true;
    sel.appendChild(opt);
  });
}

function renderInfoLine() {
  const el = document.getElementById('info-line');
  const combo = currentCombo();
  if (!combo) { el.innerHTML = 'Aucune donnée pour cette sélection.'; return; }
  const digits = priceDigits(combo.last_price ?? combo.predicted ?? 1);
  const dateLabel = state.horizon === 'W1'
    ? `cutoff ${combo.cutoff_date} → cible ${combo.target_date}`
    : `run du ${combo.run_date} → prévision J+1 après ${combo.last_date}`;
  el.innerHTML = `<b>${state.model}</b> — ${state.asset} — ${dateLabel}. `
    + `Dernier prix connu : <b>${fmt(combo.last_price, digits)}</b>. `
    + `Prévision : <b>${fmt(combo.predicted, digits)}</b> `
    + `[IC95% ${fmt(combo.pi_lower, digits)} – ${fmt(combo.pi_upper, digits)}]. `
    + `n = ${combo.n_val} pas de backtest.`;
}

function drawCurve(svg, xs, ysG, ysH, yMax, xDomain, refX, piLo, piHi, color, isGaussianSide) {
  const W = svg.clientWidth || 420, H = 220;
  const padL = 8, padR = 8, padT = 10, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xScale = (x) => padL + (x - xDomain[0]) / (xDomain[1] - xDomain[0]) * plotW;
  const yScale = (y) => padT + plotH - (y / yMax) * plotH;
  const ys = isGaussianSide ? ysG : ysH;

  let path = `M ${xScale(xs[0])} ${yScale(ys[0])}`;
  for (let i = 1; i < xs.length; i++) path += ` L ${xScale(xs[i])} ${yScale(ys[i])}`;
  let area = path + ` L ${xScale(xs[xs.length - 1])} ${yScale(0)} L ${xScale(xs[0])} ${yScale(0)} Z`;

  const ticks = [piLo, refX, piHi];
  const tickLabels = ticks.map(t => fmt(t, priceDigits(t)));

  svg.innerHTML = `
    <line class="axis-line" x1="${padL}" y1="${yScale(0)}" x2="${W - padR}" y2="${yScale(0)}"></line>
    <path d="${area}" fill="${hexToRgba(color, isDark ? 0.22 : 0.14)}" stroke="none"></path>
    <path d="${path}" fill="none" stroke="${color}" stroke-width="2"></path>
    <line class="ref-line" x1="${xScale(refX)}" y1="${padT}" x2="${xScale(refX)}" y2="${yScale(0)}" stroke="${color}"></line>
    ${ticks.map((t, i) => `<text x="${xScale(t)}" y="${H - 4}" text-anchor="middle">${tickLabels[i]}</text>`).join('')}
    <rect id="hover-rect" x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent"></rect>
    <line id="crosshair" x1="0" y1="${padT}" x2="0" y2="${yScale(0)}" stroke="${color}" stroke-width="1" stroke-dasharray="2 2" style="display:none;"></line>
  `;

  const hoverRect = svg.querySelector('#hover-rect');
  const crosshair = svg.querySelector('#crosshair');
  hoverRect.addEventListener('mousemove', (evt) => {
    const rect = svg.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const x = xDomain[0] + (mx - padL) / plotW * (xDomain[1] - xDomain[0]);
    const idx = Math.max(0, Math.min(xs.length - 1, Math.round((x - xs[0]) / (xs[1] - xs[0]))));
    crosshair.setAttribute('x1', xScale(xs[idx])); crosshair.setAttribute('x2', xScale(xs[idx]));
    crosshair.style.display = 'block';
    tooltipEl.innerHTML = `<div>${fmt(xs[idx], priceDigits(xs[idx]))} &nbsp;→&nbsp; densité ${fmt(ys[idx], 5)}</div>`;
    tooltipEl.style.display = 'block';
    tooltipEl.style.left = (evt.clientX + 14) + 'px';
    tooltipEl.style.top = (evt.clientY + 14) + 'px';
  });
  hoverRect.addEventListener('mouseleave', () => { crosshair.style.display = 'none'; hideTooltip(); });
}

function contentSkeleton() {
  return `
    <div class="card">
      <div class="compare-grid">
        <div>
          <div class="compare-card-title" style="color:${GAUSSIAN_COLOR};">
            <span>Loi gaussienne</span>${infoDot('approx')}
          </div>
          <div class="compare-card-sub">Queues fines — même prédit, même IC95% que la loi de Student.</div>
          <div class="curve-wrap"><svg id="svg-gaussian" width="100%" height="220"></svg></div>
          <div class="stat-tiles">
            <div class="stat-tile"><div class="label">RMSE ${infoDot('rmse')}</div><div class="value" id="rmse-gaussian"></div></div>
            <div class="stat-tile"><div class="label">CRPS ${infoDot('crpsgauss')}</div><div class="value" id="crps-gaussian"></div></div>
          </div>
        </div>
        <div>
          <div class="compare-card-title" style="color:${HEAVY_COLOR};">
            <span>Loi à queues épaisses (Student-t)</span>
          </div>
          <div class="compare-card-sub">
            <label class="threshold-field">ν ${infoDot('dfslider')}
              <input type="range" id="df-slider" min="${DATA.df_grid[0]}" max="${DATA.df_grid[DATA.df_grid.length - 1]}" step="1" value="${state.df}">
              <span><b id="df-value">${state.df}</b></span>
            </label>
          </div>
          <div class="curve-wrap"><svg id="svg-heavy" width="100%" height="220"></svg></div>
          <div class="stat-tiles">
            <div class="stat-tile"><div class="label">RMSE ${infoDot('rmse')}</div><div class="value" id="rmse-heavy"></div></div>
            <div class="stat-tile"><div class="label">CRPS ${infoDot('crpsheavy')}</div><div class="value" id="crps-heavy"></div></div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderComparison() {
  const root = document.getElementById('content');
  const combo = currentCombo();
  if (!combo) { root.innerHTML = '<div class="card"><div class="no-data">Aucune donnée pour cette sélection.</div></div>'; return; }
  if (!root.querySelector('#svg-gaussian')) root.innerHTML = contentSkeleton();

  const digits = priceDigits(combo.last_price ?? combo.predicted ?? 1);
  const mu = combo.predicted, piLo = combo.pi_lower, piHi = combo.pi_upper;
  const sigma = Math.max((piHi - piLo) / (2 * Z95), 1e-12);
  const scale = studentScale(piLo, piHi, state.df);
  const pad = 0.4 * (piHi - piLo);
  const xDomain = [piLo - pad, piHi + pad];
  const N = 240;
  const xs = Array.from({ length: N }, (_, i) => xDomain[0] + i * (xDomain[1] - xDomain[0]) / (N - 1));
  const ysG = xs.map(x => gaussianPdf(x, mu, sigma));
  const ysH = xs.map(x => studentPdf(x, mu, scale, state.df));
  const yMax = Math.max(...ysG, ...ysH) * 1.08;

  drawCurve(document.getElementById('svg-gaussian'), xs, ysG, ysH, yMax, xDomain, mu, piLo, piHi, GAUSSIAN_COLOR, true);
  drawCurve(document.getElementById('svg-heavy'), xs, ysG, ysH, yMax, xDomain, mu, piLo, piHi, HEAVY_COLOR, false);

  document.getElementById('rmse-gaussian').textContent = fmt(combo.rmse, digits);
  document.getElementById('rmse-heavy').textContent = fmt(combo.rmse, digits);
  document.getElementById('crps-gaussian').textContent = fmt(combo.crps_gaussian, digits);
  document.getElementById('crps-heavy').textContent = fmt(combo.crps_studentt[String(state.df)], digits);
  document.getElementById('df-value').textContent = state.df;

  document.getElementById('df-slider').oninput = (evt) => {
    state.df = parseInt(evt.target.value, 10);
    renderComparison();
  };
}

function renderAll() {
  renderInfoLine();
  renderComparison();
}

function boot() {
  state.asset = DATA.assets[0] ? DATA.assets[0].ticker : null;
  state.model = availableModels(state.asset, state.horizon)[0] || null;
  buildTabBar();
  renderControls();
  document.querySelectorAll('#horizon-toggle button').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.disabled) return;
      state.horizon = btn.dataset.h;
      onAssetOrHorizonChange();
    });
  });
  document.getElementById('model-select').addEventListener('change', (evt) => {
    state.model = evt.target.value;
    renderAll();
  });
  renderAll();

  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = `Généré le ${DATA.generated_at} depuis ${DATA.run_root} (D+1) et validation/tracking.db (W+1, TSDiff uniquement).`;
  document.body.appendChild(note);
}
boot();
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default=str(REPO_ROOT / "Run"))
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    run_root = Path(args.run_root)
    run_data = collect_run_data(run_root)
    if not run_data["records"]:
        print(f"Aucun metrics.json trouvé sous {run_root}")

    daily = build_daily_payload(run_data)
    weekly = build_weekly_payload(args.db_path)
    html = render_html(daily, weekly, run_data["records"], str(run_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n_daily = sum(len(v) for v in daily.values())
    print(f"[generate_distributions_dashboard] écrit -> {out_path} "
          f"({n_daily} combinaisons D+1, {len(weekly)} actif(s) W+1)")


if __name__ == "__main__":
    main()
