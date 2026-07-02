"""
dashboard_builder.py — Dashboard multi-actifs DEITA (BTC/ETH/SPY/ZN=F)

Orchestre RegimeAgent (RegimeHMM + RegimeBOCPD) sur les 4 actifs de assets.py,
calcule les analyses de regime_analytics.py, et génère un dashboard HTML unique
à 5 onglets (4 actifs + comparaison). Aucune modification du moteur de régime :
RegimeHMM/RegimeBOCPD/RegimeAgent sont appelés tels quels, 4 fois.

Exécution :
    python -m calibration.regime.dashboard_builder
"""

import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from calibration.regime.assets import (
    ASSETS,
    DATA_START,
    DATA_END,
    TRAIN_END,
    _EVENT_COLORS,
    _REGIME_BG,
    _REGIME_HEX,
    _REGIME_LABELS,
    events_for_ticker,
)
from calibration.regime.regime_agent import RegimeAgent
from calibration.regime import regime_analytics as ra


# ── Utilitaires de sérialisation ────────────────────────────────────────────────

def _num(v):
    """float JSON-safe : NaN/inf -> None."""
    f = float(v)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)


# ── 1. Pipeline de calcul ────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    Pour chaque actif de assets.ASSETS :
      1. Télécharge les prix via yfinance (DATA_START -> DATA_END), auto_adjust=True.
      2. Aplatit les colonnes si MultiIndex.
      3. Instancie un RegimeAgent, .fit(prices, train_end=TRAIN_END).
      4. .predict(prices) pour l'état courant (dernier RegimeState).
      5. .predict_history(prices) pour le DataFrame complet.
    Retourne { ticker: {"prices": df_ohlcv, "history": df_history, "state": RegimeState} }.
    """
    data_end = DATA_END or datetime.today().strftime("%Y-%m-%d")
    results = {}

    for asset in ASSETS:
        ticker = asset["ticker"]
        print(f"[dashboard_builder] {ticker} : telechargement des donnees ({DATA_START} -> {data_end})...")
        prices = yf.download(ticker, start=DATA_START, end=data_end, auto_adjust=True, progress=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        agent = RegimeAgent()
        agent.fit(prices, train_end=TRAIN_END)
        state = agent.predict(prices)
        history = agent.predict_history(prices)

        print(
            f"[dashboard_builder] {ticker} : {len(prices)} jours "
            f"({prices.index[0].date()} -> {prices.index[-1].date()}) | "
            f"regime dominant = {state.dominant_regime()} | stress_score = {state.stress_score:.3f}"
        )

        results[ticker] = {"prices": prices, "history": history, "state": state}

    return results


# ── 2. Analytics multi-actifs ────────────────────────────────────────────────────

def compute_all_analytics(results: dict) -> dict:
    """
    Appelle regime_analytics.* sur les résultats de run_pipeline() :
      - segments + width stats par actif et agrégés
      - vol_spike_hit_rate, par actif (statistique descriptive simple, cf. §note dans le HTML —
        pas de test de significativité, échantillon d'événements limité par actif)
      - rolling_cross_correlation + stress_conditioned_correlation, inter-actifs
    Retourne un dict structuré prêt à sérialiser en JSON pour le template HTML.
    """
    per_asset = {}
    all_segments = []
    returns_by_asset = {}
    stress_masks = {}
    calm_masks = {}

    for asset in ASSETS:
        ticker = asset["ticker"]
        short = asset["short"]
        history = results[ticker]["history"]
        prices = results[ticker]["prices"]

        segments = ra.segment_regimes(history)
        width_stats = ra.regime_width_stats(segments)
        hit_rate = ra.vol_spike_hit_rate(history, lookback=3, quantile=0.75)
        # nombre de transitions = nombre de segments - 1 (le 1er segment n'est pas un "changement",
        # cohérent avec regime_change.iloc[0] = False dans vol_spike_hit_rate).
        n_regime_changes = max(0, len(segments) - 1)

        per_asset[ticker] = {
            "segments": segments,
            "width_stats": width_stats,
            "hit_rate": hit_rate,
            "n_regime_changes": n_regime_changes,
        }

        tagged = segments.copy()
        tagged["asset"] = short
        all_segments.append(tagged)

        returns = prices["Close"].pct_change().dropna()
        returns_by_asset[short] = returns
        stress_masks[short] = (history["p_stress"] > 0.5).reindex(returns.index).fillna(False)
        calm_masks[short] = (history["regime"] == "calm").reindex(returns.index).fillna(False)

    combined_segments = pd.concat(all_segments, ignore_index=True)
    cross_correlation = ra.rolling_cross_correlation(returns_by_asset, window=63)
    stress_conditioned = ra.stress_conditioned_correlation(returns_by_asset, stress_masks, calm_masks)

    return {
        "per_asset": per_asset,
        "comparison": {
            "combined_segments": combined_segments,
            "cross_correlation": cross_correlation,
            "stress_conditioned": stress_conditioned,
        },
    }


# ── 3. Construction des payloads JSON par onglet ─────────────────────────────────

def _asset_tab_payload(asset: dict, prices: pd.DataFrame, history: pd.DataFrame, analytics_asset: dict) -> dict:
    """Construit le payload JSON-safe pour l'onglet d'un actif (prix/vol/BOCPD/composition/événements)."""
    ticker = asset["ticker"]
    close = prices["Close"].reindex(history.index).ffill()
    dates_str = [str(d.date()) for d in history.index]
    regimes = history["regime"].tolist()

    first_date, last_date = history.index[0], history.index[-1]

    # ── Segments de régime "Jour" -> rectangles de fond (paper y0-1, chart-agnostiques) ──
    regime_shapes = []
    i = 0
    while i < len(regimes):
        j, r = i, regimes[i]
        while j < len(regimes) and regimes[j] == r:
            j += 1
        regime_shapes.append({"x0": dates_str[i], "x1": dates_str[j - 1], "regime": r})
        i = j

    # ── Événements : lignes (toutes) + annotations texte (prix uniquement) ────────
    events = events_for_ticker(ticker)
    sorted_events = [
        (d, lbl, cat) for d, (lbl, cat) in sorted(events.items())
        if first_date <= pd.Timestamp(d) <= last_date
    ]
    event_lines = [
        {"x": d, "cat": cat, "color": _EVENT_COLORS.get(cat, "#888")}
        for d, _, cat in sorted_events
    ]
    event_annotations = [
        {
            "x": d, "y": 0.96 if k % 2 == 0 else 0.84, "text": f"<b>{lbl}</b>",
            "cat": cat, "color": _EVENT_COLORS.get(cat, "#888"),
        }
        for k, (d, lbl, cat) in enumerate(sorted_events)
    ]
    events_table = [
        {"date": d, "label": lbl, "cat": cat, "color": _EVENT_COLORS.get(cat, "#888")}
        for d, lbl, cat in sorted_events
    ]

    return {
        "ticker": ticker,
        "label": asset["label"],
        "color": asset["color"],
        "dates": dates_str,
        "close": [_num(v) if not np.isnan(v) else None for v in close],
        "sigma": [_num(v) if not np.isnan(v) else None for v in history["sigma_t"]],
        "vol_of_vol": [_num(v) if not np.isnan(v) else None for v in history["vol_of_vol"]],
        "cp": [_num(v) for v in history["changepoint_prob"]],
        "regimes": regimes,
        "regime_shapes": regime_shapes,
        "event_lines": event_lines,
        "event_annotations": event_annotations,
        "events_table": events_table,
        "first_date": str(first_date.date()),
        "last_date": str(last_date.date()),
    }


def _comparison_payload(results: dict, analytics: dict) -> dict:
    combined = analytics["comparison"]["combined_segments"]
    cross_corr = analytics["comparison"]["cross_correlation"]
    stress_cond = analytics["comparison"]["stress_conditioned"]

    # ── Box plot largeur des régimes (4 actifs x 3 régimes) ────────────────────────
    box_traces = []
    for asset in ASSETS:
        sub = combined[combined["asset"] == asset["short"]]
        box_traces.append({
            "name": asset["short"],
            "color": asset["color"],
            "x": sub["regime"].tolist(),
            "y": [int(v) for v in sub["n_days_calendar"]],
        })

    # ── Hit-rates vol -> changement de régime (statistique descriptive, cf. note HTML) ──────
    hit_rates = {}
    for asset in ASSETS:
        ticker = asset["ticker"]
        hit_rates[ticker] = {
            "label": asset["label"],
            "color": asset["color"],
            "hit_rate": _num(analytics["per_asset"][ticker]["hit_rate"]),
            "n_regime_changes": analytics["per_asset"][ticker]["n_regime_changes"],
        }

    # ── Corrélation glissante inter-actifs + bandes de stress marché (union des 4 actifs) ──
    cc_dates = [str(d.date()) for d in cross_corr.index]
    cc_series = {
        col: [_num(v) if not (isinstance(v, float) and np.isnan(v)) else None for v in cross_corr[col]]
        for col in cross_corr.columns
    }

    market_stress = stress_cond["stress_mask"]
    stress_bands = [
        {"x0": str(s["start"].date()), "x1": str(s["end"].date())}
        for s in ra.segment_boolean_mask(market_stress)
    ]

    # ── Tableau récapitulatif corrélation stress vs calme (6 paires) ───────────────
    # + test de significativité (transformation de Fisher) pour distinguer un vrai effet
    # du bruit d'échantillonnage — cf. BRIEF_dashboard_v4_corrections.md §1.
    r_crit_stress = ra.fisher_r_critical(stress_cond["n_stress"])
    r_crit_calm = ra.fisher_r_critical(stress_cond["n_calm"])

    pairs_table = []
    shorts = [a["short"] for a in ASSETS]
    for a, b in itertools.combinations(shorts, 2):
        s_val = stress_cond["stress"].loc[a, b] if a in stress_cond["stress"].index else float("nan")
        c_val = stress_cond["calm"].loc[a, b] if a in stress_cond["calm"].index else float("nan")
        s_ok = not (isinstance(s_val, float) and np.isnan(s_val))
        c_ok = not (isinstance(c_val, float) and np.isnan(c_val))
        pairs_table.append({
            "pair": f"{a}-{b}",
            "stress": _num(s_val) if s_ok else None,
            "calm": _num(c_val) if c_ok else None,
            "stress_sig": bool(r_crit_stress is not None and s_ok and abs(s_val) > r_crit_stress),
            "calm_sig": bool(r_crit_calm is not None and c_ok and abs(c_val) > r_crit_calm),
        })

    return {
        "box_traces": box_traces,
        "hit_rates": hit_rates,
        "cross_correlation": {"dates": cc_dates, "series": cc_series},
        "stress_bands": stress_bands,
        "pairs_table": pairs_table,
        "corr_significance": {
            "n_stress": stress_cond["n_stress"],
            "n_calm": stress_cond["n_calm"],
            "r_crit_stress": _num(r_crit_stress) if r_crit_stress is not None else None,
            "r_crit_calm": _num(r_crit_calm) if r_crit_calm is not None else None,
        },
    }


# ── 4. Génération HTML ────────────────────────────────────────────────────────────

def build_multi_asset_html(results: dict, analytics: dict) -> str:
    tabs_payload = {}
    for asset in ASSETS:
        ticker = asset["ticker"]
        tabs_payload[ticker] = _asset_tab_payload(
            asset, results[ticker]["prices"], results[ticker]["history"], analytics["per_asset"][ticker]
        )

    comparison_payload = _comparison_payload(results, analytics)

    j_tabs = json.dumps(tabs_payload)
    j_comparison = json.dumps(comparison_payload)
    j_assets = json.dumps(ASSETS)
    j_regime_bg = json.dumps(_REGIME_BG)
    j_regime_hex = json.dumps(_REGIME_HEX)
    j_regime_labels = json.dumps(_REGIME_LABELS)
    j_event_colors = json.dumps(_EVENT_COLORS)

    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M")
    tab_ids = [a["ticker"] for a in ASSETS] + ["COMPARISON"]
    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if a is ASSETS[0] else ""}" data-tab="{a["ticker"]}">{a["short"]}</button>'
        for a in ASSETS
    ) + '<button class="tab-btn" data-tab="COMPARISON">Comparaison</button>'

    asset_panels = "".join(
        _asset_panel_html(a, tabs_payload[a["ticker"]]["first_date"], tabs_payload[a["ticker"]]["last_date"])
        for a in ASSETS
    )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEITA &#8212; Dashboard Multi-Actifs</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f0f1a;color:#ecf0f1;font-family:'Segoe UI',sans-serif;padding:0 22px 22px}}
h1{{text-align:center;font-size:1.35rem;letter-spacing:1px;margin:14px 0 3px}}
.sub{{text-align:center;color:#7f8c8d;font-size:.82rem;margin-bottom:14px}}
.tabbar{{position:sticky;top:0;background:#0f0f1a;display:flex;gap:6px;justify-content:center;
  padding:10px 0;z-index:50;border-bottom:1px solid #1c2a3a;flex-wrap:wrap}}
.tab-btn{{background:#16213e;color:#95a5a6;border:1px solid #1c2a3a;border-radius:6px;
  padding:7px 16px;font-size:.82rem;cursor:pointer;transition:.15s}}
.tab-btn:hover{{color:#ecf0f1}}
.tab-btn.active{{background:#2980b9;color:#fff;border-color:#2980b9}}
.tab-panel{{display:none;padding-top:14px}}
.tab-panel.active{{display:block}}
.card{{background:#16213e;border-radius:8px;padding:12px 14px;margin-bottom:14px}}
.card-label{{font-size:.72rem;text-transform:uppercase;letter-spacing:1.2px;color:#566573;margin-bottom:8px}}
.chart-note{{color:#7f8c8d;font-size:.72rem;margin-bottom:8px}}
.sig-star{{color:#f39c12;font-weight:700}}
.legend{{display:flex;gap:16px;justify-content:center;margin-bottom:14px;flex-wrap:wrap;font-size:.8rem}}
.li{{display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}}
.li input{{accent-color:#2980b9}}
.dot{{width:11px;height:11px;border-radius:2px;flex-shrink:0}}
.sep{{width:1px;height:18px;background:#2c3e50;margin:0 4px}}
.row2{{display:grid;grid-template-columns:280px 1fr;gap:14px;margin-bottom:14px}}
.grid2x2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:.76rem}}
thead th{{color:#7f8c8d;text-align:left;padding:4px 10px;border-bottom:1px solid #1c2a3a;white-space:nowrap}}
tbody td{{padding:3px 10px;border-bottom:1px solid #151d2b}}
.tag{{padding:1px 7px;border-radius:3px;font-size:.7rem;color:#fff;display:inline-block}}
.scale-sel{{display:flex;gap:6px;align-items:center;justify-content:flex-end;margin-bottom:6px}}
.scale-btn{{background:#0f0f1a;color:#95a5a6;border:1px solid #1c2a3a;border-radius:4px;
  padding:3px 10px;font-size:.72rem;cursor:pointer}}
.scale-btn.active{{background:#2980b9;color:#fff}}
.scale-label{{font-size:.72rem;color:#7f8c8d;margin-right:2px}}
.date-pick{{background:#0f0f1a;color:#ecf0f1;border:1px solid #1c2a3a;border-radius:4px;
  padding:2px 6px;font-size:.72rem}}
.date-pick:disabled{{opacity:.4}}
.hitrate{{font-size:1.5rem;font-weight:600;color:#f39c12;text-align:center}}
.hitrate-label{{font-size:.72rem;color:#7f8c8d;text-align:center;margin-bottom:6px}}
.declutter-msg{{display:none;text-align:center;font-size:.72rem;color:#566573;padding:2px}}
details summary{{cursor:pointer;color:#95a5a6;font-size:.78rem;padding:4px 0}}
footer{{text-align:center;color:#3d5166;font-size:.72rem;margin-top:14px}}
</style>
</head>
<body>
<h1>DEITA &#8212; Dashboard Multi-Actifs</h1>
<p class="sub">Bitcoin &middot; Ethereum &middot; S&amp;P 500 (SPY) &middot; US Treasury 10Y Note Futures (ZN=F) &nbsp;&middot;&nbsp; HMM 2 &#233;tats + seuil ADX 25 + BOCPD</p>

<div class="tabbar">{tab_buttons}</div>

{asset_panels}

<div class="tab-panel" data-tab="COMPARISON">
  <div class="card">
    <div class="card-label">Largeur des r&#233;gimes par actif (jours calendaires, box plot)</div>
    <div id="chart-box" style="height:340px"></div>
  </div>

  <div class="card">
    <div class="card-label">Volatilit&#233; comme d&#233;clencheur de changement de r&#233;gime</div>
    <p class="chart-note">Statistique descriptive simple, &#224; lire avec prudence : % des changements de
      r&#233;gime pr&#233;c&#233;d&#233;s (dans les 3 jours avant) d'un &#963;<sub>t</sub> au-dessus de son 75<sup>e</sup>
      percentile glissant (60j). <strong>Ce n'est pas un test de significativit&#233;</strong> — l'&#233;chantillon
      d'&#233;v&#233;nements par actif est faible (voir "n=" sous chaque chiffre), et le r&#233;gime <em>stress</em> est
      lui-m&#234;me d&#233;fini en partie par &#963;<sub>t</sub> &#233;lev&#233;e dans le mod&#232;le HMM, donc une partie du lien est
      m&#233;canique plut&#244;t que pr&#233;dictive. &#192; prendre comme indication exploratoire, pas comme signal de
      trading valid&#233;.</p>
    <div class="grid2x2" id="leadlag-grid"></div>
  </div>

  <div class="card">
    <div class="card-label">Corr&#233;lation glissante inter-actifs (63j)</div>
    <p class="chart-note">Corr&#233;lation de Pearson glissante (fen&#234;tre 63 jours, rendements journaliers). BTC = Bitcoin, ETH = Ethereum, SPX = S&amp;P 500 (SPY), ZN = US Treasury 10Y Note Futures (ZN=F), le benchmark mondial des taux. Fond rouge = jours o&#249; au moins un des 4 actifs est en r&#233;gime de stress (p_stress &gt; 0.5).</p>
    <div id="chart-crosscorr" style="height:320px"></div>
  </div>

  <div class="card">
    <div class="card-label">Corr&#233;lation moyenne inter-actifs &#8212; stress vs calme (hypoth&#232;se de contagion)</div>
    <p class="chart-note">Corr&#233;lation moyenne des rendements journaliers entre chaque paire d'actifs, calcul&#233;e s&#233;par&#233;ment sur deux sous-&#233;chantillons de jours : Stress = au moins 1 actif sur 4 en r&#233;gime stress ce jour-l&#224;. Calme = les 4 actifs simultan&#233;ment en r&#233;gime calme ce jour-l&#224;. Hypoth&#232;se test&#233;e : la corr&#233;lation inter-actifs augmente en p&#233;riode de stress (contagion).</p>
    <table><thead><tr><th>Paire</th><th>Corr&#233;lation (stress)</th><th>Corr&#233;lation (calme)</th></tr></thead>
    <tbody id="pairs-body"></tbody></table>
    <p class="chart-note" id="corr-sig-note"></p>
  </div>
</div>

<footer>DEITA Benchmark &middot; Dashboard multi-actifs &middot; g&#233;n&#233;r&#233; le {generated_on}</footer>

<script>
const TAB_DATA = {j_tabs};
const COMPARISON = {j_comparison};
const ASSETS = {j_assets};
const REGIME_BG = {j_regime_bg};
const REGIME_HEX = {j_regime_hex};
const REGIME_LABELS = {j_regime_labels};
const EVENT_COLORS = {j_event_colors};

const BG='#16213e', GRID='rgba(255,255,255,0.05)', FONT={{family:'Segoe UI,sans-serif',color:'#ecf0f1',size:11}};
const baseLayout=()=>({{paper_bgcolor:BG,plot_bgcolor:BG,font:FONT,hovermode:'x unified',
  legend:{{bgcolor:'rgba(0,0,0,0)',font:{{size:10}}}},
  xaxis:{{gridcolor:GRID,zerolinecolor:GRID,type:'date',rangeslider:{{visible:false}}}}}});

// Un ticker comme "ZN=F" contient un '=' invalide dans un s&#233;lecteur/id CSS non &#233;chapp&#233;
// (document.querySelectorAll('.regime-cb-ZN=F') l&#232;ve une exception qui casse en silence toute
// l'interactivit&#233; de l'onglet). SHORT_OF fournit un identifiant DOM s&#251;r (ex. "ZN") pour tout
// id/class/s&#233;lecteur construit dynamiquement ; tabId (le ticker brut) reste utilis&#233; tel quel
// pour les acc&#232;s objet (TAB_DATA[tabId], TABS[tabId], ...) et l'attribut data-tab.
const SHORT_OF = {{}};
ASSETS.forEach(a => {{ SHORT_OF[a.ticker] = a.short; }});

const TABS = {{}};
ASSETS.forEach(a => {{
  TABS[a.ticker] = {{
    initialized:false,
    state:{{regimes:{{calm:true,trending:true,stress:true}}, cats:{{crypto:true,macro:true,monetaire:true,geopolitique:true}}, scale:'annee'}},
    currentXRange:[TAB_DATA[a.ticker].first_date, TAB_DATA[a.ticker].last_date],
    _programmatic:false,
  }};
}});
let comparisonInitialized = false;

// ── Navigation par onglets (lazy init) ──────────────────────────────────────────
function switchTab(tabId) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.dataset.tab===tabId));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab===tabId));
  if (tabId === 'COMPARISON') {{
    if (!comparisonInitialized) {{ initComparisonTab(); comparisonInitialized = true; }}
  }} else if (!TABS[tabId].initialized) {{
    initAssetTab(tabId); TABS[tabId].initialized = true;
  }}
}}
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

// ── Construction des shapes/annotations ─────────────────────────────────────────
function buildShapes(tabId) {{
  const d = TAB_DATA[tabId], st = TABS[tabId].state, shapes = [];
  d.regime_shapes.forEach(s => {{
    if (!st.regimes[s.regime]) return;
    shapes.push({{type:'rect',xref:'x',yref:'paper',x0:s.x0,x1:s.x1,y0:0,y1:1,
      fillcolor:REGIME_BG[s.regime],line:{{width:0}},layer:'below'}});
  }});
  d.event_lines.forEach(e => {{
    if (!st.cats[e.cat]) return;
    shapes.push({{type:'line',xref:'x',yref:'paper',x0:e.x,x1:e.x,y0:0,y1:1,
      line:{{color:e.color,width:1.2,dash:'dot'}}}});
  }});
  return shapes;
}}

function inRange(x, range) {{ return x >= range[0] && x <= range[1]; }}

function buildAnnotations(tabId) {{
  const d = TAB_DATA[tabId], st = TABS[tabId].state;
  const visible = d.event_annotations.filter(a => st.cats[a.cat] && inRange(a.x, TABS[tabId].currentXRange));
  if (visible.length > 8) return {{annotations:[], overflow:true}};
  return {{
    annotations: visible.map(a => ({{x:a.x,y:a.y,xref:'x',yref:'paper',text:a.text,showarrow:false,
      font:{{size:8,color:a.color}},textangle:-40,xanchor:'left'}})),
    overflow:false,
  }};
}}

function refreshTab(tabId) {{
  const shapes = buildShapes(tabId);
  ['price','vol','cp'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{shapes}}));
  const {{annotations, overflow}} = buildAnnotations(tabId);
  Plotly.relayout(`chart-price-${{SHORT_OF[tabId]}}`, {{annotations}});
  const msg = document.getElementById(`declutter-${{SHORT_OF[tabId]}}`);
  if (msg) msg.style.display = overflow ? 'block' : 'none';
}}

// ── Composition des r&#233;gimes : donut recalcul&#233; sur la fen&#234;tre visible ──────────────
function updateComposition(tabId) {{
  const d = TAB_DATA[tabId], range = TABS[tabId].currentXRange;
  let nCalm = 0, nTrend = 0, nStress = 0, total = 0;
  for (let i = 0; i < d.dates.length; i++) {{
    if (d.dates[i] >= range[0] && d.dates[i] <= range[1]) {{
      total++;
      if (d.regimes[i] === 'calm') nCalm++;
      else if (d.regimes[i] === 'trending') nTrend++;
      else nStress++;
    }}
  }}
  const pct = v => total ? v / total * 100 : 0;
  const trace = {{
    type:'pie', labels:['Calme','Tendanciel','Stress'],
    values:[pct(nCalm), pct(nTrend), pct(nStress)],
    marker:{{colors:[REGIME_HEX.calm, REGIME_HEX.trending, REGIME_HEX.stress]}},
    hole:0.42, textinfo:'label+percent', textfont:{{size:11,color:'#ecf0f1'}}, showlegend:false,
  }};
  Plotly.react(`chart-dist-${{SHORT_OF[tabId]}}`, [trace], {{
    paper_bgcolor:BG, plot_bgcolor:BG, font:FONT, margin:{{l:10,r:10,t:26,b:10}},
    title:{{text:`${{range[0]}} &#8594; ${{range[1]}} &middot; ${{total}} j`, font:{{size:10,color:'#7f8c8d'}}}},
  }}, {{responsive:true,displayModeBar:false}});
}}

// ── Zoom temporel (boutons Jour/Mois/Trimestre/Ann&#233;e + s&#233;lecteur de date) ──────────
const ZOOM_WINDOW_DAYS = {{jour:60, mois:365, trimestre:1095}};

function clampDate(dateObj, minStr, maxStr) {{
  const s = dateObj.toISOString().slice(0, 10);
  if (s < minStr) return minStr;
  if (s > maxStr) return maxStr;
  return s;
}}

function applyZoom(tabId, scale, anchorDateStr) {{
  const d = TAB_DATA[tabId];
  TABS[tabId].state.scale = scale;
  let r0, r1;
  if (scale === 'annee') {{
    r0 = d.first_date; r1 = d.last_date;
  }} else {{
    const anchor = anchorDateStr ? new Date(anchorDateStr + 'T00:00:00') : new Date(d.last_date + 'T00:00:00');
    const half = ZOOM_WINDOW_DAYS[scale] / 2;
    const start = new Date(anchor); start.setDate(start.getDate() - half);
    const end = new Date(anchor); end.setDate(end.getDate() + half);
    r0 = clampDate(start, d.first_date, d.last_date);
    r1 = clampDate(end, d.first_date, d.last_date);
  }}
  TABS[tabId]._programmatic = true;
  ['price','vol','cp'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{
    'xaxis.range[0]': r0, 'xaxis.range[1]': r1,
  }}));
  TABS[tabId]._programmatic = false;
  TABS[tabId].currentXRange = [r0, r1];
  onRangeChange(tabId);

  const dp = document.getElementById(`datepick-${{SHORT_OF[tabId]}}`);
  if (dp) dp.disabled = (scale === 'annee');
}}

// ── Initialisation d'un onglet actif ────────────────────────────────────────────
function initAssetTab(tabId) {{
  const d = TAB_DATA[tabId];
  const shapes = buildShapes(tabId);

  const priceTrace = {{type:'scatter',x:d.dates,y:d.close,mode:'lines',
    line:{{color:'#ecf0f1',width:1.5}},name:d.label,showlegend:true}};
  const regimeTraces = ['calm','trending','stress'].map(r => ({{
    type:'scatter',x:[null],y:[null],mode:'markers',
    marker:{{color:REGIME_HEX[r],size:10,symbol:'square'}},name:REGIME_LABELS[r],showlegend:true}}));

  Plotly.newPlot(`chart-price-${{SHORT_OF[tabId]}}`, [priceTrace].concat(regimeTraces),
    Object.assign({{}},baseLayout(),{{margin:{{l:60,r:18,t:8,b:38}},shapes,annotations:[],
      yaxis:{{title:'Prix (USD)',gridcolor:GRID,tickformat:',.0f',type:'log'}}}}),
    {{responsive:true,displayModeBar:false}});

  const sigmaTrace = {{type:'scatter',x:d.dates,y:d.sigma,mode:'lines',
    line:{{color:'#9b59b6',width:1.5}},name:'&#963;&#8339; GARCH (%)',showlegend:true}};
  const vovTrace = {{type:'scatter',x:d.dates,y:d.vol_of_vol,mode:'lines',
    line:{{color:'#e67e22',width:1.2,dash:'dot'}},fill:'tozeroy',fillcolor:'rgba(230,126,34,0.10)',
    name:'Vol-of-Vol (20j)',showlegend:true}};
  Plotly.newPlot(`chart-vol-${{SHORT_OF[tabId]}}`, [sigmaTrace, vovTrace],
    Object.assign({{}},baseLayout(),{{margin:{{l:60,r:18,t:8,b:38}},shapes,
      yaxis:{{title:'Vol (%)',gridcolor:GRID}}}}), {{responsive:true,displayModeBar:false}});

  const cpTrace = {{type:'scatter',x:d.dates,y:d.cp,mode:'lines',line:{{color:'#f39c12',width:1.5}},
    fill:'tozeroy',fillcolor:'rgba(243,156,18,0.12)',name:'changepoint_prob',showlegend:true}};
  const threshTrace = {{type:'scatter',x:[d.dates[0],d.dates[d.dates.length-1]],y:[0.5,0.5],
    mode:'lines',line:{{color:'#e74c3c',width:1,dash:'dash'}},name:'seuil 0.5',showlegend:true}};
  Plotly.newPlot(`chart-cp-${{SHORT_OF[tabId]}}`, [cpTrace, threshTrace],
    Object.assign({{}},baseLayout(),{{margin:{{l:60,r:18,t:8,b:38}},shapes,
      yaxis:{{title:'P(chgt)',gridcolor:GRID,range:[0,1.05],dtick:0.25}}}}), {{responsive:true,displayModeBar:false}});

  updateComposition(tabId);

  // ── Table &#233;v&#233;nements ──────────────────────────────────────────────────────
  const tbody = document.getElementById(`evt-body-${{SHORT_OF[tabId]}}`);
  d.events_table.forEach(e => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{e.date}}</td><td>${{e.label}}</td><td><span class="tag" style="background:${{e.color}}">${{e.cat}}</span></td>`;
    tbody.appendChild(tr);
  }});

  // ── Sync zoom : chart-price est la SEULE source d'&#233;coute (drag/zoom souris natif) ────
  // vol/cp n'ont aucun listener : ils ne font que suivre. Avec 3 charts qui s'&#233;coutaient
  // mutuellement (v1/v2), un Plotly.relayout() programmatique pouvait re-d&#233;clencher
  // 'plotly_relayout' de fa&#231;on asynchrone sur les autres charts, qui relayoutaient &#224; leur
  // tour et red&#233;clenchaient l'&#233;v&#233;nement en boucle (le garde-fou _syncing, remis &#224; false
  // de fa&#231;on synchrone, n'arrivait pas &#224; bloquer un rebond asynchrone) — page compl&#232;tement
  // gel&#233;e au clic. Une seule source (price) rend un tel cycle structurellement impossible.
  document.getElementById(`chart-price-${{SHORT_OF[tabId]}}`).on('plotly_relayout', e=>{{
    if (TABS[tabId]._programmatic) return;  // ignore les relayouts d&#233;clench&#233;s par applyZoom
    if (e['xaxis.range[0]']!==undefined) {{
      // Un drag-zoom natif retourne des bornes avec heure (ex. '2020-07-31 00:22:46.6081') ;
      // on tronque au jour pour rester coh&#233;rent avec les bornes 'YYYY-MM-DD' produites par
      // applyZoom (boutons/date picker), utilis&#233;es telles quelles dans les comparaisons de
      // updateComposition/buildAnnotations.
      const r0 = String(e['xaxis.range[0]']).slice(0,10), r1 = String(e['xaxis.range[1]']).slice(0,10);
      Plotly.relayout(`chart-vol-${{SHORT_OF[tabId]}}`, {{'xaxis.range[0]':r0,'xaxis.range[1]':r1}});
      Plotly.relayout(`chart-cp-${{SHORT_OF[tabId]}}`, {{'xaxis.range[0]':r0,'xaxis.range[1]':r1}});
      TABS[tabId].currentXRange = [r0, r1];
      onRangeChange(tabId);
    }}
  }});

  // ── Checkboxes l&#233;gende (r&#233;gimes + cat&#233;gories d'&#233;v&#233;nements) ─────────────────────────
  document.querySelectorAll(`.regime-cb-${{SHORT_OF[tabId]}}`).forEach(cb => {{
    cb.addEventListener('change', () => {{ TABS[tabId].state.regimes[cb.value] = cb.checked; refreshTab(tabId); }});
  }});
  document.querySelectorAll(`.cat-cb-${{SHORT_OF[tabId]}}`).forEach(cb => {{
    cb.addEventListener('change', () => {{ TABS[tabId].state.cats[cb.value] = cb.checked; refreshTab(tabId); }});
  }});

  // ── Zoom temporel (boutons Jour/Mois/Trimestre/Ann&#233;e + s&#233;lecteur de date) ──────────
  document.querySelectorAll(`.scale-btn-${{SHORT_OF[tabId]}}`).forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll(`.scale-btn-${{SHORT_OF[tabId]}}`).forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const dp = document.getElementById(`datepick-${{SHORT_OF[tabId]}}`);
      applyZoom(tabId, btn.dataset.scale, dp && dp.value ? dp.value : null);
    }});
  }});
  const dp = document.getElementById(`datepick-${{SHORT_OF[tabId]}}`);
  dp.addEventListener('change', () => {{
    const activeBtn = document.querySelector(`.scale-btn-${{SHORT_OF[tabId]}}.active`);
    applyZoom(tabId, activeBtn.dataset.scale, dp.value);
  }});
}}

function onRangeChange(tabId) {{
  const {{annotations, overflow}} = buildAnnotations(tabId);
  Plotly.relayout(`chart-price-${{SHORT_OF[tabId]}}`, {{annotations}});
  const msg = document.getElementById(`declutter-${{SHORT_OF[tabId]}}`);
  if (msg) msg.style.display = overflow ? 'block' : 'none';
  updateComposition(tabId);
}}

// ── Onglet Comparaison ───────────────────────────────────────────────────────────
function initComparisonTab() {{
  const boxData = COMPARISON.box_traces.map(t => ({{
    type:'box', name:t.name, x:t.x, y:t.y, marker:{{color:t.color}},
  }}));
  Plotly.newPlot('chart-box', boxData, Object.assign({{}},baseLayout(),{{
    boxmode:'group', margin:{{l:60,r:18,t:8,b:38}},
    xaxis:{{type:'category',categoryarray:['calm','trending','stress'],
      ticktext:['Calme','Tendanciel','Stress'],tickvals:['calm','trending','stress']}},
    yaxis:{{title:'Dur&#233;e (jours calendaires)',gridcolor:GRID}},
  }}), {{responsive:true,displayModeBar:false}});

  const grid = document.getElementById('leadlag-grid');
  ASSETS.forEach(a => {{
    const hr = COMPARISON.hit_rates[a.ticker];
    const wrap = document.createElement('div');
    wrap.innerHTML = `
      <div class="hitrate-label">${{hr.label}}</div>
      <div class="hitrate">${{(hr.hit_rate*100).toFixed(0)}}%</div>
      <div class="hitrate-label">des changements de r&#233;gime pr&#233;c&#233;d&#233;s d'un pic de vol (3j)<br>(n=${{hr.n_regime_changes}} changements observ&#233;s)</div>`;
    grid.appendChild(wrap);
  }});

  const cc = COMPARISON.cross_correlation;
  const palette = ['#f7931a','#627eea','#2ecc71','#3498db','#e67e22','#9b59b6'];
  const ccTraces = Object.keys(cc.series).map((col,i) => ({{
    type:'scatter', mode:'lines', x:cc.dates, y:cc.series[col], name:col,
    line:{{width:1.5,color:palette[i % palette.length]}},
  }}));
  const stressShapes = COMPARISON.stress_bands.map(b => ({{
    type:'rect',xref:'x',yref:'paper',x0:b.x0,x1:b.x1,y0:0,y1:1,
    fillcolor:'rgba(231,76,60,0.14)',line:{{width:0}},layer:'below'}}));
  Plotly.newPlot('chart-crosscorr', ccTraces, Object.assign({{}},baseLayout(),{{
    margin:{{l:50,r:18,t:8,b:38}}, shapes:stressShapes,
    yaxis:{{title:'Corr&#233;lation (63j)',gridcolor:GRID,range:[-1,1]}},
  }}), {{responsive:true,displayModeBar:false}});

  const pbody = document.getElementById('pairs-body');
  const fmtSig = (v, sig) => v === null ? '&#8212;'
    : `${{v.toFixed(3)}}${{sig ? ' <span class="sig-star" title="Significatif &#224; 95%">*</span>' : ''}}`;
  COMPARISON.pairs_table.forEach(p => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{p.pair}}</td><td>${{fmtSig(p.stress, p.stress_sig)}}</td><td>${{fmtSig(p.calm, p.calm_sig)}}</td>`;
    pbody.appendChild(tr);
  }});

  const sig = COMPARISON.corr_significance;
  document.getElementById('corr-sig-note').innerHTML =
    `* = corr&#233;lation significativement diff&#233;rente de 0 au seuil de confiance 95% ` +
    `(test de Fisher). Stress : n=${{sig.n_stress}} jours, seuil |r| &gt; ${{sig.r_crit_stress!==null ? sig.r_crit_stress.toFixed(3) : '&#8212;'}}. ` +
    `Calme : n=${{sig.n_calm}} jours, seuil |r| &gt; ${{sig.r_crit_calm!==null ? sig.r_crit_calm.toFixed(3) : '&#8212;'}}.`;
}}

// ── D&#233;marrage ──────────────────────────────────────────────────────────────────
switchTab('{ASSETS[0]["ticker"]}');
</script>
</body>
</html>"""


def _asset_panel_html(asset: dict, first_date: str, last_date: str) -> str:
    ticker = asset["ticker"]
    # Identifiant DOM sûr : un ticker comme "ZN=F" contient un '=' invalide dans un id/class/
    # sélecteur CSS non échappé (cf. commentaire SHORT_OF côté JS). "short" (ex. "ZN") est déjà
    # garanti alphanumérique par assets.py. `data-tab` garde le ticker brut (simple attribut
    # HTML, comparé en JS via égalité de chaîne — jamais parsé comme sélecteur CSS).
    dom = asset["short"]
    active = " active" if asset is ASSETS[0] else ""
    return f"""
<div class="tab-panel{active}" data-tab="{ticker}">
  <div class="legend">
    <div class="li"><input type="checkbox" class="regime-cb-{dom}" value="calm" checked><div class="dot" style="background:{_REGIME_HEX['calm']}"></div>Calme</div>
    <div class="li"><input type="checkbox" class="regime-cb-{dom}" value="trending" checked><div class="dot" style="background:{_REGIME_HEX['trending']}"></div>Tendanciel</div>
    <div class="li"><input type="checkbox" class="regime-cb-{dom}" value="stress" checked><div class="dot" style="background:{_REGIME_HEX['stress']}"></div>Stress</div>
    <div class="sep"></div>
    <div class="li"><input type="checkbox" class="cat-cb-{dom}" value="crypto" checked><div class="dot" style="background:{_EVENT_COLORS['crypto']}"></div>Crypto</div>
    <div class="li"><input type="checkbox" class="cat-cb-{dom}" value="macro" checked><div class="dot" style="background:{_EVENT_COLORS['macro']}"></div>Macro</div>
    <div class="li"><input type="checkbox" class="cat-cb-{dom}" value="monetaire" checked><div class="dot" style="background:{_EVENT_COLORS['monetaire']}"></div>Mon&#233;taire</div>
    <div class="li"><input type="checkbox" class="cat-cb-{dom}" value="geopolitique" checked><div class="dot" style="background:{_EVENT_COLORS['geopolitique']}"></div>G&#233;opolitique</div>
  </div>

  <div class="card">
    <div class="card-label">Prix {asset["label"]} (USD, &#233;chelle log) &#8212; fond color&#233; par r&#233;gime d&#233;tect&#233;</div>
    <div class="scale-sel">
      <span class="scale-label">Centrer sur</span>
      <input type="date" id="datepick-{dom}" class="date-pick" min="{first_date}" max="{last_date}" disabled>
      <button class="scale-btn scale-btn-{dom}" data-scale="jour">Jour</button>
      <button class="scale-btn scale-btn-{dom}" data-scale="mois">Mois</button>
      <button class="scale-btn scale-btn-{dom}" data-scale="trimestre">Trimestre</button>
      <button class="scale-btn scale-btn-{dom} active" data-scale="annee">Ann&#233;e</button>
    </div>
    <div id="chart-price-{dom}" style="height:360px"></div>
    <div class="declutter-msg" id="declutter-{dom}">Zoomez pour voir les libell&#233;s (trop d'&#233;v&#233;nements visibles)</div>
  </div>

  <div class="card">
    <div class="card-label">Volatilit&#233; conditionnelle GARCH(1,1) &middot; &#963;<sub>t</sub> (violet) et Vol-of-Vol rolling 20j (orange)</div>
    <div id="chart-vol-{dom}" style="height:140px"></div>
  </div>

  <div class="card">
    <div class="card-label">Probabilit&#233; de changement de r&#233;gime &#8212; BOCPD &middot; seuil 0.5</div>
    <div id="chart-cp-{dom}" style="height:120px"></div>
  </div>

  <div class="row2">
    <div class="card" style="margin-bottom:0">
      <div class="card-label">Composition moyenne des r&#233;gimes</div>
      <div id="chart-dist-{dom}" style="height:230px"></div>
    </div>
    <div class="card" style="margin-bottom:0">
      <details>
        <summary>&#201;v&#233;nements de march&#233; r&#233;f&#233;renc&#233;s</summary>
        <div style="overflow-y:auto;max-height:230px;margin-top:6px">
          <table><thead><tr><th>Date</th><th>&#201;v&#233;nement</th><th>Cat&#233;gorie</th></tr></thead>
          <tbody id="evt-body-{dom}"></tbody></table>
        </div>
      </details>
    </div>
  </div>
</div>
"""


def main():
    results = run_pipeline()
    analytics = compute_all_analytics(results)
    html = build_multi_asset_html(results, analytics)
    out = Path(__file__).parent / "output" / "regime_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[dashboard_builder] HTML genere -> {out.resolve()}")
    return str(out.resolve())


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    main()
