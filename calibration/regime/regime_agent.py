"""
regime_agent.py — Assemblage final du moteur de régime DEITA

Orchestre RegimeHMM (Maéva) + RegimeBOCPD (Kyrio) et produit un RegimeState complet.
Génère également la page HTML de démonstration interactive.

Dépendances : numpy, pandas, json, pathlib, datetime (stdlib).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from calibration.regime.regime_state import RegimeState
from calibration.regime.regime_hmm import RegimeHMM
from calibration.regime.regime_bocpd import RegimeBOCPD


# ── Événements de marché (réutilisés depuis Benchmark 1.7) ────────────────────

MARKET_EVENTS = {
    "2017-11-29": ("BTC ATH $10k",        "crypto"),
    "2018-01-17": ("BTC ATH $20k",        "crypto"),
    "2018-12-15": ("BTC bas $3.2k",       "crypto"),
    "2020-03-12": ("COVID crash",          "macro"),
    "2020-03-15": ("Fed taux 0%",         "monetaire"),
    "2020-05-11": ("BTC halving #3",       "crypto"),
    "2020-12-16": ("BTC franchit $20k",    "crypto"),
    "2021-02-08": ("Tesla 1.5G$ BTC",     "crypto"),
    "2021-04-14": ("Coinbase IPO",         "crypto"),
    "2021-09-07": ("El Salvador BTC",      "geopolitique"),
    "2021-11-10": ("BTC ATH $69k",        "crypto"),
    "2022-01-05": ("Fed pivot hawkish",    "monetaire"),
    "2022-02-24": ("Invasion Ukraine",     "geopolitique"),
    "2022-05-09": ("LUNA collapse",        "crypto"),
    "2022-06-15": ("Fed +75bp",           "monetaire"),
    "2022-09-15": ("ETH Merge PoS",        "crypto"),
    "2022-11-08": ("FTX collapse",         "crypto"),
    "2023-03-10": ("SVB faillite",         "macro"),
    "2023-07-26": ("Fed pic 5.25%",       "monetaire"),
    "2024-01-10": ("BTC ETF spot",         "crypto"),
    "2024-03-14": ("BTC ATH $73k",        "crypto"),
    "2024-04-19": ("BTC halving #4",       "crypto"),
    "2025-01-23": ("BTC ATH $109k",       "crypto"),
}

_EVENT_COLORS = {
    "crypto":       "#e67e22",
    "macro":        "#e74c3c",
    "monetaire":    "#2980b9",
    "geopolitique": "#8e44ad",
}

_REGIME_BG = {
    "calm":     "rgba(39, 174, 96, 0.18)",
    "trending": "rgba(41, 128, 185, 0.18)",
    "stress":   "rgba(231, 76, 60, 0.22)",
}

_REGIME_HEX = {
    "calm":     "#27ae60",
    "trending": "#2980b9",
    "stress":   "#e74c3c",
}

_REGIME_LABELS = {
    "calm":     "Calme",
    "trending": "Tendanciel",
    "stress":   "Stress",
}


class RegimeAgent:
    """
    Assemblage final du moteur de régime DEITA.

    Orchestre :
      - RegimeHMM  (Maéva) : GARCH(1,1) + HMM 2 états + seuil ADX + vol_bucket
      - RegimeBOCPD (Kyrio) : BOCPD pour changepoint_prob et is_transitioning

    Interface publique
    ------------------
        agent = RegimeAgent()
        agent.fit(prices, train_end)       # entraîne HMM + BOCPD
        state = agent.predict(prices)      # RetourneRegimeState complet
        agent.generate_html(prices)        # génère output/regime.html
    """

    def __init__(self) -> None:
        self._hmm = RegimeHMM()
        self._bocpd = RegimeBOCPD()
        self._is_fitted = False

    # ── API publique ───────────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame, train_end: str) -> None:
        """
        Entraîne HMM et BOCPD sur l'historique jusqu'à train_end inclus.

        Paramètres
        ----------
        prices : pd.DataFrame
            Données OHLCV complètes (yfinance, colonnes Open/High/Low/Close/Volume,
            index DatetimeIndex quotidien).
        train_end : str
            Date de fin d'entraînement inclusive au format "YYYY-MM-DD".
        """
        prices_train = prices[prices.index <= train_end]
        self._hmm.fit(prices, train_end)
        self._bocpd.fit(prices_train)
        self._is_fitted = True

    def predict(
        self,
        prices: pd.DataFrame,
        as_of: Optional[datetime] = None,
    ) -> RegimeState:
        """
        Retourne un RegimeState complet pour la date as_of.

        Contrainte point-in-time : seules les données strictement antérieures à
        as_of sont utilisées. Si as_of est None, utilise la dernière date disponible.

        Paramètres
        ----------
        prices : pd.DataFrame
            Données OHLCV complètes (filtre point-in-time appliqué en interne).
        as_of : datetime, optional
            Date de prédiction. None → dernière date de prices.

        Retourne
        --------
        RegimeState avec tous les champs remplis et validés.
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeAgent doit être entraîné — appeler fit() d'abord.")

        if as_of is None:
            as_of = prices.index[-1].to_pydatetime()

        state = self._hmm.predict(prices, as_of)
        cp, trans = self._bocpd.predict(prices, as_of)
        state.changepoint_prob = cp
        state.is_transitioning = trans
        state.validate()
        return state

    def generate_html(
        self,
        prices: pd.DataFrame,
        output_path: str = "calibration/regime/output/regime.html",
    ) -> str:
        """
        Génère la page HTML interactive de démonstration et la sauvegarde sur disque.

        Contenu du HTML
        ---------------
        - Prix BTC-USD (échelle log) avec fond coloré par régime détecté
          (vert = calme, bleu = tendanciel, rouge = stress)
        - Marqueurs verticaux pointillés des événements MARKET_EVENTS avec étiquettes
        - Courbe changepoint_prob (BOCPD) en sous-graphe avec seuil 0.5
        - Fichier HTML autonome (Plotly via CDN, pas de serveur nécessaire)
        - Les deux graphiques sont synchronisés sur l'axe X (zoom/pan liés)

        Paramètres
        ----------
        prices : pd.DataFrame
            Données OHLCV complètes.
        output_path : str
            Chemin de sortie pour le fichier HTML. Le dossier parent est créé
            automatiquement si nécessaire.

        Retourne
        --------
        str : chemin absolu du fichier généré.
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeAgent doit être entraîné — appeler fit() d'abord.")

        df = self._predict_history(prices)
        html = self._build_html(prices, df)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"[RegimeAgent] HTML genere -> {out.resolve()}")
        return str(out.resolve())

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _predict_history(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le régime pour toutes les dates de prices en une passe (batch).

        Note : pas de contrainte point-in-time ici — on utilise le modèle entraîné
        pour classer l'ensemble de l'historique. Réservé à la visualisation.

        Retourne un DataFrame indexé comme prices avec les colonnes :
            regime (str), p_calm, p_trending, p_stress, vol_bucket, changepoint_prob
        """
        # ── 1. Features + passe HMM ────────────────────────────────────────────
        features = self._hmm._compute_features(prices)
        features = features.dropna()

        X_scaled = self._hmm._scaler.transform(features)
        probs_matrix = self._hmm._hmm.predict_proba(X_scaled)  # shape (n, 2)

        stress_idx = next(
            i for i, lbl in self._hmm._state_labels.items() if lbl == "stress"
        )
        p_stress = probs_matrix[:, stress_idx]
        p_non_stress = 1.0 - p_stress

        adx = features["adx"].values
        thresh = RegimeHMM.ADX_TRENDING_THRESHOLD
        p_trending = np.where(adx > thresh, p_non_stress, 0.0)
        p_calm = np.where(adx <= thresh, p_non_stress, 0.0)

        # ── 2. vol_bucket ──────────────────────────────────────────────────────
        q33, q66 = self._hmm._vol_thresholds
        sigma_t = features["sigma_t"].values
        vol_bucket = np.where(sigma_t < q33, 0, np.where(sigma_t < q66, 1, 2))

        # ── 3. Régime dominant (argmax) ────────────────────────────────────────
        stacked = np.stack([p_calm, p_trending, p_stress], axis=1)
        idx = np.argmax(stacked, axis=1)
        _map = {0: "calm", 1: "trending", 2: "stress"}
        regimes = [_map[int(i)] for i in idx]

        df = pd.DataFrame(
            {
                "regime":     regimes,
                "p_calm":     p_calm,
                "p_trending": p_trending,
                "p_stress":   p_stress,
                "vol_bucket": vol_bucket.astype(int),
            },
            index=features.index,
        )

        df["sigma_t"]    = features["sigma_t"].values
        df["vol_of_vol"] = features["sigma_t"].rolling(20).std().values

        # ── 4. BOCPD changepoint_prob (passe complète sur tout l'historique) ───
        returns = prices["Close"].pct_change().dropna()
        signal = self._bocpd._to_signal(returns)
        cp_raw = self._bocpd._run_bocpd(signal)
        cp_series = pd.Series(cp_raw, index=returns.index)
        df["changepoint_prob"] = cp_series.reindex(df.index).fillna(0.0)

        return df

    def _build_html(self, prices: pd.DataFrame, df: pd.DataFrame) -> str:
        """Construit le HTML complet avec Plotly (CDN)."""

        close = prices["Close"].reindex(df.index).ffill()
        dates_str  = [str(d.date()) for d in df.index]
        close_vals = [round(float(v), 2) if not np.isnan(v) else None for v in close]
        cp_vals    = [round(float(v), 4) for v in df["changepoint_prob"]]
        sigma_vals = [round(float(v), 4) if not np.isnan(v) else None for v in df["sigma_t"]]
        vov_vals   = [round(float(v), 4) if not np.isnan(v) else None for v in df["vol_of_vol"]]
        regimes    = df["regime"].tolist()

        first_date = df.index[0]
        last_date  = df.index[-1]

        # ── Distribution des régimes (donut) ──────────────────────────────────
        n_total = len(df)
        rc = df["regime"].value_counts()
        dist_vals = [round(rc.get(r, 0) / n_total * 100, 1) for r in ["calm", "trending", "stress"]]

        # ── Rectangles de fond (régimes) pour les 3 graphiques temporels ──────
        shapes_price, shapes_vol, shapes_cp = [], [], []
        i = 0
        while i < len(regimes):
            j, r = i, regimes[i]
            while j < len(regimes) and regimes[j] == r:
                j += 1
            col = _REGIME_BG[r]
            x0, x1 = dates_str[i], dates_str[j - 1]
            for lst in (shapes_price, shapes_vol, shapes_cp):
                lst.append({
                    "type": "rect", "xref": "x", "yref": "paper",
                    "x0": x0, "x1": x1, "y0": 0, "y1": 1,
                    "fillcolor": col, "line": {"width": 0}, "layer": "below",
                })
            i = j

        # ── Événements MARKET_EVENTS ───────────────────────────────────────────
        event_annotations = []
        sorted_events = [
            (d, lbl, cat)
            for d, (lbl, cat) in sorted(MARKET_EVENTS.items())
            if first_date <= pd.Timestamp(d) <= last_date
        ]
        for k, (date_str, label, cat) in enumerate(sorted_events):
            col = _EVENT_COLORS.get(cat, "#888")
            line_shape = {
                "type": "line", "xref": "x", "yref": "paper",
                "x0": date_str, "x1": date_str, "y0": 0, "y1": 1,
                "line": {"color": col, "width": 1.2, "dash": "dot"},
            }
            for lst in (shapes_price, shapes_vol, shapes_cp):
                lst.append(line_shape)
            ypos = 0.96 if k % 2 == 0 else 0.84
            event_annotations.append({
                "x": date_str, "y": ypos,
                "xref": "x", "yref": "paper",
                "text": f"<b>{label}</b>",
                "showarrow": False,
                "font": {"size": 8, "color": col},
                "textangle": -40,
                "xanchor": "left",
            })

        # ── Traces Plotly ──────────────────────────────────────────────────────
        price_trace = {
            "type": "scatter", "x": dates_str, "y": close_vals,
            "mode": "lines", "line": {"color": "#ecf0f1", "width": 1.5},
            "name": "BTC-USD", "showlegend": True,
        }
        regime_traces = [
            {
                "type": "scatter", "x": [None], "y": [None], "mode": "markers",
                "marker": {"color": _REGIME_HEX[r], "size": 10, "symbol": "square"},
                "name": _REGIME_LABELS[r], "showlegend": True,
            }
            for r in ["calm", "trending", "stress"]
        ]
        sigma_trace = {
            "type": "scatter", "x": dates_str, "y": sigma_vals,
            "mode": "lines", "line": {"color": "#9b59b6", "width": 1.5},
            "name": "σₜ GARCH (%)", "showlegend": True,
        }
        vov_trace = {
            "type": "scatter", "x": dates_str, "y": vov_vals,
            "mode": "lines",
            "line": {"color": "#e67e22", "width": 1.2, "dash": "dot"},
            "fill": "tozeroy", "fillcolor": "rgba(230,126,34,0.10)",
            "name": "Vol-of-Vol (rolling 20j)", "showlegend": True,
        }
        cp_trace = {
            "type": "scatter", "x": dates_str, "y": cp_vals,
            "mode": "lines",
            "line": {"color": "#f39c12", "width": 1.5},
            "fill": "tozeroy", "fillcolor": "rgba(243,156,18,0.12)",
            "name": "changepoint_prob", "showlegend": True,
        }
        threshold_trace = {
            "type": "scatter",
            "x": [dates_str[0], dates_str[-1]], "y": [0.5, 0.5],
            "mode": "lines",
            "line": {"color": "#e74c3c", "width": 1, "dash": "dash"},
            "name": "seuil 0.5", "showlegend": True,
        }
        dist_trace = {
            "type": "pie",
            "labels": ["Calme", "Tendanciel", "Stress"],
            "values": dist_vals,
            "marker": {"colors": ["#27ae60", "#2980b9", "#e74c3c"]},
            "hole": 0.42,
            "textinfo": "label+percent",
            "textfont": {"size": 11, "color": "#ecf0f1"},
            "showlegend": False,
        }

        # ── Table des événements ──────────────────────────────────────────────
        events_for_table = [
            {"date": d, "label": lbl, "cat": cat, "color": _EVENT_COLORS.get(cat, "#888")}
            for d, lbl, cat in sorted_events
        ]

        # ── Sérialisation JSON ─────────────────────────────────────────────────
        j_price_data  = json.dumps([price_trace] + regime_traces)
        j_vol_data    = json.dumps([sigma_trace, vov_trace])
        j_cp_data     = json.dumps([cp_trace, threshold_trace])
        j_dist_data   = json.dumps([dist_trace])
        j_shapes_p    = json.dumps(shapes_price)
        j_shapes_vol  = json.dumps(shapes_vol)
        j_shapes_cp   = json.dumps(shapes_cp)
        j_annotations = json.dumps(event_annotations)
        j_events      = json.dumps(events_for_table)

        generated_on = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEITA &#8212; Moteur de R&#233;gime BTC-USD</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f0f1a;color:#ecf0f1;font-family:'Segoe UI',sans-serif;padding:18px 22px}}
h1{{text-align:center;font-size:1.35rem;letter-spacing:1px;margin-bottom:3px}}
.sub{{text-align:center;color:#7f8c8d;font-size:.82rem;margin-bottom:18px}}
.card{{background:#16213e;border-radius:8px;padding:12px 14px;margin-bottom:14px}}
.card-label{{font-size:.72rem;text-transform:uppercase;letter-spacing:1.2px;color:#566573;margin-bottom:8px}}
.legend{{display:flex;gap:18px;justify-content:center;margin-bottom:14px;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:5px;font-size:.8rem}}
.dot{{width:11px;height:11px;border-radius:2px;flex-shrink:0}}
.sep{{width:1px;height:18px;background:#2c3e50;margin:0 4px}}
.row2{{display:grid;grid-template-columns:280px 1fr;gap:14px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:.76rem}}
thead th{{color:#7f8c8d;text-align:left;padding:4px 10px;border-bottom:1px solid #1c2a3a;white-space:nowrap}}
tbody td{{padding:3px 10px;border-bottom:1px solid #151d2b}}
.tag{{padding:1px 7px;border-radius:3px;font-size:.7rem;color:#fff;display:inline-block}}
footer{{text-align:center;color:#3d5166;font-size:.72rem;margin-top:14px}}
</style>
</head>
<body>
<h1>DEITA &#8212; Moteur de R&#233;gime BTC-USD</h1>
<p class="sub">Historique {str(first_date.date())} &#8594; {str(last_date.date())} &nbsp;&middot;&nbsp; HMM 2 &#233;tats + seuil ADX 25 + BOCPD (Adams &amp; MacKay 2007)</p>

<div class="legend">
  <div class="li"><div class="dot" style="background:#27ae60"></div>Calme</div>
  <div class="li"><div class="dot" style="background:#2980b9"></div>Tendanciel</div>
  <div class="li"><div class="dot" style="background:#e74c3c"></div>Stress</div>
  <div class="sep"></div>
  <div class="li"><div class="dot" style="background:#e67e22"></div>Crypto</div>
  <div class="li"><div class="dot" style="background:#e74c3c"></div>Macro</div>
  <div class="li"><div class="dot" style="background:#2980b9"></div>Mon&#233;taire</div>
  <div class="li"><div class="dot" style="background:#8e44ad"></div>G&#233;opolitique</div>
</div>

<div class="card">
  <div class="card-label">Prix BTC-USD (USD, &#233;chelle log) &#8212; fond color&#233; par r&#233;gime d&#233;tect&#233;</div>
  <div id="chart-price" style="height:360px"></div>
</div>

<div class="card">
  <div class="card-label">Volatilit&#233; conditionnelle GARCH(1,1) &middot; &#963;<sub>t</sub> (violet) et Vol-of-Vol rolling 20j (orange)</div>
  <div id="chart-vol" style="height:140px"></div>
</div>

<div class="card">
  <div class="card-label">Probabilit&#233; de changement de r&#233;gime &#8212; BOCPD &middot; P(run &#8804; 3j) &middot; seuil 0.5</div>
  <div id="chart-cp" style="height:120px"></div>
</div>

<div class="row2">
  <div class="card" style="margin-bottom:0">
    <div class="card-label">Distribution des r&#233;gimes &#8212; {str(first_date.date())} &#8594; {str(last_date.date())}</div>
    <div id="chart-dist" style="height:230px"></div>
  </div>
  <div class="card" style="margin-bottom:0;overflow-y:auto;max-height:270px">
    <div class="card-label">&#201;v&#233;nements de march&#233; r&#233;f&#233;renc&#233;s</div>
    <table><thead><tr><th>Date</th><th>&#201;v&#233;nement</th><th>Cat&#233;gorie</th></tr></thead>
    <tbody id="evt-body"></tbody></table>
  </div>
</div>

<footer>DEITA Benchmark &middot; g&#233;n&#233;r&#233; le {generated_on}</footer>

<script>
const BG='#16213e', GRID='rgba(255,255,255,0.05)', FONT={{family:'Segoe UI,sans-serif',color:'#ecf0f1',size:11}};
const baseLayout={{paper_bgcolor:BG,plot_bgcolor:BG,font:FONT,hovermode:'x unified',
  legend:{{bgcolor:'rgba(0,0,0,0)',font:{{size:10}}}},
  xaxis:{{gridcolor:GRID,zerolinecolor:GRID,type:'date',rangeslider:{{visible:false}}}}}};

// ── Prix ──────────────────────────────────────────────────────────────────────
Plotly.newPlot('chart-price',
  {j_price_data},
  Object.assign({{}},baseLayout,{{
    margin:{{l:60,r:18,t:8,b:38}},
    shapes:{j_shapes_p},
    annotations:{j_annotations},
    yaxis:{{title:'Prix (USD)',gridcolor:GRID,tickformat:',.0f',type:'log'}},
  }}),
  {{responsive:true,displayModeBar:false}});

// ── Volatilité ────────────────────────────────────────────────────────────────
Plotly.newPlot('chart-vol',
  {j_vol_data},
  Object.assign({{}},baseLayout,{{
    margin:{{l:60,r:18,t:8,b:38}},
    shapes:{j_shapes_vol},
    yaxis:{{title:'Vol (%)',gridcolor:GRID}},
  }}),
  {{responsive:true,displayModeBar:false}});

// ── BOCPD ─────────────────────────────────────────────────────────────────────
Plotly.newPlot('chart-cp',
  {j_cp_data},
  Object.assign({{}},baseLayout,{{
    margin:{{l:60,r:18,t:8,b:38}},
    shapes:{j_shapes_cp},
    yaxis:{{title:'P(chgt)',gridcolor:GRID,range:[0,1.05],dtick:0.25}},
  }}),
  {{responsive:true,displayModeBar:false}});

// ── Distribution ──────────────────────────────────────────────────────────────
Plotly.newPlot('chart-dist',
  {j_dist_data},
  {{paper_bgcolor:BG,plot_bgcolor:BG,font:FONT,margin:{{l:10,r:10,t:10,b:10}}}},
  {{responsive:true,displayModeBar:false}});

// ── Sync X : zoom/pan li&#233;s sur les 3 graphiques temporels ───────────────────
let _sync=false;
function syncRange(src,r0,r1){{
  if(_sync)return; _sync=true;
  ['chart-price','chart-vol','chart-cp'].filter(id=>id!==src).forEach(id=>{{
    Plotly.relayout(id,{{'xaxis.range[0]':r0,'xaxis.range[1]':r1}});
  }}); _sync=false;
}}
['chart-price','chart-vol','chart-cp'].forEach(id=>{{
  document.getElementById(id).on('plotly_relayout',e=>{{
    if(e['xaxis.range[0]']!==undefined) syncRange(id,e['xaxis.range[0]'],e['xaxis.range[1]']);
  }});
}});

// ── Table &#233;v&#233;nements ─────────────────────────────────────────────────────────
const evts={j_events};
const tbody=document.getElementById('evt-body');
evts.forEach(e=>{{
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${{e.date}}</td><td>${{e.label}}</td>
    <td><span class="tag" style="background:${{e.color}}">${{e.cat}}</span></td>`;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""


# ── Génération autonome ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Ajout de la racine du projet au path si nécessaire
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("[RegimeAgent] Telechargement des donnees BTC-USD via yfinance...")
    try:
        import yfinance as yf
        prices = yf.download("BTC-USD", start="2017-01-01", end="2025-01-01",
                             auto_adjust=True, progress=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        if len(prices) < 500:
            raise ValueError("Données insuffisantes")
        print(f"[RegimeAgent] {len(prices)} jours telecharges ({prices.index[0].date()} -> {prices.index[-1].date()})")
    except Exception as e:
        print(f"[RegimeAgent] yfinance indisponible ({e}), utilisation de donnees synthetiques.")
        np.random.seed(42)
        n = 1800
        dates = pd.date_range("2019-01-01", periods=n, freq="B")
        r = np.concatenate([
            np.random.normal(0, 0.01, 400),
            np.random.normal(0, 0.06, 30),   # COVID-like
            np.random.normal(0, 0.01, 700),
            np.random.normal(0, 0.05, 30),   # FTX-like
            np.random.normal(0, 0.01, 640),
        ])
        close = 8000 * np.cumprod(1 + r[:n])
        prices = pd.DataFrame({
            "Open": close * 0.998, "High": close * 1.01,
            "Low": close * 0.99,  "Close": close,
            "Volume": np.random.uniform(1e9, 5e9, n),
        }, index=dates)

    agent = RegimeAgent()
    train_end = "2023-12-31"
    print(f"[RegimeAgent] Entrainement jusqu'au {train_end}...")
    agent.fit(prices, train_end=train_end)

    out_path = Path(__file__).parent / "output" / "regime.html"
    agent.generate_html(prices, output_path=str(out_path))
    print(f"[RegimeAgent] Ouvrir dans un navigateur: {out_path}")
