"""
dashboard_builder.py — Dashboard multi-actifs DEITA (BTC/ETH/SPY/ZN=F/TLT)

Orchestre RegimeAgent (RegimeHMM + RegimeBOCPD) sur les 5 actifs de assets.py,
calcule les analyses de regime_analytics.py, et génère un dashboard HTML unique
à 6 onglets (5 actifs + comparaison). Aucune modification du moteur de régime :
RegimeHMM/RegimeBOCPD/RegimeAgent sont appelés tels quels, 5 fois.

Exécution :
    python -m calibration.regime.dashboard_builder
"""

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
      - regime_transition_vol_profile, par actif (étude d'événement : profil moyen de sigma_t
        ET de volume_norm autour des transitions de régime — toutes transitions, et transitions
        vers stress uniquement, questions 1 et 2)
      - granger_causality_vol_to_stress / granger_causality_volume_to_stress, par actif : test
        formel (ADF + Granger) en complément de l'étude d'événement descriptive ci-dessus,
        pour la vol (Q1) et le volume (Q2)

    Note (BRIEF_dashboard_v11_corrections.md) : le calcul cross-actifs (rolling_cross_correlation,
    pairwise_stress_calm_correlation, market_stress_majority) a été retiré d'ici — la page
    Comparaison ne les affiche plus pour l'instant, mais les fonctions elles-mêmes restent en
    place et testées dans regime_analytics.py/test_regime_analytics.py.
    Retourne un dict structuré prêt à sérialiser en JSON pour le template HTML.
    """
    per_asset = {}
    all_segments = []

    for asset in ASSETS:
        ticker = asset["ticker"]
        short = asset["short"]
        history = results[ticker]["history"]

        segments = ra.segment_regimes(history)
        width_stats = ra.regime_width_stats(segments)
        profile_all = ra.regime_transition_vol_profile(history, window=10, alignment="start")
        profile_into_stress = ra.regime_transition_vol_profile(
            history, window=10, alignment="start", only_into="stress"
        )
        profile_into_stress_volume = ra.regime_transition_vol_profile(
            history, window=10, alignment="start", only_into="stress", column="volume_norm"
        )
        granger = ra.granger_causality_vol_to_stress(history, maxlag=10)
        granger_volume = ra.granger_causality_volume_to_stress(history, maxlag=10)
        # Question 3 : entre vol et volume, qui bouge en premier EN GÉNÉRAL (pas seulement
        # autour des débuts de régime) ? Corrélation croisée ±5j sur les variations journalières,
        # sur tout l'historique — méthode distincte de granger_causality_*_to_stress ci-dessus,
        # qui elles sont conditionnées aux transitions vers le stress.
        vol_volume_ccf = ra.lead_lag_cross_correlation(
            history["sigma_t"].diff(), history["volume_norm"].diff(), max_lag=5
        )

        per_asset[ticker] = {
            "segments": segments,
            "width_stats": width_stats,
            "profile_all": profile_all,
            "profile_into_stress": profile_into_stress,
            "profile_into_stress_volume": profile_into_stress_volume,
            "granger": granger,
            "granger_volume": granger_volume,
            "vol_volume_ccf": vol_volume_ccf,
        }

        tagged = segments.copy()
        tagged["asset"] = short
        all_segments.append(tagged)

    combined_segments = pd.concat(all_segments, ignore_index=True)

    return {
        "per_asset": per_asset,
        "comparison": {
            "combined_segments": combined_segments,
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
        "volume_norm": [_num(v) if not np.isnan(v) else None for v in history["volume_norm"]],
        "cp": [_num(v) for v in history["changepoint_prob"]],
        "regimes": regimes,
        "regime_shapes": regime_shapes,
        "event_lines": event_lines,
        "event_annotations": event_annotations,
        "events_table": events_table,
        "first_date": str(first_date.date()),
        "last_date": str(last_date.date()),
    }


# Seuil de Bonferroni (correction pour comparaisons multiples : 10 lags testés par actif dans
# granger_causality_vol_to_stress / granger_causality_volume_to_stress) — un lag isolé sous 0.05
# mais au-dessus de ce seuil n'est pas une preuve, juste ce qu'on attend par hasard en testant
# 10 hypothèses. Partagé entre la Q1 (vol) et la Q2 (volume) pour rester cohérent.
GRANGER_BONFERRONI_ALPHA = 0.05 / 10


def _event_study_from_profile(profile: pd.DataFrame, label: str, color: str) -> dict:
    """
    Construit le payload d'étude d'événement (série indexée + premier jour de réaction
    significative) à partir d'un profil regime_transition_vol_profile — que ce profil porte sur
    sigma_t (Q1, vol) ou volume_norm (Q2, volume) : la méthode est identique, seule la colonne
    source du profil diffère (cf. BRIEF_dashboard_v11_corrections.md et son extension Q2).
    Indicatif, pas un test statistique formel — le test formel est granger_causality_*_to_stress.
    """
    if profile["n_events"].iloc[0] == 0 or profile["mean_sigma"].isna().all():
        return {
            "label": label, "color": color,
            "rel_day": profile["rel_day"].tolist(), "index_pct": [None] * len(profile),
            "n_events": 0, "first_reaction_day": None,
        }

    baseline_mask = profile["rel_day"].between(-10, -5)
    baseline = profile.loc[baseline_mask, "mean_sigma"].mean()
    n_events = int(profile["n_events"].iloc[0])

    index_pct = ((profile["mean_sigma"] / baseline - 1.0) * 100.0)

    # Seuil de déviation "significative" : 1,96 x erreur standard (95%, cohérent avec le
    # seuil de Fisher déjà utilisé ailleurs dans ce dashboard, cf. fisher_r_critical), calculée
    # à partir de l'écart-type de la période de RÉFÉRENCE (-10 à -5j, fixe) — pas de l'écart-type
    # du jour testé lui-même. Utiliser le std du jour testé est instable : il fluctue d'un jour à
    # l'autre par pur artefact d'échantillonnage sur les mêmes événements, et le jour où il tombe
    # le plus bas devient mécaniquement le plus facile à faire "significatif" — ce n'est pas un
    # vrai signal, juste un creux local du bruit. Un seuil à 1 erreur standard (au lieu de 1,96)
    # est par ailleurs beaucoup trop permissif sur 21 jours testés (~68% de confiance, pas 95%) :
    # on s'attend à des franchissements parasites même si rien ne se passe avant le jour 0
    # (constaté empiriquement sur BTC/vol : un creux de bruit à -2,5% déclenchait un "premier jour
    # de réaction" à J-3 avant cette correction).
    baseline_std = profile.loc[baseline_mask, "std_sigma"].mean()
    threshold = 1.959964 * baseline_std / np.sqrt(n_events)
    deviation = (profile["mean_sigma"] - baseline).abs()
    significant = deviation > threshold
    first_reaction_day = None
    for rel_day, is_sig in zip(profile["rel_day"], significant):
        if is_sig:
            first_reaction_day = int(rel_day)
            break

    return {
        "label": label, "color": color,
        "rel_day": profile["rel_day"].tolist(),
        "index_pct": [_num(v) if not np.isnan(v) else None for v in index_pct],
        "n_events": n_events,
        "first_reaction_day": first_reaction_day,
    }


def _granger_table_payload(analytics: dict, key: str, alpha: float) -> dict:
    """
    Construit le payload JSON-safe d'un test de Granger déjà calculé (par compute_all_analytics,
    sous per_asset[ticker][key]) pour chaque actif — réutilisé pour Q1 (key="granger") et
    Q2 (key="granger_volume"), même structure de sortie dans les deux cas.
    """
    out = {}
    for asset in ASSETS:
        ticker = asset["ticker"]
        g = analytics["per_asset"][ticker][key]
        p_values = {int(lag): _num(p) for lag, p in g["p_values"].items()}
        min_lag, min_p = min(g["p_values"].items(), key=lambda kv: kv[1])
        out[asset["short"]] = {
            "label": asset["label"], "color": asset["color"],
            "adf_source_p": _num(g["adf_source_p"]), "source_differenced": g["source_differenced"],
            "adf_pstress_p": _num(g["adf_pstress_p"]), "pstress_differenced": g["pstress_differenced"],
            "p_values": p_values,
            "min_p": _num(min_p), "min_p_lag": int(min_lag),
            "significant": bool(min_p < alpha),
            "n_obs": g["n_obs"],
        }
    return out


# Nombre de lags testés par lead_lag_cross_correlation (max_lag=5 -> -5..+5 = 11 lags) — sert à
# corriger le seuil de significativité de Fisher pour comparaisons multiples (même principe que
# GRANGER_BONFERRONI_ALPHA, appliqué ici à une corrélation plutôt qu'à une p-value de Granger).
CCF_N_TESTS = 11


def _vol_volume_ccf_payload(analytics: dict) -> dict:
    """
    Construit le payload de corrélation croisée vol/volume (Question 3, cf.
    lead_lag_cross_correlation) pour chaque actif : série lag/corr/n + seuil de significativité
    de Fisher CORRIGÉ POUR COMPARAISONS MULTIPLES (Bonferroni sur les 11 lags scannés — sinon un
    pic isolé dû au hasard du nombre de tests pourrait être pris pour un vrai signal, même piège
    que celui déjà corrigé pour granger_causality_*_to_stress) par lag (n varie légèrement d'un
    lag à l'autre à cause des NaN introduits par le décalage), les diagnostics de stationnarité
    ADF (cf. ensure_stationary, appliqué en interne par lead_lag_cross_correlation), et le
    verdict (qui mène, déterminé par le lag de |corrélation| maximale).
    """
    out = {}
    for asset in ASSETS:
        ticker = asset["ticker"]
        result = analytics["per_asset"][ticker]["vol_volume_ccf"]
        ccf = result["ccf"]
        lags = [int(v) for v in ccf["lag"]]
        corr = [_num(v) if not (isinstance(v, float) and np.isnan(v)) else None for v in ccf["corr"]]
        n_obs = [int(v) for v in ccf["n"]]
        r_crit = [
            _num(rc) if (rc := ra.fisher_r_critical_bonferroni(int(n), CCF_N_TESTS)) is not None else None
            for n in ccf["n"]
        ]

        valid = ccf.dropna(subset=["corr"])
        peak_idx = valid["corr"].abs().idxmax()
        peak_lag = int(valid.loc[peak_idx, "lag"])
        peak_corr = _num(valid.loc[peak_idx, "corr"])
        if peak_lag > 0:
            leader = "volume"
        elif peak_lag < 0:
            leader = "vol"
        else:
            leader = "aucun"

        out[asset["short"]] = {
            "label": asset["label"], "color": asset["color"],
            "lags": lags, "corr": corr, "n": n_obs, "r_crit": r_crit,
            "peak_lag": peak_lag, "peak_corr": peak_corr, "leader": leader,
            "adf_a_p": _num(result["adf_a_p"]), "a_differenced": result["a_differenced"],
            "adf_b_p": _num(result["adf_b_p"]), "b_differenced": result["b_differenced"],
        }
    return out


def _comparison_payload(results: dict, analytics: dict) -> dict:
    combined = analytics["comparison"]["combined_segments"]

    # ── Box plot largeur des régimes (5 actifs x 4 régimes) ────────────────────────
    box_traces = []
    for asset in ASSETS:
        sub = combined[combined["asset"] == asset["short"]]
        box_traces.append({
            "name": asset["short"],
            "color": asset["color"],
            "x": sub["regime"].tolist(),
            "y": [int(v) for v in sub["n_days_calendar"]],
        })

    # ── Étude d'événement : la volatilité (Q1), puis le volume (Q2), annoncent-ils ou
    # confirment-ils un passage en stress ? Indicatif dans les deux cas — le test formel
    # (causalité de Granger) est dans les cartes suivantes.
    event_study = {}
    event_study_volume = {}
    for asset in ASSETS:
        ticker = asset["ticker"]
        event_study[asset["short"]] = _event_study_from_profile(
            analytics["per_asset"][ticker]["profile_into_stress"], asset["label"], asset["color"]
        )
        event_study_volume[asset["short"]] = _event_study_from_profile(
            analytics["per_asset"][ticker]["profile_into_stress_volume"], asset["label"], asset["color"]
        )

    # ── Test formel : causalité de Granger, vol -> régime stress (Q1) et volume -> régime
    # stress (Q2), même méthode et même seuil corrigé pour les deux.
    granger = _granger_table_payload(analytics, "granger", GRANGER_BONFERRONI_ALPHA)
    granger_volume = _granger_table_payload(analytics, "granger_volume", GRANGER_BONFERRONI_ALPHA)

    # ── Question 3 : entre vol et volume, qui bouge en premier en général (pas seulement
    # autour des débuts de régime) ? Corrélation croisée ±5j sur variations journalières.
    vol_volume_ccf = _vol_volume_ccf_payload(analytics)

    return {
        "box_traces": box_traces,
        "event_study": event_study,
        "event_study_volume": event_study_volume,
        "granger": granger,
        "granger_volume": granger_volume,
        "granger_alpha": _num(GRANGER_BONFERRONI_ALPHA),
        "vol_volume_ccf": vol_volume_ccf,
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
.legend{{display:flex;gap:16px;justify-content:center;margin-bottom:14px;flex-wrap:wrap;font-size:.8rem}}
.li{{display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}}
.li input{{accent-color:#2980b9}}
.dot{{width:11px;height:11px;border-radius:2px;flex-shrink:0}}
.sep{{width:1px;height:18px;background:#2c3e50;margin:0 4px}}
.row2{{display:grid;grid-template-columns:280px 1fr;gap:14px;margin-bottom:14px}}
.grid2x2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:14px}}
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
<p class="sub">Bitcoin &middot; Ethereum &middot; S&amp;P 500 (SPY) &middot; US Treasury 10Y Note Futures (ZN=F) &middot; US Treasury 20+Y (TLT) &nbsp;&middot;&nbsp; HMM 2 &#233;tats + seuil ADX 25 + DI+/DI- + BOCPD</p>

<div class="tabbar">{tab_buttons}</div>

{asset_panels}

<div class="tab-panel" data-tab="COMPARISON">
  <div class="card">
    <div class="card-label">Largeur des r&#233;gimes par actif (jours calendaires, box plot)</div>
    <div id="chart-box" style="height:340px"></div>
  </div>

  <div class="card">
    <div class="card-label">La volatilit&#233; annonce-t-elle ou confirme-t-elle un passage en stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
    <p class="chart-note">Volatilit&#233; moyenne autour de chaque entr&#233;e en r&#233;gime stress (jour 0 = jour du
      basculement), index&#233;e sur la p&#233;riode -10&#224;-5 jours = 0% pour rester comparable entre crypto et
      obligations. Un losange marque, pour chaque actif, le premier jour o&#249; l'&#233;cart &#224; la p&#233;riode
      pr&#233;-&#233;v&#233;nement d&#233;passe 1,96&#215; l'erreur standard de la p&#233;riode de r&#233;f&#233;rence (seuil &#224; 95%)
      &#8212; indicatif, pas un test statistique formel.</p>
    <div id="chart-event-study" style="height:380px"></div>
    <details style="margin-top:8px">
      <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
      <div style="overflow-x:auto;margin-top:6px">
        <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n transitions vers stress</th></tr></thead>
        <tbody id="event-study-table"></tbody></table>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="card-label">Test formel &#8212; la volatilit&#233; pass&#233;e pr&#233;dit-elle le r&#233;gime stress futur&nbsp;? (causalit&#233; de Granger)</div>
    <p class="chart-note">Test de Granger (&#963;<sub>t</sub> retard&#233; am&#233;liore-t-il la pr&#233;diction de p_stress,
      au-del&#224; de sa propre persistance&nbsp;?), pr&#233;c&#233;d&#233; d'un test ADF de stationnarit&#233; (diff&#233;renciation
      si n&#233;cessaire) &#8212; sinon les p-values ne seraient pas interpr&#233;tables. 10 lags test&#233;s par actif :
      seuil de significativit&#233; corrig&#233; pour comparaisons multiples (Bonferroni, &#945;=0.005 au lieu de
      0.05) pour &#233;viter de conclure sur un franchissement isol&#233; d&#251; au hasard. Case color&#233;e = p-value
      sous ce seuil.</p>
    <div id="chart-granger" style="height:260px"></div>
    <p class="chart-note" id="granger-verdict"></p>
    <details style="margin-top:8px">
      <summary>Voir le d&#233;tail par actif (stationnarit&#233; ADF, p-value minimale)</summary>
      <div style="overflow-x:auto;margin-top:6px">
        <table><thead><tr><th>Actif</th><th>ADF &#963;<sub>t</sub> (p)</th><th>ADF p_stress (p)</th><th>p-value min (lag)</th><th>Verdict</th></tr></thead>
        <tbody id="granger-table"></tbody></table>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="card-label">Le volume annonce-t-il ou confirme-t-il un passage en stress&nbsp;? (&#233;tude d'&#233;v&#233;nement)</div>
    <p class="chart-note">Volume normalis&#233; moyen (ratio &#224; la moyenne 30j) autour de chaque entr&#233;e en
      r&#233;gime stress (jour 0 = jour du basculement), index&#233; sur la p&#233;riode -10&#224;-5 jours = 0% (m&#234;me
      m&#233;thode que pour la volatilit&#233; ci-dessus). Un losange marque, pour chaque actif, le premier jour
      o&#249; l'&#233;cart &#224; la p&#233;riode pr&#233;-&#233;v&#233;nement d&#233;passe 1,96&#215; l'erreur standard de la p&#233;riode de
      r&#233;f&#233;rence (seuil &#224; 95%) &#8212; indicatif, pas un test statistique formel.</p>
    <div id="chart-event-study-volume" style="height:380px"></div>
    <details style="margin-top:8px">
      <summary>Voir le d&#233;tail par actif (jour de premi&#232;re r&#233;action, nombre d'&#233;v&#233;nements)</summary>
      <div style="overflow-x:auto;margin-top:6px">
        <table><thead><tr><th>Actif</th><th>Premier jour de r&#233;action</th><th>n transitions vers stress</th></tr></thead>
        <tbody id="event-study-volume-table"></tbody></table>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="card-label">Test formel &#8212; le volume pass&#233; pr&#233;dit-il le r&#233;gime stress futur&nbsp;? (causalit&#233; de Granger)</div>
    <p class="chart-note">M&#234;me test que pour la volatilit&#233; ci-dessus (ADF + Granger + correction de
      Bonferroni, &#945;=0.005), appliqu&#233; &#224; volume_norm au lieu de &#963;<sub>t</sub> : le volume retard&#233;
      am&#233;liore-t-il la pr&#233;diction de p_stress, au-del&#224; de sa propre persistance&nbsp;? Utilit&#233; :
      savoir si le volume est un signal pr&#233;curseur ind&#233;pendant de la volatilit&#233;, ou s'il ne fait
      que confirmer lui aussi.</p>
    <div id="chart-granger-volume" style="height:260px"></div>
    <p class="chart-note" id="granger-volume-verdict"></p>
    <details style="margin-top:8px">
      <summary>Voir le d&#233;tail par actif (stationnarit&#233; ADF, p-value minimale)</summary>
      <div style="overflow-x:auto;margin-top:6px">
        <table><thead><tr><th>Actif</th><th>ADF volume (p)</th><th>ADF p_stress (p)</th><th>p-value min (lag)</th><th>Verdict</th></tr></thead>
        <tbody id="granger-volume-table"></tbody></table>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="card-label">Entre volatilit&#233; et volume, lequel bouge en premier en g&#233;n&#233;ral&nbsp;? (corr&#233;lation crois&#233;e &#177;5j)</div>
    <p class="chart-note">Corr&#233;lation entre les VARIATIONS journali&#232;res de &#963;<sub>t</sub> et de volume_norm
      (pas les niveaux, v&#233;rifi&#233;es stationnaires par test ADF), sur tout l'historique de chaque actif
      &#8212; pas seulement autour des d&#233;buts de r&#233;gime (question distincte des tests de Granger
      ci-dessus). Lag &gt; 0 (&#224; droite) : le volume pass&#233; est li&#233; &#224; la vol d'aujourd'hui &#8212; le volume
      pr&#233;c&#232;de. Lag &lt; 0 (&#224; gauche) : la vol d'aujourd'hui est li&#233;e au volume futur &#8212; la vol
      pr&#233;c&#232;de. Lag 0 : corr&#233;lation le m&#234;me jour. Seuil de significativit&#233; corrig&#233; pour
      comparaisons multiples (Bonferroni sur les 11 lags scann&#233;s, comme pour les tests de Granger).</p>
    <div id="chart-vol-volume-ccf" style="height:320px"></div>
    <p class="chart-note" id="vol-volume-ccf-verdict"></p>
    <details style="margin-top:8px">
      <summary>Voir le d&#233;tail par actif (stationnarit&#233; ADF, lag du pic, corr&#233;lation, seuil de Fisher corrig&#233;)</summary>
      <div style="overflow-x:auto;margin-top:6px">
        <table><thead><tr><th>Actif</th><th>ADF &#916;&#963; (p)</th><th>ADF &#916;volume (p)</th><th>Lag du pic</th><th>Corr&#233;lation au pic</th><th>Seuil Fisher (Bonferroni)</th><th>Qui pr&#233;c&#232;de&nbsp;?</th></tr></thead>
        <tbody id="vol-volume-ccf-table"></tbody></table>
      </div>
    </details>
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
// Ligne de r&#233;f&#233;rence y=1 du panneau volume (volume_norm est un ratio &#224; sa propre moyenne 30j) —
// constante partag&#233;e entre initAssetTab (trac&#233; initial) et refreshTab (qui recalcule les fonds
// de r&#233;gime via Plotly.relayout({{shapes}}) et &#233;craserait cette ligne si elle n'&#233;tait pas
// r&#233;-ajout&#233;e &#224; chaque refresh).
const VOLUME_REF_LINE = {{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:1,y1:1,
  line:{{color:'#2c3e50',width:1,dash:'dot'}}}};

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
    state:{{regimes:{{calm:true,bull:true,bear:true,stress:true}}, cats:{{crypto:true,macro:true,monetaire:true,geopolitique:true}}, scale:'annee'}},
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
  // chart-volume a en plus sa ligne de r&#233;f&#233;rence y=1, propre &#224; ce panneau (cf. VOLUME_REF_LINE) —
  // il faut la r&#233;-ajouter ici, sinon ce relayout(shapes) la remplacerait par les seuls fonds de r&#233;gime.
  Plotly.relayout(`chart-volume-${{SHORT_OF[tabId]}}`, {{shapes: shapes.concat([VOLUME_REF_LINE])}});
  const {{annotations, overflow}} = buildAnnotations(tabId);
  Plotly.relayout(`chart-price-${{SHORT_OF[tabId]}}`, {{annotations}});
  const msg = document.getElementById(`declutter-${{SHORT_OF[tabId]}}`);
  if (msg) msg.style.display = overflow ? 'block' : 'none';
}}

// ── Composition des r&#233;gimes : donut recalcul&#233; sur la fen&#234;tre visible ──────────────
function updateComposition(tabId) {{
  const d = TAB_DATA[tabId], range = TABS[tabId].currentXRange;
  let nCalm = 0, nBull = 0, nBear = 0, nStress = 0, total = 0;
  for (let i = 0; i < d.dates.length; i++) {{
    if (d.dates[i] >= range[0] && d.dates[i] <= range[1]) {{
      total++;
      if (d.regimes[i] === 'calm') nCalm++;
      else if (d.regimes[i] === 'bull') nBull++;
      else if (d.regimes[i] === 'bear') nBear++;
      else nStress++;
    }}
  }}
  const pct = v => total ? v / total * 100 : 0;
  const trace = {{
    type:'pie', labels:['Calme','Haussier','Baissier','Stress'],
    values:[pct(nCalm), pct(nBull), pct(nBear), pct(nStress)],
    marker:{{colors:[REGIME_HEX.calm, REGIME_HEX.bull, REGIME_HEX.bear, REGIME_HEX.stress]}},
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
  ['price','vol','cp','volume'].forEach(k => Plotly.relayout(`chart-${{k}}-${{SHORT_OF[tabId]}}`, {{
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
  const regimeTraces = ['calm','bull','bear','stress'].map(r => ({{
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

  // showlegend:false volontaire : seule trace du panneau (le titre de la carte suffit à la
  // labelliser), une légende cliquable permettrait de la masquer par un clic accidentel — le
  // panneau doit rester affiché en permanence, sans mécanisme de désélection.
  const volumeTrace = {{type:'bar',x:d.dates,y:d.volume_norm,
    marker:{{color:'#7f8c8d'}},name:'Volume (norm.)',showlegend:false}};
  // xaxis.range forcé explicitement à [first_date, last_date] : un graphique en barres réserve
  // par défaut une demi-largeur de barre de marge de chaque côté (contrairement aux graphiques en
  // lignes prix/vol/BOCPD), ce qui décale légèrement son échelle au premier rendu (avant tout
  // zoom) si on laisse Plotly l'autodéterminer — vérifié : ~12h de décalage sur chaque bord.
  // margin.r:144 (au lieu de 18) : avec showlegend:false, la zone de tracé se serait sinon
  // élargie pour occuper l'espace auparavant réservé à la légende (mesuré à ~144px avant sa
  // suppression), décalant horizontalement ce panneau par rapport à vol/prix/BOCPD juste
  // au-dessus/en-dessous — la marge est conservée à l'identique, seule la légende disparaît.
  Plotly.newPlot(`chart-volume-${{SHORT_OF[tabId]}}`, [volumeTrace],
    Object.assign({{}},baseLayout(),{{margin:{{l:60,r:144,t:8,b:38}},shapes:shapes.concat([VOLUME_REF_LINE]),
      xaxis:Object.assign({{}},baseLayout().xaxis,{{range:[d.first_date,d.last_date]}}),
      yaxis:{{title:'x moyenne 30j',gridcolor:GRID}}}}), {{responsive:true,displayModeBar:false}});

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
      Plotly.relayout(`chart-volume-${{SHORT_OF[tabId]}}`, {{'xaxis.range[0]':r0,'xaxis.range[1]':r1}});
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
    xaxis:{{type:'category',categoryarray:['calm','bull','bear','stress'],
      ticktext:['Calme','Haussier','Baissier','Stress'],tickvals:['calm','bull','bear','stress']}},
    yaxis:{{title:'Dur&#233;e (jours calendaires)',gridcolor:GRID}},
  }}), {{responsive:true,displayModeBar:false}});

  // ── Étude d'événement : profil indexé autour de l'entrée en stress (Q1 vol, Q2 volume) ──
  function renderEventStudy(data, chartId, tableId) {{
    const traces = Object.keys(data).map(short => {{
      const a = data[short];
      return {{
        type: 'scatter', mode: 'lines', name: short,
        x: a.rel_day, y: a.index_pct,
        line: {{ color: a.color, width: 2 }},
        legendgroup: short,
        hovertemplate: `${{short}} : %{{y:.1f}}%<extra></extra>`,
      }};
    }});
    const markers = Object.keys(data).map(short => {{
      const a = data[short];
      const y = a.rel_day.map(d => d === a.first_reaction_day ? a.index_pct[a.rel_day.indexOf(d)] : null);
      return {{
        type: 'scatter', mode: 'markers', name: short, legendgroup: short, showlegend: false,
        x: a.rel_day, y: y,
        marker: {{ color: a.color, size: 11, symbol: 'diamond', line: {{ color: '#fff', width: 1 }} }},
        hovertemplate: `${{short}} : premi&#232;re r&#233;action au jour %{{x}}<extra></extra>`,
      }};
    }});
    Plotly.newPlot(chartId, [...traces, ...markers], Object.assign({{}}, baseLayout(), {{
      margin: {{ l: 55, r: 18, t: 10, b: 45 }},
      xaxis: {{ title: 'Jours relatifs au d&#233;but du r&#233;gime stress (0 = jour du basculement)',
                gridcolor: GRID, dtick: 1, zeroline: false }},
      yaxis: {{ title: '&#201;cart vs p&#233;riode pr&#233;-&#233;v&#233;nement (%)',
                gridcolor: GRID, zeroline: true, zerolinewidth: 2, zerolinecolor: '#566573' }},
      shapes: [{{ type: 'line', x0: 0, x1: 0, xref: 'x', y0: 0, y1: 1, yref: 'paper',
                 line: {{ color: '#7f8c8d', width: 1, dash: 'dash' }} }}],
    }}), {{ responsive: true, displayModeBar: false }});

    const body = document.getElementById(tableId);
    Object.keys(data).forEach(short => {{
      const a = data[short];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{short}}</td>` +
        `<td>${{a.first_reaction_day !== null ? (a.first_reaction_day >= 0 ? '+' : '') + a.first_reaction_day + ' j' : 'aucun &#233;cart significatif d&#233;tect&#233;'}}</td>` +
        `<td>${{a.n_events}}</td>`;
      body.appendChild(tr);
    }});
  }}

  // ── Test formel : causalité de Granger (Q1 vol -> régime stress, Q2 volume -> régime stress) ──
  function renderGrangerCard(grData, alpha, chartId, tableId, verdictId, sourceLabel) {{
    const shorts = Object.keys(grData);
    const lags = Object.keys(grData[shorts[0]].p_values).map(Number).sort((a,b) => a-b);

    const zP = shorts.map(short => lags.map(lag => grData[short].p_values[lag]));
    const zSig = zP.map(row => row.map(p => p < alpha ? 1 : 0));
    const textVals = zP.map(row => row.map(p => p < alpha ? `${{p.toFixed(3)}}*` : p.toFixed(3)));

    Plotly.newPlot(chartId, [{{
      type: 'heatmap',
      x: lags.map(l => `${{l}}`),
      y: shorts,
      z: zSig,
      customdata: zP,
      text: textVals,
      texttemplate: '%{{text}}',
      textfont: {{ size: 10, color: '#ecf0f1' }},
      colorscale: [[0, BG], [1, '#e74c3c']],
      zmin: 0, zmax: 1,
      showscale: false,
      xgap: 3, ygap: 3,
      hovertemplate: 'lag %{{x}} : p=%{{customdata:.4f}}<extra>%{{y}}</extra>',
    }}], Object.assign({{}}, baseLayout(), {{
      margin: {{ l: 55, r: 18, t: 10, b: 40 }},
      xaxis: {{ title: 'Lag (jours)', gridcolor: GRID, type: 'category' }},
      yaxis: {{ gridcolor: GRID, automargin: true }},
    }}), {{ responsive: true, displayModeBar: false }});

    const body = document.getElementById(tableId);
    shorts.forEach(short => {{
      const g = grData[short];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{short}}</td>` +
        `<td>${{g.adf_source_p.toFixed(4)}}${{g.source_differenced ? ' (diff.)' : ''}}</td>` +
        `<td>${{g.adf_pstress_p.toFixed(4)}}${{g.pstress_differenced ? ' (diff.)' : ''}}</td>` +
        `<td>${{g.min_p.toFixed(4)}} (lag ${{g.min_p_lag}})</td>` +
        `<td>${{g.significant ? '<b style="color:#e74c3c">causal (Granger)</b>' : 'non significatif'}}</td>`;
      body.appendChild(tr);
    }});

    const sigAssets = shorts.filter(short => grData[short].significant);
    document.getElementById(verdictId).innerHTML = sigAssets.length > 0
      ? `<b>${{sigAssets.join(', ')}}</b> : ${{sourceLabel}} pr&#233;dit significativement (Granger, seuil corrig&#233; &#945;=${{alpha}}) le r&#233;gime stress futur. Pour les autres actifs, le test ne rejette pas H0 &#8212; pas de preuve d'un effet pr&#233;dictif, au-del&#224; de la propre persistance du r&#233;gime.`
      : `Aucun actif ne montre de lien de Granger significatif (seuil corrig&#233; &#945;=${{alpha}}) entre ${{sourceLabel}} et r&#233;gime stress futur.`;
  }}

  // ── Question 3 : entre vol et volume, qui bouge en premier en général ? ────────
  function renderVolVolumeCcf(ccfData, chartId, tableId, verdictId) {{
    const shorts = Object.keys(ccfData);
    const traces = shorts.map(short => {{
      const a = ccfData[short];
      return {{
        type: 'scatter', mode: 'lines+markers', name: short,
        x: a.lags, y: a.corr,
        line: {{ color: a.color, width: 2 }}, marker: {{ size: 4 }},
        hovertemplate: `${{short}} : %{{y:.3f}}<extra></extra>`,
      }};
    }});
    Plotly.newPlot(chartId, traces, Object.assign({{}}, baseLayout(), {{
      margin: {{ l: 55, r: 18, t: 30, b: 45 }},
      xaxis: {{ title: 'Lag (jours de trading)', gridcolor: GRID, dtick: 1, zeroline: false }},
      yaxis: {{ title: 'Corr&#233;lation (variations journali&#232;res)', gridcolor: GRID,
                zeroline: true, zerolinewidth: 2, zerolinecolor: '#566573' }},
      shapes: [{{ type: 'line', x0: 0, x1: 0, xref: 'x', y0: 0, y1: 1, yref: 'paper',
                 line: {{ color: '#7f8c8d', width: 1, dash: 'dash' }} }}],
      annotations: [
        {{ x: 1, y: 1.1, xref: 'paper', yref: 'paper', xanchor: 'right', showarrow: false,
          text: 'volume pr&#233;c&#232;de &#8594;', font: {{ size: 10, color: '#7f8c8d' }} }},
        {{ x: 0, y: 1.1, xref: 'paper', yref: 'paper', xanchor: 'left', showarrow: false,
          text: '&#8592; vol pr&#233;c&#232;de', font: {{ size: 10, color: '#7f8c8d' }} }},
      ],
    }}), {{ responsive: true, displayModeBar: false }});

    const body = document.getElementById(tableId);
    const leaderLabel = {{ volume: 'volume', vol: 'volatilit&#233;', aucun: 'contemporain (aucun)' }};
    shorts.forEach(short => {{
      const a = ccfData[short];
      const rcrit = a.r_crit[a.lags.indexOf(a.peak_lag)];
      const isSig = rcrit !== null && Math.abs(a.peak_corr) > rcrit;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{short}}</td>` +
        `<td>${{a.adf_a_p.toFixed(4)}}${{a.a_differenced ? ' (diff.)' : ''}}</td>` +
        `<td>${{a.adf_b_p.toFixed(4)}}${{a.b_differenced ? ' (diff.)' : ''}}</td>` +
        `<td>${{a.peak_lag >= 0 ? '+' : ''}}${{a.peak_lag}}</td>` +
        `<td>${{a.peak_corr.toFixed(3)}}${{isSig ? '<b style="color:#f39c12">*</b>' : ''}}</td>` +
        `<td>${{rcrit !== null ? rcrit.toFixed(3) : '&#8212;'}}</td>` +
        `<td>${{leaderLabel[a.leader]}}</td>`;
      body.appendChild(tr);
    }});

    const leaders = shorts.map(short => ccfData[short].leader);
    const nVolume = leaders.filter(l => l === 'volume').length;
    const nVol = leaders.filter(l => l === 'vol').length;
    document.getElementById(verdictId).innerHTML =
      `Sur ${{shorts.length}} actifs, le pic de corr&#233;lation se situe c&#244;t&#233; <b>volume pr&#233;c&#232;de</b> pour ${{nVolume}} actif(s), c&#244;t&#233; <b>vol pr&#233;c&#232;de</b> pour ${{nVol}} actif(s) (* = corr&#233;lation significative au seuil de Fisher corrig&#233; pour comparaisons multiples, Bonferroni sur 11 lags).`;
  }}

  renderEventStudy(COMPARISON.event_study, 'chart-event-study', 'event-study-table');
  renderEventStudy(COMPARISON.event_study_volume, 'chart-event-study-volume', 'event-study-volume-table');
  renderGrangerCard(COMPARISON.granger, COMPARISON.granger_alpha,
    'chart-granger', 'granger-table', 'granger-verdict', 'la volatilit&#233; pass&#233;e');
  renderGrangerCard(COMPARISON.granger_volume, COMPARISON.granger_alpha,
    'chart-granger-volume', 'granger-volume-table', 'granger-volume-verdict', 'le volume pass&#233;');
  renderVolVolumeCcf(COMPARISON.vol_volume_ccf, 'chart-vol-volume-ccf', 'vol-volume-ccf-table', 'vol-volume-ccf-verdict');
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
    <div class="li"><input type="checkbox" class="regime-cb-{dom}" value="bull" checked><div class="dot" style="background:{_REGIME_HEX['bull']}"></div>Haussier</div>
    <div class="li"><input type="checkbox" class="regime-cb-{dom}" value="bear" checked><div class="dot" style="background:{_REGIME_HEX['bear']}"></div>Baissier</div>
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
    <div class="card-label">Volume normalis&#233; (ratio &#224; la moyenne 30j)</div>
    <div id="chart-volume-{dom}" style="height:120px"></div>
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
