"""
DEITA Benchmark 1.6  —  ARIMA-OLS vs SARIMA-OLS  (mensuel / journalier)
========================================================================
Modifiez le bloc CONFIGURATION ci-dessous, puis lancez :
  python "Benchmark 1.6.py"

Comparaison :
  ARIMA-OLS  : diff reguliere d=1, AR-OLS sur lags courts  (sans saisonnalite)
  SARIMA-OLS : double diff d=1 + D=1(s), AR-OLS sur lags [1,s,s+1]  (saisonnier)
  => mesure l'apport reel de la composante saisonniere sur les cryptos

3 fenetres interactives + PNG :
  Fig 1 — Vue globale + zoom centree sur la periode de test
  Fig 2 — Previsions M+1..M+N (ou J+1..J+N) avec contexte recent
  Fig 3 — Comparaison : KPIs + erreurs + OHLC (3 agregations)
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Modifier ici avant de lancer
# ═══════════════════════════════════════════════════════════════════════════════

TICKER         = "ETH-USD"          # Symbole yfinance : ETH-USD, BTC-USD, AAPL...
FREQ           = "M"                # "M" = mensuel  |  "J" = journalier
START          = "auto"             # "auto" = recommande  |  ou "2017-01-01"
END            = "today"            # "today"  |  ou "2026-06-30"
TEST_RATIO     = 0.15               # Fraction test (0.15 = 15 %)
ACTIVE_MODELS  = ["arima", "sarima"]  # ["arima"]  |  ["sarima"]  |  les deux
FORECAST_STEPS = 3                  # Nombre de pas a predire (M+1..M+N)
SEED           = 42                 # Reproductibilite
SHOW_PI        = True               # Afficher les intervalles de prediction 95 %
SAVE_PNG       = True               # Sauvegarder le dashboard PNG

# ═══════════════════════════════════════════════════════════════════════════════

import math, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

try:
    import certifi as _cert; _ca = _cert.where()
    for _k in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ.setdefault(_k, _ca)
except ImportError:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

# ── Frequences disponibles ────────────────────────────────────────────────────
# p_ar : ordre AR pour ARIMA (lags de la diff simple)
# s    : periode saisonniere pour SARIMA

FREQ_CONFIG = {
    "M": dict(interval="1mo", label="mensuelle",  start_default="2017-01-01",
              s=12, p_ar=3,  zoom_ctx=6,  step_lbl="M", date_fmt="%Y-%m"),
    "J": dict(interval="1d",  label="journaliere", start_default="2020-01-01",
              s=5,  p_ar=5,  zoom_ctx=7,  step_lbl="J", date_fmt="%Y-%m-%d"),
}

# ── Apparence des modeles ─────────────────────────────────────────────────────

MODEL_LABELS = {"arima": "ARIMA-OLS",  "sarima": "SARIMA-OLS"}
MODEL_COLORS = {"arima": "#e74c3c",    "sarima": "#2980b9"}    # rouge / bleu
MODEL_LS     = {"arima": "--",         "sarima": "-"}           # tirete / plein
MODEL_HATCH  = {"arima": "///",        "sarima": ""}            # hachures / plein

# ── Evenements de marche ──────────────────────────────────────────────────────

MARKET_EVENTS = {
    "2017-11-29": ("BTC ATH $10k",       "crypto"),
    "2018-01-17": ("BTC ATH $20k",       "crypto"),
    "2018-12-15": ("BTC bas $3.2k",      "crypto"),
    "2020-03-12": ("COVID crash",         "macro"),
    "2020-03-15": ("Fed taux 0%",        "monetaire"),
    "2020-05-11": ("BTC halving #3",      "crypto"),
    "2020-12-16": ("BTC franchit $20k",   "crypto"),
    "2021-02-08": ("Tesla 1.5G$ BTC",    "crypto"),
    "2021-04-14": ("Coinbase IPO",        "crypto"),
    "2021-09-07": ("El Salvador BTC",     "geopolitique"),
    "2021-11-10": ("BTC ATH $69k",       "crypto"),
    "2022-01-05": ("Fed pivot hawkish",   "monetaire"),
    "2022-02-24": ("Invasion Ukraine",    "geopolitique"),
    "2022-05-09": ("LUNA collapse",       "crypto"),
    "2022-06-15": ("Fed +75bp",          "monetaire"),
    "2022-09-15": ("ETH Merge PoS",       "crypto"),
    "2022-11-08": ("FTX collapse",        "crypto"),
    "2023-03-10": ("SVB faillite",        "macro"),
    "2023-07-26": ("Fed pic 5.25%",      "monetaire"),
    "2024-01-10": ("BTC ETF spot",        "crypto"),
    "2024-03-14": ("BTC ATH $73k",       "crypto"),
    "2024-04-19": ("BTC halving #4",      "crypto"),
    "2025-01-23": ("BTC ATH $109k",      "crypto"),
}
EVENT_COLORS = {"crypto":"#e67e22","macro":"#e74c3c",
                "monetaire":"#2980b9","geopolitique":"#8e44ad"}
EVENT_LABELS = {"crypto":"Crypto","macro":"Macro",
                "monetaire":"Monet.","geopolitique":"Geopo."}

# ── ANSI terminal ─────────────────────────────────────────────────────────────

_GR,_RE,_YL,_BO,_RS = "\033[92m","\033[91m","\033[93m","\033[1m","\033[0m"
_USE_COLOR = True
import re as _re; _ANSI = _re.compile(r'\033\[[0-9;]*m')
def _c(t, code): return f"{code}{t}{_RS}" if _USE_COLOR else t
def _vl(s):      return len(_ANSI.sub("", s))
def _rp(s, w):   return " "*max(0, w-_vl(s)) + s
def _lp(s, w):   return s + " "*max(0, w-_vl(s))

# ── Loi normale sans scipy (_stats_pythran bloque par AppLocker) ──────────────

def _ncdf(x):
    v = np.asarray(x, float)
    return np.vectorize(lambda xi: 0.5*(1 + math.erf(xi/math.sqrt(2))))(v)
def _npdf(x):
    return np.exp(-0.5*np.asarray(x, float)**2) / math.sqrt(2*math.pi)

# ── Session SSL (curl_cffi) ───────────────────────────────────────────────────

def _ssl_session():
    try:
        from curl_cffi.requests import Session
        return Session(impersonate="chrome", verify=False)
    except Exception:
        return None

# ── Donnees ───────────────────────────────────────────────────────────────────

def fetch_prices(ticker, start, end, cfg):
    print(f"[DATA] {ticker}  [{start} -> {end}]  {cfg['label']} ...")
    sess = _ssl_session()
    raw  = yf.download(ticker, start=start, end=end, interval=cfg["interval"],
                       progress=False, auto_adjust=True,
                       **({"session": sess} if sess else {}))
    if raw.empty: sys.exit(f"[ERR] Aucune donnee pour {ticker}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    c = (pd.to_numeric(raw["Close"], errors="coerce")
           .replace([np.inf, -np.inf], np.nan).dropna())
    c.index = pd.DatetimeIndex(c.index).tz_localize(None)
    print(f"[DATA] {len(c)} observations chargees.\n")
    return c.astype(float)

def split_series(prices, ratio):
    k = int(len(prices) * (1 - ratio))
    return prices.iloc[:k], prices.iloc[k:]

def _future_dates(last_d, n, cfg):
    dates = []; d = last_d
    for _ in range(n):
        d = (d + pd.DateOffset(months=1) if cfg["interval"] == "1mo"
             else d + pd.tseries.offsets.BDay(1))
        dates.append(d)
    return dates

# ── Naif ──────────────────────────────────────────────────────────────────────

def compute_naive(train, test):
    a   = test.values.astype(float)
    p   = np.concatenate([[train.iloc[-1]], a[:-1]])
    sig = float(np.std(np.diff(train.values)))
    return {"RMSE": round(float(np.sqrt(np.mean((a-p)**2))), 4),
            "MAE":  round(float(np.mean(np.abs(a-p))), 4),
            "MAPE (%)": round(float(np.mean(np.abs((a-p)/(a+1e-8)))*100), 2),
            "Dir. Acc (%)": round(float(np.mean(
                np.sign(np.diff(a)) == np.sign(np.diff(p)))*100), 2),
            "Train Time (s)": 0.0,
            "predictions": p, "lower": p-1.96*sig, "upper": p+1.96*sig,
            "index": test.index, "actual": a}

# ── Metriques ─────────────────────────────────────────────────────────────────

def crps_g(preds, lo, hi, actual, z=1.96):
    mu, l, h, y = (np.asarray(x, float) for x in (preds, lo, hi, actual))
    sig = np.where((h-l) > 0, (h-l)/(2*z), 1e-8); zs = (y-mu)/sig
    return round(float(np.mean(sig*(zs*(2*_ncdf(zs)-1) + 2*_npdf(zs) - 1/np.sqrt(np.pi)))), 4)

def cov95(a, lo, hi):
    a, lo, hi = map(np.asarray, (a, lo, hi))
    return round(float(np.mean((a >= lo) & (a <= hi))*100), 2)

def winkler(a, lo, hi, alpha=0.05):
    a, lo, hi = map(np.asarray, (a, lo, hi))
    return round(float(np.mean((hi-lo) + np.where(a < lo, (2/alpha)*(lo-a), 0.)
                                       + np.where(a > hi, (2/alpha)*(a-hi), 0.))), 2)

def dm_test(ea, eb):
    d = np.asarray(ea, float)**2 - np.asarray(eb, float)**2; T = len(d)
    if T < 5: return 0., 1.
    db = np.mean(d); tr = max(1, int(T**(1/3))); vd = np.var(d, ddof=0)
    for j in range(1, tr+1):
        vd += 2*(1 - j/(tr+1))*np.mean((d[j:]-db)*(d[:-j]-db))
    if vd <= 0: return 0., 1.
    DM = db / np.sqrt(max(vd, 1e-12)/T)
    p  = 2*(1 - float(_ncdf(abs(DM))))
    return round(float(DM), 3), round(float(p), 3)

def _build_result(actual, preds, lo, hi, index, elapsed):
    a, p, l, h = (np.asarray(x, float) for x in (actual, preds, lo, hi))
    rmse = float(np.sqrt(np.mean((a-p)**2))); mae = float(np.mean(np.abs(a-p)))
    mape = float(np.mean(np.abs((a-p)/(a+1e-8)))*100)
    da   = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p)))*100) if len(a)>1 else 0.
    return {"RMSE": round(rmse,4), "MAE": round(mae,4), "MAPE (%)": round(mape,2),
            "Dir. Acc (%)": round(da,2), "CRPS": crps_g(p,l,h,a),
            "Train Time (s)": round(elapsed,2),
            "predictions": p, "lower": l, "upper": h, "index": index, "actual": a}

# ── ARIMA-OLS (pure numpy) ────────────────────────────────────────────────────
#
# ARIMA(p,1,0) :
#   1) Diff simple : z = diff(y)
#   2) AR-OLS sur lags [1..p] de z
#   3) Inversion : y_next = y[-1] + z_next
#   4) PI = 1.96 * sigma_residus * 1.5

def _arima_step(y, p=3):
    y = np.asarray(y, float); n = len(y)

    def _fallback():
        last = float(y[-1])
        vol  = float(np.std(np.diff(y[-min(p+3, n-1):]))) if n > 2 else last*0.1
        return last, last-1.96*vol, last+1.96*vol

    if n < p + 3: return _fallback()
    z = np.diff(y)
    if len(z) <= p + 1: return _fallback()

    lags = list(range(1, p+1))
    X = np.column_stack([z[p-g: len(z)-g] for g in lags]); Y = z[p:]
    try:
        theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return _fallback()

    resid  = Y - X @ theta
    sigma  = max(float(np.std(resid, ddof=min(p, len(resid)-1))), 10.)
    z_next = float(theta @ np.array([z[-g] for g in lags]))
    y_pred = float(y[-1]) + z_next
    pi     = 1.96 * sigma * 1.5
    return y_pred, y_pred - pi, y_pred + pi


def run_arima(train, test, cfg):
    t0 = time.time(); p = cfg["p_ar"]
    hist = list(train.astype(float).values); pred_l, lo_l, hi_l = [], [], []
    for i in range(len(test)):
        pred, lo, hi = _arima_step(np.array(hist), p)
        pred_l.append(pred); lo_l.append(lo); hi_l.append(hi)
        hist.append(float(test.iloc[i]))
        print(f"  ARIMA : {i+1}/{len(test)}", end="\r", flush=True)
    print()
    return _build_result(test.values.astype(float), pred_l, lo_l, hi_l,
                         test.index, time.time()-t0)

def arima_forecast(series, n, cfg):
    hist = list(series.astype(float).values); steps = []; p = cfg["p_ar"]
    for i in range(n):
        pred, lo, hi = _arima_step(np.array(hist), p)
        steps.append((pred, lo, hi)); hist.append(pred)
        print(f"  ARIMA next : {cfg['step_lbl']}+{i+1}", end="\r", flush=True)
    print(); return steps

# ── SARIMA-OLS (pure numpy) ───────────────────────────────────────────────────
#
# Approximation SARIMA(1,1,1)(1,1,1)[s] :
#   1) Double diff : D=1 saisonniere (lag s) + d=1 reguliere => serie z
#   2) AR-OLS sur lags [1, s, s+1] de z
#   3) Inversion : y_next = y[-s] + w[-1] + z_next
#   4) PI = 1.96 * sigma_residus * 1.5

def _sarima_step(y, s=12):
    y = np.asarray(y, float); n = len(y); lags = [1, s, s+1]; lmax = s+1

    def _fallback():
        last = float(y[-1])
        vol  = float(np.std(np.diff(y[-min(lmax+2, n-1):]))) if n > 2 else last*0.1
        return last, last-1.96*vol, last+1.96*vol

    if n < 2*s + lmax + 2: return _fallback()
    w = y[s:] - y[:-s]; z = np.diff(w)
    if len(z) <= lmax + 2: return _fallback()

    X = np.column_stack([z[lmax-g: len(z)-g] for g in lags]); Y = z[lmax:]
    try:
        theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return _fallback()

    resid  = Y - X @ theta
    sigma  = max(float(np.std(resid, ddof=min(len(lags), len(resid)-1))), 10.)
    z_next = float(theta @ np.array([z[-g] for g in lags]))
    y_pred = float(y[-s]) + float(w[-1]) + z_next
    pi     = 1.96 * sigma * 1.5
    return y_pred, y_pred - pi, y_pred + pi


def run_sarima(train, test, cfg):
    t0 = time.time(); s = cfg["s"]
    hist = list(train.astype(float).values); pred_l, lo_l, hi_l = [], [], []
    for i in range(len(test)):
        pred, lo, hi = _sarima_step(np.array(hist), s)
        pred_l.append(pred); lo_l.append(lo); hi_l.append(hi)
        hist.append(float(test.iloc[i]))
        print(f"  SARIMA : {i+1}/{len(test)}", end="\r", flush=True)
    print()
    return _build_result(test.values.astype(float), pred_l, lo_l, hi_l,
                         test.index, time.time()-t0)

def sarima_forecast(series, n, cfg):
    hist = list(series.astype(float).values); steps = []
    for i in range(n):
        pred, lo, hi = _sarima_step(np.array(hist), cfg["s"])
        steps.append((pred, lo, hi)); hist.append(pred)
        print(f"  SARIMA next : {cfg['step_lbl']}+{i+1}", end="\r", flush=True)
    print(); return steps

# ── Runners unifies ───────────────────────────────────────────────────────────

def run_all(train, test, models, cfg, seed):
    np.random.seed(seed); results = {}
    for k in models:
        print(f"[RUN] {MODEL_LABELS[k]} ...")
        try:
            res = run_arima(train, test, cfg) if k == "arima" else run_sarima(train, test, cfg)
            results[k] = res
            print(f"[RUN] {MODEL_LABELS[k]} | RMSE={res['RMSE']}  "
                  f"CRPS={res['CRPS']}  Dir.Acc={res['Dir. Acc (%)']:.1f}%\n")
        except Exception as e:
            print(f"[ERR] {MODEL_LABELS[k]} : {e}")
    return results

def forecast_all(series, models, n, cfg, seed):
    np.random.seed(seed); preds = {}
    for k in models:
        print(f"[NEXT] {MODEL_LABELS[k]} ({n} pas) ...")
        try:
            steps = arima_forecast(series, n, cfg) if k == "arima" else sarima_forecast(series, n, cfg)
            preds[k] = steps
            print(f"[NEXT] {MODEL_LABELS[k]}: " +
                  " | ".join(f"{cfg['step_lbl']}+{i+1}: {p:.0f}$ [{l:.0f},{h:.0f}]"
                              for i,(p,l,h) in enumerate(steps)))
        except Exception as e:
            print(f"[ERR] {MODEL_LABELS[k]} next : {e}")
    return preds

# ── Terminal KPI ──────────────────────────────────────────────────────────────

def print_kpi_table(results, naive, ticker, start, end, cfg):
    metrics = ["RMSE","MAE","MAPE (%)","Dir. Acc (%)","CRPS","Train Time (s)"]
    mh = {"RMSE":"RMSE","MAE":"MAE","MAPE (%)":"MAPE%",
          "Dir. Acc (%)":"DirAcc%","CRPS":"CRPS","Train Time (s)":"Time(s)"}
    W = 82; cm, cv = 14, 10
    print(f"\n{'='*W}\n  {_c('RESULTATS — ARIMA-OLS vs SARIMA-OLS  ('+cfg['label']+')',_BO)}")
    print(f"  {ticker}  {start} -> {end}\n{'='*W}")
    print(f"  {_lp('Modele',cm)}" + "".join(_rp(mh[m], cv) for m in metrics))
    sep = "  " + "-"*(cm + cv*len(metrics)); print(sep)
    keys = list(results.keys())
    best = {}
    for m in metrics:
        v = {k: results[k].get(m) for k in keys if isinstance(results[k].get(m), (int,float))}
        if v: best[m] = (max if m == "Dir. Acc (%)" else min)(v, key=v.get)
    print(f"  {_lp(_c('Naif',_YL),cm)}" +
          "".join(_rp(_c(str(naive.get(m,'--')),_YL), cv) for m in metrics))
    print(sep)
    for k in keys:
        r   = results[k]; row = f"  {_lp(MODEL_LABELS[k],cm)}"
        for m in metrics:
            v = r.get(m, "--"); nv = naive.get(m)
            if not isinstance(v, (int,float)): row += _rp(str(v), cv); continue
            if best.get(m) == k: row += _rp(_c(str(v), _GR), cv)
            elif isinstance(nv,(int,float)) and m not in ("Dir. Acc (%)","CRPS") and v > nv:
                row += _rp(_c(str(v)+"!", _RE), cv)
            elif isinstance(nv,(int,float)) and m == "Dir. Acc (%)" and v < nv:
                row += _rp(_c(str(v)+"!", _RE), cv)
            else: row += _rp(str(v), cv)
        print(row)
    print(sep); print(f"  {_c('V',_GR)} meilleur  {_c('!',_RE)} pire que naif\n{'='*W}\n")

# ── Evenements ────────────────────────────────────────────────────────────────

def _filter_events(start, end):
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    evts = [(pd.Timestamp(d), lbl, cat)
            for d,(lbl,cat) in sorted(MARKET_EVENTS.items())
            if t0 <= pd.Timestamp(d) <= t1]
    return [(d,lbl,cat,i+1) for i,(d,lbl,cat) in enumerate(evts)]

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS DE DESSIN  (helpers partages par fenetres interactives et PNG)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_date(train, test, results, naive, events, date_from=None, date_to=None):
    """Filtre toutes les donnees plot sur un intervalle de dates."""
    t0 = pd.Timestamp(date_from) if date_from else pd.Timestamp.min
    t1 = pd.Timestamp(date_to)   if date_to   else pd.Timestamp.max
    tr = train[(train.index >= t0) & (train.index <= t1)]
    te = test[ (test.index  >= t0) & (test.index  <= t1)]
    evts = [(d,l,c,n) for d,l,c,n in events if t0 <= d <= t1]
    ni = naive["index"]; nm = (ni >= t0) & (ni <= t1)
    nf = {**naive, "index": ni[nm],
          "predictions": naive["predictions"][nm],
          "lower":       naive["lower"][nm],
          "upper":       naive["upper"][nm],
          "actual":      naive["actual"][nm]}
    rf = {}
    for k, res in results.items():
        ri = res["index"]; rm = (ri >= t0) & (ri <= t1)
        if rm.any():
            rf[k] = {**res, "index": ri[rm],
                     "predictions": res["predictions"][rm],
                     "lower": res["lower"][rm] if res.get("lower") is not None else None,
                     "upper": res["upper"][rm] if res.get("upper") is not None else None,
                     "actual": res["actual"][rm]}
    return tr, te, rf, nf, evts

def _draw_series(ax, train, test, results, naive, events, show_pi,
                 zoom=False, next_preds=None, cfg=None,
                 show_hpb=True, show_annotations=True):
    """
    Walk-forward + (optionnel) prevision en continuation sur le meme axe.
    show_hpb         : labels H/P/B sur le premier pas de prevision.
    show_annotations : Cov.% et Ens:XX$ dans le graphe.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ctx = max(len(test), min(3*len(test), len(train)//3))
    train_vis = train.iloc[-ctx:] if zoom else train

    # ── Historique ────────────────────────────────────────────────────────────
    if not train_vis.empty:
        ax.plot(train_vis.index, train_vis.values,
                color="#cccccc", lw=1.0, label="Train")
    if not test.empty:
        ax.plot(test.index, test.values,
                color="black", lw=2.0, label="Reel (test)", zorder=5)
    if len(naive["index"]) > 0:
        ax.plot(naive["index"], naive["predictions"],
                color="#aaaaaa", lw=0.9, ls=":", alpha=0.8, label="Naif")

    # ── Walk-forward predictions : Moyen + Haut/Bas (PI 95 %) ────────────────
    cov_lines = []  # pour la legende des PI
    for k, res in results.items():
        col = MODEL_COLORS[k]; lbl = MODEL_LABELS[k]
        p_arr  = res["predictions"]
        lo_arr = res.get("lower"); hi_arr = res.get("upper")
        ax.plot(res["index"], p_arr, color=col, lw=1.8, ls=MODEL_LS[k],
                alpha=0.92, label=lbl, zorder=3)
        if show_pi and lo_arr is not None:
            # Bande semi-transparente
            ax.fill_between(res["index"], lo_arr, hi_arr, color=col, alpha=0.10, zorder=1)
            # Lignes Haut et Bas explicites (pointilles fins)
            ax.plot(res["index"], hi_arr, color=col, lw=0.8, ls=":", alpha=0.50, zorder=2)
            ax.plot(res["index"], lo_arr, color=col, lw=0.8, ls=":", alpha=0.50, zorder=2)
            # Couverture empirique en annotation au bord droit
            if show_annotations:
                cov = cov95(res["actual"], lo_arr, hi_arr)
                ax.annotate(f" Cov.{cov:.0f}%",
                            xy=(res["index"][-1], float(hi_arr[-1])),
                            fontsize=7, color=col, alpha=0.9,
                            xytext=(4, 0), textcoords="offset points", va="bottom")
        if show_pi:
            cov_lines.append(Line2D([0],[0], color=col, lw=0.8, ls=":",
                                    alpha=0.6, label=f"Haut/Bas PI 95% ({lbl})"))

    # ── Separateur train|test ─────────────────────────────────────────────────
    if not train.empty:
        ax.axvline(train.index[-1], color="#777777", lw=1.0, ls=":", alpha=0.7)
        ax.text(train.index[-1], 0.985, "  Train|Test",
                transform=ax.get_xaxis_transform(),
                fontsize=7, color="#666", va="top")
    # Zone test (verte)
    if not test.empty:
        ax.axvspan(test.index[0], test.index[-1],
                   alpha=0.06, color="#27ae60", zorder=0)

    # ── Previsions en continuation (si next_preds fourni) ────────────────────
    if next_preds and cfg and not test.empty:
        n_fc = max((len(v) for v in next_preds.values() if v), default=0)
        if n_fc > 0:
            fut = _future_dates(test.index[-1], n_fc, cfg)
            last_v = float(test.iloc[-1])
            is_mo  = cfg["interval"] == "1mo"
            td1    = pd.DateOffset(months=1) if is_mo else pd.Timedelta(days=1)

            # Zone forecast (orange clair)
            ax.axvspan(test.index[-1], fut[-1] + td1,
                       alpha=0.05, color="#e67e22", zorder=0)
            ax.axvline(test.index[-1], color="#e67e22",
                       lw=1.2, ls="--", alpha=0.55, zorder=3)
            ax.text(test.index[-1], 0.985, f"  {cfg['step_lbl']}+1..",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7, color="#e67e22", va="top")

            keys = [k for k in ACTIVE_MODELS if k in next_preds]
            # Decalage horizontal entre modeles pour eviter overlap
            n_m  = len(keys)
            day_offs = np.linspace(-2, 2, n_m) if n_m > 1 else [0.]

            for mi, k in enumerate(keys):
                steps = next_preds[k]; col = MODEL_COLORS[k]
                pf = [s[0] for s in steps]
                lf = [s[1] for s in steps]
                hf = [s[2] for s in steps]
                off = pd.Timedelta(days=day_offs[mi])

                # Ligne de continuation depuis dernier reel
                x_line = [test.index[-1]] + fut[:len(steps)]
                y_line = [last_v] + pf
                ax.plot(x_line, y_line, color=col, lw=1.2, ls="--",
                        alpha=0.65, zorder=3)

                # PI bande + lignes Haut/Bas forecast
                if show_pi:
                    ax.fill_between(fut[:len(steps)], lf, hf,
                                    color=col, alpha=0.08, zorder=1)
                    ax.plot(fut[:len(steps)], hf, color=col,
                            lw=0.7, ls=":", alpha=0.45)
                    ax.plot(fut[:len(steps)], lf, color=col,
                            lw=0.7, ls=":", alpha=0.45)

                # Errorbars par pas
                for si in range(len(steps)):
                    fd    = fut[si] + off
                    p0, lo0, hi0 = pf[si], lf[si], hf[si]
                    alpha = 0.95 if si == 0 else 0.45
                    ax.errorbar(fd, p0,
                                yerr=[[max(0., p0-lo0)], [max(0., hi0-p0)]],
                                fmt="o" if si == 0 else "^",
                                color=col, ecolor=col,
                                elinewidth=1.8 if si == 0 else 1.1,
                                capsize=4, capthick=1.2,
                                ms=8 if si == 0 else 5,
                                alpha=alpha, zorder=5)

                # Labels H / P / B sur M+1 (si demande)
                if show_hpb:
                    v_sign = 1 if mi % 2 == 0 else -1
                    p0, lo0, hi0 = pf[0], lf[0], hf[0]
                    span = max(hi0 - lo0, 1.)
                    pad  = span * 0.06
                    fd0  = fut[0] + off
                    ax.text(fd0, hi0 + pad*v_sign if v_sign > 0 else hi0 + pad,
                            f"H:{hi0:.0f}$", fontsize=7, color=col,
                            ha="center", va="bottom", alpha=0.88)
                    ax.text(fd0, p0,
                            f" P:{p0:.0f}$", fontsize=7.5, color=col,
                            ha="left", va="center", fontweight="bold")
                    ax.text(fd0, lo0 - pad*v_sign if v_sign > 0 else lo0 - pad,
                            f"B:{lo0:.0f}$", fontsize=7, color=col,
                            ha="center", va="top", alpha=0.88)

            # Ensemble (si 2+ modeles)
            if n_m >= 2:
                for si in range(n_fc):
                    av = [next_preds[k][si] for k in keys if si < len(next_preds[k])]
                    ep = float(np.mean([t[0] for t in av]))
                    el = float(np.min([t[1] for t in av]))
                    eh = float(np.max([t[2] for t in av]))
                    ae = 0.95 if si == 0 else 0.4
                    ax.errorbar(fut[si], ep,
                                yerr=[[max(0.,ep-el)],[max(0.,eh-ep)]],
                                fmt="*", color="darkorange",
                                ecolor="darkorange", elinewidth=1.8,
                                capsize=5, ms=11, alpha=ae, zorder=6)
                    if si == 0 and show_annotations:
                        ax.text(fut[si], eh + (eh-el)*0.04,
                                f"Ens:{ep:.0f}$", ha="center", va="bottom",
                                fontsize=7.5, color="darkorange",
                                fontweight="bold")

    # ── Evenements (avec espacement minimum pour eviter overlap) ─────────────
    xf = ax.get_xaxis_transform(); cats = set()
    last_evt_d = pd.Timestamp("1900-01-01")
    # Gap minimum : ~6 semaines en mensuel, ~10 jours en journalier
    min_gap = pd.Timedelta(days=45 if (cfg and cfg["interval"]=="1mo") else 10)
    for d, lbl, cat, num in events:
        if not train_vis.empty and d < train_vis.index[0]: continue
        if (d - last_evt_d) < min_gap: continue   # saute si trop proche du precedent
        col_e = EVENT_COLORS.get(cat, "#888")
        ax.axvline(d, color=col_e, lw=0.6, ls="--", alpha=0.35, zorder=1)
        ax.text(d, 0.985, f" {num}", transform=xf, rotation=90,
                va="top", ha="right", fontsize=5.5, color=col_e, alpha=0.75)
        cats.add(cat); last_evt_d = d

    # ── Legendes (deux niveaux, pas de chevauchement) ─────────────────────────
    # Legende modeles
    model_handles = []
    for k in results:
        model_handles.append(
            Line2D([0],[0], color=MODEL_COLORS[k], lw=1.8, ls=MODEL_LS[k],
                   label=MODEL_LABELS[k]))
    if naive and len(naive["index"]) > 0:
        model_handles.insert(0, Line2D([0],[0], color="#aaaaaa", lw=0.9, ls=":",
                                       alpha=0.8, label="Naif"))
    if not train_vis.empty:
        model_handles.insert(0, Line2D([0],[0], color="#cccccc", lw=1.2,
                                       label="Train"))
    if not test.empty:
        model_handles.insert(len(model_handles)//2,
            Line2D([0],[0], color="black", lw=2.0, label="Reel (test)"))
    if show_pi:
        model_handles.append(
            Patch(facecolor="#888", alpha=0.15,
                  label="PI 95 % (1.96σ)  — — = Haut/Bas"))
    if next_preds and cfg:
        keys = [k for k in ACTIVE_MODELS if k in next_preds]
        if len(keys) >= 2:
            model_handles.append(
                Line2D([0],[0], color="darkorange", lw=0, marker="*",
                       ms=10, label="Ensemble Moy."))
    leg1 = ax.legend(handles=model_handles, fontsize=7.5,
                     ncol=3, loc="upper left",
                     framealpha=0.92, borderpad=0.5, labelspacing=0.3)
    ax.add_artist(leg1)

    # Legende evenements
    if cats:
        ax.legend(handles=[Line2D([0],[0], color=EVENT_COLORS[c], lw=1.2,
                                  ls="--", alpha=0.7, label=EVENT_LABELS[c])
                           for c in EVENT_COLORS if c in cats],
                  fontsize=7, loc="lower right", title="Evts",
                  title_fontsize=7, framealpha=0.88, borderpad=0.4)

    ax.set_ylabel("Prix ($)", fontsize=9.5)
    ax.grid(True, alpha=0.18, lw=0.5); ax.set_facecolor("#fafbfc")


def _draw_forecast_zoom(ax, next_preds, test, cfg, show_pi=True):
    """
    Vue centree uniquement sur les previsions (+ quelques points de contexte).
    H / P / B annotes sur chaque pas de prevision pour tous les modeles.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not next_preds or test.empty:
        ax.text(0.5, 0.5, "Aucune prevision disponible",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#999")
        ax.set_facecolor("#fafbfc"); return

    n_fc = max((len(v) for v in next_preds.values() if v), default=0)
    if n_fc == 0:
        ax.text(0.5, 0.5, "Aucune prevision disponible",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#999")
        ax.set_facecolor("#fafbfc"); return

    ctx    = min(cfg.get("zoom_ctx", 6), len(test))
    test_ctx = test.iloc[-ctx:]
    fut    = _future_dates(test.index[-1], n_fc, cfg)
    is_mo  = cfg["interval"] == "1mo"
    td1    = pd.DateOffset(months=1) if is_mo else pd.Timedelta(days=1)
    last_v = float(test.iloc[-1])

    # Contexte reel (derniers points)
    ax.plot(test_ctx.index, test_ctx.values,
            color="black", lw=2.2, label="Reel (contexte)",
            marker="o", ms=4, zorder=5)

    # Separateur reel / previsions
    ax.axvline(test.index[-1], color="#e67e22", lw=1.4, ls="--", alpha=0.65, zorder=3)
    ax.text(test.index[-1], 0.985, f"  {cfg['step_lbl']}+1..",
            transform=ax.get_xaxis_transform(), fontsize=8, color="#e67e22", va="top")
    ax.axvspan(test.index[-1], fut[-1] + td1, alpha=0.06, color="#e67e22", zorder=0)

    keys     = [k for k in ACTIVE_MODELS if k in next_preds]
    n_m      = len(keys)
    day_offs = np.linspace(-3, 3, n_m) if n_m > 1 else [0.]

    for mi, k in enumerate(keys):
        steps = next_preds[k]; col = MODEL_COLORS[k]
        pf = [s[0] for s in steps]; lf = [s[1] for s in steps]; hf = [s[2] for s in steps]
        off = pd.Timedelta(days=day_offs[mi])

        # Ligne de continuation depuis dernier reel connu
        ax.plot([test.index[-1]] + fut[:len(steps)], [last_v] + pf,
                color=col, lw=1.5, ls="--", alpha=0.70, zorder=3)

        # Bande PI
        if show_pi:
            ax.fill_between(fut[:len(steps)], lf, hf, color=col, alpha=0.12, zorder=1)
            ax.plot(fut[:len(steps)], hf, color=col, lw=1.0, ls=":", alpha=0.55)
            ax.plot(fut[:len(steps)], lf, color=col, lw=1.0, ls=":", alpha=0.55)

        # Errorbars uniquement — les valeurs sont dans la boite recapitulative
        for si in range(len(steps)):
            fd           = fut[si] + off
            p0, lo0, hi0 = pf[si], lf[si], hf[si]
            alpha_pt     = 0.95 if si == 0 else 0.70
            ax.errorbar(fd, p0,
                        yerr=[[max(0., p0-lo0)], [max(0., hi0-p0)]],
                        fmt="o" if si == 0 else "^",
                        color=col, ecolor=col,
                        elinewidth=2.2 if si == 0 else 1.6,
                        capsize=5, capthick=1.5,
                        ms=10 if si == 0 else 7,
                        alpha=alpha_pt, zorder=5)
            # Juste le numero de pas au-dessus du marqueur M+1
            if si == 0:
                ax.annotate(f"{cfg['step_lbl']}+1",
                            xy=(fd, hi0), xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=8, color=col, ha="center", va="bottom",
                            alpha=0.75)

    # Ensemble (si 2+ modeles)
    ens_vals = []
    if n_m >= 2:
        for si in range(n_fc):
            av = [next_preds[k][si] for k in keys if si < len(next_preds[k])]
            ep = float(np.mean([t[0] for t in av]))
            el = float(np.min( [t[1] for t in av]))
            eh = float(np.max( [t[2] for t in av]))
            ae = 0.95 if si == 0 else 0.55
            ens_vals.append((ep, el, eh))
            ax.errorbar(fut[si], ep,
                        yerr=[[max(0., ep-el)], [max(0., eh-ep)]],
                        fmt="*", color="darkorange",
                        ecolor="darkorange", elinewidth=2.2,
                        capsize=6, ms=14, alpha=ae, zorder=6)

    # ── Retrecir l'axe pour liberer deux colonnes de boites a droite ────────
    pos    = ax.get_position()          # Bbox en coordonnees figure [0,1]
    new_w  = pos.width * 0.55
    ax.set_position([pos.x0, pos.y0, new_w, pos.height])
    fig_obj   = ax.get_figure()
    sl        = cfg["step_lbl"]

    # Espace disponible a droite (coordonnees figure)
    gap       = 0.012
    right_x0  = pos.x0 + new_w + gap   # bord gauche de la zone droite
    right_x1  = pos.x1                 # bord droit (= gridspec right)
    right_mid = (right_x0 + right_x1) / 2
    top_y     = pos.y1                 # haut de l'axe
    bot_y     = pos.y0                 # bas  de l'axe

    # ── Deux colonnes cote a cote : ARIMA gauche, SARIMA droite ──────────────
    for mi, k in enumerate(keys):
        steps_k = next_preds[k]
        pf2 = [s[0] for s in steps_k]
        lf2 = [s[1] for s in steps_k]
        hf2 = [s[2] for s in steps_k]
        col = MODEL_COLORS[k]
        lines = [f" {MODEL_LABELS[k]} ", "─" * 18]
        for si in range(len(steps_k)):
            lines.append(f"{sl}+{si+1}  H: {hf2[si]:>6.0f}$")
            lines.append(f"     P: {pf2[si]:>6.0f}$")
            lines.append(f"     B: {lf2[si]:>6.0f}$")
        # mi pair → colonne gauche  |  mi impair → colonne droite
        x_fig = right_x0 if mi % 2 == 0 else right_mid + 0.005
        fig_obj.text(x_fig, top_y, "\n".join(lines),
                     ha="left", va="top",
                     fontsize=9.5, color=col,
                     fontfamily="monospace", linespacing=1.35,
                     bbox=dict(boxstyle="round,pad=0.50", fc="white", ec=col,
                               alpha=0.95, lw=1.6))

    # ── Boite ensemble centree en bas (entre les deux colonnes) ──────────────
    if n_m >= 2 and ens_vals:
        e_lines = [" Ensemble (moy.) ", "─" * 18]
        for si, (ep, el, eh) in enumerate(ens_vals):
            e_lines.append(f"{sl}+{si+1}  P: {ep:>6.0f}$")
        fig_obj.text((right_x0 + right_x1) / 2, bot_y,
                     "\n".join(e_lines),
                     ha="center", va="bottom",
                     fontsize=9.5, color="darkorange",
                     fontfamily="monospace", linespacing=1.35,
                     bbox=dict(boxstyle="round,pad=0.50", fc="white",
                               ec="darkorange", alpha=0.95, lw=1.4))

    # Legende compacte dans l'axe (en haut a gauche)
    handles = [Line2D([0],[0], color="black", lw=2.2, marker="o", ms=5,
                      label="Reel (contexte)")]
    for k in keys:
        handles.append(Line2D([0],[0], color=MODEL_COLORS[k], lw=1.5, ls="--",
                               label=MODEL_LABELS[k]))
    if n_m >= 2:
        handles.append(Line2D([0],[0], color="darkorange", lw=0,
                               marker="*", ms=12, label="Ensemble Moy."))
    if show_pi:
        handles.append(Patch(facecolor="#888", alpha=0.18,
                             label="PI 95 % (1.96σ)"))
    ax.legend(handles=handles, fontsize=8.5, loc="upper left",
              ncol=1, framealpha=0.92, borderpad=0.5, labelspacing=0.35)
    ax.set_ylabel("Prix ($)", fontsize=9.5)
    ax.grid(True, alpha=0.20, lw=0.5); ax.set_facecolor("#fafbfc")


def _draw_kpi(ax, results, naive):
    ax.axis("off")
    a    = np.asarray(naive["actual"], float)
    nrms = naive.get("RMSE"); nda = naive.get("Dir. Acc (%)", 0.)
    nwkl = winkler(a, naive["lower"], naive["upper"])
    cols = ["Modele","RMSE","MAE","MAPE%","Dir%","CRPS","Cov%","Winkler","DM/Naif"]
    G="#d0f0d0"; R="#f7d0d0"; N="#f5f5e6"; W="white"
    rows   = [["Naif",
               f"{nrms:.2f}" if isinstance(nrms,float) else "--",
               f"{naive.get('MAE',0.):.2f}", "--", f"{nda:.1f}", "--",
               f"{cov95(a,naive['lower'],naive['upper']):.1f}",
               f"{nwkl:.1f}", "--"]]
    colors = [[N]*len(cols)]; md = []
    for k, res in results.items():
        p = np.asarray(res["predictions"],float)
        l = np.asarray(res.get("lower",p),float); h = np.asarray(res.get("upper",p),float)
        cv = cov95(a,l,h); wk = winkler(a,l,h)
        me = a - p; ne = a - np.asarray(naive["predictions"],float)
        ds, dp = dm_test(me, ne)
        dms = (f"W({dp:.2f})" if dp<0.05 and ds<0 else
               f"L({dp:.2f})" if dp<0.05 else f"T({dp:.2f})")
        rc = res.get("CRPS"); rr = res.get("RMSE")
        md.append({"k":k,"rmse":rr,"mae":res.get("MAE"),"mape":res.get("MAPE (%)"),"da":res.get("Dir. Acc (%)",0.),"crps":rc,"cov":cv,"wkl":wk,"dm":dms})
        rows.append([MODEL_LABELS[k],
                     f"{rr:.2f}" if isinstance(rr,float) else "--",
                     f"{res.get('MAE',0.):.2f}",
                     f"{res.get('MAPE (%)',0.):.1f}%",
                     f"{res.get('Dir. Acc (%)',0.):.1f}",
                     f"{rc:.4f}" if isinstance(rc,float) else "--",
                     f"{cv:.1f}", f"{wk:.1f}", dms])
        colors.append([W]*len(cols))
    col_idx = {"rmse":1,"mae":2,"mape":3,"da":4,"crps":5,"cov":6,"wkl":7,"dm":8}
    for field, hi_is_best in [("rmse",False),("mae",False),("da",True),("crps",False)]:
        v = [(i,d[field]) for i,d in enumerate(md) if isinstance(d[field],float)]
        if v: colors[(max if hi_is_best else min)(v,key=lambda x:x[1])[0]+1][col_idx[field]] = G
    for i,d in enumerate(md):
        if isinstance(d["rmse"],float) and isinstance(nrms,float) and d["rmse"]>nrms \
                and colors[i+1][1]!=G: colors[i+1][1] = R
        if isinstance(d["da"],float) and d["da"]<nda and colors[i+1][4]!=G: colors[i+1][4] = R
    for i,d in enumerate(md):
        cv_v = d["cov"]
        if abs(cv_v-95) < abs((md[1-i]["cov"] if len(md)>1 else 100)-95) \
                and 90<=cv_v<=100 and colors[i+1][6]!=G: colors[i+1][6] = G
        if (cv_v<90 or cv_v>100) and colors[i+1][6]!=G: colors[i+1][6] = R
        if isinstance(d["wkl"],float) and d["wkl"]>nwkl and colors[i+1][7]!=G: colors[i+1][7] = R
        if d["dm"].startswith("W"): colors[i+1][8] = G
        elif d["dm"].startswith("L"): colors[i+1][8] = R
    wv = [(i,d["wkl"]) for i,d in enumerate(md)]
    if wv: colors[min(wv,key=lambda x:x[1])[0]+1][7] = G
    cov_v = [(i,d["cov"]) for i,d in enumerate(md)]
    if cov_v: colors[min(cov_v,key=lambda x:abs(x[1]-95))[0]+1][6] = G

    t = ax.table(cellText=rows, colLabels=cols, cellColours=colors,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1., 1.75)
    for j in range(len(cols)):
        t[0,j].set_facecolor("#3a5a8a"); t[0,j].set_text_props(color="white", fontweight="bold")
    for j in range(len(cols)):
        t[1,j].set_text_props(fontstyle="italic", color="#555")
    for i in range(len(rows)+1):
        t[i,0].set_text_props(ha="left")


def _draw_errors(ax, results, naive, cfg):
    """
    Barres d'erreurs cote a cote.
    ARIMA = hachures '///'  |  SARIMA = plein  |  pointilles = RMSE
    """
    a = np.asarray(naive["actual"], float); idx = naive["index"]
    x = np.arange(len(idx)); keys = list(results.keys())
    w = 0.38; offs = np.linspace(-w/2, w/2, len(keys)) if len(keys)>1 else [0.]
    for k, off in zip(keys, offs):
        errs = a - np.asarray(results[k]["predictions"], float)
        col  = MODEL_COLORS[k]; htch = MODEL_HATCH[k]
        ax.bar(x+off, errs, width=w, color=col, alpha=0.70,
               hatch=htch, edgecolor="white", linewidth=0.5,
               label=MODEL_LABELS[k], zorder=3)
        rmse = results[k].get("RMSE", 0.)
        ax.axhline( rmse, color=col, lw=1.8, ls=":", alpha=0.85,
                    label=f"RMSE {MODEL_LABELS[k]} : {rmse:.0f}$")
        ax.axhline(-rmse, color=col, lw=1.8, ls=":", alpha=0.85)
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    step = max(1, len(idx)//10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime(cfg["date_fmt"]) for d in idx[::step]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Erreur reel - predit ($)", fontsize=9.5)
    ax.set_title("Erreurs par periode  |  hachures=ARIMA  plein=SARIMA"
                 "  |  pointilles=RMSE",
                 fontsize=9.5, pad=4)
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5); ax.set_facecolor("#fafbfc")


def _draw_ohlc(ax, next_preds, naive, train, results, models):
    """
    Barres OHLC pour chaque modele + 3 agregations ensemble :
      Agg.1 — Moyenne simple  (poids egaux)
      Agg.2 — Ponderee 1/RMSE (modele plus precis = plus de poids)
      Agg.3 — Mediane  (robuste aux valeurs aberrantes)
    Corps=1sigma  Moustaches=PI95%
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    last = float(np.asarray(naive["actual"], float)[-1])
    sig0 = float(np.std(np.diff(train.values)))
    j1   = {k: v[0] for k, v in next_preds.items() if v}

    items = [{"label":"Naif","color":"#888888",
              "pred":last,"lo":last-1.96*sig0,"hi":last+1.96*sig0,"sig":sig0}]
    for k in models:
        if k not in j1: continue
        p, lo, hi = j1[k]; sig = (hi-lo)/(2*1.96) if hi > lo else 0.
        items.append({"label":MODEL_LABELS[k],"color":MODEL_COLORS[k],
                      "pred":p,"lo":lo,"hi":hi,"sig":sig})

    if len(j1) >= 2:
        pv = [j1[k][0] for k in j1]; lv = [j1[k][1] for k in j1]; hv = [j1[k][2] for k in j1]
        sigs = [(h-l)/(2*1.96) for l,h in zip(lv,hv)]
        ep1 = float(np.mean(pv)); s1 = float(np.mean(sigs))
        items.append({"label":"Agg.1 Moy.",  "color":"#d4ac0d",
                      "pred":ep1,"lo":ep1-1.96*s1,"hi":ep1+1.96*s1,"sig":s1})
        wts = [1./max(results[k].get("RMSE",1.),1.) for k in j1]; sw = sum(wts)
        ep2 = float(sum(ww*pp for ww,pp in zip(wts,pv))/sw)
        s2  = float(sum(ww*ss for ww,ss in zip(wts,sigs))/sw)
        items.append({"label":"Agg.2 Pond.", "color":"#e67e22",
                      "pred":ep2,"lo":ep2-1.96*s2,"hi":ep2+1.96*s2,"sig":s2})
        ep3 = float(np.median(pv)); s3 = float(np.median(sigs))
        items.append({"label":"Agg.3 Med.",  "color":"#c0392b",
                      "pred":ep3,"lo":ep3-1.96*s3,"hi":ep3+1.96*s3,"sig":s3})

    n  = len(items); ys = list(range(n-1, -1, -1))
    xmn = min(it["lo"] for it in items); xmx = max(it["hi"] for it in items)
    xr  = max(xmx-xmn, 1.); toff = xr*0.012
    ax.axvline(last, color="black", lw=1.5, ls="--", zorder=10,
               label=f"Last close: {last:.0f}$")

    for i, it in enumerate(items):
        y=ys[i]; p=it["pred"]; lo=it["lo"]; hi=it["hi"]; sig=it["sig"]; col=it["color"]
        blo, bhi = p-sig, p+sig
        ax.plot([lo, hi], [y, y], color=col, lw=2., alpha=0.55, zorder=2)
        for xc in [lo, hi]:
            ax.plot([xc,xc], [y-.2, y+.2], color=col, lw=2., alpha=0.55, zorder=2)
        ax.barh(y, bhi-blo, left=blo, height=0.44, color=col, alpha=0.80,
                zorder=3, edgecolor="white", linewidth=0.5)
        ax.plot([p, p], [y-.22, y+.22], color="white", lw=2., zorder=4)
        ax.text(hi+toff, y+0.22, f"H:{hi:.0f}$",
                va="bottom", ha="left", fontsize=7.5, color=col, alpha=0.78)
        ax.text(hi+toff, y,      f"P:{p:.0f}$",
                va="center", ha="left", fontsize=8.5, color=col, fontweight="bold")
        ax.text(hi+toff, y-0.22, f"B:{lo:.0f}$",
                va="top",    ha="left", fontsize=7.5, color=col, alpha=0.78)

    n_agg = 3 if len(j1) >= 2 else 0
    if n_agg:
        sep_y = ys[len(items)-n_agg] + 0.5
        ax.axhline(sep_y, color="#cccccc", lw=0.8, ls="--", zorder=1)
        ax.text(xmn - xr*0.02, sep_y+0.05, "Agregations",
                fontsize=7.5, color="#888", va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels([it["label"] for it in items], fontsize=9.5)
    ax.set_xlim(xmn - xr*0.08, xmx + xr*0.30); ax.set_ylim(-.7, n-.3)
    ax.set_xlabel("Prix ($)", fontsize=10)
    ax.set_title("Next-step OHLC  |  Corps=1sigma  Moustaches=PI95%"
                 "  |  3 agregations ensemble",
                 fontsize=10, pad=4)
    ax.legend(handles=[
        Patch(facecolor="#888", alpha=0.8,  label="Corps = pred +/- 1 sigma"),
        Line2D([0],[0], color="#888", lw=2., alpha=0.5, label="Moustaches = PI 95%"),
        Line2D([0],[0], color="black", lw=1.5, ls="--", label=f"Last: {last:.0f}$"),
        Patch(facecolor="#d4ac0d", alpha=0.8, label="Agg.1 Moy. simple"),
        Patch(facecolor="#e67e22", alpha=0.8, label="Agg.2 Pond. 1/RMSE"),
        Patch(facecolor="#c0392b", alpha=0.8, label="Agg.3 Mediane"),
    ], fontsize=8, loc="lower right", framealpha=0.92)
    ax.grid(True, axis="x", alpha=0.25); ax.set_facecolor("#fafbfc")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES INTERACTIVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_data(train, test, results, naive, events, ticker, cfg,
             show_pi=True, zoom_from="2025-01-01", zoom_to=None,
             next_preds=None):
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
    n_fc = max((len(v) for v in next_preds.values()), default=0) if next_preds else 0
    fig = plt.figure(figsize=(16, 20),
                     num=f"DEITA 1.6 — {ticker} | Walk-Forward + Previsions")
    fig.patch.set_facecolor("#f7f8fa")
    fig.suptitle(
        f"DEITA 1.6  |  {ticker}  ({cfg['label']})  "
        f"—  Walk-Forward + Previsions {cfg['step_lbl']}+1..+{n_fc}",
        fontsize=12, fontweight="bold", y=0.993)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 3.8, 3.8],
                           hspace=0.38, left=0.07, right=0.97,
                           top=0.97, bottom=0.04, figure=fig)

    # ── Subplot 1 : Vue globale + forecast en continuation ───────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Vue globale — train / validation / previsions",
                  fontsize=9.5, pad=3)
    _draw_series(ax1, train, test, results, naive, events, show_pi,
                 zoom=False, next_preds=next_preds, cfg=cfg,
                 show_hpb=False, show_annotations=False)

    # ── Subplot 2 : Zoom test + forecast ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(
        f"Zoom validation ({len(test)} {cfg['step_lbl']}) + "
        f"previsions {cfg['step_lbl']}+1..+{n_fc}  |  PI 95 % (1.96σ)",
        fontsize=9.5, pad=3)
    _draw_series(ax2, train, test, results, naive, events, show_pi,
                 zoom=True, next_preds=next_preds, cfg=cfg,
                 show_hpb=False, show_annotations=False)

    # ── Subplot 3 : Zoom previsions uniquement (H / P / B sur chaque pas) ──
    ax3 = fig.add_subplot(gs[2])
    n_fc_lbl = max((len(v) for v in next_preds.values()), default=0) if next_preds else 0
    ax3.set_title(
        f"Zoom previsions {cfg['step_lbl']}+1..+{n_fc_lbl}  "
        f"|  PI 95 % (1.96σ)  |  H = Haut  P = Prediction  B = Bas",
        fontsize=9.5, pad=3)
    _draw_forecast_zoom(ax3, next_preds, test, cfg, show_pi)
    return fig


def plot_comparison(results, naive, next_preds, train, ticker, cfg):
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16,12),
                     num=f"DEITA 1.6 — {ticker} | Comparaison ARIMA vs SARIMA")
    fig.patch.set_facecolor("#f7f8fa")
    fig.suptitle(f"DEITA 1.6  |  {ticker}  ({cfg['label']})  —  ARIMA-OLS vs SARIMA-OLS",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3,1, height_ratios=[1.8,2.0,2.8], hspace=0.52,
                           left=0.07, right=0.97, top=0.94, bottom=0.07, figure=fig)
    ax_kpi = fig.add_subplot(gs[0]); ax_kpi.axis("off")
    ax_kpi.set_title("KPIs  |  vert=meilleur  rouge=pire que naif  |  Coverage cible 95%",
                     fontsize=9, fontweight="bold", pad=5, loc="left")
    _draw_kpi(ax_kpi, results, naive)
    _draw_errors(fig.add_subplot(gs[1]), results, naive, cfg)
    _draw_ohlc(fig.add_subplot(gs[2]), next_preds, naive, train, results,
               [k for k in ACTIVE_MODELS if k in results])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  PNG DASHBOARD (4 panneaux)
# ══════════════════════════════════════════════════════════════════════════════

def save_dashboard(train, test, results, naive, next_preds, series,
                   ticker, start, end, cfg, path, show_pi=True,
                   zoom_from="2025-01-01", zoom_to=None):
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
    events = _filter_events(start, end)
    n_fc = max((len(v) for v in next_preds.values()), default=0) if next_preds else 0

    fig = plt.figure(figsize=(16, 46)); fig.patch.set_facecolor("#f7f8fa")
    fig.suptitle(
        f"DEITA 1.6  |  {ticker}  ({cfg['label']})  "
        f"—  ARIMA-OLS vs SARIMA-OLS  |  PI 95 % (1.96σ)  H/P/B = Haut/Pred/Bas",
        fontsize=13, fontweight="bold", y=0.999)

    outer = gridspec.GridSpec(4, 1, height_ratios=[3.2, 4.2, 4.2, 6.2],
                              hspace=0.30, left=0.07, right=0.97,
                              top=0.996, bottom=0.010, figure=fig)

    # ── 1. Vue globale + forecast ────────────────────────────────────────────
    ax1 = fig.add_subplot(outer[0])
    ax1.set_title("1. Serie complete — Walk-Forward + Previsions",
                  fontsize=10, fontweight="bold", pad=3, loc="left", color="#2c3e50")
    _draw_series(ax1, train, test, results, naive, events, show_pi,
                 zoom=False, next_preds=next_preds, cfg=cfg,
                 show_hpb=False, show_annotations=False)

    # ── 2. Zoom validation + forecast ───────────────────────────────────────
    ax2 = fig.add_subplot(outer[1])
    ax2.set_title(
        f"2. Zoom validation ({len(test)} {cfg['step_lbl']}) + "
        f"Previsions {cfg['step_lbl']}+1..+{n_fc}",
        fontsize=10, fontweight="bold", pad=3, loc="left", color="#2c3e50")
    _draw_series(ax2, train, test, results, naive, events, show_pi,
                 zoom=True, next_preds=next_preds, cfg=cfg,
                 show_hpb=False, show_annotations=False)

    # ── 3. Zoom previsions uniquement (H / P / B sur chaque pas) ────────────
    ax3 = fig.add_subplot(outer[2])
    ax3.set_title(
        f"3. Zoom Previsions {cfg['step_lbl']}+1..+{n_fc}  |  H/P/B = Haut/Pred/Bas",
        fontsize=10, fontweight="bold", pad=3, loc="left", color="#2c3e50")
    _draw_forecast_zoom(ax3, next_preds, test, cfg, show_pi)

    # ── 4. Comparaison KPI + erreurs + OHLC ─────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[3],
                                             height_ratios=[1.6, 2.0, 2.6],
                                             hspace=0.52)
    ax4a = fig.add_subplot(inner[0]); ax4a.axis("off")
    ax4a.set_title("4. Comparaison ARIMA-OLS vs SARIMA-OLS",
                   fontsize=10, fontweight="bold", pad=3, loc="left", color="#2c3e50")
    _draw_kpi(ax4a, results, naive)
    _draw_errors(fig.add_subplot(inner[1]), results, naive, cfg)
    _draw_ohlc(fig.add_subplot(inner[2]), next_preds, naive, train, results,
               [k for k in ACTIVE_MODELS if k in results])

    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"[PNG] -> {path}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _place_windows(pairs):
    """
    Positionne les fenetres matplotlib sur l'ecran.
    pairs : liste de (fig, x_pixels, y_pixels)
    Fonctionne avec les backends Qt (PyQt5/PySide2/Qt6) et Tk (TkAgg).
    """
    for fig, x, y in pairs:
        try:
            mgr = fig.canvas.manager
            if not hasattr(mgr, "window"):
                continue
            win = mgr.window
            if hasattr(win, "move"):           # Qt
                win.move(int(x), int(y))
            elif hasattr(win, "wm_geometry"):  # Tk / TkAgg
                win.wm_geometry(f"+{int(x)}+{int(y)}")
        except Exception:
            pass


def main():
    global _USE_COLOR, ACTIVE_MODELS, FORECAST_STEPS
    ap = argparse.ArgumentParser(description="DEITA Benchmark 1.6  —  ARIMA vs SARIMA")
    ap.add_argument("--ticker",     default=TICKER)
    ap.add_argument("--freq",       default=FREQ,   choices=["M","J"])
    ap.add_argument("--start",      default=START)
    ap.add_argument("--end",        default=END)
    ap.add_argument("--test-ratio", default=TEST_RATIO, type=float)
    ap.add_argument("--models",     default=ACTIVE_MODELS, nargs="+",
                    choices=["arima","sarima"])
    ap.add_argument("--forecast",   default=FORECAST_STEPS, type=int)
    ap.add_argument("--seed",       default=SEED,   type=int)
    ap.add_argument("--no-pi",      action="store_true")
    ap.add_argument("--no-png",     action="store_true")
    ap.add_argument("--no-show",    action="store_true")
    ap.add_argument("--no-color",   action="store_true")
    args = ap.parse_args()

    if args.no_color: _USE_COLOR = False
    ACTIVE_MODELS  = args.models
    FORECAST_STEPS = args.forecast
    cfg = FREQ_CONFIG[args.freq]

    start = cfg["start_default"] if args.start == "auto" else args.start
    end   = pd.Timestamp.today().strftime("%Y-%m-%d") if args.end == "today" else args.end

    prices = fetch_prices(args.ticker, start, end, cfg)
    train, test = split_series(prices, args.test_ratio)
    print(f"[SPLIT] Train : {len(train)} {cfg['step_lbl']} "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test  : {len(test)}  {cfg['step_lbl']} "
          f"({test.index[0].date()}  -> {test.index[-1].date()})\n")
    if args.freq == "M" and len(train) < 36:
        print("[WARN] Moins de 36 mois — SARIMA(s=12) peut etre instable.\n")

    naive   = compute_naive(train, test)
    results = run_all(train, test, args.models, cfg, args.seed)
    if not results:
        sys.exit("[ERR] Aucun modele n'a produit de resultat.")

    print_kpi_table(results, naive, args.ticker, start, end, cfg)

    n_fc = max(1, args.forecast)
    print(f"\n[NEXT] Previsions {cfg['step_lbl']}+1..+{n_fc} ...")
    series     = pd.concat([train, test])
    next_preds = forecast_all(series, args.models, n_fc, cfg, args.seed)

    show_pi = not args.no_pi
    events  = _filter_events(start, end)

    if not args.no_png and SAVE_PNG:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"deita_dashboard_1.6_{args.freq.lower()}.png")
        print(f"\n[PNG] Dashboard ...")
        save_dashboard(train, test, results, naive, next_preds, series,
                       args.ticker, start, end, cfg, path, show_pi)

    if not args.no_show:
        print("[DASH] Ouverture des 2 fenetres ...")
        f1 = plot_data(train, test, results, naive, events,
                       args.ticker, cfg, show_pi,
                       zoom_from="2025-01-01", next_preds=next_preds)
        f2 = plot_comparison(results, naive, next_preds, train, args.ticker, cfg)
        import matplotlib.pyplot as plt
        # Fenetre 1 (Walk-Forward) a gauche  |  Fenetre 2 (KPI+OHLC) a droite
        _place_windows([(f1, 0, 30), (f2, 860, 30)])
        print("[DASH] Fermer les fenetres pour quitter.\n")
        plt.show()


if __name__ == "__main__":
    main()
