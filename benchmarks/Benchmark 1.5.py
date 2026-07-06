"""
DEITA Benchmark 1.5 - Zoom J-7 + Previsions multi-pas
======================================================
Extension de Benchmark 1.4.

Nouveautes v1.5
---------------
  - Fenetre 3 (zoom) : les N derniers jours reels + previsions J+1 et J+2
    en errorbars par modele sur axe temporel. J+2 = speculatif (la prediction
    J+1 est utilisee comme "valeur observee" pour l'etape suivante).
  - compute_next_steps_n() : walk-forward sur n_steps passes successifs.
  - PNG etendu a 5 panneaux (zoom en bas).
  - Tous les autres panneaux de 1.4 conserves a l'identique.

Logique J+2 "speculatif"
------------------------
  Etape 1 : fit(train+test) -> pred J+1, PI J+1
  Etape 2 : fit(train+test+pred_J1) -> pred J+2, PI J+2
  => J+2 accumule l'incertitude de J+1. Ce n'est pas une vraie prevision
     walk-forward (pas de valeur reelle connue), d'ou le label "(spec.)".

Usage
-----
  python "Benchmark 1.5.py"
  python "Benchmark 1.5.py" --conformal
  python "Benchmark 1.5.py" --zoom-days 10     # 10 jours de contexte
  python "Benchmark 1.5.py" --zoom-steps 1     # J+1 seulement
  python "Benchmark 1.5.py" --models arima sarima prophet
  python "Benchmark 1.5.py" --no-dashboard
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

try:
    import certifi as _certifi
    _ca = _certifi.where()
    os.environ.setdefault("CURL_CA_BUNDLE",     _ca)
    os.environ.setdefault("SSL_CERT_FILE",      _ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
except ImportError:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

# ── Constantes ────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = ["arima", "sarima", "prophet", "lstm"]

MODEL_LABELS = {
    "arima":   "ARIMA-GARCH",
    "sarima":  "SARIMA",
    "prophet": "Prophet",
    "lstm":    "LSTM",
}

MODEL_COLORS = {
    "arima":   "tab:blue",
    "sarima":  "tab:green",
    "prophet": "tab:purple",
    "lstm":    "tab:red",
}

FREQ_CONFIG = {
    "J": ("1d",  "journaliere",  False),
    "H": ("1h",  "horaire",      False),
    "S": ("1wk", "hebdomadaire", True),
}

MARKET_EVENTS = {
    "2018-01-17": ("BTC ATH $20k (dec.)",    "crypto"),
    "2018-12-15": ("BTC bas $3.2k",           "crypto"),
    "2019-06-18": ("Annonce Libra (Meta)",    "crypto"),
    "2020-03-12": ("COVID crash -50%",        "macro"),
    "2020-03-15": ("Fed: taux 0%",            "monetaire"),
    "2020-05-11": ("BTC halving #3",          "crypto"),
    "2020-12-16": ("BTC franchit $20k",       "crypto"),
    "2021-01-29": ("WSB / GameStop",          "macro"),
    "2021-02-08": ("Tesla: 1.5G$ en BTC",    "crypto"),
    "2021-04-14": ("Coinbase IPO Nasdaq",     "crypto"),
    "2021-05-12": ("Tesla stop pmt BTC",      "crypto"),
    "2021-05-19": ("Chine ban crypto",        "geopolitique"),
    "2021-09-07": ("El Salvador: BTC legal",  "geopolitique"),
    "2021-11-10": ("BTC ATH $69k",            "crypto"),
    "2022-01-05": ("Fed pivot hawkish",       "monetaire"),
    "2022-02-24": ("Invasion Ukraine",        "geopolitique"),
    "2022-03-16": ("Fed +25bp",               "monetaire"),
    "2022-05-05": ("Fed +50bp",               "monetaire"),
    "2022-05-09": ("LUNA/UST collapse",       "crypto"),
    "2022-06-13": ("Celsius gele fonds",      "crypto"),
    "2022-06-15": ("Fed +75bp",               "monetaire"),
    "2022-09-15": ("ETH Merge (PoS)",         "crypto"),
    "2022-11-08": ("FTX collapse",            "crypto"),
    "2023-03-10": ("SVB faillite",            "macro"),
    "2023-06-05": ("SEC vs Coinbase",         "geopolitique"),
    "2023-07-26": ("Fed pic: 5.25%",          "monetaire"),
    "2024-01-10": ("BTC ETF spot US",         "crypto"),
    "2024-03-14": ("BTC ATH $73k",            "crypto"),
    "2024-04-19": ("BTC halving #4",          "crypto"),
}

EVENT_COLORS = {
    "crypto":       "#e67e22",
    "macro":        "#e74c3c",
    "monetaire":    "#2980b9",
    "geopolitique": "#8e44ad",
}

EVENT_CAT_LABEL = {
    "crypto":       "Crypto",
    "macro":        "Macro",
    "monetaire":    "Monetaire",
    "geopolitique": "Geopolitique",
}

# ── ANSI ──────────────────────────────────────────────────────────────────────

_GR = "\033[92m"; _RE = "\033[91m"; _YL = "\033[93m"
_BO = "\033[1m";  _RS = "\033[0m"
_USE_COLOR = True

import re as _re
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

def _c(t, code): return f"{code}{t}{_RS}" if _USE_COLOR else t
def _vlen(s):    return len(_ANSI_RE.sub("", s))
def _rpad(s, w): return " " * max(0, w - _vlen(s)) + s
def _lpad(s, w): return s + " " * max(0, w - _vlen(s))

# ── Utilitaires ───────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except ImportError:
        pass

def _make_session():
    try:
        from curl_cffi.requests import Session as CurlSession
        return CurlSession(impersonate="chrome", verify=False)
    except Exception:
        return None

def fetch_prices(ticker, start, end, freq):
    interval, label, _ = FREQ_CONFIG[freq]
    print(f"[DATA] Telechargement {ticker} [{start} -> {end}] frequence={label} ...")
    session = _make_session()
    raw = yf.download(
        ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=True,
        **({"session": session} if session else {}),
    )
    if raw.empty:
        sys.exit(f"[ERREUR] Aucune donnee pour {ticker}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = pd.to_numeric(raw["Close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    print(f"[DATA] {len(close)} observations chargees.\n")
    return close.astype(float)

def split_series(prices, test_ratio):
    split = int(len(prices) * (1 - test_ratio))
    return prices.iloc[:split], prices.iloc[split:]

def _infer_next_date(series):
    """Infere la date suivant la derniere observation."""
    idx = pd.DatetimeIndex(series.index)
    try:
        freq = pd.infer_freq(idx)
        if freq:
            return idx[-1] + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        pass
    diffs = np.diff(idx.asi8) / 1e9 / 86400
    med = float(np.median(diffs))
    if 0.9  <= med <= 1.5:   return idx[-1] + pd.tseries.offsets.BDay(1)
    if 6.5  <= med <= 7.5:   return idx[-1] + pd.Timedelta(weeks=1)
    if 0.03 <= med <= 0.06:  return idx[-1] + pd.Timedelta(hours=1)
    return idx[-1] + pd.Timedelta(days=med)

# ── Naif ──────────────────────────────────────────────────────────────────────

def compute_naive(train, test):
    actual = test.values.astype(float)
    preds  = np.concatenate([[train.iloc[-1]], actual[:-1]])
    resid  = actual - preds
    sigma  = float(np.std(np.diff(train.values)))
    return {
        "RMSE":           round(float(np.sqrt(np.mean(resid**2))), 4),
        "CRPS":           "--",
        "MAE":            round(float(np.mean(np.abs(resid))), 4),
        "MAPE (%)":       round(float(np.mean(np.abs(resid/(actual+1e-8)))*100), 2),
        "Dir. Acc (%)":   round(float(np.mean(
                              np.sign(np.diff(actual))==np.sign(np.diff(preds)))*100), 2),
        "Train Time (s)": 0.0,
        "predictions": preds, "lower": preds-1.96*sigma,
        "upper": preds+1.96*sigma, "index": test.index, "actual": actual,
    }

# ── Metriques ─────────────────────────────────────────────────────────────────

def crps_gaussian_approx(predictions, lower, upper, actual, z=1.96):
    from scipy.stats import norm as sp_norm
    mu  = np.asarray(predictions, float); lo = np.asarray(lower, float)
    hi  = np.asarray(upper, float);       y  = np.asarray(actual, float)
    sig = np.where((hi-lo)>0, (hi-lo)/(2*z), 1e-8)
    zsc = (y-mu)/sig
    return round(float(np.mean(sig*(zsc*(2*sp_norm.cdf(zsc)-1)
                                    +2*sp_norm.pdf(zsc)-1/np.sqrt(np.pi)))), 4)

def coverage(actual, lower, upper):
    y,lo,hi = map(np.asarray,(actual,lower,upper))
    return round(float(np.mean((y>=lo)&(y<=hi))*100), 2)

def avg_width(lower, upper):
    return round(float(np.mean(np.asarray(upper)-np.asarray(lower))), 2)

def winkler_score(actual, lower, upper, alpha=0.05):
    y,lo,hi = map(np.asarray,(actual,lower,upper))
    return round(float(np.mean(
        (hi-lo)
        + np.where(y<lo,(2/alpha)*(lo-y),0.)
        + np.where(y>hi,(2/alpha)*(y-hi),0.)
    )), 2)

def diebold_mariano(ea, eb):
    from scipy.stats import norm as sp_norm
    d = np.asarray(ea,float)**2 - np.asarray(eb,float)**2
    T = len(d)
    if T<5: return 0., 1.
    dbar  = np.mean(d)
    trunc = max(1, int(np.floor(T**(1/3))))
    var_d = np.var(d, ddof=0)
    for j in range(1, trunc+1):
        var_d += 2*(1-j/(trunc+1))*np.mean((d[j:]-dbar)*(d[:-j]-dbar))
    if var_d<=0: return 0., 1.
    DM  = dbar/np.sqrt(max(var_d,1e-12)/T)
    p   = 2*(1-sp_norm.cdf(abs(DM)))
    return round(float(DM),3), round(float(p),3)

# ── Modeles ───────────────────────────────────────────────────────────────────

def load_runner(key):
    sd = os.path.dirname(os.path.abspath(__file__))
    if sd not in sys.path: sys.path.insert(0, sd)
    try:
        if key=="arima":
            from arima_model import run_arima_garch;   return run_arima_garch
        if key=="sarima":
            from sarima_model import run_sarima;       return run_sarima
        if key=="prophet":
            from prophet_model import run_prophet;     return run_prophet
        if key=="lstm":
            from lstm_model import run_lstm;           return run_lstm
    except ImportError as e:
        print(f"[WARN] {MODEL_LABELS.get(key,key)} : dependance manquante - {e}")
        return None

def run_benchmark(train, test, models, seed):
    results = {}
    for key in models:
        runner = load_runner(key)
        if not runner: continue
        label = MODEL_LABELS[key]
        print(f"[RUN] {label} ...")
        set_seed(seed)
        try:
            res = runner(train, test)
        except Exception as e:
            print(f"[ERREUR] {label} : {e}"); continue
        if res.get("lower") is not None:
            res["CRPS"] = crps_gaussian_approx(
                res["predictions"], res["lower"], res["upper"], res["actual"])
        else:
            res["CRPS"] = "--"
        results[key] = res
        print(f"[RUN] {label} | RMSE={res.get('RMSE','?')}  "
              f"CRPS={res.get('CRPS','?')}  "
              f"Dir.Acc={res.get('Dir. Acc (%)','?')}%\n")
    return results

# ── Calibration conforme ──────────────────────────────────────────────────────

def conformal_calibrate(results: dict, train: pd.Series, seed: int,
                        cal_ratio: float = 0.20) -> dict:
    n_cal  = max(20, min(100, int(len(train) * cal_ratio)))
    train1 = train.iloc[:-n_cal]
    cal    = train.iloc[-n_cal:]
    calibrated = {}
    for key, res in results.items():
        label = MODEL_LABELS.get(key, key)
        if len(train1) < 30:
            calibrated[key] = res; continue
        print(f"[CONF] {label} : calibration sur {len(cal)} pts ...")
        set_seed(seed)
        try:
            if key == "prophet":
                sd = os.path.dirname(os.path.abspath(__file__))
                if sd not in sys.path: sys.path.insert(0, sd)
                from prophet_model import run_prophet_batch
                cal_res = run_prophet_batch(train1, cal)
            else:
                runner = load_runner(key)
                if not runner:
                    calibrated[key] = res; continue
                cal_res = runner(train1, cal)
        except Exception as e:
            print(f"[CONF] {label} : echec - {e}")
            calibrated[key] = res; continue
        resid = np.abs(np.asarray(cal_res["actual"]) - np.asarray(cal_res["predictions"]))
        n     = len(resid)
        qidx  = min(int(np.ceil((n + 1) * 0.95)), n) - 1
        q95   = float(np.sort(resid)[qidx])
        preds = np.asarray(res["predictions"])
        calibrated[key] = {
            **res,
            "lower": preds - q95,
            "upper": preds + q95,
            "conformal_q95": q95,
        }
        calibrated[key]["CRPS"] = crps_gaussian_approx(
            preds, preds - q95, preds + q95, res["actual"])
        print(f"[CONF] {label} : q95={q95:.2f}$  "
              f"Coverage avant={coverage(res['actual'],res['lower'],res['upper']):.1f}%  "
              f"apres={coverage(res['actual'],preds-q95,preds+q95):.1f}%")
    return calibrated

# ── Tableaux terminal ─────────────────────────────────────────────────────────

_MH = {"RMSE":"RMSE","CRPS":"CRPS","MAE":"MAE",
       "MAPE (%)":"MAPE%","Dir. Acc (%)":"DirAcc%","Train Time (s)":"Time(s)"}

def print_table1(results, naive, ticker, freq, start, end, n_tr, n_te):
    _, fl, _ = FREQ_CONFIG[freq]
    metrics  = list(_MH.keys()); all_keys = list(results.keys()); W = 80
    print(f"\n{'='*W}\n  {_c('TABLEAU 1 - Performances principales',_BO)}")
    print(f"  {ticker}  ({fl}, {start} -> {end})  Train:{n_tr} Test:{n_te}")
    print(f"{'='*W}")
    cm, cv = 16, 10
    print(f"  {_lpad('Modele',cm)}" + "".join(_rpad(_MH[m],cv) for m in metrics))
    sep = "  "+"-"*(cm+cv*len(metrics)); print(sep)
    best = {}
    for m in metrics:
        v = {k:results[k].get(m) for k in all_keys if isinstance(results[k].get(m),(int,float))}
        best[m] = (max if m=="Dir. Acc (%)" else min)(v, key=v.get) if v else None
    def _rv(val, m, mk, isn=False):
        if isn: return _c(str(val),_YL)
        if not isinstance(val,(int,float)): return str(val)
        nv = naive.get(m)
        if best.get(m)==mk: return _c(str(val),_GR)
        if isinstance(nv,(int,float)):
            w = val<nv if m=="Dir. Acc (%)" else val>nv
            if w: return _c(str(val)+"!",_RE)
        return str(val)
    def _row(lbl, data, mk=None, isn=False):
        l = _c(lbl,_YL) if isn else lbl
        print(f"  {_lpad(l,cm)}"+"".join(_rpad(_rv(data.get(m,"N/A"),m,mk,isn),cv) for m in metrics))
    _row("Naif (y_t)", naive, isn=True); print(sep)
    printed = set()
    for grp,keys in [("-- Statistique --",["arima","sarima","prophet"]),("-- RNN --",["lstm"])]:
        ig = [k for k in all_keys if k in keys]
        if ig:
            print(f"  {grp}")
            for k in ig: _row(MODEL_LABELS[k],results[k],mk=k); printed.add(k)
    for k in [k for k in all_keys if k not in printed]:
        _row(MODEL_LABELS[k],results[k],mk=k)
    print(sep); print(f"  {_c('[V]',_GR)} meilleur  {_c('[!]',_RE)} pire que naif  {_c('jaune',_YL)} naif")
    print(f"{'='*W}\n")
    rows = {"Naif":{m:naive.get(m,"N/A") for m in metrics}}
    for k in all_keys: rows[MODEL_LABELS[k]] = {m:results[k].get(m,"N/A") for m in metrics}
    df = pd.DataFrame(rows).T; df.index.name="Modele"; return df

def print_table2(results, naive, ticker, freq):
    _, fl, _ = FREQ_CONFIG[freq]; W = 70
    print(f"\n{'='*W}\n  {_c('TABLEAU 2 - Diebold-Mariano (paires)',_BO)}")
    print(f"  {ticker} ({fl}) | Ligne A vs Col B\n{'='*W}")
    items = [("naif","Naif",naive)]+[(k,MODEL_LABELS[k],results[k]) for k in results]
    errs  = {key:np.asarray(r["actual"],float)-np.asarray(r["predictions"],float)
             for key,_,r in items}
    names = [l for _,l,_ in items]
    cw    = max(14,max(len(n) for n in names)+2)
    sep   = "  "+"-"*(cw*(len(names)+1))
    print(f"  {_lpad('',cw)}"+"".join(_rpad(n,cw) for n in names)); print(sep)
    dm_rows = {}
    for ka,la,_ in items:
        row={};line=f"  {_rpad(la,cw)}"
        for kb,lb,_ in items:
            if ka==kb: cs,cc="--","--"
            else:
                dm,p=diebold_mariano(errs[ka],errs[kb])
                if p<0.05: cs=f"L ({p:.2f})" if dm>0 else f"W ({p:.2f})"; cc=_c(cs,_RE if dm>0 else _GR)
                else: cs=f"T ({p:.2f})"; cc=cs
            row[lb]=cs; line+=_rpad(cc,cw)
        print(line); dm_rows[la]=row
    print(sep); print(f"  {_c('W',_GR)} victoire  T egalite  {_c('L',_RE)} defaite")
    print(f"{'='*W}\n")
    df=pd.DataFrame(dm_rows).T; df.index.name="vs"; return df

def print_table3(results, naive, ticker, freq, alpha=0.05):
    _, fl, _ = FREQ_CONFIG[freq]; cov_tgt=(1-alpha)*100; W=70
    print(f"\n{'='*W}\n  {_c('TABLEAU 3 - PI a {:.0f}%'.format(cov_tgt),_BO)}")
    print(f"  {ticker} ({fl}) | Cible:{cov_tgt:.0f}% Zone:[{cov_tgt-5:.0f}%,{cov_tgt+5:.0f}%]\n{'='*W}")
    cm,cv=20,14; sep="  "+"-"*(cm+cv*3)
    print(f"  {_lpad('Methode',cm)}{_rpad('Coverage (%)',cv)}{_rpad('Width ($)',cv)}{_rpad('Winkler',cv)}")
    print(sep)
    actual=np.asarray(naive["actual"],float); rows={}
    def _fmt(lbl,lo,hi,ref=None,isn=False):
        cv_v=coverage(actual,lo,hi); wid=avg_width(lo,hi); wkl=winkler_score(actual,lo,hi,alpha)
        cv_s=_c(str(cv_v)+"!",_RE) if (cv_v<cov_tgt-5 or cv_v>cov_tgt+5) else str(cv_v)
        wk_s=(_c(str(wkl)+"!",_RE) if ref and wkl>ref and not isn else str(wkl))
        ls=_c(lbl,_YL) if isn else lbl
        print(f"  {_lpad(ls,cm)}{_rpad(cv_s,cv)}{_rpad(str(wid),cv)}{_rpad(wk_s,cv)}")
        return {"Coverage (%)":cv_v,"Width ($)":wid,"Winkler":wkl}
    nw=_fmt("Gaussien naif",naive["lower"],naive["upper"],isn=True); rows["Gaussien naif"]=nw; print(sep)
    for k,r in results.items():
        if r.get("lower") is None: continue
        rows[MODEL_LABELS[k]]=_fmt(MODEL_LABELS[k],r["lower"],r["upper"],ref=nw["Winkler"])
    print(sep); print(f"  {_c('[!]',_RE)} hors zone | Winkler > naif -> recalibrer"); print(f"{'='*W}\n")
    df=pd.DataFrame(rows).T; df.index.name="Methode"; return df

# ── Next-step (multi-pas) ─────────────────────────────────────────────────────

def compute_next_steps_n(full_series, models, seed, n_steps=2):
    """
    Walk-forward sur n_steps.
    J+1 : fit(full_series) -> pred / PI
    J+2 : fit(full_series + pred_J1) -> pred / PI  [speculatif]
    Retourne : {key: [(pred, lo, hi), ...]}  (liste de n_steps tuples)
    """
    sd = os.path.dirname(os.path.abspath(__file__))
    if sd not in sys.path: sys.path.insert(0, sd)

    all_preds = {}
    for key in models:
        label = MODEL_LABELS.get(key, key)
        print(f"[NEXT] {label} ({n_steps} pas) ...")
        set_seed(seed)
        steps   = []
        history = full_series.copy()

        for step in range(n_steps):
            next_date = _infer_next_date(history)
            try:
                if key == "arima":
                    from arima_model import next_step_arima_garch
                    pred, lo, hi = next_step_arima_garch(history)
                elif key == "sarima":
                    from sarima_model import next_step_sarima
                    pred, lo, hi = next_step_sarima(history)
                elif key == "prophet":
                    from prophet_model import next_step_prophet
                    pred, lo, hi = next_step_prophet(history, next_date=next_date)
                elif key == "lstm":
                    from lstm_model import next_step_lstm
                    pred, lo, hi = next_step_lstm(history)
                else:
                    break
                steps.append((float(pred), float(lo), float(hi)))
                # Append la prediction (pas une valeur reelle) pour l'etape suivante
                new_pt = pd.Series([float(pred)],
                                   index=pd.DatetimeIndex([next_date]))
                history = pd.concat([history, new_pt])
            except Exception as e:
                print(f"[NEXT] {label} J+{step+1}: ERREUR - {e}")
                break

        if steps:
            all_preds[key] = steps
            summary = " | ".join(
                f"J+{i+1}: {p:.2f}$ [{l:.2f}, {h:.2f}]"
                for i, (p, l, h) in enumerate(steps))
            print(f"[NEXT] {label}: {summary}")

    return all_preds


def _j1_preds(next_preds_n):
    """Extrait les predictions J+1 au format legacy {key:(pred,lo,hi)} pour OHLC."""
    return {k: v[0] for k, v in next_preds_n.items() if v}

# ── Helpers graphiques ────────────────────────────────────────────────────────

def _filter_events(start, end):
    t0,t1 = pd.Timestamp(start), pd.Timestamp(end)
    evts = [(pd.Timestamp(d),lbl,cat)
            for d,(lbl,cat) in sorted(MARKET_EVENTS.items())
            if t0<=pd.Timestamp(d)<=t1]
    return [(d,lbl,cat,i+1) for i,(d,lbl,cat) in enumerate(evts)]


def _build_kpi_table(results, naive):
    actual=np.asarray(naive["actual"],float)
    naif_errs=actual-np.asarray(naive["predictions"],float)
    naif_rmse=naive.get("RMSE"); naif_dacc=naive.get("Dir. Acc (%)",0.)
    naif_cov=coverage(actual,naive["lower"],naive["upper"])
    naif_wkl=winkler_score(actual,naive["lower"],naive["upper"])
    col_labels=["Modele","RMSE","CRPS","Dir.Acc%","Cover%","Winkler","vs Naif","DM/Naif"]
    _G="#d0f0d0"; _R="#f7d0d0"; _N="#f5f5e6"; _W="white"
    rows=[["Naif (ref)",
           f"{naif_rmse:.2f}" if isinstance(naif_rmse,float) else "--",
           "--",f"{naif_dacc:.1f}",f"{naif_cov:.1f}",f"{naif_wkl:.1f}","--","--"]]
    colors=[[_N]*len(col_labels)]
    mdata=[]
    for k in results:
        res=results[k]; r_rmse=res.get("RMSE"); r_crps=res.get("CRPS")
        r_dacc=res.get("Dir. Acc (%)"); r_lo=res.get("lower"); r_hi=res.get("upper")
        r_cov=coverage(actual,r_lo,r_hi) if r_lo is not None else None
        r_wkl=winkler_score(actual,r_lo,r_hi) if r_lo is not None else None
        vs_val=None
        if isinstance(r_rmse,float) and isinstance(naif_rmse,float) and naif_rmse>0:
            vs_val=(naif_rmse-r_rmse)/naif_rmse*100; vs_str=f"{vs_val:+.1f}%"
        else: vs_str="--"
        m_errs=actual-np.asarray(res["predictions"],float)
        dm,p=diebold_mariano(m_errs,naif_errs)
        dm_str=f"W ({p:.2f})" if p<0.05 and dm<0 else (f"L ({p:.2f})" if p<0.05 else f"T ({p:.2f})")
        mdata.append({"k":k,"rmse":r_rmse,"crps":r_crps,"dacc":r_dacc,
                      "cov":r_cov,"wkl":r_wkl,"vs_val":vs_val,"vs_str":vs_str,"dm_str":dm_str})
        rows.append([MODEL_LABELS[k],
                     f"{r_rmse:.2f}" if isinstance(r_rmse,float) else "--",
                     f"{r_crps:.4f}" if isinstance(r_crps,float) else "--",
                     f"{r_dacc:.1f}" if isinstance(r_dacc,float) else "--",
                     f"{r_cov:.1f}" if r_cov is not None else "--",
                     f"{r_wkl:.1f}" if r_wkl is not None else "--",
                     vs_str, dm_str])
        colors.append([_W]*len(col_labels))
    rv=[(i,d["rmse"]) for i,d in enumerate(mdata) if isinstance(d["rmse"],float)]
    if rv:
        colors[min(rv,key=lambda x:x[1])[0]+1][1]=_G
        if isinstance(naif_rmse,float):
            [colors.__setitem__(i+1,colors[i+1][:1]+[_R]+colors[i+1][2:])
             for i,v in rv if v>naif_rmse and colors[i+1][1]!=_G]
    cv2=[(i,d["crps"]) for i,d in enumerate(mdata) if isinstance(d["crps"],float)]
    if cv2: colors[min(cv2,key=lambda x:x[1])[0]+1][2]=_G
    dv=[(i,d["dacc"]) for i,d in enumerate(mdata) if isinstance(d["dacc"],float)]
    if dv:
        colors[max(dv,key=lambda x:x[1])[0]+1][3]=_G
        if isinstance(naif_dacc,float):
            for i,v in dv:
                if v<naif_dacc and colors[i+1][3]!=_G: colors[i+1][3]=_R
    cv4=[(i,d["cov"]) for i,d in enumerate(mdata) if d["cov"] is not None]
    if cv4:
        bi,bv=min(cv4,key=lambda x:abs(x[1]-95))
        if 90<=bv<=100: colors[bi+1][4]=_G
        for i,v in cv4:
            if (v<90 or v>100) and colors[i+1][4]!=_G: colors[i+1][4]=_R
    wv=[(i,d["wkl"]) for i,d in enumerate(mdata) if d["wkl"] is not None]
    if wv:
        colors[min(wv,key=lambda x:x[1])[0]+1][5]=_G
        if isinstance(naif_wkl,float):
            for i,v in wv:
                if v>naif_wkl and colors[i+1][5]!=_G: colors[i+1][5]=_R
    for i,d in enumerate(mdata):
        if d["vs_val"] is not None: colors[i+1][6]=_G if d["vs_val"]>0 else _R
    for i,d in enumerate(mdata):
        if d["dm_str"].startswith("W"): colors[i+1][7]=_G
        elif d["dm_str"].startswith("L"): colors[i+1][7]=_R
    return rows, col_labels, colors

# ── Fonctions de dessin ───────────────────────────────────────────────────────

def _draw_info(ax, train, test, ticker, freq, start, end, seed, events):
    ax.axis("off")
    _, flabel, _ = FREQ_CONFIG[freq]
    all_prices = pd.concat([train, test])
    vol        = float(np.std(np.diff(train.values)))
    last       = float(test.iloc[-1])
    ds = (f"Ticker    : {ticker}\n"
          f"Frequence : {flabel}\n"
          f"Periode   : {start}  ->  {end}\n"
          f"N total   : {len(all_prices)} obs   Seed : {seed}\n\n"
          f"Train : {len(train):>4} pts  "
          f"({str(train.index[0].date())} -> {str(train.index[-1].date())})\n"
          f"Test  : {len(test):>4} pts  "
          f"({str(test.index[0].date())}  -> {str(test.index[-1].date())})\n\n"
          f"Prix  : min={all_prices.min():.2f}$  max={all_prices.max():.2f}$\n"
          f"        last={last:.2f}$ ({str(test.index[-1].date())})\n"
          f"Vol.  : 1sig={vol:.2f}$  2sig={2*vol:.2f}$")
    ax.text(0.01, 0.98, ds, transform=ax.transAxes, fontsize=9, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5",facecolor="white",edgecolor="#b0b8c8",alpha=0.95))
    if events:
        ev_lines = []
        for d,lbl,cat,num in events[:20]:
            ev_lines.append(f" {num:>2}. [{EVENT_CAT_LABEL.get(cat,cat)[:3].upper()}] "
                            f"{str(d.date())}  {lbl}")
        if len(events)>20: ev_lines.append(f" ... (+{len(events)-20} autres)")
        ev_lines += [""]
        for cat,col in EVENT_COLORS.items():
            if any(c==cat for _,_,c,_ in events):
                ev_lines.append(f" [{EVENT_CAT_LABEL[cat][:3].upper()}] = {EVENT_CAT_LABEL[cat]}")
        ax.text(0.44, 0.98, "\n".join(ev_lines), transform=ax.transAxes,
                fontsize=8, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",facecolor="white",edgecolor="#b0b8c8",alpha=0.95))
    ax.set_title("Resume du dataset et evenements marquants",
                 fontsize=10, pad=3, loc="left", color="#444444")


def _draw_ts(ax, train, test, results, naive, events, show_pi):
    from matplotlib.lines import Line2D
    ax.plot(train.index, train.values, color="#cccccc", lw=0.9, label="Train")
    ax.plot(test.index,  test.values,  color="black",   lw=1.5, label="Reel (test)", zorder=5)
    ax.plot(naive["index"], naive["predictions"],
            color="#888888", lw=0.9, ls="--", alpha=0.75, label="Naif")
    for k,res in results.items():
        ax.plot(res["index"],res["predictions"],color=MODEL_COLORS[k],lw=1.3,alpha=0.85,
                label=MODEL_LABELS[k])
        if show_pi and res.get("lower") is not None:
            ax.fill_between(res["index"],res["lower"],res["upper"],
                            color=MODEL_COLORS[k],alpha=0.09)
    if show_pi:
        ax.fill_between(naive["index"],naive["lower"],naive["upper"],
                        color="#888888",alpha=0.07,label="PI naif 95%")
    xform = ax.get_xaxis_transform()
    cats_seen = set()
    for d,lbl,cat,num in events:
        col = EVENT_COLORS.get(cat,"#888888")
        ax.axvline(d, color=col, lw=0.9, ls="--", alpha=0.65, zorder=1)
        ax.text(d, 0.985, f" {num}", transform=xform, rotation=90,
                va="top", ha="right", fontsize=6.5, color=col, alpha=0.9, fontweight="bold")
        cats_seen.add(cat)
    leg1 = ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    ax.add_artist(leg1)
    if cats_seen:
        ev_h = [Line2D([0],[0],color=EVENT_COLORS[c],lw=1.5,ls="--",
                       label=EVENT_CAT_LABEL[c]) for c in EVENT_COLORS if c in cats_seen]
        ax.legend(handles=ev_h,fontsize=7.5,loc="lower right",title="Evenements",
                  title_fontsize=7.5,framealpha=0.9,edgecolor="#cccccc")
    ax.set_ylabel("Prix ($)",fontsize=10)
    ax.set_title("Walk-forward sur test set"+(" + PI 95%" if show_pi else "")
                 +(f"  |  {len(events)} evenements" if events else ""),fontsize=10,pad=4)
    ax.grid(True,alpha=0.25,lw=0.5); ax.set_facecolor("#fafbfc")


def _draw_kpi(ax, results, naive):
    rows, col_labels, cell_colors = _build_kpi_table(results, naive)
    ax.axis("off")
    ax.set_title("KPIs — meilleur en vert / pire que naif en rouge  "
                 "|  vs Naif = amelioration RMSE  |  DM/Naif = Diebold-Mariano",
                 fontsize=9, fontweight="bold", pad=5, loc="left")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor("#3a5a8a"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    for j in range(len(col_labels)):
        tbl[1,j].set_text_props(fontstyle="italic",color="#555555")
    for i in range(len(rows)+1):
        tbl[i,0].set_text_props(ha="left")


def _draw_ohlc(ax, next_preds, naive, train, models):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    last_known  = float(np.asarray(naive["actual"],float)[-1])
    sigma_naive = float(np.std(np.diff(train.values)))
    bar_items = [{"key":"naif","label":"Naif (y_t)","color":"#888888",
                  "pred":last_known,"lo95":last_known-1.96*sigma_naive,
                  "hi95":last_known+1.96*sigma_naive,"sigma":sigma_naive}]
    for k in models:
        if k not in next_preds: continue
        pred,lo95,hi95=next_preds[k]
        sig=(hi95-lo95)/(2*1.96) if (hi95-lo95)>0 else 0.
        bar_items.append({"key":k,"label":MODEL_LABELS[k],"color":MODEL_COLORS[k],
                          "pred":pred,"lo95":lo95,"hi95":hi95,"sigma":sig})
    if len(next_preds)>=2:
        ep=[next_preds[k][0] for k in next_preds]; el=[next_preds[k][1] for k in next_preds]
        eh=[next_preds[k][2] for k in next_preds]
        bar_items.append({"key":"ensemble","label":"Ensemble","color":"darkorange",
                          "pred":float(np.mean(ep)),"lo95":float(np.min(el)),"hi95":float(np.max(eh)),
                          "sigma":float(np.mean([(h-l)/(2*1.96) for l,h in zip(el,eh)]))})
    x_min=min(b["lo95"] for b in bar_items); x_max=max(b["hi95"] for b in bar_items)
    x_rng=max(x_max-x_min,1.); toff=x_rng*0.012
    n=len(bar_items); ys=list(range(n-1,-1,-1))
    ax.axvline(last_known,color="black",lw=1.5,ls="--",zorder=10)
    yticks,ylabels=[],[]
    for i,item in enumerate(bar_items):
        y,pred,lo95,hi95,sig,col = ys[i],item["pred"],item["lo95"],item["hi95"],item["sigma"],item["color"]
        blo,bhi=pred-sig,pred+sig
        ax.plot([lo95,hi95],[y,y],color=col,lw=1.8,alpha=0.55,zorder=2)
        for xc in [lo95,hi95]: ax.plot([xc,xc],[y-.18,y+.18],color=col,lw=1.8,alpha=0.55,zorder=2)
        ax.barh(y,bhi-blo,left=blo,height=0.44,color=col,alpha=0.80,zorder=3,linewidth=0.5,edgecolor="white")
        ax.plot([pred,pred],[y-.22,y+.22],color="white",lw=2.,zorder=4)
        ax.text(hi95+toff,y,f"{pred:.0f}$",va="center",ha="left",fontsize=8.5,color=col,fontweight="bold")
        yticks.append(y); ylabels.append(item["label"])
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels,fontsize=9.5)
    if n>=2: ax.axhline(ys[-1]+0.5,color="#cccccc",lw=0.8,ls="--",zorder=1)
    margin=x_rng*0.08; ax.set_xlim(x_min-margin,x_max+x_rng*0.16); ax.set_ylim(-.7,n-.3)
    ax.set_xlabel("Prix ($)",fontsize=10)
    ax.set_title("Previsions J+1 (next-step)  |  Corps:1sigma(68%)  |  Moustaches:95%PI"
                 f"  |  Ref:{last_known:.2f}$",fontsize=10,pad=4)
    ax.legend(handles=[
        Patch(facecolor="#888888",alpha=0.8,label="Corps = pred +/- 1 sigma"),
        Line2D([0],[0],color="#888888",lw=1.8,alpha=0.55,label="Moustaches = PI 95%"),
        Line2D([0],[0],color="black",lw=1.5,ls="--",label=f"Last close: {last_known:.2f}$"),
        Patch(facecolor="darkorange",alpha=0.8,label="Ensemble"),
    ],fontsize=8,loc="lower right",framealpha=0.92,edgecolor="#cccccc")
    ax.grid(True,axis="x",alpha=0.25,lw=0.5); ax.set_facecolor("#fafbfc")


def _draw_zoom(ax, full_series, next_preds_n, models, n_days=7):
    """
    Zoom sur les n_days derniers jours reels + errorbars par modele a J+1 et J+2.
    J+2 est speculatif (pred J+1 utilisee comme input, pas de valeur observee).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    last_n    = full_series.iloc[-n_days:]
    last_val  = float(full_series.iloc[-1])
    last_date = full_series.index[-1]

    # Pas de temps typique
    idx       = pd.DatetimeIndex(full_series.index)
    med_days  = float(np.median(np.diff(idx.asi8) / 1e9 / 86400))
    step_td   = pd.Timedelta(days=max(0.04, med_days))

    # Dates futures : J+1, J+2, ...
    n_steps = max((len(v) for v in next_preds_n.values()), default=0) if next_preds_n else 0
    future_dates = []
    d = last_date
    for _ in range(n_steps):
        d = d + step_td
        future_dates.append(d)

    # Ligne des prix reels
    ax.plot(last_n.index, last_n.values,
            "o-", color="black", lw=2.2, ms=6, zorder=6, label="Prix reels")
    ax.plot(last_date, last_val,
            "s", color="black", ms=11, zorder=7,
            markerfacecolor="white", markeredgewidth=2)

    # Separateur present / futur
    ax.axvline(last_date, color="#555555", lw=1.5, ls="--", alpha=0.55, zorder=2)
    ax.text(last_date, 0.975, "  Aujourd'hui",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", alpha=0.75))

    # Zone future ombree
    if future_dates:
        x_end = future_dates[-1] + step_td * 0.65
        ax.axvspan(last_date, x_end, alpha=0.055, color="#e67e22", zorder=0)

    # Labels J+1 / J+2 en haut
    for si, fd in enumerate(future_dates):
        lbl = f"J+{si+1}" + (" (spec.)" if si > 0 else "")
        ax.text(fd, 0.975, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5,
                fontweight="bold" if si == 0 else "normal",
                color="#333333" if si == 0 else "#888888",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.8))

    # Offsets horizontaux pour separer les modeles visuellement
    model_keys = [k for k in models if k in next_preds_n]
    n_m        = len(model_keys)
    raw_off    = np.linspace(-0.30, 0.30, n_m) if n_m > 1 else np.array([0.0])

    legend_handles = [
        Line2D([0],[0], color="black", lw=2, marker="o", ms=6, label="Prix reels"),
        Line2D([0],[0], color="black", lw=0, marker="s", ms=9,
               markerfacecolor="white", markeredgewidth=2, label=f"Last close: {last_val:.2f}$"),
    ]

    for mi, key in enumerate(model_keys):
        steps = next_preds_n[key]
        col   = MODEL_COLORS[key]
        lbl   = MODEL_LABELS[key]
        off_h = raw_off[mi] * 24        # fraction de jour en heures
        off_td = pd.Timedelta(hours=off_h)

        for si, (pred, lo, hi) in enumerate(steps):
            fd      = future_dates[si] + off_td
            alpha   = 1.0 if si == 0 else 0.45
            capsize = 6 if si == 0 else 4
            ms_v    = 10 if si == 0 else 7
            lw_eb   = 2.2 if si == 0 else 1.5
            fmt     = "o" if si == 0 else "^"

            ax.errorbar(
                fd, pred,
                yerr=[[max(0., pred - lo)], [max(0., hi - pred)]],
                fmt=fmt, color=col, ecolor=col,
                elinewidth=lw_eb, capsize=capsize, capthick=1.6,
                ms=ms_v, alpha=alpha, zorder=4,
            )
            # Etiquette prix
            ax.text(fd, hi + (hi - lo) * 0.05 + last_val * 0.002,
                    f"{pred:.0f}$",
                    ha="center", va="bottom",
                    fontsize=7.5 if si == 0 else 7,
                    color=col, alpha=alpha,
                    fontweight="bold" if si == 0 else "normal")

        # Une seule entree legende par modele (J+1)
        legend_handles.append(
            Line2D([0],[0], color=col, lw=0, marker="o", ms=8, label=lbl)
        )

    # Bande de consensus (min/max des PI sur tous les modeles) par pas
    for si in range(n_steps):
        avail = [next_preds_n[k][si] for k in model_keys if si < len(next_preds_n[k])]
        if len(avail) < 2: continue
        lo_min = min(t[1] for t in avail)
        hi_max = max(t[2] for t in avail)
        fd_c   = future_dates[si]
        half   = step_td * 0.38
        ax.fill_betweenx(
            [lo_min, hi_max],
            [fd_c - half], [fd_c + half],
            alpha=0.07, color="#444444", zorder=0,
        )

    # Ensemble marker
    for si in range(n_steps):
        avail = [next_preds_n[k][si] for k in model_keys if si < len(next_preds_n[k])]
        if len(avail) < 2: continue
        ens_p = float(np.mean([t[0] for t in avail]))
        ens_l = float(np.min([t[1] for t in avail]))
        ens_h = float(np.max([t[2] for t in avail]))
        fd    = future_dates[si]
        alpha_ens = 1.0 if si == 0 else 0.45

        ax.errorbar(fd, ens_p,
                    yerr=[[max(0., ens_p - ens_l)], [max(0., ens_h - ens_p)]],
                    fmt="*", color="darkorange", ecolor="darkorange",
                    elinewidth=2.5, capsize=7, capthick=2,
                    ms=15, alpha=alpha_ens, zorder=5,
                    label=None)
        ax.text(fd, ens_h + (ens_h - ens_l) * 0.05 + last_val * 0.002,
                f"~{ens_p:.0f}$",
                ha="center", va="bottom",
                fontsize=8, color="darkorange",
                fontweight="bold", alpha=alpha_ens)

    if n_m >= 2:
        legend_handles.append(
            Line2D([0],[0], color="darkorange", lw=0, marker="*", ms=12, label="Ensemble")
        )
    legend_handles += [
        Patch(facecolor="none", edgecolor="none", alpha=0, label="--"),
        Line2D([0],[0], color="darkorange", lw=0, marker="^", ms=7,
               alpha=0.5, label="J+2 (speculatif)"),
    ]

    # Limites axes
    all_y = list(last_n.values)
    for k in model_keys:
        for pred, lo, hi in next_preds_n[k]:
            all_y += [lo, hi]
    y_min, y_max = min(all_y), max(all_y)
    y_rng = max(y_max - y_min, 1.)
    ax.set_ylim(y_min - y_rng * 0.08, y_max + y_rng * 0.22)

    x_start = last_n.index[0] - step_td * 0.4
    x_end   = (future_dates[-1] if future_dates else last_date) + step_td * 0.8
    ax.set_xlim(x_start, x_end)

    ax.set_ylabel("Prix ($)", fontsize=10)
    spec_note = "  |  J+2 = speculatif (pred J+1 utilisee comme input)" if n_steps >= 2 else ""
    ax.set_title(f"Zoom J-{n_days} : {n_days} derniers jours + Previsions court terme{spec_note}",
                 fontsize=10, pad=4)
    ax.legend(handles=legend_handles, fontsize=8, ncol=min(len(legend_handles), 5),
              loc="lower left", framealpha=0.92, edgecolor="#cccccc")
    ax.grid(True, alpha=0.22, lw=0.5)
    ax.set_facecolor("#fafbfc")

# ── Fenetres interactives ─────────────────────────────────────────────────────

def plot_timeseries_panel(train, test, results, naive, ticker, freq,
                          start, end, seed, models, show_pi=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _, flabel, _ = FREQ_CONFIG[freq]
    events = _filter_events(start, end)
    fig = plt.figure(figsize=(16,11), num=f"DEITA 1.5 — Serie temporelle | {ticker}")
    fig.patch.set_facecolor("#f7f8fa")
    gs = gridspec.GridSpec(2,1,height_ratios=[1.7,3.9],hspace=0.38,
                           left=0.07,right=0.97,top=0.95,bottom=0.06,figure=fig)
    fig.suptitle(f"DEITA Benchmark 1.5  |  {ticker}  ({flabel})",
                 fontsize=13,fontweight="bold",y=0.99)
    _draw_info(fig.add_subplot(gs[0]), train, test, ticker, freq, start, end, seed, events)
    _draw_ts(fig.add_subplot(gs[1]), train, test, results, naive, events, show_pi)
    return fig


def plot_performance_panel(results, naive, next_preds_n, train, models, ticker):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    j1 = _j1_preds(next_preds_n)
    fig = plt.figure(figsize=(16,10),
                     num=f"DEITA 1.5 — Performance & Prevision J+1 | {ticker}")
    fig.patch.set_facecolor("#f7f8fa")
    gs = gridspec.GridSpec(2,1,height_ratios=[2.2,2.4],hspace=0.52,
                           left=0.08,right=0.97,top=0.94,bottom=0.06,figure=fig)
    fig.suptitle(f"DEITA Benchmark 1.5  |  {ticker}  — Performance & Prevision J+1",
                 fontsize=13,fontweight="bold",y=0.99)
    _draw_kpi(fig.add_subplot(gs[0]), results, naive)
    _draw_ohlc(fig.add_subplot(gs[1]), j1, naive, train, models)
    return fig


def plot_zoom_panel(full_series, next_preds_n, models, ticker, freq, n_days=7):
    import matplotlib.pyplot as plt
    _, flabel, _ = FREQ_CONFIG[freq]
    n_steps = max((len(v) for v in next_preds_n.values()), default=1) if next_preds_n else 1
    step_lbl = f"J+1{'  +  J+2 speculatif' if n_steps >= 2 else ''}"
    fig = plt.figure(figsize=(14, 7),
                     num=f"DEITA 1.5 — Zoom J-{n_days} & {step_lbl} | {ticker}")
    fig.patch.set_facecolor("#f7f8fa")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.10)
    fig.suptitle(
        f"DEITA Benchmark 1.5  |  {ticker}  ({flabel})"
        f"  —  Zoom J-{n_days} & Previsions court terme",
        fontsize=12, fontweight="bold")
    _draw_zoom(ax, full_series, next_preds_n, models, n_days=n_days)
    return fig

# ── PNG combine (5 panneaux) ──────────────────────────────────────────────────

def save_combined_dashboard(train, test, results, naive, next_preds_n,
                            ticker, freq, start, end, seed, models,
                            path, show_pi=True, n_days=7):
    """
    Cree et sauvegarde un PNG 5 panneaux :
      [1] Resume dataset + evenements
      [2] Serie temporelle + PI + evenements
      [3] Tableau KPI
      [4] Barres OHLC J+1
      [5] Zoom J-N + errorbars J+1/J+2
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _, flabel, _ = FREQ_CONFIG[freq]
    events = _filter_events(start, end)
    j1     = _j1_preds(next_preds_n)

    fig = plt.figure(figsize=(16, 28))
    fig.patch.set_facecolor("#f7f8fa")
    fig.suptitle(f"DEITA Benchmark 1.5  |  {ticker}  ({flabel})",
                 fontsize=14, fontweight="bold", y=0.994)
    gs = gridspec.GridSpec(5, 1,
                           height_ratios=[1.5, 3.5, 1.9, 2.3, 2.8],
                           hspace=0.42,
                           left=0.07, right=0.97,
                           top=0.988, bottom=0.020,
                           figure=fig)

    _draw_info(fig.add_subplot(gs[0]), train, test, ticker, freq, start, end, seed, events)
    _draw_ts(fig.add_subplot(gs[1]), train, test, results, naive, events, show_pi)
    _draw_kpi(fig.add_subplot(gs[2]), results, naive)
    _draw_ohlc(fig.add_subplot(gs[3]), j1, naive, train, models)
    _draw_zoom(fig.add_subplot(gs[4]), pd.concat([train, test]),
               next_preds_n, models, n_days=n_days)

    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PNG] Dashboard sauvegarde -> {path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global _USE_COLOR
    p = argparse.ArgumentParser(
        description="DEITA Benchmark 1.5 - Zoom J-7 + Previsions J+1/J+2",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--ticker",      default="ETH-USD")
    p.add_argument("--start",       default="2020-01-01")
    p.add_argument("--end",         default="2024-12-31")
    p.add_argument("--freq",        default="J", choices=["J","H","S"])
    p.add_argument("--test-ratio",  type=float, default=0.15)
    p.add_argument("--models",      nargs="+", default=AVAILABLE_MODELS,
                   choices=AVAILABLE_MODELS, metavar="MODEL")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--conformal",   action="store_true",
                   help="Calibration conforme des PI (val set = 20%% du train)")
    p.add_argument("--dashboard",   metavar="PATH", default=None,
                   help="Chemin PNG custom (defaut: deita_dashboard.png)")
    p.add_argument("--no-dashboard",action="store_true",
                   help="Terminal uniquement, pas de visuels")
    p.add_argument("--zoom-days",   type=int, default=7, metavar="N",
                   help="Nombre de jours de contexte dans la fenetre zoom (defaut: 7)")
    p.add_argument("--zoom-steps",  type=int, default=2, metavar="N",
                   help="Nombre de pas a predire J+1..J+N (defaut: 2)")
    p.add_argument("--save-csv",    metavar="PATH", default=None)
    p.add_argument("--save-data",   metavar="PATH", default=None)
    p.add_argument("--no-pi",       action="store_true")
    p.add_argument("--no-color",    action="store_true")
    args = p.parse_args()

    if args.no_color: _USE_COLOR = False

    _, _, swarn = FREQ_CONFIG[args.freq]
    if swarn and "sarima" in args.models:
        print("[WARN] SARIMA saisonnalite (s=5j) inadaptee au mode hebdomadaire.\n")
    if args.freq == "H":
        print("[INFO] Frequence horaire : lookback yfinance ~60 jours.\n")

    prices = fetch_prices(args.ticker, args.start, args.end, args.freq)
    train, test = split_series(prices, args.test_ratio)
    print(f"[SPLIT] Train : {len(train)} pts "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test  : {len(test)}  pts "
          f"({test.index[0].date()}  -> {test.index[-1].date()})\n")
    if len(train) < 60:
        print("[WARN] Moins de 60 pts : LSTM risque d'etre instable.\n")
    if args.save_data:
        prices.to_csv(args.save_data, header=["Close"])
        print(f"[DATA] Prix -> {args.save_data}\n")

    naive   = compute_naive(train, test)
    results = run_benchmark(train, test, args.models, args.seed)
    if not results:
        sys.exit("[ERREUR] Aucun modele n'a produit de resultat.")

    if args.conformal:
        print("\n[CONF] Calibration conforme (derniers 20% du train) ...")
        results = conformal_calibrate(results, train, args.seed)
        for k in results:
            if results[k].get("lower") is not None:
                results[k]["CRPS"] = crps_gaussian_approx(
                    results[k]["predictions"], results[k]["lower"],
                    results[k]["upper"], results[k]["actual"])
        print()

    df_t1 = print_table1(results, naive, args.ticker, args.freq,
                         args.start, args.end, len(train), len(test))
    df_t2 = print_table2(results, naive, args.ticker, args.freq)
    df_t3 = print_table3(results, naive, args.ticker, args.freq)

    if args.save_csv:
        base, ext = os.path.splitext(args.save_csv); ext = ext or ".csv"
        for suf,df in [("_t1",df_t1),("_t2",df_t2),("_t3",df_t3)]:
            df.to_csv(f"{base}{suf}{ext}"); print(f"[CSV] -> {base}{suf}{ext}")

    if args.no_dashboard:
        return

    # Previsions multi-pas (J+1 et J+2 par defaut)
    n_zoom_steps = max(1, args.zoom_steps)
    print(f"\n[NEXT] Calcul des previsions J+1..J+{n_zoom_steps} ...")
    full_series  = pd.concat([train, test])
    next_preds_n = compute_next_steps_n(full_series, args.models, args.seed,
                                        n_steps=n_zoom_steps)

    # PNG 5 panneaux
    dash_path = args.dashboard or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deita_dashboard.png")
    print(f"\n[PNG] Generation du dashboard 5 panneaux ...")
    save_combined_dashboard(
        train, test, results, naive, next_preds_n,
        args.ticker, args.freq, args.start, args.end,
        args.seed, args.models, dash_path,
        show_pi=not args.no_pi,
        n_days=args.zoom_days,
    )

    # Fenetres interactives
    print("[DASH] Ouverture des 3 fenetres interactives ...")
    fig1 = plot_timeseries_panel(
        train, test, results, naive,
        ticker=args.ticker, freq=args.freq,
        start=args.start, end=args.end,
        seed=args.seed, models=args.models,
        show_pi=not args.no_pi,
    )
    fig2 = plot_performance_panel(
        results, naive, next_preds_n, train, args.models, args.ticker)
    fig3 = plot_zoom_panel(
        full_series, next_preds_n, args.models,
        ticker=args.ticker, freq=args.freq,
        n_days=args.zoom_days,
    )

    import matplotlib.pyplot as plt
    print("[DASH] Fermer les fenetres pour quitter.\n")
    plt.show()


if __name__ == "__main__":
    main()
