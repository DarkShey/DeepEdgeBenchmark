"""
DEITA Benchmark 1.3 - Dashboard Interactif
==========================================
Extension de Benchmark 1.2 avec fenetres matplotlib interactives.

Nouveautes v1.3
---------------
  - Fenetres interactives : 2 fenetres matplotlib s'ouvrent a l'execution
    (plus besoin de PNG / visionneur systeme).
  - Fenetre 1 — Serie temporelle + contexte :
      * Panneau recapitulatif (dataset + legende des evenements numerotes)
      * Graphique de prix avec evenements marquants en traits verticaux colores
        (crypto / macro / monetaire / geopolitique) + numeros
  - Fenetre 2 — Performance & Prevision J+1 :
      * Tableau KPI avec mise en couleur : vert = meilleur / rouge = pire que naif
        Colonnes : RMSE | CRPS | Dir.Acc% | Coverage% | Winkler | vs Naif | DM/Naif
      * Barres OHLC horizontales : corps 1sigma + moustaches 95% + barre ensemble
  - Toutes les sorties terminal de Benchmark 1.1/1.2 sont conservees.

Usage
-----
  python "Benchmark 1.3.py"
  python "Benchmark 1.3.py" --ticker BTC-USD --models arima sarima
  python "Benchmark 1.3.py" --dashboard bench.png   # sauvegarde optionnelle
  python "Benchmark 1.3.py" --no-dashboard           # terminal uniquement
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))

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

# ── Evenements marquants ──────────────────────────────────────────────────────
# Format : "YYYY-MM-DD": (label, categorie)
# Categories : crypto / macro / monetaire / geopolitique

MARKET_EVENTS = {
    "2018-01-17": ("BTC ATH $20k (dec.)",   "crypto"),
    "2018-12-15": ("BTC bas $3.2k",          "crypto"),
    "2019-06-18": ("Annonce Libra (Meta)",   "crypto"),
    "2020-03-12": ("COVID crash -50%",       "macro"),
    "2020-03-15": ("Fed: taux 0%",           "monetaire"),
    "2020-05-11": ("BTC halving #3",         "crypto"),
    "2020-12-16": ("BTC franchit $20k",      "crypto"),
    "2021-01-29": ("WSB / GameStop",         "macro"),
    "2021-02-08": ("Tesla: 1.5G$ en BTC",   "crypto"),
    "2021-04-14": ("Coinbase IPO Nasdaq",    "crypto"),
    "2021-05-12": ("Tesla stop pmt BTC",     "crypto"),
    "2021-05-19": ("Chine ban crypto",       "geopolitique"),
    "2021-09-07": ("El Salvador: BTC legal", "geopolitique"),
    "2021-11-10": ("BTC ATH $69k",           "crypto"),
    "2022-01-05": ("Fed pivot hawkish",      "monetaire"),
    "2022-02-24": ("Invasion Ukraine",       "geopolitique"),
    "2022-03-16": ("Fed +25bp",              "monetaire"),
    "2022-05-05": ("Fed +50bp",              "monetaire"),
    "2022-05-09": ("LUNA/UST collapse",      "crypto"),
    "2022-06-13": ("Celsius gele fonds",     "crypto"),
    "2022-06-15": ("Fed +75bp",              "monetaire"),
    "2022-09-15": ("ETH Merge (PoS)",        "crypto"),
    "2022-11-08": ("FTX collapse",           "crypto"),
    "2023-03-10": ("SVB faillite",           "macro"),
    "2023-06-05": ("SEC vs Coinbase",        "geopolitique"),
    "2023-07-26": ("Fed pic: 5.25%",         "monetaire"),
    "2024-01-10": ("BTC ETF spot US",        "crypto"),
    "2024-03-14": ("BTC ATH $73k",           "crypto"),
    "2024-04-19": ("BTC halving #4",         "crypto"),
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

_GR = "\033[92m"
_RE = "\033[91m"
_YL = "\033[93m"
_BO = "\033[1m"
_RS = "\033[0m"
_USE_COLOR = True

import re as _re
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RS}" if _USE_COLOR else text

def _vlen(s: str) -> int:
    return len(_ANSI_RE.sub("", s))

def _rpad(s: str, width: int) -> str:
    return " " * max(0, width - _vlen(s)) + s

def _lpad(s: str, width: int) -> str:
    return s + " " * max(0, width - _vlen(s))

# ── Utilitaires ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _make_session():
    try:
        from curl_cffi.requests import Session as CurlSession
        return CurlSession(impersonate="chrome", verify=False)
    except Exception:
        return None


def fetch_prices(ticker: str, start: str, end: str, freq: str) -> pd.Series:
    interval, label, _ = FREQ_CONFIG[freq]
    print(f"[DATA] Telechargement {ticker} [{start} -> {end}] frequence={label} ...")
    session = _make_session()
    raw = yf.download(
        ticker, start=start, end=end,
        interval=interval, progress=False, auto_adjust=True,
        **({"session": session} if session is not None else {}),
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


def split_series(prices: pd.Series, test_ratio: float):
    split = int(len(prices) * (1 - test_ratio))
    return prices.iloc[:split], prices.iloc[split:]

# ── Benchmark naif ────────────────────────────────────────────────────────────

def compute_naive(train: pd.Series, test: pd.Series) -> dict:
    actual      = test.values.astype(float)
    naive_preds = np.concatenate([[train.iloc[-1]], actual[:-1]])
    residuals   = actual - naive_preds
    sigma       = float(np.std(np.diff(train.values)))
    return {
        "RMSE":           round(float(np.sqrt(np.mean(residuals**2))), 4),
        "CRPS":           "--",
        "MAE":            round(float(np.mean(np.abs(residuals))), 4),
        "MAPE (%)":       round(float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100), 2),
        "Dir. Acc (%)":   round(float(np.mean(
                              np.sign(np.diff(actual)) == np.sign(np.diff(naive_preds))
                          ) * 100), 2),
        "Train Time (s)": 0.0,
        "predictions":    naive_preds,
        "lower":          naive_preds - 1.96 * sigma,
        "upper":          naive_preds + 1.96 * sigma,
        "index":          test.index,
        "actual":         actual,
    }

# ── Metriques ─────────────────────────────────────────────────────────────────

def crps_gaussian_approx(predictions, lower, upper, actual, z: float = 1.96) -> float:
    from scipy.stats import norm as sp_norm
    mu    = np.asarray(predictions, float)
    lo    = np.asarray(lower,       float)
    hi    = np.asarray(upper,       float)
    y     = np.asarray(actual,      float)
    sigma = np.where((hi - lo) > 0, (hi - lo) / (2 * z), 1e-8)
    zsc   = (y - mu) / sigma
    return round(float(np.mean(
        sigma * (zsc * (2*sp_norm.cdf(zsc) - 1) + 2*sp_norm.pdf(zsc) - 1/np.sqrt(np.pi))
    )), 4)


def coverage(actual, lower, upper) -> float:
    y, lo, hi = map(np.asarray, (actual, lower, upper))
    return round(float(np.mean((y >= lo) & (y <= hi)) * 100), 2)


def avg_width(lower, upper) -> float:
    return round(float(np.mean(np.asarray(upper) - np.asarray(lower))), 2)


def winkler_score(actual, lower, upper, alpha: float = 0.05) -> float:
    y, lo, hi = map(np.asarray, (actual, lower, upper))
    width  = hi - lo
    pen_lo = np.where(y < lo, (2/alpha) * (lo - y), 0.0)
    pen_hi = np.where(y > hi, (2/alpha) * (y - hi), 0.0)
    return round(float(np.mean(width + pen_lo + pen_hi)), 2)


def diebold_mariano(errors_a, errors_b) -> tuple:
    from scipy.stats import norm as sp_norm
    d = np.asarray(errors_a, float)**2 - np.asarray(errors_b, float)**2
    T = len(d)
    if T < 5:
        return 0.0, 1.0
    dbar  = np.mean(d)
    trunc = max(1, int(np.floor(T ** (1/3))))
    var_d = np.var(d, ddof=0)
    for j in range(1, trunc + 1):
        cov_j = np.mean((d[j:] - dbar) * (d[:-j] - dbar))
        var_d += 2 * (1 - j / (trunc + 1)) * cov_j
    if var_d <= 0:
        return 0.0, 1.0
    DM    = dbar / np.sqrt(max(var_d, 1e-12) / T)
    p_val = 2 * (1 - sp_norm.cdf(abs(DM)))
    return round(float(DM), 3), round(float(p_val), 3)

# ── Chargement des modeles ────────────────────────────────────────────────────

def load_runner(model_key: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        if model_key == "arima":
            from arima_model import run_arima_garch; return run_arima_garch
        if model_key == "sarima":
            from sarima_model import run_sarima;     return run_sarima
        if model_key == "prophet":
            from prophet_model import run_prophet;   return run_prophet
        if model_key == "lstm":
            from lstm_model import run_lstm;         return run_lstm
    except ImportError as exc:
        print(f"[WARN] {MODEL_LABELS[model_key]} ignore : {exc}")
        return None

# ── Run ───────────────────────────────────────────────────────────────────────

def run_benchmark(train: pd.Series, test: pd.Series, models: list, seed: int) -> dict:
    results = {}
    for key in models:
        runner = load_runner(key)
        if runner is None:
            continue
        label = MODEL_LABELS[key]
        print(f"[RUN] {label} ...")
        set_seed(seed)
        try:
            result = runner(train, test)
        except Exception as exc:
            print(f"[ERREUR] {label} : {exc}")
            continue
        if result.get("lower") is not None and result.get("upper") is not None:
            result["CRPS"] = crps_gaussian_approx(
                result["predictions"], result["lower"], result["upper"], result["actual"]
            )
        else:
            result["CRPS"] = "--"
        results[key] = result
        print(f"[RUN] {label} | RMSE={result.get('RMSE','?')}  "
              f"CRPS={result.get('CRPS','?')}  "
              f"Dir.Acc={result.get('Dir. Acc (%)','?')}%\n")
    return results

# ── Tableau 1 ─────────────────────────────────────────────────────────────────

_METRIC_HEADERS = {
    "RMSE": "RMSE", "CRPS": "CRPS", "MAE": "MAE",
    "MAPE (%)": "MAPE%", "Dir. Acc (%)": "DirAcc%", "Train Time (s)": "Time(s)",
}

def print_table1(results, naive, ticker, freq, start, end, n_train, n_test):
    _, flabel, _ = FREQ_CONFIG[freq]
    metrics, all_keys, W = list(_METRIC_HEADERS.keys()), list(results.keys()), 80
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 1 - Performances principales', _BO)}")
    print(f"  {ticker}  ({flabel}, {start} -> {end})")
    print(f"  Train : {n_train} pts  |  Test : {n_test} pts")
    print(f"{'='*W}")
    col_m, col_v = 16, 10
    hdr = _lpad("Modele", col_m) + "".join(_rpad(_METRIC_HEADERS[m], col_v) for m in metrics)
    sep = "  " + "-" * (col_m + col_v * len(metrics))
    print(f"  {hdr}")
    print(sep)

    best_key = {}
    for m in metrics:
        vals = {k: results[k].get(m) for k in all_keys
                if isinstance(results[k].get(m), (int, float))}
        if not vals:       best_key[m] = None
        elif m == "Dir. Acc (%)": best_key[m] = max(vals, key=vals.get)
        else:              best_key[m] = min(vals, key=vals.get)

    def _render(val, m, mk, is_naive=False):
        if is_naive:       return _c(str(val), _YL)
        if not isinstance(val, (int, float)): return str(val)
        nv = naive.get(m)
        if best_key.get(m) == mk: return _c(str(val), _GR)
        if isinstance(nv, (int, float)):
            worse = val < nv if m == "Dir. Acc (%)" else val > nv
            if worse:      return _c(str(val) + "!", _RE)
        return str(val)

    def _row(label, data, mk=None, is_naive=False):
        lbl = _c(label, _YL) if is_naive else label
        row = f"  {_lpad(lbl, col_m)}"
        for m in metrics:
            row += _rpad(_render(data.get(m, "N/A"), m, mk, is_naive), col_v)
        print(row)

    _row("Naif (y_t)", naive, is_naive=True)
    print(sep)
    printed = set()
    for grp, keys in [("-- Statistique --", ["arima","sarima","prophet"]),
                      ("-- RNN --",         ["lstm"])]:
        in_grp = [k for k in all_keys if k in keys]
        if in_grp:
            print(f"  {grp}")
            for k in in_grp:
                _row(MODEL_LABELS[k], results[k], mk=k)
                printed.add(k)
    for k in [k for k in all_keys if k not in printed]:
        _row(MODEL_LABELS[k], results[k], mk=k)
    print(sep)
    print(f"  {_c('[V]',_GR)} meilleur  |  {_c('[!]',_RE)} pire que naif  "
          f"|  {_c('jaune',_YL)} = naif")
    print(f"{'='*W}\n")

    rows = {"Naif": {m: naive.get(m, "N/A") for m in metrics}}
    for k in all_keys:
        rows[MODEL_LABELS[k]] = {m: results[k].get(m, "N/A") for m in metrics}
    df = pd.DataFrame(rows).T; df.index.name = "Modele"
    return df

# ── Tableau 2 ─────────────────────────────────────────────────────────────────

def print_table2(results, naive, ticker, freq):
    _, flabel, _ = FREQ_CONFIG[freq]
    W = 70
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 2 - Test Diebold-Mariano (paires)', _BO)}")
    print(f"  {ticker}  ({flabel})  |  Ligne A vs Col B")
    print(f"{'='*W}")
    all_items = [("naif","Naif",naive)] + [(k,MODEL_LABELS[k],results[k]) for k in results]
    errors = {key: np.asarray(res["actual"],float) - np.asarray(res["predictions"],float)
              for key, _, res in all_items}
    names = [l for _,l,_ in all_items]
    col_w = max(14, max(len(n) for n in names) + 2)
    sep   = "  " + "-" * (col_w * (len(names) + 1))
    print(f"  {_lpad('', col_w)}" + "".join(_rpad(n, col_w) for n in names))
    print(sep)
    dm_rows = {}
    for ka, la, _ in all_items:
        row, line = {}, f"  {_rpad(la, col_w)}"
        for kb, lb, _ in all_items:
            if ka == kb:
                cell_str, cell_col = "--", "--"
            else:
                dm, p = diebold_mariano(errors[ka], errors[kb])
                if p < 0.05:
                    cell_str = f"L ({p:.2f})" if dm > 0 else f"W ({p:.2f})"
                    cell_col = _c(cell_str, _RE if dm > 0 else _GR)
                else:
                    cell_str = f"T ({p:.2f})"; cell_col = cell_str
            row[lb] = cell_str; line += _rpad(cell_col, col_w)
        print(line); dm_rows[la] = row
    print(sep)
    print(f"  {_c('W',_GR)} victoire  |  T egalite  |  {_c('L',_RE)} defaite")
    print(f"{'='*W}\n")
    df = pd.DataFrame(dm_rows).T; df.index.name = "vs"
    return df

# ── Tableau 3 ─────────────────────────────────────────────────────────────────

def print_table3(results, naive, ticker, freq, alpha=0.05):
    _, flabel, _ = FREQ_CONFIG[freq]
    cov_tgt, W = (1 - alpha) * 100, 70
    cov_lo, cov_hi = cov_tgt - 5, cov_tgt + 5
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 3 - Intervalles PI a {:.0f}%'.format(cov_tgt), _BO)}")
    print(f"  {ticker}  |  Cible : {cov_tgt:.0f}%  Acceptable : [{cov_lo:.0f}%, {cov_hi:.0f}%]")
    print(f"{'='*W}")
    col_m, col_v = 20, 14
    sep = "  " + "-" * (col_m + col_v * 3)
    print(f"  {_lpad('Methode', col_m)}"
          f"{_rpad('Coverage (%)', col_v)}{_rpad('Width ($)', col_v)}{_rpad('Winkler', col_v)}")
    print(sep)
    actual = np.asarray(naive["actual"], float)
    rows = {}

    def _fmt(label, lo, hi, ref_wkl=None, is_naive=False):
        cov_v = coverage(actual, lo, hi)
        wid_v = avg_width(lo, hi)
        wkl_v = winkler_score(actual, lo, hi, alpha)
        cov_s = _c(str(cov_v)+"!", _RE) if (cov_v<cov_lo or cov_v>cov_hi) else str(cov_v)
        wkl_s = (_c(str(wkl_v)+"!", _RE)
                 if ref_wkl is not None and wkl_v > ref_wkl and not is_naive
                 else str(wkl_v))
        lbl_s = _c(label, _YL) if is_naive else label
        print(f"  {_lpad(lbl_s, col_m)}{_rpad(cov_s, col_v)}"
              f"{_rpad(str(wid_v), col_v)}{_rpad(wkl_s, col_v)}")
        return {"Coverage (%)": cov_v, "Width ($)": wid_v, "Winkler": wkl_v}

    nw = _fmt("Gaussien naif", naive["lower"], naive["upper"], is_naive=True)
    rows["Gaussien naif"] = nw
    print(sep)
    for k, res in results.items():
        if res.get("lower") is None: continue
        rows[MODEL_LABELS[k]] = _fmt(MODEL_LABELS[k], res["lower"], res["upper"],
                                     ref_wkl=nw["Winkler"])
    print(sep)
    print(f"  {_c('[!]',_RE)} Coverage hors [{cov_lo:.0f}%,{cov_hi:.0f}%] | "
          f"Winkler > naif -> recalibrer")
    print(f"{'='*W}\n")
    df = pd.DataFrame(rows).T; df.index.name = "Methode"
    return df

# ── Previsions J+1 ────────────────────────────────────────────────────────────

def compute_next_steps(full_series: pd.Series, models: list, seed: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    next_preds = {}
    for key in models:
        label = MODEL_LABELS.get(key, key)
        print(f"[NEXT] {label} ...")
        set_seed(seed)
        try:
            if key == "arima":
                from arima_model import next_step_arima_garch
                pred, lo, hi = next_step_arima_garch(full_series)
            elif key == "sarima":
                from sarima_model import next_step_sarima
                pred, lo, hi = next_step_sarima(full_series)
            elif key == "prophet":
                from prophet_model import next_step_prophet
                pred, lo, hi = next_step_prophet(full_series)
            elif key == "lstm":
                from lstm_model import next_step_lstm
                pred, lo, hi = next_step_lstm(full_series)
            else:
                continue
            next_preds[key] = (float(pred), float(lo), float(hi))
            print(f"[NEXT] {label}: pred={pred:.2f}$  95%PI=[{lo:.2f}$, {hi:.2f}$]")
        except Exception as exc:
            print(f"[NEXT] {label}: ERREUR - {exc}")
    return next_preds

# ── Helpers dashboard ─────────────────────────────────────────────────────────

def _filter_events(start: str, end: str) -> list:
    """Retourne [(timestamp, label, categorie, num)] filtre sur [start, end]."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    evts = [
        (pd.Timestamp(d), lbl, cat)
        for d, (lbl, cat) in sorted(MARKET_EVENTS.items())
        if t0 <= pd.Timestamp(d) <= t1
    ]
    return [(d, lbl, cat, i + 1) for i, (d, lbl, cat) in enumerate(evts)]


def _build_kpi_table(results: dict, naive: dict) -> tuple:
    """
    Retourne (rows, col_labels, cell_colors) pour matplotlib table.
    rows et cell_colors incluent la ligne naif ET les modeles.
    """
    actual    = np.asarray(naive["actual"], float)
    naif_errs = actual - np.asarray(naive["predictions"], float)
    naif_rmse = naive.get("RMSE")
    naif_dacc = naive.get("Dir. Acc (%)", 0.0)
    naif_cov  = coverage(actual, naive["lower"], naive["upper"])
    naif_wkl  = winkler_score(actual, naive["lower"], naive["upper"])

    col_labels = ["Modele", "RMSE", "CRPS", "Dir.Acc%",
                  "Cover%", "Winkler", "vs Naif", "DM/Naif"]

    _GREEN = "#d0f0d0"
    _RED   = "#f7d0d0"
    _NAIF  = "#f5f5e6"
    _WHITE = "white"

    rows   = []
    colors = []

    # Ligne naif
    rows.append([
        "Naif (ref)",
        f"{naif_rmse:.2f}" if isinstance(naif_rmse, float) else "--",
        "--", f"{naif_dacc:.1f}", f"{naif_cov:.1f}", f"{naif_wkl:.1f}",
        "--", "--",
    ])
    colors.append([_NAIF] * len(col_labels))

    # Lignes modeles : collecter les donnees brutes avant de colorier
    mdata = []
    for k in results:
        res   = results[k]
        r_rmse = res.get("RMSE")
        r_crps = res.get("CRPS")
        r_dacc = res.get("Dir. Acc (%)")
        r_lo   = res.get("lower")
        r_hi   = res.get("upper")
        r_cov  = coverage(actual, r_lo, r_hi) if r_lo is not None else None
        r_wkl  = winkler_score(actual, r_lo, r_hi) if r_lo is not None else None

        vs_val = None
        if isinstance(r_rmse, float) and isinstance(naif_rmse, float) and naif_rmse > 0:
            vs_val = (naif_rmse - r_rmse) / naif_rmse * 100
            vs_str = f"{vs_val:+.1f}%"
        else:
            vs_str = "--"

        m_errs = actual - np.asarray(res["predictions"], float)
        dm, p  = diebold_mariano(m_errs, naif_errs)
        if p < 0.05:
            dm_str = f"W ({p:.2f})" if dm < 0 else f"L ({p:.2f})"
        else:
            dm_str = f"T ({p:.2f})"

        mdata.append({
            "k": k, "rmse": r_rmse, "crps": r_crps, "dacc": r_dacc,
            "cov": r_cov, "wkl": r_wkl, "vs_val": vs_val, "vs_str": vs_str,
            "dm_str": dm_str,
        })
        rows.append([
            MODEL_LABELS[k],
            f"{r_rmse:.2f}" if isinstance(r_rmse, float) else "--",
            f"{r_crps:.4f}" if isinstance(r_crps, float) else "--",
            f"{r_dacc:.1f}" if isinstance(r_dacc, float) else "--",
            f"{r_cov:.1f}" if r_cov is not None else "--",
            f"{r_wkl:.1f}" if r_wkl is not None else "--",
            vs_str, dm_str,
        ])
        colors.append([_WHITE] * len(col_labels))

    # RMSE (col 1) — lower is better
    rv = [(i, d["rmse"]) for i, d in enumerate(mdata) if isinstance(d["rmse"], float)]
    if rv:
        colors[min(rv, key=lambda x: x[1])[0] + 1][1] = _GREEN
        if isinstance(naif_rmse, float):
            for i, v in rv:
                if v > naif_rmse and colors[i+1][1] != _GREEN:
                    colors[i+1][1] = _RED

    # CRPS (col 2) — lower is better
    cv2 = [(i, d["crps"]) for i, d in enumerate(mdata) if isinstance(d["crps"], float)]
    if cv2:
        colors[min(cv2, key=lambda x: x[1])[0] + 1][2] = _GREEN

    # Dir.Acc (col 3) — higher is better
    dv = [(i, d["dacc"]) for i, d in enumerate(mdata) if isinstance(d["dacc"], float)]
    if dv:
        colors[max(dv, key=lambda x: x[1])[0] + 1][3] = _GREEN
        if isinstance(naif_dacc, float):
            for i, v in dv:
                if v < naif_dacc and colors[i+1][3] != _GREEN:
                    colors[i+1][3] = _RED

    # Coverage (col 4) — closest to 95%, outside [90,100] = rouge
    cv4 = [(i, d["cov"]) for i, d in enumerate(mdata) if d["cov"] is not None]
    if cv4:
        bi, bv = min(cv4, key=lambda x: abs(x[1] - 95))
        if 90 <= bv <= 100:
            colors[bi + 1][4] = _GREEN
        for i, v in cv4:
            if (v < 90 or v > 100) and colors[i+1][4] != _GREEN:
                colors[i+1][4] = _RED

    # Winkler (col 5) — lower is better
    wv = [(i, d["wkl"]) for i, d in enumerate(mdata) if d["wkl"] is not None]
    if wv:
        colors[min(wv, key=lambda x: x[1])[0] + 1][5] = _GREEN
        for i, v in wv:
            if isinstance(naif_wkl, float) and v > naif_wkl and colors[i+1][5] != _GREEN:
                colors[i+1][5] = _RED

    # vs Naif (col 6)
    for i, d in enumerate(mdata):
        if d["vs_val"] is not None:
            colors[i+1][6] = _GREEN if d["vs_val"] > 0 else _RED

    # DM/Naif (col 7)
    for i, d in enumerate(mdata):
        if d["dm_str"].startswith("W"):   colors[i+1][7] = _GREEN
        elif d["dm_str"].startswith("L"): colors[i+1][7] = _RED

    return rows, col_labels, colors

# ── Figure 1 : Serie temporelle + evenements ──────────────────────────────────

def plot_timeseries_panel(
    train, test, results, naive,
    ticker, freq, start, end, seed, models,
    show_pi=True,
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    _, flabel, _ = FREQ_CONFIG[freq]
    events = _filter_events(start, end)

    fig = plt.figure(figsize=(16, 11), num=f"DEITA 1.3 — Serie temporelle | {ticker}")
    fig.patch.set_facecolor("#f7f8fa")

    gs = gridspec.GridSpec(
        2, 1, height_ratios=[1.7, 3.9], hspace=0.38,
        left=0.07, right=0.97, top=0.95, bottom=0.06, figure=fig,
    )
    ax_info = fig.add_subplot(gs[0])
    ax_ts   = fig.add_subplot(gs[1])

    fig.suptitle(
        f"DEITA Benchmark 1.3  |  {ticker}  ({flabel})",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Panneau info (gauche) ────────────────────────────────────────────────
    ax_info.axis("off")
    all_prices  = pd.concat([train, test])
    vol_daily   = float(np.std(np.diff(train.values)))
    last_actual = float(test.iloc[-1])

    dataset_text = (
        f"Ticker    : {ticker}\n"
        f"Frequence : {flabel}\n"
        f"Periode   : {start}  ->  {end}\n"
        f"N total   : {len(all_prices)} obs   Seed : {seed}\n\n"
        f"Train : {len(train):>4} pts  "
        f"({str(train.index[0].date())} -> {str(train.index[-1].date())})\n"
        f"Test  : {len(test):>4} pts  "
        f"({str(test.index[0].date())}  -> {str(test.index[-1].date())})\n\n"
        f"Prix  : min={all_prices.min():.2f}$  max={all_prices.max():.2f}$\n"
        f"        last={last_actual:.2f}$ ({str(test.index[-1].date())})\n"
        f"Vol.  : 1sig={vol_daily:.2f}$  2sig={2*vol_daily:.2f}$"
    )
    ax_info.text(
        0.01, 0.98, dataset_text,
        transform=ax_info.transAxes, fontsize=9, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#b0b8c8", alpha=0.95),
    )

    # ── Legende des evenements (droite de ax_info) ──────────────────────────
    if events:
        max_ev_show = 20
        ev_shown = events[:max_ev_show]
        lines = []
        for d, lbl, cat, num in ev_shown:
            abbr = EVENT_CAT_LABEL.get(cat, cat)[:3].upper()
            lines.append(f" {num:>2}. [{abbr}] {str(d.date())}  {lbl}")
        if len(events) > max_ev_show:
            lines.append(f" ... (+{len(events) - max_ev_show} autres)")
        lines.append("")
        for cat, col in EVENT_COLORS.items():
            if any(c == cat for _, _, c, _ in events):
                lines.append(f" [{EVENT_CAT_LABEL[cat][:3].upper()}] = {EVENT_CAT_LABEL[cat]}")
        ax_info.text(
            0.44, 0.98, "\n".join(lines),
            transform=ax_info.transAxes, fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#b0b8c8", alpha=0.95),
        )

    ax_info.set_title("Resume du dataset et evenements marquants",
                      fontsize=10, pad=3, loc="left", color="#444444")

    # ── Graphique de prix ────────────────────────────────────────────────────
    ax_ts.plot(train.index, train.values,
               color="#cccccc", lw=0.9, label="Train", zorder=1)
    ax_ts.plot(test.index, test.values,
               color="black", lw=1.5, label="Reel (test)", zorder=5)
    ax_ts.plot(naive["index"], naive["predictions"],
               color="#888888", lw=0.9, ls="--", alpha=0.75, label="Naif")

    for k, res in results.items():
        ax_ts.plot(res["index"], res["predictions"],
                   color=MODEL_COLORS[k], lw=1.3, alpha=0.85, label=MODEL_LABELS[k])
        if show_pi and res.get("lower") is not None:
            ax_ts.fill_between(res["index"], res["lower"], res["upper"],
                               color=MODEL_COLORS[k], alpha=0.09)
    if show_pi:
        ax_ts.fill_between(naive["index"], naive["lower"], naive["upper"],
                           color="#888888", alpha=0.07, label="PI naif 95%")

    # Evenements sur le graphique
    xform = ax_ts.get_xaxis_transform()
    cats_seen = set()
    for d, lbl, cat, num in events:
        col = EVENT_COLORS.get(cat, "#888888")
        ax_ts.axvline(d, color=col, lw=0.9, ls="--", alpha=0.65, zorder=1)
        ax_ts.text(d, 0.985, f" {num}", transform=xform,
                   rotation=90, va="top", ha="right",
                   fontsize=6.5, color=col, alpha=0.9, fontweight="bold")
        cats_seen.add(cat)

    # Double legende : modeles (haut gauche) + categories evenements (bas droite)
    leg_models = ax_ts.legend(fontsize=8, ncol=4, loc="upper left",
                              framealpha=0.9, edgecolor="#cccccc")
    ax_ts.add_artist(leg_models)
    if cats_seen:
        ev_handles = [
            Line2D([0], [0], color=EVENT_COLORS[cat], lw=1.5, ls="--",
                   label=EVENT_CAT_LABEL[cat])
            for cat in EVENT_COLORS if cat in cats_seen
        ]
        ax_ts.legend(handles=ev_handles, fontsize=7.5, loc="lower right",
                     title="Evenements", title_fontsize=7.5,
                     framealpha=0.9, edgecolor="#cccccc")

    ax_ts.set_ylabel("Prix ($)", fontsize=10)
    ax_ts.set_title(
        "Walk-forward sur test set" + (" + PI 95%" if show_pi else "")
        + (f"  |  {len(events)} evenements" if events else ""),
        fontsize=10, pad=4,
    )
    ax_ts.grid(True, alpha=0.25, lw=0.5)
    ax_ts.set_facecolor("#fafbfc")

    return fig

# ── Figure 2 : Tableau KPI + OHLC J+1 ────────────────────────────────────────

def plot_performance_panel(results, naive, next_preds, train, models, ticker):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(16, 10),
                     num=f"DEITA 1.3 — Performance & Prevision J+1 | {ticker}")
    fig.patch.set_facecolor("#f7f8fa")

    gs = gridspec.GridSpec(
        2, 1, height_ratios=[2.2, 2.4], hspace=0.52,
        left=0.08, right=0.97, top=0.94, bottom=0.06, figure=fig,
    )
    ax_table = fig.add_subplot(gs[0])
    ax_ohlc  = fig.add_subplot(gs[1])

    fig.suptitle(
        f"DEITA Benchmark 1.3  |  {ticker}  —  Performance & Prevision J+1",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Tableau KPI ──────────────────────────────────────────────────────────
    rows, col_labels, cell_colors = _build_kpi_table(results, naive)

    ax_table.axis("off")
    ax_table.set_title(
        "KPIs — meilleur en vert / pire que naif en rouge\n"
        "  [vs Naif] = amelioration RMSE vs naif  |  [DM/Naif] = test Diebold-Mariano",
        fontsize=10, fontweight="bold", pad=5, loc="left",
    )

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    # En-tete : fond bleu, texte blanc gras
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#3a5a8a")
        cell.set_text_props(color="white", fontweight="bold")

    # Ligne naif : italique
    for j in range(len(col_labels)):
        table[1, j].set_text_props(fontstyle="italic", color="#555555")

    # Colonne Modele : alignement gauche
    for i in range(len(rows) + 1):
        table[i, 0].set_text_props(ha="left")

    # ── Barres OHLC J+1 ──────────────────────────────────────────────────────
    actual_arr  = np.asarray(naive["actual"], float)
    last_known  = float(actual_arr[-1])
    sigma_naive = float(np.std(np.diff(train.values)))

    bar_items = [
        {"key": "naif", "label": "Naif (y_t)", "color": "#888888",
         "pred": last_known,
         "lo95": last_known - 1.96 * sigma_naive,
         "hi95": last_known + 1.96 * sigma_naive,
         "sigma": sigma_naive},
    ]
    for k in models:
        if k not in next_preds:
            continue
        pred, lo95, hi95 = next_preds[k]
        sigma = (hi95 - lo95) / (2 * 1.96) if (hi95 - lo95) > 0 else 0.0
        bar_items.append({
            "key": k, "label": MODEL_LABELS[k], "color": MODEL_COLORS[k],
            "pred": pred, "lo95": lo95, "hi95": hi95, "sigma": sigma,
        })
    if len(next_preds) >= 2:
        ep = [next_preds[k][0] for k in next_preds]
        el = [next_preds[k][1] for k in next_preds]
        eh = [next_preds[k][2] for k in next_preds]
        bar_items.append({
            "key": "ensemble", "label": "Ensemble", "color": "darkorange",
            "pred":  float(np.mean(ep)),
            "lo95":  float(np.min(el)),
            "hi95":  float(np.max(eh)),
            "sigma": float(np.mean([(h - l) / (2 * 1.96) for l, h in zip(el, eh)])),
        })

    x_min  = min(b["lo95"] for b in bar_items)
    x_max  = max(b["hi95"] for b in bar_items)
    x_rng  = max(x_max - x_min, 1.0)
    toff   = x_rng * 0.012

    n  = len(bar_items)
    ys = list(range(n - 1, -1, -1))

    ax_ohlc.axvline(last_known, color="black", lw=1.5, ls="--", zorder=10)

    yticks, ylabels = [], []
    for i, item in enumerate(bar_items):
        y, pred = ys[i], item["pred"]
        lo95, hi95, sigma = item["lo95"], item["hi95"], item["sigma"]
        color, label = item["color"], item["label"]
        blo, bhi = pred - sigma, pred + sigma

        ax_ohlc.plot([lo95, hi95], [y, y], color=color, lw=1.8, alpha=0.55, zorder=2)
        for x_cap in [lo95, hi95]:
            ax_ohlc.plot([x_cap, x_cap], [y - 0.18, y + 0.18],
                         color=color, lw=1.8, alpha=0.55, zorder=2)
        ax_ohlc.barh(y, bhi - blo, left=blo, height=0.44,
                     color=color, alpha=0.80, zorder=3,
                     linewidth=0.5, edgecolor="white")
        ax_ohlc.plot([pred, pred], [y - 0.22, y + 0.22],
                     color="white", lw=2.0, zorder=4)
        ax_ohlc.text(hi95 + toff, y, f"{pred:.0f}$",
                     va="center", ha="left", fontsize=8.5,
                     color=color, fontweight="bold")
        yticks.append(y)
        ylabels.append(label)

    ax_ohlc.set_yticks(yticks)
    ax_ohlc.set_yticklabels(ylabels, fontsize=9.5)

    if n >= 2:
        ax_ohlc.axhline(ys[-1] + 0.5, color="#cccccc", lw=0.8, ls="--", zorder=1)

    margin = x_rng * 0.08
    ax_ohlc.set_xlim(x_min - margin, x_max + x_rng * 0.16)
    ax_ohlc.set_ylim(-0.7, n - 0.3)
    ax_ohlc.set_xlabel("Prix ($)", fontsize=10)
    ax_ohlc.set_title(
        "Previsions J+1 (next-step)  |  Corps : 1sigma (68%)  |  Moustaches : 95% PI"
        f"  |  Ref. last close : {last_known:.2f}$",
        fontsize=10, pad=4,
    )
    legend_elems = [
        Patch(facecolor="#888888", alpha=0.8, label="Corps = pred +/- 1 sigma (68%)"),
        Line2D([0], [0], color="#888888", lw=1.8, alpha=0.55, label="Moustaches = PI 95%"),
        Line2D([0], [0], color="black", lw=1.5, ls="--",
               label=f"Last close : {last_known:.2f}$"),
        Patch(facecolor="darkorange", alpha=0.8, label="Ensemble"),
    ]
    ax_ohlc.legend(handles=legend_elems, fontsize=8, loc="lower right",
                   framealpha=0.92, edgecolor="#cccccc")
    ax_ohlc.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax_ohlc.set_facecolor("#fafbfc")

    return fig

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    p = argparse.ArgumentParser(
        description="DEITA Benchmark 1.3 - Dashboard interactif (2 fenetres matplotlib)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ticker",       default="ETH-USD")
    p.add_argument("--start",        default="2020-01-01")
    p.add_argument("--end",          default="2024-12-31")
    p.add_argument("--freq",         default="J", choices=["J", "H", "S"])
    p.add_argument("--test-ratio",   type=float, default=0.15)
    p.add_argument("--models",       nargs="+", default=AVAILABLE_MODELS,
                   choices=AVAILABLE_MODELS, metavar="MODEL")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--dashboard",    metavar="PATH", default=None,
                   help="Sauvegarde optionnelle en PNG (ex: bench.png -> bench_ts.png + bench_perf.png)")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Pas de fenetres graphiques (terminal uniquement)")
    p.add_argument("--save-csv",     metavar="PATH", default=None)
    p.add_argument("--save-data",    metavar="PATH", default=None)
    p.add_argument("--no-pi",        action="store_true")
    p.add_argument("--no-color",     action="store_true")
    args = p.parse_args()

    if args.no_color:
        _USE_COLOR = False

    _, _, seasonal_warn = FREQ_CONFIG[args.freq]
    if seasonal_warn and "sarima" in args.models:
        print("[WARN] Frequence hebdomadaire : SARIMA (s=5j) inadapte.\n")
    if args.freq == "H":
        print("[INFO] Frequence horaire : lookback yfinance ~60 jours.\n")

    prices = fetch_prices(args.ticker, args.start, args.end, args.freq)
    train, test = split_series(prices, args.test_ratio)
    print(f"[SPLIT] Train : {len(train)} pts "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test  : {len(test)}  pts "
          f"({test.index[0].date()}  -> {test.index[-1].date()})\n")

    if len(train) < 60:
        print("[WARN] Moins de 60 pts d'entrainement : LSTM instable.\n")

    if args.save_data:
        prices.to_csv(args.save_data, header=["Close"])
        print(f"[DATA] Prix -> {args.save_data}\n")

    naive   = compute_naive(train, test)
    results = run_benchmark(train, test, args.models, args.seed)
    if not results:
        sys.exit("[ERREUR] Aucun modele n'a produit de resultat.")

    df_t1 = print_table1(results, naive, args.ticker, args.freq,
                         args.start, args.end, len(train), len(test))
    df_t2 = print_table2(results, naive, args.ticker, args.freq)
    df_t3 = print_table3(results, naive, args.ticker, args.freq)

    if args.save_csv:
        base, ext = os.path.splitext(args.save_csv)
        ext = ext or ".csv"
        for suf, df in [("_t1", df_t1), ("_t2", df_t2), ("_t3", df_t3)]:
            path_csv = f"{base}{suf}{ext}"
            df.to_csv(path_csv)
            print(f"[CSV] -> {path_csv}")

    if args.no_dashboard:
        return

    # ── Dashboard interactif ──────────────────────────────────────────────────
    print("\n[NEXT] Calcul des previsions J+1 ...")
    full_series = pd.concat([train, test])
    next_preds  = compute_next_steps(full_series, args.models, args.seed)

    print("\n[DASH] Generation des fenetres ...")

    fig1 = plot_timeseries_panel(
        train, test, results, naive,
        ticker=args.ticker, freq=args.freq,
        start=args.start, end=args.end,
        seed=args.seed, models=args.models,
        show_pi=not args.no_pi,
    )
    fig2 = plot_performance_panel(
        results, naive, next_preds, train,
        args.models, args.ticker,
    )

    # Sauvegarde PNG optionnelle
    if args.dashboard:
        base, ext = os.path.splitext(args.dashboard)
        ext = ext or ".png"
        p1 = f"{base}_ts{ext}"
        p2 = f"{base}_perf{ext}"
        fig1.savefig(p1, dpi=130, bbox_inches="tight", facecolor=fig1.get_facecolor())
        fig2.savefig(p2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
        print(f"[DASH] Serie temporelle -> {p1}")
        print(f"[DASH] Performance      -> {p2}")

    import matplotlib.pyplot as plt
    print("[DASH] Affichage des 2 fenetres (fermer pour quitter) ...")
    plt.show()


if __name__ == "__main__":
    main()
