"""
DEITA Benchmark 1.2 - Dashboard Visuel
=======================================
Extension de Benchmark 1.1 avec rendu graphique externe.

Nouveautes v1.2
---------------
  - Fenetre externe : le dashboard est sauvegarde en PNG puis ouvert dans le
    visionneur systeme (os.startfile sur Windows, open sur macOS, xdg-open Linux).
  - Panneau recapitulatif du dataset : ticker, dates, n_train/test, prix min/max/last,
    volatilite journaliere (1sigma et 2sigma en dollars).
  - Previsions J+1 (next-step) de chaque modele sous forme de barres OHLC horizontales :
      * Corps (body)        = intervalle a 1 sigma (68 %)
      * Moustaches (wicks)  = intervalle a 95 % (2 sigma)
      * Barre ensemble      = PI agrege : moy(preds) + moy(1sigma) + min/max(95%)
  - Toutes les sorties terminal de Benchmark 1.1 sont conservees.

Modeles disponibles
-------------------
  arima   - ARIMA(2,0,2)-GARCH(1,1) sur log-rendements, walk-forward
  sarima  - SARIMA(1,1,1)(1,0,1)[5] walk-forward (re-fit a chaque pas)
  prophet - Prophet (saisonnalites hebdo + annuelle), fit-once
  lstm    - LSTM(64) + Dense(1) sur fenetre glissante de 30 jours, walk-forward

Options
-------
  --ticker      Symbole yfinance (defaut : ETH-USD)
  --start       Date de debut YYYY-MM-DD (defaut : 2020-01-01)
  --end         Date de fin   YYYY-MM-DD (defaut : 2024-12-31)
  --freq        J=journalier (defaut)  H=horaire  S=hebdomadaire
  --test-ratio  Part du test set (defaut : 0.15)
  --models      Modeles a lancer, ex : --models arima prophet
  --seed        Graine aleatoire (defaut : 42)
  --dashboard   Chemin PNG du dashboard (defaut: deita_dashboard.png)
  --no-dashboard  Desactive la generation du dashboard
  --plot        Chemin du graphique comparatif legacy (ex : bench.png)
  --save-csv    Chemin CSV des metriques (ex : results.csv)
  --save-data   Chemin CSV des prix bruts (ex : prices.csv)
  --no-pi       Masque les intervalles de confiance sur le graphique
  --no-color    Desactive les couleurs ANSI dans le terminal

Exemples
--------
  python "Benchmark 1.2.py"
  python "Benchmark 1.2.py" --models arima prophet --dashboard dash.png
  python "Benchmark 1.2.py" --ticker BTC-USD --start 2019-01-01 --end 2024-12-31
  python "Benchmark 1.2.py" --no-dashboard --save-csv results.csv
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Fix SSL sur Windows (Python 3.14 + curl_cffi)
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

METRIC_KEYS_T1 = ["RMSE", "CRPS", "MAE", "MAPE (%)", "Dir. Acc (%)", "Train Time (s)"]

# ANSI couleurs terminal
_GR = "\033[92m"
_RE = "\033[91m"
_YL = "\033[93m"
_BO = "\033[1m"
_RS = "\033[0m"

# ── Couleur conditionnelle + alignement ANSI ─────────────────────────────────

_USE_COLOR = True

import re as _re
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

def _c(text: str, code: str) -> str:
    if _USE_COLOR:
        return f"{code}{text}{_RS}"
    return text

def _vlen(s: str) -> int:
    return len(_ANSI_RE.sub("", s))

def _rpad(s: str, width: int) -> str:
    return " " * max(0, width - _vlen(s)) + s

def _lpad(s: str, width: int) -> str:
    return s + " " * max(0, width - _vlen(s))

# ── Utilitaires de base ───────────────────────────────────────────────────────

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
        sys.exit(f"[ERREUR] Aucune donnee pour {ticker} entre {start} et {end}.")
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

    mae   = float(np.mean(np.abs(residuals)))
    rmse  = float(np.sqrt(np.mean(residuals**2)))
    mape  = float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100)
    d_acc = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(naive_preds))) * 100)

    sigma = float(np.std(np.diff(train.values)))
    lower = naive_preds - 1.96 * sigma
    upper = naive_preds + 1.96 * sigma

    return {
        "RMSE":           round(rmse,  4),
        "CRPS":           "--",
        "MAE":            round(mae,   4),
        "MAPE (%)":       round(mape,  2),
        "Dir. Acc (%)":   round(d_acc, 2),
        "Train Time (s)": 0.0,
        "predictions":    naive_preds,
        "lower":          lower,
        "upper":          upper,
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
    vals  = sigma * (zsc * (2*sp_norm.cdf(zsc) - 1) + 2*sp_norm.pdf(zsc) - 1/np.sqrt(np.pi))
    return round(float(np.mean(vals)), 4)


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
            from arima_model import run_arima_garch
            return run_arima_garch
        if model_key == "sarima":
            from sarima_model import run_sarima
            return run_sarima
        if model_key == "prophet":
            from prophet_model import run_prophet
            return run_prophet
        if model_key == "lstm":
            from lstm_model import run_lstm
            return run_lstm
    except ImportError as exc:
        print(f"[WARN] {MODEL_LABELS[model_key]} ignore - dependance manquante : {exc}")
        return None

# ── Execution du benchmark ────────────────────────────────────────────────────

def run_benchmark(train: pd.Series, test: pd.Series,
                  models: list, seed: int) -> dict:
    results = {}
    for key in models:
        label  = MODEL_LABELS[key]
        runner = load_runner(key)
        if runner is None:
            continue

        print(f"[RUN] {label} ...")
        set_seed(seed)
        try:
            result = runner(train, test)
        except Exception as exc:
            print(f"[ERREUR] {label} a echoue : {exc}")
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
    "RMSE":           "RMSE",
    "CRPS":           "CRPS",
    "MAE":            "MAE",
    "MAPE (%)":       "MAPE%",
    "Dir. Acc (%)":   "DirAcc%",
    "Train Time (s)": "Time(s)",
}

def print_table1(results: dict, naive: dict, ticker: str, freq: str,
                 start: str, end: str, n_train: int, n_test: int) -> pd.DataFrame:
    _, flabel, _ = FREQ_CONFIG[freq]
    metrics  = list(_METRIC_HEADERS.keys())
    all_keys = list(results.keys())
    W        = 80

    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 1 - Performances principales', _BO)}")
    print(f"  {ticker}  ({flabel}, {start} -> {end})")
    print(f"  Train : {n_train} pts  |  Test : {n_test} pts")
    print(f"  CRPS : approx. gaussienne via sigma = (PI_upper - PI_lower) / 3.92")
    print(f"{'='*W}")

    col_m = 16
    col_v = 10

    hdr = _lpad("Modele", col_m)
    for m in metrics:
        hdr += _rpad(_METRIC_HEADERS[m], col_v)
    print(f"  {hdr}")
    sep = "  " + "-" * (col_m + col_v * len(metrics))
    print(sep)

    best_key = {}
    for m in metrics:
        vals = {k: results[k].get(m) for k in all_keys
                if isinstance(results[k].get(m), (int, float))}
        if not vals:
            best_key[m] = None
        elif m == "Dir. Acc (%)":
            best_key[m] = max(vals, key=vals.get)
        else:
            best_key[m] = min(vals, key=vals.get)

    def _render_val(val, metric, model_key, is_naive=False):
        if is_naive:
            return _c(str(val), _YL)
        if not isinstance(val, (int, float)):
            return str(val)
        naive_val = naive.get(metric)
        is_best = (best_key.get(metric) == model_key)
        is_worse = False
        if isinstance(naive_val, (int, float)):
            if metric == "Dir. Acc (%)":
                is_worse = val < naive_val
            else:
                is_worse = val > naive_val
        if is_best:
            return _c(str(val), _GR)
        if is_worse:
            return _c(str(val) + "!", _RE)
        return str(val)

    def _print_row(label, data_dict, model_key=None, is_naive=False):
        lbl_str = _c(label, _YL) if is_naive else label
        row = f"  {_lpad(lbl_str, col_m)}"
        for m in metrics:
            val = data_dict.get(m, "N/A")
            cell = _render_val(val, m, model_key, is_naive)
            row += _rpad(cell, col_v)
        print(row)

    _print_row("Naif (y_t)", naive, is_naive=True)
    print(sep)

    GROUPS = [
        ("-- Statistique --", ["arima", "sarima", "prophet"]),
        ("-- RNN --",         ["lstm"]),
    ]
    printed = set()
    for grp_label, grp_keys in GROUPS:
        in_grp = [k for k in all_keys if k in grp_keys]
        if not in_grp:
            continue
        print(f"  {grp_label}")
        for k in in_grp:
            _print_row(MODEL_LABELS[k], results[k], model_key=k)
            printed.add(k)

    remaining = [k for k in all_keys if k not in printed]
    if remaining:
        print(f"  -- Autres --")
        for k in remaining:
            _print_row(MODEL_LABELS[k], results[k], model_key=k)

    print(sep)
    print(f"  {_c('[V]', _GR)} meilleur (vert)  |  {_c('[!]', _RE)} pire que le naif (rouge)  "
          f"|  {_c('italique', _YL)} naif (reference)")
    print(f"{'='*W}\n")

    all_rows = {"Naif": {m: naive.get(m, "N/A") for m in metrics}}
    for k in all_keys:
        all_rows[MODEL_LABELS[k]] = {m: results[k].get(m, "N/A") for m in metrics}
    df = pd.DataFrame(all_rows).T
    df.index.name = "Modele"
    return df

# ── Tableau 2 ─────────────────────────────────────────────────────────────────

def print_table2(results: dict, naive: dict, ticker: str, freq: str) -> pd.DataFrame:
    _, flabel, _ = FREQ_CONFIG[freq]
    W = 70
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 2 - Test Diebold-Mariano (victoires par paire)', _BO)}")
    print(f"  {ticker}  ({flabel})  |  Lecture : cellule (ligne A, col B) = A vs B")
    print(f"{'='*W}")

    all_items = [("naif", "Naif", naive)] + [
        (k, MODEL_LABELS[k], results[k]) for k in results
    ]

    errors = {}
    for key, _, res in all_items:
        actual = np.asarray(res["actual"], float)
        preds  = np.asarray(res["predictions"], float)
        errors[key] = actual - preds

    names = [lbl for _, lbl, _ in all_items]
    keys  = [k   for k, _, _ in all_items]

    col_w = max(14, max(len(n) for n in names) + 2)
    hdr   = f"  {_lpad('', col_w)}" + "".join(_rpad(n, col_w) for n in names)
    sep   = "  " + "-" * (col_w * (len(names) + 1))
    print(hdr)
    print(sep)

    dm_rows = {}
    for i, (key_a, lbl_a, _) in enumerate(all_items):
        row  = {}
        line = f"  {_rpad(lbl_a, col_w)}"
        for j, (key_b, lbl_b, _) in enumerate(all_items):
            if i == j:
                cell_str = "--"
                cell_col = "--"
            else:
                dm_stat, p_val = diebold_mariano(errors[key_a], errors[key_b])
                if p_val < 0.05:
                    if dm_stat > 0:
                        cell_str = f"L ({p_val:.2f})"
                        cell_col = _c(cell_str, _RE)
                    else:
                        cell_str = f"W ({p_val:.2f})"
                        cell_col = _c(cell_str, _GR)
                else:
                    cell_str = f"T ({p_val:.2f})"
                    cell_col = cell_str
            row[lbl_b] = cell_str
            line += _rpad(cell_col, col_w)
        print(line)
        dm_rows[lbl_a] = row

    print(sep)
    print(f"  {_c('W', _GR)} = victoire (p<0.05)  |  T = egalite (p>=0.05)  |  "
          f"{_c('L', _RE)} = defaite (p<0.05)")
    print(f"  Variance HAC Newey-West, truncation = floor(T^(1/3))")
    print(f"{'='*W}\n")

    df = pd.DataFrame(dm_rows).T
    df.index.name = "vs"
    return df

# ── Tableau 3 ─────────────────────────────────────────────────────────────────

def print_table3(results: dict, naive: dict, ticker: str, freq: str,
                 alpha: float = 0.05) -> pd.DataFrame:
    _, flabel, _ = FREQ_CONFIG[freq]
    W       = 70
    cov_tgt = (1 - alpha) * 100
    cov_lo  = cov_tgt - 5
    cov_hi  = cov_tgt + 5
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 3 - Intervalles de confiance (PI a {:.0f}%)'.format(cov_tgt), _BO)}")
    print(f"  {ticker}  ({flabel})")
    print(f"  Cible coverage : {cov_tgt:.0f}%  "
          f"|  Zone acceptable : [{cov_lo:.0f}%, {cov_hi:.0f}%]")
    print(f"{'='*W}")

    col_m = 20
    col_v = 14
    hdr   = (f"  {_lpad('Methode', col_m)}"
             f"{_rpad('Coverage (%)', col_v)}"
             f"{_rpad('Width ($)', col_v)}"
             f"{_rpad('Winkler', col_v)}")
    sep   = "  " + "-" * (col_m + col_v * 3)
    print(hdr)
    print(sep)

    rows = {}

    def _fmt_row(label, actual, lower, upper, ref_winkler=None, is_naive=False):
        cov_v = coverage(actual, lower, upper)
        wid_v = avg_width(lower, upper)
        wkl_v = winkler_score(actual, lower, upper, alpha)

        cov_str = str(cov_v)
        if cov_v < cov_lo or cov_v > cov_hi:
            cov_str = _c(str(cov_v) + "!", _RE)

        wkl_str = str(wkl_v)
        if ref_winkler is not None and wkl_v > ref_winkler and not is_naive:
            wkl_str = _c(str(wkl_v) + "!", _RE)

        lbl_str = _c(label, _YL) if is_naive else label
        print(f"  {_lpad(lbl_str, col_m)}"
              f"{_rpad(cov_str, col_v)}"
              f"{_rpad(str(wid_v), col_v)}"
              f"{_rpad(wkl_str, col_v)}")
        return {"Coverage (%)": cov_v, "Width ($)": wid_v, "Winkler": wkl_v}

    actual = np.asarray(naive["actual"], float)

    naif_metrics = _fmt_row(
        "Gaussien naif",
        actual, naive["lower"], naive["upper"],
        ref_winkler=None, is_naive=True,
    )
    naif_winkler = naif_metrics["Winkler"]
    rows["Gaussien naif"] = naif_metrics

    print(sep)

    for k, res in results.items():
        if res.get("lower") is None or res.get("upper") is None:
            continue
        m = _fmt_row(
            MODEL_LABELS[k],
            actual, res["lower"], res["upper"],
            ref_winkler=naif_winkler,
        )
        rows[MODEL_LABELS[k]] = m

    print(sep)
    print(f"  {_c('[!] Coverage', _RE)} hors [{cov_lo:.0f}%, {cov_hi:.0f}%] -> recalibrer")
    print(f"  {_c('[!] Winkler', _RE)}   > Gaussien naif -> intervalle moins informatif")
    print(f"  Winkler = Width + (2/alpha)*max(0, lb-y) + (2/alpha)*max(0, y-ub)")
    print(f"{'='*W}\n")

    df = pd.DataFrame(rows).T
    df.index.name = "Methode"
    return df

# ── Graphique comparatif legacy ───────────────────────────────────────────────

def save_comparison_plot(results: dict, naive: dict,
                         train: pd.Series, test: pd.Series,
                         ticker: str, freq: str, path: str,
                         show_pi: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, label, _ = FREQ_CONFIG[freq]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"DEITA Benchmark | {ticker} ({label})", fontsize=13, fontweight="bold")

    ax1.plot(train.index, train.values, color="lightgray", lw=1, label="Train")
    ax1.plot(test.index,  test.values,  color="black",     lw=1.2, label="Test (reel)")
    ax1.plot(naive["index"], naive["predictions"],
             color="gray", lw=1, ls="--", alpha=0.7, label="Naif")
    for k, res in results.items():
        ax1.plot(res["index"], res["predictions"],
                 color=MODEL_COLORS[k], lw=1, alpha=0.8, label=MODEL_LABELS[k])
    ax1.set_ylabel("Prix ($)")
    ax1.legend(fontsize=8, ncol=4, loc="upper left")
    ax1.set_title("Vue globale", fontsize=10)

    ax2.plot(test.index, test.values, color="black", lw=1.5, label="Reel", zorder=5)
    ax2.plot(naive["index"], naive["predictions"],
             color="gray", lw=1, ls="--", alpha=0.6, label="Naif")
    if show_pi:
        ax2.fill_between(naive["index"], naive["lower"], naive["upper"],
                         color="gray", alpha=0.10, label="PI naif 95%")
    for k, res in results.items():
        ax2.plot(res["index"], res["predictions"],
                 color=MODEL_COLORS[k], lw=1.3, label=MODEL_LABELS[k])
        if show_pi and res.get("lower") is not None:
            ax2.fill_between(res["index"], res["lower"], res["upper"],
                             color=MODEL_COLORS[k], alpha=0.12)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Prix ($)")
    ax2.legend(fontsize=9, ncol=3)
    ax2.set_title("Zoom test - predictions + intervalles 95%", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[PLOT] Graphique sauvegarde -> {path}")

# ── Previsions J+1 (next-step) ────────────────────────────────────────────────

def compute_next_steps(full_series: pd.Series, models: list, seed: int) -> dict:
    """
    Appelle next_step_X() de chaque modele sur la serie complete (train + test).
    Retourne un dict {key: (pred, lo_95, hi_95)}.
    """
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
            print(f"[NEXT] {label}: pred={pred:.2f}$  "
                  f"95%PI=[{lo:.2f}$, {hi:.2f}$]")
        except Exception as exc:
            print(f"[NEXT] {label}: ERREUR - {exc}")

    return next_preds

# ── Ouverture automatique du fichier ─────────────────────────────────────────

def _open_file(path: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
        print(f"[DASH] Dashboard ouvert dans le visionneur systeme.")
    except Exception as exc:
        print(f"[WARN] Impossible d'ouvrir {path} automatiquement : {exc}")

# ── Dashboard visuel ──────────────────────────────────────────────────────────

def plot_dashboard(
    train: pd.Series,
    test: pd.Series,
    results: dict,
    naive: dict,
    next_preds: dict,
    ticker: str,
    freq: str,
    start: str,
    end: str,
    seed: int,
    models: list,
    show_pi: bool = True,
):
    """
    Cree un dashboard matplotlib 3 panneaux :
      1. Recapitulatif du dataset (texte)
      2. Series temporelle : train + test + predictions + PI
      3. Barres OHLC horizontales pour les previsions J+1
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    _, flabel, _ = FREQ_CONFIG[freq]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#f7f8fa")

    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[1.2, 2.8, 2.2],
        hspace=0.45,
        figure=fig,
        left=0.08, right=0.97, top=0.95, bottom=0.06,
    )
    ax_info = fig.add_subplot(gs[0])
    ax_ts   = fig.add_subplot(gs[1])
    ax_ohlc = fig.add_subplot(gs[2])

    fig.suptitle(
        f"DEITA Benchmark 1.2  |  {ticker}  ({flabel})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panneau 1 : Recapitulatif dataset ────────────────────────────────────
    ax_info.axis("off")

    all_prices  = pd.concat([train, test])
    vol_daily   = float(np.std(np.diff(train.values)))
    last_actual = float(test.iloc[-1])
    next_date_str = ""
    try:
        idx = pd.DatetimeIndex(test.index)
        freq_offset = pd.tseries.frequencies.to_offset(pd.infer_freq(idx))
        if freq_offset is not None:
            next_date_str = str((idx[-1] + freq_offset).date())
    except Exception:
        pass

    info_lines = [
        f"Ticker : {ticker:<12} Frequence : {flabel:<14} Seed : {seed}",
        f"Periode : {start}  ->  {end}     N total : {len(all_prices)} obs",
        f"",
        f"Train  : {len(train):>4} pts   {str(train.index[0].date())} -> {str(train.index[-1].date())}",
        f"Test   : {len(test):>4} pts   {str(test.index[0].date())}  -> {str(test.index[-1].date())}",
        f"",
        f"Prix   : min={all_prices.min():.2f}$   max={all_prices.max():.2f}$"
        f"   last={last_actual:.2f}$ ({str(test.index[-1].date())})",
        f"Vol.   : 1sigma={vol_daily:.2f}$  "
        f"[{last_actual - vol_daily:.2f}$, {last_actual + vol_daily:.2f}$]"
        f"   2sigma={2*vol_daily:.2f}$",
    ]
    if next_date_str:
        info_lines.append(f"J+1    : {next_date_str}")

    ax_info.text(
        0.01, 0.98,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#b0b8c8",
            alpha=0.95,
        ),
    )
    ax_info.set_title(
        "Resume du dataset",
        fontsize=10, pad=3, loc="left", color="#444444",
    )

    # ── Panneau 2 : Serie temporelle ─────────────────────────────────────────
    ax_ts.plot(train.index, train.values,
               color="#cccccc", lw=0.9, label="Train", zorder=1)
    ax_ts.plot(test.index, test.values,
               color="black", lw=1.5, label="Reel (test)", zorder=5)
    ax_ts.plot(naive["index"], naive["predictions"],
               color="#888888", lw=0.9, ls="--", alpha=0.75, label="Naif")

    for k, res in results.items():
        ax_ts.plot(
            res["index"], res["predictions"],
            color=MODEL_COLORS[k], lw=1.3, alpha=0.85, label=MODEL_LABELS[k],
        )
        if show_pi and res.get("lower") is not None:
            ax_ts.fill_between(
                res["index"], res["lower"], res["upper"],
                color=MODEL_COLORS[k], alpha=0.09,
            )

    if show_pi:
        ax_ts.fill_between(
            naive["index"], naive["lower"], naive["upper"],
            color="#888888", alpha=0.07, label="PI naif 95%",
        )

    ax_ts.axvline(test.index[-1], color="#cccccc", lw=1, ls=":", zorder=0)
    ax_ts.set_ylabel("Prix ($)", fontsize=10)
    ax_ts.set_title(
        "Predictions walk-forward sur le test set"
        + (" + PI 95%" if show_pi else ""),
        fontsize=10, pad=4,
    )
    ax_ts.legend(fontsize=8, ncol=4, loc="upper left",
                 framealpha=0.92, edgecolor="#cccccc")
    ax_ts.grid(True, alpha=0.25, lw=0.5)
    ax_ts.set_facecolor("#fafbfc")

    # ── Panneau 3 : Barres OHLC J+1 ──────────────────────────────────────────
    last_known = float(test.iloc[-1])
    sigma_naive = float(np.std(np.diff(train.values)))

    # Construire la liste des barres (naif + modeles + ensemble)
    bar_items = []

    # Naif
    bar_items.append({
        "key":    "naif",
        "label":  "Naif (y_t)",
        "color":  "#888888",
        "pred":   last_known,
        "lo95":   last_known - 1.96 * sigma_naive,
        "hi95":   last_known + 1.96 * sigma_naive,
        "sigma":  sigma_naive,
    })

    # Modeles
    for k in models:
        if k not in next_preds:
            continue
        pred, lo95, hi95 = next_preds[k]
        sigma = (hi95 - lo95) / (2 * 1.96) if (hi95 - lo95) > 0 else 0.0
        bar_items.append({
            "key":   k,
            "label": MODEL_LABELS[k],
            "color": MODEL_COLORS[k],
            "pred":  pred,
            "lo95":  lo95,
            "hi95":  hi95,
            "sigma": sigma,
        })

    # Ensemble
    if len(next_preds) >= 2:
        ep     = [next_preds[k][0] for k in next_preds]
        el     = [next_preds[k][1] for k in next_preds]
        eh     = [next_preds[k][2] for k in next_preds]
        e_pred = float(np.mean(ep))
        e_lo95 = float(np.min(el))
        e_hi95 = float(np.max(eh))
        e_sig  = float(np.mean([(h - l) / (2 * 1.96) for l, h in zip(el, eh)]))
        bar_items.append({
            "key":   "ensemble",
            "label": "Ensemble",
            "color": "darkorange",
            "pred":  e_pred,
            "lo95":  e_lo95,
            "hi95":  e_hi95,
            "sigma": e_sig,
        })

    # Calculer la plage x avant de tracer
    all_lo = [b["lo95"] for b in bar_items]
    all_hi = [b["hi95"] for b in bar_items]
    x_min  = min(all_lo)
    x_max  = max(all_hi)
    x_rng  = max(x_max - x_min, 1.0)
    text_offset = x_rng * 0.012

    n  = len(bar_items)
    ys = list(range(n - 1, -1, -1))  # naif en haut, ensemble en bas

    # Reference last_known
    ax_ohlc.axvline(
        last_known, color="black", lw=1.5, ls="--", zorder=10,
        label=f"Last close: {last_known:.2f}$",
    )

    yticks, ylabels = [], []
    for i, item in enumerate(bar_items):
        y      = ys[i]
        pred   = item["pred"]
        lo95   = item["lo95"]
        hi95   = item["hi95"]
        sigma  = item["sigma"]
        color  = item["color"]
        label  = item["label"]
        blo    = pred - sigma   # corps 1sigma
        bhi    = pred + sigma

        # Moustaches (wicks) = PI 95%
        ax_ohlc.plot([lo95, hi95], [y, y],
                     color=color, lw=1.8, alpha=0.55, zorder=2, solid_capstyle="butt")
        # Extremites des moustaches
        cap_h = 0.18
        ax_ohlc.plot([lo95, lo95], [y - cap_h, y + cap_h],
                     color=color, lw=1.8, alpha=0.55, zorder=2)
        ax_ohlc.plot([hi95, hi95], [y - cap_h, y + cap_h],
                     color=color, lw=1.8, alpha=0.55, zorder=2)

        # Corps = 1sigma (barre pleine)
        ax_ohlc.barh(y, bhi - blo, left=blo, height=0.44,
                     color=color, alpha=0.80, zorder=3,
                     linewidth=0.5, edgecolor="white")

        # Trait blanc vertical au centre (prix predit)
        ax_ohlc.plot([pred, pred], [y - 0.22, y + 0.22],
                     color="white", lw=2.0, zorder=4)

        # Valeur predite a droite de la barre
        ax_ohlc.text(
            hi95 + text_offset, y,
            f"{pred:.0f}$",
            va="center", ha="left",
            fontsize=8.5, color=color, fontweight="bold",
        )

        yticks.append(y)
        ylabels.append(label)

    # Axe Y
    ax_ohlc.set_yticks(yticks)
    ax_ohlc.set_yticklabels(ylabels, fontsize=9.5)

    # Separateur visuel entre ensemble et modeles individuels
    if len(bar_items) >= 2:
        sep_y = ys[-1] + 0.5  # entre ensemble (dernier) et modele precedent
        ax_ohlc.axhline(sep_y, color="#cccccc", lw=0.8, ls="--", zorder=1)

    # Limites x avec marges
    margin = x_rng * 0.08
    ax_ohlc.set_xlim(x_min - margin, x_max + x_rng * 0.15)
    ax_ohlc.set_ylim(-0.7, n - 0.3)

    ax_ohlc.set_xlabel("Prix ($)", fontsize=10)
    ax_ohlc.set_title(
        "Previsions J+1 (next-step)  |  Corps : 1sigma (68%)  |  Moustaches : 95% PI",
        fontsize=10, pad=4,
    )

    legend_elems = [
        Patch(facecolor="#888888", alpha=0.8, label="Corps = prediction +/- 1 sigma (68%)"),
        Line2D([0], [0], color="#888888", lw=1.8, alpha=0.55,
               label="Moustaches = PI 95% du modele"),
        Line2D([0], [0], color="black", lw=1.5, ls="--",
               label=f"Last close: {last_known:.2f}$"),
        Patch(facecolor="darkorange", alpha=0.8, label="Ensemble (agregation des modeles)"),
    ]
    ax_ohlc.legend(
        handles=legend_elems, fontsize=8, loc="lower right",
        framealpha=0.92, edgecolor="#cccccc",
    )
    ax_ohlc.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax_ohlc.set_facecolor("#fafbfc")

    return fig

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    p = argparse.ArgumentParser(
        description="DEITA Benchmark 1.2 - Dashboard visuel (fenetre externe + OHLC J+1)",
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
                   help="Chemin PNG du dashboard (defaut : deita_dashboard.png)")
    p.add_argument("--no-dashboard", action="store_true",
                   help="Desactive la generation et l'ouverture du dashboard")
    p.add_argument("--plot",         metavar="PATH", default=None,
                   help="Graphique comparatif legacy (optionnel)")
    p.add_argument("--save-csv",     metavar="PATH", default=None)
    p.add_argument("--save-data",    metavar="PATH", default=None)
    p.add_argument("--no-pi",        action="store_true")
    p.add_argument("--no-color",     action="store_true",
                   help="Desactive les couleurs ANSI dans le terminal")
    args = p.parse_args()

    if args.no_color:
        _USE_COLOR = False

    _, _, seasonal_warn = FREQ_CONFIG[args.freq]
    if seasonal_warn and "sarima" in args.models:
        print("[WARN] Frequence hebdomadaire : periode saisonniere SARIMA (s=5j) inadaptee.\n")
    if args.freq == "H":
        print("[INFO] Frequence horaire : lookback yfinance limite a ~60 jours.\n")

    # ── Donnees ──
    prices = fetch_prices(args.ticker, args.start, args.end, args.freq)
    train, test = split_series(prices, args.test_ratio)

    print(f"[SPLIT] Train : {len(train)} pts "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test  : {len(test)}  pts "
          f"({test.index[0].date()}  -> {test.index[-1].date()})\n")

    if len(train) < 60:
        print("[WARN] Moins de 60 points d'entrainement : LSTM risque d'etre instable.\n")

    if args.save_data:
        prices.to_csv(args.save_data, header=["Close"])
        print(f"[DATA] Prix sauvegardes -> {args.save_data}\n")

    # ── Benchmark naif ──
    naive = compute_naive(train, test)

    # ── Modeles ──
    results = run_benchmark(train, test, args.models, args.seed)
    if not results:
        sys.exit("[ERREUR] Aucun modele n'a produit de resultat.")

    # ── Tableau 1 ──
    df_t1 = print_table1(
        results, naive,
        args.ticker, args.freq, args.start, args.end,
        len(train), len(test),
    )

    # ── Tableau 2 ──
    df_t2 = print_table2(results, naive, args.ticker, args.freq)

    # ── Tableau 3 ──
    df_t3 = print_table3(results, naive, args.ticker, args.freq)

    # ── Export CSV ──
    if args.save_csv:
        base, ext = os.path.splitext(args.save_csv)
        ext = ext or ".csv"
        for suffix, df in [("_t1", df_t1), ("_t2", df_t2), ("_t3", df_t3)]:
            path_csv = f"{base}{suffix}{ext}"
            df.to_csv(path_csv)
            print(f"[CSV] -> {path_csv}")

    # ── Graphique legacy ──
    if args.plot:
        save_comparison_plot(
            results, naive, train, test,
            args.ticker, args.freq, args.plot,
            show_pi=not args.no_pi,
        )

    # ── Dashboard 1.2 ──
    if not args.no_dashboard:
        print("\n[NEXT] Calcul des previsions J+1 (next-step) ...")
        full_series = pd.concat([train, test])
        next_preds  = compute_next_steps(full_series, args.models, args.seed)

        if not next_preds:
            print("[WARN] Aucune prevision J+1 disponible - dashboard partiel.")

        print("\n[DASH] Generation du dashboard ...")
        import matplotlib.pyplot as plt

        fig = plot_dashboard(
            train, test, results, naive, next_preds,
            args.ticker, args.freq, args.start, args.end,
            args.seed, args.models,
            show_pi=not args.no_pi,
        )

        dash_path = args.dashboard
        if dash_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dash_path  = os.path.join(script_dir, "deita_dashboard.png")

        fig.savefig(
            dash_path, dpi=130,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"[DASH] Sauvegarde -> {dash_path}")

        _open_file(dash_path)


if __name__ == "__main__":
    main()
