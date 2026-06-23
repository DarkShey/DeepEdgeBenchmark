"""
DEITA Benchmark - Comparaison tete-a-tete des modeles
======================================================
Produit les 3 tableaux standards du template DEITA :

  Tableau 1 - Performances principales (RMSE + CRPS)
              Benchmark naif en premiere ligne de reference.
              Vert = meilleur de la colonne, [!] = pire que le naif.

  Tableau 2 - Test Diebold-Mariano par paire
              W = victoire (p<0.05), T = egalite, L = defaite.

  Tableau 3 - Intervalles de confiance (PI a 95%)
              Coverage (%), Width ($), Score de Winkler.

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
  --plot        Chemin du graphique comparatif (ex : bench.png)
  --save-csv    Chemin CSV des metriques (ex : results.csv)
  --save-data   Chemin CSV des prix bruts (ex : prices.csv)
  --no-pi       Masque les intervalles de confiance sur le graphique
  --no-color    Desactive les couleurs ANSI dans le terminal

Exemples
--------
  python benchmark.py
  python benchmark.py --models arima prophet --plot bench.png
  python benchmark.py --ticker BTC-USD --start 2019-01-01 --end 2024-12-31 --save-csv r.csv
  python benchmark.py --models arima sarima prophet
  python benchmark.py --start 2024-01-01 --end 2024-12-31
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
_GR = "\033[92m"   # vert  (meilleur)
_RE = "\033[91m"   # rouge (pire que naif)
_YL = "\033[93m"   # jaune (reference naif)
_BO = "\033[1m"    # gras
_RS = "\033[0m"    # reset

# ── Couleur conditionnelle + alignement ANSI ─────────────────────────────────

_USE_COLOR = True   # modifie par --no-color

import re as _re
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

def _c(text: str, code: str) -> str:
    """Applique une couleur ANSI si activee."""
    if _USE_COLOR:
        return f"{code}{text}{_RS}"
    return text

def _vlen(s: str) -> int:
    """Longueur visible (hors codes ANSI)."""
    return len(_ANSI_RE.sub("", s))

def _rpad(s: str, width: int) -> str:
    """Aligne a droite sur 'width' caracteres visibles."""
    return " " * max(0, width - _vlen(s)) + s

def _lpad(s: str, width: int) -> str:
    """Aligne a gauche sur 'width' caracteres visibles."""
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
    """
    Benchmark naif : y_hat_{t+1} = y_t (derniere valeur connue).
    Reference absolue : tout modele qui ne le bat pas ne justifie pas sa presence.
    Intervalle gaussien naif : pred +/- 1.96 * std(variations journalieres de train).
    """
    actual       = test.values.astype(float)
    naive_preds  = np.concatenate([[train.iloc[-1]], actual[:-1]])
    residuals    = actual - naive_preds

    mae    = float(np.mean(np.abs(residuals)))
    rmse   = float(np.sqrt(np.mean(residuals**2)))
    mape   = float(np.mean(np.abs(residuals / (actual + 1e-8))) * 100)
    smape  = float(np.mean(2*np.abs(residuals) / (np.abs(actual) + np.abs(naive_preds) + 1e-8)) * 100)
    d_acc  = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(naive_preds))) * 100)

    sigma  = float(np.std(np.diff(train.values)))
    lower  = naive_preds - 1.96 * sigma
    upper  = naive_preds + 1.96 * sigma

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

# ── Metriques supplementaires ─────────────────────────────────────────────────

def crps_gaussian_approx(predictions, lower, upper, actual, z: float = 1.96) -> float:
    """
    Approximation du CRPS via la formule gaussienne.
    sigma est derive du PI a 95% : sigma = (upper - lower) / (2 * 1.96).
    CRPS(N(mu,sigma), y) = sigma*(z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    """
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
    """
    Score de Winkler : penalise largeur + depassements hors intervalle.
    alpha = 1 - niveau de confiance (0.05 pour PI a 95%). Plus faible = meilleur.
    """
    y, lo, hi = map(np.asarray, (actual, lower, upper))
    width    = hi - lo
    pen_lo   = np.where(y < lo, (2/alpha) * (lo - y), 0.0)
    pen_hi   = np.where(y > hi, (2/alpha) * (y - hi), 0.0)
    return round(float(np.mean(width + pen_lo + pen_hi)), 2)


def diebold_mariano(errors_a, errors_b) -> tuple[float, float]:
    """
    Test Diebold-Mariano bilateral (perte = MSE, variance HAC Newey-West).
    Returns (DM_stat, p_value).
    d_t = MSE_A_t - MSE_B_t.
    DM > 0 => A perd plus => A est moins bon que B.
    p < 0.05 => difference statistiquement significative.
    """
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
                  models: list[str], seed: int) -> dict:
    """
    Lance chaque modele sur le meme split train/test.
    Enrichit chaque result avec CRPS (approximation gaussienne).
    """
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

        # Ajout du CRPS (approximation gaussienne si intervalles disponibles)
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

# ── Tableau 1 : Performances principales ─────────────────────────────────────
#  Lignes = modeles  |  Colonnes = metriques  (style PDF DEITA)

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
    """
    Tableau 1 — Performances principales.
    Lignes : modeles (Naif en premier). Colonnes : metriques.
    Vert [V] = meilleur de la colonne (hors naif).
    Rouge [!] = pire que le naif sur cette metrique.
    """
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

    col_m = 16   # largeur colonne modele
    col_v = 10   # largeur colonne metrique

    # Ligne d'en-tete des metriques
    hdr = _lpad("Modele", col_m)
    for m in metrics:
        hdr += _rpad(_METRIC_HEADERS[m], col_v)
    print(f"  {hdr}")
    sep = "  " + "-" * (col_m + col_v * len(metrics))
    print(sep)

    # Pre-calcule les meilleurs et pires (hors naif) par metrique
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
        """Formate une valeur avec couleur et marqueur."""
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

    # --- Naif (reference) ---
    _print_row("Naif (y_t)", naive, is_naive=True)
    print(sep)

    # --- Groupes de modeles ---
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

    # DataFrame pour export
    all_rows = {"Naif": {m: naive.get(m, "N/A") for m in metrics}}
    for k in all_keys:
        all_rows[MODEL_LABELS[k]] = {m: results[k].get(m, "N/A") for m in metrics}
    df = pd.DataFrame(all_rows).T
    df.index.name = "Modele"
    return df

# ── Tableau 2 : Diebold-Mariano par paire ─────────────────────────────────────

def print_table2(results: dict, naive: dict, ticker: str, freq: str) -> pd.DataFrame:
    """
    Tableau 2 - Test Diebold-Mariano bilateral.
    Cellule (A, B) : A vs B.
      W = victoire de A (p<0.05 et A plus precis)
      T = egalite statistique (p>=0.05)
      L = defaite de A (p<0.05 et B plus precis)
    """
    _, flabel, _ = FREQ_CONFIG[freq]
    W = 70
    print(f"\n{'='*W}")
    print(f"  {_c('TABLEAU 2 - Test Diebold-Mariano (victoires par paire)', _BO)}")
    print(f"  {ticker}  ({flabel})  |  Lecture : cellule (ligne A, col B) = A vs B")
    print(f"{'='*W}")

    # Inclure le naif
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

# ── Tableau 3 : Intervalles de confiance ──────────────────────────────────────

def print_table3(results: dict, naive: dict, ticker: str, freq: str,
                 alpha: float = 0.05) -> pd.DataFrame:
    """
    Tableau 3 - Evaluation des intervalles de confiance a (1-alpha)*100%.
    Metrics : Coverage (%), Width ($), Score de Winkler (plus faible = meilleur).
    Reference : intervalle gaussien naif.
    Regle d'acceptation : coverage dans [(1-alpha-0.05)*100, (1-alpha+0.05)*100].
    """
    _, flabel, _ = FREQ_CONFIG[freq]
    W        = 70
    cov_tgt  = (1 - alpha) * 100
    cov_lo   = cov_tgt - 5
    cov_hi   = cov_tgt + 5
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

    # Gaussien naif (reference)
    naif_metrics = _fmt_row(
        "Gaussien naif",
        actual, naive["lower"], naive["upper"],
        ref_winkler=None, is_naive=True,
    )
    naif_winkler = naif_metrics["Winkler"]
    rows["Gaussien naif"] = naif_metrics

    print(sep)

    # Chaque modele
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

# ── Graphique comparatif ──────────────────────────────────────────────────────

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

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    p = argparse.ArgumentParser(
        description="DEITA Benchmark - 3 tableaux standards (RMSE/CRPS, DM test, PI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ticker",     default="ETH-USD")
    p.add_argument("--start",      default="2020-01-01")
    p.add_argument("--end",        default="2024-12-31")
    p.add_argument("--freq",       default="J", choices=["J", "H", "S"])
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--models",     nargs="+", default=AVAILABLE_MODELS,
                   choices=AVAILABLE_MODELS, metavar="MODEL")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--plot",       metavar="PATH", default=None)
    p.add_argument("--save-csv",   metavar="PATH", default=None)
    p.add_argument("--save-data",  metavar="PATH", default=None)
    p.add_argument("--no-pi",      action="store_true")
    p.add_argument("--no-color",   action="store_true",
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
        p_t1 = f"{base}_t1{ext}"
        p_t2 = f"{base}_t2{ext}"
        p_t3 = f"{base}_t3{ext}"
        df_t1.to_csv(p_t1)
        df_t2.to_csv(p_t2)
        df_t3.to_csv(p_t3)
        print(f"[CSV] Tableau 1 -> {p_t1}")
        print(f"[CSV] Tableau 2 -> {p_t2}")
        print(f"[CSV] Tableau 3 -> {p_t3}")

    # ── Graphique ──
    if args.plot:
        save_comparison_plot(
            results, naive, train, test,
            args.ticker, args.freq, args.plot,
            show_pi=not args.no_pi,
        )


if __name__ == "__main__":
    main()
