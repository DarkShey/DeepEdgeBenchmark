"""
DEITA Benchmark — Comparaison tete-a-tete des modeles
======================================================
Point d'entree unique pour entrainer et comparer plusieurs modeles de prevision
sur exactement le meme jeu de donnees (meme split train/test, meme actif).

Modeles disponibles
-------------------
  arima   – ARIMA(2,0,2)-GARCH(1,1) sur log-rendements, walk-forward
  sarima  – SARIMA(1,1,1)(1,0,1)[5] walk-forward (re-fit a chaque pas)
  prophet – Prophet (saisonnalites hebdo + annuelle), fit-once
  lstm    – LSTM(64) + Dense(1) sur fenetre glissante de 30 jours, walk-forward

Best practices du dataset d'entrainement
-----------------------------------------
  1. UNE SEULE requete de donnees partagee par tous les modeles -> pas de drift.
  2. Split train/test FIXE (test_ratio) -> meme periode de test pour tous.
  3. Seed aleatoire fixe -> LSTM reproductible d'une execution a l'autre.
  4. Sauvegarde optionnelle du dataset brut (--save-data) pour audit.
  5. Les modeles reçoivent les prix bruts : chaque modele gere son propre
     pretraitement (log-rendements pour ARIMA, MinMax pour LSTM, etc.)

Options disponibles
--------------------
  --ticker      Symbole yfinance (defaut : ETH-USD)
                Exemples : BTC-USD  SPY  GC=F  ETH-USD
  --start       Date de debut YYYY-MM-DD (defaut : 2020-01-01)
  --end         Date de fin   YYYY-MM-DD (defaut : 2024-12-31)
  --freq        Frequence des donnees :
                  J = journaliere (defaut) -> interval 1d, recommande
                  H = horaire              -> interval 1h, max ~60 jours lookback
                  S = hebdomadaire         -> interval 1wk
  --test-ratio  Part de la serie reservee au test (defaut : 0.15 = 15 %)
  --models      Modeles a executer, separes par des espaces (defaut : tous)
                Ex : --models arima prophet
  --seed        Graine aleatoire pour reproductibilite (defaut : 42)
  --plot        Chemin de sauvegarde du graphique comparatif (ex : bench.png)
  --save-csv    Chemin de sauvegarde du tableau de metriques   (ex : results.csv)
  --save-data   Chemin de sauvegarde des prix telecharges      (ex : prices.csv)
  --no-pi       Masquer les intervalles de confiance sur le graphique

Exemples d'utilisation
-----------------------
  # Comparaison complete sur ETH-USD (defaut)
  python benchmark.py

  # Seulement ARIMA et Prophet, graphique sauvegarde
  python benchmark.py --models arima prophet --plot bench.png

  # BTC-USD sur 5 ans, tous les modeles, export CSV
  python benchmark.py --ticker BTC-USD --start 2019-01-01 --end 2024-12-31 --save-csv results.csv

  # LSTM desactive (lent), resultats exportes
  python benchmark.py --models arima sarima prophet --save-csv fast_bench.csv

  # Test rapide sur 1 an (train 85 % ≈ 10 mois, test ≈ 45 jours)
  python benchmark.py --start 2024-01-01 --end 2024-12-31

Notes
-----
  - SARIMA re-fit le modele a CHAQUE pas de test -> peut prendre plusieurs minutes.
  - LSTM necessite TensorFlow. Sans GPU, ~2-5 min sur 1 500 pts de train.
  - Pour H (horaire), yfinance limite le lookback a ~60 jours calendaires.
  - Pour S (hebdomadaire) avec SARIMA, la periode saisonniere (s=5 jours) n'est
    plus pertinente. Un avertissement est affiche dans ce cas.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Fix SSL on Windows (Python 3.14 + curl_cffi) : point curl at the certifi bundle
try:
    import certifi as _certifi
    _ca = _certifi.where()
    os.environ.setdefault("CURL_CA_BUNDLE",    _ca)
    os.environ.setdefault("SSL_CERT_FILE",     _ca)
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
    # freq_code: (yfinance_interval, label_fr, seasonal_warning)
    "J": ("1d",  "journaliere",   False),
    "H": ("1h",  "horaire",       False),
    "S": ("1wk", "hebdomadaire",  True),
}

METRIC_KEYS = [
    "RMSE", "MAE", "MAPE (%)", "SMAPE (%)",
    "Dir. Acc (%)", "PI Cov 95% (%)", "Train Time (s)",
]

# ── Utilitaires ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fixe les graines aleatoires pour numpy et TensorFlow (si disponible)."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _make_session():
    """Cree une session curl_cffi avec SSL desactive (contournement Windows)."""
    try:
        from curl_cffi.requests import Session as CurlSession
        return CurlSession(impersonate="chrome", verify=False)
    except Exception:
        return None


def fetch_prices(ticker: str, start: str, end: str, freq: str) -> pd.Series:
    """
    Telecharge les prix de cloture une seule fois pour tous les modeles.

    Retourne une pd.Series tz-naive avec les prix de cloture journaliers.
    """
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
    close = close.astype(float)

    print(f"[DATA] {len(close)} observations chargees.\n")
    return close


def split_series(prices: pd.Series, test_ratio: float):
    """Split train/test sur la meme base pour tous les modeles."""
    split = int(len(prices) * (1 - test_ratio))
    return prices.iloc[:split], prices.iloc[split:]


# ── Chargement conditionnel des modeles ──────────────────────────────────────

def load_runner(model_key: str):
    """
    Importe dynamiquement la fonction run_X() du fichier modele correspondant.
    Retourne None si les dependances sont manquantes.
    """
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


# ── Execution du benchmark ───────────────────────────────────────────────────

def run_benchmark(train: pd.Series, test: pd.Series,
                  models: list[str], seed: int) -> dict:
    """
    Lance chaque modele sur le meme split train/test.
    Retourne un dict  {model_key: result_dict}  où result_dict contient
    metriques + tableaux predictions/lower/upper/actual/index.
    """
    results = {}

    for key in models:
        label = MODEL_LABELS[key]
        runner = load_runner(key)
        if runner is None:
            continue

        print(f"[RUN] {label} ...")
        set_seed(seed)    # reset avant chaque modele stochastique (LSTM)
        t_start = time.time()

        try:
            result = runner(train, test)
        except Exception as exc:
            print(f"[ERREUR] {label} a echoue : {exc}")
            continue

        elapsed = time.time() - t_start
        results[key] = result
        print(f"[RUN] {label} termine en {elapsed:.1f}s  |  "
              f"RMSE={result.get('RMSE','?')}  "
              f"Dir.Acc={result.get('Dir. Acc (%)','?')}%\n")

    return results


# ── Affichage de la table de comparaison ─────────────────────────────────────

def print_comparison_table(results: dict, ticker: str, freq: str,
                           start: str, end: str,
                           n_train: int, n_test: int) -> pd.DataFrame:
    """Affiche la table de metriques alignee dans le terminal."""
    _, label, _ = FREQ_CONFIG[freq]

    header = (
        f"\n{'='*70}\n"
        f"  DEITA Benchmark | {ticker}  ({label}, {start} -> {end})\n"
        f"  Train : {n_train} pts  |  Test : {n_test} pts\n"
        f"{'='*70}"
    )
    print(header)

    if not results:
        print("  Aucun resultat disponible.")
        return pd.DataFrame()

    model_keys = list(results.keys())
    col_w = max(14, max(len(MODEL_LABELS[k]) for k in model_keys) + 2)

    header_row = f"  {'Metrique':<22}" + "".join(
        f"{MODEL_LABELS[k]:>{col_w}}" for k in model_keys
    )
    sep = "  " + "-" * (22 + col_w * len(model_keys))
    print(header_row)
    print(sep)

    rows = {}
    for metric in METRIC_KEYS:
        row_vals = {}
        line = f"  {metric:<22}"
        for k in model_keys:
            val = results[k].get(metric, "N/A")
            row_vals[k] = val
            line += f"{str(val):>{col_w}}"
        rows[metric] = row_vals
        print(line)

    print(sep)

    # Meilleur modele par RMSE
    rmse_vals = {k: results[k].get("RMSE", np.inf) for k in model_keys
                 if isinstance(results[k].get("RMSE"), (int, float))}
    if rmse_vals:
        best = min(rmse_vals, key=rmse_vals.get)
        print(f"\n  Meilleur RMSE : {MODEL_LABELS[best]} ({rmse_vals[best]})")

    # Meilleure precision directionnelle
    dir_vals = {k: results[k].get("Dir. Acc (%)", -np.inf) for k in model_keys
                if isinstance(results[k].get("Dir. Acc (%)"), (int, float))}
    if dir_vals:
        best_dir = max(dir_vals, key=dir_vals.get)
        print(f"  Meilleure Dir. Acc : {MODEL_LABELS[best_dir]} ({dir_vals[best_dir]}%)")

    print(f"{'='*70}\n")

    # Construction du DataFrame de sortie
    df_rows = {}
    for metric in METRIC_KEYS:
        df_rows[metric] = {MODEL_LABELS[k]: results[k].get(metric, "N/A")
                           for k in model_keys}
    df = pd.DataFrame(df_rows).T
    df.index.name = "Metrique"
    return df


# ── Graphique comparatif ──────────────────────────────────────────────────────

def save_comparison_plot(results: dict, train: pd.Series, test: pd.Series,
                         ticker: str, freq: str, path: str,
                         show_pi: bool = True) -> None:
    """
    Sauvegarde un graphique a deux panneaux :
      - Panneau haut  : historique complet (train gris + test noir) + toutes predictions
      - Panneau bas   : zoom sur la periode de test uniquement
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, label, _ = FREQ_CONFIG[freq]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"DEITA Benchmark | {ticker} ({label})", fontsize=13, fontweight="bold")

    # ── Panneau haut : vue globale ──
    ax1.plot(train.index, train.values, color="lightgray", lw=1, label="Train")
    ax1.plot(test.index,  test.values,  color="black",     lw=1.2, label="Test (reel)")
    for k, res in results.items():
        ax1.plot(res["index"], res["predictions"],
                 color=MODEL_COLORS[k], lw=1, alpha=0.8, label=MODEL_LABELS[k])
    ax1.set_ylabel("Prix ($)")
    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    ax1.set_title("Vue globale", fontsize=10)

    # ── Panneau bas : zoom test ──
    ax2.plot(test.index, test.values, color="black", lw=1.5, label="Reel", zorder=5)
    for k, res in results.items():
        ax2.plot(res["index"], res["predictions"],
                 color=MODEL_COLORS[k], lw=1.3, label=MODEL_LABELS[k])
        if show_pi and "lower" in res and "upper" in res:
            ax2.fill_between(res["index"], res["lower"], res["upper"],
                             color=MODEL_COLORS[k], alpha=0.12)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Prix ($)")
    ax2.legend(fontsize=9, ncol=2)
    ax2.set_title("Zoom periode de test — predictions + intervalles 95%", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[PLOT] Graphique sauvegarde -> {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="DEITA Benchmark - comparaison tete-a-tete des modeles de prevision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Donnees
    p.add_argument("--ticker", default="ETH-USD",
                   help="Symbole yfinance (defaut : ETH-USD)")
    p.add_argument("--start",  default="2020-01-01",
                   help="Date de debut YYYY-MM-DD (defaut : 2020-01-01)")
    p.add_argument("--end",    default="2024-12-31",
                   help="Date de fin   YYYY-MM-DD (defaut : 2024-12-31)")
    p.add_argument("--freq",   default="J", choices=["J", "H", "S"],
                   help="Frequence : J=journaliere (defaut)  H=horaire  S=hebdomadaire")
    p.add_argument("--test-ratio", type=float, default=0.15,
                   help="Part du test set (defaut : 0.15 = 15%%)")

    # Modeles
    p.add_argument("--models", nargs="+", default=AVAILABLE_MODELS,
                   choices=AVAILABLE_MODELS, metavar="MODEL",
                   help=f"Modeles a executer (defaut : tous). Choix : {AVAILABLE_MODELS}")

    # Reproductibilite
    p.add_argument("--seed", type=int, default=42,
                   help="Graine aleatoire (defaut : 42)")

    # Sorties
    p.add_argument("--plot",      metavar="PATH", default=None,
                   help="Sauvegarde le graphique comparatif (ex : bench.png)")
    p.add_argument("--save-csv",  metavar="PATH", default=None,
                   help="Sauvegarde le tableau de metriques en CSV (ex : results.csv)")
    p.add_argument("--save-data", metavar="PATH", default=None,
                   help="Sauvegarde les prix telecharges en CSV (ex : prices.csv)")
    p.add_argument("--no-pi", action="store_true",
                   help="Masque les intervalles de confiance sur le graphique")

    args = p.parse_args()

    # Validation freq / avertissements
    _, _, seasonal_warn = FREQ_CONFIG[args.freq]
    if seasonal_warn and "sarima" in args.models:
        print("[WARN] Frequence hebdomadaire : la periode saisonniere SARIMA (s=5 jours)"
              " n'est pas adaptee aux donnees S. Les resultats SARIMA seront degrades.\n")

    if args.freq == "H":
        print("[INFO] Frequence horaire : yfinance limite le lookback a ~60 jours"
              " calendaires. Assurez-vous que start/end sont recents.\n")

    # ── Donnees ──
    prices = fetch_prices(args.ticker, args.start, args.end, args.freq)
    train, test = split_series(prices, args.test_ratio)

    print(f"[SPLIT] Train : {len(train)} pts ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[SPLIT] Test  : {len(test)}  pts ({test.index[0].date()}  -> {test.index[-1].date()})\n")

    if len(train) < 60:
        print("[WARN] Moins de 60 points d'entrainement — les modeles DL (LSTM) "
              "risquent d'etre instables.\n")

    # Sauvegarde optionnelle des donnees brutes
    if args.save_data:
        prices.to_csv(args.save_data, header=["Close"])
        print(f"[DATA] Prix sauvegardes -> {args.save_data}\n")

    # ── Benchmark ──
    results = run_benchmark(train, test, args.models, args.seed)

    if not results:
        sys.exit("[ERREUR] Aucun modele n'a produit de resultat.")

    # ── Table de comparaison ──
    df_metrics = print_comparison_table(
        results, args.ticker, args.freq, args.start, args.end,
        len(train), len(test),
    )

    # ── Export CSV ──
    if args.save_csv and not df_metrics.empty:
        df_metrics.to_csv(args.save_csv)
        print(f"[CSV] Metriques sauvegardees -> {args.save_csv}")

    # ── Graphique ──
    if args.plot:
        save_comparison_plot(
            results, train, test,
            args.ticker, args.freq, args.plot,
            show_pi=not args.no_pi,
        )


if __name__ == "__main__":
    main()
