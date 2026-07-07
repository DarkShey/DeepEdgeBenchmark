"""
run_benchmark.py — Benchmark multi-actifs x multi-horizons, verdicts régime-calibrés
======================================================================================
Point d'entrée du nouveau benchmark DeepEdgeBenchmark. Compare tous les modèles de
models/ (ARIMA-GARCH, SARIMA, Prophet, LSTM, Naive) sur les 5 actifs de
calibration/regime/assets.py, à 12 horizons (J+1..J+6, S+1..S+4, M+1, M+2), avec un
verdict réussi/échoué/pending par (modèle x actif x horizon) selon que le prix réel
tombe dans l'intervalle de confiance à 95% (élargi selon le régime détecté, cf.
regime_overlay.py). Sortie : base SQLite (db.py) + fenêtre matplotlib interactive avec
cases à cocher (modèles / actifs / horizons).

Mécanisme T (cf. config.py) : les T derniers jours réels téléchargés sont mis de côté
et servent de "futur déjà connu" pour valider les horizons courts/moyens sans attendre
le vrai futur — voir config.T et la section "Mécanisme clé" du brief d'implémentation.

Quick start
-----------
    python -m benchmarks.run_benchmark
    python -m benchmarks.run_benchmark --models "Naive,ARIMA-GARCH" --assets "SPY,BTC-USD"
    python -m benchmarks.run_benchmark --headless Run/benchmark_check.png
"""

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from benchmarks import config, db, regime_overlay
from benchmarks.multi_horizon import MODEL_ADAPTERS


# ── Sélection interactive (terminal) ─────────────────────────────────────────
def prompt_selection(label, options, key_fn=lambda x: str(x)):
    """Numbered multi-select prompt. Empty input (Enter) selects everything."""
    print(f"\n{label} (numéros séparés par des virgules, Entrée = tous) :")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {key_fn(opt)}")
    raw = input("> ").strip()
    if not raw:
        return list(options)
    idxs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            idxs.append(int(chunk))
    chosen = [options[i - 1] for i in idxs if 1 <= i <= len(options)]
    return chosen or list(options)


# ── Pipeline de données (téléchargement + T-trim + split 85/15) ─────────────
def download_full_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit(f"Aucune donnée reçue pour {ticker} entre {start} et {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna(subset=["Close"])
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    return raw


def split_data(full_data: pd.DataFrame, t_days: int, train_val_split: float) -> dict:
    """Applique le trim T puis le split train/validation 85/15 sur les données restantes.
    `combined_future` = validation + queue cachée : vérité terrain pour les verdicts."""
    if t_days > 0:
        effective = full_data.iloc[:-t_days]
        hidden_future = full_data.iloc[len(effective):]
    else:
        effective = full_data
        hidden_future = full_data.iloc[0:0]

    split_idx = int(len(effective) * train_val_split)
    train = effective.iloc[:split_idx]
    val = effective.iloc[split_idx:]
    combined_future = pd.concat([val, hidden_future])

    return {
        "full": full_data, "effective": effective, "hidden_future": hidden_future,
        "train": train, "val": val, "combined_future": combined_future,
    }


def compute_verdict(combined_future: pd.DataFrame, h_days: int, lo: float, hi: float):
    """Retourne (target_date, actual, verdict). verdict='pending' si le futur caché
    (val + queue T) ne couvre pas encore cet horizon."""
    if h_days <= len(combined_future):
        target_date = combined_future.index[h_days - 1]
        actual = float(combined_future["Close"].iloc[h_days - 1])
        verdict = "success" if lo <= actual <= hi else "fail"
        return target_date, actual, verdict
    return None, None, "pending"


# ── Un modèle x un actif -> une ligne de résultat par horizon ────────────────
def run_model_for_asset(model_name: str, adapter_fn, ticker: str, split: dict,
                        regime_state, epochs: int, seed: int) -> list:
    train_close = split["train"]["Close"]
    distinct_days = sorted(set(config.HORIZONS.values()))

    if model_name == "LSTM":
        raw_results = adapter_fn(train_close, distinct_days, epochs=epochs, seed=seed)
    else:
        raw_results = adapter_fn(train_close, distinct_days)

    anchor_date = train_close.index[-1]
    stress_score = regime_state.stress_score
    regime_tag = regime_state.dominant_regime()
    vol_bucket = regime_state.vol_bucket

    rows = []
    for label, h_days in config.HORIZONS.items():
        point, lo, hi = raw_results[h_days]
        lo2, hi2 = regime_overlay.scale_interval(point, lo, hi, stress_score)
        target_date, actual, verdict = compute_verdict(split["combined_future"], h_days, lo2, hi2)
        rows.append({
            "model": model_name, "ticker": ticker, "horizon_label": label,
            "horizon_days": h_days, "anchor_date": str(anchor_date.date()),
            "target_date": str(target_date.date()) if target_date is not None else None,
            "point_forecast": point, "pi_lower": lo2, "pi_upper": hi2, "actual": actual,
            "verdict": verdict, "regime_tag": regime_tag,
            "stress_score": stress_score, "vol_bucket": vol_bucket,
        })
    return rows


# ── Fenêtre interactive (matplotlib + cases à cocher) ────────────────────────
def launch_gui(results_df: pd.DataFrame, headless: str = None) -> None:
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.widgets import CheckButtons

    models = sorted(results_df["model"].unique())
    tickers = sorted(results_df["ticker"].unique())
    horizon_labels = [h for h in config.HORIZONS if h in set(results_df["horizon_label"])]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 4.2],
                          height_ratios=[2.4, 1.7, 0.4], hspace=0.4, wspace=0.4)

    ax_models   = fig.add_subplot(gs[0, 0]); ax_models.set_title("Modèles", fontsize=9)
    ax_assets   = fig.add_subplot(gs[0, 1]); ax_assets.set_title("Actifs", fontsize=9)
    ax_horizons = fig.add_subplot(gs[0, 2]); ax_horizons.set_title("Horizons", fontsize=9)
    ax_heatmap  = fig.add_subplot(gs[0, 3])
    ax_table    = fig.add_subplot(gs[1, :]); ax_table.axis("off")
    ax_summary  = fig.add_subplot(gs[2, :]); ax_summary.axis("off")

    cb_models   = CheckButtons(ax_models, models, [True] * len(models))
    cb_assets   = CheckButtons(ax_assets, tickers, [True] * len(tickers))
    cb_horizons = CheckButtons(ax_horizons, horizon_labels, [True] * len(horizon_labels))

    VERDICT_CODE = {"fail": -1.0, "pending": 0.0, "success": 1.0}
    CMAP = ListedColormap([(0.80, 0.22, 0.22), (0.75, 0.75, 0.75), (0.22, 0.62, 0.32)])
    CMAP.set_bad(color="white")

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v)

    def active(cb, labels):
        return [l for l, checked in zip(labels, cb.get_status()) if checked]

    def redraw(_event=None):
        active_models   = active(cb_models, models)
        active_tickers  = active(cb_assets, tickers)
        active_horizons = active(cb_horizons, horizon_labels)

        sub = results_df[
            results_df["model"].isin(active_models)
            & results_df["ticker"].isin(active_tickers)
            & results_df["horizon_label"].isin(active_horizons)
        ]

        ax_heatmap.clear()
        ax_table.clear(); ax_table.axis("off")
        ax_summary.clear(); ax_summary.axis("off")

        if sub.empty:
            ax_heatmap.text(0.5, 0.5, "Aucune sélection", ha="center", va="center")
            fig.canvas.draw_idle()
            return

        row_keys = sorted({f"{r.ticker}/{r.model}" for r in sub.itertuples()})
        col_keys = [h for h in config.HORIZONS if h in active_horizons]
        grid = np.full((len(row_keys), len(col_keys)), np.nan)
        for r in sub.itertuples():
            grid[row_keys.index(f"{r.ticker}/{r.model}"), col_keys.index(r.horizon_label)] = \
                VERDICT_CODE[r.verdict]

        ax_heatmap.imshow(np.ma.masked_invalid(grid), cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
        ax_heatmap.set_xticks(range(len(col_keys)))
        ax_heatmap.set_xticklabels(col_keys, rotation=45, ha="right", fontsize=8)
        ax_heatmap.set_yticks(range(len(row_keys)))
        ax_heatmap.set_yticklabels(row_keys, fontsize=8)
        ax_heatmap.set_title("Verdicts (vert=réussi, rouge=échoué, gris=pending)", fontsize=9)

        detail_cols = ["ticker", "model", "horizon_label", "target_date", "point_forecast",
                      "pi_lower", "pi_upper", "actual", "verdict", "regime_tag"]
        col_labels = ["Actif", "Modèle", "Horizon", "Date cible", "Prévision", "IC bas",
                     "IC haut", "Réel", "Verdict", "Régime"]
        max_rows = 15
        detail = sub[detail_cols].sort_values(["ticker", "model", "horizon_label"]).head(max_rows)
        cell_text = [[fmt(v) for v in row] for row in detail.values]
        if cell_text:
            tbl = ax_table.table(cellText=cell_text, colLabels=col_labels,
                                 cellLoc="center", loc="center")
            # scale(1, y>1) overflows the axes box regardless of row count (matplotlib
            # auto-fits rows to 1/n_rows of the box, so scaling y stretches past it) —
            # keep y<=1 and rely on fontsize for legibility instead.
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 0.9)
        note = "" if len(sub) <= max_rows else f"  ({max_rows}/{len(sub)} lignes affichées)"
        ax_table.set_title(f"Détail{note}", fontsize=9, loc="left", pad=14)

        checked = sub[sub["verdict"] != "pending"]
        if len(checked):
            rate = 100.0 * (checked["verdict"] == "success").mean()
            summary = (f"Sélection : {len(sub)} verdicts   |   Vérifiables : {len(checked)}   |   "
                      f"Taux de réussite (couverture IC95%) : {rate:.1f}%")
        else:
            summary = f"Sélection : {len(sub)} verdicts   |   Aucun encore vérifiable (tous 'pending')"
        ax_summary.text(0.0, 0.5, summary, fontsize=10, va="center")

        fig.canvas.draw_idle()

    cb_models.on_clicked(redraw)
    cb_assets.on_clicked(redraw)
    cb_horizons.on_clicked(redraw)

    redraw()
    fig.suptitle("DeepEdgeBenchmark — Verdicts multi-actifs x multi-horizons",
                fontsize=13, fontweight="bold")

    if headless:
        fig.savefig(headless, dpi=130, bbox_inches="tight")
        print(f"[run_benchmark] figure sauvegardée -> {headless}")
    else:
        plt.show()


# ── Orchestration ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="DeepEdgeBenchmark — multi-asset x multi-horizon verdicts")
    p.add_argument("--models", default=None,
                   help="liste de modèles séparés par des virgules (contourne le prompt interactif)")
    p.add_argument("--assets", default=None,
                   help="liste de tickers séparés par des virgules (contourne le prompt interactif)")
    p.add_argument("--epochs", type=int, default=20, help="épochs LSTM (rollout multi-horizon)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", metavar="PATH", default=None,
                   help="sauvegarde la figure dans PATH au lieu d'ouvrir la fenêtre interactive")
    args = p.parse_args()

    model_names = list(MODEL_ADAPTERS.keys())
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        selected_models = [m for m in model_names if m in wanted]
    else:
        selected_models = prompt_selection("Modèles à activer", model_names)

    if args.assets:
        wanted = {t.strip() for t in args.assets.split(",")}
        selected_assets = [a for a in config.ASSETS if a["ticker"] in wanted]
    else:
        selected_assets = prompt_selection(
            "Actifs à activer", config.ASSETS,
            key_fn=lambda a: f'{a["ticker"]} ({a["label"]})',
        )

    if not selected_models or not selected_assets:
        raise SystemExit("Aucun modèle ou aucun actif sélectionné — rien à faire.")

    data_end = config.DATA_END or datetime.today().strftime("%Y-%m-%d")
    conn = db.init_db(config.DB_PATH)
    run_id = db.insert_run(conn, config.T, config.DATA_START, data_end, config.TRAIN_VAL_SPLIT)
    print(f"[run_benchmark] run_id={run_id}  T={config.T}  split={config.TRAIN_VAL_SPLIT}  "
          f"fenêtre {config.DATA_START} -> {data_end}")

    all_rows = []
    for asset in selected_assets:
        ticker = asset["ticker"]
        print(f"\n[run_benchmark] {ticker} : téléchargement ...")
        full_data = download_full_data(ticker, config.DATA_START, data_end)
        split = split_data(full_data, config.T, config.TRAIN_VAL_SPLIT)
        print(f"  {len(full_data)} jours -> train={len(split['train'])} val={len(split['val'])} "
              f"caché(T={config.T})={len(split['hidden_future'])}")

        print(f"[run_benchmark] {ticker} : calibration régime ...")
        regime_state = regime_overlay.fit_predict_regime(split["train"], split["train"].index[-1])
        print(f"  régime dominant = {regime_state.dominant_regime()}  "
              f"stress_score={regime_state.stress_score:.3f}  vol_bucket={regime_state.vol_bucket}")

        for model_name in selected_models:
            adapter_fn = MODEL_ADAPTERS[model_name]
            t0 = time.time()
            try:
                rows = run_model_for_asset(model_name, adapter_fn, ticker, split,
                                           regime_state, args.epochs, args.seed)
                all_rows.extend(rows)
                print(f"  {model_name:<12} ok ({time.time() - t0:.1f}s)")
            except Exception as exc:
                print(f"  {model_name:<12} ECHEC : {exc}")

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        raise SystemExit("Aucun résultat calculé (tous les modèles ont échoué).")

    db.insert_results_df(conn, run_id, results_df)
    print(f"\n[run_benchmark] {len(results_df)} verdicts écrits dans {config.DB_PATH} (run_id={run_id})")

    launch_gui(results_df, headless=args.headless)


if __name__ == "__main__":
    main()
