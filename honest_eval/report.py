"""
report.py — the honest dashboard (Points 1, 3, 4)
=================================================
Renders the regenerated dashboard where every performance claim is measured
against the corrected naive with explicit statistical uncertainty:

  * Variations panel (Point 1): Δpred vs Δreal scatter + change time series.
    The old levels overlay is demoted to a faint background — it created the
    illusion of skill.
  * KPI table: MASE, Theil's U, DirAcc ± 95% CI, DM vs naive, plain-language
    verdict ("no better than naive" when U≈1 & DM n.s.).
  * Error-vs-horizon curves (Point 3): RMSE/MASE as h grows, model vs naive.
  * Volatility & direction tabs (Point 4): each target with its own baseline
    and a "beats / does not beat" verdict.

Pure matplotlib; figures are returned so the caller can save and/or show them.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import metrics

_COLORS = {"arima": "tab:blue", "sarima": "tab:green",
           "prophet": "tab:purple", "lstm": "tab:red", "ensemble": "darkorange"}
_GREEN, _RED, _NAIF, _WHITE = "#d0f0d0", "#f7d0d0", "#f5f5e6", "white"


# ── Point 1: variations panel ────────────────────────────────────────────────

def plot_variations_panel(d1_results, naive_rows, ticker=""):
    """Δpred vs Δreal (the honest view) with the levels overlay demoted.

    ``d1_results`` : {name: {"prev","actual","pred","index"}}   (D+1 rows)
    ``naive_rows`` : {"prev","actual","index"}                  (reference)
    """
    fig = plt.figure(figsize=(15, 9), num=f"Honest eval — variations | {ticker}")
    fig.suptitle(f"Point 1 — scoring CHANGES, not levels   ({ticker})",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.22,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_sc = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_lv = fig.add_subplot(gs[1, :])

    prev = naive_rows["prev"]
    actual = naive_rows["actual"]
    dreal = actual - prev
    idx = naive_rows["index"]

    lim = np.nanpercentile(np.abs(dreal), 99) * 1.1 if len(dreal) else 1.0
    ax_sc.axline((0, 0), slope=1, color="#999", ls="--", lw=0.8)
    ax_sc.axhline(0, color="#ccc", lw=0.6); ax_sc.axvline(0, color="#ccc", lw=0.6)
    for name, r in d1_results.items():
        dpred = r["pred"] - r["prev"]
        cc = metrics.change_correlation(r["pred"], r["prev"], r["actual"])
        ax_sc.scatter(r["actual"] - r["prev"], dpred, s=10, alpha=0.5,
                      color=_COLORS.get(name, "gray"),
                      label=f"{name}  ρ(Δ)={cc:+.2f}" if np.isfinite(cc) else f"{name}  ρ=n/a")
    ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel("Δreal = price_t − price_{t-1}")
    ax_sc.set_ylabel("Δpred = pred_t − price_{t-1}")
    ax_sc.set_title("Predicted change vs realised change\n(points on the 45° line = real skill)",
                    fontsize=10)
    ax_sc.legend(fontsize=8, loc="upper left"); ax_sc.grid(alpha=0.2)

    ax_ts.plot(idx, dreal, color="black", lw=1.0, label="Δreal", zorder=5)
    for name, r in d1_results.items():
        ax_ts.plot(r["index"], r["pred"] - r["prev"], lw=0.9, alpha=0.7,
                   color=_COLORS.get(name, "gray"), label=f"Δpred {name}")
    ax_ts.axhline(0, color="#888", lw=0.6, ls="--")
    ax_ts.set_title("Change time series (naive Δpred ≡ 0)", fontsize=10)
    ax_ts.legend(fontsize=8, ncol=2); ax_ts.grid(alpha=0.2)

    # demoted levels overlay — faint, labelled as illusory
    ax_lv.plot(idx, actual, color="#bbbbbb", lw=1.2, label="Actual (levels)")
    for name, r in d1_results.items():
        ax_lv.plot(r["index"], r["pred"], lw=0.8, alpha=0.35,
                   color=_COLORS.get(name, "gray"), label=name)
    ax_lv.set_title("Levels overlay (DEMOTED — corr(pred_t, price_{t-1}) ≈ 1 "
                    "creates a false impression of skill)", fontsize=9, color="#888")
    ax_lv.legend(fontsize=7, ncol=5); ax_lv.grid(alpha=0.15)
    ax_lv.set_ylabel("Price")
    return fig


# ── Point 1: KPI table ───────────────────────────────────────────────────────

def plot_kpi_table(kpi_rows, ticker="", horizon=1):
    """Render the honest KPI table.

    ``kpi_rows`` : list of dicts with keys name, rmse, mase, theil_u,
    change_corr, dir_acc, dir_ci95, dir_p, coverage, dm, dm_p, verdict.
    """
    cols = ["Model", "RMSE", "MASE", "Theil U", "ρ(Δ)",
            "DirAcc% [95% CI]", "p(coin)", "Cover%", "DM/naive (p)", "Verdict"]
    fig = plt.figure(figsize=(16, 0.6 + 0.5 * (len(kpi_rows) + 2)),
                     num=f"Honest eval — KPI | {ticker}")
    ax = fig.add_subplot(111); ax.axis("off")
    ax.set_title(f"Honest KPI table — {ticker}  |  horizon D+{horizon}\n"
                 "U≈1 & DM n.s. ⇒ the model adds nothing vs naive",
                 fontsize=12, fontweight="bold", loc="left", pad=12)

    text, colours = [], []
    for r in kpi_rows:
        lo, hi = r.get("dir_ci95", (float("nan"), float("nan")))
        da = r.get("dir_acc", float("nan"))
        text.append([
            r["name"],
            f"{r['rmse']:.3f}" if np.isfinite(r.get("rmse", np.nan)) else "--",
            f"{r['mase']:.3f}" if np.isfinite(r.get("mase", np.nan)) else "--",
            f"{r['theil_u']:.3f}" if np.isfinite(r.get("theil_u", np.nan)) else "--",
            f"{r['change_corr']:+.2f}" if np.isfinite(r.get("change_corr", np.nan)) else "n/a",
            f"{da*100:.1f} [{lo*100:.0f},{hi*100:.0f}]" if np.isfinite(da) else "--",
            f"{r['dir_p']:.3f}" if np.isfinite(r.get("dir_p", np.nan)) else "--",
            f"{r['coverage']:.1f}" if np.isfinite(r.get("coverage", np.nan)) else "--",
            f"{r['dm']:+.2f} ({r['dm_p']:.3f})" if np.isfinite(r.get("dm", np.nan)) else "--",
            r.get("verdict", ""),
        ])
        row_c = [_WHITE] * len(cols)
        u = r.get("theil_u", np.nan)
        if np.isfinite(u):
            row_c[3] = _GREEN if u < 1 else _RED
        v = r.get("verdict", "")
        row_c[9] = _GREEN if "beats" in v else (_RED if "worse" in v else _NAIF)
        colours.append(row_c)

    tbl = ax.table(cellText=text, colLabels=cols, cellColours=colours,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#3a5a8a"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    return fig


# ── Point 3: error-vs-horizon curves ─────────────────────────────────────────

def plot_error_vs_horizon(curves, naive_curve=None, ticker=""):
    """``curves`` : {name: DataFrame(index=h, cols rmse,mase,...)}."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   num=f"Honest eval — error vs horizon | {ticker}")
    fig.suptitle(f"Point 3 — degradation vs horizon   ({ticker})",
                 fontsize=13, fontweight="bold")
    for name, df in curves.items():
        ax1.plot(df.index, df["rmse"], marker="o", ms=3,
                 color=_COLORS.get(name, "gray"), label=name)
        ax2.plot(df.index, df["mase"], marker="o", ms=3,
                 color=_COLORS.get(name, "gray"), label=name)
    if naive_curve is not None:
        ax1.plot(naive_curve.index, naive_curve["rmse"], color="black", ls="--", label="naive")
    ax2.axhline(1.0, color="black", ls="--", lw=0.8, label="naive (U=1)")
    ax1.set_xlabel("horizon h (days)"); ax1.set_ylabel("RMSE"); ax1.set_title("RMSE vs horizon")
    ax2.set_xlabel("horizon h (days)"); ax2.set_ylabel("MASE (MAE/naive)")
    ax2.set_title("MASE vs horizon  (>1 = worse than naive)")
    for a in (ax1, ax2):
        a.grid(alpha=0.25); a.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ── Point 4: volatility & direction tabs ─────────────────────────────────────

def plot_targets_panel(vol_res=None, dir_res=None, ticker=""):
    fig = plt.figure(figsize=(15, 6), num=f"Honest eval — new targets | {ticker}")
    fig.suptitle(f"Point 4 — targets with signal   ({ticker})",
                 fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.06, right=0.97, top=0.88, bottom=0.1)

    axv = fig.add_subplot(gs[0, 0]); axv.axis("off")
    axv.set_title("Volatility target (QLIKE ↓ better)", fontsize=11, loc="left")
    if vol_res:
        rows, cols = [], ["Method", "QLIKE", "MSE(var)", "PIT p", "Cover%", "Verdict"]
        for m in ("persistence", "ewma", "garch"):
            d = vol_res.get(m, {})
            verdict = ("baseline" if m == "persistence"
                       else ("beats persistence" if d.get("beats_persistence") else "no"))
            rows.append([m, f"{d.get('qlike','--'):.4f}", f"{d.get('mse_var',0):.2e}",
                         f"{d.get('pit_p','--'):.3f}", f"{d.get('ret_coverage','--'):.1f}", verdict])
        t = axv.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
        for j in range(len(cols)):
            t[0, j].set_facecolor("#3a5a8a"); t[0, j].set_text_props(color="white", fontweight="bold")

    axd = fig.add_subplot(gs[0, 1]); axd.axis("off")
    axd.set_title("Direction target (AUC ↑, Brier ↓)", fontsize=11, loc="left")
    if dir_res:
        rows, cols = [], ["Metric", "Model", "Always-up", "Majority"]
        rows.append(["AUC", f"{dir_res.get('auc','--')}", "0.500", "0.500"])
        rows.append(["Brier", f"{dir_res.get('brier','--')}",
                     f"{dir_res.get('brier_alwaysup','--')}", f"{dir_res.get('brier_majority','--')}"])
        rows.append(["Hit-rate", f"{dir_res.get('hit_rate','--')}", "--", "--"])
        rows.append(["p vs coin", f"{dir_res.get('binom_p_vs_coin','--')}", "--", "--"])
        rows.append(["Verdict", dir_res.get("verdict", ""), "", ""])
        t = axd.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
        for j in range(len(cols)):
            t[0, j].set_facecolor("#3a5a8a"); t[0, j].set_text_props(color="white", fontweight="bold")
    return fig


def save_all(figs, base_path):
    """Save a dict {suffix: fig} to ``{base}_{suffix}.png`` and return paths."""
    import os
    base, ext = os.path.splitext(base_path)
    ext = ext or ".png"
    paths = []
    for suf, fig in figs.items():
        p = f"{base}_{suf}{ext}"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        paths.append(p)
    return paths
