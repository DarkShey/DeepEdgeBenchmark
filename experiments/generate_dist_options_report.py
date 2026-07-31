"""
generate_dist_options_report.py — builds the HTML comparison report (calibration
options 1/2/3, all 5 models) from dist_options_summary.json.

NOT committed to git on purpose: the report embeds Cambria/Consolas (Microsoft-
licensed fonts) as base64 data URIs for a self-contained, offline-renderable page.
Shipping those font bytes in a git repo is a redistribution concern -- this script
instead subsets + embeds them AT BUILD TIME from the *local* machine's own
C:\\Windows\\Fonts (each such machine already carries its own Windows/Office font
license; nothing is copied into version control). Run this locally to regenerate
the report; the output HTML (and the intermediate font subsets) stay out of git --
see .gitignore. If Cambria/Consolas aren't available (non-Windows machine), the
FALLBACK_FONTS branch below skips embedding and falls back to system font stacks.

Usage (from repo root):
    python experiments/generate_dist_options_report.py
    python experiments/generate_dist_options_report.py --out /tmp/report.html
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_JSON = ROOT / "experiments" / "dist_options_summary.json"
WINDOWS_FONTS = Path(r"C:\Windows\Fonts")
FONT_SOURCES = {
    "cambria_bold": WINDOWS_FONTS / "cambriab.ttf",
    "consolas_reg": WINDOWS_FONTS / "consola.ttf",
    "consolas_bold": WINDOWS_FONTS / "consolab.ttf",
}
SUBSET_UNICODES = "U+0020-007E,U+00A0-017F,U+2010-2027,U+2030,U+2032-2033,U+2190-2199,U+00B1,U+00B0"

MODEL_ORDER = ["SARIMA", "Prophet", "LSTM", "Naive", "ARIMA-GARCH"]
VARIANT_LABEL = {
    "normal": "Gaussienne (actuel)",
    "student_t": "Student-t (fit manuel)",
    "ged": "GED (fit manuel)",
    "cqr": "CQR",
    "native_ged": "GED natif (refit GARCH)",
    "native_skewt": "Skew-t natif (refit GARCH)",
}
VARIANT_COLOR = {
    "normal":       ("#2a78d6", "#3987e5"),
    "student_t":    ("#eb6834", "#d95926"),
    "ged":          ("#1baf7a", "#199e70"),
    "cqr":          ("#eda100", "#c98500"),
    "native_ged":   ("#e87ba4", "#d55181"),
    "native_skewt": ("#008300", "#008300"),
}
VARIANT_ORDER = ["normal", "student_t", "ged", "cqr", "native_ged", "native_skewt"]

# Hand-authored verdicts -- numbers pulled live from dist_options_summary.json's
# `mace_strict` (mean absolute calibration error, averaged PER ASSET then
# absolute-valued -- no cancellation across assets of opposite sign). An earlier
# version of this report used `mace_loose` (abs of the cross-asset mean), which
# lets a model that over-covers on one asset and under-covers on another look
# "calibrated" on average when neither asset actually is -- flagged by an
# independent robustness re-run, see HANDOFF_sigma_calibration_suivi.md. `best`/
# `status`/prose kept in sync with experiments/dist_options_summary.json as of
# the 30 juillet 2026 mace_strict fix (5-asset run, single window 2020-2024).
VERDICTS = {
    "SARIMA": dict(status="good", headline="Vaut le coût — CQR nettement devant", best="cqr",
        text="CQR ramène l'erreur de calibration de 8,07 à 4,43 points (−45 %) ; "
             "Student-t (6,22) et GED (7,45) aident aussi mais moins. Coût quasi nul "
             "(0,0006 à 0,02 s sur un backtest qui en prend déjà 70). Largeur des PI "
             "réduite en prime pour CQR/Student-t, pas juste une meilleure couverture obtenue "
             "en élargissant."),
    "Prophet": dict(status="critical", headline="Aucune option testée ne suffit", best="cqr",
        text="Sous-couverture massive à tous les niveaux sous loi normale (24 % à 50 %, "
             "70 % à 95 % — erreur moyenne 28,1 points). Student-t/GED ne changent "
             "presque rien (28,2 / 28,3) : le problème n'est pas la forme de la queue, c'est "
             "que le σ lui-même est sous-estimé. CQR aide (20,7, −26 %) sans "
             "corriger le fond. Ce n'est pas un problème de loi — l'incertitude de Prophet dans "
             "ce pipeline mérite une investigation dédiée avant d'y toucher davantage."),
    "LSTM": dict(status="warning", headline="CQR aide un peu, ne règle pas le vrai problème",
        best="cqr",
        text="CQR est la seule des trois options à faire mieux que la loi normale ici "
             "(10,96 → 8,43 points, −23 %) — mais Student-t et GED empirent "
             "nettement l'erreur (12,36 et 12,53, jusqu'à +14 %) : le σ de LSTM est "
             "une seule valeur figée sur les résidus d'entraînement, pas un chemin qui varie "
             "dans le temps comme pour ARIMA-GARCH — lui changer juste la forme de queue ne "
             "corrige pas un problème de niveau. CQR n'est qu'un pansement partiel ; la vraie "
             "cause (σ figé) demande une correction dynamique plutôt qu'un changement de "
             "forme."),
    "Naive": dict(status="good", headline="Vaut le coût — CQR nettement devant", best="cqr",
        text="CQR ramène l'erreur de 9,26 à 3,76 points (−59 %), loin devant "
             "Student-t (7,06) et surtout GED (7,81 — à peine mieux que ne rien faire). Coût "
             "négligeable. Contrairement à SARIMA/ARIMA-GARCH, changer seulement la forme de la "
             "queue (GED) ne suffit pas ici — CQR, qui recalibre directement sur les données "
             "plutôt que de supposer une famille, s'en sort nettement mieux."),
    "ARIMA-GARCH": dict(status="good", headline="Le refit natif l'emporte pour de bon",
        best="native_ged",
        text="Le refit natif (GED ou skew-t, vrai réajustement du GARCH sous la nouvelle loi) "
             "l'emporte dans l'absolu : 2,51 / 2,52 points contre 3,27 pour le swap manuel "
             "Student-t le moins cher — et CQR fait ici moins bien (4,27) que les corrections "
             "de forme. Le refit natif coûte ~2,4 s de plus par actif contre "
             "~0,004 s pour le swap manuel — négligeable si le calcul n'est pas la "
             "contrainte ; sinon, Student-t manuel reste la meilleure option gratuite."),
}
MDN_VERDICT = dict(status="critical", headline="Ne vaut pas le coût, en l'état",
    text="Erreur de calibration moyenne 13,74 points contre 10,96 pour le LSTM de production "
        "(pire, +25 %, alors que le LSTM sous-couvre déjà) — pour +23 % de temps "
        "d'entraînement (37,6 s contre 30,6 s) et une instabilité réelle d'un "
        "entraînement à l'autre : même seed, même code, la couverture à 50 % sur SPY est "
        "passée de 66 % à 14 % entre deux runs identiques avant stabilisation "
        "(clipping de gradient + patience augmentée), et reste dispersée (écart-type jusqu'à "
        "4,2 points sur 3 seeds). Un réseau à mélange de gaussiennes correctement stabilisé "
        "demanderait plus d'ingénierie que ce prototype pour espérer battre le ruban "
        "±1,96·σ actuel — pas rentable tel quel.")


def subset_fonts_b64(tmpdir: Path) -> dict:
    """{'cambria': b64, 'consolas_reg': b64, 'consolas_bold': b64} or {} if the
    local machine doesn't have these fonts (non-Windows / different install)."""
    if not all(p.exists() for p in FONT_SOURCES.values()):
        print("Cambria/Consolas not found locally -- falling back to system font "
             "stacks, no fonts embedded.", file=sys.stderr)
        return {}
    out = {}
    for key, src in FONT_SOURCES.items():
        dst = tmpdir / f"{key}.woff"
        subprocess.run([
            sys.executable, "-m", "fontTools.subset", str(src),
            f"--output-file={dst}", "--flavor=woff",
            f"--unicodes={SUBSET_UNICODES}", "--layout-features=*",
            "--no-hinting", "--desubroutinize",
        ], check=True, capture_output=True)
        out[key] = base64.b64encode(dst.read_bytes()).decode("ascii")
    return out


def fmt(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


def bar_chart_svg(rows, max_val, width=560, bar_h=22, gap=14, left_pad=190, chart_id=""):
    n = len(rows)
    plot_w = width - left_pad - 60
    row_h = bar_h + gap
    height = n * row_h + 10
    svg = [f'<svg class="barchart" data-chart="{chart_id}" viewBox="0 0 {width} {height}" '
          f'width="100%" role="img" aria-label="Comparaison par option">']
    svg.append(f'<line x1="{left_pad}" y1="4" x2="{left_pad}" y2="{height-6}" class="axis-line"/>')
    for i, (label, value, c_light, c_dark, vkey) in enumerate(rows):
        y = 8 + i * row_h
        bw = max(2, (value / max_val) * plot_w) if max_val > 0 else 2
        cy = y + bar_h / 2
        svg.append(f'<text x="{left_pad-10}" y="{cy+4}" class="bar-label" text-anchor="end">{label}</text>')
        svg.append(f'<rect x="{left_pad}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="4" '
                  f'class="bar bar-{vkey}"><title>{label}: {fmt(value)} points d\'erreur de calibration moyenne</title></rect>')
        svg.append(f'<text x="{left_pad+bw+8:.1f}" y="{cy+4}" class="bar-value">{fmt(value)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def model_section(summary, model_key):
    m = summary["models"][model_key]
    variants = ["normal"] + [v for v in VARIANT_ORDER if v in m and v != "normal"]
    rows = [(VARIANT_LABEL[v], m[v]["mace_strict"], *VARIANT_COLOR[v], v) for v in variants]
    max_val = max(r[1] for r in rows) * 1.18
    verdict = VERDICTS[model_key]
    chart_svg = bar_chart_svg(rows, max_val, chart_id=model_key)

    thead = ("<tr><th>Option</th><th>Cov 50%</th><th>Cov 80%</th><th>Cov 95%</th>"
            "<th>Largeur rel.</th><th>Pinball rel.</th><th>Surcoût</th></tr>")
    trows = []
    base_time = m["base_train_time_s_mean"]
    for v in variants:
        d = m[v]
        cov = d["cov_mean"]
        oh = d["overhead_s_mean"]
        oh_txt = f"+{oh:.3f}s" if oh < 1 else f"+{oh:.2f}s"
        rw = d["rel_width_mean"]
        trows.append(
            f'<tr class="{"row-best" if v == verdict["best"] else ""}">'
            f'<td><span class="swatch" style="--c-light:{VARIANT_COLOR[v][0]};--c-dark:{VARIANT_COLOR[v][1]}"></span>{VARIANT_LABEL[v]}</td>'
            f'<td>{fmt(cov["50"],1)}%</td><td>{fmt(cov["80"],1)}%</td><td>{fmt(cov["95"],1)}%</td>'
            f'<td>{fmt(rw["50"],2)}×</td><td>{fmt(d["rel_pinball_mean"],3)}×</td>'
            f'<td class="mono">{oh_txt}</td></tr>')
    table = f'<table class="kpi-table"><thead>{thead}</thead><tbody>{"".join(trows)}</tbody></table>'

    return f"""
    <section class="model-section" id="model-{model_key.lower().replace(' ', '-')}">
      <div class="model-head">
        <h3>{model_key}</h3>
        <span class="pill pill-{verdict['status']}">{verdict['headline']}</span>
      </div>
      <p class="model-verdict-text">{verdict['text']}</p>
      <div class="model-body">
        <div class="chart-wrap">{chart_svg}</div>
        <div class="table-wrap">{table}</div>
      </div>
      <p class="base-time">Backtest de base ({model_key}) : <span class="mono">{fmt(base_time,1)}s</span>
        par actif, avant tout surcoût d'option.</p>
    </section>"""


def verdict_card(summary, model_key):
    m = summary["models"][model_key]
    v = VERDICTS[model_key]
    best = v["best"]
    before = m["normal"]["mace_strict"]
    after = m[best]["mace_strict"]
    delta_pct = (after - before) / before * 100
    return f"""
    <a class="verdict-card status-{v['status']}" href="#model-{model_key.lower().replace(' ', '-')}">
      <div class="verdict-card-top">
        <span class="verdict-model">{model_key}</span>
        <span class="pill pill-{v['status']} pill-sm">{v['headline']}</span>
      </div>
      <div class="verdict-metric">
        <span class="verdict-before">{fmt(before,1)}</span>
        <span class="verdict-arrow">\u2192</span>
        <span class="verdict-after">{fmt(after,1)}</span>
        <span class="verdict-unit">pts d'erreur</span>
      </div>
      <div class="verdict-delta {'delta-good' if delta_pct < 0 else 'delta-bad'}">{'' if delta_pct<0 else '+'}{fmt(delta_pct,0)}%</div>
    </a>"""


def mdn_section(summary):
    mdn = summary["mdn"]
    base = summary["lstm_baseline_for_mdn_comparison"]
    v = MDN_VERDICT
    rows = [
        ("LSTM (production)", base["mace_strict"], *VARIANT_COLOR["normal"], "normal"),
        ("LSTM-MDN (K=3, moy. 3 seeds)", mdn["mace_strict"], "#4a3aa7", "#9085e9", "mdn"),
    ]
    max_val = max(r[1] for r in rows) * 1.2
    chart_svg = bar_chart_svg(rows, max_val, chart_id="mdn-cal", bar_h=26, gap=20)

    assets = list(mdn["per_asset"].keys())
    strip_w, strip_h, pad_l = 560, 46 * len(assets) + 10, 190
    parts = [f'<svg class="barchart" viewBox="0 0 {strip_w} {strip_h}" width="100%" role="img" '
            f'aria-label="Instabilit\u00e9 MDN par actif (couverture 50% sur 3 seeds)">']
    parts.append(f'<line x1="{pad_l}" y1="4" x2="{pad_l}" y2="{strip_h-6}" class="axis-line"/>')
    xmax, plot_w = 100, strip_w - pad_l - 60
    for i, a in enumerate(assets):
        pa = mdn["per_asset"][a]
        y = 8 + i * 46
        cy = y + 13
        mean, std = pa["cov_50_mean"], pa["cov_50_std"]
        lo, hi = max(0, mean - std), min(100, mean + std)
        x_target = pad_l + (50 / xmax) * plot_w
        x_lo, x_hi = pad_l + (lo / xmax) * plot_w, pad_l + (hi / xmax) * plot_w
        x_mean = pad_l + (mean / xmax) * plot_w
        parts.append(f'<text x="{pad_l-10}" y="{cy+4}" class="bar-label" text-anchor="end">{a}</text>')
        parts.append(f'<line x1="{x_target:.1f}" y1="{y-2}" x2="{x_target:.1f}" y2="{y+28}" class="target-line"/>')
        parts.append(f'<line x1="{x_lo:.1f}" y1="{cy}" x2="{x_hi:.1f}" y2="{cy}" class="range-line-mdn"/>')
        parts.append(f'<circle cx="{x_mean:.1f}" cy="{cy}" r="6" class="range-dot-mdn">'
                     f'<title>{a}: cov 50% = {fmt(mean,1)}% (\u00b1{fmt(std,1)}, cible 50%)</title></circle>')
        parts.append(f'<text x="{x_hi+10:.1f}" y="{cy+4}" class="bar-value">{fmt(mean,1)}% \u00b1{fmt(std,1)}</text>')
    parts.append("</svg>")
    instability_svg = "\n".join(parts)

    return f"""
    <section class="mdn-section" id="mdn-option3">
      <div class="model-head">
        <h3>Option 3 — Mixture Density Network (LSTM)</h3>
        <span class="pill pill-{v['status']}">{v['headline']}</span>
      </div>
      <p class="model-verdict-text">{v['text']}</p>
      <div class="mdn-grid">
        <div class="chart-wrap">
          <h4 class="chart-subtitle">Erreur de calibration moyenne (5 actifs)</h4>
          {chart_svg}
        </div>
        <div class="chart-wrap">
          <h4 class="chart-subtitle">Couverture à 50% par actif — moyenne \u00b1 écart-type sur 3 seeds
            <span class="chart-subtitle-note">(cible : 50%, ligne pointillée)</span></h4>
          {instability_svg}
        </div>
      </div>
      <p class="base-time">Coût : <span class="mono">{fmt(mdn['mean_train_time_s'],1)}s</span> en moyenne
        par entraînement (×3 seeds pour mesurer la stabilité) contre
        <span class="mono">{fmt(base['train_time_s'],1)}s</span> pour le LSTM de production —
        et {mdn['seeds']} seeds différents produisent des résultats mesurablement différents
        avec un code et des données identiques.</p>
    </section>"""


def build_html(summary: dict, fonts_b64: dict) -> str:
    if fonts_b64:
        font_faces = f"""
@font-face {{
  font-family: "Cambria Report"; src: url(data:font/woff;base64,{fonts_b64['cambria_bold']}) format("woff");
  font-weight: 700; font-style: normal; font-display: swap;
}}
@font-face {{
  font-family: "Consolas Report"; src: url(data:font/woff;base64,{fonts_b64['consolas_reg']}) format("woff");
  font-weight: 400; font-style: normal; font-display: swap;
}}
@font-face {{
  font-family: "Consolas Report"; src: url(data:font/woff;base64,{fonts_b64['consolas_bold']}) format("woff");
  font-weight: 700; font-style: normal; font-display: swap;
}}"""
        display_font = '"Cambria Report", Cambria, Georgia, serif'
        mono_font = '"Consolas Report", ui-monospace, Consolas, monospace'
    else:
        font_faces = ""
        display_font = "Cambria, Georgia, serif"
        mono_font = 'ui-monospace, Consolas, "SFMono-Regular", monospace'

    legend_items = "".join(
        f'<span class="legend-item"><span class="swatch" style="--c-light:{VARIANT_COLOR[v][0]};--c-dark:{VARIANT_COLOR[v][1]}"></span>{VARIANT_LABEL[v]}</span>'
        for v in ["normal", "student_t", "ged", "cqr", "native_ged", "native_skewt"])
    verdict_cards = "".join(verdict_card(summary, m) for m in MODEL_ORDER)
    model_sections = "".join(model_section(summary, m) for m in MODEL_ORDER)

    return f"""<title>Calibration des PI — au-delà de la gaussienne</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{font_faces}
:root {{
  color-scheme: light;
  --page: #f7f6f2; --surface: #fcfcfa; --surface-2: #f1efe8; --ink: #14140f;
  --ink-2: #4c4b42; --ink-muted: #86847a; --hairline: #e2e0d6; --baseline: #c7c4b6;
  --accent: #0f5c56; --accent-ink: #0a4440; --accent-wash: #e3efec;
  --good: #0ca30c; --good-ink: #075e07; --warning: #b06a00; --warning-bg: #fbeed6;
  --critical: #b8332f; --critical-bg: #fbe4e2; --good-bg: #dff2df;
  --shadow: 0 1px 2px rgba(20,20,15,0.06), 0 8px 24px -12px rgba(20,20,15,0.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --page: #111110; --surface: #171714; --surface-2: #1e1e1a; --ink: #f3f2ec;
    --ink-2: #c7c5b8; --ink-muted: #8f8d80; --hairline: #2c2c27; --baseline: #3a3a33;
    --accent: #4bb3a8; --accent-ink: #7fd0c6; --accent-wash: #16302c;
    --good: #4ec24e; --good-ink: #7fdc7f; --warning: #e0ab4b; --warning-bg: #3a2e12;
    --critical: #e8746f; --critical-bg: #3a1917; --good-bg: #133014;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #111110; --surface: #171714; --surface-2: #1e1e1a; --ink: #f3f2ec;
  --ink-2: #c7c5b8; --ink-muted: #8f8d80; --hairline: #2c2c27; --baseline: #3a3a33;
  --accent: #4bb3a8; --accent-ink: #7fd0c6; --accent-wash: #16302c;
  --good: #4ec24e; --good-ink: #7fdc7f; --warning: #e0ab4b; --warning-bg: #3a2e12;
  --critical: #e8746f; --critical-bg: #3a1917; --good-bg: #133014;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --page: #f7f6f2; --surface: #fcfcfa; --surface-2: #f1efe8; --ink: #14140f;
  --ink-2: #4c4b42; --ink-muted: #86847a; --hairline: #e2e0d6; --baseline: #c7c4b6;
  --accent: #0f5c56; --accent-ink: #0a4440; --accent-wash: #e3efec;
  --good: #0ca30c; --good-ink: #075e07; --warning: #b06a00; --warning-bg: #fbeed6;
  --critical: #b8332f; --critical-bg: #fbe4e2; --good-bg: #dff2df;
  --shadow: 0 1px 2px rgba(20,20,15,0.06), 0 8px 24px -12px rgba(20,20,15,0.18);
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ background: var(--page); color: var(--ink); font-family: "Segoe UI", system-ui, -apple-system, sans-serif; line-height: 1.55; -webkit-font-smoothing: antialiased; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px 96px; }}
a {{ color: var(--accent-ink); }}
.mono {{ font-family: {mono_font}; font-variant-numeric: tabular-nums; }}
header.hero {{ padding: 56px 24px 40px; border-bottom: 1px solid var(--hairline); background: linear-gradient(180deg, var(--accent-wash), var(--page) 140%); }}
.hero-inner {{ max-width: 1080px; margin: 0 auto; }}
.eyebrow {{ font-family: {mono_font}; text-transform: uppercase; letter-spacing: 0.09em; font-size: 12.5px; color: var(--accent-ink); font-weight: 700; margin: 0 0 14px; }}
h1 {{ font-family: {display_font}; font-weight: 700; font-size: clamp(28px, 4vw, 42px); line-height: 1.12; margin: 0 0 14px; text-wrap: balance; letter-spacing: -0.01em; }}
.hero-sub {{ font-size: 17px; color: var(--ink-2); max-width: 640px; margin: 0 0 22px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ font-family: {mono_font}; font-size: 12.5px; padding: 5px 11px; border-radius: 999px; background: var(--surface); border: 1px solid var(--hairline); color: var(--ink-2); }}
section.block {{ padding: 48px 0 8px; }}
h2 {{ font-family: {display_font}; font-size: 26px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.005em; }}
.section-intro {{ color: var(--ink-2); max-width: 720px; margin: 0 0 28px; }}
.verdict-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 8px; }}
.verdict-card {{ display: block; text-decoration: none; color: inherit; background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px; padding: 18px; box-shadow: var(--shadow); border-left: 4px solid var(--ink-muted); transition: transform 0.12s ease; }}
.verdict-card:hover {{ transform: translateY(-2px); }}
.verdict-card.status-good {{ border-left-color: var(--good); }}
.verdict-card.status-warning {{ border-left-color: var(--warning); }}
.verdict-card.status-critical {{ border-left-color: var(--critical); }}
.verdict-card-top {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 12px; }}
.verdict-model {{ font-weight: 700; font-size: 15.5px; }}
.verdict-metric {{ display: flex; align-items: baseline; gap: 6px; font-family: {mono_font}; }}
.verdict-before {{ color: var(--ink-muted); font-size: 15px; text-decoration: line-through; text-decoration-color: var(--ink-muted); }}
.verdict-arrow {{ color: var(--ink-muted); }}
.verdict-after {{ font-size: 21px; font-weight: 700; }}
.verdict-unit {{ font-size: 11.5px; color: var(--ink-muted); }}
.verdict-delta {{ font-family: {mono_font}; font-size: 13px; margin-top: 4px; font-weight: 700; }}
.delta-good {{ color: var(--good-ink); }}
.delta-bad {{ color: var(--critical); }}
.pill {{ font-family: {mono_font}; font-size: 11px; font-weight: 700; padding: 4px 9px; border-radius: 999px; white-space: nowrap; }}
.pill-sm {{ font-size: 10px; padding: 3px 8px; }}
.pill-good {{ background: var(--good-bg); color: var(--good-ink); }}
.pill-warning {{ background: var(--warning-bg); color: var(--warning); }}
.pill-critical {{ background: var(--critical-bg); color: var(--critical); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px 22px; margin: 4px 0 30px; padding: 14px 18px; background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 13.5px; color: var(--ink-2); }}
.swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; background: var(--c-light); flex: none; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .swatch {{ background: var(--c-dark); }} }}
:root[data-theme="dark"] .swatch {{ background: var(--c-dark); }}
.model-section {{ background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; padding: 26px 26px 20px; margin-bottom: 22px; box-shadow: var(--shadow); scroll-margin-top: 20px; }}
.model-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }}
.model-head h3 {{ font-family: {display_font}; font-size: 21px; margin: 0; }}
.model-verdict-text {{ color: var(--ink-2); max-width: 780px; margin: 0 0 22px; font-size: 15px; }}
.model-body {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 26px; align-items: start; }}
@media (max-width: 760px) {{ .model-body {{ grid-template-columns: 1fr; }} }}
.chart-wrap {{ overflow-x: auto; }}
.chart-subtitle {{ font-size: 13px; font-weight: 700; color: var(--ink-2); margin: 0 0 10px; }}
.chart-subtitle-note {{ font-weight: 400; color: var(--ink-muted); }}
.base-time {{ font-size: 12.5px; color: var(--ink-muted); margin: 14px 0 0; }}
.barchart {{ display: block; }}
.axis-line {{ stroke: var(--baseline); stroke-width: 1; }}
.target-line {{ stroke: var(--ink-muted); stroke-width: 1.5; stroke-dasharray: 3 3; }}
.bar-label {{ font-size: 12px; fill: var(--ink-2); font-family: "Segoe UI", system-ui, sans-serif; }}
.bar-value {{ font-size: 12px; fill: var(--ink); font-family: {mono_font}; font-weight: 700; }}
.bar {{ fill: var(--ink-muted); cursor: default; }}
.bar-normal {{ fill: #2a78d6; }} .bar-student_t {{ fill: #eb6834; }} .bar-ged {{ fill: #1baf7a; }}
.bar-cqr {{ fill: #eda100; }} .bar-native_ged {{ fill: #e87ba4; }} .bar-native_skewt {{ fill: #008300; }} .bar-mdn {{ fill: #4a3aa7; }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .bar-normal {{ fill: #3987e5; }}
  :root:where(:not([data-theme="light"])) .bar-student_t {{ fill: #d95926; }}
  :root:where(:not([data-theme="light"])) .bar-ged {{ fill: #199e70; }}
  :root:where(:not([data-theme="light"])) .bar-cqr {{ fill: #c98500; }}
  :root:where(:not([data-theme="light"])) .bar-native_ged {{ fill: #d55181; }}
  :root:where(:not([data-theme="light"])) .bar-mdn {{ fill: #9085e9; }}
}}
:root[data-theme="dark"] .bar-normal {{ fill: #3987e5; }}
:root[data-theme="dark"] .bar-student_t {{ fill: #d95926; }}
:root[data-theme="dark"] .bar-ged {{ fill: #199e70; }}
:root[data-theme="dark"] .bar-cqr {{ fill: #c98500; }}
:root[data-theme="dark"] .bar-native_ged {{ fill: #d55181; }}
:root[data-theme="dark"] .bar-mdn {{ fill: #9085e9; }}
.range-line-mdn {{ stroke: #4a3aa7; stroke-width: 3; stroke-linecap: round; }}
.range-dot-mdn {{ fill: #4a3aa7; stroke: var(--surface); stroke-width: 2; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .range-line-mdn, :root:where(:not([data-theme="light"])) .range-dot-mdn {{ fill: #9085e9; }} }}
.kpi-table {{ width: 100%; border-collapse: collapse; font-size: 12.8px; }}
.kpi-table th {{ text-align: left; font-weight: 700; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.02em; font-size: 10.5px; padding: 0 8px 8px 0; border-bottom: 1px solid var(--hairline); }}
.kpi-table td {{ padding: 7px 8px 7px 0; border-bottom: 1px solid var(--hairline); font-variant-numeric: tabular-nums; }}
.kpi-table td:first-child {{ display: flex; align-items: center; gap: 7px; white-space: nowrap; }}
.kpi-table tr.row-best {{ background: var(--good-bg); }}
.kpi-table tr:last-child td {{ border-bottom: none; }}
.mdn-section {{ background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; padding: 26px 26px 20px; margin-bottom: 22px; box-shadow: var(--shadow); border-top: 3px solid #4a3aa7; }}
.mdn-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 8px; }}
@media (max-width: 760px) {{ .mdn-grid {{ grid-template-columns: 1fr; }} }}
.takeaways {{ background: var(--accent-wash); border: 1px solid var(--hairline); border-radius: 12px; padding: 26px 28px; }}
.takeaways h2 {{ color: var(--accent-ink); }}
.takeaways ul {{ margin: 14px 0 0; padding-left: 20px; }}
.takeaways li {{ margin-bottom: 10px; color: var(--ink); }}
.takeaways li b {{ color: var(--accent-ink); }}
footer {{ margin-top: 56px; padding-top: 24px; border-top: 1px solid var(--hairline); color: var(--ink-muted); font-size: 13px; }}
footer p {{ max-width: 760px; }}
footer code {{ font-family: {mono_font}; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<header class="hero">
  <div class="hero-inner">
    <p class="eyebrow">DeepEdgeBenchmark — calibration probabiliste</p>
    <h1>Au-delà de la gaussienne : les trois options testées sur les 5 modèles</h1>
    <p class="hero-sub">Extension de l'option 1 (loi alternative) à tous les modèles, plus
      options 2 (CQR) et 3 (MDN, LSTM) — comparaison seule, rien de branché en production.
      Walk-forward existant de chaque modèle réutilisé tel quel ; fenêtre de test
      séparée en calibration (40%, jamais notée) et évaluation (60%, seule fenêtre
      scorée).</p>
    <div class="chips">
      <span class="chip">5 actifs — SPY, BTC, ETH, ZN=F, TLT</span>
      <span class="chip">2020–2024</span>
      <span class="chip">5 modèles</span>
      <span class="chip">option 1 manuelle + native — option 2 CQR — option 3 MDN</span>
    </div>
  </div>
</header>
<div class="wrap">
  <section class="block" id="verdicts">
    <h2>Verdict par modèle</h2>
    <p class="section-intro">Erreur de calibration = moyenne des écarts absolus entre couverture
      mesurée et cible, sur les niveaux 50/80/95%, moyennée sur les 5 actifs. Cliquer une carte
      pour le détail.</p>
    <div class="verdict-grid">{verdict_cards}</div>
  </section>
  <section class="block" id="details">
    <h2>Détail par modèle</h2>
    <p class="section-intro">Barres : erreur de calibration moyenne par option (plus court = mieux
      calibré). Tableau : couverture aux 3 niveaux, largeur relative au cas gaussien, perte
      pinball relative, et surcoût de calcul par rapport au backtest de base.</p>
    <div class="legend">{legend_items}</div>
    {model_sections}
  </section>
  <section class="block" id="mdn">
    <h2>Option 3 — gros plan MDN</h2>
    <p class="section-intro">Seul modèle testé qui demande un vrai réentraînement (pas
      un post-traitement) — architecture identique au LSTM de production, seule la tête de
      sortie change (Dense(1) → mélange de 3 gaussiennes, perte MSE → log-vraisemblance).</p>
    {mdn_section(summary)}
  </section>
  <section class="block takeaways" id="takeaways">
    <h2>Ce qu'il faut retenir</h2>
    <p class="section-intro" style="margin-bottom:18px"><b>Correction du 30 juillet 2026</b> :
      les chiffres de ce rapport utilisent désormais <code>mace_strict</code> (erreur absolue
      moyennée <i>par actif</i>) plutôt que <code>mace_loose</code> (erreur absolue de la
      moyenne inter-actifs) — la version précédente laissait des écarts de signe opposé
      s'annuler entre actifs (BTC sur-couvre, SPY sous-couvre → « calibré » en moyenne sans
      qu'aucun des deux actifs le soit). Repéré par une vérification indépendante, voir
      <code>HANDOFF_sigma_calibration_suivi.md</code>. Les chiffres sont moins spectaculaires
      qu'avant mais plus honnêtes ; deux conclusions s'en trouvent inversées (LSTM, ARIMA-GARCH).</p>
    <ul>
      <li><b>CQR est l'option qui aide le plus souvent</b> — SARIMA, Naive, Prophet, et même
        LSTM (les quatre modèles où le σ ne varie pas dans le temps par construction) :
        -23 à -59% d'erreur de calibration, pour un surcoût de calcul négligeable sur des
        backtests qui prennent déjà des dizaines de secondes.</li>
      <li><b>Le swap de loi seul (Student-t/GED) est plus fragile qu'il n'y paraissait.</b>
        Il aide nettement pour SARIMA, mais pour Naive il fait à peine mieux que ne rien
        faire (GED : -16% contre -59% pour CQR), et pour LSTM il <i>dégrade</i> franchement
        la calibration (jusqu'à +14%) — le σ de LSTM est une seule valeur figée sur les résidus
        d'entraînement, pas un chemin qui varie dans le temps ; lui changer la forme de queue ne
        corrige pas un problème de niveau.</li>
      <li><b>Il ne corrige rien quand le problème est ailleurs.</b> Prophet sous-couvre
        massivement à <i>tous</i> les niveaux — changer la forme de la queue autour d'un
        σ déjà trop petit ne le rend pas plus grand ; même CQR (-26%) ne suffit pas à le
        rendre correctement calibré.</li>
      <li><b>Pour ARIMA-GARCH, c'est l'inverse : le refit natif l'emporte pour de bon</b> sur
        le swap manuel gratuit (2,51 contre 3,27 points) — et CQR, ici seulement, fait moins
        bien que les corrections de forme. Logique : c'est le seul modèle dont le σ est déjà
        dynamique par construction (GARCH), donc le vrai problème restant est bien la forme de
        la queue, pas son niveau — l'inverse du diagnostic pour les 4 autres modèles.</li>
      <li><b>Le MDN (option 3) n'est pas rentable en l'état</b> — plus cher, moins bien
        calibré en moyenne que le ruban ±1,96·σ actuel, et instable d'un entraînement à
        l'autre à seed égal. Le vrai problème de LSTM (σ figé dans le temps) reste à
        résoudre, mais pas par cette voie telle qu'implémentée ici.</li>
      <li><b>Attention, tout ce qui précède reste sur une seule fenêtre</b> (2020–2024).
        Un travail de suivi (<code>HANDOFF_sigma_calibration_suivi.md</code>) a testé les mêmes
        options sur 3 fenêtres temporelles et montré que les gagnants statiques ci-dessus
        (CQR pour SARIMA/Naive) ne généralisent pas d'une fenêtre à l'autre — seul ARIMA-GARCH
        natif reste robuste. La correction qui tient sur les 3 fenêtres pour les 4 autres
        modèles est un σ dynamique par EWMA causale, pas un choix de loi statique : à lire
        avant d'adopter quoi que ce soit en dur sur la base de ce rapport seul.</li>
    </ul>
  </section>
  <footer>
    <p>Méthodologie complète dans les docstrings de
      <code>experiments/dist_options_common.py</code>,
      <code>experiments/all_models_dist_options.py</code> et
      <code>experiments/lstm_mdn_prototype.py</code>. Données brutes :
      <code>experiments/all_models_dist_options_results.json</code>,
      <code>experiments/lstm_mdn_results.json</code>,
      <code>experiments/dist_options_summary.json</code>. Statut d'exécution et reprise :
      <code>HANDOFF_dist_options_comparison.md</code>. σ estimé depuis la largeur du PI 95%
      existant de chaque modèle (même convention que
      <code>generate_distributions_dashboard.py</code>), pas re-dérivé en interne —
      approximation assumée pour SARIMA/Prophet/LSTM/Naive, exacte pour ARIMA-GARCH natif.</p>
  </footer>
</div>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "experiments" / "dist_options_report.html"))
    p.add_argument("--summary", default=str(SUMMARY_JSON))
    args = p.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fonts_b64 = subset_fonts_b64(Path(tmp))
        html = build_html(summary, fonts_b64)

    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Saved -> {args.out}  ({len(html)/1024:.0f} KB) "
         f"[fonts embedded: {'yes' if fonts_b64 else 'no, fallback stacks'}]")


if __name__ == "__main__":
    main()
