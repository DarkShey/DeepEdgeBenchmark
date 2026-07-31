# -*- coding: utf-8 -*-
"""
generate_dist_options_pdf.py — PDF summary of the all-models PI-calibration
options comparison (1/2/3), reportlab Times-Roman house style (matches
comparaison_lois_garch_resultats.pdf). Unlike generate_dist_options_report.py
(HTML), this has no font-licensing concern (reportlab's base14 Times-Roman
needs no embedding) and is deliberately what gets shared/committed -- colleagues
without access to a privately-hosted page can open a PDF directly.

Verdict text/numbers below use `mace_strict` (mean absolute calibration error,
averaged PER ASSET then absolute-valued) from experiments/dist_options_summary.json
-- NOT `mace_loose`, which lets opposite-signed per-asset errors cancel and
flatters every option (see HANDOFF_sigma_calibration_suivi.md, the independent
robustness re-run that flagged this). Kept in sync with
experiments/generate_dist_options_report.py's VERDICTS dict.

Usage (from repo root):
    python experiments/generate_dist_options_pdf.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = json.loads((ROOT / "experiments" / "dist_options_summary.json").read_text())
OUT_PDF = ROOT / "comparaison_options_calibration_tous_modeles.pdf"

title_style = ParagraphStyle("Title", fontName="Times-Roman", fontSize=15,
                             leading=18, alignment=TA_CENTER, spaceAfter=2)
subtitle_style = ParagraphStyle("Subtitle", fontName="Times-Roman", fontSize=11,
                                leading=14, alignment=TA_CENTER, spaceAfter=2)
date_style = ParagraphStyle("Date", fontName="Times-Roman", fontSize=10,
                            leading=13, alignment=TA_CENTER, spaceBefore=10, spaceAfter=18)
h1_style = ParagraphStyle("H1", fontName="Times-Bold", fontSize=12.5, leading=15,
                          spaceBefore=14, spaceAfter=6)
body_style = ParagraphStyle("Body", fontName="Times-Roman", fontSize=10.2,
                            leading=14.3, alignment=TA_JUSTIFY, spaceAfter=8)
callout_style = ParagraphStyle("Callout", fontName="Times-Italic", fontSize=9.8,
                               leading=13.6, alignment=TA_JUSTIFY, spaceAfter=10,
                               leftIndent=10, rightIndent=10)
caption_style = ParagraphStyle("Caption", fontName="Times-Roman", fontSize=8.6,
                               leading=11.5, alignment=TA_CENTER, spaceBefore=4, spaceAfter=14)
li_style = ParagraphStyle("Li", fontName="Times-Roman", fontSize=10.2, leading=14.2,
                          alignment=TA_JUSTIFY, spaceAfter=9, leftIndent=12,
                          bulletIndent=0, bulletFontName="Times-Bold")
h2_style = ParagraphStyle("H2", fontName="Times-Bold", fontSize=11.3, leading=14,
                          spaceBefore=12, spaceAfter=5)
model_verdict_style = ParagraphStyle("ModelVerdict", fontName="Times-Roman", fontSize=9.4,
                                     leading=13, alignment=TA_JUSTIFY, spaceAfter=6)
mono_caption_style = ParagraphStyle("MonoCaption", fontName="Times-Roman", fontSize=8.2,
                                    leading=10.6, alignment=TA_JUSTIFY, spaceBefore=4,
                                    spaceAfter=12, textColor=colors.grey)


def p(text, style=body_style):
    return Paragraph(text, style)


def fmt(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


# Best option per model + French verdict labels for the table -- kept in sync
# with generate_dist_options_report.py's VERDICTS dict (best/headline).
VERDICTS = {
    "SARIMA":      ("cqr",          "CQR",                "Vaut le coût — CQR nettement devant"),
    "Prophet":     ("cqr",          "CQR",                "Aucune option testée ne suffit"),
    "LSTM":        ("cqr",          "CQR",                "Aide un peu, ne règle pas le vrai problème"),
    "Naive":       ("cqr",          "CQR",                "Vaut le coût — CQR nettement devant"),
    "ARIMA-GARCH": ("native_ged",   "GED natif (refit)",  "Le refit natif l'emporte pour de bon"),
}

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
    "normal": "#2a78d6", "student_t": "#eb6834", "ged": "#1baf7a",
    "cqr": "#eda100", "native_ged": "#e87ba4", "native_skewt": "#008300",
}
VARIANT_ORDER = ["normal", "student_t", "ged", "cqr", "native_ged", "native_skewt"]


AXIS_LABEL_H = 13  # extra strip at the bottom of every bar chart carrying the axis caption


def bar_chart_drawing(rows, max_val, width=460, bar_h=13, gap=9, left_pad=132,
                      axis_label="Erreur de calibration — points de % d'écart à la "
                                 "couverture visée (0 = parfait, plus court = mieux)"):
    """rows: [(label, value, color_hex), ...]. Horizontal bars, value at tip --
    same layout logic as the SVG version in generate_dist_options_report.py.
    Carries its own axis caption (baked into the image, not just surrounding
    prose) so the chart is self-explanatory even skimmed on its own."""
    row_h = bar_h + gap
    height = len(rows) * row_h + 6 + AXIS_LABEL_H
    d = Drawing(width, height)
    plot_w = width - left_pad - 46
    d.add(Line(left_pad, AXIS_LABEL_H, left_pad, height - 2,
               strokeColor=colors.HexColor("#c7c4b6"), strokeWidth=0.6))
    if axis_label:
        d.add(String(left_pad, 2, axis_label, fontName="Times-Italic", fontSize=6.6,
                     fillColor=colors.HexColor("#86847a")))
    for i, (label, value, color_hex) in enumerate(rows):
        y = height - 4 - (i + 1) * row_h + gap / 2
        bw = max(2, (value / max_val) * plot_w) if max_val > 0 else 2
        d.add(String(left_pad - 6, y + bar_h / 2 - 3, label, fontName="Times-Roman",
                     fontSize=7.6, textAnchor="end", fillColor=colors.HexColor("#4c4b42")))
        d.add(Rect(left_pad, y, bw, bar_h, fillColor=colors.HexColor(color_hex),
                   strokeColor=None))
        d.add(String(left_pad + bw + 5, y + bar_h / 2 - 3, fmt(value), fontName="Times-Bold",
                     fontSize=7.6, fillColor=colors.HexColor("#14140f")))
    return d


def kpi_table(model_key, variants):
    m = SUMMARY["models"][model_key]
    best_key = VERDICTS[model_key][0]
    head = ["Option", "Cov 50%", "Cov 80%", "Cov 95%", "Larg. rel.", "Pinball rel.", "Surcoût"]
    data = [head]
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_i, v in enumerate(variants, start=1):
        d = m[v]
        cov = d["cov_mean"]
        oh = d["overhead_s_mean"]
        oh_txt = f"+{fmt(oh,3)}s" if oh < 1 else f"+{fmt(oh,1)}s"
        data.append([VARIANT_LABEL[v], f"{fmt(cov['50'],1)}%", f"{fmt(cov['80'],1)}%",
                    f"{fmt(cov['95'],1)}%", f"{fmt(d['rel_width_mean']['50'],2)}×",
                    f"{fmt(d['rel_pinball_mean'],3)}×", oh_txt])
        if v == best_key:
            style_cmds.append(("BACKGROUND", (0, row_i), (-1, row_i), colors.HexColor("#dff2df")))
    col_widths = [3.6*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.7*cm, 1.9*cm, 1.6*cm]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds))
    return t


def model_section(model_key):
    m = SUMMARY["models"][model_key]
    variants = ["normal"] + [v for v in VARIANT_ORDER if v in m and v != "normal"]
    rows = [(VARIANT_LABEL[v], m[v]["mace_strict"], VARIANT_COLOR[v]) for v in variants]
    max_val = max(r[1] for r in rows) * 1.15
    best_key, best_label, headline = VERDICTS[model_key]

    flow = [
        p(f"{model_key} — <i>{headline}</i>", h2_style),
        bar_chart_drawing(rows, max_val),
        Spacer(1, 4),
        kpi_table(model_key, variants),
        p(f"Backtest de base : {fmt(m['base_train_time_s_mean'],1)}s par actif, "
         "avant tout surcoût d'option.", mono_caption_style),
    ]
    return flow


def mdn_section():
    mdn = SUMMARY["mdn"]
    base = SUMMARY["lstm_baseline_for_mdn_comparison"]
    rows = [("LSTM (production)", base["mace_strict"], VARIANT_COLOR["normal"]),
           ("LSTM-MDN (K=3, moy. 3 seeds)", mdn["mace_strict"], "#4a3aa7")]
    max_val = max(r[1] for r in rows) * 1.15

    head = ["Actif", "Cov 50% (moy. ± écart-type)", "Cov 95% (moy.)", "CRPS (moy.)",
           "Temps d'entraînement"]
    data = [head]
    for asset, pa in mdn["per_asset"].items():
        data.append([asset, f"{fmt(pa['cov_50_mean'],1)}% ±{fmt(pa['cov_50_std'],1)}",
                    f"{fmt(pa['cov_95_mean'],1)}%", fmt(pa['crps_mean'], 2),
                    f"{fmt(pa['train_time_s_mean'],1)}s"])
    col_widths = [2.2*cm, 4.2*cm, 2.3*cm, 2.3*cm, 3.0*cm]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    return [
        p("Option 3 — gros plan MDN (LSTM)", h2_style),
        p("Seul modèle testé qui demande un vrai réentraînement (pas un post-traitement) — "
         "architecture identique au LSTM de production, seule la tête de sortie change "
         "(Dense(1) → mélange de 3 gaussiennes, perte MSE → log-vraisemblance).",
         model_verdict_style),
        bar_chart_drawing(rows, max_val),
        Spacer(1, 4),
        t,
        p("Couverture à 50/95% et CRPS moyennés sur 3 seeds par actif — l'écart-type sur la "
         "couverture à 50% est l'instabilité d'un entraînement à l'autre, code et données "
         f"identiques (seeds {mdn['seeds']}). Coût moyen {fmt(mdn['mean_train_time_s'],1)}s "
         f"contre {fmt(base['train_time_s'],1)}s pour le LSTM de production.",
         mono_caption_style),
    ]

story = []

# ---- Title ----
story.append(p("Comparaison des options de calibration des PI — tous modèles",
              title_style))
story.append(p("DeepEdgeBenchmark — option 1 (loi alternative) étendue aux 5 modèles, "
              "options 2 (CQR) et 3 (MDN) testées en comparaison", subtitle_style))
story.append(p("Résultats au 29 juillet 2026 — corrigé le 30 juillet 2026", date_style))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceAfter=14))

# ---- Correction callout ----
story.append(Paragraph(
    "<b>Correction du 30 juillet 2026</b> : les chiffres ci-dessous utilisent désormais "
    "<font face=\"Courier\">mace_strict</font> (erreur de calibration moyennée "
    "<i>par actif</i>, puis mise en valeur absolue) plutôt que "
    "<font face=\"Courier\">mace_loose</font> (valeur absolue de la moyenne inter-actifs). "
    "La version loose laissait des écarts de signe opposé s'annuler entre actifs "
    "(BTC sur-couvre, SPY sous-couvre → « calibré » en moyenne sans qu'aucun "
    "des deux actifs le soit) — repéré par une vérification indépendante "
    "(voir <font face=\"Courier\">HANDOFF_sigma_calibration_suivi.md</font>). Les chiffres sont "
    "moins spectaculaires qu'hier mais honnêtes ; deux verdicts s'en trouvent inversés "
    "(LSTM, ARIMA-GARCH — voir §3).",
    callout_style))

# ---- 1. Method ----
story.append(p("1&nbsp;&nbsp; Méthode", h1_style))
story.append(p(
    "Chaque modèle est réévalué via son propre backtest walk-forward existant, "
    "non modifié. La fenêtre de test (2020–2024, 15&nbsp;% en test) est séparée "
    "chronologiquement en une fenêtre de calibration (40&nbsp;%, jamais notée) et une "
    "fenêtre d'évaluation (60&nbsp;%, seule fenêtre servant au calcul des KPIs).",
    body_style))
story.append(Paragraph(
    "<b>Qu'est-ce que « l'erreur » du tableau et des graphiques&nbsp;?</b> Chaque prévision "
    "vient avec un intervalle de confiance à 3 niveaux (50&nbsp;%, 80&nbsp;%, 95&nbsp;%) — "
    "censé contenir le vrai prix 50/80/95&nbsp;% du temps. On mesure combien de fois, en "
    "pratique, le vrai prix est tombé dans cet intervalle (la <i>couverture</i>), et on "
    "compare à la cible. <b>« Erreur de calibration »</b> = l'écart moyen, en points de "
    "pourcentage, entre couverture mesurée et couverture visée, aux 3 niveaux et sur les "
    "5 actifs — 0 = l'intervalle tient exactement sa promesse. <i>Exemple</i>&nbsp;: une "
    "erreur de 8 points veut dire qu'en moyenne l'intervalle censé être juste 1 fois sur 2 "
    "(50&nbsp;%) l'est en réalité 58&nbsp;% ou 42&nbsp;% du temps — trop large ou trop "
    "étroit, dans un sens ou dans l'autre. Ce n'est <u>pas</u> une erreur de prédiction de "
    "prix (RMSE) — un modèle peut prédire le prix tout aussi bien avant et après correction, "
    "seule la fiabilité de son intervalle change (cf. « Pinball rel. » dans les tableaux "
    "§3, qui lui reflète la qualité de l'intervalle dans son ensemble, largeur et "
    "position).",
    callout_style))

# ---- 2. Verdict table ----
story.append(p("2&nbsp;&nbsp; Verdict par modèle", h1_style))

header = ["Modèle", "Erreur calib. avant", "Meilleure option", "Erreur calib. après",
         "Δ", "Surcoût", "Verdict"]
rows = [header]
for model, (best_key, best_label, verdict) in VERDICTS.items():
    m = SUMMARY["models"][model]
    before = m["normal"]["mace_strict"]
    after = m[best_key]["mace_strict"]
    delta_pct = (after - before) / before * 100
    oh = m[best_key]["overhead_s_mean"]
    base_t = m["base_train_time_s_mean"]
    oh_txt = f"+{fmt(oh,3)}s" if oh < 1 else f"+{fmt(oh,1)}s"
    rows.append([model, fmt(before, 2), best_label, fmt(after, 2),
                f"{delta_pct:+.0f}%", f"{oh_txt} / {fmt(base_t,0)}s", verdict])

col_widths = [2.3*cm, 2.3*cm, 2.7*cm, 2.3*cm, 1.3*cm, 2.4*cm, 3.7*cm]
tbl = Table(rows, colWidths=col_widths, repeatRows=1)
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
    ("ALIGN", (1, 0), (4, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tbl)
story.append(p(
    "Erreur de calibration en points de pourcentage (MACE stricte, 0 = parfait). Surcoût : "
    "coût de l'option / coût du backtest de base du modèle (mêmes unités, "
    "secondes). Détail complet (4 à 6 options par modèle, largeur, pinball, CRPS) dans "
    "experiments/dist_options_summary.json.",
    caption_style))

# ---- 3. Detail per model ----
story.append(p("3&nbsp;&nbsp; Détail par modèle", h1_style))
story.append(p(
    "Barres : erreur de calibration stricte par option (plus court = mieux calibré). "
    "Tableau : couverture aux 3 niveaux, largeur relative au cas gaussien, perte pinball "
    "relative, et surcoût de calcul par rapport au backtest de base.",
    body_style))
for model_key in MODEL_ORDER:
    story.append(KeepTogether(model_section(model_key)))

# ---- 4. MDN spotlight ----
story.append(p("4&nbsp;&nbsp; Option 3 — gros plan MDN", h1_style))
story.append(KeepTogether(mdn_section()))

# ---- 5. Interpretation ----
story.append(p("5&nbsp;&nbsp; Ce qu'il faut retenir", h1_style))

sarima = SUMMARY["models"]["SARIMA"]
naive = SUMMARY["models"]["Naive"]
lstm = SUMMARY["models"]["LSTM"]
prophet = SUMMARY["models"]["Prophet"]
arima = SUMMARY["models"]["ARIMA-GARCH"]
mdn = SUMMARY["mdn"]
lstm_base = SUMMARY["lstm_baseline_for_mdn_comparison"]

takeaways = [
    f"<b>CQR est l'option qui aide le plus souvent</b> — SARIMA "
    f"({fmt(sarima['normal']['mace_strict'])}→{fmt(sarima['cqr']['mace_strict'])}), Naive "
    f"({fmt(naive['normal']['mace_strict'])}→{fmt(naive['cqr']['mace_strict'])}), Prophet, et "
    f"même LSTM ({fmt(lstm['normal']['mace_strict'])}→{fmt(lstm['cqr']['mace_strict'])}) "
    "— les quatre modèles dont le σ ne varie pas dans le temps par construction. "
    "-23 à -59&nbsp;% d'erreur de calibration pour un surcoût de calcul négligeable.",

    f"<b>Le swap de loi seul (Student-t/GED) est plus fragile qu'il n'y paraissait sous "
    "l'ancienne agrégation.</b> Net pour SARIMA, mais pour Naive GED fait à peine mieux "
    f"que ne rien faire ({fmt(naive['ged']['mace_strict'])} contre "
    f"{fmt(naive['normal']['mace_strict'])} sans rien faire, contre "
    f"{fmt(naive['cqr']['mace_strict'])} pour CQR) ; pour LSTM il <i>dégrade</i> franchement "
    f"la calibration (jusqu'à {fmt(lstm['ged']['mace_strict'])} contre "
    f"{fmt(lstm['normal']['mace_strict'])} sans rien faire) — le σ de LSTM est une seule "
    "valeur figée sur les résidus d'entraînement, pas un chemin qui varie dans le "
    "temps ; lui changer la forme de queue ne corrige pas un problème de niveau.",

    f"<b>Ça ne corrige rien quand le problème est ailleurs.</b> Prophet sous-couvre "
    "massivement à <i>tous</i> les niveaux (erreur "
    f"{fmt(prophet['normal']['mace_strict'])} points) — changer la forme de la queue autour "
    f"d'un σ déjà trop petit ne le rend pas plus grand ; même CQR "
    f"({fmt(prophet['cqr']['mace_strict'])}) ne suffit pas à le rendre correctement "
    "calibré.",

    f"<b>Pour ARIMA-GARCH, c'est l'inverse : le refit natif l'emporte pour de bon</b> sur le "
    f"swap manuel gratuit ({fmt(arima['native_ged']['mace_strict'])} contre "
    f"{fmt(arima['student_t']['mace_strict'])} points) — et CQR, ici seulement, fait moins "
    f"bien ({fmt(arima['cqr']['mace_strict'])}) que les corrections de forme. Logique : c'est le "
    "seul modèle dont le σ est déjà dynamique par construction (GARCH), donc le "
    "vrai problème restant est la forme de la queue, pas son niveau — l'inverse du "
    "diagnostic pour les 4 autres modèles.",

    f"<b>Le MDN (option 3) n'est pas rentable en l'état.</b> Erreur moyenne "
    f"{fmt(mdn['mace_strict'])} contre {fmt(lstm_base['mace_strict'])} pour le LSTM de "
    f"production (pire, +{fmt((mdn['mace_strict']-lstm_base['mace_strict'])/lstm_base['mace_strict']*100,0)}"
    "&nbsp;%), pour +23&nbsp;% de temps d'entraînement, et une instabilité réelle "
    "d'un entraînement à l'autre à seed égal (couverture à 50&nbsp;% sur "
    "SPY : 66&nbsp;% puis 14&nbsp;% entre deux runs identiques avant stabilisation).",

    "<b>Tout ce qui précède reste sur une seule fenêtre</b> (2020–2024). Un "
    "travail de suivi (<font face=\"Courier\">HANDOFF_sigma_calibration_suivi.md</font>) a "
    "testé les mêmes options sur 3 fenêtres temporelles et montré que les "
    "gagnants statiques ci-dessus (CQR pour SARIMA/Naive) ne généralisent pas d'une "
    "fenêtre à l'autre — seul ARIMA-GARCH natif reste robuste. La correction qui "
    "tient sur les 3 fenêtres pour les 4 autres modèles est un σ dynamique par EWMA "
    "causale, pas un choix de loi statique — à lire avant d'adopter quoi que ce soit en "
    "dur sur la base de ce document seul.",
]
for t in takeaways:
    story.append(Paragraph(f"•&nbsp;&nbsp;{t}", li_style))

story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.4, color=colors.grey))
story.append(p(
    "Code&nbsp;: <font face=\"Courier\">experiments/dist_options_common.py</font>, "
    "<font face=\"Courier\">all_models_dist_options.py</font>, "
    "<font face=\"Courier\">lstm_mdn_prototype.py</font>, "
    "<font face=\"Courier\">summarize_dist_options.py</font> &nbsp;·&nbsp; données&nbsp;: "
    "<font face=\"Courier\">experiments/dist_options_summary.json</font> &nbsp;·&nbsp; "
    "suivi robustesse/EWMA&nbsp;: <font face=\"Courier\">HANDOFF_sigma_calibration_suivi.md</font>.",
    ParagraphStyle("Footer", fontName="Times-Roman", fontSize=8.2, leading=11,
                  alignment=TA_CENTER, spaceBefore=8, textColor=colors.grey)))

doc = SimpleDocTemplate(str(OUT_PDF), pagesize=LETTER,
                        topMargin=2.1*cm, bottomMargin=2.1*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm)
doc.build(story)
print(f"Saved -> {OUT_PDF}")
