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
)

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
    "fenêtre d'évaluation (60&nbsp;%, seule fenêtre servant au calcul des KPIs). "
    "<b>Erreur de calibration</b> (MACE stricte) = moyenne, sur les 5 actifs, des écarts "
    "absolus entre couverture mesurée et cible aux niveaux 50/80/95&nbsp;% — 0 = "
    "parfaitement calibré, sans possibilité de compensation entre actifs.",
    body_style))

# ---- 2. Verdict table ----
story.append(p("2&nbsp;&nbsp; Verdict par modèle", h1_style))

header = ["Modèle", "Erreur avant", "Meilleure option", "Erreur après", "Δ", "Surcoût", "Verdict"]
rows = [header]
for model, (best_key, best_label, verdict) in VERDICTS.items():
    m = SUMMARY["models"][model]
    before = m["normal"]["mace_strict"]
    after = m[best_key]["mace_strict"]
    delta_pct = (after - before) / before * 100
    oh = m[best_key]["overhead_s_mean"]
    base_t = m["base_train_time_s_mean"]
    oh_txt = f"+{oh:.3f}s" if oh < 1 else f"+{oh:.1f}s"
    rows.append([model, fmt(before, 2), best_label, fmt(after, 2),
                f"{delta_pct:+.0f}%", f"{oh_txt} / {fmt(base_t,0)}s", verdict])

col_widths = [2.3*cm, 1.9*cm, 2.9*cm, 1.9*cm, 1.5*cm, 2.6*cm, 4.1*cm]
tbl = Table(rows, colWidths=col_widths, repeatRows=1)
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
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

# ---- 3. Interpretation ----
story.append(p("3&nbsp;&nbsp; Ce qu'il faut retenir", h1_style))

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
