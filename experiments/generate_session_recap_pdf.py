# -*- coding: utf-8 -*-
"""
generate_session_recap_pdf.py — session recap PDF (31 juillet 2026), same
reportlab Times-Roman house style as comparaison_options_calibration_tous_modeles.pdf
and comparaison_lois_garch_resultats.pdf.

The Prophet 5-asset table (§3) is pulled live from
experiments/prophet_sigma_investigation.json so its numbers can't drift from
the actual data; the rest is a hand-authored narrative recap of the session
(not derived from a single JSON, this one is genuinely "what happened today").

Usage (from repo root):
    python experiments/generate_session_recap_pdf.py
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
PROPHET_JSON = ROOT / "experiments" / "prophet_sigma_investigation.json"
OUT_PDF = ROOT / "documentation" / "Recap_Session_20260731.pdf"

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
caption_style = ParagraphStyle("Caption", fontName="Times-Roman", fontSize=8.6,
                               leading=11.5, alignment=TA_CENTER, spaceBefore=4, spaceAfter=14)
li_style = ParagraphStyle("Li", fontName="Times-Roman", fontSize=10.2, leading=14.2,
                          alignment=TA_JUSTIFY, spaceAfter=8, leftIndent=12,
                          bulletIndent=0, bulletFontName="Times-Bold")
li_sub_style = ParagraphStyle("LiSub", fontName="Times-Roman", fontSize=9.6, leading=13.2,
                              alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=24)


def p(text, style=body_style):
    return Paragraph(text, style)


def li(text, style=li_style):
    return Paragraph(f"•&nbsp;&nbsp;{text}", style)


def fmt(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


story = []

# ---- Title ----
story.append(p("Récapitulatif de session", title_style))
story.append(p("DeepEdgeBenchmark — suivi de la calibration des PI, coordination avec "
              "le travail parallèle du collègue", subtitle_style))
story.append(p("31 juillet 2026", date_style))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceAfter=14))

# ---- 1. What happened today ----
story.append(p("1&nbsp;&nbsp; Ce qui a été fait aujourd'hui", h1_style))
story.append(p(
    "Suite directe d'hier (comparaison des options 1/2/3 de calibration, "
    "<i>HANDOFF_dist_options_comparison.md</i>) et du travail parallèle du collègue, "
    "poussé pendant la nuit (adoption d'un σ dynamique EWMA en production pour les 5 "
    "modèles, <i>HANDOFF_sigma_calibration_suivi.md</i>). La session d'aujourd'hui a "
    "porté sur la vérification de ce travail, la correction d'un biais méthodologique "
    "trouvé dans mon propre script, et la reprise de deux fils laissés en suspens.",
    body_style))
story.append(li("<b>Vérifié et intégré le travail du collègue</b> — reclassement CRPS "
               "sous la meilleure option de calibration (aucun changement de rang), "
               "puis l'adoption complète du σ EWMA en production."))
story.append(li("<b>Corrigé un bug d'agrégation dans <font face=\"Courier\">"
               "summarize_dist_options.py</font></b> — <font face=\"Courier\">mace_loose</font> "
               "laissait des écarts de signe opposé s'annuler entre actifs (repéré "
               "indépendamment par le collègue) ; ajout de <font face=\"Courier\">"
               "mace_strict</font>, la version honnête. Deux verdicts s'en trouvent "
               "inversés (LSTM, ARIMA-GARCH)."))
story.append(li("<b>Régénéré et clarifié <font face=\"Courier\">"
               "comparaison_options_calibration_tous_modeles.pdf</font></b> avec les "
               "chiffres corrigés, plus une explication en langage clair de ce que "
               "« erreur de calibration » représente (exemple concret à l'appui)."))
story.append(li("<b>Laissé une question écrite pour le collègue</b> sur "
               "<font face=\"Courier\">calibration/pi_recalibration.py</font> — un "
               "script orphelin qui recoupe fortement l'adoption EWMA, statut à "
               "clarifier avant d'y toucher (<font face=\"Courier\">"
               "NOTE_question_pi_recalibration.md</font>, pas encore de réponse)."))
story.append(li("<b>Repris et terminé le fix Prophet D+7/hebdomadaire</b> — le "
               "fit en log-prix était déjà commencé localement (non commité, pas de "
               "moi) ; ajout du paramètre <font face=\"Courier\">sigma_scale</font> "
               "(même mécanisme que <font face=\"Courier\">next_step_prophet</font> déjà "
               "en prod) et de tests dédiés — aucun n'existait avant pour ce module."))
story.append(li("<b>Étendu l'investigation Prophet du collègue aux 5 actifs</b> "
               "(SPY/BTC seulement avant) — résultats détaillés en §2, nuancent la "
               "conclusion « Prophet résolu » d'hier."))

# ---- 2. Prophet 5-asset results ----
story.append(p("2&nbsp;&nbsp; Résultats — extension de l'investigation Prophet aux 5 actifs", h1_style))
story.append(p(
    "<font face=\"Courier\">experiments/prophet_sigma_investigation.py</font> (script du "
    "collègue) teste 6 configurations candidates pour corriger le sous-couvrement massif "
    "de Prophet, sur la fenêtre W1 (2020–2024). Jusqu'ici testé sur SPY/BTC seulement — "
    "étendu ici à ETH, ZN=F et TLT. Indicateur : erreur de calibration stricte (MACE), "
    "moyenne des écarts absolus à 50/80/95&nbsp;% — 0 = parfait, mêmes conventions que le "
    "document de comparaison principal.",
    body_style))

data = json.loads(PROPHET_JSON.read_text())
assets = data["config"]["sweep_assets"]
configs = list(data["config"]["configs"])
CONFIG_LABEL = {"base": "base (référence)", "cps_low": "cps_low", "cps_high": "cps_high",
               "no_seas": "no_seas", "flat": "flat", "log": "log (adopté en prod)"}
targets = {"cov_50": 50, "cov_80": 80, "cov_95": 95}

header = ["Configuration"] + assets + ["Moyenne"]
rows = [header]
means = {}
for cfg in configs:
    row_vals = []
    for asset in assets:
        kpi = data["phaseB"][asset][cfg]
        mace = sum(abs(kpi[k] - t) for k, t in targets.items()) / 3
        row_vals.append(mace)
    mean_v = sum(row_vals) / len(row_vals)
    means[cfg] = mean_v
    rows.append([CONFIG_LABEL[cfg]] + [fmt(v) for v in row_vals] + [fmt(mean_v)])

best_cfg = min(means, key=means.get)
col_widths = [3.6*cm] + [1.85*cm] * len(assets) + [1.9*cm]
tbl = Table(rows, colWidths=col_widths, repeatRows=1)
style_cmds = [
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]
for i, cfg in enumerate(configs, start=1):
    if cfg == best_cfg:
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#dff2df")))
tbl.setStyle(TableStyle(style_cmds))
story.append(tbl)
story.append(p(
    "MACE stricte (points de %, 0 = parfait) par configuration et par actif, sur la "
    "fenêtre W1. Ligne en vert = meilleure moyenne sur les 5 actifs.",
    caption_style))

story.append(p(
    "<b>Le fix adopté (log-espace) reste le meilleur en moyenne (19,7 contre 28,8 pour "
    "la référence non corrigée) — mais il n'aide pas uniformément.</b> Excellent sur "
    "SPY (12,8→4,8) et BTC (29,3→6,5), correct sur ETH (34,4→15,6), il ne bouge quasiment "
    "pas ZN (40,5→35,8) et fait <i>pire que ne rien faire</i> sur TLT (27,2→35,8). Le "
    "défaut de production (<font face=\"Courier\">LOG_SPACE=True</font> en dur pour tous "
    "les actifs, <font face=\"Courier\">models/prophet_model.py</font>) a été validé "
    "« sur 3 fenêtres × SPY/BTC » selon son propre commentaire — cette limite de "
    "périmètre était donc réelle, pas juste une formalité. Logique après coup&nbsp;: le "
    "log-prix corrige un problème d'extrapolation agressive du trend sur des séries en "
    "forte croissance non stationnaire — SPY/BTC/ETH sont exactement ce cas ; ZN et TLT "
    "(taux/obligataire) ont une dynamique de prix très différente, où cette correction "
    "n'a pas de raison de s'appliquer de la même façon.",
    body_style))
story.append(p(
    "<b>Point important non tranché ici</b> : ce balayage teste le log-espace "
    "<i>seul</i>, pas la combinaison complète adoptée en production "
    "(log + EWMA — la correction EWMA de la phase C de ce script n'est qu'un facteur "
    "d'échelle statique appliqué à la configuration <font face=\"Courier\">base</font> "
    "non corrigée, pas le mécanisme EWMA réellement adopté). Il est possible que la "
    "couche EWMA, dynamique, rattrape une partie du problème sur ZN/TLT là où le "
    "log seul ne suffit pas — mais ce test précis (log+EWMA sur les 5 actifs) n'a pas "
    "encore été fait.",
    body_style))

# ---- 3. Updated conclusions ----
story.append(p("3&nbsp;&nbsp; Conclusions mises à jour", h1_style))
story.append(li("<b>Le diagnostic général tient toujours</b> : le problème dominant est "
               "presque partout le <i>niveau</i> de σ qui doit suivre les régimes de "
               "volatilité, pas la forme de la loi — sauf pour ARIMA-GARCH, seul modèle "
               "dont le σ était déjà dynamique par construction."))
story.append(li("<b>« Prophet résolu » doit être nuancé.</b> Résolu pour les actifs à "
               "forte croissance non stationnaire (SPY, BTC, ETH) — pas démontré pour "
               "les actifs obligataires (ZN, TLT), où le fix adopté globalement peut "
               "activement dégrader la calibration. À vérifier avec log+EWMA combiné "
               "avant de considérer le sujet clos sur ces deux actifs — potentiellement "
               "besoin d'un traitement différent par classe d'actif plutôt qu'un défaut "
               "unique pour tout Prophet."))
story.append(li("<b>LSTM reste résolu par l'EWMA seule</b> (pas de réseau à mélange de "
               "gaussiennes nécessaire) — confirme que l'abandon du MDN (plus cher, "
               "moins bon, instable) était la bonne décision."))
story.append(li("<b>Calibration et classement global des modèles restent deux axes "
               "indépendants</b> — confirmé par le reclassement CRPS du collègue, "
               "aucun changement de rang malgré des gains de calibration nets."))

# ---- 4. Exploitable now ----
story.append(p("4&nbsp;&nbsp; Ce qui est exploitable maintenant", h1_style))
story.append(li("<font face=\"Courier\">comparaison_options_calibration_tous_modeles.pdf</font> "
               "— chiffres corrects, à jour, partageable."))
story.append(li("Défauts de calibration EWMA/natifs adoptés en production pour le D+1 "
               "des 5 modèles (<font face=\"Courier\">models/*.py</font>)."))
story.append(li("Hook <font face=\"Courier\">sigma_scale</font> disponible sur les "
               "chemins D+7 et hebdomadaire de Prophet — prêt à être alimenté, pas "
               "encore branché en live."))
story.append(li("Détail par actif de l'investigation Prophet "
               "(<font face=\"Courier\">experiments/prophet_sigma_investigation.json</font>) "
               "— base pour décider d'un traitement différencié ZN/TLT."))

# ---- 5. Open items ----
story.append(p("5&nbsp;&nbsp; Reste ouvert", h1_style))
story.append(li("Tester log+EWMA combiné sur les 5 actifs pour savoir si ZN/TLT sont "
               "réellement sauvés par la couche EWMA, ou s'il faut un traitement "
               "différent par classe d'actif (§2)."))
story.append(li("Réponse du collègue sur <font face=\"Courier\">pi_recalibration.py</font> "
               "— toujours en attente."))
story.append(li("Intégration pipeline : alimenter <font face=\"Courier\">"
               "next_step_prophet(sigma_scale=...)</font> depuis <font face=\"Courier\">"
               "tracking.db</font> en production (HANDOFF_sigma_calibration_suivi.md §8.1)."))
story.append(li("ARIMA-GARCH/SARIMA/Naive dans <font face=\"Courier\">"
               "benchmarks/multi_horizon.py</font> utilisent toujours un Z_95 normal "
               "fixe pour le D+7 — non traité (hors périmètre de ce qui a été demandé "
               "aujourd'hui)."))

story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.4, color=colors.grey))
story.append(p(
    "Fichiers touchés aujourd'hui&nbsp;: <font face=\"Courier\">summarize_dist_options.py</font>, "
    "<font face=\"Courier\">generate_dist_options_report.py</font>, "
    "<font face=\"Courier\">generate_dist_options_pdf.py</font>, "
    "<font face=\"Courier\">benchmarks/multi_horizon.py</font>, "
    "<font face=\"Courier\">experiments/weekly_multimodel.py</font>, "
    "<font face=\"Courier\">benchmarks/test_multi_horizon.py</font> (nouveau), "
    "<font face=\"Courier\">experiments/prophet_sigma_investigation.json</font>, "
    "<font face=\"Courier\">NOTE_question_pi_recalibration.md</font> (nouveau).",
    ParagraphStyle("Footer", fontName="Times-Roman", fontSize=8.2, leading=11,
                  alignment=TA_CENTER, spaceBefore=8, textColor=colors.grey)))

doc = SimpleDocTemplate(str(OUT_PDF), pagesize=LETTER,
                        topMargin=2.1*cm, bottomMargin=2.1*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm)
doc.build(story)
print(f"Saved -> {OUT_PDF}")
