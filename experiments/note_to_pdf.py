# -*- coding: utf-8 -*-
"""
note_to_pdf.py -- rend une NOTE_*.md du repo en PDF, dans le style maison
reportlab Times-Roman deja utilise par `generate_session_recap_pdf.py`,
`comparaison_options_calibration_tous_modeles.pdf` et
`comparaison_lois_garch_resultats.pdf`.

Generique et sans etat : le PDF est une VUE du markdown, jamais une seconde
source de verite. Toute correction se fait dans le .md, on regenere.

Sous-ensemble Markdown couvert (celui qu'emploient les notes du repo) :
titres `#`/`##`/`###`, paragraphes, listes a puces (un niveau
d'imbrication), tableaux GFM, citations `>`, regles `---`, et en ligne
`**gras**`, `*italique*`, `` `code` ``, `[texte](lien)`.

Usage :
    python experiments/note_to_pdf.py NOTE_xxx.md
    python experiments/note_to_pdf.py NOTE_xxx.md --out documentation/Xxx.pdf
"""

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent

title_style = ParagraphStyle("Title", fontName="Times-Bold", fontSize=15, leading=18,
                             alignment=TA_CENTER, spaceAfter=4)
meta_style = ParagraphStyle("Meta", fontName="Times-Italic", fontSize=9, leading=12,
                            alignment=TA_CENTER, spaceBefore=6, spaceAfter=14)
h1_style = ParagraphStyle("H1", fontName="Times-Bold", fontSize=12.5, leading=15,
                          spaceBefore=15, spaceAfter=6)
h2_style = ParagraphStyle("H2", fontName="Times-Bold", fontSize=10.8, leading=13.5,
                          spaceBefore=11, spaceAfter=5)
body_style = ParagraphStyle("Body", fontName="Times-Roman", fontSize=9.8, leading=13.6,
                            alignment=TA_JUSTIFY, spaceAfter=7)
li_style = ParagraphStyle("Li", fontName="Times-Roman", fontSize=9.8, leading=13.4,
                          alignment=TA_JUSTIFY, spaceAfter=5, leftIndent=13)
li_sub_style = ParagraphStyle("LiSub", fontName="Times-Roman", fontSize=9.3, leading=12.6,
                              alignment=TA_JUSTIFY, spaceAfter=4, leftIndent=26)
quote_style = ParagraphStyle("Quote", fontName="Times-Italic", fontSize=9.3, leading=12.8,
                             alignment=TA_JUSTIFY, spaceAfter=8, leftIndent=14, rightIndent=10)
cell_style = ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8.1, leading=10.2)
head_style = ParagraphStyle("Head", fontName="Times-Bold", fontSize=8.1, leading=10.2)


def inline(text: str) -> str:
    """Markdown en ligne -> mini-HTML de reportlab. L'echappement precede
    toute insertion de balise, sinon un `<` du texte casserait le parseur."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.4">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)      # le lien n'a pas de sens sur papier
    text = text.replace("—", "&#8212;").replace("–", "&#8211;")
    return text


def table_flowable(rows: list, width: float) -> Table:
    header, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), head_style) for c in header]]
    data += [[Paragraph(inline(c), cell_style) for c in r] for r in body]
    n = max(len(r) for r in rows)
    # 1re colonne un peu plus large (c'est presque toujours le libelle)
    first = width * min(0.30, 1.6 / n)
    rest = (width - first) / max(1, n - 1)
    t = Table(data, colWidths=[first] + [rest] * (n - 1), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#BBBBBB")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())) and "-" in line


def parse(md: str, width: float) -> list:
    """Markdown -> flowables. Un seul passage lineaire : les notes du repo
    n'imbriquent pas de structures au-dela d'un niveau de liste."""
    lines = md.split("\n")
    story, buf, i = [], [], 0

    def flush():
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), body_style))
            buf.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush(); i += 1; continue

        if stripped.startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            flush()
            rows, i = [_split_row(stripped)], i + 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i])); i += 1
            story.append(Spacer(1, 3))
            story.append(table_flowable(rows, width))
            story.append(Spacer(1, 9))
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush()
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#888888"),
                                    spaceBefore=8, spaceAfter=10))
            i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                story.append(Paragraph(inline(text), title_style))
            else:
                story.append(Paragraph(inline(text), h1_style if level == 2 else h2_style))
            i += 1; continue

        if stripped.startswith(">"):
            flush()
            quote = [stripped.lstrip("> ").strip()]
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip("> ").strip()); i += 1
            story.append(Paragraph(inline(" ".join(q for q in quote if q)), quote_style))
            continue

        m = re.match(r"^(\s*)[-*+]\s+(.*)$", raw)
        if m:
            flush()
            indent, text = len(m.group(1)), m.group(2)
            i += 1
            while i < len(lines):                       # continuations indentees
                nxt = lines[i]
                if nxt.strip() and not re.match(r"^\s*[-*+]\s+", nxt) and \
                   not nxt.strip().startswith(("#", "|", ">")) and \
                   len(nxt) - len(nxt.lstrip()) > indent:
                    text += " " + nxt.strip(); i += 1
                else:
                    break
            style = li_sub_style if indent >= 2 else li_style
            story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{inline(text)}", style))
            continue

        buf.append(stripped)
        i += 1

    flush()
    return story


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="chemin du .md (relatif a la racine du repo ou absolu)")
    ap.add_argument("--out", default=None, help="chemin du PDF (defaut : documentation/<nom>.pdf)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = (ROOT / src) if (ROOT / src).exists() else src.resolve()
    out = Path(args.out) if args.out else ROOT / "documentation" / f"{src.stem}.pdf"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.9 * cm, bottomMargin=1.9 * cm,
                            title=src.stem, author="DeepEdgeBenchmark")
    width = doc.width

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(1.9 * cm, 1.15 * cm, f"{src.name}  —  DeepEdgeBenchmark")
        canvas.drawRightString(A4[0] - 1.9 * cm, 1.15 * cm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.build(parse(src.read_text(encoding="utf-8"), width),
              onFirstPage=footer, onLaterPages=footer)
    print(f"{src.name} -> {out}  ({out.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
