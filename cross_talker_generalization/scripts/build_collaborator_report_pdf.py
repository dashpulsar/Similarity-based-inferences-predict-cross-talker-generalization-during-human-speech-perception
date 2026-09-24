"""Render the English collaborator report from its Markdown source.

Dependencies: reportlab and Pillow. PyMuPDF is used only for preview rendering.
The optional artifacts/report_runtime directory permits isolated dependencies.
"""
from pathlib import Path
import argparse
import html
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "artifacts/report_runtime"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from PIL import Image as PILImage


def inline(text):
    text = html.escape(text)
    # Keep web citations clickable; local paths remain available in Markdown.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<link href="{m.group(2)}" color="#245C8D">{m.group(1)}</link>'
                  if m.group(2).startswith('https://') else m.group(1), text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, help="Alternative PDF path, e.g. when the previous PDF is open")
    parser.add_argument("--source", type=Path, help="Report Markdown source; defaults to the earlier collaborator report")
    parser.add_argument("--preview-dir", type=Path, help="Optional directory for page PNGs")
    parser.add_argument("--image-max-height", type=float, default=370,
                        help="Maximum figure height in PDF points (default: 370)")
    args = parser.parse_args()
    folder = PROJECT / "analysis_update_2026-09-06/collaborator_report"
    markdown = args.source.resolve() if args.source else folder / "REPORT_FOR_FLORIAN.md"
    folder = markdown.parent
    output = PROJECT.parent / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    pdf = args.output.resolve() if args.output else output / "cross_talker_analysis_report_for_florian.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=colors.HexColor("#17364b"), spaceAfter=14))
    styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#17364b"), spaceAfter=10))
    styles.add(ParagraphStyle(name="Text", fontName="Helvetica", fontSize=11, leading=15, spaceAfter=9))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=9, leading=12))
    story = []
    buffer = []
    table_buffer = []
    def flush():
        if buffer:
            story.append(Paragraph(inline(" ".join(buffer)), styles["Text"]))
            buffer.clear()
        if table_buffer:
            rows = [[Paragraph(inline(x.strip()), styles["Cell"]) for x in row] for row in table_buffer]
            widths = [90, 250, page_width-80-340] if len(rows[0]) == 3 else None
            table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9f0f4")),
                ("VALIGN", (0,0), (-1,-1), "TOP"), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 7),
                ("LINEBELOW", (0,0), (-1,-1), .3, colors.HexColor("#cbd6dd")),
            ]))
            story.extend([table, Spacer(1, 9)])
            table_buffer.clear()
    for line in markdown.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            if not re.fullmatch(r"[| :\-]+", line):
                table_buffer.append(line.strip("|").split("|"))
            continue
        if not line.strip():
            flush(); continue
        if line == "---":
            flush(); story.append(PageBreak()); continue
        if line.startswith("#"):
            flush()
            title = line.lstrip("# ")
            story.append(Paragraph(inline(title), styles["ReportTitle" if line.startswith("# ") else "Section"]))
            continue
        match = re.fullmatch(r"!\[.*?\]\((.*?)\)", line)
        if match:
            flush()
            path = (folder / match.group(1)).resolve()
            with PILImage.open(path) as image:
                w, h = image.size
            scale = min((page_width-90)/w, args.image_max_height/h)
            story.extend([Image(str(path), width=w*scale, height=h*scale), Spacer(1, 7)])
            continue
        if re.match(r"^\d+\. ", line) or line.startswith("- "):
            flush(); story.append(Paragraph(inline(line), styles["Text"]))
        else:
            buffer.append(line)
    flush()
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#60717b"))
        canvas.drawString(40, 22, "Cross-talker generalization | For discussion")
        canvas.drawRightString(page_width-40, 22, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(pdf), pagesize=(page_width,page_height), rightMargin=40, leftMargin=40,
                            topMargin=32,bottomMargin=38,title="Cross-talker generalization: analysis and figure update", author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    if args.render:
        import fitz
        previews = args.preview_dir.resolve() if args.preview_dir else PROJECT.parent / "tmp/pdfs/collaborator_report"
        previews.mkdir(parents=True, exist_ok=True)
        with fitz.open(pdf) as document:
            for i, page in enumerate(document):
                page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(str(previews/f"page_{i+1:02d}.png"))
            print(f"Rendered {len(document)} pages to {previews}")
    print(pdf)


if __name__ == "__main__":
    main()
