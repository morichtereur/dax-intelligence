"""Renders an answered query as a downloadable PDF memo — a print-ready
version of the same ledger the web app shows, for handing to someone who
isn't going to open the app."""
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT

from app.branding import company_meta
from app.llm import CITATION_RE, _pages_by_company, _resolve

INK = colors.HexColor("#171B16")
NAVY = colors.HexColor("#15222B")
PAPER_ON_NAVY = colors.HexColor("#E7E9E1")
GREEN = colors.HexColor("#1E6B45")
RED = colors.HexColor("#A0392D")
AMBER = colors.HexColor("#92661A")
RULE = colors.HexColor("#C7CBB6")
MUTED = colors.HexColor("#545843")

_FONT_DIR = __import__("pathlib").Path(__file__).parent / "assets" / "fonts"
_FONTS_REGISTERED = False


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    pdfmetrics.registerFont(TTFont("PlexSerif-Bold", str(_FONT_DIR / "IBMPlexSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("PlexSerif", str(_FONT_DIR / "IBMPlexSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("PlexMono", str(_FONT_DIR / "IBMPlexMono-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("PlexMono-Medium", str(_FONT_DIR / "IBMPlexMono-Medium.ttf")))
    pdfmetrics.registerFont(TTFont("PlexMono-SemiBold", str(_FONT_DIR / "IBMPlexMono-SemiBold.ttf")))
    _FONTS_REGISTERED = True


import re as _re
_BOLD_RE = _re.compile(r"\*\*(.+?)\*\*")


def _render_paragraph(text: str, chunks: list[dict]) -> str:
    """Escapes raw text, then re-expresses the small slice of Markdown Claude
    actually produces (**bold**, plus verified/flagged citations) as
    reportlab's inline markup — Paragraph has no Markdown support of its own."""
    pages_by_company = _pages_by_company(chunks)
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\n", "<br/>")
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)

    def _sub(m):
        raw_company, year, page = m.group(1).strip(), m.group(2), int(m.group(3))
        _, verified = _resolve(raw_company, year, page, pages_by_company)
        color = "#1E6B45" if verified else "#A0392D"
        return f'<font face="PlexMono" color="{color}">{m.group(0)}</font>'

    return CITATION_RE.sub(_sub, escaped)


def build_memo_pdf(
    query: str,
    answer: str,
    chunks: list[dict],
    citations: list[dict],
    best_score: float,
    low_confidence: bool,
    low_confidence_threshold: float,
) -> bytes:
    _register_fonts()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=40 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )

    styles = {
        "title": ParagraphStyle("title", fontName="PlexSerif-Bold", fontSize=20, textColor=PAPER_ON_NAVY, leading=24),
        "subtitle": ParagraphStyle("subtitle", fontName="PlexMono", fontSize=7.5, textColor=colors.HexColor("#9AA79C"), leading=11, tracking=1),
        "query": ParagraphStyle("query", fontName="PlexSerif", fontSize=12.5, textColor=INK, leading=17, spaceBefore=14, spaceAfter=10),
        "sectionHead": ParagraphStyle("sectionHead", fontName="PlexMono-SemiBold", fontSize=8.5, textColor=MUTED, leading=12, spaceBefore=6, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, textColor=INK, leading=15, spaceAfter=8),
        "memoHead": ParagraphStyle("memoHead", fontName="Helvetica-Bold", fontSize=11, textColor=INK, leading=15, spaceBefore=6, spaceAfter=4),
        "verify": ParagraphStyle("verify", fontName="PlexMono", fontSize=8, textColor=MUTED, leading=12, spaceBefore=2, spaceAfter=10),
        "banner": ParagraphStyle("banner", fontName="Helvetica", fontSize=9, textColor=AMBER, leading=13, spaceBefore=4, spaceAfter=10),
        "footer": ParagraphStyle("footer", fontName="PlexMono", fontSize=7, textColor=MUTED, leading=10),
        "th": ParagraphStyle("th", fontName="PlexMono-SemiBold", fontSize=7.5, textColor=PAPER_ON_NAVY),
        "td": ParagraphStyle("td", fontName="PlexMono", fontSize=8, textColor=INK),
        "tdMuted": ParagraphStyle("tdMuted", fontName="PlexMono", fontSize=8, textColor=MUTED),
    }

    story = []

    def header_footer(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, A4[1] - 34 * mm, A4[0], 34 * mm, fill=1, stroke=0)
        canvas.setFillColor(PAPER_ON_NAVY)
        canvas.setFont("PlexSerif-Bold", 20)
        canvas.drawString(20 * mm, A4[1] - 18 * mm, "DAX Intelligence")
        canvas.setFillColor(colors.HexColor("#9AA79C"))
        canvas.setFont("PlexMono", 7.5)
        n_companies = len({c["company"] for c in chunks})
        source_word = "SOURCE" if n_companies == 1 else "SOURCES"
        years_present = sorted({str(c["year"]) for c in chunks if c["year"]})
        year_tag = f"FY{years_present[0]}–FY{years_present[-1]}" if len(years_present) > 1 else f"FY{years_present[0]}" if years_present else ""
        canvas.drawString(20 * mm, A4[1] - 25 * mm, f"RETRIEVAL-GROUNDED RESEARCH MEMO   ·   {n_companies} {source_word} CITED   ·   {year_tag}")
        canvas.setFont("PlexMono", 7)
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 25 * mm, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        canvas.restoreState()

        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("PlexMono", 7)
        canvas.drawString(20 * mm, 9 * mm, "ChromaDB retrieval + hybrid BM25/cross-encoder re-rank · Claude Sonnet synthesis · built by Moritz Richter")
        canvas.drawRightString(A4[0] - 20 * mm, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    story.append(Paragraph(query, styles["query"]))

    if low_confidence:
        story.append(Paragraph(
            f'<font face="PlexMono" color="#92661A"><b>WEAK MATCH</b></font> — best retrieved excerpt scored '
            f'{best_score:.2f} similarity, below the {low_confidence_threshold:.2f} confidence bar. This memo may '
            f'rest on tangential evidence — check the audit trail before relying on it.',
            styles["banner"]
        ))

    story.append(Paragraph("ANALYST NOTE", styles["sectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=INK, spaceAfter=8))
    for para in answer.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.lstrip("#").strip()
            story.append(KeepTogether([Paragraph(_render_paragraph(heading, chunks), styles["memoHead"])]))
        else:
            # Individually kept-together: reportlab's mid-paragraph page
            # splitting has shown a rare rendering glitch on LLM-generated
            # text with inline citation markup (content bleeding into the
            # header band on the next page). Pushing whole paragraphs to the
            # next page instead avoids the split entirely — each paragraph
            # here is short enough that this costs at most a partial blank
            # page, never a broken one.
            story.append(KeepTogether([Paragraph(_render_paragraph(para, chunks), styles["body"])]))

    unverified = [c for c in citations if not c["verified"]]
    if citations:
        verified_n = len(citations) - len(unverified)
        if unverified:
            bad = ", ".join(f"{c['company']} p.{c['page']}" for c in unverified)
            story.append(Paragraph(
                f'<font color="#A0392D"><b>{len(unverified)} UNVERIFIED</b></font> — {verified_n}/{len(citations)} '
                f'citations matched retrieved excerpts exactly. Flagged: {bad}.',
                styles["verify"]
            ))
        else:
            story.append(Paragraph(
                f'<font color="#1E6B45"><b>VERIFIED</b></font> — {verified_n}/{len(citations)} citations matched '
                f'retrieved excerpts exactly.',
                styles["verify"]
            ))
    else:
        story.append(Paragraph(
            '<font color="#A0392D"><b>NO CITATIONS</b></font> — this memo made no verifiable citation; '
            'treat any claim above as unsupported.',
            styles["verify"]
        ))

    audit_block = [
        Spacer(1, 6),
        Paragraph(f"AUDIT TRAIL — {len(chunks)} EXCERPTS RETRIEVED", styles["sectionHead"]),
        HRFlowable(width="100%", thickness=0.75, color=RULE, spaceAfter=6),
    ]

    cited_pages = {(c["company"], c["year"], c["page"]) for c in citations if c["verified"]}
    years_present = {str(c["year"]) for c in chunks if c["year"]}
    table_data = [[
        Paragraph("COMPANY", styles["th"]), Paragraph("SECTION", styles["th"]),
        Paragraph("PAGE", styles["th"]), Paragraph("SCORE", styles["th"]), Paragraph("STATUS", styles["th"]),
    ]]
    for c in chunks:
        meta = company_meta(c["company"])
        is_cited = any(
            cc == c["company"]
            and (cy is None or cy == str(c["year"] or ""))
            and abs(cp - int(c["approx_page"] or 0)) <= 3
            for cc, cy, cp in cited_pages
        )
        status = Paragraph('<font color="#1E6B45"><b>CITED</b></font>' if is_cited else "retrieved", styles["tdMuted"])
        score_text = f"{c['score']:.3f}" if c["score"] is not None else "keyword"
        year_tag = f"FY{str(c['year'])[-2:]} · " if len(years_present) > 1 and c["year"] else ""
        table_data.append([
            Paragraph(meta["name"], styles["td"]),
            Paragraph((c["section"] or "").replace("_", " ").title(), styles["tdMuted"]),
            Paragraph(f"{year_tag}p.~{c['approx_page']}", styles["tdMuted"]),
            Paragraph(score_text, styles["tdMuted"]),
            status,
        ])

    table = Table(table_data, colWidths=[42 * mm, 42 * mm, 20 * mm, 22 * mm, 24 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
    ]))
    audit_block.append(table)
    story.append(KeepTogether(audit_block))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return buf.getvalue()
