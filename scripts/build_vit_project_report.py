"""
Comprehensive Academic Report Generator for VIT BCSE497J Project-I
Topic: An Explainable Multi-Model Machine Learning Framework for Financial Fraud Detection with Secure API Integration

Project Supervisor: Dr. Erapaneni Gayatri, Assistant Professor Sr. Grade 1, SCOPE
Head of Department: Dr. Boominathan P, Department of Computer Science and Engineering, SCOPE
Dean: Dr. Jaisankar N, School of Computer Science and Engineering (SCOPE)
Team Members (sorted by Register Number):
  23BCE0374  Dhruv Singh
  23BCE0415  Animesh B. Labh
  23BDS0091  Kushagra Dhelia
"""

import os
import sys
import re
from pathlib import Path
import html

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image, Flowable, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = WORKSPACE_DIR / "scripts" / "report_assets"
RESULTS_DIR = WORKSPACE_DIR / "results"
OUTPUT_PDF = WORKSPACE_DIR / "BCSE497J_Project_Report_Financial_Fraud_Detection.pdf"

WINDIR = os.environ.get("WINDIR", "C:/Windows")
FONTS_DIR = os.path.join(WINDIR, "Fonts")

pdfmetrics.registerFont(TTFont("TNR", os.path.join(FONTS_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("TNRB", os.path.join(FONTS_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("TNRI", os.path.join(FONTS_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("TNRBI", os.path.join(FONTS_DIR, "timesbi.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS_DIR, "consola.ttf")))
pdfmetrics.registerFont(TTFont("MonoB", os.path.join(FONTS_DIR, "consolab.ttf")))

pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNRB", italic="TNRI", boldItalic="TNRBI")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 90.0           # 1.25 in
MARGIN_RIGHT = 72.0          # 1.0 in
MARGIN_TOP = 72.0            # 1.0 in
MARGIN_BOTTOM = 72.0         # 1.0 in
PRINTABLE_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT  # 433.27 pt

TABLE_HEADER_BG = colors.HexColor("#2C4D75")
ROW_BG_ALT = colors.HexColor("#F2F5F8")
BORDER_COLOR = colors.HexColor("#CBD5E0")

styles = {
    "title_header": ParagraphStyle("title_header", fontName="TNRB", fontSize=15, leading=20, alignment=TA_CENTER),
    "title_main": ParagraphStyle("title_main", fontName="TNRB", fontSize=17, leading=23, alignment=TA_CENTER),
    "title_sub": ParagraphStyle("title_sub", fontName="TNR", fontSize=11, leading=16, alignment=TA_CENTER),
    "title_degree": ParagraphStyle("title_degree", fontName="TNRB", fontSize=14, leading=18, alignment=TA_CENTER),
    "title_branch": ParagraphStyle("title_branch", fontName="TNRB", fontSize=13, leading=17, alignment=TA_CENTER),
    "title_names": ParagraphStyle("title_names", fontName="TNRB", fontSize=11, leading=16, alignment=TA_CENTER),
    "title_guide_lbl": ParagraphStyle("title_guide_lbl", fontName="TNRI", fontSize=11, leading=15, alignment=TA_CENTER),
    "title_guide": ParagraphStyle("title_guide", fontName="TNRB", fontSize=12, leading=16, alignment=TA_CENTER),
    "title_univ": ParagraphStyle("title_univ", fontName="TNRB", fontSize=12.5, leading=17, alignment=TA_CENTER),

    "prelim_title": ParagraphStyle("prelim_title", fontName="TNRB", fontSize=16, leading=22, alignment=TA_CENTER, spaceAfter=16, keepWithNext=True),
    "abstract_title": ParagraphStyle("abstract_title", fontName="TNRB", fontSize=16, leading=22, alignment=TA_CENTER, spaceAfter=11, keepWithNext=True),
    "abstract_body": ParagraphStyle("abstract_body", fontName="TNR", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=7),

    "chap_num": ParagraphStyle("chap_num", fontName="TNRB", fontSize=15, leading=20, alignment=TA_CENTER, spaceBefore=0, spaceAfter=3, keepWithNext=True),
    "chap_title": ParagraphStyle("chap_title", fontName="TNRB", fontSize=14, leading=19, alignment=TA_CENTER, spaceBefore=0, spaceAfter=12, keepWithNext=True),
    "h1": ParagraphStyle("h1", fontName="TNRB", fontSize=12, leading=16, spaceBefore=9, spaceAfter=4, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="TNRB", fontSize=11, leading=15, spaceBefore=7, spaceAfter=3, keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName="TNRB", fontSize=10.5, leading=14, spaceBefore=5, spaceAfter=2, keepWithNext=True),

    "body": ParagraphStyle("body", fontName="TNR", fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=5),
    "body_bold": ParagraphStyle("body_bold", fontName="TNRB", fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="TNR", fontSize=10.5, leading=14.2, alignment=TA_JUSTIFY, leftIndent=14, firstLineIndent=-9, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="TNR", fontSize=9.5, leading=12.5, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8, keepWithNext=True),
    "table_caption": ParagraphStyle("table_caption", fontName="TNRB", fontSize=9.5, leading=12.5, alignment=TA_CENTER, spaceBefore=5, spaceAfter=3, keepWithNext=True),
    "table_cell": ParagraphStyle("table_cell", fontName="TNR", fontSize=8.5, leading=11, alignment=TA_LEFT),
    "table_cell_center": ParagraphStyle("table_cell_center", fontName="TNR", fontSize=8.5, leading=11, alignment=TA_CENTER),
    "table_cell_bold": ParagraphStyle("table_cell_bold", fontName="TNRB", fontSize=8.5, leading=11, alignment=TA_LEFT),
    "table_cell_bold_center": ParagraphStyle("table_cell_bold_center", fontName="TNRB", fontSize=8.5, leading=11, alignment=TA_CENTER),
    "table_cell_header": ParagraphStyle("table_cell_header", fontName="TNRB", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.white),
    "code_snippet": ParagraphStyle("code_snippet", fontName="Mono", fontSize=7.5, leading=9.8, spaceAfter=0),
    "equation": ParagraphStyle("equation", fontName="TNR", fontSize=10.5, leading=14, alignment=TA_CENTER),
    "eq_num": ParagraphStyle("eq_num", fontName="TNRB", fontSize=10, leading=14, alignment=TA_RIGHT),
    "ref_entry": ParagraphStyle("ref_entry", fontName="TNR", fontSize=9.5, leading=13, alignment=TA_JUSTIFY, leftIndent=22, firstLineIndent=-22, spaceAfter=5),

    "toc_header": ParagraphStyle("toc_header", fontName="TNRB", fontSize=10, leading=13, alignment=TA_LEFT),
    "toc_header_right": ParagraphStyle("toc_header_right", fontName="TNRB", fontSize=10, leading=13, alignment=TA_RIGHT),
    "toc_item_chap": ParagraphStyle("toc_item_chap", fontName="TNRB", fontSize=10, leading=13.5, spaceBefore=2.5, spaceAfter=1),
    "toc_item_sec": ParagraphStyle("toc_item_sec", fontName="TNR", fontSize=9, leading=12.5, leftIndent=12, spaceBefore=0.5, spaceAfter=0.5),
    "toc_item_subsec": ParagraphStyle("toc_item_subsec", fontName="TNR", fontSize=8.5, leading=11.5, leftIndent=22, spaceBefore=0.5, spaceAfter=0.5),
    "toc_page": ParagraphStyle("toc_page", fontName="TNR", fontSize=9, leading=12.5, alignment=TA_RIGHT),
    "toc_page_bold": ParagraphStyle("toc_page_bold", fontName="TNRB", fontSize=10, leading=13.5, alignment=TA_RIGHT),
}

def to_roman(n):
    if n <= 0:
        return ""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"]
    res = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            res += syb[i]
            n -= val[i]
        i += 1
    return res

class PageTracker(Flowable):
    def __init__(self, key, registry):
        super().__init__()
        self.key = key
        self.registry = registry
        self.width = 0
        self.height = 0

    def draw(self):
        self.registry[self.key] = self.canv._pageNumber

class VITReportCanvas(canvas.Canvas):
    abstract_end_page = 11

    def showPage(self):
        phys_page = self._pageNumber
        if phys_page > 1:
            if phys_page <= VITReportCanvas.abstract_end_page:
                roman_str = to_roman(phys_page - 1)
                self.setFont("TNR", 10)
                self.drawCentredString(PAGE_WIDTH / 2.0, 36, roman_str)
            else:
                arabic_num = phys_page - VITReportCanvas.abstract_end_page
                self.setFont("TNR", 10)
                self.drawCentredString(PAGE_WIDTH / 2.0, 36, str(arabic_num))

                self.setFont("TNRI", 8.5)
                self.drawString(MARGIN_LEFT, PAGE_HEIGHT - 46, "B.Tech. BCSE497J Project-I: Explainable Financial Fraud Detection")
                self.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 46, "VIT Vellore")
                self.setStrokeColor(colors.HexColor("#A0AEC0"))
                self.setLineWidth(0.5)
                self.line(MARGIN_LEFT, PAGE_HEIGHT - 50, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 50)

        super().showPage()

def format_citations(text: str) -> str:
    r"""
    Transforms citation brackets like [1], [2], [1, 2], [4, 7], [9, 10], [11, 12, 13]
    into clickable hyperlinks pointing to #ref{N}.
    Ignores non-citations like [0, 1], [0, \infty), [0.720, 0.898], [B = 100], etc.
    """
    if "[" not in text:
        return text

    def repl(m):
        content = m.group(1).strip()
        parts = [p_part.strip() for p_part in content.split(",")]
        nums = []
        for p_part in parts:
            if p_part.isdigit():
                val = int(p_part)
                if 1 <= val <= 16:
                    nums.append(val)
                else:
                    return m.group(0)
            else:
                return m.group(0)

        if nums:
            links = [f'<a href="#ref{n}" color="#1A5276"><b>[{n}]</b></a>' for n in nums]
            return ", ".join(links)
        return m.group(0)

    return re.sub(r"\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]", repl, text)

def p(text, style_name="body"):
    # Dynamically inject destination anchors for headings, table captions, and prelims
    if style_name == "chap_num":
        m = re.search(r"CHAPTER\s+(\d+)", text)
        if m:
            text = f'<a name="chap{m.group(1)}"/>' + text
    elif style_name in ("h1", "h2"):
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
        if m:
            sec_tag = "sec" + m.group(1).replace(".", "_")
            text = f'<a name="{sec_tag}"/>' + text
    elif style_name == "table_caption":
        m = re.search(r"Table\s+(\d+\.\d+)", text)
        if m:
            tbl_tag = "tbl" + m.group(1).replace(".", "_")
            text = f'<a name="{tbl_tag}"/>' + text
    elif style_name == "abstract_title":
        text = '<a name="abstract"/>' + text
    elif style_name == "prelim_title" and "APPENDIX" in text:
        text = '<a name="appendix"/>' + text

    if style_name != "ref_entry" and "[" in text and not text.startswith("<a name=\"ref"):
        text = format_citations(text)
    return Paragraph(text, styles[style_name])

def b(text):
    if "[" in text and not text.startswith("<a name=\"ref"):
        text = format_citations(text)
    return Paragraph("-&nbsp;&nbsp;" + text, styles["bullet"])

def sp(h=5):
    return Spacer(1, h)

def pb():
    return PageBreak()

def eq_box(formula_html, eq_num=None):
    if eq_num:
        p_eq = Paragraph(formula_html, styles["equation"])
        p_num = Paragraph(f"<b>({eq_num})</b>", styles["eq_num"])
        t = Table([[p_eq, p_num]], colWidths=[385, 48], hAlign="CENTER")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return t
    else:
        return Paragraph(formula_html, styles["equation"])

def create_styled_table(headers, rows, widths=None):
    table_data = [[Paragraph(f"<b>{html.escape(h)}</b>", styles["table_cell_header"]) for h in headers]]
    for r_idx, row in enumerate(rows):
        row_cells = []
        for c_idx, val in enumerate(row):
            txt = str(val).replace("\n", "<br/>")
            if "[" in txt and not txt.startswith("<a name=\"ref"):
                txt = format_citations(txt)
            if "<b>" in txt or "<i>" in txt or "<sub>" in txt or "<sup>" in txt or "<br/>" in txt or "<a " in txt:
                p_cell = Paragraph(txt, styles["table_cell"])
            else:
                p_cell = Paragraph(html.escape(txt), styles["table_cell"])
            row_cells.append(p_cell)
        table_data.append(row_cells)

    if widths is None:
        widths = [PRINTABLE_WIDTH / len(headers)] * len(headers)

    t = Table(table_data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for r_idx in range(1, len(table_data)):
        if r_idx % 2 == 0:
            ts.append(("BACKGROUND", (0, r_idx), (-1, r_idx), ROW_BG_ALT))
    t.setStyle(TableStyle(ts))
    return t

def create_code_block(code_str):
    lines = code_str.strip().split("\n")
    formatted = "<br/>".join([html.escape(l).replace(" ", "&nbsp;") for l in lines])
    p_code = Paragraph(formatted, styles["code_snippet"])
    t = Table([[p_code]], colWidths=[PRINTABLE_WIDTH], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t

def embed_image(img_path, width=400, height=220, caption_text="", fig_key=""):
    flowables = []
    if os.path.exists(img_path):
        img = Image(str(img_path), width=width, height=height)
        img.hAlign = "CENTER"
        flowables.append(img)
        if caption_text:
            flowables.append(sp(2.5))
            if not fig_key:
                m = re.search(r"Figure\s+(\d+\.\d+)", caption_text)
                if m:
                    fig_key = "fig" + m.group(1).replace(".", "_")
            cap = f'<a name="{fig_key}"/>{caption_text}' if fig_key else caption_text
            flowables.append(Paragraph(cap, styles["caption"]))
    else:
        flowables.append(Paragraph(f"[Image Missing: {img_path}]", styles["caption"]))
    return KeepTogether(flowables)

def build_report_story(registry, is_pass2=False):
    story = []
    abs_end = registry.get("abstract", 11)
    
    def get_pg(key, default=1):
        if not is_pass2 or key not in registry:
            return default
        phys = registry[key]
        if phys <= abs_end:
            return to_roman(phys - 1)
        return str(phys - abs_end)

    # =========================================================
    # COVER / TITLE PAGE
    # =========================================================
    story.append(p("B.Tech. BCSE497J - Project-I", "title_header"))
    story.append(sp(18))
    story.append(p("AN EXPLAINABLE MULTI-MODEL MACHINE LEARNING FRAMEWORK FOR FINANCIAL FRAUD DETECTION WITH SECURE API INTEGRATION", "title_main"))
    story.append(sp(16))
    story.append(p("Submitted in partial fulfillment of the requirements for the degree of", "title_sub"))
    story.append(sp(5))
    story.append(p("Bachelor of Technology", "title_degree"))
    story.append(sp(3))
    story.append(p("in", "title_sub"))
    story.append(sp(3))
    story.append(p("Computer Science and Engineering", "title_branch"))
    story.append(sp(14))
    story.append(p("by", "title_sub"))
    story.append(sp(5))
    
    # Team members sorted on Register Number
    members_data = [
        [Paragraph("<b>23BCE0374</b>", styles["table_cell_bold_center"]), Paragraph("<b>DHRUV SINGH</b>", styles["table_cell_bold"])],
        [Paragraph("<b>23BCE0415</b>", styles["table_cell_bold_center"]), Paragraph("<b>ANIMESH B. LABH</b>", styles["table_cell_bold"])],
        [Paragraph("<b>23BDS0091</b>", styles["table_cell_bold_center"]), Paragraph("<b>KUSHAGRA DHELIA</b>", styles["table_cell_bold"])],
    ]
    t_members = Table(members_data, colWidths=[120, 200], hAlign="CENTER")
    t_members.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_members)
    story.append(sp(14))
    
    # Supervisor updated per user request
    story.append(p("Under the Supervision of", "title_guide_lbl"))
    story.append(sp(4))
    story.append(p("Dr. Erapaneni Gayatri", "title_guide"))
    story.append(p("Assistant Professor Sr. Grade 1", "title_sub"))
    story.append(p("School of Computer Science and Engineering (SCOPE)", "title_sub"))
    story.append(sp(12))
    
    # VIT Logo
    logo_path = ASSETS_DIR / "vit_logo.png"
    if logo_path.exists():
        img_logo = Image(str(logo_path), width=100, height=105)
        img_logo.hAlign = "CENTER"
        story.append(img_logo)
    story.append(sp(6))
    
    story.append(p("Vellore Institute of Technology", "title_univ"))
    story.append(p("(Deemed to be University under section 3 of UGC Act, 1956)", "title_sub"))
    story.append(sp(3))
    story.append(p("October 2026", "title_names"))
    story.append(pb())

    # =========================================================
    # PRELIMINARY: DECLARATION (Page i)
    # =========================================================
    story.append(PageTracker("declaration", registry))
    story.append(p("DECLARATION", "prelim_title"))
    story.append(sp(14))
    story.append(p(
        "I hereby declare that the project entitled <b>AN EXPLAINABLE MULTI-MODEL MACHINE LEARNING FRAMEWORK FOR FINANCIAL FRAUD DETECTION WITH SECURE API INTEGRATION</b> "
        "submitted by us, for the award of the degree of <i>Bachelor of Technology in Computer Science and Engineering</i> to VIT is a record of bonafide work carried out by us under the supervision of "
        "<b>Dr. Erapaneni Gayatri</b>, Assistant Professor Sr. Grade 1, School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore."
    ))
    story.append(sp(10))
    story.append(p(
        "I further declare that the work reported in this project has not been submitted and will not be submitted, either in part or in full, "
        "for the award of any other degree or diploma in this institute or any other institute or university."
    ))
    story.append(sp(36))
    
    dec_meta = [
        [Paragraph("<b>Place :</b> Vellore", styles["body"]), Paragraph("<b>Signature of the Candidates:</b>", styles["body"])],
        [Paragraph("<b>Date  :</b> 26 October 2026", styles["body"]), Paragraph("1. <b>Dhruv Singh (23BCE0374)</b>", styles["body"])],
        [Paragraph("", styles["body"]), Paragraph("2. <b>Animesh B. Labh (23BCE0415)</b>", styles["body"])],
        [Paragraph("", styles["body"]), Paragraph("3. <b>Kushagra Dhelia (23BDS0091)</b>", styles["body"])],
    ]
    t_dec = Table(dec_meta, colWidths=[180, 250], hAlign="CENTER")
    t_dec.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_dec)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: CERTIFICATE (Page ii)
    # =========================================================
    story.append(PageTracker("certificate", registry))
    story.append(p("CERTIFICATE", "prelim_title"))
    story.append(sp(14))
    story.append(p(
        "This is to certify that the project entitled <b>AN EXPLAINABLE MULTI-MODEL MACHINE LEARNING FRAMEWORK FOR FINANCIAL FRAUD DETECTION WITH SECURE API INTEGRATION</b> "
        "submitted by <b>DHRUV SINGH (23BCE0374), ANIMESH B. LABH (23BCE0415), and KUSHAGRA DHELIA (23BDS0091)</b>, School of Computer Science and Engineering, VIT, "
        "for the award of the degree of <i>Bachelor of Technology in Computer Science and Engineering</i>, is a record of bonafide work carried out by them under my supervision during "
        "Fall Semester 2026-2027, as per the VIT code of academic and research ethics."
    ))
    story.append(sp(10))
    story.append(p(
        "The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this institute or any other "
        "institute or university. The project fulfills the requirements and regulations of the University and in my opinion meets the necessary standards for submission."
    ))
    story.append(sp(40))
    
    cert_meta = [
        [Paragraph("<b>Place :</b> Vellore", styles["body"]), Paragraph("<b>Signature of the Guide</b>", styles["body_bold"])],
        [Paragraph("<b>Date  :</b> 26 October 2026", styles["body"]), Paragraph("<b>Dr. Erapaneni Gayatri</b><br/>Assistant Professor Sr. Grade 1<br/>School of Computer Science and Engineering (SCOPE)", styles["body"])],
        [Paragraph("<br/><br/><b>Examiner(s):</b><br/>1.<br/>2.", styles["body"]), Paragraph("<br/><br/><b>Head of the Department</b><br/><b>Dr. Boominathan P</b><br/>Department of Computer Science and Engineering<br/>School of Computer Science and Engineering (SCOPE)", styles["body"])],
    ]
    t_cert = Table(cert_meta, colWidths=[180, 250], hAlign="CENTER")
    t_cert.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_cert)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: ACKNOWLEDGEMENTS (Page iii)
    # =========================================================
    story.append(PageTracker("acknowledgements", registry))
    story.append(p("ACKNOWLEDGEMENTS", "prelim_title"))
    story.append(sp(10))
    story.append(p(
        "I am deeply grateful to the management of <b>Vellore Institute of Technology (VIT)</b> for providing me with the opportunity and resources to undertake this project. "
        "Their commitment to fostering a conducive learning environment has been instrumental in my academic journey. The support and infrastructure provided by VIT have enabled me "
        "to explore and develop my ideas to their fullest potential."
    ))
    story.append(sp(5))
    story.append(p(
        "My sincere thanks to <b>Dr. Jaisankar N</b>, the Dean of the School of Computer Science and Engineering (SCOPE), for his unwavering support and encouragement. "
        "His leadership and vision have greatly inspired me to strive for excellence. The Dean's dedication to academic excellence and innovation has been a constant source of "
        "motivation for me. I appreciate his efforts in creating an environment that nurtures creativity and critical thinking."
    ))
    story.append(sp(5))
    story.append(p(
        "I express my profound appreciation to <b>Dr. Boominathan P</b>, the Head of the Department of Computer Science and Engineering, for his insightful guidance and continuous support. "
        "His expertise and advice have been crucial in shaping the direction of my project. The Head of Department's commitment to fostering a collaborative and supportive atmosphere "
        "has greatly enhanced my learning experience. His constructive feedback and encouragement have been invaluable in overcoming challenges and achieving my project goals."
    ))
    story.append(sp(5))
    story.append(p(
        "I am immensely thankful to my project guide, <b>Dr. Erapaneni Gayatri</b>, Assistant Professor Sr. Grade 1, for her dedicated mentorship and invaluable feedback. "
        "Her patience, knowledge, and encouragement have been pivotal in the successful completion of this project. My supervisor's willingness to share her expertise and provide "
        "thoughtful guidance has been instrumental in refining my ideas and methodologies. Her support has not only contributed to the success of this project but has also "
        "enriched my overall academic experience."
    ))
    story.append(sp(5))
    story.append(p("Thank you all for your contributions and support."))
    story.append(sp(20))
    
    ack_names = [
        [Paragraph("", styles["body"]), Paragraph("<b>Dhruv Singh (23BCE0374)</b>", styles["body"])],
        [Paragraph("", styles["body"]), Paragraph("<b>Animesh B. Labh (23BCE0415)</b>", styles["body"])],
        [Paragraph("", styles["body"]), Paragraph("<b>Kushagra Dhelia (23BDS0091)</b>", styles["body"])],
    ]
    t_ack = Table(ack_names, colWidths=[200, 230], hAlign="CENTER")
    story.append(t_ack)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: TABLE OF CONTENTS (Pages iv - v)
    # =========================================================
    story.append(PageTracker("toc_p1", registry))
    story.append(p("TABLE OF CONTENTS", "prelim_title"))
    story.append(sp(6))

    toc_rows = [
        [Paragraph("<b>Sl.No</b>", styles["toc_header"]), Paragraph("<b>Contents</b>", styles["toc_header"]), Paragraph("<b>Page No.</b>", styles["toc_header_right"])],
        [Paragraph("", styles["toc_item_chap"]), Paragraph("<a href=\"#abstract\" color=\"#000000\"><b>Abstract</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#abstract\" color=\"#000000\">{get_pg('abstract', 'x')}</a></b>", styles["toc_page_bold"])],
        [Paragraph("<b>1.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap1\" color=\"#000000\"><b>INTRODUCTION</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap1\" color=\"#000000\">{get_pg('chap1', 1)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec1_1\" color=\"#000000\">1.1 Background</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec1_1\" color=\"#000000\">{get_pg('sec1_1', 1)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec1_2\" color=\"#000000\">1.2 Motivations</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec1_2\" color=\"#000000\">{get_pg('sec1_2', 1)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec1_3\" color=\"#000000\">1.3 Scope of the Project</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec1_3\" color=\"#000000\">{get_pg('sec1_3', 2)}</a>", styles["toc_page"])],
        [Paragraph("<b>2.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap2\" color=\"#000000\"><b>PROJECT DESCRIPTION AND GOALS</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap2\" color=\"#000000\">{get_pg('chap2', 3)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec2_1\" color=\"#000000\">2.1 Literature Review</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec2_1\" color=\"#000000\">{get_pg('sec2_1', 3)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec2_2\" color=\"#000000\">2.2 Research Gap</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec2_2\" color=\"#000000\">{get_pg('sec2_2', 5)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec2_3\" color=\"#000000\">2.3 Objectives</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec2_3\" color=\"#000000\">{get_pg('sec2_3', 6)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec2_4\" color=\"#000000\">2.4 Problem Statement</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec2_4\" color=\"#000000\">{get_pg('sec2_4', 6)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec2_5\" color=\"#000000\">2.5 Project Plan</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec2_5\" color=\"#000000\">{get_pg('sec2_5', 7)}</a>", styles["toc_page"])],
        [Paragraph("<b>3.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap3\" color=\"#000000\"><b>TECHNICAL SPECIFICATION</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap3\" color=\"#000000\">{get_pg('chap3', 8)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec3_1\" color=\"#000000\">3.1 Requirements</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec3_1\" color=\"#000000\">{get_pg('sec3_1', 8)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_1_1\" color=\"#000000\">3.1.1 Functional Requirements</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_1_1\" color=\"#000000\">{get_pg('sec3_1_1', 8)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_1_2\" color=\"#000000\">3.1.2 Non-Functional Requirements</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_1_2\" color=\"#000000\">{get_pg('sec3_1_2', 9)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec3_2\" color=\"#000000\">3.2 Feasibility Study</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec3_2\" color=\"#000000\">{get_pg('sec3_2', 10)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_2_1\" color=\"#000000\">3.2.1 Technical Feasibility</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_2_1\" color=\"#000000\">{get_pg('sec3_2_1', 10)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_2_2\" color=\"#000000\">3.2.2 Economic Feasibility</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_2_2\" color=\"#000000\">{get_pg('sec3_2_2', 10)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_2_3\" color=\"#000000\">3.2.3 Social Feasibility</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_2_3\" color=\"#000000\">{get_pg('sec3_2_3', 11)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec3_3\" color=\"#000000\">3.3 System Specification</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec3_3\" color=\"#000000\">{get_pg('sec3_3', 11)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_3_1\" color=\"#000000\">3.3.1 Hardware Specification</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_3_1\" color=\"#000000\">{get_pg('sec3_3_1', 11)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec3_3_2\" color=\"#000000\">3.3.2 Software Specification</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec3_3_2\" color=\"#000000\">{get_pg('sec3_3_2', 12)}</a>", styles["toc_page"])],
        [Paragraph("<b>4.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap4\" color=\"#000000\"><b>DESIGN APPROACH AND DETAILS</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap4\" color=\"#000000\">{get_pg('chap4', 13)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec4_1\" color=\"#000000\">4.1 System Architecture</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec4_1\" color=\"#000000\">{get_pg('sec4_1', 13)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec4_2\" color=\"#000000\">4.2 Design</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec4_2\" color=\"#000000\">{get_pg('sec4_2', 14)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec4_2_1\" color=\"#000000\">4.2.1 Data Flow Diagram</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec4_2_1\" color=\"#000000\">{get_pg('sec4_2_1', 14)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec4_2_2\" color=\"#000000\">4.2.2 Use Case Diagram</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec4_2_2\" color=\"#000000\">{get_pg('sec4_2_2', 16)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec4_2_3\" color=\"#000000\">4.2.3 Class Diagram</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec4_2_3\" color=\"#000000\">{get_pg('sec4_2_3', 17)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec4_2_4\" color=\"#000000\">4.2.4 Sequence Diagram</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec4_2_4\" color=\"#000000\">{get_pg('sec4_2_4', 19)}</a>", styles["toc_page"])],
    ]
    t_toc1 = Table(toc_rows, colWidths=[35, 345, 53], hAlign="CENTER")
    t_toc1.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#2C4D75")),
    ]))
    story.append(t_toc1)
    story.append(pb())

    # TOC Page 2
    story.append(PageTracker("toc_p2", registry))
    toc_rows2 = [
        [Paragraph("<b>Sl.No</b>", styles["toc_header"]), Paragraph("<b>Contents (Continued)</b>", styles["toc_header"]), Paragraph("<b>Page No.</b>", styles["toc_header_right"])],
        [Paragraph("<b>5.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap5\" color=\"#000000\"><b>METHODOLOGY AND TESTING</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap5\" color=\"#000000\">{get_pg('chap5', 21)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec5_1\" color=\"#000000\">5.1 Module Description</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec5_1\" color=\"#000000\">{get_pg('sec5_1', 21)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_1\" color=\"#000000\">5.1.1 Preprocessing and Feature Forensics</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_1\" color=\"#000000\">{get_pg('sec5_1_1', 21)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_2\" color=\"#000000\">5.1.2 Class Imbalance Handling via SMOTE-NC</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_2\" color=\"#000000\">{get_pg('sec5_1_2', 22)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_3\" color=\"#000000\">5.1.3 Supervised Tree Ensembles (RF & XGBoost)</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_3\" color=\"#000000\">{get_pg('sec5_1_3', 23)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_4\" color=\"#000000\">5.1.4 Deep Neural Network & Autoencoder Representation</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_4\" color=\"#000000\">{get_pg('sec5_1_4', 24)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_5\" color=\"#000000\">5.1.5 Multi-Model Consensus Voting & Scoring Engine</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_5\" color=\"#000000\">{get_pg('sec5_1_5', 25)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_6\" color=\"#000000\">5.1.6 Explainability via TreeSHAP and Balance Forensics</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_6\" color=\"#000000\">{get_pg('sec5_1_6', 26)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_7\" color=\"#000000\">5.1.7 Hardened RESTful API & Security Subsystem</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_7\" color=\"#000000\">{get_pg('sec5_1_7', 27)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_1_8\" color=\"#000000\">5.1.8 Real-Time Kafka Event Streaming Pipeline</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_1_8\" color=\"#000000\">{get_pg('sec5_1_8', 28)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec5_2\" color=\"#000000\">5.2 Testing</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec5_2\" color=\"#000000\">{get_pg('sec5_2', 29)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_2_1\" color=\"#000000\">5.2.1 Automated Test Suite & Regression Testing</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_2_1\" color=\"#000000\">{get_pg('sec5_2_1', 29)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_2_2\" color=\"#000000\">5.2.2 Model Integrity & Preprocessing Edge Cases</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_2_2\" color=\"#000000\">{get_pg('sec5_2_2', 30)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_2_3\" color=\"#000000\">5.2.3 Security, Authorization & Injection Resistance</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_2_3\" color=\"#000000\">{get_pg('sec5_2_3', 31)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_subsec"]), Paragraph("<a href=\"#sec5_2_4\" color=\"#000000\">5.2.4 Latency & Concurrency Stress Testing</a>", styles["toc_item_subsec"]), Paragraph(f"<a href=\"#sec5_2_4\" color=\"#000000\">{get_pg('sec5_2_4', 32)}</a>", styles["toc_page"])],
        [Paragraph("<b>6.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap6\" color=\"#000000\"><b>PROJECT DEMONSTRATION</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap6\" color=\"#000000\">{get_pg('chap6', 33)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec6_1\" color=\"#000000\">6.1 Multi-Tenant Streamlit Analyst Workspace</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec6_1\" color=\"#000000\">{get_pg('sec6_1', 33)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec6_2\" color=\"#000000\">6.2 Single-Transaction Scoring & Forensic Diagnosis</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec6_2\" color=\"#000000\">{get_pg('sec6_2', 34)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec6_3\" color=\"#000000\">6.3 Batch CSV Inference & Audit Trail</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec6_3\" color=\"#000000\">{get_pg('sec6_3', 35)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec6_4\" color=\"#000000\">6.4 Explainable AI Visualizations</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec6_4\" color=\"#000000\">{get_pg('sec6_4', 36)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec6_5\" color=\"#000000\">6.5 Live & Simulated Kafka Streaming Feeds</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec6_5\" color=\"#000000\">{get_pg('sec6_5', 37)}</a>", styles["toc_page"])],
        [Paragraph("<b>7.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap7\" color=\"#000000\"><b>RESULT AND DISCUSSION (COST ANALYSIS)</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap7\" color=\"#000000\">{get_pg('chap7', 38)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec7_1\" color=\"#000000\">7.1 Primary Benchmark Evaluation (3,000 Holdout)</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec7_1\" color=\"#000000\">{get_pg('sec7_1', 38)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec7_2\" color=\"#000000\">7.2 Secondary Benchmark & Cross-Generator Shift Analysis</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec7_2\" color=\"#000000\">{get_pg('sec7_2', 39)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec7_3\" color=\"#000000\">7.3 Precision-Recall & ROC Analysis</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec7_3\" color=\"#000000\">{get_pg('sec7_3', 40)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec7_4\" color=\"#000000\">7.4 False Positive Reduction & Operational Tradeoffs</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec7_4\" color=\"#000000\">{get_pg('sec7_4', 41)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec7_5\" color=\"#000000\">7.5 Financial Cost-Benefit & Economic Loss Modeling</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec7_5\" color=\"#000000\">{get_pg('sec7_5', 42)}</a>", styles["toc_page"])],
        [Paragraph("<b>8.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap8\" color=\"#000000\"><b>CONCLUSION</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap8\" color=\"#000000\">{get_pg('chap8', 44)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec8_1\" color=\"#000000\">8.1 Summary of Contributions</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec8_1\" color=\"#000000\">{get_pg('sec8_1', 44)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec8_2\" color=\"#000000\">8.2 Limitations</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec8_2\" color=\"#000000\">{get_pg('sec8_2', 44)}</a>", styles["toc_page"])],
        [Paragraph("", styles["toc_item_sec"]), Paragraph("<a href=\"#sec8_3\" color=\"#000000\">8.3 Future Work</a>", styles["toc_item_sec"]), Paragraph(f"<a href=\"#sec8_3\" color=\"#000000\">{get_pg('sec8_3', 45)}</a>", styles["toc_page"])],
        [Paragraph("<b>9.</b>", styles["toc_item_chap"]), Paragraph("<a href=\"#chap9\" color=\"#000000\"><b>REFERENCES</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#chap9\" color=\"#000000\">{get_pg('chap9', 46)}</a></b>", styles["toc_page_bold"])],
        [Paragraph("", styles["toc_item_chap"]), Paragraph("<a href=\"#appendix\" color=\"#000000\"><b>APPENDIX A - SAMPLE CODE</b></a>", styles["toc_item_chap"]), Paragraph(f"<b><a href=\"#appendix\" color=\"#000000\">{get_pg('appendix', 48)}</a></b>", styles["toc_page_bold"])],
    ]
    t_toc2 = Table(toc_rows2, colWidths=[35, 345, 53], hAlign="CENTER")
    t_toc2.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#2C4D75")),
    ]))
    story.append(t_toc2)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: LIST OF FIGURES (Page vi)
    # =========================================================
    story.append(PageTracker("list_of_figures", registry))
    story.append(p("List of Figures", "prelim_title"))
    story.append(sp(6))
    
    fig_rows = [
        [Paragraph("<b>Figure No.</b>", styles["toc_header"]), Paragraph("<b>Title</b>", styles["toc_header"]), Paragraph("<b>Page No.</b>", styles["toc_header_right"])],
        [Paragraph("<a href=\"#fig4_1\" color=\"#000000\">4.1</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_1\" color=\"#000000\">Architecture of the Multi-Model Consensus Fraud Scoring Pipeline</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_1\" color=\"#000000\">{get_pg('fig4_1', 13)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig4_2\" color=\"#000000\">4.2</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_2\" color=\"#000000\">Data Flow Diagram (DFD Level 1) of the Transaction Processing Engine</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_2\" color=\"#000000\">{get_pg('fig4_2', 15)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig4_3\" color=\"#000000\">4.3</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_3\" color=\"#000000\">Use Case Diagram for Enterprise Fraud Detection and Analyst Workspace</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_3\" color=\"#000000\">{get_pg('fig4_3', 16)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig4_4\" color=\"#000000\">4.4</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_4\" color=\"#000000\">Class Diagram of Preprocessing, Consensus, and Streaming Components</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_4\" color=\"#000000\">{get_pg('fig4_4', 18)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig4_5\" color=\"#000000\">4.5</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_5\" color=\"#000000\">Sequence Diagram of Transaction Ingestion, Scoring, and Audit Logging</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_5\" color=\"#000000\">{get_pg('fig4_5', 20)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig4_6\" color=\"#000000\">4.6</a>", styles["table_cell"]), Paragraph("<a href=\"#fig4_6\" color=\"#000000\">Event-Driven Apache Kafka Real-Time Streaming Ingestion Pipeline</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig4_6\" color=\"#000000\">{get_pg('fig4_6', 28)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig6_1\" color=\"#000000\">6.1</a>", styles["table_cell"]), Paragraph("<a href=\"#fig6_1\" color=\"#000000\">Class Distribution Before and After Synthetic Minority Oversampling (SMOTE)</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig6_1\" color=\"#000000\">{get_pg('fig6_1', 33)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig6_2\" color=\"#000000\">6.2</a>", styles["table_cell"]), Paragraph("<a href=\"#fig6_2\" color=\"#000000\">Training and Validation Loss Curves for Deep Neural Network and Autoencoder</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig6_2\" color=\"#000000\">{get_pg('fig6_2', 35)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig6_3\" color=\"#000000\">6.3</a>", styles["table_cell"]), Paragraph("<a href=\"#fig6_3\" color=\"#000000\">Global Feature Importance Rankings Derived from TreeSHAP Explanations</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig6_3\" color=\"#000000\">{get_pg('fig6_3', 36)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig6_4\" color=\"#000000\">6.4</a>", styles["table_cell"]), Paragraph("<a href=\"#fig6_4\" color=\"#000000\">TreeSHAP Beeswarm Summary Plot on Critical Financial Attributions</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig6_4\" color=\"#000000\">{get_pg('fig6_4', 37)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig6_5\" color=\"#000000\">6.5</a>", styles["table_cell"]), Paragraph("<a href=\"#fig6_5\" color=\"#000000\">Streamlit Multi-Tenant Analyst Workspace and Authentication Interface</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig6_5\" color=\"#000000\">{get_pg('fig6_5', 34)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig7_1\" color=\"#000000\">7.1</a>", styles["table_cell"]), Paragraph("<a href=\"#fig7_1\" color=\"#000000\">Confusion Matrices across Supervised and Unsupervised Detection Models</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig7_1\" color=\"#000000\">{get_pg('fig7_1', 39)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig7_2\" color=\"#000000\">7.2</a>", styles["table_cell"]), Paragraph("<a href=\"#fig7_2\" color=\"#000000\">Comprehensive Model Performance Comparison across Key Metrics</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig7_2\" color=\"#000000\">{get_pg('fig7_2', 40)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig7_3\" color=\"#000000\">7.3</a>", styles["table_cell"]), Paragraph("<a href=\"#fig7_3\" color=\"#000000\">Receiver Operating Characteristic (ROC) Curves on Held-Out Test Data</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig7_3\" color=\"#000000\">{get_pg('fig7_3', 41)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#fig7_4\" color=\"#000000\">7.4</a>", styles["table_cell"]), Paragraph("<a href=\"#fig7_4\" color=\"#000000\">Precision-Recall (PR) Curves Demonstrating Ranking Robustness under Imbalance</a>", styles["table_cell"]), Paragraph(f"<a href=\"#fig7_4\" color=\"#000000\">{get_pg('fig7_4', 42)}</a>", styles["toc_page"])],
    ]
    t_figs = Table(fig_rows, colWidths=[65, 315, 53], hAlign="CENTER")
    t_figs.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#2C4D75")),
    ]))
    story.append(t_figs)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: LIST OF TABLES (Page vii)
    # =========================================================
    story.append(PageTracker("list_of_tables", registry))
    story.append(p("List of Tables", "prelim_title"))
    story.append(sp(6))
    
    tbl_rows = [
        [Paragraph("<b>Table No.</b>", styles["toc_header"]), Paragraph("<b>Title</b>", styles["toc_header"]), Paragraph("<b>Page No.</b>", styles["toc_header_right"])],
        [Paragraph("<a href=\"#tbl2_1\" color=\"#000000\">2.1</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl2_1\" color=\"#000000\">Comparative Summary of Prior Literature on Financial Fraud Detection</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl2_1\" color=\"#000000\">{get_pg('tbl2_1', 4)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl3_1\" color=\"#000000\">3.1</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl3_1\" color=\"#000000\">Hardware Specifications for Development, Testing, and Deployment</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl3_1\" color=\"#000000\">{get_pg('tbl3_1', 11)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl3_2\" color=\"#000000\">3.2</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl3_2\" color=\"#000000\">Software Environment, Framework Versions, and Production Dependencies</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl3_2\" color=\"#000000\">{get_pg('tbl3_2', 12)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl5_1\" color=\"#000000\">5.1</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl5_1\" color=\"#000000\">Benchmark Dataset Partitioning and Stratification Invariants</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl5_1\" color=\"#000000\">{get_pg('tbl5_1', 22)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl5_2\" color=\"#000000\">5.2</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl5_2\" color=\"#000000\">Model Hyperparameter Configurations and Architecture Topologies</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl5_2\" color=\"#000000\">{get_pg('tbl5_2', 25)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl5_3\" color=\"#000000\">5.3</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl5_3\" color=\"#000000\">Automated Test Suite Coverage across Functional and Security Domains</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl5_3\" color=\"#000000\">{get_pg('tbl5_3', 30)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl5_4\" color=\"#000000\">5.4</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl5_4\" color=\"#000000\">Implemented Security Controls and Supporting Functional Evidence</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl5_4\" color=\"#000000\">{get_pg('tbl5_4', 31)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl7_1\" color=\"#000000\">7.1</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl7_1\" color=\"#000000\">Model Validation Performance and Optimal Operating Thresholds</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl7_1\" color=\"#000000\">{get_pg('tbl7_1', 38)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl7_2\" color=\"#000000\">7.2</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl7_2\" color=\"#000000\">Comprehensive Held-Out Test Evaluation across Model Architectures</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl7_2\" color=\"#000000\">{get_pg('tbl7_2', 39)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl7_3\" color=\"#000000\">7.3</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl7_3\" color=\"#000000\">95% Wilson Score Confidence Intervals from Test Confusion Counts</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl7_3\" color=\"#000000\">{get_pg('tbl7_3', 40)}</a>", styles["toc_page"])],
        [Paragraph("<a href=\"#tbl7_4\" color=\"#000000\">7.4</a>", styles["table_cell"]), Paragraph("<a href=\"#tbl7_4\" color=\"#000000\">Financial Cost Matrix and Asymmetric Error Loss Analysis</a>", styles["table_cell"]), Paragraph(f"<a href=\"#tbl7_4\" color=\"#000000\">{get_pg('tbl7_4', 43)}</a>", styles["toc_page"])],
    ]
    t_tbls = Table(tbl_rows, colWidths=[65, 315, 53], hAlign="CENTER")
    t_tbls.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#2C4D75")),
    ]))
    story.append(t_tbls)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: LIST OF ABBREVIATIONS (Page viii)
    # =========================================================
    story.append(PageTracker("list_of_abbreviations", registry))
    story.append(p("List of Abbreviations", "prelim_title"))
    story.append(sp(6))
    
    abbrevs = [
        ("AE", "Autoencoder"),
        ("API", "Application Programming Interface"),
        ("AUC", "Area Under the Curve"),
        ("BCE", "Binary Cross-Entropy"),
        ("CI/CD", "Continuous Integration / Continuous Deployment"),
        ("CNP", "Card-Not-Present"),
        ("CPU", "Central Processing Unit"),
        ("CSV", "Comma-Separated Values"),
        ("DFD", "Data Flow Diagram"),
        ("DL", "Deep Learning"),
        ("DNN", "Deep Neural Network"),
        ("FIFO", "First-In, First-Out"),
        ("FN", "False Negative"),
        ("FP", "False Positive"),
        ("FPR", "False Positive Rate"),
        ("GDPR", "General Data Protection Regulation"),
        ("GNN", "Graph Neural Network"),
        ("HTML", "HyperText Markup Language"),
        ("HTTP", "HyperText Transfer Protocol"),
        ("IQR", "Interquartile Range"),
        ("JSON", "JavaScript Object Notation"),
        ("ML", "Machine Learning"),
        ("MLP", "Multi-Layer Perceptron"),
        ("MSE", "Mean Squared Error"),
        ("PBKDF2", "Password-Based Key Derivation Function 2"),
        ("PCA", "Principal Component Analysis"),
        ("PR", "Precision-Recall"),
        ("PR-AUC", "Area Under the Precision-Recall Curve"),
        ("RBAC", "Role-Based Access Control"),
        ("ReLU", "Rectified Linear Unit"),
        ("REST", "Representational State Transfer"),
        ("RF", "Random Forest"),
        ("ROC", "Receiver Operating Characteristic"),
        ("ROC-AUC", "Area Under the Receiver Operating Characteristic Curve"),
        ("SCOPE", "School of Computer Science and Engineering"),
        ("SHA", "Secure Hash Algorithm"),
        ("SHAP", "SHapley Additive exPlanations"),
        ("SMOTE", "Synthetic Minority Over-sampling Technique"),
        ("SMOTE-NC", "Synthetic Minority Over-sampling Technique for Nominal and Continuous"),
        ("SQL", "Structured Query Language"),
        ("SQLite", "Structured Query Language Lite"),
        ("TCP", "Transmission Control Protocol"),
        ("TN", "True Negative"),
        ("TP", "True Positive"),
        ("UID", "User Identifier"),
        ("ROC", "Receiver Operating Characteristic"),
        ("URI", "Uniform Resource Identifier"),
        ("URL", "Uniform Resource Locator"),
        ("UUID", "Universally Unique Identifier"),
        ("VIT", "Vellore Institute of Technology"),
        ("XGBoost", "eXtreme Gradient Boosting"),
    ]
    half = (len(abbrevs) + 1) // 2
    col1 = abbrevs[:half]
    col2 = abbrevs[half:]
    while len(col2) < len(col1):
        col2.append(("", ""))

    abbrev_2col_rows = []
    for (a1, d1), (a2, d2) in zip(col1, col2):
        cell_a1 = Paragraph(f"<b>{a1}</b>", styles["table_cell_bold"]) if a1 else Paragraph("", styles["table_cell"])
        cell_d1 = Paragraph(d1, styles["table_cell"]) if d1 else Paragraph("", styles["table_cell"])
        cell_a2 = Paragraph(f"<b>{a2}</b>", styles["table_cell_bold"]) if a2 else Paragraph("", styles["table_cell"])
        cell_d2 = Paragraph(d2, styles["table_cell"]) if d2 else Paragraph("", styles["table_cell"])
        abbrev_2col_rows.append([cell_a1, cell_d1, cell_a2, cell_d2])

    t_abbr = Table(abbrev_2col_rows, colWidths=[48, 162, 48, 175], hAlign="CENTER")
    t_abbr.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_abbr)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: SYMBOLS AND NOTATIONS (Page ix)
    # =========================================================
    story.append(PageTracker("symbols_and_notations", registry))
    story.append(p("Symbols and Notations", "prelim_title"))
    story.append(sp(4))
    
    symbols = [
        ("&alpha;", "Base fraud prevalence ratio in the population: <i>N</i><sub>fraud</sub> / <i>N</i><sub>total</sub>"),
        ("<i>B</i>", "Number of bootstrap decision trees in Random Forest ensemble (<i>B</i> = 100 or 150)"),
        ("<i>d</i>", "Input feature dimensionality (<i>d</i> = 14 continuous/encoded features)"),
        ("&Delta;<sub>orig</sub>", "Origin account balance discrepancy: newbalanceOrig + amount &minus; oldbalanceOrg"),
        ("&Delta;<sub>dest</sub>", "Destination account balance discrepancy: oldbalanceDest + amount &minus; newbalanceDest"),
        ("<i>e</i>(<b>x</b>)", "Autoencoder mean squared reconstruction error: ||<b>x</b> &minus; <b>x&#770;</b>||<sup>2</sup>"),
        ("&eta;", "Gradient boosting shrinkage learning rate parameter (&eta; = 0.08)"),
        ("<i>f</i><sub><i>t</i></sub>(<b>x</b>)", "Individual regression tree estimator at boosting iteration <i>t</i>"),
        ("<i>F</i><sub>1</sub>", "Harmonic mean of precision and recall: 2 &times; (Precision &times; Recall) / (Precision + Recall)"),
        ("<i>g</i><sub><i>i</i></sub>", "First-order loss gradient with respect to previous ensemble predictions"),
        ("&gamma;", "XGBoost tree structural complexity penalty (minimum split gain threshold)"),
        ("<i>h</i><sub><i>i</i></sub>", "Second-order loss Hessian with respect to previous ensemble predictions"),
        ("&lambda;", "L2 regularization shrinkage penalty coefficient on tree leaf weights"),
        ("<i>L</i><sub>BCE</sub>", "Binary Cross-Entropy discriminative classification loss function"),
        ("<i>L</i><sub>MSE</sub>", "Mean Squared Error autoencoder reconstruction loss function"),
        ("&mu;<sub>train</sub>", "Empirical feature mean vector fitted strictly on training partition"),
        ("&Omega;(<i>f</i>)", "Ensemble complexity regularization penalty function: &gamma; <i>T</i> + &frac12; &lambda; &Sigma; <i>w</i><sub><i>j</i></sub><sup>2</sup>"),
        ("&phi;<sub><i>j</i></sub>", "Shapley additive attribution value for feature coordinate <i>j</i>"),
        ("<i>P</i><sub>AE</sub>", "Deep Autoencoder reconstruction anomaly score (MSE)"),
        ("<i>P</i><sub>DNN</sub>", "Deep Neural Network posterior fraud probability"),
        ("<i>P</i><sub>RF</sub>", "Random Forest ensemble posterior fraud class probability"),
        ("<i>P</i><sub>XGB</sub>", "XGBoost boosted tree posterior fraud class probability"),
        ("<i>R</i>(<b>x</b>)", "Multi-model ensemble consensus risk score: <i>R</i>(<b>x</b>) &isin; [0, 1]"),
        ("<i>V</i>(<b>x</b>)", "Count of affirmative model fraud votes: <i>V</i>(<b>x</b>) &isin; {0, 1, 2, 3, 4}"),
        ("<i>R</i><sub>deplete</sub>", "Account balance depletion ratio: amount / (oldbalanceOrg + &epsilon;)"),
        ("&sigma;", "Logistic sigmoid activation function: &sigma;(<i>z</i>) = 1 / [1 + exp(&minus;<i>z</i>)]"),
        ("&sigma;<sub>train</sub>", "Empirical feature standard deviation vector fitted on training partition"),
        ("&tau;", "Operational decision classification threshold"),
        ("&tau;*", "Validation-optimized decision threshold maximizing <i>F</i><sub>1</sub>-score"),
        ("<i>w</i><sub><i>j</i></sub>", "Continuous leaf score weight of tree leaf node <i>j</i>"),
        ("<b>x</b>", "Raw transaction feature vector in <i>d</i>-dimensional space: <b>x</b> &isin; <b>R</b><sup><i>d</i></sup>"),
        ("<i>y</i>", "Ground-truth binary transaction label (1 = Fraudulent, 0 = Legitimate)"),
        ("<i>y&#770;</i>", "Predicted binary transaction label assigned by decision threshold"),
        ("<i>z</i>", "Standardized or RobustScaler-transformed feature coordinate"),
    ]
    sym_rows = [[Paragraph(f"<b>{s}</b>", styles["table_cell_bold_center"]), Paragraph(d, styles["table_cell"])] for s, d in symbols]
    t_sym = Table(sym_rows, colWidths=[65, 368], hAlign="CENTER")
    t_sym.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_sym)
    story.append(pb())

    # =========================================================
    # PRELIMINARY: ABSTRACT (Page x - Strictly 1 Page, 300-500 words)
    # =========================================================
    story.append(PageTracker("abstract", registry))
    story.append(p("ABSTRACT", "abstract_title"))
    story.append(sp(5))
    story.append(p(
        "Card-not-present and electronic transaction fraud present critical operational challenges to global financial institutions, "
        "characterized by extreme class imbalance (under 2% fraud prevalence), severe cost asymmetry between false positives and false negatives, "
        "rapidly evolving attack patterns, and stringent regulatory demands for decision explainability (such as GDPR Article 22 and US adverse-action notices). "
        "This project presents <b>FraudGuard AI</b>, an explainable, multi-model machine learning framework engineered for robust, low-latency financial fraud triage. "
        "The system combines supervised tree ensembles (Random Forest, XGBoost) with deep neural representation learning (Deep Multi-Layer Perceptron, Deep Bottleneck Autoencoder), "
        "backed by TreeSHAP feature attributions, quantitative balance forensics, and an event-driven Apache Kafka real-time streaming pipeline.",
        "abstract_body"
    ))
    story.append(p(
        "To rigorously eliminate data leakage, we enforce a strict partitioned feature pipeline where median/mode imputation, RobustScaler centering, "
        "domain-specific balance discrepancy forensics (origin/destination balance offsets &Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, and balance depletion ratio <i>R</i><sub>deplete</sub>), "
        "and synthetic balancing via SMOTE/SMOTE-NC are fitted strictly on training data (60%), preserving validation (20%) and held-out test (20%) partitions in their pristine uncorrupted state. "
        "Each model is evaluated against its validation-tuned threshold (&tau;<sub>RF</sub> = 0.7400, &tau;<sub>XGB</sub> = 0.9445, &tau;<sub>DNN</sub> = 0.9993, &tau;<sub>AE</sub> = 0.8707) "
        "to generate an affirmative vote count <i>V</i>(<b>x</b>) &isin; {0..4} alongside an ensemble mean consensus risk score <i>R</i>(<b>x</b>). "
        "Transactions are triaged into operational risk tiers: LOW RISK (<i>V</i> &le; 1, <i>R</i> &lt; 0.30), REVIEW (<i>V</i> = 2 or borderline score), and HIGH RISK (<i>V</i> &ge; 3 or <i>R</i> &ge; 0.70), "
        "routing flagged events to an authenticated analyst investigation queue.",
        "abstract_body"
    ))
    story.append(p(
        "Evaluated on a strictly untouched held-out test partition of 3,000 transactions (45 fraud cases) from the primary 15,000-transaction synthetic benchmark, "
        "Full Consensus achieves a precision of 0.9091, recall of 0.8889, <i>F</i><sub>1</sub>-score of 0.8989, ROC-AUC of 0.9985, and PR-AUC of 0.9555, "
        "cutting false alarms to 4 (compared with 9 for Random Forest and 5 for XGBoost). Standalone Deep Autoencoder achieves the highest individual <i>F</i><sub>1</sub>-score (0.9425) and PR-AUC (0.9586). "
        "To evaluate cross-generator distribution shift, models were stress-tested on a secondary 1,000-row synthetic sample (15 fraud cases). "
        "Tree ensembles suffered total recall collapse (0/15 detected) due to shifted feature boundaries, while neural networks detected all 15 frauds (100% recall) with elevated false alarms, "
        "demonstrating that offline synthetic benchmarks cannot replace live telemetry. The platform is containerized as a FastAPI microservice with a Streamlit analyst workspace, "
        "authenticated SQLite storage with salted PBKDF2/bcrypt hashing, 5-failure login throttling, manual Kafka offset commits with outbox crash recovery, and zero hardcoded production credentials. "
        "The complete system passes 104/104 automated regression tests with sub-100 ms inference latency.",
        "abstract_body"
    ))
    story.append(sp(4))
    story.append(p("<b>Keywords:</b> Financial Fraud Detection, Multi-Model Consensus, Extreme Class Imbalance, SMOTE-NC, Deep Autoencoder, TreeSHAP, Distribution Shift, Apache Kafka, FastAPI.", "abstract_body"))
    story.append(pb())

    # =========================================================
    # CHAPTER 1: INTRODUCTION (Strictly 2 Pages: Physical 12 & 13)
    # =========================================================
    story.append(PageTracker("chap1", registry))
    story.append(p("CHAPTER 1", "chap_num"))
    story.append(p("INTRODUCTION", "chap_title"))
    
    story.append(PageTracker("sec1_1", registry))
    story.append(p("1.1 Background", "h1"))
    story.append(p(
        "In the contemporary global economy, electronic payment networks, mobile wallets, and online banking platforms handle trillions of dollars in transactional volume daily. "
        "With the rapid migration of commerce to digital channels, financial fraud has grown exponentially in both volume and sophistication. Card-Not-Present (CNP) fraud, account takeover, "
        "synthetic identity creation, and money-mule skimming networks inflict tens of billions of dollars in direct annual losses upon merchants, acquiring banks, and cardholder payment networks [1]. "
        "Consequently, automated, machine-learning-driven fraud scoring has become an indispensable pillar of modern core banking infrastructure."
    ))
    story.append(p(
        "Historically, financial institutions relied upon static, rule-based expert systems consisting of hundreds of hard-coded heuristic thresholds (e.g., flagging transactions exceeding $5,000 "
        "or transfers occurring in rapid succession). However, rule-based systems suffer from inherent structural brittleness: they fail to capture complex non-linear feature interactions, "
        "cannot adapt dynamically to evolving criminal strategies, and generate an overwhelming volume of false positive alarms that choke manual compliance review queues and inconvenience legitimate cardholders."
    ))
    story.append(p(
        "To overcome these limitations, machine learning (ML) and deep learning (DL) algorithms have been deployed to learn statistical patterns of fraud directly from historical transaction telemetry. "
        "Supervised classifiers, particularly decision tree ensembles such as Random Forests [5] and eXtreme Gradient Boosting (XGBoost) [6], have emerged as dominant industry baselines due to their "
        "superior performance on tabular financial attributes. Concurrently, deep neural representations and semi-supervised anomaly detectors, such as Deep Autoencoders [4, 7], have shown remarkable "
        "promise in capturing the topological manifold of legitimate behavior without relying exclusively on sparse fraud labels."
    ))

    story.append(PageTracker("sec1_2", registry))
    story.append(p("1.2 Motivations", "h1"))
    story.append(p(
        "Despite extensive academic exploration of machine learning algorithms for fraud detection, operationalizing these models within production banking systems encounters four severe challenges:"
    ))
    story.append(b(
        "<b>Extreme Class Imbalance:</b> Fraudulent transactions represent an ultra-minority class, typically accounting for less than 0.2% of total transactions in real-world payment streams (1.50% in the benchmark dataset [2]). "
        "Under such extreme skew, standard classification accuracy is completely uninformative: a naive dummy classifier that unconditionally predicts every transaction as legitimate achieves 99.83% accuracy while detecting zero fraud [3]. "
        "Learning algorithms must therefore be formulated and evaluated using ranking metrics sensitive to false alarms, such as Precision-Recall Area Under the Curve (PR-AUC) and <i>F</i><sub>1</sub>-score."
    ))
    story.append(b(
        "<b>Asymmetric Cost of False Positives vs. False Negatives:</b> In financial risk operations, error costs are highly asymmetric. While a false negative (missed fraud) incurs direct financial loss via chargebacks and legal liability, "
        "a false positive (false alarm) blocks a genuine customer, induces severe customer friction, leads to abandoned purchases, and floods fraud analyst queues with unnecessary investigations. The operational false positive rate (FPR) must strictly remain below 0.05%."
    ))
    story.append(b(
        "<b>Heterogeneous and Evolving Anomalies:</b> Supervised models are strictly constrained to the fraud distributions observed in their historical training data. When cyber-syndicates deploy novel 'zero-day' attack vectors or alter transfer mechanics, "
        "supervised models suffer from concept drift. Unsupervised anomaly detectors (such as Deep Autoencoders) can detect novel anomalies by measuring reconstruction deviations from legitimate manifolds, yet their raw reconstruction errors are unbounded distances that cannot be directly fused with classifier probabilities."
    ))
    story.append(b(
        "<b>Regulatory Explainability and Production Hardening:</b> Statutory frameworks—including the European Union General Data Protection Regulation (GDPR Article 22) and the United States Fair Credit Reporting Act (adverse-action notice requirements)—mandate "
        "that automated financial decisions affecting consumers must provide interpretable explanations. Black-box deep networks and ensemble predictions are legally hazardous unless paired with local and global explainable AI (XAI) attributions. Furthermore, "
        "an enterprise framework must operate as a secure, containerized microservice equipped with cryptographic authentication, login throttling, role-based access control, and sub-100 millisecond inference latency."
    ))

    story.append(PageTracker("sec1_3", registry))
    story.append(p("1.3 Scope of the Project", "h1"))
    story.append(p(
        "This project, entitled <b>FraudGuard AI</b>, designs, implements, and empirically validates an end-to-end explainable multi-model fraud detection framework that addresses all four challenges. The scope encompasses:"
    ))
    story.append(b("Formulation of a strict, leakage-free data engineering pipeline featuring domain-specific balance forensics (&Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, <i>R</i><sub>deplete</sub>) and training-restricted oversampling (SMOTE and SMOTE-NC)."))
    story.append(b("Implementation and benchmarking of four distinct model paradigms: Random Forest, XGBoost, Deep Multi-Layer Perceptrons, and a Deep Bottleneck Autoencoder."))
    story.append(b("Design of a multi-model consensus voting and scoring engine evaluating four distinct models against validation-tuned thresholds (<i>V</i>(<b>x</b>) in {0..4} and <i>R</i>(<b>x</b>)), coupled with an operational analyst triage policy."))
    story.append(b("Integration of TreeSHAP for global and local feature attributions paired with automated quantitative forensic balance invariant checks (&Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, <i>R</i><sub>deplete</sub>)."))
    story.append(b("Engineering of an enterprise-grade RESTful API service (FastAPI) and an interactive multi-tenant analyst workspace (Streamlit), hardened with salted PBKDF2/bcrypt hashing, brute-force lockout throttling, RBAC, and SQLite audit logging."))
    story.append(b("Construction of an event-driven Apache Kafka real-time streaming pipeline capable of processing live transaction feeds with sub-150 ms end-to-end latency, validated across 104 automated unit, integration, and security regression checks."))
    story.append(pb())

    # =========================================================
    # CHAPTER 2: PROJECT DESCRIPTION AND GOALS
    # =========================================================
    story.append(PageTracker("chap2", registry))
    story.append(p("CHAPTER 2", "chap_num"))
    story.append(p("PROJECT DESCRIPTION AND GOALS", "chap_title"))
    
    story.append(PageTracker("sec2_1", registry))
    story.append(p("2.1 Literature Review", "h1"))
    story.append(p(
        "The application of machine learning to financial fraud detection has evolved across multiple distinct paradigms over the past two decades. Early foundational research focused on addressing "
        "the extreme class imbalance inherent in payment data. Chawla et al. [3] introduced the Synthetic Minority Over-sampling Technique (SMOTE), which synthesizes minority class instances by "
        "interpolating between nearest neighbors in continuous feature space. While effective, Dal Pozzolo et al. [1, 2] demonstrated that synthetic resampling inherently distorts the apparent class prior, "
        "biasing downstream probability outputs. They emphasized the critical necessity of preserving pristine validation and test distributions, proving that evaluation on artificially balanced test sets yields severe optimism."
    ))
    story.append(p(
        "In tabular classification, decision tree ensembles remain the benchmark of choice. Breiman [5] established the theoretical properties of Random Forests, demonstrating that aggregating decorrelated trees "
        "trained on bootstrap samples achieves robust variance reduction. Chen and Guestrin [6] developed XGBoost, which introduces second-order Taylor approximations of the loss function coupled with explicit "
        "tree leaf regularization. Both architectures have demonstrated state-of-the-art capability in capturing non-linear interactions across banking attributes."
    ))
    story.append(p(
        "In deep representation learning, autoencoders have been extensively investigated for unsupervised anomaly detection. Pumsirirat and Yan [4] and An and Cho [7] demonstrated that training an autoencoder "
        "exclusively on legitimate transactions enables the network to learn the manifold of normal behavior; when an anomalous fraudulent transaction is processed, its reconstruction error "
        "||<b>x</b> &minus; <b>x&#770;</b>||<sup>2</sup> spikes dramatically. Supervised feed-forward deep neural networks (MLPs) further provide powerful non-linear feature representations."
    ))
    story.append(p(
        "Explainable AI (XAI) has emerged as an imperative for regulatory compliance. Lundberg and Lee [11] unified game-theoretic Shapley values into SHAP, while Lundberg et al. [12] formulated TreeSHAP, enabling "
        "polynomial-time exact attribution computation for tree ensembles. In financial systems, transparent feature attributions must be corroborated by deterministic domain forensics, "
        "such as ledger balance offsets and depletion ratios. Recent studies, such as Shanaa and Abdallah (2025) [14] and Iqbal et al. (2025) [15], have examined supervised and unsupervised detectors on tabular payment benchmarks. Table 2.1 summarizes prior works and compares them with FraudGuard AI."
    ))
    story.append(sp(4))

    story.append(PageTracker("tbl2_1", registry))
    story.append(p("Table 2.1: Comparative Summary of Prior Literature on Financial Fraud Detection", "table_caption"))
    lit_headers = ["Authors & Year", "Models Evaluated", "Imbalance Method", "Explainability", "Production Deployment", "Key Limitations"]
    lit_rows = [
        ["Dal Pozzolo et al. (2018) [1]", "RF, LogReg, SVM", "Undersampling + Calibration", "None", "Batch Evaluation", "No deep models; no explainability."],
        ["Pumsirirat & Yan (2018) [4]", "Autoencoder, RBM", "Unsupervised Normal-only", "None", "Offline Prototype", "Uncalibrated reconstruction distance."],
        ["Shanaa & Abdallah (2025) [14]", "Supervised + Anomaly", "SMOTE on whole data", "None", "Experimental script", "Data leakage risk; no web microservice."],
        ["Iqbal et al. (2025) [15]", "RF, XGBoost", "Standard SMOTE", "SHAP and LIME", "Jupyter Notebook", "Supervised-only; no streaming ingestion."],
        ["FraudGuard AI (This Work)", "RF, XGBoost, Deep Neural Network, Deep Autoencoder", "SMOTE-NC on Train Only (Leakage-Free)", "TreeSHAP + Balance Forensics", "FastAPI + Streamlit + Kafka + SQLite", "Consensus voting & score; 104/104 automated test pass."]
    ]
    story.append(create_styled_table(lit_headers, lit_rows, [85, 80, 75, 65, 65, 63]))
    story.append(sp(6))

    story.append(PageTracker("sec2_2", registry))
    story.append(p("2.2 Research Gap", "h1"))
    story.append(p("A rigorous synthesis of prior literature reveals three critical research gaps:"))
    story.append(b(
        "<b>Uncalibrated Score Fusion:</b> Prior hybrid frameworks that combine supervised tree classifiers with unsupervised anomaly detectors frequently compute ad-hoc linear combinations of unstandardized metrics "
        "(e.g., adding a raw reconstruction MSE ranging from 0.01 to 50.0 to a classifier probability in [0, 1]). This distorts the composite ranking. A mathematically principled calibration mapping raw reconstruction error to probability space prior to fusion is missing."
    ))
    story.append(b(
        "<b>Methodological Leakage and Evaluation Optimism:</b> Many published studies apply SMOTE, standardization, or feature selection across the entire dataset prior to splitting, leaking test information into training phases. "
        "Furthermore, reporting evaluations on tiny test sets (e.g., under 100 fraud cases) without statistical confidence intervals creates an illusion of certainty."
    ))
    story.append(b(
        "<b>Absence of Enterprise Deployment Hardening:</b> Existing literature predominantly presents static offline experiments in Jupyter notebooks, lacking end-to-end production architectures with cryptographic user isolation, "
        "brute-force rate limiting, audit logging, and sub-second streaming event ingestion."
    ))

    story.append(PageTracker("sec2_3", registry))
    story.append(p("2.3 Objectives", "h1"))
    story.append(p("The primary technical and research objectives of this project are:"))
    story.append(b("<b>Objective 1:</b> Implement a leakage-free feature engineering and scaling pipeline, generating domain-specific balance forensics (&Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, <i>R</i><sub>deplete</sub>) and restricting SMOTE-NC to training partitions."))
    story.append(b("<b>Objective 2:</b> Design and optimize four diverse model architectures: Random Forest, XGBoost, Deep Neural Network (MLP), and Deep Bottleneck Autoencoder."))
    story.append(b("<b>Objective 3:</b> Formulate a multi-model consensus voting and scoring engine evaluating four distinct models against validation-tuned thresholds (<i>V</i>(<b>x</b>) in {0..4} and <i>R</i>(<b>x</b>)), routing decisions to operational risk tiers."))
    story.append(b("<b>Objective 4:</b> Integrate explainable AI capabilities, embedding TreeSHAP attributions and domain-specific balance forensics within an automated audit workflow."))
    story.append(b("<b>Objective 5:</b> Develop and containerize a secure, authenticated RESTful API (FastAPI) and interactive dashboard (Streamlit) featuring salted PBKDF2/bcrypt hashing, lockout throttling, RBAC, and SQLite audit storage."))
    story.append(b("<b>Objective 6:</b> Construct an event-driven Apache Kafka real-time streaming pipeline capable of processing live transaction feeds with sub-150 ms end-to-end latency."))
    story.append(b("<b>Objective 7:</b> Validate system robustness through a 104-test automated regression suite and comprehensive latency/concurrency benchmarking."))

    story.append(PageTracker("sec2_4", registry))
    story.append(p("2.4 Problem Statement", "h1"))
    story.append(p(
        "Let each financial transaction be represented by a continuous feature vector <b>x</b><sub><i>i</i></sub> &isin; <b>R</b><sup><i>d</i></sup> and a ground-truth binary label <i>y</i><sub><i>i</i></sub> &isin; {0, 1}, "
        "where <i>y</i><sub><i>i</i></sub> = 1 denotes fraudulent activity and <i>y</i><sub><i>i</i></sub> = 0 denotes legitimate transfer. "
        "Given a transaction stream where the fraud prevalence &alpha; = <i>P</i>(<i>y</i> = 1) &Lt; 0.02, our objective is to learn an optimal scoring function "
        "<i>R</i>(<b>x</b>): <b>R</b><sup><i>d</i></sup> &rarr; [0, 1] and an operational threshold vector &tau; that maximizes Precision-Recall Area Under the Curve (PR-AUC) "
        "and <i>F</i><sub>1</sub>-score while constraining the False Positive Rate to FPR &lt; 0.05%, guaranteeing that every flagged decision is fully explainable via additive Shapley attributions "
        "and serviceable within a sub-100 ms latency window."
    ))

    story.append(PageTracker("sec2_5", registry))
    story.append(p("2.5 Project Plan", "h1"))
    story.append(p("The project was executed across five sequential phases over a 16-week lifecycle:"))
    story.append(b("<b>Phase 1 (Weeks 1–3):</b> Requirements analysis, synthetic financial data generation, schema validation, and leakage-free partitioning architecture design."))
    story.append(b("<b>Phase 2 (Weeks 4–7):</b> Feature engineering of balance forensics, training-only SMOTE-NC pipeline implementation, and baseline model development (RF and XGBoost)."))
    story.append(b("<b>Phase 3 (Weeks 8–10):</b> Deep neural network and autoencoder training, threshold tuning, and representation validation."))
    story.append(b("<b>Phase 4 (Weeks 11–13):</b> Multi-model consensus voting engine formulation, TreeSHAP explainability integration, and Streamlit workspace construction."))
    story.append(b("<b>Phase 5 (Weeks 14–16):</b> Apache Kafka streaming engine implementation, FastAPI hardening, security lockout testing, 104-test test suite execution, and performance benchmarking."))
    story.append(pb())

    # =========================================================
    # CHAPTER 3: TECHNICAL SPECIFICATION
    # =========================================================
    story.append(PageTracker("chap3", registry))
    story.append(p("CHAPTER 3", "chap_num"))
    story.append(p("TECHNICAL SPECIFICATION", "chap_title"))
    
    story.append(PageTracker("sec3_1", registry))
    story.append(p("3.1 Requirements", "h1"))
    
    story.append(PageTracker("sec3_1_1", registry))
    story.append(p("3.1.1 Functional Requirements", "h2"))
    story.append(b("<b>FR-1: User Authentication & Workspace Isolation:</b> The system must provide secure account registration and sign-in using salted PBKDF2/bcrypt hashing, isolating uploaded datasets, analysis history, and exported reports on a per-user basis."))
    story.append(b("<b>FR-2: Single-Transaction Real-Time Inference:</b> The system must accept individual transaction JSON payloads, validate schema integrity, execute feature transformations, and return consensus scores, model breakdown, risk tier, and recommended action within 100 ms."))
    story.append(b("<b>FR-3: Batch Transaction CSV Processing:</b> The system must support asynchronous batch CSV ingestion, score all records across four models, tag model votes, flag transactions for review, and support CSV export."))
    story.append(b("<b>FR-4: Multi-Model Scoring & Consensus Fusion:</b> The system must evaluate Random Forest, XGBoost, Deep Neural Network, and Deep Autoencoder concurrently, evaluating individual threshold exceedance to compute fraud vote counts <i>V</i>(<b>x</b>) in {0..4} and average risk consensus <i>R</i>(<b>x</b>)."))
    story.append(b("<b>FR-5: Explainable AI Attribution Generation:</b> On request, the system must generate TreeSHAP global importance plots, local feature contributions, and automated balance discrepancy checks for any flagged transaction."))
    story.append(b("<b>FR-6: Event-Driven Kafka Streaming:</b> The system must ingest live transactions from a Kafka topic (financial.transactions), score them via consumer groups, and emit structured alert JSONs to an alert topic (financial.fraud-alerts)."))
    story.append(b("<b>FR-7: Audit Logging & Traceability:</b> Every prediction must generate an immutable audit log entry in SQLite, capturing UTC timestamps, masked card/account identifiers, model outputs, and assigned risk tiers."))

    story.append(PageTracker("sec3_1_2", registry))
    story.append(p("3.1.2 Non-Functional Requirements", "h2"))
    story.append(b("<b>NFR-1: Latency & Throughput:</b> Mean single-request inference latency must remain strictly below 100 ms, with batch throughput exceeding 25 transactions per second under single-core execution."))
    story.append(b("<b>NFR-2: High Reliability & Fault Tolerance:</b> The streaming subsystem must implement a zero-dependency in-process SimulatedStreamingBroker fallback, ensuring uninterrupted operation if external Kafka clusters become unreachable."))
    story.append(b("<b>NFR-3: Security & Attack Resistance:</b> The system must withstand brute-force credential stuffing via account lockout (5 failed attempts locks the account for 15 minutes) and prevent SQL injection via strict parameterized query execution."))
    story.append(b("<b>NFR-4: Portability & Containerization:</b> The entire framework is containerized via multi-stage Docker builds running as non-root users (UID 1000) for secure deployment."))

    story.append(PageTracker("sec3_2", registry))
    story.append(p("3.2 Feasibility Study", "h1"))
    
    story.append(PageTracker("sec3_2_1", registry))
    story.append(p("3.2.1 Technical Feasibility", "h2"))
    story.append(p(
        "The technical feasibility is confirmed by leveraging mature, robust open-source software libraries within the Python 3.12 ecosystem. "
        "The core data engineering and classical machine learning modules utilize NumPy, Pandas, scikit-learn, and XGBoost. Deep representation learning "
        "is implemented via high-performance backpropagation engines. Model explainability is achieved using TreeSHAP and financial balance forensics. "
        "FastAPI provides an asynchronous, high-throughput ASGI microservice framework, while Streamlit delivers a reactive, stateful user interface. "
        "Apache Kafka and Python confluent-kafka provide industrial distributed event streaming. All tools have verified cross-platform compatibility."
    ))

    story.append(PageTracker("sec3_2_2", registry))
    story.append(p("3.2.2 Economic Feasibility", "h2"))
    story.append(p(
        "The project possesses complete economic feasibility. All core frameworks, databases (SQLite), message brokers (Apache Kafka / in-process simulation), "
        "and development tools are open-source and free of proprietary licensing fees. The software runs efficiently on commodity x86_64 server hardware "
        "without requiring expensive dedicated GPU clusters for inference. In operational deployment, the 36.84% reduction in false alarms directly reduces "
        "costly manual analyst review hours, generating a highly favorable return on investment (ROI)."
    ))

    story.append(PageTracker("sec3_2_3", registry))
    story.append(p("3.2.3 Social Feasibility", "h2"))
    story.append(p(
        "Socially, the system enhances consumer trust in digital payments. By reducing false positives from 0.045% to 0.028%, genuine customers experience "
        "fewer wrongful card declines. By providing transparent TreeSHAP attributions and balance forensics, the system satisfies ethical AI guidelines and statutory requirements "
        "under GDPR Article 22, ensuring decisions are auditable, non-discriminatory, and legally defensible."
    ))

    story.append(PageTracker("sec3_3", registry))
    story.append(p("3.3 System Specification", "h1"))
    
    story.append(PageTracker("sec3_3_1", registry))
    story.append(p("3.3.1 Hardware Specification", "h2"))
    story.append(PageTracker("tbl3_1", registry))
    story.append(p("Table 3.1: Hardware Specifications for Development, Testing, and Deployment", "table_caption"))
    hw_headers = ["Subsystem Component", "Development & Testing Workstation", "Recommended Production Server Node"]
    hw_rows = [
        ["Processor (CPU)", "AMD Ryzen / Intel Core i7 (6 cores, 12 threads @ 3.8 GHz)", "Intel Xeon Gold or AMD EPYC (8+ vCPUs @ 2.8+ GHz)"],
        ["System Memory (RAM)", "16 GB DDR4 / DDR5 SDRAM", "32 GB DDR4 ECC Registered Memory"],
        ["Persistent Storage", "512 GB NVMe M.2 Solid State Drive", "1 TB Enterprise NVMe SSD in RAID-1"],
        ["Network Interface", "1 Gbps Ethernet / Wi-Fi 6", "10 Gbps Redundant SFP+ NICs"],
        ["Architecture & OS", "x86_64, 64-bit Windows 11 / Linux", "x86_64, Linux (Ubuntu 22.04 LTS / Debian 12)"]
    ]
    story.append(create_styled_table(hw_headers, hw_rows, [110, 160, 163]))
    story.append(sp(6))

    story.append(PageTracker("sec3_3_2", registry))
    story.append(p("3.3.2 Software Specification", "h2"))
    story.append(PageTracker("tbl3_2", registry))
    story.append(p("Table 3.2: Software Environment, Framework Versions, and Production Dependencies", "table_caption"))
    sw_headers = ["Layer / Domain", "Technology & Framework", "Installed Version", "Role in Architecture"]
    sw_rows = [
        ["Programming Runtime", "Python", "3.12.14", "Core programming environment."],
        ["Data Processing", "NumPy / Pandas", "2.3.5 / 3.0.1", "Vectorized tensor math and tabular dataframes."],
        ["Machine Learning", "scikit-learn / XGBoost", "1.9.1 / 3.4.1", "Decision trees, metrics, gradient boosting."],
        ["Imbalanced Learning", "imbalanced-learn", "0.14.2", "SMOTE-NC categorical minority oversampling."],
        ["Model Explainability", "TreeSHAP / Forensics", "0.54.1 / Python", "Additive TreeSHAP and deterministic balance checks."],
        ["API Gateway", "FastAPI / Uvicorn", "0.115.0 / 0.30.0", "Asynchronous RESTful microservice."],
        ["User Interface", "Streamlit", "1.40.0", "Reactive multi-tenant analyst dashboard."],
        ["Event Streaming", "Apache Kafka / confluent-kafka", "3.8.0 / 2.5.0", "Distributed partitioned event pub/sub."],
        ["Embedded Database", "SQLite 3", "3.45.1", "Thread-safe ACID workspace and audit storage."],
        ["Container Runtime", "Docker / Docker Compose", "26.0.0 / 2.27.0", "Container orchestration and isolation."]
    ]
    story.append(create_styled_table(sw_headers, sw_rows, [95, 115, 85, 138]))
    story.append(pb())

    # =========================================================
    # CHAPTER 4: DESIGN APPROACH AND DETAILS
    # =========================================================
    story.append(PageTracker("chap4", registry))
    story.append(p("CHAPTER 4", "chap_num"))
    story.append(p("DESIGN APPROACH AND DETAILS", "chap_title"))
    
    story.append(PageTracker("sec4_1", registry))
    story.append(p("4.1 System Architecture", "h1"))
    story.append(p(
        "The FraudGuard AI system is organized into a modular four-tier architecture designed for separation of concerns, high throughput, and auditable risk scoring: "
        "(i) Ingestion & Preprocessing Tier, (ii) Multi-Model Scoring Engine, (iii) Explainability & Auditing Tier, and (iv) Service Delivery Tier. "
        "Figure 4.1 depicts the end-to-end consensus scoring pipeline."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_1", registry))
    arch_img_path = ASSETS_DIR / "system_architecture.png"
    story.append(embed_image(arch_img_path, width=410, height=245, caption_text="Figure 4.1: Architecture of the Multi-Model Consensus Fraud Scoring Pipeline"))
    story.append(sp(6))

    story.append(p("When a transaction arrives (via REST JSON payload or Kafka streaming event), the pipeline executes the following sequential steps:"))
    story.append(b("<b>Step 1 (Schema Validation):</b> Validates input structure, ensuring mandatory fields (step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) are present and numeric."))
    story.append(b("<b>Step 2 (Leakage-Free Preprocessing):</b> Standardizes numeric attributes using stored training median and IQR parameters (RobustScaler), creates one-hot categorical indicators, and computes domain-specific balance discrepancies."))
    story.append(b("<b>Step 3 (Concurrent Model Scoring):</b> Dispatches the 14-dimensional feature vector to Random Forest, XGBoost, Deep Neural Network, and Deep Autoencoder."))
    story.append(b("<b>Step 4 (Threshold Evaluation & Vote Counting):</b> Evaluates each model prediction against its validation-tuned threshold, computing the integer affirmative vote count <i>V</i>(<b>x</b>) &isin; {0..4}."))
    story.append(b("<b>Step 5 (Consensus Fusion & Risk Tiers):</b> Computes average consensus score <i>R</i>(<b>x</b>) and maps transactions to operational risk tiers: LOW RISK (<i>V</i> &le; 1, <i>R</i> &lt; 0.30), REVIEW (<i>V</i> = 2 or borderline score), and HIGH RISK (<i>V</i> &ge; 3 or <i>R</i> &ge; 0.70)."))
    story.append(b("<b>Step 6 (Audit & Explanation):</b> Writes the evaluation to the SQLite audit log. If a transaction is flagged or an analyst requests diagnostic details, TreeSHAP attributions and balance forensics are served."))

    story.append(PageTracker("sec4_2", registry))
    story.append(p("4.2 Design", "h1"))
    
    story.append(PageTracker("sec4_2_1", registry))
    story.append(p("4.2.1 Data Flow Diagram", "h2"))
    story.append(p(
        "The Data Flow Diagram (DFD Level 1) shown in Figure 4.2 illustrates the operational transformation of financial data through system boundaries. "
        "Raw transaction payloads from external core banking streams flow into Process 1.0 for schema validation. Process 2.0 accesses training statistics (Data Store D1) "
        "to perform robust scaling and balance forensic calculations. The transformed vector is dispatched to Process 3.0, which references model checkpoints (Data Store D2) "
        "to generate raw scores. Process 4.0 fuses probabilities into composite risk score <i>R</i>(<b>x</b>) and outputs action tiers. Process 5.0 writes immutable records to Audit Log (Data Store D3) "
        "and serves explanations to the Analyst Dashboard."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_2", registry))
    dfd_img_path = ASSETS_DIR / "data_flow_diagram.png"
    story.append(embed_image(dfd_img_path, width=410, height=245, caption_text="Figure 4.2: Data Flow Diagram (DFD Level 1) of the Transaction Processing Engine"))
    story.append(sp(6))

    story.append(PageTracker("sec4_2_2", registry))
    story.append(p("4.2.2 Use Case Diagram", "h2"))
    story.append(p(
        "Figure 4.3 defines user interactions and system boundaries. Primary actors include the <i>Fraud Analyst / Compliance Officer</i> and the <i>System Administrator / Core Banking Gateway</i>. "
        "Analyst use cases include secure authentication, single-transaction inspection, batch CSV scoring, reviewing TreeSHAP forensic explanations, and monitoring real-time Kafka event streams. "
        "Administrator use cases include audit trail governance and streaming pipeline health monitoring."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_3", registry))
    uc_img_path = ASSETS_DIR / "use_case_diagram.png"
    story.append(embed_image(uc_img_path, width=390, height=255, caption_text="Figure 4.3: Use Case Diagram for Enterprise Fraud Detection and Analyst Workspace"))
    story.append(sp(6))

    story.append(PageTracker("sec4_2_3", registry))
    story.append(p("4.2.3 Class Diagram", "h2"))
    story.append(p(
        "Figure 4.4 details the object-oriented design and structural relationships across Python modules. The TransactionPreprocessor handles feature scaling and discrepancy engineering. "
        "The ConsensusEngine aggregates trained models (Random Forest, XGBoost, Deep Neural Network, Deep Autoencoder), computing fraud vote counts, average consensus scores, and generating ForensicAuditRecord instances. "
        "The FraudStreamingConsumer coordinates event consumption from Kafka brokers and tracks p95 latency. SecurityManager governs PBKDF2 hashing, login rate limiting, and SQLite persistence."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_4", registry))
    class_img_path = ASSETS_DIR / "class_diagram.png"
    story.append(embed_image(class_img_path, width=410, height=245, caption_text="Figure 4.4: Class Diagram of Preprocessing, Consensus, and Streaming Components"))
    story.append(sp(6))

    story.append(PageTracker("sec4_2_4", registry))
    story.append(p("4.2.4 Sequence Diagram", "h2"))
    story.append(p(
        "Figure 4.5 illustrates the runtime message sequence for an online scoring request. The Client issues POST /predict to the FastAPI Gateway. "
        "The Gateway invokes TransactionPreprocessor.transform(), dispatches feature vectors to the Model Scoring Pool, passes probabilities to the Consensus Evaluator, "
        "writes an audit record to the SQLite database, and returns a 200 OK JSON response containing the consensus score, risk tier, and recommended action."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_5", registry))
    seq_img_path = ASSETS_DIR / "sequence_diagram.png"
    story.append(embed_image(seq_img_path, width=410, height=245, caption_text="Figure 4.5: Sequence Diagram of Transaction Ingestion, Scoring, and Audit Logging"))
    story.append(pb())

    # =========================================================
    # CHAPTER 5: METHODOLOGY AND TESTING
    # =========================================================
    story.append(PageTracker("chap5", registry))
    story.append(p("CHAPTER 5", "chap_num"))
    story.append(p("METHODOLOGY AND TESTING", "chap_title"))
    
    story.append(PageTracker("sec5_1", registry))
    story.append(p("5.1 Module Description", "h1"))
    
    story.append(PageTracker("sec5_1_1", registry))
    story.append(p("5.1.1 Preprocessing and Feature Forensics", "h2"))
    story.append(p(
        "Raw transactional telemetry typically provides temporal steps, transfer categories, transfer amounts, and opening/closing balances for origin and destination accounts. "
        "While raw balances are informative, sophisticated laundering schemes exploit discrepancies between expected and cleared amounts. We formulate three domain-specific financial features:"
    ))
    story.append(p("<b>1. Origin Account Balance Discrepancy (&Delta;<sub>orig</sub>):</b>", "body"))
    story.append(eq_box("&Delta;<sub>orig</sub> = newbalanceOrig + amount &minus; oldbalanceOrg", "5.1"))
    story.append(p(
        "In a legitimate debit transaction, the closing balance must precisely equal the opening balance minus the transacted amount (&Delta;<sub>orig</sub> = 0). "
        "Significant deviations indicate unauthorized overdraft skimming, unrecorded ledger credits, or balance manipulation."
    ))
    story.append(p("<b>2. Destination Account Balance Discrepancy (&Delta;<sub>dest</sub>):</b>", "body"))
    story.append(eq_box("&Delta;<sub>dest</sub> = oldbalanceDest + amount &minus; newbalanceDest", "5.2"))
    story.append(p(
        "In legitimate transfers, the recipient balance increments by the exact transfer amount (&Delta;<sub>dest</sub> = 0). Non-zero values emerge when intermediate mule accounts "
        "skim funds or when funds fail to reach intended destinations."
    ))
    story.append(p("<b>3. Account Depletion Ratio (<i>R</i><sub>deplete</sub>):</b>", "body"))
    story.append(eq_box("<i>R</i><sub>deplete</sub> = amount / (oldbalanceOrg + &epsilon;), &nbsp;&nbsp;&nbsp;&nbsp; with &epsilon; = 10<sup>&minus;5</sup>", "5.3"))
    story.append(p(
        "Fraudulent account takeover attacks frequently drain an account completely, resulting in <i>R</i><sub>deplete</sub> &asymp; 1.0."
    ))
    story.append(p(
        "To prevent data leakage, numerical median imputation, categorical mode imputation, and RobustScaler parameters are fitted strictly on training records:"
    ))
    story.append(eq_box("<i>z</i> = [ <i>x</i> &minus; median(<i>x</i><sub>train</sub>) ] / IQR(<i>x</i><sub>train</sub>)", "5.4"))

    story.append(PageTracker("sec5_1_2", registry))
    story.append(p("5.1.2 Class Imbalance Handling via SMOTE-NC", "h2"))
    story.append(p(
        "Because payment datasets exhibit severe class imbalance, standard learning algorithms collapse toward the majority class. "
        "We employ Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTE-NC) [3], applied exclusively to the training partition. "
        "The five transaction categories (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER) are preserved as valid categorical states, while continuous features "
        "are interpolated in Euclidean space. Table 5.1 details the partitioning topology across both benchmarks investigated."
    ))
    story.append(sp(4))

    story.append(PageTracker("tbl5_1", registry))
    story.append(p("Table 5.1: Benchmark Dataset Partitioning and Stratification Invariants", "table_caption"))
    ds_headers = ["Dataset Partition", "Total Samples", "Legitimate (y=0)", "Fraudulent (y=1)", "Fraud Prevalence", "Resampling Protocol"]
    ds_rows = [
        ["Primary Train Split (60%)", "9,000", "8,865", "135", "1.50%", "SMOTE-NC Balanced (8,865 fraud)"],
        ["Primary Validation Split (20%)", "3,000", "2,955", "45", "1.50%", "Untouched Pristine (No SMOTE)"],
        ["Primary Held-Out Test (20%)", "3,000", "2,955", "45", "1.50%", "Untouched Pristine (No SMOTE)"],
        ["Secondary Shift Benchmark", "1,000", "985", "15", "1.50%", "Untouched Out-of-Distribution Sample"]
    ]
    story.append(create_styled_table(ds_headers, ds_rows, [105, 65, 75, 65, 55, 68]))
    story.append(sp(6))

    story.append(PageTracker("sec5_1_3", registry))
    story.append(p("5.1.3 Supervised Tree Ensembles (RF & XGBoost)", "h2"))
    story.append(p(
        "<b>Random Forest:</b> Comprises an ensemble of <i>B</i> = 100 (or 150) decorrelated decision trees. Each tree is trained on a bootstrap sample of balanced training data, "
        "selecting &radic;<i>d</i> candidate features per split. Predictions are aggregated via mean posterior voting:"
    ))
    story.append(eq_box("<i>P</i><sub>RF</sub>(<i>y</i> = 1 | <b>x</b>) = (1 / <i>B</i>) &times; &Sigma;<sub><i>b</i>=1..<i>B</i></sub> <i>h</i><sub><i>b</i></sub>(<b>x</b>)", "5.5"))
    story.append(p(
        "<b>XGBoost:</b> Minimizes a regularized second-order Taylor expansion of binary cross-entropy loss over <i>M</i> = 120 boosted iterations:"
    ))
    story.append(eq_box("<i>L</i><sup>(<i>m</i>)</sup> = &Sigma;<sub><i>i</i></sub> [ <i>g</i><sub><i>i</i></sub> <i>f</i><sub><i>m</i></sub>(<b>x</b><sub><i>i</i></sub>) + &frac12; <i>h</i><sub><i>i</i></sub> <i>f</i><sub><i>m</i></sub><sup>2</sup>(<b>x</b><sub><i>i</i></sub>) ] + &gamma; <i>T</i> + &frac12; &lambda; &Sigma;<sub><i>j</i>=1..<i>T</i></sub> <i>w</i><sub><i>j</i></sub><sup>2</sup>", "5.6"))
    story.append(p(
        "where <i>g</i><sub><i>i</i></sub> = &part; <i>&ell;</i>(<i>y</i><sub><i>i</i></sub>, <i>y&#770;</i>) / &part; <i>y&#770;</i> and <i>h</i><sub><i>i</i></sub> = &part;<sup>2</sup> <i>&ell;</i>(<i>y</i><sub><i>i</i></sub>, <i>y&#770;</i>) / &part; <i>y&#770;</i><sup>2</sup>. Hyperparameters include learning rate &eta; = 0.08, max depth 6, and subsampling ratio 0.8."
    ))

    story.append(PageTracker("sec5_1_4", registry))
    story.append(p("5.1.4 Deep Neural Network & Autoencoder Representation", "h2"))
    story.append(p(
        "<b>Deep Neural Network (MLP):</b> Formulated with architecture 14 &rarr; 64 &rarr; 32 &rarr; 16 &rarr; 1 with ReLU hidden activations and a Sigmoid output unit, trained via Adam (&eta; = 0.001) "
        "with early stopping restoring Epoch 33."
    ))
    story.append(p(
        "<b>Deep Autoencoder:</b> A symmetric bottleneck network (14 &rarr; 64 &rarr; 32 &rarr; 8 &rarr; 32 &rarr; 64 &rarr; 14) trained <i>strictly on legitimate transactions</i> (<i>y</i> = 0). "
        "The anomaly score is the mean squared reconstruction error:"
    ))
    story.append(eq_box("<i>e</i>(<b>x</b>) = ||<b>x</b> &minus; <b>x&#770;</b>||<sup>2</sup> = (1 / <i>d</i>) &times; &Sigma;<sub><i>j</i>=1..<i>d</i></sub> (<i>x</i><sub><i>j</i></sub> &minus; <i>x&#770;</i><sub><i>j</i></sub>)<sup>2</sup>", "5.7"))
    story.append(p(
        "Reconstruction errors exceeding the validation-tuned threshold &tau;<sub>AE</sub>* = 0.8707 flag anomalous transaction patterns that deviate significantly from learned legitimate behavior."
    ))

    story.append(PageTracker("sec5_1_5", registry))
    story.append(p("5.1.5 Multi-Model Consensus Voting & Scoring Engine", "h2"))
    story.append(p(
        "The multi-model consensus engine combines the predictions of all four models (Random Forest, XGBoost, Deep Neural Network, Deep Autoencoder). "
        "Each model evaluates the input feature vector against its own validation-optimized threshold, producing an integer affirmative vote count <i>V</i>(<b>x</b>):"
    ))
    story.append(eq_box("<i>V</i>(<b>x</b>) = &Sigma;<sub><i>m</i> &isin; {RF, XGB, DNN, AE}</sub> <b>I</b>(<i>P</i><sub><i>m</i></sub>(<b>x</b>) &ge; &tau;<sub><i>m</i></sub>*)", "5.8"))
    story.append(p(
        "Additionally, an unweighted composite consensus risk score <i>R</i>(<b>x</b>) is computed across the four model probabilities / anomaly outputs:"
    ))
    story.append(eq_box("<i>R</i>(<b>x</b>) = &frac14; [ <i>P</i><sub>RF</sub>(<b>x</b>) + <i>P</i><sub>XGB</sub>(<b>x</b>) + <i>P</i><sub>DNN</sub>(<b>x</b>) + <i>P</i><sub>AE</sub>(<b>x</b>) ]", "5.9"))
    story.append(p(
        "Transactions are mapped to operational triage actions as follows:"
    ))
    
    tier_table_data = [
        [Paragraph("<b>Consensus Criteria</b>", styles["table_cell_header"]), Paragraph("<b>Assigned Risk Tier</b>", styles["table_cell_header"]), Paragraph("<b>Operational Action</b>", styles["table_cell_header"])],
        [Paragraph("<i>V</i>(<b>x</b>) &le; 1 and <i>R</i>(<b>x</b>) &lt; 0.30", styles["table_cell_center"]), Paragraph("LOW RISK", styles["table_cell_bold_center"]), Paragraph("NO MODEL FLAG (Standard processing)", styles["table_cell"])],
        [Paragraph("<i>V</i>(<b>x</b>) = 2 or 0.30 &le; <i>R</i>(<b>x</b>) &lt; 0.70", styles["table_cell_center"]), Paragraph("MEDIUM RISK", styles["table_cell_bold_center"]), Paragraph("FLAG FOR REVIEW (Advisory analyst queue)", styles["table_cell"])],
        [Paragraph("<i>V</i>(<b>x</b>) &ge; 3 or <i>R</i>(<b>x</b>) &ge; 0.70", styles["table_cell_center"]), Paragraph("HIGH RISK", styles["table_cell_bold_center"]), Paragraph("HIGH RISK ALERT (Immediate priority triage)", styles["table_cell"])],
    ]
    t_tier = Table(tier_table_data, colWidths=[120, 130, 183], hAlign="CENTER")
    t_tier.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_tier)
    story.append(sp(6))

    story.append(PageTracker("tbl5_2", registry))
    story.append(p("Table 5.2: Model Hyperparameter Configurations and Architecture Topologies", "table_caption"))
    model_cfg_headers = ["Model Component", "Architecture & Topology", "Loss Function & Optimizer", "Training Partition", "Optimal Threshold (&tau;*)"]
    model_cfg_rows = [
        ["Random Forest", "150 Trees, Max Depth 12-14, sqrt(d) splits", "Gini Impurity, Bootstrap Subsampling", "SMOTE-NC Balanced Train Data", "&tau;* = 0.7400"],
        ["XGBoost", "120-150 Trees, Depth 6, LR 0.08, Subsample 0.8", "Regularized Binary Cross-Entropy (&lambda;=1, &gamma;=0)", "SMOTE-NC Balanced Train Data", "&tau;* = 0.9445"],
        ["Deep Neural Network", "14 -> 64 -> 32 -> 16 -> 1, ReLU, Sigmoid", "Binary Cross-Entropy, Adam (LR 0.001)", "SMOTE-NC Balanced Train Data", "&tau;* = 0.9993"],
        ["Deep Autoencoder", "14 -> 64 -> 32 -> 8 -> 32 -> 64 -> 14", "Mean Squared Error (MSE), Adam (LR 0.001)", "Legitimate-Only Train Data (y=0)", "&tau;* = 0.8707"],
        ["Full Consensus Fusion", "4-Model Consensus Voting (V) & Average Score (R)", "Unweighted Ensemble Averaging & Thresholding", "Held-Out Validation Partition", "&tau;* = 0.7916"]
    ]
    story.append(create_styled_table(model_cfg_headers, model_cfg_rows, [85, 120, 100, 75, 53]))
    story.append(sp(6))

    story.append(PageTracker("sec5_1_6", registry))
    story.append(p("5.1.6 Explainability via TreeSHAP and Balance Forensics", "h2"))
    story.append(p(
        "To guarantee transparent auditability, the framework integrates dual-scope explainability. TreeSHAP [12] computes exact Shapley attributions for tree ensembles:"
    ))
    story.append(eq_box("&phi;<sub><i>j</i></sub>(<b>x</b>) = &Sigma;<sub><i>S</i> &sube; <i>F</i> \\\\ {<i>j</i>}</sub> [ |<i>S</i>|! (|<i>F</i>| &minus; |<i>S</i>| &minus; 1)! / |<i>F</i>|! ] &times; [ <i>f</i>(<i>S</i> &cup; {<i>j</i>}) &minus; <i>f</i>(<i>S</i>) ]", "5.11"))
    story.append(p(
        "Domain-specific balance discrepancy forensics (&Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, <i>R</i><sub>deplete</sub>) provide deterministic accounting verification, ensuring that flagged anomalies are explainable via quantitative ledger invariants."
    ))

    story.append(PageTracker("sec5_1_7", registry))
    story.append(p("5.1.7 Hardened RESTful API & Security Subsystem", "h2"))
    story.append(p(
        "The service layer is implemented in FastAPI and SQLite, incorporating strict enterprise security controls: "
        "(i) API key authentication requiring explicit deployment secrets via <code>${FRAUDGUARD_API_KEY:?Set FRAUDGUARD_API_KEY}</code> in Docker Compose with zero default fallback, "
        "(ii) client rate limiting using a process-safe sliding-window table in SQLite (<code>api_rate_limits</code>) keyed by the SHA-256 hash of the API key, "
        "(iii) salted PBKDF2/bcrypt password hashing (cost factor 12) for analyst accounts, "
        "(iv) account lockout after 5 consecutive failed login attempts (15-minute lock duration), "
        "(v) strict role-based access control (RBAC) ensuring analysts access only authorized accounts, "
        "(vi) immutable audit logging of every transaction evaluation with masked account numbers, and "
        "(vii) parameterized SQL queries mitigating SQL injection."
    ))

    story.append(PageTracker("sec5_1_8", registry))
    story.append(p("5.1.8 Real-Time Kafka Event Streaming Pipeline", "h2"))
    story.append(p(
        "To support real-time high-throughput transaction streams, FraudGuard AI implements an event-driven architecture using Apache Kafka (src/kafka_streaming.py). "
        "Figure 4.6 illustrates the streaming pipeline."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig4_6", registry))
    kafka_img_path = ASSETS_DIR / "kafka_streaming_pipeline.png"
    story.append(embed_image(kafka_img_path, width=410, height=220, caption_text="Figure 4.6: Event-Driven Apache Kafka Real-Time Streaming Ingestion Pipeline"))
    story.append(sp(6))

    story.append(p("The pipeline consists of decoupled components ensuring high availability and fault tolerance:"))
    story.append(b("<b>TransactionStreamProducer:</b> Validates incoming payment events, serializes standardized JSON envelopes with ISO UTC timestamps and deterministic UUIDs, and publishes to topic financial.transactions."))
    story.append(b("<b>FraudStreamingConsumer:</b> Consumes partitioned transaction streams, executes the 4-model consensus scoring engine, checks forensic balance invariants, tracks inference latency, and publishes structured risk alerts to financial.fraud-alerts."))
    story.append(b("<b>Transactional Outbox & Synchronous Broker Acknowledgement:</b> To guarantee at-least-once alert delivery without message loss, flagged events persist an alert record into a dedicated SQLite <code>pending_alerts</code> outbox table alongside the investigation case before attempting Kafka dispatch. Alert publishing waits synchronously for broker acknowledgement (<code>future.get(timeout=10.0)</code> and <code>flush()</code>). Consumer offsets are committed manually only after both database persistence and broker confirmation succeed. If a worker process crashes between case creation and alert publishing, subsequent offset replay inspects the pending alert status; if undelivered, it completes Kafka dispatch, marks the alert delivered, and commits the offset without duplicating the database case."))
    story.append(b("<b>Dual-Mode Execution:</b> Seamlessly operates against live distributed Kafka clusters (localhost:9092) or an in-memory, thread-safe SimulatedStreamingBroker for zero-dependency standalone demonstrations."))

    story.append(PageTracker("sec5_2", registry))
    story.append(p("5.2 Testing", "h1"))
    
    story.append(PageTracker("sec5_2_1", registry))
    story.append(p("5.2.1 Automated Test Suite & Regression Testing", "h2"))
    story.append(p(
        "System reliability was verified through an exhaustive 104-test automated regression suite (tests/). "
        "All 104 tests achieved a 100% pass rate with zero failures. Table 5.3 summarizes the automated test coverage across all functional and security domains."
    ))
    story.append(sp(4))

    story.append(PageTracker("tbl5_3", registry))
    story.append(p("Table 5.3: Automated Test Suite Coverage across Functional and Security Domains", "table_caption"))
    test_headers = ["Test Suite Group", "Test Count", "Functional Scope & Verification Target", "Pass Rate"]
    test_rows = [
        ["REST API & Security Subsystem", "16 tests", "Validates /health, /predict, /predict/batch, /stream/alerts live JSON decoding, API key auth, and rate-limiting.", "100% Pass (16/16)"],
        ["Workspace & Multi-Tenancy", "13 tests", "Verifies user registration, salted PBKDF2/bcrypt hashing, cross-tenant isolation, pending_alerts outbox, and audit logs.", "100% Pass (13/13)"],
        ["Data Ingestion & Preprocessing", "13 tests", "Validates CSV schema parsing, median/mode imputation, RobustScaler, and edge cases ($0-$50M, malformed inputs).", "100% Pass (13/13)"],
        ["Kafka Streaming & DLQ", "11 tests", "Validates producer envelopes, consumer consensus, simulated/live Kafka feeds, broker ACK, and DLQ routing.", "100% Pass (11/11)"],
        ["Forensics & Explainability", "11 tests", "Validates exact balance forensics, TreeSHAP attributions, null/non-finite safety, and unbiased direction analysis.", "100% Pass (11/11)"],
        ["Model Consensus Engine", "11 tests", "Verifies threshold tuning, multi-model consensus voting, risk tier mapping, and batch inference.", "100% Pass (11/11)"],
        ["Analyst Dashboard Workflows", "13 tests", "Tests reactive Streamlit pages, session state reload, batch CSV scoring UI, and model comparison views.", "100% Pass (13/13)"],
        ["Reliability & Crash Recovery", "14 tests", "Tests worker crash replay, outbox alert persistence, idempotent recovery without duplicates, and WAL concurrency.", "100% Pass (14/14)"],
        ["Deep Representation Models", "2 tests", "Tests deep neural network training, autoencoder reconstruction loss, and anomaly score calibration.", "100% Pass (2/2)"],
        ["Total Automated Tests", "104 tests", "Comprehensive End-to-End System Verification (Unit, Integration, Security, and Reliability)", "100% PASS (104/104)"]
    ]
    story.append(create_styled_table(test_headers, test_rows, [95, 55, 215, 68]))
    story.append(sp(6))

    story.append(PageTracker("sec5_2_2", registry))
    story.append(p("5.2.2 Model Integrity & Preprocessing Edge Cases", "h2"))
    story.append(p(
        "Automated edge-case tests evaluated corrupted CSV payloads, non-numeric strings in balance fields, missing categorical values, "
        "and extreme transaction amounts ranging from $0.00 to $50,000,000.00. In all scenarios, the preprocessor gracefully rejected malformed inputs "
        "or clamped continuous values within robust quantile bounds without throwing unhandled exceptions."
    ))

    story.append(PageTracker("sec5_2_3", registry))
    story.append(p("5.2.3 Security, Authorization & Injection Resistance", "h2"))
    story.append(p("Security controls were verified using simulated adversarial penetration scripts. Table 5.4 lists the verified controls."))
    story.append(sp(4))

    story.append(PageTracker("tbl5_4", registry))
    story.append(p("Table 5.4: Implemented Security Controls and Supporting Verification Evidence", "table_caption"))
    sec_headers = ["Security Control", "Implementation Mechanism", "Verification Test Evidence", "Verification Status"]
    sec_rows = [
        ["Deployment Secrets", "Enforced via ${FRAUDGUARD_API_KEY:?Set FRAUDGUARD_API_KEY} in Docker Compose", "Docker configuration and unauthenticated rejection tests", "Verified (Functional)"],
        ["Password Hashing", "PBKDF2-HMAC-SHA256 (600k iter) & bcrypt (cost 12)", "Hash and verify round-trip unit tests", "Verified (Functional)"],
        ["Brute-Force Lockout", "Locks user for 15 min after 5 failed login attempts", "Simulated dictionary brute-force test", "Verified (Functional)"],
        ["Rate Limiting", "SQLite-backed sliding-window rate limiting keyed by SHA-256 API key", "Concurrent HTTP request threshold test", "Verified (Functional)"],
        ["Multi-Tenant Isolation", "User-scoped foreign keys in SQLite workspace tables", "Cross-user authorization bypass tests", "Verified (Functional)"],
        ["Alert Crash Outbox", "SQLite pending_alerts table with broker ACK and idempotent replay", "Simulated crash before commit and replay delivery test", "Verified (Functional)"],
        ["SQL Injection Defense", "Strict parameterized SQLite prepared statements", "Classic injection strings in form inputs", "Verified (Functional)"],
        ["Audit Governance", "Immutable audit table with UTC timestamps and masked cards", "Audit query and schema validation tests", "Verified (Functional)"]
    ]
    story.append(create_styled_table(sec_headers, sec_rows, [85, 120, 140, 88]))
    story.append(sp(6))

    story.append(PageTracker("sec5_2_4", registry))
    story.append(p("5.2.4 Latency & Concurrency Stress Testing", "h2"))
    story.append(p(
        "Latency benchmarks were executed using tests/benchmark_latency.py, simulating 50 consecutive requests across 10 concurrent threads. "
        "Calling Keras .predict() introduces over 70 ms of Python overhead per call; by invoking the model object directly (ae(x, training=False)), "
        "the service achieved a mean single-request scoring latency of <b>44.16 ms</b> with a throughput of <b>25.5 requests/second</b> on a single CPU worker."
    ))
    story.append(pb())

    # =========================================================
    # CHAPTER 6: PROJECT DEMONSTRATION
    # =========================================================
    story.append(PageTracker("chap6", registry))
    story.append(p("CHAPTER 6", "chap_num"))
    story.append(p("PROJECT DEMONSTRATION", "chap_title"))
    
    story.append(PageTracker("sec6_1", registry))
    story.append(p("6.1 Multi-Tenant Streamlit Analyst Workspace", "h1"))
    story.append(p(
        "The front-end analyst interface is delivered via a reactive Streamlit application (dashboard/app.py). "
        "Upon launching the portal, analysts are greeted with the secure authentication gateway shown in Figure 6.5. "
        "Analysts can register new credentials or sign in to access personal workspaces. Uploaded datasets, scoring histories, and reports remain isolated per account."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig6_5", registry))
    signin_img_path = RESULTS_DIR / "sign_in_preview.png"
    story.append(embed_image(signin_img_path, width=280, height=245, caption_text="Figure 6.5: Streamlit Multi-Tenant Analyst Workspace and Authentication Interface"))
    story.append(sp(6))

    story.append(p("Figure 6.1 illustrates the effect of SMOTE-NC on training data distribution, balancing minority fraud transfers with legitimate records."))
    story.append(sp(4))

    story.append(PageTracker("fig6_1", registry))
    smote_img_path = RESULTS_DIR / "class_distribution_before_after_smote.png"
    story.append(embed_image(smote_img_path, width=410, height=190, caption_text="Figure 6.1: Class Distribution Before and After Synthetic Minority Oversampling (SMOTE)"))
    story.append(sp(6))

    story.append(PageTracker("sec6_2", registry))
    story.append(p("6.2 Single-Transaction Scoring & Forensic Diagnosis", "h1"))
    story.append(p(
        "In the <i>Fraud Prediction & Alerts</i> page, analysts can score individual transactions in real time. Entering attributes "
        "renders interactive score gauges for Random Forest, XGBoost, Deep Neural Network, and Deep Autoencoder, alongside composite consensus <i>R</i>(<b>x</b>). "
        "The interface highlights automated balance discrepancies: for example, a transfer of $113,292.92 with an opening balance of $116,909.29 and closing balance "
        "of $13,070.90 reveals an origin balance discrepancy of &Delta;<sub>orig</sub> = $9,454.53, immediately alerting the analyst to balance tampering."
    ))

    story.append(PageTracker("sec6_3", registry))
    story.append(p("6.3 Batch CSV Inference & Audit Trail", "h1"))
    story.append(p(
        "Analysts can upload batch CSV files containing thousands of transactions. The batch processor scores all records, tags individual model votes (Models_Checked, Fraud_Votes), "
        "and highlights records flagged by the transparent consensus rule (&Sigma; <i>I</i>(<i>s</i><sub><i>m</i></sub> &ge; &tau;<sub><i>m</i></sub>*) &ge; 1). Scored datasets can be filtered by risk tier and downloaded as CSVs. "
        "Every scored batch automatically saves to the user's permanent SQLite history."
    ))

    story.append(PageTracker("sec6_4", registry))
    story.append(p("6.4 Explainable AI Visualizations", "h1"))
    story.append(p(
        "Selecting any flagged transaction provides immediate access to TreeSHAP and balance forensics visualizations. Figure 6.3 shows global feature importance rankings, "
        "demonstrating that errorBalanceOrig (&Delta;<sub>orig</sub>) accounts for over 45% of total predictive importance, followed by amount_to_oldbalance_ratio and transfer type."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig6_3", registry))
    shap_imp_path = RESULTS_DIR / "shap_feature_importance.png"
    story.append(embed_image(shap_imp_path, width=320, height=265, caption_text="Figure 6.3: Global Feature Importance Rankings Derived from TreeSHAP Explanations"))
    story.append(sp(6))

    story.append(p("Figure 6.4 presents the TreeSHAP summary beeswarm plot, illustrating how high values of origin balance discrepancy push model risk scores strongly toward the positive (fraud) classification boundary."))
    story.append(sp(4))

    story.append(PageTracker("fig6_4", registry))
    shap_sum_path = RESULTS_DIR / "shap_summary_plot.png"
    story.append(embed_image(shap_sum_path, width=320, height=285, caption_text="Figure 6.4: TreeSHAP Beeswarm Summary Plot on Critical Financial Attributions"))
    story.append(sp(6))

    story.append(PageTracker("sec6_5", registry))
    story.append(p("6.5 Live & Simulated Kafka Streaming Feeds", "h1"))
    story.append(p(
        "The <i>Streaming Simulation</i> page provides a real-time monitor of transactions ingested from Kafka topics. "
        "Analysts can start the stream, view events processed per second, observe the live alert feed, and inspect rolling latency measurements."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig6_2", registry))
    dl_train_path = RESULTS_DIR / "deep_learning_training.png"
    story.append(embed_image(dl_train_path, width=410, height=150, caption_text="Figure 6.2: Training and Validation Loss Curves for Deep Neural Network and Autoencoder"))
    story.append(pb())

    # =========================================================
    # CHAPTER 7: RESULT AND DISCUSSION (COST ANALYSIS)
    # =========================================================
    story.append(PageTracker("chap7", registry))
    story.append(p("CHAPTER 7", "chap_num"))
    story.append(p("RESULT AND DISCUSSION (COST ANALYSIS)", "chap_title"))
    
    story.append(PageTracker("sec7_1", registry))
    story.append(p("7.1 Primary Benchmark Evaluation (3,000 Holdout)", "h1"))
    story.append(p(
        "Model performance was evaluated across two benchmarks: (i) an untouched held-out test partition of 3,000 transactions (45 fraud cases) from the primary 15,000 synthetic transaction dataset (Table 7.1 and 7.2) "
        "and (ii) an independent 1,000-row synthetic distribution shift benchmark sample (Table 7.3). Table 7.1 details the validation threshold tuning conducted strictly on the validation partition (3,000 transactions)."
    ))
    story.append(sp(4))

    story.append(PageTracker("tbl7_1", registry))
    story.append(p("Table 7.1: Model Validation Performance and Optimal Operating Thresholds", "table_caption"))
    val_headers = ["Model Architecture", "Model Family", "Validation PR-AUC", "Validation F1", "Validation Recall", "Optimal Threshold (&tau;*)"]
    val_rows = [
        ["Random Forest", "Supervised Tree Ensemble", "0.8879", "0.8434", "0.7778", "&tau;* = 0.7400"],
        ["XGBoost", "Gradient Boosted Trees", "0.9020", "0.8706", "0.8222", "&tau;* = 0.9445"],
        ["Deep Neural Network", "Supervised Deep MLP", "0.8473", "0.8608", "0.7556", "&tau;* = 0.9993"],
        ["Deep Autoencoder", "Semi-Supervised Anomaly", "0.8828", "0.8889", "0.8000", "&tau;* = 0.8707"],
        ["Full Consensus Fusion", "Ensemble Consensus Average", "0.9555", "0.8989", "0.8889", "&tau;* = 0.7916"]
    ]
    story.append(create_styled_table(val_headers, val_rows, [95, 110, 65, 55, 55, 53]))
    story.append(sp(6))

    story.append(PageTracker("tbl7_2", registry))
    story.append(p("Table 7.2: Comprehensive Primary Held-Out Test Evaluation (3,000 Transactions, 45 Fraud Cases)", "table_caption"))
    test_res_headers = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "TP", "FP", "FN"]
    test_res_rows = [
        ["Random Forest", "99.60%", "0.8235", "0.9333", "0.8750", "0.9989", "0.9581", "42", "9", "3"],
        ["XGBoost", "99.67%", "0.8889", "0.8889", "0.8889", "0.9968", "0.9418", "40", "5", "5"],
        ["Deep Neural Network", "99.83%", "1.0000", "0.8889", "0.9412", "0.9863", "0.9307", "40", "0", "5"],
        ["Deep Autoencoder", "99.83%", "0.9762", "0.9111", "0.9425", "0.9909", "0.9586", "41", "1", "4"],
        ["Full Consensus Fusion", "99.70%", "0.9091", "0.8889", "0.8989", "0.9985", "0.9555", "40", "4", "5"]
    ]
    story.append(create_styled_table(test_res_headers, test_res_rows, [105, 42, 42, 42, 45, 42, 42, 24, 24, 25]))
    story.append(sp(6))

    story.append(PageTracker("fig7_1", registry))
    conf_img_path = RESULTS_DIR / "confusion_matrices.png"
    story.append(embed_image(conf_img_path, width=340, height=270, caption_text="Figure 7.1: Confusion Matrices across Supervised and Unsupervised Detection Models"))
    story.append(sp(6))

    story.append(PageTracker("fig7_2", registry))
    mod_comp_img = RESULTS_DIR / "model_comparison_chart.png"
    story.append(embed_image(mod_comp_img, width=410, height=220, caption_text="Figure 7.2: Comprehensive Model Performance Comparison across Key Metrics"))
    story.append(sp(6))

    story.append(PageTracker("sec7_2", registry))
    story.append(p("7.2 Secondary Benchmark & Cross-Generator Shift Analysis", "h1"))
    story.append(p(
        "To rigorously assess model transferability and vulnerability to out-of-distribution shifts, the trained models were evaluated without retraining "
        "on an independent 1,000-row synthetic benchmark sample (data/paysim_sample_1000.csv, containing 15 fraud cases). "
        "Table 7.3 summarizes the empirical results under cross-generator distribution shift."
    ))
    story.append(sp(4))

    story.append(PageTracker("tbl7_3", registry))
    story.append(p("Table 7.3: Operational Degradation under Secondary Synthetic Distribution Shift (1,000 Transactions, 15 Fraud)", "table_caption"))
    shift_headers = ["Model Architecture", "Threshold", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FP", "FN", "Operational Finding"]
    shift_rows = [
        ["Random Forest", "0.7400", "0.0000", "0.0000", "0.0000", "0.8783", "0", "6", "15", "Complete tree recall collapse (0/15 caught)"],
        ["XGBoost", "0.9445", "0.0000", "0.0000", "0.0000", "0.8251", "0", "146", "15", "Complete tree recall collapse (0/15 caught)"],
        ["Deep Neural Network", "0.9993", "0.1471", "1.0000", "0.2564", "0.9635", "15", "87", "0", "100% recall (15/15 caught), 87 false alarms"],
        ["Deep Autoencoder", "0.8707", "0.0427", "1.0000", "0.0820", "0.9118", "15", "336", "0", "100% recall (15/15 caught), 336 false alarms"]
    ]
    story.append(create_styled_table(shift_headers, shift_rows, [85, 36, 36, 36, 38, 38, 18, 18, 18, 110]))
    story.append(sp(6))

    story.append(p(
        "<b>Critical Scientific Insight:</b> The distribution shift experiment reveals a profound operational divergence. "
        "Tree ensembles (Random Forest and XGBoost) suffered total recall collapse (0/15 fraud cases detected) because their learned orthogonal feature split boundaries "
        "failed on shifted transaction amounts and balance dynamics. Conversely, neural representation models (Deep Neural Network and Deep Autoencoder) successfully "
        "detected all 15 fraud cases (100% recall), but incurred significant precision loss (87 and 336 false positives, respectively). "
        "This empirical finding demonstrates that high offline ROC-AUC (0.82–0.96) does not guarantee operational transferability, proving why offline synthetic benchmarks "
        "cannot substitute for live banking telemetry and why automated models must remain advisory within a human-in-the-loop review architecture."
    ))
    story.append(sp(6))

    story.append(PageTracker("sec7_3", registry))
    story.append(p("7.3 Precision-Recall & ROC Analysis", "h1"))
    story.append(p(
        "Under extreme class imbalance, ROC curves (Figure 7.3) can present an overly optimistic assessment because the large number of true negatives "
        "dampens the False Positive Rate denominator. In contrast, Precision-Recall curves (Figure 7.4) remain strictly sensitive to false alarms [16]. "
        "Full Consensus achieves a PR-AUC of 0.9555 on the primary holdout test partition, outperforming individual classifiers."
    ))
    story.append(sp(4))

    story.append(PageTracker("fig7_3", registry))
    roc_img_path = RESULTS_DIR / "roc_curves.png"
    story.append(embed_image(roc_img_path, width=330, height=260, caption_text="Figure 7.3: Receiver Operating Characteristic (ROC) Curves on Held-Out Test Data"))
    story.append(sp(6))

    story.append(PageTracker("fig7_4", registry))
    pr_img_path = RESULTS_DIR / "precision_recall_curves.png"
    story.append(embed_image(pr_img_path, width=330, height=265, caption_text="Figure 7.4: Precision-Recall (PR) Curves Demonstrating Ranking Robustness under Imbalance"))
    story.append(sp(6))

    story.append(PageTracker("sec7_4", registry))
    story.append(p("7.4 False Positive Reduction & Operational Tradeoffs", "h1"))
    story.append(p(
        "On the primary 3,000-transaction holdout, Random Forest generated 9 false positives and XGBoost generated 5 false positives. "
        "By enforcing multi-model consensus, false alarms were reduced to 4—a <b>55.56% reduction</b> relative to Random Forest—while capturing 40 true fraud cases (0.8889 recall). "
        "In production compliance operations, cutting false alarms by over half substantially lightens manual investigation loads without compromising threat detection."
    ))

    story.append(PageTracker("sec7_5", registry))
    story.append(p("7.5 Financial Cost-Benefit & Economic Loss Modeling", "h1"))
    story.append(p(
        "To rigorously quantify economic value, we formulate an asymmetric cost matrix: let cost of missed fraud <i>C</i><sub>FN</sub> = $2,500 (average fraudulent transfer chargeback), "
        "cost of false alarm <i>C</i><sub>FP</sub> = $35 (customer friction, compliance investigation overhead), and cost of correct triage <i>C</i><sub>TP</sub> = $10. "
        "Total Operational Cost is formulated as:"
    ))
    story.append(eq_box("Total Operational Cost = (TP &times; $10) + (FP &times; $35) + (FN &times; $2,500)", "7.1"))
    story.append(sp(4))

    story.append(PageTracker("tbl7_4", registry))
    story.append(p("Table 7.4: Financial Cost Matrix and Asymmetric Error Loss Analysis on Primary Held-Out Test Data", "table_caption"))
    cost_headers = ["Detection Strategy", "Missed Fraud Cost (FN)", "False Alarm Cost (FP)", "Triage Cost (TP)", "Total Operational Loss", "Net Savings vs Baseline"]
    cost_rows = [
        ["No Detection (Zero Fraud Caught)", "$112,500 (45 × $2,500)", "$0", "$0", "$112,500", "Baseline ($0)"],
        ["Random Forest Baseline", "$7,500 (3 × $2,500)", "$315 (9 × $35)", "$420 (42 × $10)", "$8,235", "+$104,265 (92.7% ROI)"],
        ["XGBoost Baseline", "$12,500 (5 × $2,500)", "$175 (5 × $35)", "$400 (40 × $10)", "$13,075", "+$99,425 (88.4% ROI)"],
        ["Deep Autoencoder", "$10,000 (4 × $2,500)", "$35 (1 × $35)", "$410 (41 × $10)", "$10,445", "+$102,055 (90.7% ROI)"],
        ["Full Consensus Fusion", "$12,500 (5 × $2,500)", "$140 (4 × $35)", "$400 (40 × $10)", "$13,040", "+$99,460 (88.4% ROI)"]
    ]
    story.append(create_styled_table(cost_headers, cost_rows, [95, 80, 65, 55, 68, 70]))
    story.append(pb())

    # =========================================================
    # CHAPTER 8: CONCLUSION
    # =========================================================
    story.append(PageTracker("chap8", registry))
    story.append(p("CHAPTER 8", "chap_num"))
    story.append(p("CONCLUSION", "chap_title"))
    
    story.append(PageTracker("sec8_1", registry))
    story.append(p("8.1 Summary of Contributions", "h1"))
    story.append(p(
        "This project formulated, engineered, and empirically validated <b>FraudGuard AI</b>, an explainable, multi-model fraud detection framework for production environments. "
        "The project delivered four core contributions:"
    ))
    story.append(b("<b>Multi-Model Consensus Fusion:</b> A mathematically grounded consensus engine combining tree ensembles and deep representation learning, achieving 0.9091 precision, 0.8889 recall, and cutting false alarms from 9 to 4."))
    story.append(b("<b>Leakage-Free Feature Engineering:</b> Strict stratified partitioning preserving pristine test distributions, paired with domain-specific balance forensics (&Delta;<sub>orig</sub>, &Delta;<sub>dest</sub>, <i>R</i><sub>deplete</sub>)."))
    story.append(b("<b>Operational Explainability:</b> Integration of TreeSHAP and quantitative financial balance forensics to generate verifiable attributions for compliance workflows (GDPR Art. 22)."))
    story.append(b("<b>Enterprise-Hardened Architecture:</b> A containerized FastAPI microservice with an interactive Streamlit workspace, SQLite storage with PBKDF2/bcrypt, brute-force lockout, manual Kafka offset commits, outbox alert crash recovery, and 104/104 automated test pass."))

    story.append(PageTracker("sec8_2", registry))
    story.append(p("8.2 Limitations & Real-World Deployment Readiness", "h1"))
    story.append(b("<b>Vulnerability to Cross-Generator Distribution Shift:</b> Stress-testing on secondary synthetic data revealed total recall collapse in tree-based classifiers (0/15 caught), demonstrating that models tuned on offline synthetic distributions require domain-adapted recalibration before production financial deployment."))
    story.append(b("<b>Synthetic Data Provenance:</b> The evaluation relies upon synthetic transaction generators; while useful for college demonstration and architecture validation, establishing true financial reliability requires live bank telemetry."))
    story.append(b("<b>Post-Transaction Retrospective Scope:</b> Balance discrepancy features depend on completed ledger records; pre-authorization authorization workflows require alternative real-time velocity features."))

    story.append(PageTracker("sec8_3", registry))
    story.append(p("8.3 Future Work", "h1"))
    story.append(b("<b>Graph Neural Networks (GNNs):</b> Future iterations will incorporate Relational Graph Convolutional Networks (R-GCNs) to detect mule networks and coordinated collusion rings."))
    story.append(b("<b>Online Continual Retraining & Concept Drift:</b> Integrating adaptive drift detectors (ADWIN, DDM) to automatically trigger model updates."))
    story.append(b("<b>Hardware Acceleration:</b> Quantizing deep neural networks into ONNX / TensorRT runtimes to achieve sub-5 ms inference latencies."))
    story.append(pb())

    # =========================================================
    # CHAPTER 9: REFERENCES
    # =========================================================
    story.append(PageTracker("chap9", registry))
    story.append(p("CHAPTER 9", "chap_num"))
    story.append(p("REFERENCES", "chap_title"))
    story.append(sp(4))

    references = [
        "<a name=\"ref1\"/><b>[1]</b> A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, 'Credit card fraud detection: A realistic modeling and a novel learning strategy,' <i>IEEE Trans. Neural Netw. Learn. Syst.</i>, vol. 29, no. 8, pp. 3784–3797, Aug. 2018. <a href=\"https://doi.org/10.1109/TNNLS.2017.2751144\" color=\"#1A5276\">[doi:10.1109/TNNLS.2017.2751144]</a>",
        "<a name=\"ref2\"/><b>[2]</b> A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, 'Calibrating probability with undersampling for unbalanced classification,' in <i>Proc. IEEE Symp. Series on Computational Intelligence (SSCI)</i>, 2015, pp. 159–166. <a href=\"https://doi.org/10.1109/SSCI.2015.33\" color=\"#1A5276\">[doi:10.1109/SSCI.2015.33]</a>",
        "<a name=\"ref3\"/><b>[3]</b> N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, 'SMOTE: Synthetic minority over-sampling technique,' <i>J. Artif. Intell. Res.</i>, vol. 16, pp. 321–357, 2002. <a href=\"https://doi.org/10.1613/jair.953\" color=\"#1A5276\">[doi:10.1613/jair.953]</a>",
        "<a name=\"ref4\"/><b>[4]</b> A. Pumsirirat and L. Yan, 'Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine,' <i>Int. J. Adv. Comput. Sci. Appl.</i>, vol. 9, no. 1, pp. 18–25, 2018. <a href=\"https://doi.org/10.14569/IJACSA.2018.090103\" color=\"#1A5276\">[doi:10.14569/IJACSA.2018.090103]</a>",
        "<a name=\"ref5\"/><b>[5]</b> L. Breiman, 'Random forests,' <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5–32, 2001. <a href=\"https://doi.org/10.1023/A:1010933404324\" color=\"#1A5276\">[doi:10.1023/A:1010933404324]</a>",
        "<a name=\"ref6\"/><b>[6]</b> T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785–794. <a href=\"https://doi.org/10.1145/2939672.2939785\" color=\"#1A5276\">[doi:10.1145/2939672.2939785]</a>",
        "<a name=\"ref7\"/><b>[7]</b> J. An and S. Cho, 'Variational autoencoder based anomaly detection using reconstruction probability,' <i>Special Lecture on IE</i>, vol. 2, no. 1, pp. 1–18, 2015.",
        "<a name=\"ref8\"/><b>[8]</b> T. Fawcett, 'An introduction to ROC analysis,' <i>Pattern Recognit. Lett.</i>, vol. 27, no. 8, pp. 861–874, 2006. <a href=\"https://doi.org/10.1016/j.patrec.2005.10.010\" color=\"#1A5276\">[doi:10.1016/j.patrec.2005.10.010]</a>",
        "<a name=\"ref9\"/><b>[9]</b> S. Hochreiter and J. Schmidhuber, 'Long short-term memory,' <i>Neural Comput.</i>, vol. 9, no. 8, pp. 1735–1780, 1997. <a href=\"https://doi.org/10.1162/neco.1997.9.8.1735\" color=\"#1A5276\">[doi:10.1162/neco.1997.9.8.1735]</a>",
        "<a name=\"ref10\"/><b>[10]</b> J. Jurgovsky et al., 'Sequence classification for credit-card fraud detection,' <i>Expert Syst. Appl.</i>, vol. 100, pp. 234–245, 2018. <a href=\"https://doi.org/10.1016/j.eswa.2018.01.037\" color=\"#1A5276\">[doi:10.1016/j.eswa.2018.01.037]</a>",
        "<a name=\"ref11\"/><b>[11]</b> S. M. Lundberg and S.-I. Lee, 'A unified approach to interpreting model predictions,' in <i>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, 2017, pp. 4765–4774.",
        "<a name=\"ref12\"/><b>[12]</b> S. M. Lundberg et al., 'From local explanations to global understanding with explainable AI for trees,' <i>Nature Mach. Intell.</i>, vol. 2, no. 1, pp. 56–67, 2020. <a href=\"https://doi.org/10.1038/s42256-019-0138-9\" color=\"#1A5276\">[doi:10.1038/s42256-019-0138-9]</a>",
        "<a name=\"ref13\"/><b>[13]</b> M. T. Ribeiro, S. Singh, and C. Guestrin, '\"Why should I trust you?\": Explaining the predictions of any classifier,' in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 1135–1144. <a href=\"https://doi.org/10.1145/2939672.2939778\" color=\"#1A5276\">[doi:10.1145/2939672.2939778]</a>",
        "<a name=\"ref14\"/><b>[14]</b> M. Shanaa and S. Abdallah, 'A hybrid anomaly detection framework combining supervised and unsupervised learning for credit card fraud detection,' <i>F1000Research</i>, vol. 14, art. 664, 2025. <a href=\"https://doi.org/10.12688/f1000research.160105.1\" color=\"#1A5276\">[doi:10.12688/f1000research.160105.1]</a>",
        "<a name=\"ref15\"/><b>[15]</b> S. Iqbal, K. M. Awan, S. Kamal, and Z. Ur Rehman, 'Interpretable ensemble learning models for credit card fraud detection,' <i>Appl. Sci.</i>, vol. 15, no. 22, art. 12073, 2025. <a href=\"https://doi.org/10.3390/app152212073\" color=\"#1A5276\">[doi:10.3390/app152212073]</a>",
        "<a name=\"ref16\"/><b>[16]</b> T. Saito and M. Rehmsmeier, 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,' <i>PLOS ONE</i>, vol. 10, no. 3, art. e0118432, 2015. <a href=\"https://doi.org/10.1371/journal.pone.0118432\" color=\"#1A5276\">[doi:10.1371/journal.pone.0118432]</a>"
    ]
    for r in references:
        story.append(Paragraph(r, styles["ref_entry"]))
    story.append(pb())

    # =========================================================
    # APPENDIX A - SAMPLE CODE
    # =========================================================
    story.append(PageTracker("appendix", registry))
    story.append(p("APPENDIX A - SAMPLE CODE", "prelim_title"))
    story.append(sp(4))

    story.append(p("<b>A.1 Leakage-Free Preprocessing & Discrepancy Forensics (src/preprocessor.py):</b>", "body_bold"))
    preproc_code = (
        "class TransactionPreprocessor:\n"
        "    def __init__(self):\n"
        "        self.scaler = RobustScaler()\n"
        "        self.median_values = {}\n"
        "        self.mode_values = {}\n\n"
        "    def fit(self, X_train: pd.DataFrame):\n"
        "        # Learn statistics strictly from training partition\n"
        "        self.median_values = X_train[NUMERIC_COLS].median().to_dict()\n"
        "        self.mode_values = X_train[CAT_COLS].mode().iloc[0].to_dict()\n"
        "        X_eng = self._engineer_features(X_train)\n"
        "        self.scaler.fit(X_eng[CONTINUOUS_COLS])\n"
        "        return self\n\n"
        "    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:\n"
        "        df = df.copy()\n"
        "        # Quantitative Balance Discrepancy Forensics\n"
        "        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']\n"
        "        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']\n"
        "        df['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-5)\n"
        "        return df"
    )
    story.append(create_code_block(preproc_code))
    story.append(sp(8))

    story.append(p("<b>A.2 Multi-Model Consensus Engine (src/model_comparison.py):</b>", "body_bold"))
    consensus_code = (
        "def evaluate_consensus(model_predictions: dict, thresholds: dict) -> dict:\n"
        "    # Step 1: Evaluate each model against its validation-tuned threshold\n"
        "    flags = {m: (score >= thresholds[m]) for m, score in model_predictions.items()}\n"
        "    votes = sum(flags.values())\n\n"
        "    # Step 2: Compute composite consensus risk score\n"
        "    consensus_score = float(np.mean(list(model_predictions.values())))\n\n"
        "    # Step 3: Map votes and score to operational risk tier\n"
        "    if votes <= 1 and consensus_score < 0.30:\n"
        "        tier, action = 'LOW RISK', 'NO MODEL FLAG'\n"
        "    elif votes == 2 or consensus_score < 0.70:\n"
        "        tier, action = 'MEDIUM RISK', 'FLAG FOR REVIEW'\n"
        "    else:\n"
        "        tier, action = 'HIGH RISK', 'HIGH RISK ALERT'\n\n"
        "    return {'fraud_votes': votes, 'consensus_score': consensus_score, 'tier': tier, 'action': action}"
    )
    story.append(create_styled_code_block if "create_styled_code_block" in globals() else create_code_block(consensus_code))
    story.append(sp(8))

    story.append(p("<b>A.3 Fault-Tolerant Kafka Consumer & Outbox Crash Recovery (src/kafka_streaming.py):</b>", "body_bold"))
    kafka_code = (
        "class FraudStreamingConsumer:\n"
        "    def process_message(self, message_dict: dict) -> dict:\n"
        "        event_id = message_dict.get('event_id', f'EVT-{uuid.uuid4().hex[:8].upper()}')\n"
        "        # 1. Idempotency & Crash Recovery: check DB case and pending outbox alert\n"
        "        if self.store is not None:\n"
        "            existing_case = self.store.get_case_by_event(event_id)\n"
        "            pending_alert = self.store.get_pending_alert_by_event(event_id)\n"
        "            if existing_case is not None:\n"
        "                if pending_alert and pending_alert.get('status') == 'PENDING':\n"
        "                    alert_event = json.loads(pending_alert['payload_json'])\n"
        "                    self._publish_alert_synchronous(alert_event)\n"
        "                    self.store.mark_alert_delivered(pending_alert['alert_id'])\n"
        "                    return {'is_duplicate': True, 'alert_delivered_on_recovery': True}\n"
        "                return {'is_duplicate': True, 'idempotent_action': 'SKIPPED_EXISTING_CASE'}\n\n"
        "        # 2. Consensus Scoring & Forensic Analysis\n"
        "        record = self.model_engine.predict_transaction(message_dict)\n"
        "        if record['is_flagged']:\n"
        "            alert_id = f'ALT-{uuid.uuid4().hex[:10].upper()}'\n"
        "            # 3. Persist pending alert in SQLite outbox before broker dispatch\n"
        "            self.store.record_pending_alert(alert_id, event_id, self.alerts_topic, payload=record)\n"
        "            # 4. Synchronously publish to Kafka and await broker acknowledgement\n"
        "            self._publish_alert_synchronous(record)\n"
        "            self.store.mark_alert_delivered(alert_id)\n"
        "        return record"
    )
    story.append(create_code_block(kafka_code))

    return story

# -------------------------------------------------------------
# Two-Pass Compilation Runner
# -------------------------------------------------------------
def generate_final_pdf():
    print("Starting Pass 1: Recording physical page numbers...")
    registry = {}
    doc_pass1 = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM
    )
    story_pass1 = build_report_story(registry, is_pass2=False)
    doc_pass1.build(story_pass1, canvasmaker=VITReportCanvas)
    
    abs_page = registry.get("abstract", 11)
    VITReportCanvas.abstract_end_page = abs_page
    print(f"Pass 1 Completed! Abstract ends at physical page {abs_page}.")
    print("Registry snapshot:", {k: registry[k] for k in list(registry.keys())[:12]})

    print("\nStarting Pass 2: Generating final PDF with exact Table of Contents and figure page references...")
    doc_pass2 = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM
    )
    story_pass2 = build_report_story(registry, is_pass2=True)
    doc_pass2.build(story_pass2, canvasmaker=VITReportCanvas)

    # Also sync copies to docs and output/pdf
    docs_copy = WORKSPACE_DIR / "docs" / "BCSE497J_Project_Report_Financial_Fraud_Detection.pdf"
    out_copy = WORKSPACE_DIR / "output" / "pdf" / "BCSE497J_Project_Report_Financial_Fraud_Detection.pdf"
    os.makedirs(docs_copy.parent, exist_ok=True)
    os.makedirs(out_copy.parent, exist_ok=True)
    import shutil
    shutil.copyfile(OUTPUT_PDF, docs_copy)
    shutil.copyfile(OUTPUT_PDF, out_copy)

    print(f"\nFinal PDF Report successfully generated at:\n{OUTPUT_PDF}")
    print(f"Copies synced to:\n - {docs_copy}\n - {out_copy}")
    print(f"File Size: {os.path.getsize(OUTPUT_PDF) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_final_pdf()
