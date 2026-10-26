"""
Builds the final Review 3 presentation for VIT BCSE497J Project-I:
Topic: FraudGuard AI: An Explainable Financial Fraud Detection System Using Machine Learning and Deep Learning
Presentation Files:
  - Review 3- 28-10- 2026 (1).pptx (Target submission presentation)
  - Review 3- 28-10- 2026 (Final).pptx (Backup final copy)
  - docs/Review 3- 28-10- 2026 (1).pptx (Synced docs archive)
"""

import os
import shutil
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

WORKSPACE = Path(__file__).resolve().parent.parent
PPTX_FILE = WORKSPACE / "Review 3- 28-10- 2026 (1).pptx"
ASSETS_DIR = WORKSPACE / "scripts" / "report_assets"
RESULTS_DIR = WORKSPACE / "results"

# Standard Professional Academic Color Palette
COLOR_NAVY = RGBColor(0x1F, 0x4E, 0x79)       # Deep Classic Navy
COLOR_DARK = RGBColor(0x20, 0x20, 0x20)       # High-contrast Charcoal
COLOR_MUTED = RGBColor(0x55, 0x55, 0x55)      # Secondary Charcoal/Gray
COLOR_BLUE = RGBColor(0x2E, 0x75, 0xB6)       # Academic Accent Blue
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)      # Pure White
COLOR_CARD_BG = RGBColor(0xF4, 0xF7, 0xFA)    # Soft Card Background
COLOR_CARD_BG_ALT = RGBColor(0xEA, 0xF0, 0xF6)# Tinted Card Background
COLOR_BORDER = RGBColor(0xCB, 0xD5, 0xE0)     # Clean Border Gray
COLOR_HIGHLIGHT = RGBColor(0x05, 0x96, 0x69)  # Emerald Green
COLOR_RED = RGBColor(0xC0, 0x00, 0x00)        # Accent Red
COLOR_AMBER = RGBColor(0xD9, 0x77, 0x06)      # Amber Warning

def update_footers(slide, review_date="28/10/26"):
    """Updates the footer date placeholder to the Review 3 date."""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.name.startswith("Date"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = review_date
            p.font.name = "Times New Roman"
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_MUTED
        elif shape.has_text_frame and shape.name.startswith("Footer"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "VIT SCOPE  |  BCSE497J Project-I"
            p.font.name = "Times New Roman"
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_MUTED

def clean_slide_body(slide, is_slide1=False, is_slide2=False, is_slide7=False):
    """Removes previous body shapes, tables, and text while preserving titles, footers, and approved template media."""
    for shape in list(slide.shapes):
        # Always keep footer elements, date, and title
        if any(shape.name.startswith(p) for p in ("Date", "Footer", "Slide Number", "object 2", "Title 1", "Title 6")):
            continue
        # Slide 1: keep Picture 4 (VIT logo) and object 3 (details box)
        if is_slide1 and shape.name in ("Picture 4", "object 3"):
            continue
        # Slide 2: keep Picture 6 (approval email)
        if is_slide2 and shape.name == "Picture 6":
            continue
        # Slide 7: keep Picture 7 and Picture 8 (SDG logos)
        if is_slide7 and shape.name in ("Picture 7", "Picture 8"):
            continue
        # Remove everything else (old tables, textboxes, generated cards, pictures)
        slide.shapes._spTree.remove(shape._element)

def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_BORDER):
    """Adds a rounded rectangle card background for grouping content."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_header_banner(slide, title_text):
    """Updates the slide title with Times New Roman bold styling."""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.name in ("object 2", "Title 1"):
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.name = "Times New Roman"
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = COLOR_NAVY
            return shape
    return None

def add_p(tf, text, font_size=14, bold=False, color=COLOR_DARK, align=PP_ALIGN.LEFT):
    p = tf.add_paragraph() if len(tf.paragraphs[0].text) > 0 else tf.paragraphs[0]
    p.text = text
    p.font.name = "Times New Roman"
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return p

def build_presentation():
    print(f"Loading template presentation: {PPTX_FILE}...")
    prs = pptx.Presentation(str(PPTX_FILE))

    # Ensure presentation has at least 18 slides (14 core + 4 dedicated backup slides)
    while len(prs.slides) < 18:
        prs.slides.add_slide(prs.slide_layouts[1])

    # =========================================================
    # SLIDE 1: Title Slide
    # =========================================================
    print("Building Slide 1: Title Slide...")
    slide1 = prs.slides[0]
    title_shape = slide1.shapes[1]
    tf1 = title_shape.text_frame
    tf1.word_wrap = True
    tf1.clear()

    p_deg = tf1.paragraphs[0]
    p_deg.text = "B.Tech Computer Science and Engineering  |  BCSE497J – Project-I (Review 3)"
    p_deg.font.name = "Times New Roman"
    p_deg.font.size = Pt(16.5)
    p_deg.font.bold = True
    p_deg.font.color.rgb = COLOR_NAVY
    p_deg.alignment = PP_ALIGN.CENTER

    p_main = tf1.add_paragraph()
    p_main.text = "FraudGuard AI: An Explainable Financial Fraud Detection System Using Machine Learning and Deep Learning"
    p_main.font.name = "Times New Roman"
    p_main.font.size = Pt(21)
    p_main.font.bold = True
    p_main.font.color.rgb = COLOR_DARK
    p_main.alignment = PP_ALIGN.CENTER

    p_sub = tf1.add_paragraph()
    p_sub.text = "With Secure RESTful API and Real-Time Kafka Streaming Ingestion"
    p_sub.font.name = "Times New Roman"
    p_sub.font.size = Pt(13)
    p_sub.font.italic = True
    p_sub.font.color.rgb = COLOR_BLUE
    p_sub.alignment = PP_ALIGN.CENTER

    details_shape = slide1.shapes[2]
    tf1_det = details_shape.text_frame
    tf1_det.word_wrap = True
    tf1_det.clear()

    add_p(tf1_det, "Project Team Members (SCOPE):", 15, True, COLOR_NAVY, PP_ALIGN.CENTER)
    add_p(tf1_det, "23BCE0374  Dhruv Singh    |    23BCE0415  Animesh B. Labh    |    23BDS0091  Kushagra Dhelia", 13.5, False, COLOR_DARK, PP_ALIGN.CENTER)
    add_p(tf1_det, "", 8, False)
    add_p(tf1_det, "Project Supervisor / Faculty Guide:", 15, True, COLOR_NAVY, PP_ALIGN.CENTER)
    add_p(tf1_det, "Dr. Erapaneni Gayatri, Assistant Professor Sr. Grade 1", 14, True, COLOR_DARK, PP_ALIGN.CENTER)
    add_p(tf1_det, "School of Computer Science and Engineering (SCOPE), VIT Vellore", 13, False, COLOR_MUTED, PP_ALIGN.CENTER)
    update_footers(slide1)

    # =========================================================
    # SLIDE 2: Approval Mail From Guide
    # =========================================================
    print("Building Slide 2: Approval Mail From Guide...")
    slide2 = prs.slides[1]
    add_header_banner(slide2, "Approval Mail From Guide")
    update_footers(slide2)

    # Clean any old generated shapes on slide 2
    for shape in list(slide2.shapes):
        if shape.name.startswith("Rounded Rectangle") or shape.name.startswith("TextBox") or shape.name.startswith("Content"):
            slide2.shapes._spTree.remove(shape._element)

    # Left: Concise faculty guide verification card (no technical claims, no self-written checklist)
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(4.8), Inches(4.8), COLOR_CARD_BG, COLOR_BORDER)
    tb2 = slide2.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(4.5), Inches(4.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    add_p(tf2, "Faculty Supervisor Verification", 16, True, COLOR_NAVY)
    add_p(tf2, "Project Guide:", 13, True, COLOR_DARK)
    add_p(tf2, "Dr. Erapaneni Gayatri", 14, True, COLOR_HIGHLIGHT)
    add_p(tf2, "Assistant Professor Sr. Grade 1", 12, False, COLOR_DARK)
    add_p(tf2, "School of Computer Science and Engineering (SCOPE)", 11.5, False, COLOR_MUTED)
    add_p(tf2, "Vellore Institute of Technology, Vellore", 11.5, False, COLOR_MUTED)
    add_p(tf2, "", 10, False)
    add_p(tf2, "Review 3 Scope & Approval Confirmation:", 13, True, COLOR_NAVY)
    add_p(tf2, "• The official supervisor approval email for BCSE497J Project-I Review 3 is attached on the right.", 11.5, False, COLOR_DARK)
    add_p(tf2, "• Verified academic deliverables include the Review 3 presentation deck, comprehensive 41-page project report, and working prototype codebase.", 11.5, False, COLOR_DARK)
    add_p(tf2, "• Detailed project outcomes and automated test verification are documented on Slide 13.", 11.5, False, COLOR_DARK)

    # Right: Reposition the guide approval mail image prominently
    for shape in slide2.shapes:
        if shape.name == "Picture 6":
            shape.left = Inches(5.8)
            shape.top = Inches(1.8)
            shape.width = Inches(6.7)
            shape.height = Inches(4.8)

    # =========================================================
    # SLIDE 3: Aim
    # =========================================================
    print("Building Slide 3: Aim...")
    slide3 = prs.slides[2]
    add_header_banner(slide3, "Aim of the Project")
    clean_slide_body(slide3)
    update_footers(slide3)

    card_w = Inches(3.7)
    card_h = Inches(4.7)
    top_pos = Inches(1.8)

    # Card 1: Multi-Model Benchmarking
    add_card(slide3, Inches(0.8), top_pos, card_w, card_h)
    tb3_1 = slide3.shapes.add_textbox(Inches(0.95), top_pos + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True
    add_p(tf3_1, "1. Multi-Model Comparative Benchmarking", 15, True, COLOR_NAVY)
    add_p(tf3_1, "Formulate, train, and empirically benchmark four diverse fraud detection architectures:", 12, False, COLOR_DARK)
    add_p(tf3_1, "  • Supervised tree ensembles: Random Forest and XGBoost.", 11.5, False, COLOR_DARK)
    add_p(tf3_1, "  • Deep neural representations: Deep Neural Network (MLP) and Deep Bottleneck Autoencoder.", 11.5, False, COLOR_DARK)
    add_p(tf3_1, "", 6, False)
    add_p(tf3_1, "Compare individual discrimination performance across Precision, Recall, F1-Score, PR-AUC, and ROC-AUC on identical holdout partitions.", 12, False, COLOR_DARK)

    # Card 2: Leakage-Free Imbalance Handling
    add_card(slide3, Inches(4.8), top_pos, card_w, card_h)
    tb3_2 = slide3.shapes.add_textbox(Inches(4.95), top_pos + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True
    add_p(tf3_2, "2. Leakage-Free Imbalance Handling", 15, True, COLOR_NAVY)
    add_p(tf3_2, "Address extreme class imbalance on the benchmark dataset:", 12, False, COLOR_DARK)
    add_p(tf3_2, "  • Actual Benchmark Prevalence: 225/15,000 = 1.5% fraud rate across 15,000 synthetic records.", 12, True, COLOR_HIGHLIGHT)
    add_p(tf3_2, "", 6, False)
    add_p(tf3_2, "Prevent methodological data leakage by restricting SMOTE-NC oversampling and RobustScaler interquartile scaling strictly to training partitions, ensuring validation and testing holdouts remain uncorrupted.", 12, False, COLOR_DARK)

    # Card 3: Explainability & Analyst Review
    add_card(slide3, Inches(8.8), top_pos, card_w, card_h)
    tb3_3 = slide3.shapes.add_textbox(Inches(8.95), top_pos + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf3_3 = tb3_3.text_frame
    tf3_3.word_wrap = True
    add_p(tf3_3, "3. Explainable Predictions & Analyst Review", 15, True, COLOR_NAVY)
    add_p(tf3_3, "Embed explainable AI capabilities to assist human fraud analysts during investigation:", 12, False, COLOR_DARK)
    add_p(tf3_3, "  • Local TreeSHAP waterfall feature attributions identifying exact risk drivers.", 11.5, False, COLOR_DARK)
    add_p(tf3_3, "  • Quantitative balance discrepancy forensics (deltaOrig, deltaDest).", 11.5, False, COLOR_DARK)
    add_p(tf3_3, "", 6, False)
    add_p(tf3_3, "Support human-in-the-loop review through an interactive dashboard with multi-model comparison and persistent audit history.", 12, False, COLOR_DARK)

    # =========================================================
    # SLIDE 4: Abstract
    # =========================================================
    print("Building Slide 4: Abstract...")
    slide4 = prs.slides[3]
    add_header_banner(slide4, "Abstract")
    clean_slide_body(slide4)
    update_footers(slide4)

    box_w = Inches(5.6)
    box_h = Inches(4.7)

    add_card(slide4, Inches(0.8), top_pos, box_w, box_h)
    tb4_1 = slide4.shapes.add_textbox(Inches(0.95), top_pos + Inches(0.15), box_w - Inches(0.3), box_h - Inches(0.3))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True
    add_p(tf4_1, "Problem Context, Dataset & Methodology", 15, True, COLOR_NAVY)
    add_p(tf4_1, "• Industrial Challenge: Digital payment expansion requires resilient fraud detection. Traditional rule engines suffer from rigidity and false alarm fatigue, while black-box models lack interpretability for human analysts.", 11.5, False, COLOR_DARK)
    add_p(tf4_1, "• Benchmark Dataset: Evaluated on 15,000 locally generated synthetic transactions containing 225 fraud cases (1.50% fraud prevalence) across 9 mobile transaction fields.", 11.5, False, COLOR_DARK)
    add_p(tf4_1, "• Leakage-Free Preprocessing: Stratified 60/20/20 train/validation/test split. Quantile RobustScaler fitting and SMOTE-NC minority synthesis are restricted strictly to training partitions; domain balance forensics (deltaOrig, deltaDest, R_deplete) are computed per transaction.", 11.5, False, COLOR_DARK)
    add_p(tf4_1, "• Multi-Model Architecture: Benchmarks 4 diverse paradigms: Random Forest, XGBoost, Deep Neural Network (MLP), and Deep Bottleneck Autoencoder.", 11.5, False, COLOR_DARK)

    add_card(slide4, Inches(6.8), top_pos, box_w, box_h)
    tb4_2 = slide4.shapes.add_textbox(Inches(6.95), top_pos + Inches(0.15), box_w - Inches(0.3), box_h - Inches(0.3))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    add_p(tf4_2, "Implemented Prototype, Workflow & Experiments", 15, True, COLOR_NAVY)
    add_p(tf4_2, "• Implemented Academic Prototype: Containerized FastAPI REST service, interactive Streamlit workspace, and event-driven Apache Kafka streaming consumer with transactional SQLite outbox recovery.", 11.5, False, COLOR_DARK)
    add_p(tf4_2, "• Primary Holdout Evaluation (3,000 txns, 45 frauds): Individual models achieve strong holdout discrimination: Random Forest (Recall: 0.9333, TP: 42), DNN (Precision: 1.0000, FP: 0), Autoencoder (F1: 0.9425, FP: 1), and XGBoost baseline (F1: 0.8889).", 11.5, False, COLOR_DARK)
    add_p(tf4_2, "• Dashboard Workflow: Evaluates transactions against individual thresholds; an any-model review rule flags transactions if any detector triggers, supported by TreeSHAP local attributions and SQLite audit persistence.", 11.5, False, COLOR_DARK)
    add_p(tf4_2, "• Offline Weighted-Fusion Experiment: In a separate offline experiment, a validation-tuned weighted ensemble reduced false positives from 9 to 4 compared with Random Forest, while true positives moved from 42 to 40.", 11.5, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 5: Literature Review
    # =========================================================
    print("Building Slide 5: Literature Review...")
    slide5 = prs.slides[4]
    add_header_banner(slide5, "Literature Review")
    clean_slide_body(slide5)
    update_footers(slide5)

    tbl_s5 = slide5.shapes.add_table(6, 6, Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.8))
    t5 = tbl_s5.table
    col_w5 = [Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.7), Inches(1.8), Inches(2.8)]
    for idx, w in enumerate(col_w5):
        t5.columns[idx].width = w

    headers5 = ["Authors & Year", "Model Architectures", "Imbalance Method", "Explainability", "Deployment", "Key Limitations / Research Gaps"]
    for c_idx, h in enumerate(headers5):
        cell = t5.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Times New Roman"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ["Dal Pozzolo et al. (2018) [1]", "Random Forest, LogReg, SVM", "Undersampling + Calibration", "None", "Batch Evaluation", "No neural representation learning; lacks feature-level local explainability."],
        ["Pumsirirat & Yan (2018) [4]", "Autoencoder, RBM", "Normal-only Unsupervised", "None", "Offline Prototype", "Focuses on uncalibrated reconstruction distance; lacks comparative supervised baseline."],
        ["Shanaa & Abdallah (2025) [14]", "Supervised Tree Models", "Resampling Techniques", "Feature Importance", "Experimental Scripts", "Evaluated on static datasets; lacks real-time ingestion and end-to-end web deployment."],
        ["Iqbal et al. (2025) [15]", "Random Forest, XGBoost", "Standard SMOTE", "SHAP and LIME", "Jupyter Notebook", "Supervised-only; lacks unsupervised anomaly detection and fault-tolerant streaming architecture."],
        ["FraudGuard AI (This Work)", "RF, XGBoost, DNN, Autoencoder", "SMOTE-NC (Train Only)", "TreeSHAP + Forensics", "Implemented Prototype", "Four-model comparison, threshold calibration, Kafka streaming with outbox recovery, 104 passing tests."]
    ]

    for r_idx, row in enumerate(lit_data):
        for c_idx, val in enumerate(row):
            cell = t5.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx == 4:
                cell.fill.fore_color.rgb = RGBColor(0xE6, 0xF4, 0xEA)
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = COLOR_CARD_BG
            else:
                cell.fill.fore_color.rgb = COLOR_WHITE

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Times New Roman"
            p.font.size = Pt(10)
            p.font.bold = (r_idx == 4)
            p.font.color.rgb = COLOR_HIGHLIGHT if (r_idx == 4 and c_idx == 0) else COLOR_DARK

    # =========================================================
    # SLIDE 6: Research Gap
    # =========================================================
    print("Building Slide 6: Research Gap...")
    slide6 = prs.slides[5]
    add_header_banner(slide6, "Research Gap")
    clean_slide_body(slide6)
    update_footers(slide6)

    gap_data = [
        ("Gap 1: Anomaly Score Normalization and Threshold Selection in Autoencoders",
         "Prior hybrid detection frameworks frequently use raw, unstandardized reconstruction errors (MSE) directly alongside classifier probabilities using ad-hoc heuristics without systematic validation.",
         "FraudGuard AI Solution: Normalizes reconstruction error into an anomaly score and selects its threshold on validation data (tau*) to maximize F1, establishing a disciplined operational boundary before multi-model comparison."),
        ("Gap 2: Methodological Data Leakage in Preprocessing Pipelines",
         "Preprocessing transformations (scaling, imputation, oversampling) are frequently fit across entire datasets prior to train/test partitioning, leading to overly optimistic benchmark metrics.",
         "FraudGuard AI Solution: Enforces strict parameter freezing on the 60% training partition only; the 20% validation and 20% test partitions remain untouched to reflect true unseen distributions."),
        ("Gap 3: Absence of Streaming Fault Tolerance in Academic Prototypes",
         "Academic fraud detection prototypes typically operate as offline scripts without crash resilience, risking dropped transactions during stream interruptions.",
         "FraudGuard AI Solution: Implements an Apache Kafka consumer with transactional SQLite outbox persistence (pending_alerts), manual offset commits, and synchronous broker acknowledgements to enable retry and replay upon worker crashes.")
    ]

    for idx, (title, gap_text, sol_text) in enumerate(gap_data):
        top_y = Inches(1.8 + idx * 1.6)
        add_card(slide6, Inches(0.8), top_y, Inches(11.7), Inches(1.45))
        tb6 = slide6.shapes.add_textbox(Inches(0.95), top_y + Inches(0.08), Inches(11.4), Inches(1.3))
        tf6 = tb6.text_frame
        tf6.word_wrap = True
        add_p(tf6, title, 13.5, True, COLOR_NAVY)
        add_p(tf6, f"• Existing Limitation: {gap_text}", 11, False, COLOR_DARK)
        add_p(tf6, f"• {sol_text}", 11, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 7: Objectives and SDGs
    # =========================================================
    print("Building Slide 7: Objectives...")
    slide7 = prs.slides[6]
    add_header_banner(slide7, "Project Objectives & Sustainable Development Goals")
    clean_slide_body(slide7, is_slide7=True)
    update_footers(slide7)

    # Left: Objectives Box (Measurable, verifiable technical objectives)
    add_card(slide7, Inches(0.8), Inches(1.8), Inches(7.5), Inches(4.8))
    tb7 = slide7.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(7.2), Inches(4.5))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    add_p(tf7, "Measurable Technical Objectives", 15, True, COLOR_NAVY)
    add_p(tf7, "1. Implement a leakage-free feature engineering pipeline, generating domain balance forensics (deltaOrig, deltaDest) and restricting SMOTE-NC strictly to training partitions.", 11.5, False, COLOR_DARK)
    add_p(tf7, "2. Train, optimize, and benchmark four diverse model architectures: Random Forest, XGBoost, Deep Neural Network (MLP), and Deep Bottleneck Autoencoder.", 11.5, False, COLOR_DARK)
    add_p(tf7, "3. Implement an interactive analyst dashboard supporting batch CSV ingestion, four-model comparison, and an any-model review rule with four operational risk tiers (LOW, MEDIUM, HIGH, CRITICAL).", 11.5, False, COLOR_DARK)
    add_p(tf7, "4. Integrate explainable AI capabilities, embedding local TreeSHAP attributions and domain-specific balance forensics within an automated audit workflow.", 11.5, False, COLOR_DARK)
    add_p(tf7, "5. Construct a containerized FastAPI RESTful microservice and SQLite workspace store featuring salted PBKDF2-HMAC-SHA256 password hashing, API-key authentication, and thread-safe WAL mode.", 11.5, False, COLOR_DARK)
    add_p(tf7, "6. Construct an event-driven Apache Kafka real-time streaming pipeline featuring transactional outbox persistence, manual offset commits, and synchronous broker acknowledgements for crash recovery.", 11.5, False, COLOR_DARK)
    add_p(tf7, "7. Validate system robustness through a 104-test automated regression suite achieving a 100% pass rate across all functional, security, and streaming modules.", 11.5, False, COLOR_DARK)

    # Right: SDGs (Potential Relevance)
    add_card(slide7, Inches(8.5), Inches(1.8), Inches(4.0), Inches(4.8))
    tb7_r = slide7.shapes.add_textbox(Inches(8.65), Inches(1.9), Inches(3.7), Inches(4.5))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True
    add_p(tf7_r, "Identified SDGs (Potential Relevance)", 15, True, COLOR_NAVY)
    add_p(tf7_r, "• SDG 8 (Decent Work & Economic Growth): Target 8.10 – Potential relevance to strengthening digital payment resilience and protecting financial transaction integrity against fraud.", 11, False, COLOR_DARK)
    add_p(tf7_r, "• SDG 9 (Industry, Innovation & Infrastructure): Target 9.5 – Potential relevance to upgrading technological capabilities and fostering robust, explainable automated infrastructure.", 11, False, COLOR_DARK)
    add_p(tf7_r, "• SDG 16 (Peace, Justice & Strong Institutions): Target 16.5 – Potential relevance to supporting compliance auditing and mitigating illicit financial flows and financial malpractices.", 11, False, COLOR_DARK)

    # Position SDG pictures inside the right box nicely
    for shape in slide7.shapes:
        if shape.name == "Picture 7":
            shape.left = Inches(8.8)
            shape.top = Inches(4.9)
            shape.width = Inches(1.5)
            shape.height = Inches(1.4)
        elif shape.name == "Picture 8":
            shape.left = Inches(10.6)
            shape.top = Inches(4.9)
            shape.width = Inches(1.5)
            shape.height = Inches(1.4)

    # =========================================================
    # SLIDE 8: Architecture
    # =========================================================
    print("Building Slide 8: Architecture Diagram...")
    slide8 = prs.slides[7]
    add_header_banner(slide8, "System Architecture & Execution Paths")
    clean_slide_body(slide8)
    update_footers(slide8)

    arch_img = ASSETS_DIR / "system_architecture.png"
    if arch_img.exists():
        slide8.shapes.add_picture(str(arch_img), Inches(0.8), Inches(1.7), Inches(7.2), Inches(4.8))

    add_card(slide8, Inches(8.2), Inches(1.7), Inches(4.3), Inches(4.8))
    tb8 = slide8.shapes.add_textbox(Inches(8.35), Inches(1.8), Inches(4.0), Inches(4.5))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    add_p(tf8, "Three Distinct Execution Paths", 15, True, COLOR_NAVY)
    add_p(tf8, "1. Offline Training Pipeline:", 12.5, True, COLOR_BLUE)
    add_p(tf8, "• Stratified 60/20/20 train/validation/test partitioning.", 11, False, COLOR_DARK)
    add_p(tf8, "• Training-only parameter estimation (RobustScaler & SMOTE-NC).", 11, False, COLOR_DARK)
    add_p(tf8, "• Independent model training and validation threshold tuning (tau*).", 11, False, COLOR_DARK)
    add_p(tf8, "", 4, False)
    add_p(tf8, "2. Dashboard Inference & Review:", 12.5, True, COLOR_BLUE)
    add_p(tf8, "• Streamlit analyst workspace loads trained models.", 11, False, COLOR_DARK)
    add_p(tf8, "• Evaluates transactions against thresholds via any-model review rule.", 11, False, COLOR_DARK)
    add_p(tf8, "• Displays local TreeSHAP waterfall feature attributions.", 11, False, COLOR_DARK)
    add_p(tf8, "• SQLite workspace persistence store (WAL mode) saves user accounts, uploaded datasets, and review dispositions.", 11, False, COLOR_DARK)
    add_p(tf8, "", 4, False)
    add_p(tf8, "3. API & Real-Time Streaming:", 12.5, True, COLOR_BLUE)
    add_p(tf8, "• FastAPI REST gateway returns primary model prediction with comparison breakdown and vote fraction.", 11, False, COLOR_DARK)
    add_p(tf8, "• Apache Kafka consumer with transactional SQLite outbox and manual offset commits for crash recovery.", 11, False, COLOR_DARK)

    # =========================================================
    # SLIDE 9: Requirements
    # =========================================================
    print("Building Slide 9: Requirements...")
    slide9 = prs.slides[8]
    add_header_banner(slide9, "Functional Requirements & System Specifications")
    clean_slide_body(slide9)
    update_footers(slide9)

    add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb9_1 = slide9.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True
    add_p(tf9_1, "Implemented Functional Requirements", 15, True, COLOR_NAVY)
    add_p(tf9_1, "• FR-1: Real-Time JSON Ingestion & Streaming: Ingest transaction payloads via synchronous FastAPI REST endpoints (/predict) or event-driven Apache Kafka topics (financial.transactions).", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-2: Four-Model Scoring & Risk Bands: Concurrently evaluate records across 4 models; triage individual predictions into four operational score bands: LOW, MEDIUM, HIGH, and CRITICAL.", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-3: Any-Model Review Rule: Flag transactions for human review whenever any individual model score meets or exceeds its calibrated threshold tau*.", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-4: Explainable Decision Auditing: Generate local TreeSHAP feature attribution charts and quantitative balance discrepancy indicators (deltaOrig, deltaDest) for flagged records.", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-5: Batch Processing & Export: Ingest and score multi-thousand row CSV datasets with persistent history, review status tracking, and filtered CSV export.", 11.5, False, COLOR_DARK)

    add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb9_2 = slide9.shapes.add_textbox(Inches(6.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True
    add_p(tf9_2, "Security, Reliability & Quality Controls", 15, True, COLOR_NAVY)
    add_p(tf9_2, "• NFR-1: Local Cryptographic Hashing: Salted PBKDF2-HMAC-SHA256 (600,000 iterations) for secure local user credential storage in SQLite.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-2: Strict Deployment Secrets: Zero-fallback environment secret enforcement via ${FRAUDGUARD_API_KEY:?Set FRAUDGUARD_API_KEY} in Docker Compose.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-3: Transactional Outbox & Crash Recovery: Persistent SQLite pending_alerts table with manual Kafka offset commits enabling message replay and retry upon consumer crash.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-4: Thread-Safe Persistence: SQLite WAL mode supporting concurrent analyst read/write sessions without database locking.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-5: Responsive Interaction: Direct model tensor invocation delivering responsive sub-second interactive scoring across dashboard workflows and streaming feeds.", 11.5, False, COLOR_DARK)

    # =========================================================
    # SLIDE 10: Modules
    # =========================================================
    print("Building Slide 10: Modules...")
    slide10 = prs.slides[9]
    add_header_banner(slide10, "Software Modules & Implementation")
    clean_slide_body(slide10)
    update_footers(slide10)

    kafka_img = ASSETS_DIR / "kafka_streaming_pipeline.png"
    if kafka_img.exists():
        slide10.shapes.add_picture(str(kafka_img), Inches(0.8), Inches(1.7), Inches(6.5), Inches(4.8))

    add_card(slide10, Inches(7.5), Inches(1.7), Inches(5.0), Inches(4.8))
    tb10 = slide10.shapes.add_textbox(Inches(7.65), Inches(1.8), Inches(4.7), Inches(4.5))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    add_p(tf10, "Implemented System Modules (Repository)", 15, True, COLOR_NAVY)
    add_p(tf10, "• Module 1: Preprocessor & Feature Forensics (src/preprocessor.py): Quantile RobustScaler fitting, median imputation, discrepancy features: deltaOrig = newOrig + amount - oldOrig, deltaDest = oldDest + amount - newDest.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 2: Model Comparison & Evaluation (src/model_comparison.py): Evaluates four diverse models across holdout partitions; calculates precision, recall, F1, PR-AUC, and ROC-AUC.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 3: Explainability Pipeline (src/explainability.py): TreeSHAP local attributions, feature importance waterfall generation, and directional explanation factors. (Note: src/transaction_explanation.py does not exist in checked repository).", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 4: RESTful API Gateway (src/api.py): FastAPI endpoints (/predict, /health, /metrics), API-key authentication, rate-limiting, and primary model prediction with comparison breakdowns.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 5: Fault-Tolerant Streaming (src/kafka_streaming.py): Kafka consumer/producer, transactional outbox (pending_alerts), synchronous broker ACK, and manual offset commits.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 6: Workspace Persistence Store (src/workspace_store.py): Thread-safe SQLite store managing user accounts, uploaded datasets, prediction history, and case review escalation.", 11, False, COLOR_DARK)

    # =========================================================
    # SLIDE 11: Results
    # =========================================================
    print("Building Slide 11: Results...")
    slide11 = prs.slides[10]
    add_header_banner(slide11, "Experimental Evaluation & Performance Results")
    clean_slide_body(slide11)
    update_footers(slide11)

    # Table showing the four individual models (no duplicate table, no weighted fusion here)
    tbl_s11 = slide11.shapes.add_table(5, 10, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.3))
    t11 = tbl_s11.table
    widths11 = [Inches(2.2), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.1), Inches(0.8), Inches(0.8), Inches(0.8)]
    for idx, w in enumerate(widths11):
        t11.columns[idx].width = w

    headers11 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "TP", "FP", "FN"]
    for c_idx, h in enumerate(headers11):
        cell = t11.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Times New Roman"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    res_data = [
        ["Random Forest", "99.60%", "0.8235", "0.9333", "0.8750", "0.9989", "0.9581", "42", "9", "3"],
        ["XGBoost", "99.67%", "0.8889", "0.8889", "0.8889", "0.9968", "0.9418", "40", "5", "5"],
        ["Deep Neural Network", "99.83%", "1.0000", "0.8889", "0.9412", "0.9863", "0.9307", "40", "0", "5"],
        ["Deep Autoencoder", "99.83%", "0.9762", "0.9111", "0.9425", "0.9909", "0.9586", "41", "1", "4"]
    ]

    for r_idx, row in enumerate(res_data):
        for c_idx, val in enumerate(row):
            cell = t11.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx % 2 == 1:
                cell.fill.fore_color.rgb = COLOR_CARD_BG
            else:
                cell.fill.fore_color.rgb = COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Times New Roman"
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_DARK
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    add_card(slide11, Inches(0.8), Inches(4.2), Inches(6.0), Inches(2.4))
    tb11_1 = slide11.shapes.add_textbox(Inches(0.95), Inches(4.25), Inches(5.7), Inches(2.2))
    tf11_1 = tb11_1.text_frame
    tf11_1.word_wrap = True
    add_p(tf11_1, "Key Model Strengths & Individual Trade-Offs", 13.5, True, COLOR_NAVY)
    add_p(tf11_1, "• Random Forest: Highest recall (0.9333, 42 TP) among the four individual models, identifying the maximum number of fraud incidents.", 10.5, False, COLOR_DARK)
    add_p(tf11_1, "• Deep Neural Network: Zero false positives (0 FP, 1.0000 precision) on this particular holdout, eliminating false alarms on legitimate transactions.", 10.5, False, COLOR_DARK)
    add_p(tf11_1, "• Deep Autoencoder: Highest individual-model F1 score (0.9425) on this holdout, balancing high recall (0.9111) with only 1 false positive.", 10.5, False, COLOR_DARK)
    add_p(tf11_1, "• XGBoost: Validation-selected benchmark baseline; selected based on validation set performance, confirming stability on holdout testing.", 10.5, False, COLOR_DARK)

    add_card(slide11, Inches(7.1), Inches(4.2), Inches(5.4), Inches(2.4))
    tb11_2 = slide11.shapes.add_textbox(Inches(7.25), Inches(4.25), Inches(5.1), Inches(2.2))
    tf11_2 = tb11_2.text_frame
    tf11_2.word_wrap = True
    add_p(tf11_2, "Methodological Scope & Results Citation", 13.5, True, COLOR_NAVY)
    add_p(tf11_2, "• Evaluation Holdout: 3,000 untouched holdout transactions (60/20/20 split; 225 total frauds = 1.50% prevalence).", 10.5, False, COLOR_DARK)
    add_p(tf11_2, "• Threshold Selection: Each model is evaluated independently against its own validation-tuned operating threshold (tau*).", 10.5, False, COLOR_DARK)
    add_p(tf11_2, "• Saved Result Files: Verified directly from results/model_results.csv and results/holdout_evaluation_report.json.", 10.5, True, COLOR_HIGHLIGHT)
    add_p(tf11_2, "• Note: Offline weighted fusion, distribution shift, and cost simulations are detailed on dedicated backup slides.", 10.5, False, COLOR_MUTED)

    # =========================================================
    # SLIDE 12: Conclusion
    # =========================================================
    print("Building Slide 12: Conclusion and Future Work...")
    slide12 = prs.slides[11]
    add_header_banner(slide12, "Conclusion & Future Work")
    clean_slide_body(slide12)
    update_footers(slide12)

    add_card(slide12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb12_1 = slide12.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True
    add_p(tf12_1, "Conclusions & Operational Insights", 15, True, COLOR_NAVY)
    add_p(tf12_1, "• Multi-Model Trade-Offs: Supervised tree ensembles offer superior recall and native SHAP interpretability; deep neural architectures achieve zero or near-zero false alarms and unsupervised anomaly detection; no single model dominates across all operating criteria.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• Synthetic Benchmark Limitations: Evaluated on 15,000 locally generated synthetic transactions. While matching mobile payment schemas, synthetic distributions do not reflect live adversarial fraud evolution; continuous telemetry is required.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• Explainability Scope: TreeSHAP local attributions and ledger forensics provide critical decision transparency for human analysts, though technical explainability alone does not constitute statutory legal compliance.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• Implemented Streaming Reliability: Transactional SQLite outbox (pending_alerts), manual offset commits, and synchronous broker acknowledgements provide verified message resilience across tested crash and disconnect scenarios.", 11.5, False, COLOR_DARK)

    add_card(slide12, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb12_2 = slide12.shapes.add_textbox(Inches(6.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True
    add_p(tf12_2, "Future Research Roadmap", 15, True, COLOR_NAVY)
    add_p(tf12_2, "• Graph Neural Networks (GNNs): Incorporate Relational Graph Convolutional Networks (R-GCNs) to detect money mule rings, collusion subgraphs, and synthetic identity fraud networks.", 11.5, False, COLOR_DARK)
    add_p(tf12_2, "• Online Concept Drift Adaptation: Integrate streaming drift detectors (ADWIN, DDM) to trigger automated pipeline recalibration under evolving payment distributions.", 11.5, False, COLOR_DARK)
    add_p(tf12_2, "• Pre-Authorization Telemetry: Expand beyond retrospective ledger balance features by incorporating real-time behavioral velocity and device fingerprinting.", 11.5, False, COLOR_DARK)
    add_p(tf12_2, "• Model Quantization: Deploy ONNX Runtime and TensorRT quantization for ultra-low latency edge inference.", 11.5, False, COLOR_DARK)

    # =========================================================
    # SLIDE 13: Outcomes
    # =========================================================
    print("Building Slide 13: Outcome Achieved...")
    slide13 = prs.slides[12]
    add_header_banner(slide13, "Outcomes Achieved")
    clean_slide_body(slide13)
    update_footers(slide13)

    add_card(slide13, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tb13 = slide13.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(4.5))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    add_p(tf13, "Tangible Project Outcomes & Verified Deliverables", 16, True, COLOR_NAVY)
    add_p(tf13, "", 6, False)

    add_p(tf13, "1. Working Academic Prototype & Microservice:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Multi-container Docker Compose architecture defining API, dashboard, Kafka, and ZooKeeper services with strict deployment secret enforcement (${FRAUDGUARD_API_KEY}).", 12, False, COLOR_DARK)
    add_p(tf13, "• Modular Python software combining FastAPI REST endpoints, Streamlit analyst workspace, and Kafka real-time streaming ingestion.", 12, False, COLOR_DARK)
    add_p(tf13, "", 8, False)

    add_p(tf13, "2. Comprehensive Automated Regression Verification:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• 104 automated tests achieving a 100% PASS rate across 9 test suites covering preprocessing, model scoring, explainability, API security, streaming recovery, and SQLite persistence.", 12, False, COLOR_DARK)
    add_p(tf13, "• Explicitly attributed to the saved test report in results/test_report.txt and results/test_summary.json.", 12, False, COLOR_DARK)
    add_p(tf13, "", 8, False)

    add_p(tf13, "3. Open-Source Codebase & Reproducible Artifacts:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Public GitHub Repository: https://github.com/animeshlabh57-rgb/AI-Based-Financial-Fraud-Detection-System", 12, False, COLOR_DARK)
    add_p(tf13, "• Contains complete modular source code, trained model weights, evaluation benchmark scripts, and Docker orchestrations.", 12, False, COLOR_DARK)
    add_p(tf13, "", 8, False)

    add_p(tf13, "4. Academic Documentation & Publication Trajectory:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Comprehensive 41-page B.Tech Project-I Academic Project Report (BCSE497J) formatted in accordance with VIT IEEE guidelines.", 12, False, COLOR_DARK)
    add_p(tf13, "• Future Publication Trajectory: Target preparation for submission to a peer-reviewed Scopus-indexed conference / journal (kept separate from achieved outcomes).", 12, False, COLOR_DARK)

    # =========================================================
    # SLIDE 14: References
    # =========================================================
    print("Building Slide 14: References...")
    slide14 = prs.slides[13]
    add_header_banner(slide14, "References")
    clean_slide_body(slide14)
    update_footers(slide14)

    add_card(slide14, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tb14 = slide14.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(4.5))
    tf14 = tb14.text_frame
    tf14.word_wrap = True

    references = [
        "[1] A. Dal Pozzolo, G. Boracchi, O. Caelen, et al., 'Credit card fraud detection: A realistic modeling and a novel learning strategy,' IEEE Trans. Neural Netw. Learn. Syst., vol. 29, no. 8, pp. 3784–3797, Aug. 2018.",
        "[2] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, 'Calibrating probability with undersampling for unbalanced classification,' in Proc. IEEE SSCI, 2015, pp. 159–166.",
        "[3] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, 'SMOTE: Synthetic minority over-sampling technique,' J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.",
        "[4] A. Pumsirirat and L. Yan, 'Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine,' Int. J. Adv. Comput. Sci. Appl., vol. 9, no. 1, pp. 18–25, 2018.",
        "[5] L. Breiman, 'Random forests,' Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.",
        "[6] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in Proc. 22nd ACM SIGKDD, 2016, pp. 785–794.",
        "[7] J. An and S. Cho, 'Variational autoencoder based anomaly detection using reconstruction probability,' Special Lecture on IE, vol. 2, no. 1, pp. 1–18, 2015.",
        "[8] S. M. Lundberg et al., 'From local explanations to global understanding with explainable AI for trees,' Nature Mach. Intell., vol. 2, no. 1, pp. 56–67, 2020.",
        "[14] M. Shanaa and S. Abdallah, 'A hybrid anomaly detection framework combining supervised and unsupervised learning for credit card fraud detection,' F1000Research, vol. 14, art. 664, 2025.",
        "[15] S. Iqbal, K. M. Awan, S. Kamal, and Z. Ur Rehman, 'Interpretable ensemble learning models for credit card fraud detection,' Appl. Sci., vol. 15, no. 22, art. 12073, 2025."
    ]

    for ref in references:
        add_p(tf14, ref, 9.5, False, COLOR_DARK)
        add_p(tf14, "", 1.5, False)

    add_p(tf14, "Repository & Saved Experimental Evidence:", 10, True, COLOR_NAVY)
    add_p(tf14, "• GitHub Codebase: https://github.com/animeshlabh57-rgb/AI-Based-Financial-Fraud-Detection-System", 9.5, False, COLOR_DARK)
    add_p(tf14, "• Primary Results: results/model_results.csv  |  Secondary Shift: results/independent_benchmark_report.json  |  Tests: results/test_report.txt", 9.5, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 15: Backup Slide 1: Dataset & Experimental Methodology
    # =========================================================
    print("Building Slide 15: Backup - Dataset & Methodology...")
    slide15 = prs.slides[14]
    add_header_banner(slide15, "Dataset Provenance & Experimental Methodology")
    clean_slide_body(slide15)
    update_footers(slide15)

    card_w15 = Inches(5.6)
    card_h15 = Inches(4.8)

    add_card(slide15, Inches(0.8), top_pos, card_w15, card_h15)
    tb15_1 = slide15.shapes.add_textbox(Inches(0.95), top_pos + Inches(0.15), card_w15 - Inches(0.3), card_h15 - Inches(0.3))
    tf15_1 = tb15_1.text_frame
    tf15_1.word_wrap = True
    add_p(tf15_1, "Benchmark Dataset & Preprocessing Pipeline", 15, True, COLOR_NAVY)
    add_p(tf15_1, "• Dataset Source: 15,000 locally generated synthetic transactions mimicking mobile money transfer semantics (not an official PaySim release or real banking data).", 11.5, False, COLOR_DARK)
    add_p(tf15_1, "• Fraud Prevalence: Exactly 225 fraud cases out of 15,000 transactions (1.50% prevalence; 225/15,000).", 11.5, True, COLOR_HIGHLIGHT)
    add_p(tf15_1, "• Input Schema: 9 attributes including transaction step, transfer type (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN), amount, and balances.", 11.5, False, COLOR_DARK)
    add_p(tf15_1, "• Domain Balance Forensics:", 12, True, COLOR_DARK)
    add_p(tf15_1, "  - Origin balance discrepancy: deltaOrig = newbalanceOrig + amount - oldbalanceOrg", 11, False, COLOR_DARK)
    add_p(tf15_1, "  - Destination balance discrepancy: deltaDest = oldbalanceDest + amount - newbalanceDest", 11, False, COLOR_DARK)
    add_p(tf15_1, "  - Depletion indicator (origDepleted: sender account fully emptied) and destination surge factor.", 11, False, COLOR_DARK)

    add_card(slide15, Inches(6.8), top_pos, card_w15, card_h15)
    tb15_2 = slide15.shapes.add_textbox(Inches(6.95), top_pos + Inches(0.15), card_w15 - Inches(0.3), card_h15 - Inches(0.3))
    tf15_2 = tb15_2.text_frame
    tf15_2.word_wrap = True
    add_p(tf15_2, "Experimental Partitions & Leakage Prevention", 15, True, COLOR_NAVY)
    add_p(tf15_2, "• Strict Stratified Partitions (60 / 20 / 20 Split):", 12, True, COLOR_DARK)
    add_p(tf15_2, "  - Training Set (60%): 9,000 transactions (135 fraud, 8,865 legitimate).", 11, False, COLOR_DARK)
    add_p(tf15_2, "  - Validation Set (20%): 3,000 transactions (45 fraud, 2,955 legitimate).", 11, False, COLOR_DARK)
    add_p(tf15_2, "  - Holdout Test Set (20%): 3,000 transactions (45 fraud, 2,955 legitimate).", 11, False, COLOR_DARK)
    add_p(tf15_2, "", 6, False)
    add_p(tf15_2, "• Leakage-Free Isolation:", 12, True, COLOR_DARK)
    add_p(tf15_2, "  - Quantile RobustScaler parameters and median imputers are fit strictly on the 60% training set.", 11, False, COLOR_DARK)
    add_p(tf15_2, "  - SMOTE-NC synthetic oversampling is restricted strictly to the training partition.", 11, False, COLOR_DARK)
    add_p(tf15_2, "  - Validation and test sets remain completely untouched during all data engineering.", 11, False, COLOR_DARK)
    add_p(tf15_2, "", 6, False)
    add_p(tf15_2, "• Validation-Based Threshold Selection: Model operating thresholds tau* are selected on the validation partition to optimize F1-score; final metrics are evaluated once on untouched holdout test data.", 11, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 16: Backup Slide 2: Additional Offline Weighted-Fusion & Cost Analysis
    # =========================================================
    print("Building Slide 16: Backup - Weighted-Fusion & Cost...")
    slide16 = prs.slides[15]
    add_header_banner(slide16, "Additional Offline Weighted-Fusion & Cost Analysis Experiment")
    clean_slide_body(slide16)
    update_footers(slide16)

    add_card(slide16, Inches(0.8), top_pos, card_w15, card_h15)
    tb16_1 = slide16.shapes.add_textbox(Inches(0.95), top_pos + Inches(0.15), card_w15 - Inches(0.3), card_h15 - Inches(0.3))
    tf16_1 = tb16_1.text_frame
    tf16_1.word_wrap = True
    add_p(tf16_1, "Offline Weighted-Fusion Formulation & Results", 15, True, COLOR_NAVY)
    add_p(tf16_1, "• Distinct Experimental Scope: This is an additional offline research experiment evaluated on the 3,000-transaction holdout, distinct from the dashboard's individual-model review rule.", 11.5, False, COLOR_DARK)
    add_p(tf16_1, "• Fusion Formulation: Combines normalized model scores using fixed empirical weights:", 11.5, False, COLOR_DARK)
    add_p(tf16_1, "  S_fusion(x) = 0.25*S_RF + 0.35*S_XGB + 0.20*S_DNN + 0.20*S_AE", 11.5, True, COLOR_BLUE)
    add_p(tf16_1, "  Evaluated against validation-selected composite threshold tau* = 0.7916.", 11, False, COLOR_DARK)
    add_p(tf16_1, "", 6, False)
    add_p(tf16_1, "• Performance Trade-Off:", 12, True, COLOR_HIGHLIGHT)
    add_p(tf16_1, "  'In the offline weighted-fusion experiment, false positives decreased from 9 to 4 compared with Random Forest (a 55.6% reduction), while true positives decreased from 42 to 40.'", 11, True, COLOR_DARK)
    add_p(tf16_1, "", 6, False)
    add_p(tf16_1, "• Holdout Test Metrics: Precision: 0.9091 | Recall: 0.8889 | F1-Score: 0.8989 | ROC-AUC: 0.9985 | PR-AUC: 0.9555.", 11, False, COLOR_DARK)

    add_card(slide16, Inches(6.8), top_pos, card_w15, card_h15)
    tb16_2 = slide16.shapes.add_textbox(Inches(6.95), top_pos + Inches(0.15), card_w15 - Inches(0.3), card_h15 - Inches(0.3))
    tf16_2 = tb16_2.text_frame
    tf16_2.word_wrap = True
    add_p(tf16_2, "Financial Cost-Benefit Scenario Modeling", 15, True, COLOR_NAVY)
    add_p(tf16_2, "• Asymmetric Cost Framework:", 12, True, COLOR_DARK)
    add_p(tf16_2, "  - Missed Fraud Cost (C_FN) = $2,500 (direct financial loss + chargeback penalty).", 11, False, COLOR_DARK)
    add_p(tf16_2, "  - False Alarm Investigation Cost (C_FP) = $35 (compliance analyst triage overhead).", 11, False, COLOR_DARK)
    add_p(tf16_2, "  - True Positive Triage Cost (C_TP) = $10 (verification workflow cost).", 11, False, COLOR_DARK)
    add_p(tf16_2, "", 6, False)
    add_p(tf16_2, "• Loss Comparison Across Scenarios:", 12, True, COLOR_DARK)
    add_p(tf16_2, "  - Baseline Loss (Unmonitored System): $112,500 across 45 fraud incidents (45 * $2,500).", 11, False, COLOR_DARK)
    add_p(tf16_2, "  - Weighted-Fusion Operational Loss: $13,040 ($12,500 FN + $140 FP + $400 TP).", 11, False, COLOR_DARK)
    add_p(tf16_2, "", 6, False)
    add_p(tf16_2, "• Simulated Financial Outcome: '88.4% simulated loss reduction under assumed costs' (+$99,460 simulated savings; scenario calculation, not measured bank savings).", 11, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 17: Backup Slide 3: Cross-Generator Distribution Shift & Robustness
    # =========================================================
    print("Building Slide 17: Backup - Distribution Shift & Robustness...")
    slide17 = prs.slides[16]
    add_header_banner(slide17, "Cross-Generator Distribution Shift & Robustness")
    clean_slide_body(slide17)
    update_footers(slide17)

    # Secondary shift table showing Recall alongside False Positives
    tbl_s17 = slide17.shapes.add_table(5, 6, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.2))
    t17 = tbl_s17.table
    widths17 = [Inches(2.5), Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.8), Inches(2.0)]
    for idx, w in enumerate(widths17):
        t17.columns[idx].width = w

    headers17 = ["Model Architecture", "Test Recall", "True Positives (TP)", "False Positives (FP)", "False Negatives (FN)", "Operational Behavior"]
    for c_idx, h in enumerate(headers17):
        cell = t17.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Times New Roman"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    shift_data = [
        ["Random Forest", "0.00% (0/15)", "0", "0", "15", "Total Recall Collapse"],
        ["XGBoost", "0.00% (0/15)", "0", "0", "15", "Total Recall Collapse"],
        ["Deep Neural Network", "100.0% (15/15)", "15", "87", "0", "100% Recall, Elevated FPs"],
        ["Deep Autoencoder", "100.0% (15/15)", "15", "336", "0", "100% Recall, Very High FPs"]
    ]

    for r_idx, row in enumerate(shift_data):
        for c_idx, val in enumerate(row):
            cell = t17.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx in (0, 1):
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xED, 0xEC) if c_idx == 1 else COLOR_WHITE
            else:
                cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF9, 0xE7) if c_idx == 3 else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Times New Roman"
            p.font.size = Pt(9.5)
            p.font.bold = (c_idx in (1, 3))
            p.font.color.rgb = COLOR_RED if (r_idx in (0, 1) and c_idx == 1) else (COLOR_AMBER if (r_idx in (2, 3) and c_idx == 3) else COLOR_DARK)
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    add_card(slide17, Inches(0.8), Inches(4.1), Inches(5.7), Inches(2.5))
    tb17_1 = slide17.shapes.add_textbox(Inches(0.95), Inches(4.15), Inches(5.4), Inches(2.3))
    tf17_1 = tb17_1.text_frame
    tf17_1.word_wrap = True
    add_p(tf17_1, "Critical Robustness Findings (1,000 txns, 15 frauds)", 13.5, True, COLOR_NAVY)
    add_p(tf17_1, "• Tree Recall Collapse: Tree ensembles (Random Forest, XGBoost) suffered complete recall failure (0/15 caught) when evaluated against an independently generated synthetic dataset due to orthogonal decision boundary shifts.", 10.5, False, COLOR_DARK)
    add_p(tf17_1, "• Elevated False Positives in Deep Models: While deep models caught 100% of frauds (15/15), the Deep Neural Network produced 87 false positives and the Autoencoder produced 336 false positives.", 10.5, False, COLOR_DARK)
    add_p(tf17_1, "• Honest Generalization Lesson: Reporting 100% recall alone without reporting false positives would be misleading; high false alarm rates create severe analyst fatigue.", 10.5, True, COLOR_RED)

    add_card(slide17, Inches(6.8), Inches(4.1), Inches(5.7), Inches(2.5))
    tb17_2 = slide17.shapes.add_textbox(Inches(6.95), Inches(4.15), Inches(5.4), Inches(2.3))
    tf17_2 = tb17_2.text_frame
    tf17_2.word_wrap = True
    add_p(tf17_2, "Academic Defense & Real-World Implications", 13.5, True, COLOR_NAVY)
    add_p(tf17_2, "• Synthetic Data Limitation: Offline synthetic benchmarks do not guarantee real-world generalization; distributional covariance shifts easily evade static tree thresholds.", 10.5, False, COLOR_DARK)
    add_p(tf17_2, "• Operational Requirement: Automated models must operate in an advisory human-in-the-loop triage capacity rather than an autonomous decision authority.", 10.5, False, COLOR_DARK)
    add_p(tf17_2, "• Pipeline Recommendation: Continuous concept drift monitoring (ADWIN) and adaptive retraining are essential before deploying to live financial environments.", 10.5, False, COLOR_DARK)
    add_p(tf17_2, "• Saved Evidence: results/independent_benchmark_report.json.", 10.5, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 18: Backup Slide 4: Streamlit System Workflow & User Interface Demonstration
    # =========================================================
    print("Building Slide 18: Backup - Streamlit Workflow Demo...")
    slide18 = prs.slides[17]
    add_header_banner(slide18, "Streamlit System Workflow & User Interface Demonstration")
    clean_slide_body(slide18)
    update_footers(slide18)

    workflow_img = ASSETS_DIR / "analyst_workflow_demo.png"
    if workflow_img.exists():
        slide18.shapes.add_picture(str(workflow_img), Inches(0.8), Inches(1.7), Inches(11.7), Inches(4.3))

    add_card(slide18, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.85))
    tb18 = slide18.shapes.add_textbox(Inches(0.95), Inches(6.12), Inches(11.4), Inches(0.75))
    tf18 = tb18.text_frame
    tf18.word_wrap = True
    add_p(tf18, "End-to-End Analyst Workflow Implemented in Streamlit Workspace:", 11, True, COLOR_NAVY)
    add_p(tf18, "(1) Batch CSV upload & schema validation  ->  (2) Four-model evaluation under any-model review rule (LOW/MED/HIGH/CRITICAL)  ->  (3) Flagged transaction inspection with local TreeSHAP waterfall attributions  ->  (4) Audit persistence in thread-safe SQLite workspace store.", 10, False, COLOR_DARK)

    # Save presentation
    docs_copy = WORKSPACE / "docs" / "Review 3- 28-10- 2026 (1).pptx"
    os.makedirs(docs_copy.parent, exist_ok=True)
    prs.save(str(docs_copy))
    print(f"Synced presentation copy to: {docs_copy}")

    final_copy = WORKSPACE / "Review 3- 28-10- 2026 (Final).pptx"
    prs.save(str(final_copy))
    print(f"Saved final presentation copy to: {final_copy}")

    target_out = PPTX_FILE
    try:
        prs.save(str(target_out))
        print(f"\nFinal presentation successfully compiled and saved to:\n{target_out}")
    except PermissionError:
        print(f"\nNOTE: '{PPTX_FILE.name}' is currently locked. Saved to '{final_copy.name}' and '{docs_copy.name}'.")

if __name__ == "__main__":
    build_presentation()
