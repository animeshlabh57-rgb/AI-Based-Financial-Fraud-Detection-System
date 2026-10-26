"""
Builds the final Review 3 presentation for VIT BCSE497J Project-I:
Topic: An Explainable Multi-Model Machine Learning Framework for Financial Fraud Detection with Secure API Integration
Presentation File: Review 3- 28-10- 2026 (1).pptx
"""

import os
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

# Standard Color Palette
COLOR_NAVY = RGBColor(0x1F, 0x4E, 0x79)
COLOR_DARK = RGBColor(0x20, 0x20, 0x20)
COLOR_MUTED = RGBColor(0x55, 0x55, 0x55)
COLOR_BLUE = RGBColor(0x2E, 0x75, 0xB6)
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_CARD_BG = RGBColor(0xF4, 0xF7, 0xFA)
COLOR_CARD_BG_ALT = RGBColor(0xEA, 0xF0, 0xF6)
COLOR_BORDER = RGBColor(0xCB, 0xD5, 0xE0)
COLOR_HIGHLIGHT = RGBColor(0x05, 0x96, 0x69) # Emerald Green
COLOR_RED = RGBColor(0xC0, 0x00, 0x00)

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
    """Removes previous body shapes, tables, and text while preserving titles, footers, and logos."""
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
    p_main.text = "An Explainable Multi-Model Machine Learning Framework for Financial Fraud Detection with Secure API Integration"
    p_main.font.name = "Times New Roman"
    p_main.font.size = Pt(21)
    p_main.font.bold = True
    p_main.font.color.rgb = COLOR_DARK
    p_main.alignment = PP_ALIGN.CENTER

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

    # Left: Verification Checklist Card
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(6.8), Inches(4.8), COLOR_CARD_BG, COLOR_BORDER)
    tb2 = slide2.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(6.5), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    add_p(tf2, "Project Guide Review & Verification: APPROVED", 15, True, COLOR_HIGHLIGHT)
    add_p(tf2, "Supervisor: Dr. Erapaneni Gayatri, Assistant Professor Sr. Grade 1, SCOPE", 12.5, True, COLOR_NAVY)
    add_p(tf2, "All Review 3 audit aspects and committee requirements have been addressed:", 11.5, False, COLOR_DARK)
    add_p(tf2, "", 4, False)
    add_p(tf2, "  [x] Literature Survey & Research Gaps: 16 peer-reviewed works evaluated; 3 concrete gaps formulated.", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Leakage-Free Preprocessing: Stratified holdout with training-only parameter estimation and balance forensics.", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Multi-Model Consensus: Fuses Random Forest, XGBoost, Deep MLP, Autoencoder; cuts false alarms by 55.6%.", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Explainability & Regulatory Compliance: TreeSHAP local attributions & ledger balance forensics (GDPR Art. 22).", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Real-Time Streaming & Fault Tolerance: Apache Kafka pipeline with transactional outbox and synchronous broker ACK.", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Comprehensive Automated Regression Testing: 104 automated tests achieving 100% PASS rate across all suites.", 11, False, COLOR_DARK)
    add_p(tf2, "  [x] Complete Academic Deliverables: 41-page IEEE-compliant project report and working software repository finalized.", 11, False, COLOR_DARK)

    # Right: Reposition the guide approval mail image
    for shape in slide2.shapes:
        if shape.name == "Picture 6":
            shape.left = Inches(7.8)
            shape.top = Inches(1.8)
            shape.width = Inches(4.7)
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

    # Card 1: Core Research Aim
    add_card(slide3, Inches(0.8), top_pos, card_w, card_h)
    tb3_1 = slide3.shapes.add_textbox(Inches(0.95), top_pos + Inches(0.12), card_w - Inches(0.3), card_h - Inches(0.24))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True
    add_p(tf3_1, "Core Research Aim", 15, True, COLOR_NAVY)
    add_p(tf3_1, "Formulate, engineer, and empirically benchmark FraudGuard AI—an explainable, multi-model fraud detection framework combining decision tree ensembles (Random Forest, XGBoost) and deep representation models (Deep Neural Network, Deep Autoencoder).", 12, False, COLOR_DARK)
    add_p(tf3_1, "", 6, False)
    add_p(tf3_1, "Address severe real-world class imbalance (<0.2% fraud) without methodological data leakage, using training-restricted oversampling (SMOTE-NC) and RobustScaler quantile normalization.", 12, False, COLOR_DARK)

    # Card 2: Operational & Economic Goals
    add_card(slide3, Inches(4.8), top_pos, card_w, card_h)
    tb3_2 = slide3.shapes.add_textbox(Inches(4.95), top_pos + Inches(0.12), card_w - Inches(0.3), card_h - Inches(0.24))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True
    add_p(tf3_2, "Operational & Cost Goals", 15, True, COLOR_NAVY)
    add_p(tf3_2, "Slash operational false alarms by over 55% relative to individual classifiers through an integer affirmative vote consensus rule with validation-tuned decision thresholds.", 12, False, COLOR_DARK)
    add_p(tf3_2, "", 6, False)
    add_p(tf3_2, "Quantify economic protection via asymmetric financial cost modeling ($2,500 missed fraud cost vs. $35 false alarm cost), achieving +$99,460 net financial savings (88.4% ROI over unmonitored baseline).", 12, False, COLOR_DARK)

    # Card 3: Compliance & Streaming
    add_card(slide3, Inches(8.8), top_pos, card_w, card_h)
    tb3_3 = slide3.shapes.add_textbox(Inches(8.95), top_pos + Inches(0.12), card_w - Inches(0.3), card_h - Inches(0.24))
    tf3_3 = tb3_3.text_frame
    tf3_3.word_wrap = True
    add_p(tf3_3, "Compliance & Microservice", 15, True, COLOR_NAVY)
    add_p(tf3_3, "Deliver actionable compliance explanations complying with statutory regulatory frameworks (GDPR Art. 22) through TreeSHAP attributions and quantitative balance discrepancy forensics.", 12, False, COLOR_DARK)
    add_p(tf3_3, "", 6, False)
    add_p(tf3_3, "Deploy a containerized FastAPI microservice, interactive Streamlit workspace, and an event-driven Apache Kafka real-time streaming pipeline with transactional outbox crash recovery.", 12, False, COLOR_DARK)

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
    add_p(tf4_1, "Problem Context & Methodology", 15, True, COLOR_NAVY)
    add_p(tf4_1, "• Industrial Challenge: Rapid digitization of commerce has triggered massive surges in electronic payment fraud. Heuristic rule systems suffer from rigidity and false alarm fatigue, while black-box deep models create serious legal liability under GDPR Art. 22.", 11.5, False, COLOR_DARK)
    add_p(tf4_1, "• Multi-Model Consensus Architecture: FraudGuard AI unifies 4 diverse paradigms: Random Forest, XGBoost, Deep Multi-Layer Perceptrons, and a Deep Bottleneck Autoencoder. Operates an affirmative vote rule with validation-tuned operating thresholds.", 11.5, False, COLOR_DARK)
    add_p(tf4_1, "• Leakage-Free Data Engineering: Strict stratified holdout preserves pristine test distributions. Combines training-restricted SMOTE-NC with domain-specific balance forensics (origin/destination discrepancies, depletion ratios).", 11.5, False, COLOR_DARK)

    add_card(slide4, Inches(6.8), top_pos, box_w, box_h)
    tb4_2 = slide4.shapes.add_textbox(Inches(6.95), top_pos + Inches(0.15), box_w - Inches(0.3), box_h - Inches(0.3))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    add_p(tf4_2, "Empirical Findings & Production Deployment", 15, True, COLOR_NAVY)
    add_p(tf4_2, "• Primary Benchmark Evaluation (3,000 Holdout txns): Full Consensus achieves Precision: 0.9091, Recall: 0.8889, F1: 0.8989, ROC-AUC: 0.9985, and PR-AUC: 0.9555. Slashes false alarms from 9 (RF) to 4 (55.6% reduction).", 11.5, False, COLOR_DARK)
    add_p(tf4_2, "• Distribution Shift Stress Test (1,000 txns, 15 frauds): Tree models suffered total recall collapse (0/15 caught) while neural representations detected 100% of frauds (15/15 caught), proving offline synthetic benchmarks cannot replace live telemetry.", 11.5, False, COLOR_DARK)
    add_p(tf4_2, "• Production Engineering & Testing: Containerized FastAPI gateway, Streamlit portal, and Kafka streaming engine with outbox crash recovery, verified across 104 automated regression tests (100% PASS) with sub-100 ms inference latency.", 11.5, False, COLOR_DARK)

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
        ["Dal Pozzolo et al. (2018) [1]", "Random Forest, LogReg, SVM", "Undersampling + Calibration", "None", "Batch Evaluation", "No deep architectures; no local explainability."],
        ["Pumsirirat & Yan (2018) [4]", "Autoencoder, RBM", "Normal-only Unsupervised", "None", "Offline Prototype", "Raw uncalibrated reconstruction distance metrics."],
        ["Shanaa & Abdallah (2025) [14]", "Supervised Tree + Anomaly", "SMOTE on whole dataset", "None", "Experimental script", "Severe data leakage risk; lacks web service."],
        ["Iqbal et al. (2025) [15]", "Random Forest, XGBoost", "Standard SMOTE", "SHAP and LIME", "Jupyter Notebook", "Supervised-only; lacks streaming ingestion and security."],
        ["FraudGuard AI (This Work)", "RF, XGBoost, Deep MLP, Autoencoder", "SMOTE-NC (Train Only)", "TreeSHAP + Forensics", "FastAPI + Streamlit + Kafka", "Consensus voting, outbox recovery, 104/104 tests pass."]
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
        ("Gap 1: Uncalibrated Score Fusion in Hybrid Detection Ensembles",
         "Prior hybrid frameworks add unstandardized reconstruction errors (raw MSE ranging from 0.01 to 50.0) directly to classifier probabilities in [0, 1] using arbitrary heuristics.",
         "FraudGuard AI Solution: Formulates validation-tuned threshold calibration (tau*) that maps continuous autoencoder reconstruction error into probability space before composite consensus fusion."),
        ("Gap 2: Methodological Data Leakage & Evaluation Optimism",
         "Widespread published practice applies SMOTE oversampling, standardization, or feature selection across the full dataset prior to splitting, leaking test information into training phases.",
         "FraudGuard AI Solution: Enforces strict parameter freezing strictly on training partitions; holdout test partitions remain untouched and reflect true real-world operational distributions."),
        ("Gap 3: Absence of Production Hardening & Streaming Fault Tolerance",
         "Academic fraud benchmarks predominantly remain as static offline Jupyter notebooks without cryptographic authentication, role-based isolation, or streaming crash resilience.",
         "FraudGuard AI Solution: Deploys an event-driven Apache Kafka pipeline with transactional SQLite outbox (pending_alerts), synchronous broker ACK, and manual offset commits guaranteeing zero message loss.")
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
    # SLIDE 7: Objectives
    # =========================================================
    print("Building Slide 7: Objectives...")
    slide7 = prs.slides[6]
    add_header_banner(slide7, "Project Objectives & Sustainable Development Goals")
    clean_slide_body(slide7, is_slide7=True)
    update_footers(slide7)

    # Left: Objectives Box
    add_card(slide7, Inches(0.8), Inches(1.8), Inches(7.5), Inches(4.8))
    tb7 = slide7.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(7.2), Inches(4.5))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    add_p(tf7, "Technical & Research Objectives", 15, True, COLOR_NAVY)
    add_p(tf7, "1. Implement a leakage-free feature engineering and scaling pipeline, generating domain-specific balance forensics (deltaOrig, deltaDest, R_deplete) and restricting SMOTE-NC to training partitions.", 11.5, False, COLOR_DARK)
    add_p(tf7, "2. Design and optimize four diverse model architectures: Random Forest, XGBoost, Deep Neural Network (MLP), and Deep Bottleneck Autoencoder.", 11.5, False, COLOR_DARK)
    add_p(tf7, "3. Formulate a multi-model consensus voting and scoring engine evaluating four distinct models against validation-tuned thresholds (V(x) in {0..4} and R(x)), routing decisions to operational risk tiers.", 11.5, False, COLOR_DARK)
    add_p(tf7, "4. Integrate explainable AI capabilities, embedding TreeSHAP attributions and domain-specific balance forensics within an automated audit workflow.", 11.5, False, COLOR_DARK)
    add_p(tf7, "5. Develop and containerize a secure, authenticated RESTful API (FastAPI) and interactive dashboard (Streamlit) featuring salted PBKDF2/bcrypt hashing, lockout throttling, RBAC, and SQLite audit storage.", 11.5, False, COLOR_DARK)
    add_p(tf7, "6. Construct an event-driven Apache Kafka real-time streaming pipeline capable of processing live transaction feeds with sub-150 ms end-to-end latency.", 11.5, False, COLOR_DARK)
    add_p(tf7, "7. Validate system robustness through a 104-test automated regression suite and comprehensive latency/concurrency benchmarking.", 11.5, False, COLOR_DARK)

    # Right: SDGs and Outcomes Box
    add_card(slide7, Inches(8.5), Inches(1.8), Inches(4.0), Inches(4.8))
    tb7_r = slide7.shapes.add_textbox(Inches(8.65), Inches(1.9), Inches(3.7), Inches(4.5))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True
    add_p(tf7_r, "Identified SDGs & Outcomes", 15, True, COLOR_NAVY)
    add_p(tf7_r, "• SDG 8 (Decent Work & Economic Growth): Target 8.10 – Strengthen financial capacity and protect transaction integrity against fraud loss.", 11, False, COLOR_DARK)
    add_p(tf7_r, "• SDG 9 (Industry, Innovation & Infrastructure): Target 9.5 – Upgrade technological capabilities and build resilient automated financial infrastructure.", 11, False, COLOR_DARK)
    add_p(tf7_r, "• SDG 16 (Peace, Justice & Strong Institutions): Target 16.5 – Substantially reduce corruption, illicit financial flows, and financial crime.", 11, False, COLOR_DARK)
    add_p(tf7_r, "", 6, False)
    add_p(tf7_r, "Expected Project Outcomes:", 13, True, COLOR_HIGHLIGHT)
    add_p(tf7_r, "• Full-Stack Production Software Product", 11, False, COLOR_DARK)
    add_p(tf7_r, "• Open-Source GitHub Repository", 11, False, COLOR_DARK)
    add_p(tf7_r, "• Comprehensive 41-page Academic Report", 11, False, COLOR_DARK)
    add_p(tf7_r, "• Target: Scopus Indexed Conference / Journal", 11, False, COLOR_DARK)

    # Position SDG pictures inside the right box nicely
    for shape in slide7.shapes:
        if shape.name == "Picture 7":
            shape.left = Inches(8.8)
            shape.top = Inches(5.1)
            shape.width = Inches(1.5)
            shape.height = Inches(1.3)
        elif shape.name == "Picture 8":
            shape.left = Inches(10.6)
            shape.top = Inches(5.1)
            shape.width = Inches(1.5)
            shape.height = Inches(1.3)

    # =========================================================
    # SLIDE 8: Framework / Architecture / Block Diagram
    # =========================================================
    print("Building Slide 8: Architecture Diagram...")
    slide8 = prs.slides[7]
    add_header_banner(slide8, "System Architecture & Consensus Pipeline")
    clean_slide_body(slide8)
    update_footers(slide8)

    arch_img = ASSETS_DIR / "system_architecture.png"
    if arch_img.exists():
        slide8.shapes.add_picture(str(arch_img), Inches(0.8), Inches(1.7), Inches(7.5), Inches(4.8))

    add_card(slide8, Inches(8.5), Inches(1.7), Inches(4.0), Inches(4.8))
    tb8 = slide8.shapes.add_textbox(Inches(8.65), Inches(1.8), Inches(3.7), Inches(4.5))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    add_p(tf8, "Modular 4-Tier Architecture", 15, True, COLOR_NAVY)
    add_p(tf8, "• Tier 1: Ingestion & Preprocessing: Schema validation, RobustScaler IQR bounds, median imputation, and ledger balance discrepancy engineering.", 11, False, COLOR_DARK)
    add_p(tf8, "• Tier 2: Multi-Model Scoring Engine: Concurrent execution of Random Forest, XGBoost, Deep MLP, and Deep Autoencoder with validation-tuned threshold evaluation.", 11, False, COLOR_DARK)
    add_p(tf8, "• Tier 3: Explainability & Auditing: TreeSHAP local attributions and automated ledger balance tampering checks complying with GDPR Art. 22.", 11, False, COLOR_DARK)
    add_p(tf8, "• Tier 4: Service Delivery & Streaming: Containerized FastAPI REST gateway, multi-tenant Streamlit portal, Apache Kafka pub/sub, SQLite persistent store.", 11, False, COLOR_DARK)

    # =========================================================
    # SLIDE 9: Functional Requirements
    # =========================================================
    print("Building Slide 9: Functional Requirements...")
    slide9 = prs.slides[8]
    add_header_banner(slide9, "Functional Requirements & System Specifications")
    clean_slide_body(slide9)
    update_footers(slide9)

    add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb9_1 = slide9.shapes.add_textbox(Inches(0.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True
    add_p(tf9_1, "Core Functional Requirements (FR)", 15, True, COLOR_NAVY)
    add_p(tf9_1, "• FR-1: Real-Time Ingestion: Ingest transactions via synchronous REST API JSON payloads or real-time event-driven Apache Kafka topic streams (financial.transactions).", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-2: Consensus Risk Scoring: Concurrently evaluate records across 4 diverse ML/DL models; compute affirmative vote count V(x) in {0..4} and composite risk score R(x).", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-3: Operational Risk Tiering: Triage transactions into 3 actionable tiers: Low Risk (No Flag), Medium Risk (Flag for Review), and High Risk (Immediate Alert).", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-4: Explainable Forensics: Generate local TreeSHAP waterfall charts and exact balance delta indicators (deltaOrig, deltaDest) for every scored transaction.", 11.5, False, COLOR_DARK)
    add_p(tf9_1, "• FR-5: Batch Processing & Export: Ingest and score multi-thousand row CSV datasets with persistent history and downloadable filtered CSV reports.", 11.5, False, COLOR_DARK)

    add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb9_2 = slide9.shapes.add_textbox(Inches(6.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True
    add_p(tf9_2, "Security & Reliability Specifications (NFR)", 15, True, COLOR_NAVY)
    add_p(tf9_2, "• NFR-1: Cryptographic Authentication: Salted PBKDF2-HMAC-SHA256 (600,000 iterations) and bcrypt hashing; account lockout after 5 consecutive failed logins (15-min lock).", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-2: Strict Deployment Secrets: Zero-fallback environment secret enforcement via ${FRAUDGUARD_API_KEY:?Set FRAUDGUARD_API_KEY} in Docker Compose.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-3: Transactional Outbox & Crash Recovery: Persistent SQLite pending_alerts table ensuring at-least-once alert delivery on worker crash and offset replay.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-4: Sub-100 ms Scoring Latency: Direct model tensor invocation achieves 44.16 ms mean scoring latency with 25.5 requests/sec single-worker throughput.", 11.5, False, COLOR_DARK)
    add_p(tf9_2, "• NFR-5: High Concurrency: SQLite WAL mode supporting concurrent analyst write sessions without thread locking or database corruption.", 11.5, False, COLOR_DARK)

    # =========================================================
    # SLIDE 10: Modules
    # =========================================================
    print("Building Slide 10: Modules...")
    slide10 = prs.slides[9]
    add_header_banner(slide10, "Software Modules & Streaming Pipeline")
    clean_slide_body(slide10)
    update_footers(slide10)

    kafka_img = ASSETS_DIR / "kafka_streaming_pipeline.png"
    if kafka_img.exists():
        slide10.shapes.add_picture(str(kafka_img), Inches(0.8), Inches(1.7), Inches(6.8), Inches(4.8))

    add_card(slide10, Inches(7.8), Inches(1.7), Inches(4.7), Inches(4.8))
    tb10 = slide10.shapes.add_textbox(Inches(7.95), Inches(1.8), Inches(4.4), Inches(4.5))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    add_p(tf10, "Implemented System Modules", 15, True, COLOR_NAVY)
    add_p(tf10, "• Module 1: Preprocessor & Feature Forensics (src/preprocessor.py): Quantile RobustScaler fitting, median imputation, discrepancy features: deltaOrig = newOrig + amount - oldOrig, deltaDest = oldDest + amount - newDest.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 2: Consensus Scoring Engine (src/model_comparison.py): Evaluates 4 diverse models against validation thresholds tau*; maps votes V(x) and composite score R(x) to operational risk tiers.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 3: Explainability & Forensics (src/transaction_explanation.py): Generates local TreeSHAP waterfall plots, global beeswarm rankings, and deterministic direction explanations.", 11, False, COLOR_DARK)
    add_p(tf10, "• Module 4: Fault-Tolerant Streaming & Security (src/kafka_streaming.py, src/api.py, src/workspace_store.py): Real-time Kafka consumer, transactional outbox (pending_alerts), synchronous broker ACK, and SQLite auth.", 11, False, COLOR_DARK)

    # =========================================================
    # SLIDE 11: Experiments and Results
    # =========================================================
    print("Building Slide 11: Experiments and Results...")
    slide11 = prs.slides[10]
    add_header_banner(slide11, "Experimental Evaluation & Performance Results")
    clean_slide_body(slide11)
    update_footers(slide11)

    tbl_s11 = slide11.shapes.add_table(6, 10, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.5))
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
        ["Deep Autoencoder", "99.83%", "0.9762", "0.9111", "0.9425", "0.9909", "0.9586", "41", "1", "4"],
        ["Full Consensus Fusion", "99.70%", "0.9091", "0.8889", "0.8989", "0.9985", "0.9555", "40", "4", "5"]
    ]

    for r_idx, row in enumerate(res_data):
        for c_idx, val in enumerate(row):
            cell = t11.cell(r_idx + 1, c_idx)
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
            p.font.size = Pt(9.5)
            p.font.bold = (r_idx == 4)
            p.font.color.rgb = COLOR_HIGHLIGHT if (r_idx == 4 and c_idx == 0) else COLOR_DARK
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    add_card(slide11, Inches(0.8), Inches(4.4), Inches(5.6), Inches(2.2))
    tb11_1 = slide11.shapes.add_textbox(Inches(0.95), Inches(4.45), Inches(5.3), Inches(2.0))
    tf11_1 = tb11_1.text_frame
    tf11_1.word_wrap = True
    add_p(tf11_1, "Cross-Generator Distribution Shift (1,000 txns)", 13.5, True, COLOR_NAVY)
    add_p(tf11_1, "• Tree Recall Collapse: Random Forest and XGBoost suffered total recall collapse (0/15 caught) due to orthogonal decision boundary shifts.", 10.5, False, COLOR_DARK)
    add_p(tf11_1, "• Neural Representation Robustness: Deep Neural Network and Autoencoder detected all 15 frauds (100% recall), albeit with elevated false alarms.", 10.5, False, COLOR_DARK)
    add_p(tf11_1, "• Key Finding: Offline synthetic benchmarks cannot replace live telemetry; automated models must remain advisory in human-in-the-loop review.", 10.5, True, COLOR_HIGHLIGHT)

    add_card(slide11, Inches(6.8), Inches(4.4), Inches(5.6), Inches(2.2))
    tb11_2 = slide11.shapes.add_textbox(Inches(6.95), Inches(4.45), Inches(5.3), Inches(2.0))
    tf11_2 = tb11_2.text_frame
    tf11_2.word_wrap = True
    add_p(tf11_2, "Financial Cost-Benefit Loss Analysis", 13.5, True, COLOR_NAVY)
    add_p(tf11_2, "• Asymmetric Cost Formulation: Missed Fraud Cost (C_FN) = $2,500; False Alarm Cost (C_FP) = $35; Triage Overhead (C_TP) = $10.", 10.5, False, COLOR_DARK)
    add_p(tf11_2, "• Baseline Loss (No Detection): $112,500 across 45 fraud incidents.", 10.5, False, COLOR_DARK)
    add_p(tf11_2, "• Full Consensus Operational Loss: $13,040 ($12,500 FN + $140 FP + $400 TP).", 10.5, False, COLOR_DARK)
    add_p(tf11_2, "• Net Operational Savings: +$99,460 (88.4% cost reduction over baseline).", 10.5, True, COLOR_HIGHLIGHT)

    # =========================================================
    # SLIDE 12: Conclusion and Future Work
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
    add_p(tf12_1, "Conclusions & Core Achievements", 15, True, COLOR_NAVY)
    add_p(tf12_1, "• Multi-Model Consensus Architecture: Successfully unified supervised tree ensembles with deep neural representation learning, achieving 0.9091 precision, 0.8889 recall, and 0.9555 PR-AUC on holdout data.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• 55.6% False Positive Reduction: Reduced false alarms from 9 down to 4, substantially lightening human compliance review burdens while maintaining high fraud recall.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• Regulatory Explainability: Embedded TreeSHAP feature attributions and quantitative balance discrepancy forensics satisfying statutory GDPR Art. 22 compliance.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• Fault-Tolerant Streaming: Deployed transactional SQLite outbox (pending_alerts) and synchronous broker ACK guaranteeing at-least-once alert delivery on worker crash.", 11.5, False, COLOR_DARK)
    add_p(tf12_1, "• 104 Passing Automated Tests: 100% verification across all unit, integration, and security regression checks.", 11.5, True, COLOR_HIGHLIGHT)

    add_card(slide12, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    tb12_2 = slide12.shapes.add_textbox(Inches(6.95), Inches(1.9), Inches(5.3), Inches(4.5))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True
    add_p(tf12_2, "Limitations & Future Roadmap", 15, True, COLOR_NAVY)
    add_p(tf12_2, "• Current Limitations:", 13, True, COLOR_DARK)
    add_p(tf12_2, "  - Sensitivity of static tree decision thresholds to cross-generator distribution shift.", 11, False, COLOR_DARK)
    add_p(tf12_2, "  - Post-transaction retrospective scope: ledger balance features depend on completed transactions; pre-authorization workflows require real-time velocity features.", 11, False, COLOR_DARK)
    add_p(tf12_2, "", 6, False)
    add_p(tf12_2, "• Future Research Roadmap:", 13, True, COLOR_DARK)
    add_p(tf12_2, "  - Graph Neural Networks (GNNs): Incorporate Relational Graph Convolutional Networks (R-GCNs) to detect money mule rings, collusion graphs, and synthetic identity networks.", 11, False, COLOR_DARK)
    add_p(tf12_2, "  - Adaptive Continual Retraining: Integrate online concept drift detectors (ADWIN, DDM) to trigger automated pipeline recalibration.", 11, False, COLOR_DARK)
    add_p(tf12_2, "  - Hardware Acceleration: Quantize deep models using ONNX Runtime and TensorRT to achieve sub-5 ms edge inference.", 11, False, COLOR_DARK)

    # =========================================================
    # SLIDE 13: Outcome Achieved
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

    add_p(tf13, "1. Enterprise-Hardened Software Product & Cloud Microservice:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Containerized FastAPI microservice with interactive multi-tenant Streamlit analyst workspace and Apache Kafka streaming engine.", 12, False, COLOR_DARK)
    add_p(tf13, "• Multi-container Docker Compose architecture (api, kafka, zookeeper, redis, prometheus) with strict deployment secret enforcement.", 12, False, COLOR_DARK)
    add_p(tf13, "• Verified against an exhaustive 104-test automated unit, integration, and security regression suite (100% PASS rate).", 12, False, COLOR_DARK)
    add_p(tf13, "", 8, False)

    add_p(tf13, "2. Open-Source Codebase & Reproducible Artifacts:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Public GitHub Repository: https://github.com/animeshlabh57-rgb/AI-Based-Financial-Fraud-Detection-System", 12, False, COLOR_DARK)
    add_p(tf13, "• Full modular Python source code, trained model weights, evaluation benchmark scripts, and Docker orchestrations.", 12, False, COLOR_DARK)
    add_p(tf13, "", 8, False)

    add_p(tf13, "3. Academic Research Deliverables & Technical Documentation:", 14, True, COLOR_HIGHLIGHT)
    add_p(tf13, "• Comprehensive 41-page B.Tech Project-I Academic Project Report (BCSE497J) prepared according to VIT IEEE formatting standards.", 12, False, COLOR_DARK)
    add_p(tf13, "• Target: Submission to a Scopus-indexed conference / journal in Financial Machine Learning & Security.", 12, False, COLOR_DARK)

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
        "[7] S. M. Lundberg et al., 'From local explanations to global understanding with explainable AI for trees,' Nature Mach. Intell., vol. 2, no. 1, pp. 56–67, 2020.",
        "[8] T. Saito and M. Rehmsmeier, 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,' PLOS ONE, vol. 10, no. 3, e0118432, 2015."
    ]

    for ref in references:
        add_p(tf14, ref, 10.5, False, COLOR_DARK)
        add_p(tf14, "", 3, False)

    # Save presentation
    docs_copy = WORKSPACE / "docs" / "Review 3- 28-10- 2026 (1).pptx"
    os.makedirs(docs_copy.parent, exist_ok=True)
    prs.save(str(docs_copy))
    print(f"Synced presentation copy to: {docs_copy}")

    target_out = PPTX_FILE
    final_copy = WORKSPACE / "Review 3- 28-10- 2026 (Final).pptx"
    prs.save(str(final_copy))
    print(f"Saved presentation to: {final_copy}")

    try:
        prs.save(str(target_out))
        print(f"\nFinal presentation successfully compiled and saved to:\n{target_out}")
    except PermissionError:
        print(f"\nNOTE: '{PPTX_FILE.name}' is currently open in Microsoft PowerPoint and locked by the OS.")
        print(f"The updated presentation is saved at:")
        print(f" - {final_copy}")
        print(f" - {docs_copy}")
        print(f"Once you close PowerPoint, run this script to update '{PPTX_FILE.name}', or open the final presentation directly!")

if __name__ == "__main__":
    build_presentation()
