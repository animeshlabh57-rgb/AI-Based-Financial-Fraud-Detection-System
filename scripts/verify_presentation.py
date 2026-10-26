import pptx

prs = pptx.Presentation("Review 3- 28-10- 2026 (1).pptx")

checks = [
    (1, "Title exact match", "FraudGuard AI: An Explainable Financial Fraud Detection System Using Machine Learning and Deep Learning"),
    (2, "Slide 2 no 'APPROVED' checklist", lambda s: "CHECKLIST" not in s.upper() and "Dr. Erapaneni Gayatri" in s),
    (3, "Slide 3 actual prevalence", "225/15,000"),
    (4, "Slide 4 implemented prototype", "Implemented Academic Prototype"),
    (4, "Slide 4 offline experiment", "Offline Weighted-Fusion Experiment"),
    (5, "Slide 5 Shanaa [14] in table", "[14]"),
    (5, "Slide 5 Iqbal [15] in table", "[15]"),
    (6, "Slide 6 anomaly score normalization", "normalizes reconstruction error into an anomaly score and selects its threshold on validation data"),
    (6, "Slide 6 outbox persistence", "pending_alerts"),
    (7, "Slide 7 potential relevance", "Potential Relevance"),
    (8, "Slide 8 SQLite workspace store", "SQLite workspace persistence store"),
    (9, "Slide 9 4 risk bands", "LOW, MEDIUM, HIGH, and CRITICAL"),
    (10, "Slide 10 explainability.py", "src/explainability.py"),
    (10, "Slide 10 model_comparison.py", "src/model_comparison.py"),
    (11, "Slide 11 Random Forest highest recall", "Highest recall (0.9333, 42 TP)"),
    (11, "Slide 11 DNN 0 FP", "Zero false positives (0 FP, 1.0000 precision)"),
    (11, "Slide 11 Autoencoder highest F1", "Highest individual-model F1 score (0.9425)"),
    (12, "Slide 12 synthetic limitation", "synthetic distributions do not reflect live adversarial fraud evolution"),
    (13, "Slide 13 working academic prototype", "Working Academic Prototype"),
    (13, "Slide 13 Docker Compose 4 services", "API, dashboard, Kafka, and ZooKeeper"),
    (13, "Slide 13 104 tests pass", "104 automated tests achieving a 100% PASS rate"),
    (14, "Slide 14 Reference 14 present", "[14]"),
    (14, "Slide 14 Reference 15 present", "[15]"),
    (15, "Slide 15 15,000 synthetic", "15,000 locally generated synthetic transactions"),
    (15, "Slide 15 60/20/20 split", "60 / 20 / 20 Split"),
    (16, "Slide 16 9 to 4 FP reduction", "false positives decreased from 9 to 4 compared with Random Forest"),
    (16, "Slide 16 88.4% simulated loss reduction", "88.4% simulated loss reduction under assumed costs"),
    (17, "Slide 17 DNN 87 FP", "87"),
    (17, "Slide 17 AE 336 FP", "336"),
    (18, "Slide 18 workflow demo", "analyst workflow")
]

print("Running Detailed Slide Verification...")
all_passed = True
for slide_idx, name, check in checks:
    slide = prs.slides[slide_idx - 1]
    slide_text = ""
    for shape in slide.shapes:
        if shape.has_text_frame:
            slide_text += shape.text_frame.text + "\n"
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    slide_text += cell.text + " "
                slide_text += "\n"
    
    if callable(check):
        res = check(slide_text)
    else:
        res = check.lower() in slide_text.lower()
    
    status = "PASS" if res else "FAIL"
    if not res:
        all_passed = False
    print(f"[{status}] Slide {slide_idx:02d}: {name}")

print("-" * 50)
print(f"Overall Result: {'ALL AUDIT CHECKS PASSED!' if all_passed else 'SOME CHECKS FAILED'}")
