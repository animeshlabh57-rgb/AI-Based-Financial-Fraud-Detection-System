"""
Generate high-resolution architecture and UML diagrams for the BCSE497J Project Report.
Uses matplotlib to generate clean, publication-quality figures.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "report_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. System Architecture Diagram
# -------------------------------------------------------------
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_blue = "#1f4e79"
    c_light_blue = "#d9e1f2"
    c_accent = "#2e75b6"
    c_gold = "#ffe699"
    c_green = "#e2efda"
    c_alert = "#fce4d6"

    # Input Box
    ax.add_patch(patches.FancyBboxPatch((0.5, 5.0), 2.0, 1.2, boxstyle="round,pad=0.1", fc=c_light_blue, ec=c_blue, lw=1.5))
    ax.text(1.5, 5.6, "Transaction Payload\n(JSON / CSV Stream)", ha="center", va="center", fontsize=9, fontweight="bold", color=c_blue)

    # Preprocessing Box
    ax.add_patch(patches.FancyBboxPatch((3.2, 5.0), 2.2, 1.2, boxstyle="round,pad=0.1", fc=c_light_blue, ec=c_blue, lw=1.5))
    ax.text(4.3, 5.6, "Feature Engineering\n& Robust Scaling\n(Leakage-Free)", ha="center", va="center", fontsize=9, fontweight="bold", color=c_blue)

    # Arrow from Payload to Preprocessor
    ax.annotate("", xy=(3.2, 5.6), xytext=(2.5, 5.6), arrowprops=dict(arrowstyle="->", lw=2, color=c_blue))

    # Arrow from Preprocessor to Models
    ax.annotate("", xy=(6.0, 5.6), xytext=(5.4, 5.6), arrowprops=dict(arrowstyle="->", lw=2, color=c_blue))

    # Model Containers Box
    ax.add_patch(patches.Rectangle((6.0, 1.8), 3.5, 4.6, fc="#f2f2f2", ec="#808080", lw=1.5, ls="--"))
    ax.text(7.75, 6.15, "Concurrent Multi-Model Scoring Engine", ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")

    # Model 1: XGBoost
    ax.add_patch(patches.FancyBboxPatch((6.3, 4.8), 2.9, 0.9, boxstyle="round,pad=0.08", fc="#ffffff", ec=c_accent, lw=1.2))
    ax.text(7.75, 5.25, "XGBoost Classifier (P_XGB)\n120 Trees, Learning Rate 0.08", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Model 2: Random Forest
    ax.add_patch(patches.FancyBboxPatch((6.3, 3.5), 2.9, 0.9, boxstyle="round,pad=0.08", fc="#ffffff", ec=c_accent, lw=1.2))
    ax.text(7.75, 3.95, "Random Forest (P_RF)\n100-150 Trees, Balanced Subsample", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Model 3: Deep Autoencoder + Platt
    ax.add_patch(patches.FancyBboxPatch((6.3, 2.2), 2.9, 0.9, boxstyle="round,pad=0.08", fc="#ffffff", ec=c_accent, lw=1.2))
    ax.text(7.75, 2.65, "Deep Autoencoder + Platt Calibrator\ne(x) -> Logistic P_AE in [0, 1]", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Weighted Consensus Node
    ax.add_patch(patches.Circle((4.3, 2.7), 1.0, fc=c_gold, ec="#b25900", lw=1.8))
    ax.text(4.3, 2.7, "Weighted\nConsensus\nR(x)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#7f3300")

    # Arrows from models to consensus
    ax.annotate("", xy=(5.3, 2.9), xytext=(6.3, 5.25), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    ax.annotate("", xy=(5.3, 2.7), xytext=(6.3, 3.95), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    ax.annotate("", xy=(5.3, 2.5), xytext=(6.3, 2.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    # Output Decision Box
    ax.add_patch(patches.FancyBboxPatch((0.5, 2.1), 2.4, 1.2, boxstyle="round,pad=0.1", fc=c_green, ec="#385723", lw=1.5))
    ax.text(1.7, 2.7, "Risk Tier & Action Mapping\n[ALLOW / CHALLENGE /\nREVIEW / BLOCK]", ha="center", va="center", fontsize=9, fontweight="bold", color="#274e13")

    # Arrow from Consensus to Decision
    ax.annotate("", xy=(2.9, 2.7), xytext=(3.3, 2.7), arrowprops=dict(arrowstyle="->", lw=2, color="#274e13"))

    # Explainability Box (On Request)
    ax.add_patch(patches.FancyBboxPatch((2.2, 0.4), 4.2, 1.1, boxstyle="round,pad=0.1", fc=c_alert, ec="#c00000", lw=1.5))
    ax.text(4.3, 0.95, "Explainable AI Engine (On-Demand)\nTreeSHAP Global / LIME Local / Balance Discrepancy", ha="center", va="center", fontsize=9, fontweight="bold", color="#900000")

    # Arrow from Decision/Consensus to Explainability
    ax.annotate("On Flagged / Request", xy=(4.3, 1.5), xytext=(4.3, 1.7), ha="center", fontsize=8, color="#900000")
    ax.annotate("", xy=(4.3, 1.5), xytext=(4.3, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color="#c00000"))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "system_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# 2. Data Flow Diagram (DFD Level 1)
# -------------------------------------------------------------
def generate_dfd_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Entity: Banking Ingestion Stream
    ax.add_patch(patches.Rectangle((0.4, 5.0), 1.8, 1.2, fc="#fff2cc", ec="#d6b656", lw=1.5))
    ax.text(1.3, 5.6, "Banking Client /\nKafka Stream\n[External Entity]", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Process 1.0: Ingestion & Schema Validation
    ax.add_patch(patches.Circle((4.0, 5.6), 0.9, fc="#dae8fc", ec="#6c8ebf", lw=1.5))
    ax.text(4.0, 5.6, "1.0\nValidate &\nStandardize", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Process 2.0: Feature Forensics
    ax.add_patch(patches.Circle((7.2, 5.6), 0.9, fc="#dae8fc", ec="#6c8ebf", lw=1.5))
    ax.text(7.2, 5.6, "2.0\nCompute\nForensic Diffs", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Data Store D1: Scaler Parameters
    ax.plot([5.2, 6.2], [4.4, 4.4], color="#333333", lw=1.5)
    ax.plot([5.2, 6.2], [3.9, 3.9], color="#333333", lw=1.5)
    ax.text(5.7, 4.15, "D1: Train Stats", ha="center", va="center", fontsize=8)

    # Process 3.0: Concurrent Multi-Model Scoring
    ax.add_patch(patches.Circle((7.2, 3.0), 0.9, fc="#d5e8d4", ec="#82b366", lw=1.5))
    ax.text(7.2, 3.0, "3.0\nMulti-Model\nScoring", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Data Store D2: Model Checkpoints & Platt Weights
    ax.plot([8.6, 9.7], [3.3, 3.3], color="#333333", lw=1.5)
    ax.plot([8.6, 9.7], [2.8, 2.8], color="#333333", lw=1.5)
    ax.text(9.15, 3.05, "D2: Model Artifacts", ha="center", va="center", fontsize=8)

    # Process 4.0: Consensus Fusion & Risk Tiers
    ax.add_patch(patches.Circle((4.0, 3.0), 0.9, fc="#d5e8d4", ec="#82b366", lw=1.5))
    ax.text(4.0, 3.0, "4.0\nConsensus\nFusion R(x)", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Process 5.0: Explainability & Audit Log
    ax.add_patch(patches.Circle((4.0, 1.1), 0.8, fc="#ffe6cc", ec="#d79b00", lw=1.5))
    ax.text(4.0, 1.1, "5.0\nSHAP/LIME &\nAudit Log", ha="center", va="center", fontsize=8, fontweight="bold")

    # Entity: Fraud Analyst Dashboard
    ax.add_patch(patches.Rectangle((0.4, 2.4), 1.8, 1.2, fc="#fff2cc", ec="#d6b656", lw=1.5))
    ax.text(1.3, 3.0, "Security Analyst /\nDashboard\n[External Entity]", ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Data Store D3: SQLite Audit Logs
    ax.plot([6.2, 7.5], [1.3, 1.3], color="#333333", lw=1.5)
    ax.plot([6.2, 7.5], [0.8, 0.8], color="#333333", lw=1.5)
    ax.text(6.85, 1.05, "D3: Audit SQLite", ha="center", va="center", fontsize=8)

    # Connecting Flow Arrows
    ax.annotate("Raw Payload", xy=(3.1, 5.6), xytext=(2.2, 5.6), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Clean Vector", xy=(6.3, 5.6), xytext=(4.9, 5.6), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("14-D Vector", xy=(7.2, 3.9), xytext=(7.2, 4.7), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Probabilities", xy=(4.9, 3.0), xytext=(6.3, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Alert / Tier", xy=(2.2, 3.0), xytext=(3.1, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Audit Record", xy=(4.0, 1.9), xytext=(4.0, 2.1), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Log Write", xy=(6.2, 1.05), xytext=(4.8, 1.05), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Audit Query", xy=(1.3, 2.4), xytext=(3.2, 1.1), arrowprops=dict(arrowstyle="<-", lw=1.2, ls="--"))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "data_flow_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# 3. Use Case Diagram
# -------------------------------------------------------------
def generate_usecase_diagram():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Boundary Box: Fraud Detection System
    ax.add_patch(patches.Rectangle((3.0, 0.4), 6.5, 6.2, fc="#fbfbfb", ec="#555555", lw=1.5))
    ax.text(6.25, 6.3, "FraudGuard AI / Enterprise System Boundary", ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")

    # Actors: Left side
    # Actor 1: Fraud Analyst
    ax.plot([1.2, 1.2], [5.2, 4.7], color="#1f4e79", lw=2)  # spine
    ax.plot([0.8, 1.6], [5.0, 5.0], color="#1f4e79", lw=2)  # arms
    ax.plot([1.2, 0.9], [4.7, 4.3], color="#1f4e79", lw=2)  # left leg
    ax.plot([1.2, 1.5], [4.7, 4.3], color="#1f4e79", lw=2)  # right leg
    ax.add_patch(patches.Circle((1.2, 5.4), 0.2, fc="#1f4e79", ec="#1f4e79"))
    ax.text(1.2, 4.0, "Fraud Analyst /\nCompliance Officer", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1f4e79")

    # Actor 2: System Administrator
    ax.plot([1.2, 1.2], [2.2, 1.7], color="#385723", lw=2)
    ax.plot([0.8, 1.6], [2.0, 2.0], color="#385723", lw=2)
    ax.plot([1.2, 0.9], [1.7, 1.3], color="#385723", lw=2)
    ax.plot([1.2, 1.5], [1.7, 1.3], color="#385723", lw=2)
    ax.add_patch(patches.Circle((1.2, 2.4), 0.2, fc="#385723", ec="#385723"))
    ax.text(1.2, 1.0, "System Admin /\nCore Banking API", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#385723")

    # Use Cases (Ellipses)
    use_cases = [
        ("UC1: Authenticate with PBKDF2/Bcrypt", (5.8, 5.6)),
        ("UC2: Single Transaction Real-Time Scoring", (6.2, 4.7)),
        ("UC3: Batch Transaction CSV Ingestion", (6.2, 3.8)),
        ("UC4: Inspect Forensic Balance & SHAP/LIME", (6.2, 2.9)),
        ("UC5: Monitor Kafka Live Streaming Events", (6.2, 2.0)),
        ("UC6: Query Multi-Tenant Audit History", (6.0, 1.1)),
    ]

    for label, pos in use_cases:
        ax.add_patch(patches.Ellipse(pos, 3.6, 0.65, fc="#e1d5e7", ec="#9673a6", lw=1.2))
        ax.text(pos[0], pos[1], label, ha="center", va="center", fontsize=8, fontweight="bold")

    # Connections from Fraud Analyst
    ax.plot([1.6, 4.0], [5.0, 5.6], color="#555555", lw=1.2)
    ax.plot([1.6, 4.4], [4.9, 4.7], color="#555555", lw=1.2)
    ax.plot([1.6, 4.4], [4.8, 3.8], color="#555555", lw=1.2)
    ax.plot([1.6, 4.4], [4.7, 2.9], color="#555555", lw=1.2)
    ax.plot([1.6, 4.4], [4.6, 2.0], color="#555555", lw=1.2)

    # Connections from Admin / Core Banking
    ax.plot([1.6, 4.0], [2.1, 5.6], color="#555555", lw=1.2)
    ax.plot([1.6, 4.4], [2.0, 4.7], color="#555555", lw=1.2)
    ax.plot([1.6, 4.2], [1.9, 1.1], color="#555555", lw=1.2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "use_case_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# 4. Class Diagram
# -------------------------------------------------------------
def generate_class_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_class_box(x, y, w, h, class_name, attributes, methods):
        # Header
        ax.add_patch(patches.Rectangle((x, y + h - 0.5), w, 0.5, fc="#dae8fc", ec="#6c8ebf", lw=1.2))
        ax.text(x + w / 2, y + h - 0.25, class_name, ha="center", va="center", fontsize=8.5, fontweight="bold")
        # Body
        ax.add_patch(patches.Rectangle((x, y), w, h - 0.5, fc="#ffffff", ec="#6c8ebf", lw=1.2))
        # Attributes
        curr_y = y + h - 0.7
        for attr in attributes:
            ax.text(x + 0.1, curr_y, attr, ha="left", va="center", fontsize=7.2, family="monospace")
            curr_y -= 0.22
        # Separator line
        ax.plot([x, x + w], [curr_y + 0.05, curr_y + 0.05], color="#6c8ebf", lw=0.8)
        curr_y -= 0.18
        # Methods
        for method in methods:
            ax.text(x + 0.1, curr_y, method, ha="left", va="center", fontsize=7.2, family="monospace")
            curr_y -= 0.22

    # Class 1: TransactionPreprocessor
    draw_class_box(0.5, 3.8, 2.7, 2.8, "TransactionPreprocessor",
                   ["- median_imputer: dict", "- robust_scaler: Scaler", "- feature_names: list[14]"],
                   ["+ fit(X_train: DataFrame)", "+ transform(payload: dict)", "+ compute_discrepancies()"])

    # Class 2: MultiModelPredictor (Base)
    draw_class_box(3.7, 3.8, 2.7, 2.8, "ConsensusEngine",
                   ["- rf_model: RandomForest", "- xgb_model: XGBClassifier", "- autoencoder: MLPRegressor", "- platt_calibrator: Logistic"],
                   ["+ predict_all(X: array)", "+ compute_consensus(p_vec)", "+ assign_risk_tier(score)", "+ explain_shap(sample)"])

    # Class 3: KafkaStreamingConsumer
    draw_class_box(6.8, 3.8, 2.8, 2.8, "FraudStreamingConsumer",
                   ["- broker: KafkaBroker", "- consensus: ConsensusEngine", "- alert_topic: str", "- metrics: LatencyTracker"],
                   ["+ start_consumer()", "+ process_event(event_json)", "+ emit_alert(verdict)", "+ track_latency_p95()"])

    # Class 4: SecurityManager & WorkspaceDB
    draw_class_box(1.5, 0.4, 3.2, 2.6, "SecurityManager",
                   ["- db_path: str", "- bcrypt_salt: bytes", "- lockout_threshold: int = 5"],
                   ["+ hash_password(pwd: str)", "+ authenticate(user, pwd)", "+ check_lockout(user)", "+ log_audit_trail(record)"])

    # Class 5: ExplanationPayload
    draw_class_box(5.5, 0.4, 3.2, 2.6, "ForensicAuditRecord",
                   ["+ transaction_id: UUID", "+ consensus_score: float", "+ risk_tier: RiskTier", "+ shap_values: list[dict]"],
                   ["+ to_json(): str", "+ get_top_contributors(k=3)", "+ flag_balance_invariants()"])

    # Association lines
    ax.annotate("", xy=(3.7, 5.2), xytext=(3.2, 5.2), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(6.8, 5.2), xytext=(6.4, 5.2), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(3.1, 3.8), xytext=(3.1, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(5.0, 3.8), xytext=(6.5, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "class_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# 5. Sequence Diagram
# -------------------------------------------------------------
def generate_sequence_diagram():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    lifelines = [
        ("Client / Stream", 1.2),
        ("FastAPI / Gateway", 3.4),
        ("Preprocessor", 5.4),
        ("Model Engines", 7.4),
        ("SQLite / Audit", 9.0),
    ]

    for name, x in lifelines:
        # Header box
        ax.add_patch(patches.Rectangle((x - 0.8, 6.2), 1.6, 0.55, fc="#dae8fc", ec="#6c8ebf", lw=1.2))
        ax.text(x, 6.47, name, ha="center", va="center", fontsize=8, fontweight="bold")
        # Lifeline dashed line
        ax.plot([x, x], [0.6, 6.2], color="#888888", ls="--", lw=1.0)

    # Sequence steps (top to bottom)
    steps = [
        (5.7, 1.2, 3.4, "1: POST /predict (Payload JSON)"),
        (5.1, 3.4, 5.4, "2: transform(payload)"),
        (4.5, 5.4, 3.4, "3: return scaled 14-D vector"),
        (3.9, 3.4, 7.4, "4: concurrent_score(vector)"),
        (3.3, 7.4, 3.4, "5: return [P_XGB, P_RF, P_AE]"),
        (2.7, 3.4, 3.4, "6: compute R(x) & Risk Tier"),
        (2.1, 3.4, 9.0, "7: INSERT audit_log record"),
        (1.5, 9.0, 3.4, "8: ack (logged_id)"),
        (0.9, 3.4, 1.2, "9: 200 OK {score, tier, action}"),
    ]

    for y, x1, x2, label in steps:
        if x1 == x2:  # Self-call
            ax.plot([x1, x1 + 0.5, x1 + 0.5, x1], [y + 0.15, y + 0.15, y - 0.15, y - 0.15], color="#1f4e79", lw=1.2)
            ax.annotate("", xy=(x1, y - 0.15), xytext=(x1 + 0.2, y - 0.15), arrowprops=dict(arrowstyle="->", lw=1.2, color="#1f4e79"))
            ax.text(x1 + 0.6, y, label, ha="left", va="center", fontsize=7.5)
        else:
            is_return = "return" in label or "ack" in label or "200" in label
            ls = "--" if is_return else "-"
            ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", lw=1.2, ls=ls, color="#1f4e79"))
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y + 0.15, label, ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sequence_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# -------------------------------------------------------------
# 6. Real-Time Kafka Streaming Architecture Diagram
# -------------------------------------------------------------
def generate_kafka_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    c_blue = "#1f4e79"
    c_orange = "#d79b00"
    c_green = "#274e13"

    # Producer
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.4), 2.2, 1.6, boxstyle="round,pad=0.1", fc="#dae8fc", ec="#6c8ebf", lw=1.5))
    ax.text(1.6, 4.4, "TransactionStreamProducer", ha="center", va="center", fontsize=9, fontweight="bold", color=c_blue)
    ax.text(1.6, 3.8, "- Schema Validation\n- Event Timestamping\n- Deterministic Partition Keys", ha="center", va="center", fontsize=7.5)

    # Kafka Topic 1: financial.transactions
    ax.add_patch(patches.Rectangle((3.6, 3.6), 2.8, 1.2, fc="#ffe6cc", ec=c_orange, lw=1.5))
    ax.text(5.0, 4.4, "Apache Kafka Topic", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_orange)
    ax.text(5.0, 3.9, "financial.transactions\n(Partitioned JSON Stream)", ha="center", va="center", fontsize=8)

    # Arrows Producer -> Topic 1
    ax.annotate("", xy=(3.6, 4.2), xytext=(2.7, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color=c_blue))

    # Consumer
    ax.add_patch(patches.FancyBboxPatch((7.2, 3.2), 2.4, 2.0, boxstyle="round,pad=0.1", fc="#d5e8d4", ec="#82b366", lw=1.5))
    ax.text(8.4, 4.7, "FraudStreamingConsumer", ha="center", va="center", fontsize=9, fontweight="bold", color=c_green)
    ax.text(8.4, 3.9, "- 4-Model Consensus Engine\n- Balance Forensic Invariants\n- Latency Tracking (~104 ms)\n- Fallback Broker Switching", ha="center", va="center", fontsize=7.5)

    # Arrow Topic 1 -> Consumer
    ax.annotate("", xy=(7.2, 4.2), xytext=(6.4, 4.2), arrowprops=dict(arrowstyle="->", lw=2, color=c_orange))

    # Kafka Topic 2: financial.fraud-alerts
    ax.add_patch(patches.Rectangle((3.6, 0.8), 2.8, 1.2, fc="#f8cecc", ec="#b85450", lw=1.5))
    ax.text(5.0, 1.6, "Apache Kafka Topic", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#b85450")
    ax.text(5.0, 1.1, "financial.fraud-alerts\n(Structured Actionable Alerts)", ha="center", va="center", fontsize=8)

    # Consumer -> Topic 2
    ax.annotate("", xy=(5.0, 2.0), xytext=(8.4, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color=c_green))

    # Dashboard / Analyst
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.8), 2.2, 1.3, boxstyle="round,pad=0.1", fc="#e1d5e7", ec="#9673a6", lw=1.5))
    ax.text(1.6, 1.6, "Analyst Dashboard\n& SOC Real-Time Feed", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#662d91")
    ax.text(1.6, 1.1, "Interactive Gauge, Stream View", ha="center", va="center", fontsize=7.5)

    # Topic 2 -> Dashboard
    ax.annotate("", xy=(2.7, 1.4), xytext=(3.6, 1.4), arrowprops=dict(arrowstyle="->", lw=2, color="#b85450"))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "kafka_streaming_pipeline.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    generate_system_architecture()
    generate_dfd_diagram()
    generate_usecase_diagram()
    generate_class_diagram()
    generate_sequence_diagram()
    generate_kafka_pipeline_diagram()
    print("All diagrams generated successfully in:", OUTPUT_DIR)
