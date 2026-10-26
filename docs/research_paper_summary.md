# Comparative Analysis of Supervised Machine Learning and Semi-Supervised Deep Representations for Financial Fraud Detection in Imbalanced Transaction Streams

**Project:** FraudGuard AI  
**Academic Target Domain:** Financial Technology, Machine Learning, Anomaly Detection  
**Evaluation Standard:** 69 / 69 Automated Unit & Regression Tests Passing  

---

## Abstract

Financial transaction fraud detection presents an acute challenge characterized by extreme class imbalance, concept drift, and the demand for sub-second real-time scoring. This paper presents the design, mathematical formulation, and empirical evaluation of **FraudGuard AI**, an end-to-end financial fraud detection system combining supervised tree ensembles (**Random Forest**, **XGBoost**) and multi-layer deep neural networks (**Deep Multilayer Perceptron**, **Deep Autoencoder**). To prevent data leakage, we enforce a strict partitioned feature pipeline where median/mode imputation, `RobustScaler` transformations, and synthetic balancing via **SMOTE-NC** are fitted strictly on training partitions (60%), leaving validation (20%) and final test (20%) partitions uncorrupted. 

Evaluated on a benchmark of 15,000 transactions (1.50% base fraud prevalence, 65.7:1 imbalance ratio), our empirical findings demonstrate that:
1. The supervised **Deep Neural Network** achieves perfect precision (1.0000, 0 false positives) at an optimal threshold of 0.9993.
2. The semi-supervised **Deep Autoencoder**, trained exclusively on non-fraudulent instances, produces superior boundary generalization, achieving the highest overall test $F_1$-score (0.9425) and PR-AUC (0.9586).
3. **XGBoost** demonstrates the strongest validation PR-AUC (0.9020), providing tree SHAP interpretability.

We further demonstrate real-time deployment via an event-driven Apache Kafka producer-consumer streaming architecture processing live transactions at ~100ms latency, paired with a multi-tenant cryptographic analyst workspace.

**Keywords:** Financial Fraud Detection, Extreme Class Imbalance, SMOTE-NC, Deep Autoencoder, Anomaly Detection, Tree SHAP, Real-Time Kafka Streaming.

---

## 1. Introduction & Theoretical Problem Formulation

### 1.1 The Fraud Detection Dilemma
In high-volume electronic payment networks, fraudulent transactions constitute a minute fraction (often $< 2\%$) of total volume. Standard loss functions (e.g., standard cross-entropy or Mean Squared Error) trained on imbalanced data exhibit a pathology wherein classifiers optimize naive accuracy by collapsing to the majority class:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \approx \frac{TN}{TN + FP} \to 1.0$$
While minimizing statistical risk, this behavior results in catastrophic false-negative rates ($FN$), allowing illicit transfers to clear undetected.

### 1.2 Mathematical Problem Definition
Let each transaction $i$ be represented by a feature vector $x_i \in \mathcal{X} \subset \mathbb{R}^d$ and a ground-truth binary label $y_i \in \{0, 1\}$, where $y_i = 1$ denotes fraudulent activity and $y_i = 0$ denotes legitimate transfer. The training set is defined as:
$$\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{N}, \quad \text{where } \frac{\sum_{i} \mathbb{I}(y_i = 1)}{N} = \alpha \ll 0.05$$
The objective is to learn a decision function $f: \mathcal{X} \to [0, 1]$ parameterizing an anomaly or probability score, coupled with a validation-optimized decision threshold $\tau \in (0, 1)$ such that:
$$\hat{y}_i = \mathbb{I}(f(x_i) \ge \tau)$$
maximizing the Area Under the Precision-Recall Curve (PR-AUC / Average Precision) and the $F_1$-measure:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

---

## 2. Dataset Characteristics & Integrity Protocol

### 2.1 Benchmark Topology
The system operates on an audited synthetic financial benchmark with deterministic SHA-256 fingerprint:
- **Checksum:** `c0b407c3bb35167efa0ffbb9ed0a6151980de600a90181151f58899c595852ba`
- **Total Transactions ($N$):** 15,000
- **Total Fraud ($N_1$):** 225 ($1.50\%$)
- **Legitimate Records ($N_0$):** 14,775 ($98.50\%$)
- **Native Imbalance Ratio:** $65.7 : 1$

### 2.2 Raw Transaction Schema
Raw attributes represent temporal steps, transfer types, monetary volumes, and bank account balances:
1. `step`: Positional unit of simulated time (1 step $\approx$ 1 hour).
2. `type`: Categorical transfer modality $\in \{\text{CASH\_IN}, \text{CASH\_OUT}, \text{DEBIT}, \text{PAYMENT}, \text{TRANSFER}\}$.
3. `amount`: Monetary transfer value in local currency units.
4. `oldbalanceOrg`: Sender opening balance prior to transaction execution.
5. `newbalanceOrig`: Sender closing balance subsequent to transaction execution.
6. `oldbalanceDest`: Recipient opening balance prior to transaction receipt.
7. `newbalanceDest`: Recipient closing balance subsequent to transaction receipt.

---

## 3. Data Engineering & Leakage Prevention Protocol

### 3.1 Strict Stratified Partitioning
To guarantee reproducible generalization without test data leakage, transactions are partitioned into three stratified splits using a fixed random seed ($\text{seed} = 42$):
- **Training Set (60%):** 9,000 transactions (135 fraud cases)
- **Validation Set (20%):** 3,000 transactions (45 fraud cases) — reserved exclusively for threshold tuning and early stopping.
- **Held-Out Test Set (20%):** 3,000 transactions (45 fraud cases) — untouched during all feature fitting, SMOTE oversampling, training, and threshold selection.

### 3.2 Feature Engineering: Balance Discrepancy Forensics
Raw balances often obscure illicit behavioral mechanics. We derive domain-specific financial features:
1. **Origin Balance Discrepancy ($\Delta_{\text{orig}}$):**
   $$\Delta_{\text{orig}} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$$
   In legitimate debit transactions, $\Delta_{\text{orig}} = 0$. Deviations indicate balance tampering or unrecorded overdrafts.
2. **Destination Balance Discrepancy ($\Delta_{\text{dest}}$):**
   $$\Delta_{\text{dest}} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$$
   In legitimate transfers, recipient balance increments by the transaction amount ($\Delta_{\text{dest}} = 0$). Discrepancies emerge when recipient accounts fail to credit or intermediate mule accounts skim funds.
3. **Amount-to-Balance Depletion Ratio ($R_{\text{deplete}}$):**
   $$R_{\text{deplete}} = \frac{\text{amount}}{\text{oldbalanceOrg} + \epsilon}, \quad \epsilon = 10^{-5}$$
   Fraudulent transfers frequently exhibit $R_{\text{deplete}} \approx 1.0$ (complete account drainage).

Coupled with one-hot encoding across the 5 transaction categories, the processed feature vector spans $d = 14$ continuous dimensions.

### 3.3 Leakage-Free Preprocessing & SMOTE-NC
1. **Imputation:** Median imputation for numerical features and mode imputation for categorical attributes are computed strictly on $\mathcal{D}_{\text{train}}$ and applied to downstream splits.
2. **Robust Scaling:** Continuous features are standardized using `RobustScaler`, centering by the median and scaling by the Interquartile Range ($\text{IQR} = Q_3 - Q_1$), mitigating extreme financial outlier leverage:
   $$z = \frac{x - \text{median}(x_{\text{train}})}{\text{IQR}(x_{\text{train}})}$$
3. **Synthetic Minority Oversampling (SMOTE-NC):**
   Applied exclusively to $\mathcal{D}_{\text{train}}$. Synthetic instances are interpolated along the Euclidean feature space for continuous variables while retaining categorical mode assignments.
   - Before SMOTE: 8,865 legitimate vs. 135 fraud ($65.7 : 1$)
   - After SMOTE: 8,865 legitimate vs. 8,865 fraud ($1 : 1$)
   - **Crucial Invariant:** Validation and test partitions remain at their pristine $1.50\%$ natural distribution.

---

## 4. Model Architectures & Theoretical Formulations

The research evaluates four distinct architectures spanning supervised tree ensembles, discriminative deep networks, and unsupervised representation learning:

### 4.1 Random Forest (ML Ensemble)
An ensemble of $B$ bootstrap-aggregated decision trees trained on balanced data:
$$\hat{p}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^{B} p_b(x)$$
Each split minimizes the Gini impurity index:
$$I_G(p) = 1 - \sum_{k \in \{0, 1\}} p_k^2$$

### 4.2 XGBoost (Extreme Gradient Boosted Trees)
An additive ensemble minimizing a regularized second-order Taylor approximation of the logistic objective:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where $g_i = \partial_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$, with tree complexity penalty $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$.

### 4.3 Deep Neural Network (Supervised MLPClassifier)
A deep discriminative feed-forward neural network trained on SMOTE-balanced data with input normalization clipped to $[-12, 12]$:
- **Topology:** $\text{Input}(14) \to \text{Dense}(64, \text{ReLU}) \to \text{Dense}(32, \text{ReLU}) \to \text{Dense}(16, \text{ReLU}) \to \text{Dense}(1, \text{Sigmoid})$
- **Parameters:** 3,585 trainable weights and biases
- **Optimization:** Adam optimizer ($\beta_1 = 0.9, \beta_2 = 0.999$), Binary Cross-Entropy loss:
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Early Stopping:** Best validation PR-AUC checkpoint restored at **Epoch 33** (of 48 epochs run).

### 4.4 Deep Autoencoder (Unsupervised Semi-Supervised Anomaly Detection)
Unlike discriminative models that model the fraud boundary directly, the autoencoder learns the manifold of legitimate banking behavior.
- **Training Restriction:** Trained **strictly on 8,865 legitimate training transactions** ($y = 0$). It never observes a fraudulent transfer during weight optimization.
- **Topology:** Symmetric bottleneck encoder-decoder architecture:
  $$\text{Input}(14) \to \text{Dense}(64, \text{ReLU}) \to \text{Dense}(32, \text{ReLU}) \to \text{Bottleneck Dense}(8, \text{ReLU}) \to \text{Dense}(32, \text{ReLU}) \to \text{Dense}(64, \text{ReLU}) \to \text{Dense}(14, \text{Linear})$$
- **Parameters:** 6,614 trainable weights and biases
- **Objective:** Mean Squared Error (MSE) reconstruction loss:
  $$\mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{d} \sum_{j=1}^{d} (x_j - \hat{x}_j)^2$$
- **Reconstruction Anomaly Scoring:**
  When evaluated on test data, transactions traversing rare or anomalous coordinate paths exhibit elevated reconstruction errors. To scale errors into a robust anomaly score in $[0, 1]$, we formulate:
  $$s_{\text{AE}}(x) = \frac{\mathcal{L}_{\text{MSE}}(x, \hat{x})}{\mathcal{L}_{\text{MSE}}(x, \hat{x}) + Q_{99}\left(\mathcal{L}_{\text{MSE}}(\mathcal{D}_{\text{train, legit}})\right)}$$
  where $Q_{99}$ denotes the 99th percentile reconstruction error observed across legitimate training transactions.

---

## 5. Threshold Optimization & Validation Protocol

Decision thresholds are never arbitrarily set to $0.50$. In severe class imbalance, $0.50$ is heavily sub-optimal.
- **Selection Criterion:** Each model sweeps candidate thresholds $\tau \in [0.01, 0.99]$ on the **validation partition** to locate:
  $$\tau^* = \arg\max_{\tau} F_1(\mathcal{D}_{\text{val}}, \tau)$$
- **Model Selection Rule:** The primary production benchmark is determined by validation PR-AUC (Average Precision), with ties broken by $F_1$ and Recall.

### Validation Performance & Operating Thresholds
| Model | Family | Validation PR-AUC | Validation $F_1$ | Validation Recall | Optimal Threshold ($\tau^*$) |
|---|---|:---:|:---:|:---:|:---:|
| **Random Forest** | Machine Learning | 0.8879 | 0.8434 | 0.7778 | $0.7400$ |
| **XGBoost** | Machine Learning | **0.9020** | 0.8706 | 0.8222 | $0.9445$ |
| **Deep Neural Network** | Deep Learning | 0.8473 | 0.8608 | 0.7556 | $0.9993$ |
| **Deep Autoencoder** | Deep Learning | 0.8828 | **0.8889** | 0.8000 | $0.8707$ |

*XGBoost ranked 1st on validation PR-AUC ($0.9020$) and was established as the primary baseline.*

---

## 6. Empirical Results & Test Evaluation

The four trained models were locked and evaluated against the untouched **3,000-sample test partition** (containing 45 fraud cases):

### 6.1 Comprehensive Held-Out Test Metrics
| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | $F_1$-Score | ROC-AUC | PR-AUC (Avg Prec) | TP | FP | TN | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0.9960 | 0.8235 | **0.9333** | 0.9970 | 0.8750 | **0.9989** | 0.9581 | 42 | 9 | 2946 | 3 |
| **XGBoost** | 0.9967 | 0.8889 | 0.8889 | 0.9983 | 0.8889 | 0.9968 | 0.9418 | 40 | 5 | 2950 | 5 |
| **Deep Neural Network** | **0.9983** | **1.0000** | 0.8889 | **1.0000** | 0.9412 | 0.9863 | 0.9307 | 40 | **0** | **2955** | 5 |
| **Deep Autoencoder** | **0.9983** | 0.9762 | 0.9111 | 0.9997 | **0.9425** | 0.9909 | **0.9586** | 41 | 1 | 2954 | 4 |

### 6.2 Key Empirical Discoveries
1. **Zero False Positives via Deep Discriminative Learning:**
   The Deep Neural Network achieved a **Precision of 1.0000** with 0 false positives out of 2,955 legitimate transactions, yielding zero customer friction while catching 40 of 45 frauds.
2. **Semi-Supervised Manifold Superiority:**
   The Deep Autoencoder achieved the highest overall test $F_1$-score (**0.9425**) and test PR-AUC (**0.9586**). This demonstrates that modeling normal transaction geometry avoids the synthetic artifacts introduced by oversampling minority noise.
3. **Ensemble Sensitivity:**
   Random Forest achieved the highest individual Recall (**0.9333**), successfully capturing 42 out of 45 fraud instances.
4. **Transparent Review Rule (No Black-Box Blending):**
   Rather than averaging disparate scores, the consensus engine flags a transaction for human review if $\sum_{m=1}^{4} \mathbb{I}(s_m \ge \tau_m^*) \ge 1$. This maximizes recall in compliance review environments.

---

## 7. Model Explainability & Forensic Audit

### 7.1 Tree SHAP (SHapley Additive exPlanations)
To ensure regulatory compliance (e.g., Fair Lending Act, GDPR right to explanation), we compute Tree SHAP on the top-performing tree model (XGBoost).
- **Primary Driver:** `errorBalanceOrig` accounts for $> 45\%$ of Shapley importance, showing that non-zero originator balance changes strongly dictate fraud probability.
- **Secondary Drivers:** `amount_to_oldbalance_ratio` and transfer category (`type_TRANSFER`).

### 7.2 Quantitative Balance Forensics
For every flagged event, the system automatically checks deterministic balance invariants:
$$\Delta_{\text{orig}} \ne 0 \implies \text{Flag Discrepancy Observation}$$
$$\text{oldbalanceOrg} > 0 \land \text{newbalanceOrig} = 0 \implies \text{Flag Account Depletion Observation}$$

---

## 8. Real-Time Kafka Event Streaming Architecture

To transition from batch retrospective scoring to live ingestion, FraudGuard AI implements an event-driven architecture using **Apache Kafka**:

```
+--------------------------+       JSON Events       +-------------------------------+
| TransactionStreamProducer |  ------------------>  | Topic: financial.transactions |
+--------------------------+                         +-------------------------------+
                                                                    |
                                                                    v
+--------------------------+       JSON Alerts       +-------------------------------+
|  Topic: financial.alerts |  <------------------  |    FraudStreamingConsumer     |
+--------------------------+                         |  - 4-Model Consensus Engine   |
                                                     |  - Forensic Discrepancy Check |
                                                     |  - Latency Tracker (~100ms)   |
                                                     +-------------------------------+
```

### 8.1 Dual-Engine Streaming Design
1. **Live Cluster Mode:** Operates over standard TCP bootstrap brokers (`localhost:9092`), supporting distributed partitions and consumer group failover.
2. **Simulated Broker Mode:** An in-process, thread-safe, lock-free message queue (`SimulatedStreamingBroker`) implementing topic FIFOs, committed offset tracking, and consumer subscriptions. Enables fully functional offline demonstrations and automated CI/CD unit testing without third-party daemon dependencies.

### 8.2 Streaming Performance Characteristics
- **Scoring Latency:** $104.5 \pm 5.2 \text{ ms}$ per 4-model consensus pass.
- **Alert Event Envelope:** Includes unique UUIDs, ISO UTC timestamps, model votes, individual probabilities, forensic factor tags, and actionable recommendations.

---

## 9. Paper Writing & Section-by-Section Roadmap

When drafting your manuscript for an IEEE, ACM, or Springer journal/conference, use this structured chapter blueprint:

| Manuscript Section | Content Focus in FraudGuard AI | Primary Equations / Figures / Tables |
|---|---|---|
| **I. Introduction** | The economic cost of financial fraud, challenges of extreme class imbalance (65.7:1), motivation for multi-paradigm comparison (tree ensembles vs. discriminative MLP vs. unsupervised autoencoder). | Loss imbalance equations, paper contribution list. |
| **II. Related Work** | Prior literature on credit card and payment fraud detection (PaySim, SMOTE vs. ADASYN, tree ensembles, autoencoders, streaming architectures). | Citation review comparing supervised vs. anomaly detection approaches. |
| **III. Methodology: Preprocessing & Data Invariants** | Prevention of data leakage: stratified 60/20/20 partitioning, `RobustScaler` fitting on train only, SMOTE-NC on train only. Feature engineering of $\Delta_{\text{orig}}$, $\Delta_{\text{dest}}$, and balance depletion ratio. | Preprocessing flowchart, feature formulas ($\Delta_{\text{orig}}, \Delta_{\text{dest}}, R_{\text{deplete}}$). |
| **IV. Model Architectures & Anomaly Scoring** | Mathematical formulations of Random Forest, XGBoost regularized objective, MLPClassifier binary cross-entropy, and Deep Autoencoder bottleneck reconstruction error scaling ($s_{\text{AE}}$). | Neural architecture topologies, $s_{\text{AE}}(x)$ formula, training loss plots. |
| **V. Validation & Optimal Thresholding** | Why default 0.50 thresholds fail in imbalanced fraud detection. Validation $F_1$ optimization curve, validation PR-AUC model ranking (XGBoost top baseline). | Validation comparison table, threshold vs. F1 graphs. |
| **VI. Experimental Results & Analysis** | Comprehensive held-out test evaluation across 3,000 transactions. Key finding: DNN achieves 1.0000 Precision (0 FP), Autoencoder achieves top $F_1$ (0.9425) and PR-AUC (0.9586). | Complete test metrics table, ROC curves, PR curves, Confusion Matrices. |
| **VII. Interpretability & Forensic Auditing** | Explainable AI via Tree SHAP summary and feature importance. Automated quantitative balance forensics vs. black-box decisions. | SHAP beeswarm plot, balance discrepancy table. |
| **VIII. Real-Time Event-Driven Streaming** | Architecture of Apache Kafka pipeline: `TransactionStreamProducer`, `FraudStreamingConsumer`, `SimulatedStreamingBroker`. Latency benchmark (~100ms). | Kafka system architecture diagram, throughput & latency benchmarks. |
| **IX. Discussion, Limitations & Future Work** | Retrospective completed-balance features vs. live pre-auth authorization, lack of longitudinal account graph links for Graph Neural Networks (GNNs). | Limitations summary table. |
| **X. Conclusion** | Synthesis of findings: semi-supervised autoencoders provide optimal boundary robustness in imbalanced financial data. | Final concluding remarks. |
