# 🛡️ AI-Based Financial Fraud Detection System

An enterprise-grade, end-to-end Financial Fraud Detection and Risk Analytics platform powered by Machine Learning, Deep Learning (Autoencoders & LSTM), and Explainable AI (XAI with SHAP & LIME).

---

## 🌟 Key Features

- **Multi-Model Architecture:**
  - **Random Forest & XGBoost**: Robust ensemble models optimized with SMOTE to handle severe class imbalance.
  - **Autoencoders**: Unsupervised deep neural networks for reconstruction error-based anomaly detection.
  - **LSTM Networks**: Deep sequential models for capturing temporal transaction patterns.
- **Explainable AI (XAI):**
  - **SHAP (SHapley Additive exPlanations)**: Global and local feature impact breakdown.
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Per-transaction decision transparency for compliance and auditing.
- **Interactive Streamlit Dashboard (`app.py`):**
  - Real-time single transaction analysis & risk scoring.
  - Batch CSV transaction scanning with dynamic alerts.
  - Comprehensive model comparison benchmarks and interactive ROC/Confusion Matrix visualizations.
  - AI Explainability explorer with visual waterfall and force plots.

---

## 📁 Repository Structure

```text
├── app.py                   # Streamlit interactive web dashboard
├── train.py                 # Traditional ML pipeline (Random Forest, XGBoost, SMOTE)
├── deep_learning.py         # Autoencoder anomaly detection pipeline
├── lstm.py                  # LSTM neural network pipeline
├── explain.py               # XAI explainability engine (SHAP & LIME)
├── data/
│   ├── demo_test.csv        # Sample test dataset for batch evaluation
│   ├── my_transactions.csv  # Demo custom transactions
│   └── creditcard.csv       # (Ignored in repo due to size - see Dataset section)
├── models/                  # Pre-trained models, encoders, and scalers
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── autoencoder.keras
│   ├── lstm.keras
│   └── scaler.pkl
├── results/                 # Evaluation plots, confusion matrices, and reports
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── model_results.csv
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/animeshlabh57-rgb/AI-Based-Financial-Fraud-Detection-System.git
cd AI-Based-Financial-Fraud-Detection-System
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset Setup
Download the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside the `data/` folder:
```text
data/creditcard.csv
```

### 4. Running the Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📊 Model Training & Evaluation

To retrain the models or reproduce the evaluation benchmarks:

- **Train Traditional ML Models (RF & XGBoost):**
  ```bash
  python train.py
  ```
- **Train Autoencoder (Anomaly Detection):**
  ```bash
  python deep_learning.py
  ```
- **Train LSTM Network:**
  ```bash
  python lstm.py
  ```
- **Generate Explainability Reports (SHAP & LIME):**
  ```bash
  python explain.py
  ```

---

## 📈 Performance & Results

Evaluation metrics across balanced test datasets:
- **High Precision & Recall**: Minimized false positives while maximizing fraud capture rate.
- **AUROC > 0.95**: Demonstrating superior discrimination between legitimate and fraudulent transactions.
- **Explainability**: Clear risk factor breakdowns for every flagged transaction.

---

## 📜 License
This project is licensed under the MIT License.
