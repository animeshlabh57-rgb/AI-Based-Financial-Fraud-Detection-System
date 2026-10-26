import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ==========================================
# FOLDERS
# ==========================================

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Class"].value_counts())


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates().reset_index(drop=True)

print("\nDataset shape after removing duplicates:")
print(df.shape)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1).copy()
y = df["Class"].astype(int)


# ==========================================
# 4. SCALE FEATURES
# ==========================================

scale_columns = []

if "Time" in X.columns:
    scale_columns.append("Time")

if "Amount" in X.columns:
    scale_columns.append("Amount")

scaler = StandardScaler()

if scale_columns:
    X[scale_columns] = scaler.fit_transform(X[scale_columns])


feature_columns = list(X.columns)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. APPLY SMOTE
# ==========================================

print("\nClass distribution BEFORE SMOTE:")
print(y_train.value_counts())

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ==========================================
# SAVE PREPROCESSING
# ==========================================

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)

joblib.dump(
    feature_columns,
    f"{MODEL_DIR}/feature_columns.pkl"
)


# ==========================================
# 7. RANDOM FOREST
# ==========================================

print("\n====================================")
print("TRAINING RANDOM FOREST")
print("====================================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(
    X_train_smote,
    y_train_smote
)

rf_prediction = rf.predict(X_test)

rf_probability = rf.predict_proba(X_test)[:, 1]


rf_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)


print("\nRandom Forest Results")

print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_auc, 4))


joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)


# ==========================================
# 8. XGBOOST
# ==========================================

print("\n====================================")
print("TRAINING XGBOOST")
print("====================================")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb.fit(
    X_train_smote,
    y_train_smote
)

xgb_prediction = xgb.predict(X_test)

xgb_probability = xgb.predict_proba(X_test)[:, 1]


xgb_precision = precision_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_prediction,
    zero_division=0
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)


print("\nXGBoost Results")

print("Precision:", round(xgb_precision, 4))
print("Recall:", round(xgb_recall, 4))
print("F1 Score:", round(xgb_f1, 4))
print("ROC-AUC:", round(xgb_auc, 4))


joblib.dump(
    xgb,
    f"{MODEL_DIR}/xgboost.pkl"
)


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Random Forest",
        "XGBoost"
    ],

    "Precision": [
        rf_precision,
        xgb_precision
    ],

    "Recall": [
        rf_recall,
        xgb_recall
    ],

    "F1-Score": [
        rf_f1,
        xgb_f1
    ],

    "ROC-AUC": [
        rf_auc,
        xgb_auc
    ]
})


print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(results)


results.to_csv(
    f"{RESULT_DIR}/model_results.csv",
    index=False
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    xgb_prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Legitimate",
        "Fraud"
    ],
    yticklabels=[
        "Legitimate",
        "Fraud"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/confusion_matrix.png",
    dpi=200
)

plt.close()


# ==========================================
# 11. ROC CURVE
# ==========================================

print("Generating ROC curve...")

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probability
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_probability
)


plt.figure(figsize=(7, 5))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest AUC={rf_auc:.3f}"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost AUC={xgb_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/roc_curve.png",
    dpi=200
)

plt.close()


# ==========================================
# 12. MODEL COMPARISON GRAPH
# ==========================================

print("Generating model comparison graph...")

plot_data = results.set_index("Model")

ax = plot_data.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_ylabel("Score")

ax.set_ylim(
    0,
    1.05
)

ax.set_title(
    "Fraud Detection Model Comparison"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/model_comparison.png",
    dpi=200
)

plt.close()


# ==========================================
# 13. FEATURE IMPORTANCE
# ==========================================

print("Generating feature importance graph...")

importance = pd.Series(
    xgb.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values(
    ascending=False
).head(15)


plt.figure(figsize=(8, 6))

importance.sort_values().plot(
    kind="barh"
)

plt.xlabel(
    "Importance"
)

plt.title(
    "Top XGBoost Features"
)

plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/feature_importance.png",
    dpi=200
)

plt.close()


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    xgb_prediction,
    target_names=[
        "Legitimate",
        "Fraud"
    ],
    zero_division=0
)

with open(
    f"{RESULT_DIR}/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# COMPLETE
# ==========================================

print("\n====================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("====================================")

print("\nGenerated files:")

print("models/random_forest.pkl")
print("models/xgboost.pkl")

print("results/model_results.csv")
print("results/confusion_matrix.png")
print("results/roc_curve.png")
print("results/model_comparison.png")
print("results/feature_importance.png")
print("results/classification_report.txt")

print("\nYou can now proceed to the deep-learning experiment.")