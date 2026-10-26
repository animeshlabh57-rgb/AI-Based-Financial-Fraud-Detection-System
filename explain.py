import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/xgboost.pkl"
SCALER_PATH = "models/scaler.pkl"

RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE FEATURES
# ==========================================

X = df.drop("Class", axis=1).copy()

y = df["Class"]


# ==========================================
# 3. LOAD SCALER
# ==========================================

print("\nLoading preprocessing objects...")

scaler = joblib.load(SCALER_PATH)


scale_columns = []

if "Time" in X.columns:
    scale_columns.append("Time")

if "Amount" in X.columns:
    scale_columns.append("Amount")


if scale_columns:

    X[scale_columns] = scaler.transform(
        X[scale_columns]
    )


# ==========================================
# 4. LOAD XGBOOST MODEL
# ==========================================

print("Loading XGBoost model...")

model = joblib.load(MODEL_PATH)


print("Model loaded successfully.")


# ==========================================
# 5. SELECT SAMPLE
# ==========================================

sample_size = min(
    1000,
    len(X)
)

sample = X.sample(
    sample_size,
    random_state=42
)


# ==========================================
# 6. SHAP EXPLAINER
# ==========================================

print("\n====================================")
print("GENERATING SHAP EXPLANATION")
print("====================================")


explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)


# ==========================================
# 7. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")


plt.figure()

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "SHAP graph saved to results/shap_summary.png"
)


# ==========================================
# 8. SHAP FEATURE IMPORTANCE
# ==========================================

print("\nGenerating SHAP feature importance...")


importance = pd.DataFrame({

    "Feature": sample.columns,

    "Importance": abs(shap_values).mean(axis=0)

})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


importance.to_csv(
    f"{RESULT_DIR}/shap_feature_importance.csv",
    index=False
)


# ==========================================
# 9. EXPLAIN ONE TRANSACTION
# ==========================================

print("\n====================================")
print("EXPLAINING ONE TRANSACTION")
print("====================================")


transaction = sample.iloc[[0]]


prediction = model.predict(
    transaction
)[0]


probability = model.predict_proba(
    transaction
)[0][1]


if prediction == 1:

    prediction_text = "FRAUDULENT"

else:

    prediction_text = "LEGITIMATE"


print(
    "\nPrediction:",
    prediction_text
)

print(
    "Fraud Probability:",
    round(probability * 100, 2),
    "%"
)


# ==========================================
# 10. LIME
# ==========================================

print("\n====================================")
print("GENERATING LIME EXPLANATION")
print("====================================")


lime_explainer = LimeTabularExplainer(

    X.values,

    feature_names=X.columns.tolist(),

    class_names=[
        "Legitimate",
        "Fraud"
    ],

    mode="classification",

    random_state=42

)


lime_explanation = lime_explainer.explain_instance(

    transaction.iloc[0].values,

    model.predict_proba,

    num_features=10

)


lime_path = (
    f"{RESULT_DIR}/lime_explanation.html"
)


lime_explanation.save_to_file(
    lime_path
)


print(
    "LIME explanation saved to:",
    lime_path
)


# ==========================================
# 11. SAVE TRANSACTION EXPLANATION
# ==========================================

explanation_text = []

explanation_text.append(
    "AI-POWERED FINANCIAL FRAUD DETECTION SYSTEM"
)

explanation_text.append(
    "\nIndividual Transaction Explanation"
)

explanation_text.append(
    f"\nPrediction: {prediction_text}"
)

explanation_text.append(
    f"Fraud Probability: {probability * 100:.2f}%"
)

explanation_text.append(
    "\nTop SHAP Features:"
)


for _, row in importance.head(10).iterrows():

    explanation_text.append(
        f"{row['Feature']}: "
        f"{row['Importance']:.6f}"
    )


with open(
    f"{RESULT_DIR}/transaction_explanation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(explanation_text)
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n====================================")
print("EXPLAINABLE AI COMPLETED")
print("====================================")

print("\nGenerated files:")

print(
    "results/shap_summary.png"
)

print(
    "results/shap_feature_importance.csv"
)

print(
    "results/lime_explanation.html"
)

print(
    "results/transaction_explanation.txt"
)

print("\nNext step: Streamlit Dashboard.")