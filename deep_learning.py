import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# SETTINGS
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

print("\nShape after removing duplicates:")
print(df.shape)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1).copy()
y = df["Class"].astype(int)


# ==========================================
# 4. SCALE DATA
# ==========================================

scale_columns = []

if "Time" in X.columns:
    scale_columns.append("Time")

if "Amount" in X.columns:
    scale_columns.append("Amount")

scaler = StandardScaler()

if scale_columns:
    X[scale_columns] = scaler.fit_transform(
        X[scale_columns]
    )


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. USE ONLY LEGITIMATE TRANSACTIONS
# ==========================================

X_train_normal = X_train[y_train == 0]

print(
    "\nNormal transactions used for Autoencoder training:",
    len(X_train_normal)
)


# ==========================================
# 7. BUILD AUTOENCODER
# ==========================================

input_dimension = X_train.shape[1]

inputs = Input(
    shape=(input_dimension,)
)

encoded = Dense(
    32,
    activation="relu"
)(inputs)

encoded = Dense(
    16,
    activation="relu"
)(encoded)

decoded = Dense(
    32,
    activation="relu"
)(encoded)

outputs = Dense(
    input_dimension,
    activation="linear"
)(decoded)


autoencoder = Model(
    inputs,
    outputs
)


# ==========================================
# 8. COMPILE MODEL
# ==========================================

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# ==========================================
# 9. TRAIN AUTOENCODER
# ==========================================

print("\n====================================")
print("TRAINING AUTOENCODER")
print("====================================")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


history = autoencoder.fit(
    X_train_normal,
    X_train_normal,

    epochs=15,

    batch_size=256,

    validation_split=0.10,

    callbacks=[early_stopping],

    verbose=1
)


# ==========================================
# 10. RECONSTRUCTION ERROR
# ==========================================

print("\nCalculating reconstruction errors...")

reconstructed_train = autoencoder.predict(
    X_train_normal,
    verbose=0
)

train_error = np.mean(
    np.square(
        X_train_normal - reconstructed_train
    ),
    axis=1
)


# ==========================================
# 11. FRAUD THRESHOLD
# ==========================================

threshold = np.percentile(
    train_error,
    95
)

print(
    "\nAnomaly threshold:",
    threshold
)


# ==========================================
# 12. TEST AUTOENCODER
# ==========================================

reconstructed_test = autoencoder.predict(
    X_test,
    verbose=0
)

test_error = np.mean(
    np.square(
        X_test - reconstructed_test
    ),
    axis=1
)


# ==========================================
# 13. PREDICTION
# ==========================================

prediction = (
    test_error > threshold
).astype(int)


# ==========================================
# 14. EVALUATION
# ==========================================

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_error
)


print("\n====================================")
print("AUTOENCODER RESULTS")
print("====================================")

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1-Score:",
    round(f1, 4)
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Autoencoder"
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1-Score": [
        f1
    ],

    "ROC-AUC": [
        roc_auc
    ]

})


results.to_csv(
    f"{RESULT_DIR}/autoencoder_results.csv",
    index=False
)


# ==========================================
# 16. SAVE MODEL
# ==========================================

autoencoder.save(
    f"{MODEL_DIR}/autoencoder.keras"
)

with open(
    f"{MODEL_DIR}/autoencoder_threshold.txt",
    "w"
) as file:

    file.write(
        str(float(threshold))
    )


# ==========================================
# 17. SAVE CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    prediction,

    target_names=[
        "Legitimate",
        "Fraud"
    ],

    zero_division=0
)

with open(
    f"{RESULT_DIR}/autoencoder_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# COMPLETE
# ==========================================

print("\n====================================")
print("AUTOENCODER EXPERIMENT COMPLETED")
print("====================================")

print("\nGenerated files:")

print(
    "models/autoencoder.keras"
)

print(
    "results/autoencoder_results.csv"
)

print(
    "results/autoencoder_report.txt"
)

print("\nNext step: LSTM experiment.")