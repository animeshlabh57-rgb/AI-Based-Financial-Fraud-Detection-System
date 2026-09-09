import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10


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
    X[scale_columns] = scaler.fit_transform(
        X[scale_columns]
    )


X = X.values
y = y.values


# ==========================================
# 5. CREATE SEQUENCES
# ==========================================

print("\nCreating transaction sequences...")

X_sequences = []
y_sequences = []

for i in range(
    SEQUENCE_LENGTH - 1,
    len(X)
):

    sequence = X[
        i - SEQUENCE_LENGTH + 1:i + 1
    ]

    label = y[i]

    X_sequences.append(sequence)
    y_sequences.append(label)


X_sequences = np.array(
    X_sequences,
    dtype=np.float32
)

y_sequences = np.array(
    y_sequences,
    dtype=np.int32
)


print(
    "Sequence shape:",
    X_sequences.shape
)

print(
    "Labels shape:",
    y_sequences.shape
)


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_sequences,
    y_sequences,

    test_size=0.20,

    random_state=42,

    stratify=y_sequences
)


print(
    "\nTraining sequences:",
    len(X_train)
)

print(
    "Testing sequences:",
    len(X_test)
)


# ==========================================
# 7. HANDLE CLASS IMBALANCE
# ==========================================

negative_count = np.sum(
    y_train == 0
)

positive_count = np.sum(
    y_train == 1
)

fraud_weight = (
    negative_count /
    max(positive_count, 1)
)

class_weights = {

    0: 1.0,

    1: fraud_weight

}


print(
    "\nClass weights:",
    class_weights
)


# ==========================================
# 8. BUILD LSTM MODEL
# ==========================================

print("\n====================================")
print("BUILDING LSTM MODEL")
print("====================================")


model = Sequential([

    LSTM(
        32,
        input_shape=(
            SEQUENCE_LENGTH,
            X_train.shape[2]
        )
    ),

    Dropout(0.3),

    Dense(
        16,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )

])


# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\n====================================")
print("TRAINING LSTM")
print("====================================")


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


history = model.fit(

    X_train,

    y_train,

    validation_split=0.10,

    epochs=10,

    batch_size=256,

    class_weight=class_weights,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ==========================================
# 11. PREDICTION
# ==========================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=0
).ravel()


predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# 12. EVALUATION
# ==========================================

precision = precision_score(

    y_test,

    predictions,

    zero_division=0

)


recall = recall_score(

    y_test,

    predictions,

    zero_division=0

)


f1 = f1_score(

    y_test,

    predictions,

    zero_division=0

)


roc_auc = roc_auc_score(

    y_test,

    probabilities

)


# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("LSTM RESULTS")
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
# 14. CLASSIFICATION REPORT
# ==========================================

report = classification_report(

    y_test,

    predictions,

    target_names=[
        "Legitimate",
        "Fraud"
    ],

    zero_division=0

)


print("\nClassification Report:")
print(report)


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results = pd.DataFrame({

    "Model": [
        "LSTM"
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

    f"{RESULT_DIR}/lstm_results.csv",

    index=False

)


# ==========================================
# 16. SAVE MODEL
# ==========================================

model.save(

    f"{MODEL_DIR}/lstm.keras"

)


# ==========================================
# 17. SAVE CLASSIFICATION REPORT
# ==========================================

with open(

    f"{RESULT_DIR}/lstm_report.txt",

    "w"

) as file:

    file.write(report)


# ==========================================
# 18. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(

    f"{RESULT_DIR}/lstm_training_history.csv",

    index=False

)


# ==========================================
# COMPLETE
# ==========================================

print("\n====================================")
print("LSTM EXPERIMENT COMPLETED")
print("====================================")

print("\nGenerated files:")

print(
    "models/lstm.keras"
)

print(
    "results/lstm_results.csv"
)

print(
    "results/lstm_report.txt"
)

print(
    "results/lstm_training_history.csv"
)

print("\nNext step: Explainable AI (SHAP + LIME).")