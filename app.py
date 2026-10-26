import os
import re
import html
import hashlib
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.inspection import permutation_importance

try:
    import tensorflow as tf
except Exception:
    tf = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "models/xgboost.pkl"
RF_MODEL_PATH = "models/random_forest.pkl"
AE_MODEL_PATH = "models/autoencoder.keras"
AE_THRESHOLD_PATH = "models/autoencoder_threshold.txt"
LSTM_MODEL_PATH = "models/lstm.keras"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

RESULTS_DIR = "results"
DATASET_PATH = "data/creditcard.csv"

# =========================================================
# 10 DEMO CARD PROFILES
# =========================================================
# Fictional/test details only. No real payment information is used.
# Each card is mapped to a different non-overlapping part of the
# anonymized transaction dataset, so every card has different history.
DEMO_CARDS = [
    {"label": "Demo Card 01 — Aarav Sharma", "cardholder": "Aarav Sharma", "card_number": "4000 0000 0000 0001", "expiry": "12/28"},
    {"label": "Demo Card 02 — Diya Kapoor", "cardholder": "Diya Kapoor", "card_number": "4000 0000 0000 0002", "expiry": "09/29"},
    {"label": "Demo Card 03 — Rohan Mehta", "cardholder": "Rohan Mehta", "card_number": "4000 0000 0000 0003", "expiry": "06/28"},
    {"label": "Demo Card 04 — Ananya Singh", "cardholder": "Ananya Singh", "card_number": "4000 0000 0000 0004", "expiry": "11/29"},
    {"label": "Demo Card 05 — Kabir Verma", "cardholder": "Kabir Verma", "card_number": "4000 0000 0000 0005", "expiry": "03/30"},
    {"label": "Demo Card 06 — Meera Iyer", "cardholder": "Meera Iyer", "card_number": "4000 0000 0000 0006", "expiry": "08/29"},
    {"label": "Demo Card 07 — Arjun Malhotra", "cardholder": "Arjun Malhotra", "card_number": "4000 0000 0000 0007", "expiry": "05/30"},
    {"label": "Demo Card 08 — Nisha Rao", "cardholder": "Nisha Rao", "card_number": "4000 0000 0000 0008", "expiry": "10/28"},
    {"label": "Demo Card 09 — Vihaan Patel", "cardholder": "Vihaan Patel", "card_number": "4000 0000 0000 0009", "expiry": "01/30"},
    {"label": "Demo Card 10 — Saanvi Gupta", "cardholder": "Saanvi Gupta", "card_number": "4000 0000 0000 0010", "expiry": "07/29"},
]

DEMO_CARD_BY_NUMBER = {
    re.sub(r"\D", "", card["card_number"]): i
    for i, card in enumerate(DEMO_CARDS)
}



# =========================================================
# SESSION STATE
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "users" not in st.session_state:
    # Demo account for project/viva demonstration.
    st.session_state.users = {
        "demo@fraud.ai": {
            "name": "Demo User",
            "password_hash": hashlib.sha256(
                "demo123".encode()
            ).hexdigest(),
        }
    }

if "uploaded_batch_data" not in st.session_state:
    st.session_state.uploaded_batch_data = None

if "uploaded_batch_predictions" not in st.session_state:
    st.session_state.uploaded_batch_predictions = None

if "uploaded_batch_probabilities" not in st.session_state:
    st.session_state.uploaded_batch_probabilities = None

if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []

if "last_single_prediction" not in st.session_state:
    st.session_state.last_single_prediction = None

if "new_transaction_source_data" not in st.session_state:
    st.session_state.new_transaction_source_data = None

if "pending_card_details" not in st.session_state:
    st.session_state.pending_card_details = None
    st.session_state.demo_account_bucket = None

if "demo_account_bucket" not in st.session_state:
    st.session_state.demo_account_bucket = None


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_all_models():
    """Load all four project models and the shared preprocessing artifacts."""
    xgb = joblib.load(MODEL_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_PATH)

    ae = None
    ae_threshold = None
    lstm = None

    if os.path.exists(AE_MODEL_PATH) and tf is not None:
        try:
            ae = tf.keras.models.load_model(AE_MODEL_PATH, compile=False)
        except Exception:
            ae = None

    if os.path.exists(AE_THRESHOLD_PATH):
        try:
            with open(AE_THRESHOLD_PATH, "r", encoding="utf-8") as f:
                ae_threshold = float(f.read().strip())
        except Exception:
            ae_threshold = None

    if os.path.exists(LSTM_MODEL_PATH) and tf is not None:
        try:
            lstm = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        except Exception:
            lstm = None

    return xgb, rf, ae, ae_threshold, lstm, scaler, feature_cols


try:
    (
        model,
        rf_model,
        autoencoder,
        autoencoder_threshold,
        lstm_model,
        scaler,
        feature_columns,
    ) = load_all_models()
except Exception as e:
    st.error(f"Unable to load the trained models: {e}")
    st.stop()


# =========================================================
# HELPERS
# =========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def reset_user_session_data():
    """Clear all transaction/evaluation data when a user starts a session."""
    st.session_state.uploaded_batch_data = None
    st.session_state.uploaded_batch_predictions = None
    st.session_state.uploaded_batch_probabilities = None
    st.session_state.transaction_history = []
    st.session_state.last_single_prediction = None
    st.session_state.new_transaction_source_data = None
    st.session_state.pending_card_details = None


def valid_email(email):
    return re.match(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        email.strip()
    ) is not None


def mask_card_number(card_number):
    digits = re.sub(r"\D", "", card_number)

    if len(digits) < 4:
        return "••••"

    return f"•••• •••• •••• {digits[-4:]}"



@st.cache_data
def load_kaggle_transaction_dataset():
    """Load the Kaggle credit-card transaction dataset from the local project."""
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        data = pd.read_csv(DATASET_PATH)
        return data
    except Exception:
        return None


def synthetic_account_bucket(card_number, cardholder="", expiry=""):
    """Return a stable demo account key for the entered card details."""
    digits = re.sub(r"\D", "", str(card_number))

    if digits in DEMO_CARD_BY_NUMBER:
        return DEMO_CARD_BY_NUMBER[digits]

    fingerprint = "|".join([
        digits,
        str(cardholder).strip().lower(),
        str(expiry).strip().lower(),
    ])
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def build_demo_account_history(data, card_number, cardholder="", expiry=""):
    """
    Map the 10 built-in demo cards to 10 different chronological sections
    of the anonymized dataset. The same card always gets the same history.
    """
    if data is None or data.empty:
        return None, None

    ordered = data.sort_values("Time", ascending=True).reset_index(drop=True)
    digits = re.sub(r"\D", "", str(card_number))

    if digits in DEMO_CARD_BY_NUMBER:
        card_index = DEMO_CARD_BY_NUMBER[digits]
        chunk_size = len(ordered) // 10

        start_index = card_index * chunk_size
        end_index = (
            (card_index + 1) * chunk_size
            if card_index < 9
            else len(ordered)
        )

        history = ordered.iloc[start_index:end_index].copy()
        return card_index, history.reset_index(drop=True)

    # Custom cards still get their own deterministic history.
    account_key = synthetic_account_bucket(
        card_number,
        cardholder,
        expiry,
    )

    history_size = min(5000, len(ordered))
    max_start = max(0, len(ordered) - history_size)

    fingerprint = (
        f"{digits}|"
        f"{str(cardholder).strip().lower()}|"
        f"{str(expiry).strip().lower()}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    start_index = int(digest[12:28], 16) % (max_start + 1)

    history = ordered.iloc[
        start_index:start_index + history_size
    ].copy().reset_index(drop=True)

    return account_key, history



def make_history_display(history, card_index=0):
    """
    Convert the anonymized dataset columns into a user-friendly bank-style
    transaction history table.

    Time/Amount/Class remain the underlying dataset fields used by the model,
    but the UI displays meaningful banking labels. Location, Channel and
    Transaction Type are deterministic demo attributes because the public
    dataset does not contain those fields.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    df = history.copy().reset_index(drop=True)

    locations = [
        "Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad",
        "Pune", "Kolkata", "Jaipur", "Ahmedabad", "Noida"
    ]
    channels = ["Online", "POS", "Mobile", "ATM"]
    types = ["Purchase", "Transfer", "Payment", "Withdrawal"]

    # Deterministic values: same card/history gives the same display.
    seed = int(card_index or 0)
    positions = np.arange(len(df))

    display = pd.DataFrame()
    display["Transaction ID"] = [
        f"TXN-{seed + 1:02d}-{i + 1:04d}" for i in range(len(df))
    ]

    # Dataset Time is elapsed seconds, so display it as a readable
    # transaction date/time relative to a fixed demo starting date.
    if "Time" in df.columns:
        seconds = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
        base_date = pd.Timestamp("2026-01-01")
        display["Date & Time"] = (
            base_date + pd.to_timedelta(seconds, unit="s")
        ).dt.strftime("%d-%m-%Y %H:%M")
    else:
        display["Date & Time"] = "N/A"

    if "Amount" in df.columns:
        display["Amount (₹)"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        ).round(2)
    else:
        display["Amount (₹)"] = np.nan

    display["Location"] = [
        locations[(i + seed) % len(locations)]
        for i in positions
    ]
    display["Channel"] = [
        channels[(i * 2 + seed) % len(channels)]
        for i in positions
    ]
    display["Transaction Type"] = [
        types[(i + seed * 2) % len(types)]
        for i in positions
    ]

    if "Class" in df.columns:
        labels = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
        display["Status"] = np.where(
            labels == 1,
            "Fraudulent",
            "Legitimate",
        )
    else:
        display["Status"] = "Legitimate"

    return display

def choose_transaction_proxy(history, amount):
    """
    Pick the historical transaction whose Amount is closest to the entered
    amount. Its anonymized V1-V28 pattern is used as the model feature proxy,
    while Amount is replaced by the newly entered amount.
    """
    if history is None or history.empty:
        return None

    amounts = pd.to_numeric(history["Amount"], errors="coerce")
    valid = amounts.notna()

    if not valid.any():
        return history.iloc[0].copy()

    distances = (amounts[valid] - float(amount)).abs()
    idx = distances.idxmin()
    return history.loc[idx].copy()



def calculate_history_risk(history, new_amount):
    """
    Calculate a transparent behavioral-risk score from the linked history.

    This is a demo behavioral layer because the public ULB dataset has no
    real card/account identifier, location, channel, or customer profile.
    The score is based on historical transaction amounts and fraud labels.
    """
    if history is None or history.empty:
        return 0.0, {
            "amount_anomaly": 0.0,
            "historical_fraud_rate": 0.0,
            "history_count": 0,
            "median_amount": 0.0,
        }

    amounts = pd.to_numeric(history["Amount"], errors="coerce").dropna()
    if amounts.empty:
        return 0.0, {
            "amount_anomaly": 0.0,
            "historical_fraud_rate": 0.0,
            "history_count": len(history),
            "median_amount": 0.0,
        }

    amount = float(new_amount)
    median_amount = float(amounts.median())
    q1 = float(amounts.quantile(0.25))
    q3 = float(amounts.quantile(0.75))
    iqr = max(q3 - q1, 1e-9)

    # Robust amount anomaly: values beyond the normal IQR range receive
    # progressively higher risk.
    upper_normal = q3 + 1.5 * iqr
    if amount <= upper_normal:
        amount_anomaly = min(abs(amount - median_amount) / max(iqr, median_amount, 1.0), 0.35)
    else:
        amount_anomaly = min(0.35 + (amount - upper_normal) / max(upper_normal, 1.0), 1.0)

    fraud_rate = 0.0
    if "Class" in history.columns:
        labels = pd.to_numeric(history["Class"], errors="coerce").dropna()
        if len(labels):
            fraud_rate = float(labels.mean())

    # Keep the behavioral layer conservative; the trained model remains the
    # primary signal.
    history_risk = min(1.0, 0.80 * amount_anomaly + 0.20 * fraud_rate)

    return history_risk, {
        "amount_anomaly": amount_anomaly,
        "historical_fraud_rate": fraud_rate,
        "history_count": len(history),
        "median_amount": median_amount,
        "upper_normal_amount": upper_normal,
    }



def build_lstm_sequence(history, current_row, seq_len=10):
    """
    Build the LSTM input from the last 9 historical transactions plus
    the current transaction. If fewer than 9 history rows exist, pad
    with the earliest available historical row.
    """
    history = history.copy() if history is not None else pd.DataFrame()

    if history.empty:
        base_rows = pd.DataFrame([current_row.copy()] * (seq_len - 1))
    else:
        history = history.sort_values("Time", ascending=True)
        base_rows = history.tail(seq_len - 1).copy()

        if len(base_rows) < seq_len - 1:
            first = base_rows.iloc[[0]].copy()
            padding = pd.concat(
                [first] * (seq_len - 1 - len(base_rows)),
                ignore_index=True,
            )
            base_rows = pd.concat(
                [padding, base_rows.reset_index(drop=True)],
                ignore_index=True,
            )

    current = current_row.copy()
    sequence_df = pd.concat(
        [base_rows, pd.DataFrame([current])],
        ignore_index=True,
    )

    X_sequence = prepare_model_input(sequence_df)
    return X_sequence.to_numpy(dtype=np.float32)




def display_history_confusion_matrices(history):
    """
    Display confusion matrices for the supervised models using the labels
    available in the selected account-history slice.
    """
    if history is None or history.empty or "Class" not in history.columns:
        st.info("Confusion matrix requires historical Class labels.")
        return

    y_true = pd.to_numeric(history["Class"], errors="coerce").fillna(0).astype(int)

    X_history = prepare_model_input(history)

    rf_pred = rf_model.predict(X_history).astype(int)
    xgb_pred = model.predict(X_history).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        cm_rf = confusion_matrix(y_true, rf_pred, labels=[0, 1])
        fig_rf = go.Figure(
            data=go.Heatmap(
                z=cm_rf,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                text=cm_rf,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_rf.update_layout(
            title="Random Forest — Account History",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with col2:
        cm_xgb = confusion_matrix(y_true, xgb_pred, labels=[0, 1])
        fig_xgb = go.Figure(
            data=go.Heatmap(
                z=cm_xgb,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                text=cm_xgb,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_xgb.update_layout(
            title="XGBoost — Account History",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig_xgb, use_container_width=True)

    st.caption(
        "These confusion matrices are calculated from the actual Class labels "
        "present in the selected card's historical transaction slice."
    )


def analyze_account_history(history, current_amount=None):
    """
    Analyze the selected demo account's transaction history with all
    available models. This powers the Model Performance page after a card
    is linked, without requiring a separate CSV upload.
    """
    if history is None or history.empty:
        return None

    history = history.sort_values("Time", ascending=True).reset_index(drop=True)

    # Keep the page responsive for very large synthetic buckets while still
    # using a substantial portion of the account history.
    analysis_history = history.tail(5000).copy()

    X_history = prepare_model_input(analysis_history)

    result = {
        "history_count": len(history),
        "rows_analyzed": len(analysis_history),
    }

    # Supervised model history scores.
    result["Random Forest"] = float(
        np.mean(rf_model.predict_proba(X_history)[:, 1])
    )
    result["XGBoost"] = float(
        np.mean(model.predict_proba(X_history)[:, 1])
    )

    # Autoencoder anomaly score.
    result["Autoencoder"] = None
    result["Autoencoder anomaly rate"] = None
    if autoencoder is not None:
        reconstructed = autoencoder.predict(
            X_history.to_numpy(dtype=np.float32),
            batch_size=256,
            verbose=0,
        )
        errors = np.mean(
            np.square(
                X_history.to_numpy(dtype=np.float32) - reconstructed
            ),
            axis=1,
        )

        result["Autoencoder"] = float(np.mean(errors))

        if autoencoder_threshold is not None and autoencoder_threshold > 0:
            result["Autoencoder anomaly rate"] = float(
                np.mean(errors > autoencoder_threshold)
            )

    # LSTM analyzes the most recent transaction sequence in the account.
    result["LSTM"] = None
    if lstm_model is not None and len(analysis_history) > 0:
        latest = analysis_history.iloc[-1].copy()
        sequence = build_lstm_sequence(
            analysis_history.iloc[:-1],
            latest,
            seq_len=10,
        )
        result["LSTM"] = float(
            np.asarray(
                lstm_model.predict(
                    sequence[np.newaxis, :, :],
                    verbose=0,
                )
            ).ravel()[0]
        )

    # Actual labels available in the public dataset are shown as historical
    # context, not as real card-level banking labels.
    if "Class" in analysis_history.columns:
        labels = pd.to_numeric(
            analysis_history["Class"],
            errors="coerce",
        ).dropna()

        result["historical_fraud_count"] = int((labels == 1).sum())
        result["historical_fraud_rate"] = (
            float((labels == 1).mean()) if len(labels) else 0.0
        )
    else:
        result["historical_fraud_count"] = None
        result["historical_fraud_rate"] = None

    if current_amount is not None:
        history_risk, stats = calculate_history_risk(
            history,
            float(current_amount),
        )
        result["new_transaction_history_risk"] = history_risk
        result["history_stats"] = stats

    return result


def analyze_all_models(history, current_row):
    """
    Run RF, XGBoost, Autoencoder and LSTM on the same history-derived
    transaction representation.

    The public ULB dataset does not contain real card/account identifiers
    or customer attributes, so this is a demonstration-level history
    linkage, not a real banking decision engine.
    """
    # Tree models use the historical transaction closest to the new amount
    # as the anonymized feature proxy.
    proxy = choose_transaction_proxy(history, float(current_row["Amount"]))
    if proxy is None:
        raise ValueError("No historical transaction is available for model analysis.")

    proxy = proxy.copy()
    proxy["Amount"] = float(current_row["Amount"])

    X_current = prepare_model_input(pd.DataFrame([proxy]))

    rf_probability = float(rf_model.predict_proba(X_current)[0][1])
    xgb_probability = float(model.predict_proba(X_current)[0][1])

    # Autoencoder: reconstruction error is an anomaly score, not a calibrated
    # probability. We normalize it against the saved normal-training threshold.
    ae_probability = None
    ae_error = None
    if autoencoder is not None:
        reconstructed = autoencoder.predict(
            X_current.to_numpy(dtype=np.float32),
            verbose=0,
        )
        ae_error = float(
            np.mean(
                np.square(
                    X_current.to_numpy(dtype=np.float32) - reconstructed
                )
            )
        )

        if autoencoder_threshold is not None and autoencoder_threshold > 0:
            ae_probability = min(
                1.0,
                ae_error / (2.0 * autoencoder_threshold),
            )

    # LSTM: last 9 historical rows + current transaction.
    lstm_probability = None
    if lstm_model is not None:
        sequence = build_lstm_sequence(history, proxy, seq_len=10)
        lstm_probability = float(
            np.asarray(
                lstm_model.predict(sequence[np.newaxis, :, :], verbose=0)
            ).ravel()[0]
        )

    # History behavior score.
    history_risk, history_stats = calculate_history_risk(
        history,
        float(current_row["Amount"]),
    )

    signals = []
    weights = []

    if rf_probability is not None:
        signals.append(rf_probability)
        weights.append(0.25)

    signals.append(xgb_probability)
    weights.append(0.30)

    if lstm_probability is not None:
        signals.append(lstm_probability)
        weights.append(0.20)

    if ae_probability is not None:
        signals.append(ae_probability)
        weights.append(0.10)

    signals.append(history_risk)
    weights.append(0.15)

    combined_score = float(
        np.average(np.asarray(signals), weights=np.asarray(weights))
    )

    return {
        "Random Forest": rf_probability,
        "XGBoost": xgb_probability,
        "Autoencoder": ae_probability,
        "LSTM": lstm_probability,
        "History Behavior": history_risk,
        "Combined Risk Score": combined_score,
        "Autoencoder Reconstruction Error": ae_error,
        "history_stats": history_stats,
        "proxy_row": proxy,
    }


def prepare_model_input(dataframe):
    """
    Prepare a DataFrame for the already-trained model.

    The uploaded data must contain the same model features used
    during training. The saved scaler is reused for Time and Amount.
    """
    missing = [
        col for col in feature_columns
        if col not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing)
        )

    X = dataframe[feature_columns].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    invalid = int(X.isnull().sum().sum())

    if invalid > 0:
        raise ValueError(
            f"The transaction data contains {invalid} "
            "missing or non-numeric model value(s)."
        )

    scale_columns = [
        col for col in ["Time", "Amount"]
        if col in X.columns
    ]

    if scale_columns:
        X[scale_columns] = scaler.transform(
            X[scale_columns]
        )

    return X


def evaluate_predictions(actual_series, predictions, probabilities):
    actual_numeric = pd.to_numeric(
        actual_series,
        errors="coerce"
    )

    valid_mask = actual_numeric.notna()

    actual = actual_numeric[
        valid_mask
    ].astype(int).to_numpy()

    predicted = np.asarray(predictions)[
        valid_mask.to_numpy()
    ]

    probability_valid = np.asarray(probabilities)[
        valid_mask.to_numpy()
    ]

    if len(actual) == 0:
        return None

    if len(np.unique(actual)) < 2:
        return {
            "actual": actual,
            "predicted": predicted,
            "probability": probability_valid,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "cm": None,
        }

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        actual,
        probability_valid
    )

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    return {
        "actual": actual,
        "predicted": predicted,
        "probability": probability_valid,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
    }


def display_metrics(evaluation):
    if evaluation is None:
        st.warning("No valid labelled transactions were available.")
        return

    if evaluation["precision"] is None:
        st.warning(
            "Both legitimate (0) and fraudulent (1) classes are "
            "required for meaningful evaluation metrics."
        )
        return

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%"
        )

    with c3:
        st.metric(
            "F1-Score",
            f"{evaluation['f1'] * 100:.2f}%"
        )

    with c4:
        st.metric(
            "ROC-AUC",
            f"{evaluation['roc_auc'] * 100:.2f}%"
        )


def display_confusion_matrix(evaluation, title="XGBoost Confusion Matrix"):
    if evaluation is None or evaluation["cm"] is None:
        return

    cm = evaluation["cm"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm.tolist(),
            x=[
                "Predicted Legitimate",
                "Predicted Fraud"
            ],
            y=[
                "Actual Legitimate",
                "Actual Fraud"
            ],
            text=cm.tolist(),
            texttemplate="%{text}",
            textfont={"size": 22},
            colorscale="Blues",
            showscale=True,
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    tn, fp, fn, tp = cm.ravel()

    a, b, c, d = st.columns(4)

    with a:
        st.metric("TN", int(tn))

    with b:
        st.metric("FP", int(fp))

    with c:
        st.metric("FN", int(fn))

    with d:
        st.metric("TP", int(tp))


def display_roc_curve(evaluation):
    if evaluation is None or evaluation["roc_auc"] is None:
        return

    actual = evaluation["actual"]
    probability = evaluation["probability"]

    fpr, tpr, _ = roc_curve(
        actual,
        probability
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"XGBoost (AUC = {evaluation['roc_auc']:.4f})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="ROC Curve — Uploaded Dataset",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def display_permutation_importance(X_model, y, max_rows=50000):
    if len(y) == 0:
        return

    if len(y) > max_rows:
        sample_indices = np.random.RandomState(
            42
        ).choice(
            len(y),
            size=max_rows,
            replace=False
        )

        X_perm = X_model.iloc[
            sample_indices
        ]

        y_perm = y[
            sample_indices
        ]
    else:
        X_perm = X_model
        y_perm = y

    try:
        result = permutation_importance(
            model,
            X_perm,
            y_perm,
            scoring="f1",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": result.importances_mean,
            }
        ).sort_values(
            "Importance",
            ascending=False
        )

        top = importance_df.head(15).sort_values(
            "Importance",
            ascending=True
        )

        fig = go.Figure(
            go.Bar(
                x=top["Importance"],
                y=top["Feature"],
                orientation="h",
            )
        )

        fig.update_layout(
            title="Permutation Feature Importance",
            xaxis_title="Mean decrease in F1-Score",
            yaxis_title="Feature",
            height=600,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        return importance_df

    except Exception as e:
        st.warning(
            f"Feature importance could not be calculated: {e}"
        )
        return None


def add_history_record(
    user_email,
    transaction_id,
    card_last4,
    amount,
    probability,
    prediction,
):
    st.session_state.transaction_history.append(
        {
            "User": user_email,
            "Transaction ID": transaction_id,
            "Card": f"•••• {card_last4}",
            "Amount": amount,
            "Fraud Probability (%)": round(
                probability * 100,
                2
            ),
            "Result": (
                "FRAUDULENT"
                if prediction == 1
                else "LEGITIMATE"
            ),
        }
    )


# =========================================================
# LOGIN / SIGN UP
# =========================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>💳 AI-Powered Financial Fraud Detection</h1>
            <p style="font-size:18px;">
                Secure Transaction Monitoring & Fraud Detection
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    login_tab, signup_tab = st.tabs(
        ["🔐 Sign In", "📝 Sign Up"]
    )

    with login_tab:

        st.subheader("Sign In")

        with st.form("login_form"):

            login_email = st.text_input(
                "Email"
            )

            login_password = st.text_input(
                "Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "🔐 Sign In",
                type="primary"
            )

        if login_button:

            email = login_email.strip().lower()

            user = st.session_state.users.get(
                email
            )

            if (
                user is not None
                and user["password_hash"]
                == hash_password(login_password)
            ):

                # Start a completely fresh application session for this user.
                reset_user_session_data()
                st.session_state.authenticated = True
                st.session_state.current_user = email

                st.success(
                    "Login successful."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )

        st.info(
            "Demo account: demo@fraud.ai / demo123"
        )

    with signup_tab:

        st.subheader("Create Account")

        with st.form("signup_form"):

            signup_name = st.text_input(
                "Full Name"
            )

            signup_email = st.text_input(
                "Email Address"
            )

            signup_password = st.text_input(
                "Password",
                type="password"
            )

            signup_confirm = st.text_input(
                "Confirm Password",
                type="password"
            )

            signup_button = st.form_submit_button(
                "📝 Create Account",
                type="primary"
            )

        if signup_button:

            email = signup_email.strip().lower()

            if not signup_name.strip():
                st.error("Please enter your name.")

            elif not valid_email(email):
                st.error("Please enter a valid email address.")

            elif len(signup_password) < 6:
                st.error(
                    "Password must contain at least 6 characters."
                )

            elif signup_password != signup_confirm:
                st.error(
                    "Passwords do not match."
                )

            elif email in st.session_state.users:
                st.error(
                    "An account with this email already exists."
                )

            else:

                st.session_state.users[email] = {
                    "name": signup_name.strip(),
                    "password_hash": hash_password(
                        signup_password
                    ),
                }

                st.success(
                    "Account created. Please sign in."
                )

    st.stop()


# =========================================================
# APPLICATION HEADER
# =========================================================

current_user = st.session_state.current_user

user_name = st.session_state.users[
    current_user
]["name"]

st.markdown(
    """
    <h1 style="text-align:center;">
        AI-Powered Financial Fraud Detection System
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="text-align:center; font-size:18px;">
        Machine Learning • Deep Learning • Explainable AI
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

st.sidebar.success(
    f"Signed in as: {user_name}"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "💳 New Transaction",
        "📁 Batch Prediction",
        "📊 Model Performance",
        "🧠 Explainable AI",
        "📜 Transaction History",
    ],
)

if st.sidebar.button(
    "🚪 Logout"
):
    reset_user_session_data()
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.rerun()


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header(
        f"Welcome, {user_name} 👋"
    )

    st.write(
        """
        This dashboard is linked to the currently selected demo card.
        Its transaction history, fraud activity and latest transaction
        prediction are shown here.
        """
    )

    st.divider()

    # =====================================================
    # CURRENT CARD / ACCOUNT SUMMARY
    # =====================================================
    account_history = st.session_state.get("new_transaction_source_data")
    card_details = st.session_state.get("pending_card_details")
    account_bucket = st.session_state.get("demo_account_bucket")
    last_prediction = st.session_state.get("last_single_prediction")

    if account_history is not None and not account_history.empty:

        card_number = str(
            card_details.get("card_number", "")
            if card_details
            else ""
        )
        card_digits = re.sub(r"\D", "", card_number)

        if card_digits in DEMO_CARD_BY_NUMBER:
            card_index = DEMO_CARD_BY_NUMBER[card_digits]
            account_id = f"DEMO-ACCOUNT-{card_index + 1:02d}"
            card_label = DEMO_CARDS[card_index]["label"]
        else:
            account_id = (
                f"DEMO-ACCOUNT-{int(account_bucket) % 1000000:06d}"
                if account_bucket is not None
                else "DEMO ACCOUNT"
            )
            card_label = (
                card_details.get("cardholder", "Custom Card")
                if card_details
                else "Custom Card"
            )

        total_history = len(account_history)

        historical_fraud = 0
        if "Class" in account_history.columns:
            labels = pd.to_numeric(
                account_history["Class"],
                errors="coerce",
            )
            historical_fraud = int((labels == 1).sum())

        st.subheader("💳 Current Card & Account")

        st.success(
            f"Dashboard connected to **{card_label} — {account_id}**"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Historical Transactions",
                f"{total_history:,}",
            )

        with c2:
            st.metric(
                "Historical Fraud",
                f"{historical_fraud:,}",
            )

        with c3:
            fraud_rate = (
                historical_fraud / total_history * 100
                if total_history
                else 0
            )
            st.metric(
                "History Fraud Rate",
                f"{fraud_rate:.2f}%",
            )

        with c4:
            if last_prediction is not None:
                st.metric(
                    "Latest Risk",
                    last_prediction["risk"],
                )
            else:
                st.metric(
                    "Latest Risk",
                    "Not analyzed",
                )

        st.subheader("📜 Recent Card Transaction History")

        display_history = account_history.copy().tail(10)

        # Convert the raw dataset columns into a more readable transaction table.
        display_rows = []
        for i, (_, row) in enumerate(display_history.iterrows(), 1):
            class_value = int(row["Class"]) if "Class" in row and pd.notna(row["Class"]) else 0
            amount = float(row["Amount"]) if "Amount" in row else 0.0
            time_value = row["Time"] if "Time" in row else ""

            display_rows.append({
                "Transaction ID": f"{account_id}-TXN-{i:04d}",
                "Date / Time": str(time_value),
                "Amount (₹)": f"₹ {amount:,.2f}",
                "Status": "Fraudulent" if class_value == 1 else "Legitimate",
            })

        st.dataframe(
            pd.DataFrame(display_rows),
            use_container_width=True,
            hide_index=True,
        )

        if last_prediction is not None:
            st.subheader("🚨 Latest New Transaction")

            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.metric(
                    "Amount",
                    f"₹ {last_prediction['amount']:,.2f}",
                )

            with p2:
                st.metric(
                    "Fraud Probability",
                    f"{last_prediction['probability'] * 100:.2f}%",
                )

            with p3:
                st.metric(
                    "Prediction",
                    (
                        "FRAUDULENT"
                        if last_prediction["prediction"] == 1
                        else "LEGITIMATE"
                    ),
                )

            with p4:
                st.metric(
                    "Risk Level",
                    last_prediction["risk"],
                )

        st.info(
            "Changing the selected demo card in **💳 New Transaction** "
            "changes the linked account history shown on this dashboard."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Historical Transactions", "—")

        with c2:
            st.metric("Historical Fraud", "—")

        with c3:
            st.metric("History Fraud Rate", "—")

        with c4:
            st.metric("Latest Risk", "—")

        st.info(
            """
            No card has been linked yet.

            Go to **💳 New Transaction**, select one of the 10 demo cards,
            review its transaction history and enter a new transaction.
            This dashboard will then automatically show that card's history
            and latest fraud-detection result.
            """
        )

    st.divider()

    st.subheader("Fraud Detection Pipeline")

    st.info(
        """
        User Sign In
        ↓
        Select Card
        ↓
        Linked Account Transaction History
        ↓
        New Transaction
        ↓
        Data Validation & Preprocessing
        ↓
        Random Forest + XGBoost
        ↓
        Autoencoder + LSTM
        ↓
        History-Based Model Analysis
        ↓
        Combined Risk Score
        ↓
        Fraud / Legitimate
        ↓
        SHAP / LIME Explainability
        """
    )

    st.subheader("Project Models")

    model_data = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "XGBoost",
                "Autoencoder",
                "LSTM",
            ],
            "Technique": [
                "Supervised Classification",
                "Gradient Boosting",
                "Anomaly Detection",
                "Sequential Deep Learning",
            ],
            "Purpose": [
                "Fraud Classification",
                "Fraud Classification",
                "Anomaly Detection",
                "Sequential Pattern Detection",
            ],
        }
    )

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# NEW TRANSACTION / CARD DETAILS
# =========================================================

elif page == "💳 New Transaction":

    st.header("💳 Card Details → Account History → All AI Models → Fraud Detection")
    st.write(
        "Enter the card details first. The system links the card to a "
        "repeatable demo account history, studies the account's previous "
        "transactions, and then checks the new transaction against that "
        "history before producing the fraud prediction."
    )

    st.warning(
        "Demo architecture: the public Kaggle ULB dataset is anonymized and "
        "does not contain card numbers or bank-account IDs. The card number "
        "therefore cannot be genuinely matched to a real account in that "
        "dataset. This app creates a repeatable synthetic account-to-history "
        "mapping for demonstration. Never enter a real card number or CVV."
    )

    # ---------------------------------------------------------
    # STEP 1 — CARD DETAILS
    # ---------------------------------------------------------
    st.subheader("1. Enter Card Details")

    st.subheader("Choose a Demo Card")

    demo_labels = [card["label"] for card in DEMO_CARDS]
    selected_demo_label = st.selectbox(
        "Select one of 10 demo cards",
        demo_labels,
        index=0,
        help="Each demo card is linked to a different transaction-history section.",
    )

    selected_profile = next(
        card for card in DEMO_CARDS
        if card["label"] == selected_demo_label
    )

    st.info(
        "Each demo card has a different transaction history. "
        "Select another card and click the button below to load its history."
    )

    with st.form("card_details_form"):

        c1, c2 = st.columns(2)

        with c1:
            cardholder = st.text_input(
                "Card Holder Name",
                value=selected_profile["cardholder"],
            )
            card_number = st.text_input(
                "Card Number (demo/test only — do not enter a real card)",
                value=selected_profile["card_number"],
                type="password",
            )
            expiry = st.text_input(
                "Expiry Date",
                value=selected_profile["expiry"],
                max_chars=5,
            )

        with c2:
            st.write("**Card information only**")
            st.caption(
                "The new transaction will be entered after the account "
                "history is loaded."
            )

        submitted_details = st.form_submit_button(
            "🔗 Link Card to Account History",
            type="primary",
            use_container_width=True,
        )

    if submitted_details:
        digits = re.sub(r"\D", "", card_number)

        if len(digits) < 4:
            st.error("Please enter a demo/test card number with at least 4 digits.")
            st.stop()

        st.session_state.pending_card_details = {
            "cardholder": cardholder.strip() or "Demo User",
            "card_number": card_number,
            "expiry": expiry.strip() or "Not specified",
            "last4": digits[-4:],
        }

        st.session_state.last_single_prediction = None

        # Automatically load the local Kaggle dataset.
        kaggle_data = load_kaggle_transaction_dataset()

        if kaggle_data is None:
            st.error(
                "The Kaggle dataset was not found. Download creditcard.csv "
                "and place it at data/creditcard.csv."
            )
        else:
            account_bucket, account_history = build_demo_account_history(
                kaggle_data,
                card_number,
                cardholder,
                expiry,
            )
            st.session_state.new_transaction_source_data = account_history
            st.session_state.demo_account_bucket = account_bucket

            st.success(
                f"{selected_demo_label} linked to its own transaction history successfully."
            )
            st.caption(
                "This demo card has its own fixed transaction-history section. "
                "Changing the selected demo card changes the history and the "
                "model analysis."
            )

    details = st.session_state.pending_card_details
    account_history = st.session_state.get("new_transaction_source_data")
    account_bucket = st.session_state.get("demo_account_bucket")

    # ---------------------------------------------------------
    # STEP 2 — LINKED ACCOUNT HISTORY
    # ---------------------------------------------------------
    if details is not None:

        st.divider()
        st.subheader("2. Linked Bank Account Transaction History")

        if account_history is None:
            st.error(
                "No local Kaggle transaction history is available. "
                "Place creditcard.csv inside the data folder and reload."
            )
        elif account_history.empty:
            st.warning("No transactions were assigned to this demo account.")
        else:
            synthetic_account_id = (
                f"DEMO-ACCOUNT-{int(account_bucket) % 1000000:06d}"
            )

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Account", synthetic_account_id)
            with a2:
                st.metric("Historical Transactions", len(account_history))
            with a3:
                if "Class" in account_history.columns:
                    fraud_hist = int(
                        pd.to_numeric(
                            account_history["Class"],
                            errors="coerce"
                        ).fillna(0).sum()
                    )
                    st.metric("Historical Fraud Labels", fraud_hist)
                else:
                    st.metric("Historical Fraud Labels", "N/A")

            st.caption(
                "The history below is the synthetic account bucket linked to "
                "the entered demo card. It is generated from the anonymized "
                "Kaggle transaction dataset."
            )

            history_display = make_history_display(
                account_history,
                card_index=account_bucket,
            )

            st.dataframe(
                history_display.tail(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Amount (₹)": st.column_config.NumberColumn(
                        "Amount (₹)",
                        format="₹ %.2f",
                    )
                },
            )

            st.caption(
                "The table shows a bank-style view of the transaction history. "
                "Amount and fraud status come from the dataset; Location, "
                "Channel and Transaction Type are generated demo fields because "
                "the source dataset does not contain those columns."
            )

            # ---------------------------------------------------------
            # STEP 3 — ENTER THE NEW TRANSACTION
            # ---------------------------------------------------------
            st.divider()
            st.subheader("3. Enter New Transaction")

            st.write(
                "Enter the transaction you want to check. The fraud decision "
                "is generated from this new transaction together with the "
                "selected card's historical behavior."
            )

            with st.form("new_transaction_form"):

                t1, t2 = st.columns(2)

                with t1:
                    new_amount = st.number_input(
                        "Transaction Amount (₹)",
                        min_value=0.01,
                        value=1000.0,
                        step=100.0,
                    )
                    new_location = st.text_input(
                        "Transaction Location",
                        placeholder="New Delhi",
                    )

                with t2:
                    new_channel = st.selectbox(
                        "Transaction Channel",
                        ["Online", "POS", "ATM", "Mobile Banking"],
                    )
                    new_type = st.selectbox(
                        "Transaction Type",
                        ["Purchase", "Transfer", "Withdrawal", "Payment"],
                    )

                analyze_transaction = st.form_submit_button(
                    "🚨 ANALYZE NEW TRANSACTION",
                    type="primary",
                    use_container_width=True,
                )

            if analyze_transaction:

                new_transaction = {
                    "amount": float(new_amount),
                    "location": new_location.strip() or "Not specified",
                    "channel": new_channel,
                    "type": new_type,
                }

                st.session_state.current_new_transaction = new_transaction

                history_risk, history_stats = calculate_history_risk(
                    account_history,
                    new_transaction["amount"],
                )

                st.subheader("4. Submitted Transaction")

                transaction_display = pd.DataFrame([{
                    "Transaction ID": (
                        f"{synthetic_account_id}-NEW-{len(account_history) + 1:04d}"
                    ),
                    "Date & Time": pd.Timestamp.now().strftime("%d-%m-%Y %H:%M"),
                    "Amount (₹)": new_transaction["amount"],
                    "Location": new_transaction["location"],
                    "Channel": new_transaction["channel"],
                    "Transaction Type": new_transaction["type"],
                    "Status": "Analyzing",
                }])

                st.dataframe(
                    transaction_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Amount (₹)": st.column_config.NumberColumn(
                            "Amount (₹)",
                            format="₹ %.2f",
                        )
                    },
                )

                # ---------------------------------------------------------
                # STEP 5 — FRAUD DETECTION
                # ---------------------------------------------------------
                proxy_row = choose_transaction_proxy(
                    account_history,
                    new_transaction["amount"],
                )

                if proxy_row is None:
                    st.error(
                        "Unable to create model input from this card's history."
                    )
                else:
                    selected_row = proxy_row.copy()
                    selected_row["Amount"] = new_transaction["amount"]

                    transaction_df = pd.DataFrame([selected_row])

                    with st.spinner(
                        "Running all AI models on the new transaction..."
                    ):
                        analysis = analyze_all_models(
                            account_history,
                            selected_row,
                        )

                        probability = analysis["Combined Risk Score"]
                        prediction = int(probability >= 0.50)

                    risk = (
                        "HIGH" if probability >= 0.80
                        else "MEDIUM" if probability >= 0.50
                        else "LOW"
                    )

                    txn_id = (
                        f"TXN-{len(st.session_state.transaction_history) + 1:06d}"
                    )

                    add_history_record(
                        current_user,
                        txn_id,
                        details["last4"],
                        new_transaction["amount"],
                        probability,
                        prediction,
                    )

                    st.session_state.last_single_prediction = {
                        "probability": probability,
                        "prediction": prediction,
                        "transaction": transaction_df,
                        "cardholder": details["cardholder"],
                        "card": mask_card_number(details["card_number"]),
                        "expiry": details["expiry"],
                        "amount": new_transaction["amount"],
                        "location": new_transaction["location"],
                        "channel": new_transaction["channel"],
                        "type": new_transaction["type"],
                        "risk": risk,
                        "account_id": synthetic_account_id,
                        "model_analysis": analysis,
                    }

                    st.divider()
                    st.subheader("5. Fraud Detection Result")

                    r1, r2, r3 = st.columns(3)

                    with r1:
                        st.metric(
                            "Fraud Probability",
                            f"{probability * 100:.2f}%",
                        )

                    with r2:
                        st.metric(
                            "Prediction",
                            (
                                "🚨 FRAUDULENT"
                                if prediction == 1
                                else "✅ LEGITIMATE"
                            ),
                        )

                    with r3:
                        st.metric("Risk Level", risk)

                    if prediction == 1:
                        st.error(
                            "🚨 FRAUD ALERT — The entered transaction has "
                            "been classified as potentially fraudulent."
                        )
                    else:
                        st.success(
                            "✅ LEGITIMATE TRANSACTION — The entered "
                            "transaction has been classified as legitimate."
                        )

                    st.subheader("6. All Model Analysis")

                    model_rows = []
                    for model_name in [
                        "Random Forest",
                        "XGBoost",
                        "Autoencoder",
                        "LSTM",
                        "History Behavior",
                    ]:
                        value = analysis.get(model_name)

                        if value is None:
                            display_value = "Unavailable"
                        else:
                            display_value = f"{value * 100:.2f}%"

                        model_rows.append({
                            "Model / Signal": model_name,
                            "Fraud / Anomaly Score": display_value,
                        })

                    st.dataframe(
                        pd.DataFrame(model_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.metric(
                        "Combined Risk Score",
                        f"{probability * 100:.2f}%",
                    )

                    st.subheader("7. New Transaction vs Card History")

                    h1, h2, h3 = st.columns(3)

                    with h1:
                        st.metric(
                            "New Amount",
                            f"₹ {new_transaction['amount']:,.2f}",
                        )

                    with h2:
                        st.metric(
                            "Historical Median",
                            f"₹ {history_stats['median_amount']:,.2f}",
                        )

                    with h3:
                        st.metric(
                            "History-Based Risk",
                            f"{history_risk * 100:.2f}%",
                        )

                    st.caption(
                        "The final decision uses the transaction you entered "
                        "and the historical transaction pattern of the selected "
                        "demo card. The public dataset's anonymized V1–V28 "
                        "features are represented using the account-history "
                        "pattern required by the trained models."
                    )

                    transaction_display["Status"] = (
                        "Fraudulent" if prediction == 1 else "Legitimate"
                    )

                    st.subheader("8. Final Transaction Status")

                    st.dataframe(
                        transaction_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Amount (₹)": st.column_config.NumberColumn(
                                "Amount (₹)",
                                format="₹ %.2f",
                            )
                        },
                    )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📁 Batch Prediction":

    st.header(
        "📁 Batch Fraud Prediction"
    )

    st.write(
        """
        Upload a large transaction dataset. The trained XGBoost
        model will score every transaction and dynamically calculate
        evaluation results when actual Class labels are present.
        """
    )

    st.info(
        """
        Required model features:
        Time, V1, V2, V3, ... V28, Amount

        Optional evaluation column:
        Class — 0 = Legitimate, 1 = Fraud
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file
            )

            st.session_state.uploaded_batch_data = data

            missing_features = [
                col
                for col in feature_columns
                if col not in data.columns
            ]

            if missing_features:

                st.error(
                    "Missing required model features: "
                    + ", ".join(missing_features)
                )

            elif len(data) == 0:

                st.warning(
                    "The uploaded dataset contains no transactions."
                )

            else:

                st.success(
                    f"Dataset uploaded successfully: "
                    f"{len(data):,} transactions."
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Transactions",
                        f"{len(data):,}"
                    )

                with c2:
                    st.metric(
                        "Features",
                        len(data.columns)
                    )

                with c3:
                    st.metric(
                        "Missing Values",
                        int(
                            data.isnull().sum().sum()
                        )
                    )

                st.subheader(
                    "Uploaded Data Preview"
                )

                st.dataframe(
                    data.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                X = prepare_model_input(
                    data
                )

                with st.spinner(
                    f"Analyzing {len(data):,} transactions..."
                ):

                    predictions = np.asarray(
                        model.predict(X)
                    )

                    probabilities = np.asarray(
                        model.predict_proba(X)[:, 1]
                    )

                st.session_state.uploaded_batch_predictions = (
                    predictions
                )

                st.session_state.uploaded_batch_probabilities = (
                    probabilities
                )

                result = data.copy()

                result[
                    "Fraud Probability (%)"
                ] = (
                    probabilities * 100
                ).round(2)

                result[
                    "Prediction"
                ] = np.where(
                    predictions == 1,
                    "FRAUDULENT",
                    "LEGITIMATE",
                )

                total = len(result)

                fraud_count = int(
                    np.sum(
                        predictions == 1
                    )
                )

                legitimate_count = int(
                    np.sum(
                        predictions == 0
                    )
                )

                fraud_rate = (
                    fraud_count / total * 100
                    if total > 0
                    else 0
                )

                st.divider()

                st.subheader(
                    "Fraud Detection Results"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Total",
                        f"{total:,}"
                    )

                with c2:
                    st.metric(
                        "Legitimate",
                        f"{legitimate_count:,}"
                    )

                with c3:
                    st.metric(
                        "Fraudulent",
                        f"{fraud_count:,}"
                    )

                with c4:
                    st.metric(
                        "Fraud Rate",
                        f"{fraud_rate:.2f}%"
                    )

                if fraud_count > 0:

                    st.error(
                        f"🚨 FRAUD ALERT: "
                        f"{fraud_count:,} suspicious transaction(s) detected."
                    )

                else:

                    st.success(
                        "✅ No fraudulent transactions detected."
                    )

                st.subheader(
                    "Transaction Predictions"
                )

                st.dataframe(
                    result.head(1000),
                    use_container_width=True,
                    hide_index=True,
                )

                if len(result) > 1000:

                    st.caption(
                        "Showing the first 1,000 rows for display performance. "
                        "All transactions were processed."
                    )

                # =================================================
                # EVALUATION
                # =================================================

                if "Class" in data.columns:

                    st.divider()

                    st.subheader(
                        "Evaluation on Uploaded Dataset"
                    )

                    evaluation = evaluate_predictions(
                        data["Class"],
                        predictions,
                        probabilities,
                    )

                    if evaluation is not None:

                        actual = evaluation["actual"]

                        st.write(
                            "**Actual Class Distribution**"
                        )

                        a, b = st.columns(2)

                        with a:
                            st.metric(
                                "Actual Legitimate",
                                f"{int(np.sum(actual == 0)):,}"
                            )

                        with b:
                            st.metric(
                                "Actual Fraud",
                                f"{int(np.sum(actual == 1)):,}"
                            )

                        display_metrics(
                            evaluation
                        )

                        st.subheader(
                            "Confusion Matrix"
                        )

                        display_confusion_matrix(
                            evaluation
                        )

                        st.subheader(
                            "ROC Curve"
                        )

                        display_roc_curve(
                            evaluation
                        )

                        st.subheader(
                            "Feature Importance on Uploaded Data"
                        )

                        display_permutation_importance(
                            X,
                            actual
                        )

                        if evaluation["precision"] is not None:

                            st.subheader(
                                "Evaluation Interpretation"
                            )

                            st.write(
                                f"""
                                The model achieved a precision of
                                {evaluation['precision'] * 100:.2f}%. This
                                represents the proportion of transactions
                                predicted as fraudulent that were actually
                                fraudulent.
                                """
                            )

                            st.write(
                                f"""
                                The recall was
                                {evaluation['recall'] * 100:.2f}%. This
                                indicates how effectively the model detected
                                actual fraudulent transactions.
                                """
                            )

                            st.write(
                                f"""
                                The F1-Score was
                                {evaluation['f1'] * 100:.2f}%, balancing
                                precision and recall.
                                """
                            )

                            st.write(
                                f"""
                                The ROC-AUC was
                                {evaluation['roc_auc'] * 100:.2f}%, indicating
                                the model's ability to distinguish between
                                legitimate and fraudulent transactions.
                                """
                            )

                else:

                    st.info(
                        """
                        No `Class` column was found. The system can still
                        generate fraud predictions, but precision, recall,
                        F1-score, ROC-AUC and the confusion matrix require
                        actual labels.
                        """
                    )

                st.divider()

                result_csv = result.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )

                st.download_button(
                    "⬇️ Download Fraud Detection Results",
                    data=result_csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                )

        except Exception as e:

            st.error(
                f"Error processing your dataset: {e}"
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    # =====================================================
    # ACCOUNT HISTORY PERFORMANCE
    # =====================================================
    account_history = st.session_state.get("new_transaction_source_data")
    card_details = st.session_state.get("pending_card_details")
    account_bucket = st.session_state.get("demo_account_bucket")

    if account_history is not None and not account_history.empty:

        account_id = (
            f"DEMO-ACCOUNT-{int(account_bucket) + 1:02d}"
            if account_bucket is not None
            else "DEMO ACCOUNT"
        )

        st.subheader("Current Card Account — History Analysis")

        st.success(
            f"Model results are calculated from the transaction history "
            f"linked to {account_id}."
        )

        try:
            current_amount = (
                card_details.get("amount")
                if card_details is not None
                else None
            )

            with st.spinner("Analyzing account transaction history with all models..."):
                history_analysis = analyze_account_history(
                    account_history,
                    current_amount=current_amount,
                )

            h1, h2, h3, h4 = st.columns(4)

            with h1:
                st.metric(
                    "Historical Transactions",
                    f"{history_analysis['history_count']:,}",
                )

            with h2:
                st.metric(
                    "Transactions Analyzed",
                    f"{history_analysis['rows_analyzed']:,}",
                )

            with h3:
                fraud_count = history_analysis["historical_fraud_count"]
                st.metric(
                    "Historical Fraud Labels",
                    "N/A" if fraud_count is None else f"{fraud_count:,}",
                )

            with h4:
                fraud_rate = history_analysis["historical_fraud_rate"]
                st.metric(
                    "Historical Fraud Rate",
                    "N/A" if fraud_rate is None else f"{fraud_rate * 100:.2f}%",
                )

            st.divider()

            # Model score table.
            model_rows = [
                {
                    "Model": "Random Forest",
                    "Analysis from History": (
                        f"{history_analysis['Random Forest'] * 100:.2f}%"
                    ),
                    "Interpretation": "Average fraud probability",
                },
                {
                    "Model": "XGBoost",
                    "Analysis from History": (
                        f"{history_analysis['XGBoost'] * 100:.2f}%"
                    ),
                    "Interpretation": "Average fraud probability",
                },
                {
                    "Model": "Autoencoder",
                    "Analysis from History": (
                        "Unavailable"
                        if history_analysis["Autoencoder"] is None
                        else f"{history_analysis['Autoencoder']:.6f}"
                    ),
                    "Interpretation": (
                        "Reconstruction error / anomaly score"
                        if history_analysis["Autoencoder"] is not None
                        else "Model not available"
                    ),
                },
                {
                    "Model": "LSTM",
                    "Analysis from History": (
                        "Unavailable"
                        if history_analysis["LSTM"] is None
                        else f"{history_analysis['LSTM'] * 100:.2f}%"
                    ),
                    "Interpretation": (
                        "Latest history-sequence fraud probability"
                        if history_analysis["LSTM"] is not None
                        else "Model not available"
                    ),
                },
            ]

            st.subheader("All Models — Account History Results")
            st.dataframe(
                pd.DataFrame(model_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Visual comparison of probability-style model outputs.
            chart_names = []
            chart_values = []

            for name in ["Random Forest", "XGBoost", "LSTM"]:
                value = history_analysis.get(name)
                if value is not None:
                    chart_names.append(name)
                    chart_values.append(value * 100)

            if chart_names:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=chart_names,
                            y=chart_values,
                            text=[f"{v:.2f}%" for v in chart_values],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Fraud Probability Signals from Account History",
                    yaxis_title="Score (%)",
                    xaxis_title="Model",
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Confusion Matrix — Account Transaction History")
            display_history_confusion_matrices(account_history)

            if history_analysis["Autoencoder anomaly rate"] is not None:
                st.metric(
                    "Autoencoder Anomaly Rate in Account History",
                    f"{history_analysis['Autoencoder anomaly rate'] * 100:.2f}%",
                )

            if (
                current_amount is not None
                and history_analysis.get("history_stats") is not None
            ):
                st.divider()
                st.subheader("New Transaction vs Account History")

                stats = history_analysis["history_stats"]

                n1, n2, n3 = st.columns(3)
                with n1:
                    st.metric(
                        "New Transaction",
                        f"₹ {float(current_amount):,.2f}",
                    )
                with n2:
                    st.metric(
                        "Historical Median",
                        f"₹ {stats['median_amount']:,.2f}",
                    )
                with n3:
                    st.metric(
                        "History-Based Risk",
                        f"{history_analysis['new_transaction_history_risk'] * 100:.2f}%",
                    )

                st.caption(
                    "This compares the entered transaction amount with the "
                    "selected account's historical transaction behavior."
                )

            st.info(
                "The account history shown here is a deterministic demo linkage "
                "created from the anonymized transaction dataset. The dataset "
                "does not contain real card/account identifiers."
            )

        except Exception as e:
            st.error(f"Unable to analyze the account history: {e}")

    else:
        st.info(
            "Enter the card details and new transaction amount in "
            "💳 New Transaction first. Model Performance will then show "
            "Random Forest, XGBoost, Autoencoder and LSTM results based on "
            "that card's linked transaction history."
        )

    # =====================================================
    # BATCH PERFORMANCE (OPTIONAL)
    # =====================================================
    st.divider()
    st.subheader("Dataset / Batch Performance")

    uploaded = st.session_state.uploaded_batch_data
    predictions = st.session_state.uploaded_batch_predictions
    probabilities = st.session_state.uploaded_batch_probabilities

    if (
        uploaded is not None
        and predictions is not None
        and probabilities is not None
        and "Class" in uploaded.columns
    ):

        st.success(
            f"Showing dynamic performance for the uploaded "
            f"{len(uploaded):,}-transaction dataset."
        )

        try:
            X_uploaded = prepare_model_input(uploaded)
            evaluation = evaluate_predictions(
                uploaded["Class"], predictions, probabilities
            )

            if evaluation is not None:
                display_metrics(evaluation)
                st.divider()
                st.subheader("Dynamic Confusion Matrix")
                display_confusion_matrix(
                    evaluation,
                    title=(
                        f"XGBoost Confusion Matrix — {len(uploaded):,} Transactions"
                    ),
                )
                st.divider()
                st.subheader("Dynamic ROC Curve")
                display_roc_curve(evaluation)
                st.divider()
                st.subheader("Dynamic Feature Importance")
                display_permutation_importance(
                    X_uploaded, evaluation["actual"]
                )

        except Exception as e:
            st.error(f"Error generating batch performance: {e}")

    else:
        st.caption(
            "Batch metrics are optional. The section above is the model analysis "
            "for the currently selected card's transaction history."
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.header("🧠 Explainable AI")

    last_prediction = st.session_state.last_single_prediction

    if last_prediction is None:
        st.info(
            "No transaction has been analyzed in this user's session yet. "
            "Run **💳 New Transaction → DETECT FRAUD** first. Explainable AI "
            "results from another user/session are not shown."
        )
    else:
        st.success(
            "Showing explainability context for the transaction most recently "
            "analyzed by the signed-in user."
        )

        st.subheader("Latest Model Decision")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Fraud Probability",
                f"{last_prediction['probability'] * 100:.2f}%",
            )
        with c2:
            st.metric(
                "Prediction",
                "FRAUDULENT" if last_prediction["prediction"] == 1 else "LEGITIMATE",
            )
        with c3:
            st.metric("Risk Level", last_prediction["risk"])

        st.subheader("Model Input Features")
        st.dataframe(
            last_prediction["transaction"][feature_columns],
            use_container_width=True,
            hide_index=True,
        )

        shap_summary = os.path.join(RESULTS_DIR, "shap_summary.png")
        if os.path.exists(shap_summary):
            st.subheader("SHAP Summary Plot")
            st.image(shap_summary, use_container_width=True)

        shap_csv = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
        if os.path.exists(shap_csv):
            st.subheader("SHAP Feature Importance")
            shap_data = pd.read_csv(shap_csv)
            st.dataframe(
                shap_data.head(20),
                use_container_width=True,
                hide_index=True,
            )

        explanation_txt = os.path.join(
            RESULTS_DIR, "transaction_explanation.txt"
        )
        if os.path.exists(explanation_txt):
            st.subheader("Individual Transaction Explanation")
            with open(explanation_txt, "r", encoding="utf-8") as file:
                st.code(file.read())

        lime_html = os.path.join(RESULTS_DIR, "lime_explanation.html")
        if os.path.exists(lime_html):
            st.subheader("LIME Explanation")
            with open(lime_html, "r", encoding="utf-8") as file:
                st.components.v1.html(
                    file.read(), height=650, scrolling=True
                )

        st.caption(
            "The SHAP/LIME files above are project explainability artifacts. "
            "They are shown only after this user has performed a transaction "
            "analysis, preventing stale explainability output from appearing "
            "on a fresh login."
        )


# =========================================================
# TRANSACTION HISTORY
# =========================================================

elif page == "📜 Transaction History":

    st.header(
        "📜 Transaction History"
    )

    user_history = [
        item
        for item in st.session_state.transaction_history
        if item.get("User") == current_user
    ]

    if not user_history:

        st.info(
            """
            No individual transactions have been analyzed
            during this session yet.
            """
        )

    else:

        history_df = pd.DataFrame(
            user_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        history_csv = history_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download Transaction History",
            data=history_csv,
            file_name="transaction_history.csv",
            mime="text/csv",
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Powered Financial Fraud Detection System | "
    "Machine Learning + Deep Learning + Explainable AI"
)
