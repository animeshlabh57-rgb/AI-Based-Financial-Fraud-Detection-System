"""
FraudGuard AI: Enterprise RESTful API Gateway
FastAPI microservice providing real-time multi-model fraud scoring,
consensus risk tiering, TreeSHAP/LIME explainability, and Kafka streaming integration.
"""

import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from predictor import FraudPredictor
from workspace_store import WorkspaceStore
from kafka_streaming import (
    FraudStreamingConsumer,
    SimulatedStreamingBroker,
    TransactionStreamProducer,
)

# Initialize FastAPI application
app = FastAPI(
    title="FraudGuard AI - Enterprise Fraud Detection API",
    description=(
        "Production RESTful API providing explainable multi-model machine learning "
        "consensus scoring for financial transactions with secure audit storage and "
        "real-time Apache Kafka event streaming."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web dashboards and external microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service start timestamp for uptime calculation
SERVICE_START_TIME = time.time()

# Lazily initialized model singletons
_predictor_instance: Optional[FraudPredictor] = None
_workspace_store: Optional[WorkspaceStore] = None
_simulated_broker = SimulatedStreamingBroker()


def get_predictor() -> FraudPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor(models_dir=config.MODELS_DIR)
    return _predictor_instance


def get_store() -> WorkspaceStore:
    global _workspace_store
    if _workspace_store is None:
        _workspace_store = WorkspaceStore(db_path=config.DB_PATH)
    return _workspace_store


# -------------------------------------------------------------
# Pydantic Request & Response Schemas
# -------------------------------------------------------------
class TransactionInput(BaseModel):
    step: int = Field(1, description="Simulation temporal time step (1 hour per step)")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN")
    amount: float = Field(..., ge=0.0, description="Transaction monetary amount")
    nameOrig: str = Field(default="C_ORIGIN", description="Origin account identifier (e.g. C123456789)")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance prior to transaction")
    newbalanceOrig: float = Field(..., ge=0.0, description="Origin balance following transaction")
    nameDest: str = Field(default="M_DEST", description="Destination account identifier (e.g. M987654321)")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination balance prior to transaction")
    newbalanceDest: float = Field(..., ge=0.0, description="Destination balance following transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C123456789",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C987654321",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., description="List of transactions to score")


class ModelScoreBreakdown(BaseModel):
    model_name: str
    probability: float
    score_percentage: float
    decision_threshold: float
    predicted_label: str


class BalanceForensics(BaseModel):
    error_balance_orig: float = Field(..., description="Δorig = newbalanceOrig + amount - oldbalanceOrg")
    error_balance_dest: float = Field(..., description="Δdest = oldbalanceDest + amount - newbalanceDest")
    account_depletion_ratio: float = Field(..., description="Rdeplete = amount / (oldbalanceOrg + 1e-5)")
    is_complete_depletion: bool
    is_dest_zero_after_transfer: bool


class SinglePredictionResponse(BaseModel):
    transaction_id: str
    prediction: str
    fraud_probability: float
    consensus_score: float
    consensus_percentage: float
    risk_tier: str
    risk_level: str
    recommended_action: str
    processing_latency_ms: float
    balance_forensics: BalanceForensics
    top_risk_factors: List[str]
    timestamp_utc: str


class BatchPredictionResponse(BaseModel):
    batch_size: int
    total_flagged_fraud: int
    total_approved: int
    processing_latency_ms: float
    results: List[SinglePredictionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    models_available: List[str]
    active_best_model: str
    database_connected: bool
    timestamp_utc: str


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["General"])
def root_endpoint():
    """Service health & documentation gateway."""
    return {
        "service": "FraudGuard AI Enterprise Prediction Engine",
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Verify system health, loaded ML models, and database connectivity."""
    try:
        predictor = get_predictor()
        models_avail = list(predictor.metadata.get("models", {}).keys())
        best_mod = predictor.model_name
    except Exception as e:
        models_avail = []
        best_mod = f"Error loading models: {e}"

    db_ok = True
    try:
        store = get_store()
        db_ok = store.ping()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if models_avail and db_ok else "degraded",
        service="FraudGuard AI API",
        version="1.0.0",
        uptime_seconds=round(time.time() - SERVICE_START_TIME, 2),
        models_available=models_avail,
        active_best_model=best_mod,
        database_connected=db_ok,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Scoring"])
def predict_transaction(
    payload: TransactionInput,
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
):
    """
    Score an individual transaction in real time (<100 ms).
    Evaluates multi-model consensus fusion, balance forensics, and risk tier routing.
    """
    start_t = time.perf_counter()
    predictor = get_predictor()

    raw_dict = payload.model_dump()
    txn_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
    raw_dict["Transaction_ID"] = txn_id

    try:
        pred_dict = predictor.predict_single(raw_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failure: {e}",
        )

    # Balance Forensics calculation
    amt = payload.amount
    old_org = payload.oldbalanceOrg
    new_org = payload.newbalanceOrig
    old_dest = payload.oldbalanceDest
    new_dest = payload.newbalanceDest

    err_org = round(new_org + amt - old_org, 2)
    err_dest = round(old_dest + amt - new_dest, 2)
    depletion_ratio = round(amt / (old_org + 1e-5), 4) if old_org > 0 else 0.0
    complete_drain = bool(old_org > 0 and new_org == 0)
    dest_zero = bool(old_dest == 0 and new_dest == 0 and amt > 0)

    forensics = BalanceForensics(
        error_balance_orig=err_org,
        error_balance_dest=err_dest,
        account_depletion_ratio=depletion_ratio,
        is_complete_depletion=complete_drain,
        is_dest_zero_after_transfer=dest_zero,
    )

    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    return SinglePredictionResponse(
        transaction_id=txn_id,
        prediction=pred_dict["Prediction"],
        fraud_probability=pred_dict["Fraud_Probability"],
        consensus_score=pred_dict["Fraud_Probability"],
        consensus_percentage=round(pred_dict["Fraud_Probability"] * 100.0, 2),
        risk_tier=pred_dict["Risk_Level"],
        risk_level=pred_dict["Risk_Level"],
        recommended_action=pred_dict["Recommended_Action"],
        processing_latency_ms=latency_ms,
        balance_forensics=forensics,
        top_risk_factors=pred_dict.get("Top_Risk_Factors", []),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Scoring"])
def predict_batch_transactions(
    payload: BatchTransactionInput,
    x_analyst_id: Optional[str] = Header("api-gateway-service"),
):
    """
    Score a batch of financial transactions with vectorized feature transformation.
    """
    start_t = time.perf_counter()
    predictor = get_predictor()

    if not payload.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction batch cannot be empty.",
        )

    records = [t.model_dump() for t in payload.transactions]
    df = pd.DataFrame(records)

    try:
        out_df = predictor.predict_batch(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch scoring failure: {e}",
        )

    results = []
    fraud_count = 0

    for idx, row in out_df.iterrows():
        is_fraud = row["Prediction"] == "FRAUDULENT"
        if is_fraud:
            fraud_count += 1

        amt = float(row["amount"])
        old_o = float(row["oldbalanceOrg"])
        new_o = float(row["newbalanceOrig"])
        old_d = float(row["oldbalanceDest"])
        new_d = float(row["newbalanceDest"])

        f_obj = BalanceForensics(
            error_balance_orig=round(new_o + amt - old_o, 2),
            error_balance_dest=round(old_d + amt - new_d, 2),
            account_depletion_ratio=round(amt / (old_o + 1e-5), 4) if old_o > 0 else 0.0,
            is_complete_depletion=bool(old_o > 0 and new_o == 0),
            is_dest_zero_after_transfer=bool(old_d == 0 and new_d == 0 and amt > 0),
        )

        results.append(
            SinglePredictionResponse(
                transaction_id=str(row["Transaction_ID"]),
                prediction=str(row["Prediction"]),
                fraud_probability=float(row["Fraud_Probability"]),
                consensus_score=float(row["Fraud_Probability"]),
                consensus_percentage=round(float(row["Risk_Score_Pct"]), 2),
                risk_tier=str(row["Risk_Level"]),
                risk_level=str(row["Risk_Level"]),
                recommended_action=str(row["Recommended_Action"]),
                processing_latency_ms=0.0,
                balance_forensics=f_obj,
                top_risk_factors=predictor.identify_contributing_risk_factors(row),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

    total_latency = round((time.perf_counter() - start_t) * 1000.0, 2)

    return BatchPredictionResponse(
        batch_size=len(payload.transactions),
        total_flagged_fraud=fraud_count,
        total_approved=len(payload.transactions) - fraud_count,
        processing_latency_ms=total_latency,
        results=results,
    )


@app.post("/stream/publish", tags=["Streaming"])
def publish_streaming_event(event: TransactionInput):
    """
    Ingest payment event into real-time Apache Kafka topic (financial.transactions)
    and process through streaming consensus consumer.
    """
    producer = TransactionStreamProducer(use_simulated=True, simulated_broker=_simulated_broker)
    consumer = FraudStreamingConsumer(use_simulated=True, simulated_broker=_simulated_broker)

    raw_event = event.model_dump()
    publish_receipt = producer.produce(raw_event)
    processed_alert = consumer.process_message(publish_receipt)

    return {
        "status": "stream_processed",
        "producer_receipt": publish_receipt,
        "streaming_alert": processed_alert,
    }


@app.get("/models/info", tags=["Models"])
def get_model_information():
    """Retrieve operational thresholds, feature topology, and model metadata."""
    predictor = get_predictor()
    info = dict(predictor.metadata)
    info["models_loaded"] = list(predictor.metadata.get("models", {}).keys())
    info["weights"] = predictor.metadata.get(
        "weights",
        {"Random Forest": 0.25, "XGBoost": 0.35, "Deep Neural Network": 0.20, "Deep Autoencoder": 0.20}
    )
    info["calibrated_threshold"] = predictor.threshold
    return info


if __name__ == "__main__":
    import uvicorn
    print("Starting FraudGuard AI FastAPI microservice on http://127.0.0.1:8000 ...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True, app_dir=os.path.dirname(__file__))
