"""
FraudGuard AI: Enterprise RESTful API Gateway
FastAPI microservice providing real-time multi-model fraud scoring,
consensus risk tiering, TreeSHAP/LIME explainability, and Kafka streaming integration.
"""

import hashlib
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from predictor import FraudPredictor
from model_comparison import FraudModelComparison
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
_comparison_instance: Optional[FraudModelComparison] = None
_workspace_store: Optional[WorkspaceStore] = None
_simulated_broker = SimulatedStreamingBroker()


def get_predictor() -> FraudPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor(models_dir=config.MODELS_DIR)
    return _predictor_instance


def get_comparison_engine() -> FraudModelComparison:
    global _comparison_instance
    if _comparison_instance is None:
        _comparison_instance = FraudModelComparison(models_dir=config.MODELS_DIR)
    return _comparison_instance


def get_store() -> WorkspaceStore:
    global _workspace_store
    if _workspace_store is None:
        _workspace_store = WorkspaceStore(db_path=config.DB_PATH)
    return _workspace_store


def authenticate_request(
    x_api_key: Optional[str] = Header(None, description="API Key for microservice authentication"),
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
) -> str:
    """Validate API Key authentication and client identity with multi-process SQLite rate limiting."""
    if config.API_KEY_ENABLED:
        if not x_api_key or x_api_key.strip() not in config.VALID_API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing 'X-API-Key' authentication header.",
            )

    analyst_id = (x_analyst_id or "").strip()
    if not analyst_id or len(analyst_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'X-Analyst-ID' must be at least 3 characters long.",
        )

    # Multi-process persistent rate limit strictly bound to authenticated API key hash
    raw_key = (x_api_key or "anonymous").strip()
    auth_identity = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    store = get_store()

    allowed = store.check_and_record_rate_limit(
        client_key=auth_identity,
        limit=config.RATE_LIMIT_PER_MINUTE,
        window_seconds=60.0,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: Maximum {config.RATE_LIMIT_PER_MINUTE} requests per minute.",
        )

    return analyst_id


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
    family: str
    prediction: str
    score: float
    threshold: float
    score_kind: str


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
    primary_model: str
    consensus_score: float
    consensus_percentage: float
    fraud_votes: int
    models_checked: int
    flagged_by: List[str]
    model_breakdown: List[ModelScoreBreakdown]
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
    api_authentication_enabled: bool
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
        "authentication_enabled": config.API_KEY_ENABLED,
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
        api_authentication_enabled=config.API_KEY_ENABLED,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Scoring"])
def predict_transaction(
    payload: TransactionInput,
    x_api_key: Optional[str] = Header(None, description="Microservice API Key"),
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
):
    """
    Score an individual transaction in real time.
    Evaluates multi-model consensus fusion (all 4 models), balance forensics, and risk tier routing.
    """
    authenticate_request(x_api_key=x_api_key, x_analyst_id=x_analyst_id)
    start_t = time.perf_counter()

    predictor = get_predictor()
    comp = get_comparison_engine()

    raw_dict = payload.model_dump()
    txn_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
    raw_dict["Transaction_ID"] = txn_id

    try:
        # 1. Primary individual model prediction
        pred_dict = predictor.predict_single(raw_dict)
        # 2. Multi-model consensus comparison across all 4 models
        comp_report = comp.predict_single(raw_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failure: {e}",
        )

    # Calculate actual consensus metrics across all 4 models
    votes = int(comp_report["Fraud_Votes"])
    checked = int(comp_report["Models_Checked"])
    consensus_fraction = round(votes / checked, 4) if checked > 0 else 0.0
    consensus_pct = round(consensus_fraction * 100.0, 2)

    model_breakdown = [
        ModelScoreBreakdown(
            model_name=m["Model"],
            family=m["Family"],
            prediction=m["Prediction"],
            score=float(m["Score"]),
            threshold=float(m["Threshold"]),
            score_kind=m["Score_Kind"],
        )
        for m in comp_report.get("models", [])
    ]

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
        primary_model=predictor.model_name,
        consensus_score=consensus_fraction,
        consensus_percentage=consensus_pct,
        fraud_votes=votes,
        models_checked=checked,
        flagged_by=comp_report.get("Flagged_By", []),
        model_breakdown=model_breakdown,
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
    x_api_key: Optional[str] = Header(None, description="Microservice API Key"),
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
):
    """
    Score a batch of financial transactions with vectorized feature transformation
    and multi-model consensus adjudication.
    """
    authenticate_request(x_api_key=x_api_key, x_analyst_id=x_analyst_id)
    start_t = time.perf_counter()

    if not payload.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction batch cannot be empty.",
        )

    # Request size guard against DoS
    if len(payload.transactions) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload too large: Batch size exceeds maximum limit of {config.MAX_BATCH_SIZE} transactions.",
        )

    predictor = get_predictor()
    comp = get_comparison_engine()

    records = [t.model_dump() for t in payload.transactions]
    df = pd.DataFrame(records)

    try:
        out_df = predictor.predict_batch(df)
        comp_df = comp.predict_batch(df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch scoring failure: {e}",
        )

    results = []
    fraud_count = 0

    for idx in range(len(out_df)):
        row = out_df.iloc[idx]
        comp_row = comp_df.iloc[idx]

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

        votes = int(comp_row["Fraud_Votes"])
        checked = int(comp_row["Models_Checked"])
        consensus_frac = round(votes / checked, 4) if checked > 0 else 0.0
        flagged_list = [name for name in comp.model_names if comp_row.get(f"{name} Prediction") == "FRAUDULENT"]

        model_breakdown = [
            ModelScoreBreakdown(
                model_name=name,
                family=config.MODEL_FAMILIES.get(name, "ML/DL"),
                prediction=str(comp_row.get(f"{name} Prediction", "LEGITIMATE")),
                score=float(comp_row.get(f"{name} Score", 0.0)),
                threshold=float(comp_row.get(f"{name} Threshold", 0.5)),
                score_kind=comp._predictors[name].score_kind,
            )
            for name in comp.model_names
        ]

        results.append(
            SinglePredictionResponse(
                transaction_id=str(row["Transaction_ID"]),
                prediction=str(row["Prediction"]),
                fraud_probability=float(row["Fraud_Probability"]),
                primary_model=predictor.model_name,
                consensus_score=consensus_frac,
                consensus_percentage=round(consensus_frac * 100.0, 2),
                fraud_votes=votes,
                models_checked=checked,
                flagged_by=flagged_list,
                model_breakdown=model_breakdown,
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
def publish_streaming_event(
    event: TransactionInput,
    mode: str = Query("auto", description="Streaming mode: 'auto', 'live' (requires live Kafka), or 'simulation'"),
    x_api_key: Optional[str] = Header(None, description="Microservice API Key"),
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
):
    """
    Ingest payment event into real-time Apache Kafka topic (financial.transactions)
    and process through streaming consensus consumer.
    """
    authenticate_request(x_api_key=x_api_key, x_analyst_id=x_analyst_id)

    raw_event = event.model_dump()

    if mode == "live":
        try:
            producer = TransactionStreamProducer(use_simulated=False, allow_fallback=False)
            consumer = FraudStreamingConsumer(use_simulated=False, allow_fallback=False)
            publish_receipt = producer.produce(raw_event)
            # Actually poll and consume from the live Kafka topic
            processed_alert = consumer.consume_one(timeout=3.0)
            if processed_alert is None:
                processed_alert = consumer.process_message(publish_receipt)
            streaming_engine = "live_kafka"
        except ConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Live Kafka streaming unavailable: {exc}. Ensure Kafka broker is running on {config.KAFKA_BOOTSTRAP_SERVERS}.",
            )
    else:
        # Simulated or auto mode with thread-safe broker queue
        producer = TransactionStreamProducer(use_simulated=True, simulated_broker=_simulated_broker)
        consumer = FraudStreamingConsumer(use_simulated=True, simulated_broker=_simulated_broker)
        publish_receipt = producer.produce(raw_event)
        # Actually poll and consume from the simulated broker topic queue
        processed_alert = consumer.consume_one(timeout=2.0)
        if processed_alert is None:
            processed_alert = consumer.process_message(publish_receipt)
        streaming_engine = "simulated_broker"

    return {
        "status": "stream_processed",
        "streaming_engine": streaming_engine,
        "producer_receipt": publish_receipt,
        "streaming_alert": processed_alert,
    }


@app.get("/stream/alerts", tags=["Streaming"])
def get_streaming_alerts(
    limit: int = Query(50, ge=1, le=200),
    mode: str = Query("simulation", description="Streaming mode: 'simulation' or 'live'"),
    x_api_key: Optional[str] = Header(None, description="Microservice API Key"),
    x_analyst_id: Optional[str] = Header("api-gateway-service", description="Analyst/Client ID"),
):
    """Retrieve recent fraud alerts published to the alerts streaming topic."""
    authenticate_request(x_api_key=x_api_key, x_analyst_id=x_analyst_id)
    alerts = []
    if mode == "simulation":
        history = _simulated_broker.get_history(config.KAFKA_ALERTS_TOPIC, limit=limit)
        for msg in history:
            try:
                alerts.append(msg.deserialize())
            except Exception:
                pass
    return {
        "status": "success",
        "total_alerts": len(alerts),
        "alerts": alerts,
    }


@app.get("/models/info", tags=["Models"])
def get_model_information():
    """Retrieve operational thresholds, feature topology, and model metadata."""
    predictor = get_predictor()
    comp = get_comparison_engine()
    info = dict(predictor.metadata)
    info["models_loaded"] = comp.model_names
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
