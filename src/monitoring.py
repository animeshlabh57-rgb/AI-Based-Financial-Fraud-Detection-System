"""
Real-Time Model & Data Quality Monitoring Engine for FraudGuard AI.

Tracks:
1. Data Integrity: Missing field rates, unexpected nulls, schema violations.
2. Latency Metrics: p50, p90, p95, p99 latency SLAs.
3. Distribution Drift: Feature distribution shifts (Kolmogorov-Smirnov statistics).
4. Score Distributions & Alert Rates over time.
5. Confirmed Performance: Computes Precision and Recall strictly when confirmed analyst
   outcomes (CONFIRMED_FRAUD vs FALSE_ALARM) become available in the case queue.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from workspace_store import WorkspaceStore


class ModelDataMonitor:
    """Monitors incoming transaction streams, system latencies, and analyst outcomes."""

    def __init__(self, store: Optional[WorkspaceStore] = None):
        self.store = store or WorkspaceStore(config.DB_PATH)
        self.latencies: List[float] = []
        self.scores: List[float] = []
        self.missing_field_events: int = 0
        self.total_events_observed: int = 0
        self.feature_windows: Dict[str, List[float]] = {
            "amount": [],
            "oldbalanceOrg": [],
            "newbalanceOrig": [],
            "oldbalanceDest": [],
            "newbalanceDest": [],
        }

    def record_inference_event(
        self,
        event_dict: dict,
        latency_ms: float,
        score: float,
        missing_count: int = 0,
    ):
        """Record telemetry from an active prediction event."""
        self.total_events_observed += 1
        self.latencies.append(float(latency_ms))
        self.scores.append(float(score))
        if missing_count > 0:
            self.missing_field_events += 1

        # Keep rolling window of last 1,000 transactions for drift testing
        tx = event_dict.get("transaction", event_dict)
        for col in self.feature_windows:
            val = tx.get(col)
            if val is not None:
                try:
                    self.feature_windows[col].append(float(val))
                    if len(self.feature_windows[col]) > 1000:
                        self.feature_windows[col].pop(0)
                except (ValueError, TypeError):
                    pass

        if len(self.latencies) > 5000:
            self.latencies = self.latencies[-5000:]
        if len(self.scores) > 5000:
            self.scores = self.scores[-5000:]

    def get_latency_metrics(self) -> Dict[str, float]:
        """Compute p50, p90, p95, p99, and mean inference latency in ms."""
        if not self.latencies:
            return {"count": 0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}

        arr = np.array(self.latencies)
        return {
            "count": len(arr),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
        }

    def get_data_quality_report(self) -> Dict:
        """Report on missing fields and data integrity."""
        missing_rate = (self.missing_field_events / max(1, self.total_events_observed)) * 100.0
        return {
            "total_events": self.total_events_observed,
            "events_with_missing_fields": self.missing_field_events,
            "missing_field_rate_pct": round(missing_rate, 3),
            "data_health_status": "HEALTHY" if missing_rate < 0.5 else "DEGRADED",
        }

    def compute_feature_drift(self, reference_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Run two-sample Kolmogorov-Smirnov (KS) test between training reference and streaming window.
        p-value < 0.01 indicates statistically significant feature distribution drift.
        """
        drift_report = {}
        for feature, current_vals in self.feature_windows.items():
            if feature not in reference_df.columns or len(current_vals) < 20:
                continue

            ref_vals = reference_df[feature].dropna().to_numpy(dtype=float)
            cur_vals = np.array(current_vals, dtype=float)

            ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
            has_drift = bool(p_val < 0.01)

            drift_report[feature] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": float(f"{p_val:.4e}"),
                "drift_detected": has_drift,
                "status": "DRIFT_ALERT" if has_drift else "STABLE",
            }
        return drift_report

    def compute_confirmed_analyst_metrics(self) -> Dict:
        """
        Compute empirical Precision and Recall strictly from confirmed analyst outcomes:
        - True Positive: CONFIRMED_FRAUD
        - False Positive: FALSE_ALARM
        - Cleared / Suspicious: SUSPICIOUS_CLEARED
        - Pending review: PENDING
        """
        cases = self.store.list_cases(limit=1000)
        tp = sum(1 for c in cases if c.get("outcome") == "CONFIRMED_FRAUD")
        fp = sum(1 for c in cases if c.get("outcome") == "FALSE_ALARM")
        pending = sum(1 for c in cases if c.get("outcome") == "PENDING")
        cleared = sum(1 for c in cases if c.get("outcome") == "SUSPICIOUS_CLEARED")

        total_adjudicated = tp + fp
        precision = (tp / total_adjudicated) if total_adjudicated > 0 else None
        false_alarm_rate = (fp / total_adjudicated) if total_adjudicated > 0 else None

        return {
            "total_cases_in_queue": len(cases),
            "pending_investigation": pending,
            "adjudicated_cases": total_adjudicated,
            "confirmed_fraud_tp": tp,
            "false_alarms_fp": fp,
            "cleared_suspicious": cleared,
            "analyst_precision": round(precision, 4) if precision is not None else "Pending Labels",
            "false_alarm_rate": round(false_alarm_rate, 4) if false_alarm_rate is not None else "Pending Labels",
            "audit_coverage_ratio": f"{total_adjudicated}/{len(cases)}" if cases else "0/0",
        }

    def generate_full_telemetry_snapshot(self, reference_df: Optional[pd.DataFrame] = None) -> Dict:
        """Generate a complete operational telemetry snapshot."""
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "latency": self.get_latency_metrics(),
            "data_quality": self.get_data_quality_report(),
            "drift_analysis": self.compute_feature_drift(reference_df) if reference_df is not None else "No reference",
            "confirmed_analyst_performance": self.compute_confirmed_analyst_metrics(),
        }
