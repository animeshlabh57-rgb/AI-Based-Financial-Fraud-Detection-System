"""
Financial Cost & Threshold Sensitivity Simulator for FraudGuard AI.

Calculates business trade-offs on labeled evaluation data:
- Detected Fraud (TP) vs Missed Fraud (FN)
- False Alarms (FP) vs Review Queue Workload (TP + FP)
- Expected Financial Loss across varying decision thresholds theta in [0.01, 0.99]
- Computes Optimal Financial Operating Threshold theta*
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config


class CostThresholdSimulator:
    """Simulates financial loss, operational workload, and optimal thresholds."""

    def __init__(
        self,
        cost_missed_fraud: float = 750.0,
        cost_analyst_review: float = 20.0,
        cost_customer_friction: float = 15.0,
    ):
        """
        Explicit financial cost assumptions (all units in currency units / USD):
        - cost_missed_fraud: Average financial loss per undetected fraud (FN).
        - cost_analyst_review: Operational cost of human investigation per flagged case (TP + FP).
        - cost_customer_friction: Customer dissatisfaction / churn penalty for false decline (FP).
        """
        self.cost_missed_fraud = float(cost_missed_fraud)
        self.cost_analyst_review = float(cost_analyst_review)
        self.cost_customer_friction = float(cost_customer_friction)

    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ) -> Dict:
        """Evaluate operational metrics and financial cost for a specific threshold."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        total_tx = len(y_true)
        total_fraud = int(np.sum(y_true == 1))
        review_workload = tp + fp

        recall = (tp / total_fraud) if total_fraud > 0 else 0.0
        precision = (tp / review_workload) if review_workload > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Financial cost components
        fraud_loss = fn * self.cost_missed_fraud
        investigation_cost = review_workload * self.cost_analyst_review
        friction_loss = fp * self.cost_customer_friction
        total_financial_cost = fraud_loss + investigation_cost + friction_loss

        return {
            "threshold": round(float(threshold), 4),
            "total_transactions": total_tx,
            "actual_frauds": total_fraud,
            "detected_fraud_tp": tp,
            "missed_fraud_fn": fn,
            "false_alarms_fp": fp,
            "approved_legitimate_tn": tn,
            "detection_rate_pct": round(recall * 100.0, 2),
            "false_alarm_rate_pct": round((fp / max(1, tn + fp)) * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "f1_score": round(f1, 4),
            "review_queue_workload": review_workload,
            "review_workload_pct": round((review_workload / total_tx) * 100.0, 2),
            "financial_breakdown": {
                "direct_fraud_loss": round(fraud_loss, 2),
                "investigation_workload_cost": round(investigation_cost, 2),
                "customer_friction_loss": round(friction_loss, 2),
                "total_expected_cost": round(total_financial_cost, 2),
            },
            "cost_assumptions": {
                "cost_per_missed_fraud": self.cost_missed_fraud,
                "cost_per_analyst_review": self.cost_analyst_review,
                "cost_per_false_alarm": self.cost_customer_friction,
            },
        }

    def simulate_cost_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        steps: int = 100,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Compute full cost curve across thresholds and locate the cost-minimizing threshold."""
        thresholds = np.linspace(0.01, 0.99, steps)
        records = []

        best_cost = float("inf")
        best_threshold_eval = None

        for th in thresholds:
            eval_res = self.evaluate_threshold(y_true, y_prob, th)
            tot_cost = eval_res["financial_breakdown"]["total_expected_cost"]

            records.append({
                "Threshold": th,
                "Total Expected Cost ($)": tot_cost,
                "Fraud Loss ($)": eval_res["financial_breakdown"]["direct_fraud_loss"],
                "Review Cost ($)": eval_res["financial_breakdown"]["investigation_workload_cost"],
                "Friction Cost ($)": eval_res["financial_breakdown"]["customer_friction_loss"],
                "Detected Fraud (TP)": eval_res["detected_fraud_tp"],
                "Missed Fraud (FN)": eval_res["missed_fraud_fn"],
                "False Alarms (FP)": eval_res["false_alarms_fp"],
                "Workload (Cases)": eval_res["review_queue_workload"],
                "Detection Rate (%)": eval_res["detection_rate_pct"],
                "F1-Score": eval_res["f1_score"],
            })

            if tot_cost < best_cost:
                best_cost = tot_cost
                best_threshold_eval = eval_res

        df = pd.DataFrame(records)
        return df, best_threshold_eval
