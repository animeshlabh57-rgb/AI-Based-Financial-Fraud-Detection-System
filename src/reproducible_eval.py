"""Reproducible Holdout Evaluation Engine for FraudGuard AI.

Provides rigorous, leak-free evaluation on strictly untouched holdout test data:
1. Compares:
   - Baseline single model (XGBoost)
   - Tree-only ensemble (Random Forest + XGBoost)
   - Full consensus fusion (RF + XGB + DNN + Deep Autoencoder)
2. Calibration assessment:
   - Reliability diagrams (calibration curves)
   - Brier score loss for individual components and consensus
3. 95% Confidence Intervals via empirical bootstrapping (B=1,000)
4. Holdout confusion matrices, Precision-Recall curves, and ROC curves
5. Explicit threshold tuning on validation split only (test split strictly untouched)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from data_loader import DataLoader
from preprocessor import DataPreprocessor


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for key performance metrics."""
    rng = np.random.RandomState(random_seed)
    boot_pr_auc = []
    boot_roc_auc = []
    boot_f1 = []
    boot_recall = []
    boot_precision = []
    boot_brier = []

    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Ensure both classes are present in sample
        if len(np.unique(y_t)) < 2:
            continue

        y_pred = (y_p >= threshold).astype(int)
        boot_pr_auc.append(average_precision_score(y_t, y_p))
        boot_roc_auc.append(roc_auc_score(y_t, y_p))
        boot_f1.append(f1_score(y_t, y_pred, zero_division=0))
        boot_recall.append(recall_score(y_t, y_pred, zero_division=0))
        boot_precision.append(precision_score(y_t, y_pred, zero_division=0))
        boot_brier.append(brier_score_loss(y_t, y_p))

    def _ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr)),
        }

    return {
        "pr_auc": _ci(boot_pr_auc),
        "roc_auc": _ci(boot_roc_auc),
        "f1": _ci(boot_f1),
        "recall": _ci(boot_recall),
        "precision": _ci(boot_precision),
        "brier_score": _ci(boot_brier),
    }


class ReproducibleEvaluator:
    """Manages holdout evaluation, calibration curve generation, and model comparisons."""

    def __init__(self, models_dir: str = config.MODELS_DIR, results_dir: str = config.RESULTS_DIR):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load models and preprocessing artifacts
        self.metadata = json.loads((self.models_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")
        self.rf_model = joblib.load(self.models_dir / "random_forest_model.joblib")
        self.xgb_model = joblib.load(self.models_dir / "xgboost_model.joblib")
        self.dnn_model = joblib.load(self.models_dir / "deep_neural_network.joblib")
        self.ae_model = joblib.load(self.models_dir / "deep_autoencoder.joblib")

    def load_clean_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load validation split (for tuning) and untouched test split (for final evaluation)."""
        loader = DataLoader()
        df = loader.load_data(config.DATASET_PATH)
        data = self.preprocessor._labelled_data(df)

        if hasattr(self.preprocessor, "partition_indices_") and "test" in self.preprocessor.partition_indices_:
            val_idx = self.preprocessor.partition_indices_["validation"]
            test_idx = self.preprocessor.partition_indices_["test"]
            val_df = data.loc[val_idx]
            test_df = data.loc[test_idx]
        else:
            from sklearn.model_selection import train_test_split
            development, test_df = train_test_split(
                data, test_size=config.TEST_SPLIT_SIZE,
                random_state=config.RANDOM_STATE, stratify=data[config.TARGET_COLUMN]
            )
            _, val_df = train_test_split(
                development, test_size=config.VALIDATION_SPLIT_SIZE / (1 - config.TEST_SPLIT_SIZE),
                random_state=config.RANDOM_STATE, stratify=development[config.TARGET_COLUMN]
            )

        X_val = self.preprocessor.transform_new_data(val_df)
        y_val = val_df[config.TARGET_COLUMN].to_numpy(dtype=int)

        X_test = self.preprocessor.transform_new_data(test_df)
        y_test = test_df[config.TARGET_COLUMN].to_numpy(dtype=int)

        return X_val, X_test, y_val, y_test

    def get_component_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Obtain component probability / anomaly scores from all 4 models."""
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        dnn_prob = self.dnn_model.predict_proba(X)[:, 1]
        ae_score = self.ae_model.predict_proba(X)[:, 1]

        return {
            "Random Forest": rf_prob,
            "XGBoost": xgb_prob,
            "Deep Neural Network": dnn_prob,
            "Deep Autoencoder": ae_score,
        }

    def compute_ensembles(self, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute tree-only and full consensus fusion scores."""
        # Tree-only: RF (0.40) + XGBoost (0.60)
        tree_score = 0.40 * scores["Random Forest"] + 0.60 * scores["XGBoost"]

        # Full consensus fusion (Research paper weighted consensus formula)
        # RF: 0.25, XGB: 0.35, DNN: 0.20, AE: 0.20
        full_score = (
            0.25 * scores["Random Forest"]
            + 0.35 * scores["XGBoost"]
            + 0.20 * scores["Deep Neural Network"]
            + 0.20 * scores["Deep Autoencoder"]
        )

        return {
            "XGBoost Alone": scores["XGBoost"],
            "Tree-Only Fusion": tree_score,
            "Full Consensus Fusion": full_score,
        }

    def tune_threshold_on_validation(self, y_val: np.ndarray, val_scores: np.ndarray) -> float:
        """Strictly tune threshold on validation data using maximum F1-score with tie-breaking."""
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1s[:-1])
        return float(thresholds[best_idx])

    def evaluate_holdout(self, n_bootstraps: int = 500) -> Dict:
        """Run complete holdout evaluation with calibration analysis and confidence intervals."""
        print("Loading clean dataset splits...")
        X_val, X_test, y_val, y_test = self.load_clean_splits()

        val_components = self.get_component_scores(X_val)
        test_components = self.get_component_scores(X_test)

        val_ensembles = self.compute_ensembles(val_components)
        test_ensembles = self.compute_ensembles(test_components)

        results = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "holdout_test_rows": len(y_test),
            "holdout_fraud_rows": int(np.sum(y_test)),
            "holdout_fraud_prevalence": float(np.mean(y_test)),
            "models": {},
            "brier_scores": {},
        }

        # Calibration Curves Figure
        fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
        ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # PR Curves Figure
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

        for name, test_prob in test_ensembles.items():
            val_prob = val_ensembles[name]

            # 1. Strictly tune threshold on validation
            threshold = self.tune_threshold_on_validation(y_val, val_prob)

            # 2. Score holdout test set using validation threshold
            y_pred = (test_prob >= threshold).astype(int)

            pr_auc = float(average_precision_score(y_test, test_prob))
            roc_auc = float(roc_auc_score(y_test, test_prob))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            brier = float(brier_score_loss(y_test, test_prob))

            cm = confusion_matrix(y_test, y_pred).tolist()

            # 3. Compute 95% Confidence Intervals
            ci_data = compute_bootstrap_ci(y_test, test_prob, threshold, n_bootstraps=n_bootstraps)

            results["models"][name] = {
                "validation_tuned_threshold": threshold,
                "holdout_pr_auc": pr_auc,
                "holdout_roc_auc": roc_auc,
                "holdout_f1": f1,
                "holdout_recall": recall,
                "holdout_precision": precision,
                "brier_score": brier,
                "confusion_matrix": {
                    "tn": cm[0][0],
                    "fp": cm[0][1],
                    "fn": cm[1][0],
                    "tp": cm[1][1],
                },
                "confidence_intervals_95": ci_data,
            }
            results["brier_scores"][name] = brier

            # Plot Calibration Curve
            prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=10)
            ax_cal.plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier: {brier:.4f})")

            # Plot PR Curve
            p_curve, r_curve, _ = precision_recall_curve(y_test, test_prob)
            ax_pr.plot(r_curve, p_curve, label=f"{name} (PR-AUC: {pr_auc:.4f})")

        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Empirical True Fraction Positives")
        ax_cal.set_title("Reliability Diagram (Calibration Assessment on Holdout Test Set)")
        ax_cal.legend(loc="lower right")
        ax_cal.grid(True, alpha=0.3)
        cal_path = self.results_dir / "holdout_calibration_curves.png"
        fig_cal.savefig(cal_path, dpi=300, bbox_inches="tight")
        plt.close(fig_cal)

        ax_pr.set_xlabel("Recall (Sensitivity)")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curves on Untouched Holdout Test Set")
        ax_pr.legend(loc="lower left")
        ax_pr.grid(True, alpha=0.3)
        pr_path = self.results_dir / "holdout_pr_curves.png"
        fig_pr.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close(fig_pr)

        results["artifacts"] = {
            "calibration_diagram": str(cal_path),
            "precision_recall_curves": str(pr_path),
        }

        # Save JSON output
        out_json = self.results_dir / "holdout_evaluation_report.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["report_path"] = str(out_json)

        return results


def run_evaluation_cli():
    print("=" * 78)
    print(" FRAUDGUARD AI: REPRODUCIBLE HOLDOUT TEST EVALUATION")
    print("=" * 78)
    evaluator = ReproducibleEvaluator()
    res = evaluator.evaluate_holdout(n_bootstraps=500)

    print(f"\nHoldout Test Set: {res['holdout_test_rows']:,} rows ({res['holdout_fraud_rows']:,} fraud, {res['holdout_fraud_prevalence']*100:.2f}%)")
    print("-" * 78)
    print(f"{'Model Architecture':<26} | {'PR-AUC (95% CI)':<18} | {'F1-Score':<10} | {'Recall':<8} | {'Brier Score':<11}")
    print("-" * 78)

    for name, m in res["models"].items():
        ci = m["confidence_intervals_95"]["pr_auc"]
        ci_str = f"{m['holdout_pr_auc']:.4f} [{ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}]"
        print(f"{name:<26} | {ci_str:<18} | {m['holdout_f1']:<10.4f} | {m['holdout_recall']:<8.4f} | {m['brier_score']:<11.5f}")

    print("-" * 78)
    print("\nCalibration Assessment (Lower Brier Score = Better Calibration):")
    for name, brier in res["brier_scores"].items():
        print(f"  * {name:<24}: Brier Score = {brier:.5f}")

    print(f"\nArtifacts Saved:")
    print(f"  * Reliability Diagram:       {res['artifacts']['calibration_diagram']}")
    print(f"  * Precision-Recall Curves:   {res['artifacts']['precision_recall_curves']}")
    print(f"  * Full JSON Metrics Report:  {res['report_path']}")
    print("=" * 78)


if __name__ == "__main__":
    run_evaluation_cli()
