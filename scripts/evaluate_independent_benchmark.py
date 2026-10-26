"""
Evaluate FraudGuard AI models against the independent PaySim mobile money benchmark.
This script demonstrates honest, transparent evaluation across independent data partitions,
highlighting distributional shift and real-world performance limitations.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

import config
from src.model_comparison import FraudModelComparison


def run_independent_benchmark(benchmark_csv: str = 'data/paysim_sample_1000.csv'):
    csv_path = Path(benchmark_csv)
    if not csv_path.exists():
        print(f"[Error] Benchmark file '{csv_path}' not found.")
        return

    print("=" * 80)
    print("FRAUDGUARD AI: INDEPENDENT DATASET BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Dataset:              {csv_path.name}")
    print(f"Path:                 {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    total_fraud = int(df['isFraud'].sum()) if 'isFraud' in df.columns else 0
    prevalence = (total_fraud / total_rows * 100.0) if total_rows else 0.0

    print(f"Total Transactions:   {total_rows:,}")
    print(f"Fraudulent Rows:      {total_fraud:,} ({prevalence:.2f}%)")
    print(f"Legitimate Rows:      {total_rows - total_fraud:,} ({100.0 - prevalence:.2f}%)")
    print("-" * 80)

    engine = FraudModelComparison(models_dir=config.MODELS_DIR)
    scored = engine.predict_batch(df)

    y_true = df['isFraud'].astype(int).to_numpy()
    results = []

    for model_name in engine.model_names:
        predictor = engine._predictors[model_name]
        y_pred = (scored[f'{model_name} Prediction'] == 'FRAUDULENT').astype(int).to_numpy()
        y_score = scored[f'{model_name} Score'].astype(float).to_numpy()
        threshold = predictor.threshold

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
        try:
            roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            roc = 0.5

        rec_data = {
            'Model': model_name,
            'Family': config.MODEL_FAMILIES.get(model_name, 'Unknown'),
            'Threshold': float(threshold),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'Average Precision (AP)': float(ap),
            'ROC-AUC': float(roc),
            'Accuracy': float(acc),
            'True Positives (TP)': int(tp),
            'False Positives (FP)': int(fp),
            'True Negatives (TN)': int(tn),
            'False Negatives (FN)': int(fn),
        }
        results.append(rec_data)

    results_df = pd.DataFrame(results)

    print("\nPERFORMANCE RESULTS (INDEPENDENT DATASET):")
    print("-" * 80)
    for r in results:
        print(f"[*] {r['Model']:<22} | Precision: {r['Precision']*100:5.2f}% | Recall: {r['Recall']*100:5.2f}% | F1: {r['F1-Score']:.4f} | AP: {r['Average Precision (AP)']:.4f} | ROC-AUC: {r['ROC-AUC']:.4f}")

    print("-" * 80)
    # Save results to output
    out_csv = Path(config.RESULTS_DIR) / 'independent_benchmark_evaluation.csv'
    out_json = Path(config.RESULTS_DIR) / 'independent_benchmark_report.json'

    results_df.to_csv(out_csv, index=False)

    report_payload = {
        'benchmark_file': csv_path.name,
        'evaluated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'dataset_summary': {
            'total_rows': total_rows,
            'fraud_rows': total_fraud,
            'fraud_prevalence_pct': round(prevalence, 2),
        },
        'methodological_notes': (
            "Models were trained exclusively on the 15,000 local synthetic benchmark dataset. "
            "Evaluation on this independent dataset exhibits distributional shift, which is expected. "
            "Reported synthetic metrics reflect rule recovery of the local generator and do not guarantee "
            "equivalent real-world performance on bank or production payment streams."
        ),
        'metrics': results,
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2)

    print(f"[Saved] CSV report:  '{out_csv}'")
    print(f"[Saved] JSON report: '{out_json}'")
    print("=" * 80)
    return results_df


if __name__ == '__main__':
    bench = sys.argv[1] if len(sys.argv) > 1 else 'data/paysim_sample_1000.csv'
    run_independent_benchmark(bench)
