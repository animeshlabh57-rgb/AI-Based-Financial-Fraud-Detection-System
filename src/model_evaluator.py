"""
Module 5: Model Evaluation & Performance Analytics Module
Computes rigorous, empirical evaluation metrics on the untouched test set:
- Precision, Recall, F1-Score, ROC-AUC, Specificity, PR-AUC, Confusion Matrix
Generates publication-quality charts and CSV comparisons:
1. Side-by-side Confusion Matrix Heatmaps
2. Comparative ROC Curves
3. Comparative Precision-Recall (PR) Curves
4. Metrics Comparison Summary CSV
5. Multi-metric Bar Comparison Chart
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)

class ModelEvaluator:
    """
    Evaluates trained models on the test set and exports visual and tabular reports.
    """

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_records = []
        self.eval_data = {}

    def evaluate_model(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        """
        Calculates all key fraud evaluation metrics from actual model predictions.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        accuracy = accuracy_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            'Model': model_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall (Sensitivity)': round(recall, 4),
            'Specificity': round(specificity, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'True Negatives': int(tn),
            'False Negatives': int(fn)
        }

        self.metrics_records.append(metrics)
        self.eval_data[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cm': cm,
            'metrics': metrics
        }
        return metrics

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Creates and saves model_comparison.csv from actual metrics.
        """
        df_metrics = pd.DataFrame(self.metrics_records)
        csv_path = os.path.join(self.results_dir, 'model_comparison.csv')
        df_metrics.to_csv(csv_path, index=False)
        print(f"[ModelEvaluator] Saved metrics comparison CSV: '{csv_path}'")
        return df_metrics

    MODEL_COLORS = {
        'Random Forest': '#2563eb',        # Primary Blue
        'XGBoost': '#d97706',              # Amber / Warm Orange
        'Deep Neural Network': '#7c3aed',  # Purple
        'Deep Autoencoder': '#059669',     # Emerald Green
    }

    def plot_confusion_matrices(self):
        """
        Plots side-by-side heatmaps of confusion matrices for all evaluated models.
        Clarifies that percentage annotations are computed relative to all test transactions.
        """
        n_models = len(self.eval_data)
        columns = min(2, n_models)
        rows = (n_models + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 5.2 * rows), dpi=200, squeeze=False)
        axes = axes.ravel()

        for ax, (model_name, data) in zip(axes, self.eval_data.items()):
            cm = data['cm']
            total_n = int(cm.sum())
            cm_pct = cm.astype('float') / total_n * 100
            labels = np.asarray([
                [f"TN: {cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"FP: {cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
                [f"FN: {cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"TP: {cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
            ])
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
                        annot_kws={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title(f"{model_name}\nDecision Threshold: {data['metrics'].get('Threshold', 0.5):.4f}",
                         fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel("Predicted Label (0: Legit, 1: Fraud)", fontsize=10, fontweight='semibold')
            ax.set_ylabel("Actual Label (0: Legit, 1: Fraud)", fontsize=10, fontweight='semibold')
            ax.set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
            ax.set_yticklabels(['Legitimate (0)', 'Fraud (1)'])

        for unused in axes[n_models:]:
            unused.set_visible(False)

        plt.suptitle("Model Evaluation: Confusion Matrices on Test Set\n(Percentages represent fraction of total test transactions: N=3,000)",
                     fontsize=13, fontweight='bold', y=0.99)
        plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
        save_path = os.path.join(self.results_dir, 'confusion_matrices.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[ModelEvaluator] Saved Confusion Matrices plot: '{save_path}'")

    def plot_roc_curves(self):
        """
        Plots comparative Receiver Operating Characteristic (ROC) Curves with consistent model colors.
        """
        plt.figure(figsize=(8.5, 6), dpi=300)

        for model_name, data in self.eval_data.items():
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
            auc_score = data['metrics']['ROC-AUC']
            color = self.MODEL_COLORS.get(model_name, '#2563eb')
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})",
                     linewidth=2.3, color=color)

        plt.plot([0, 1], [0, 1], 'k--', label='Chance Line (AUC = 0.5000)', alpha=0.5, linewidth=1.5)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='semibold')
        plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight='semibold')
        plt.title("Comparative ROC Curves (Fraud Detection)", fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        save_path = os.path.join(self.results_dir, 'roc_curves.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[ModelEvaluator] Saved ROC Curves plot:           '{save_path}'")

    def plot_precision_recall_curves(self):
        """
        Plots comparative Precision-Recall Curves with consistent model colors.
        """
        plt.figure(figsize=(8.5, 6), dpi=300)

        for model_name, data in self.eval_data.items():
            precision_vals, recall_vals, _ = precision_recall_curve(data['y_true'], data['y_prob'])
            pr_auc = data['metrics']['PR-AUC']
            color = self.MODEL_COLORS.get(model_name, '#2563eb')
            plt.plot(recall_vals, precision_vals, label=f"{model_name} (AP / PR-AUC = {pr_auc:.4f})",
                     linewidth=2.3, color=color)

        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.xlabel("Recall (Fraud Catch Rate)", fontsize=11, fontweight='semibold')
        plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='semibold')
        plt.title("Comparative Precision-Recall Curves (Average Precision)", fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc="lower left", frameon=True, shadow=True, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.4)

        save_path = os.path.join(self.results_dir, 'precision_recall_curves.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[ModelEvaluator] Saved PR Curves plot:            '{save_path}'")

    def plot_comparison_chart(self):
        """
        Generates a grouped bar chart comparing Precision, Recall, F1, and ROC-AUC using consistent model colors.
        """
        df_metrics = pd.DataFrame(self.metrics_records)
        metrics_to_plot = ['Precision', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC']
        
        df_plot = df_metrics.melt(id_vars='Model', value_vars=metrics_to_plot,
                                 var_name='Metric', value_name='Score')

        plt.figure(figsize=(10.5, 5.8), dpi=300)
        ax = sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model',
                         palette=self.MODEL_COLORS, edgecolor='black', linewidth=0.8)
        plt.title("Model Performance Benchmark (Test Set Evaluation)", fontsize=13, fontweight='bold', pad=12)
        plt.ylim(0, 1.18)
        plt.ylabel("Evaluation Score", fontsize=11, fontweight='semibold')
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Annotate numbers on bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            xytext=(0, 4), textcoords='offset points')

        plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'model_comparison_chart.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"[ModelEvaluator] Saved Model Comparison Chart:    '{save_path}'")

    def evaluate_and_generate_all(self, trainer, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Orchestrates evaluation across all models and generates all visualization artifacts.
        """
        print("\n" + "-"*65)
        print(" MODULE 5: MODEL PERFORMANCE EVALUATION")
        print("-"*65)

        for model_name in trainer.models.keys():
            y_pred, y_prob = trainer.predict_and_score(model_name, X_test)
            m = self.evaluate_model(model_name, y_test, y_pred, y_prob)
            m['Family'] = config.MODEL_FAMILIES[model_name]
            m['Threshold'] = trainer.thresholds.get(model_name, 0.5)
            m['Training Seconds'] = round(trainer.training_times[model_name], 3)
            print(f"[{model_name}] Precision: {m['Precision']:.4f} | Recall: {m['Recall (Sensitivity)']:.4f} | F1: {m['F1-Score']:.4f} | ROC-AUC: {m['ROC-AUC']:.4f}")

        df_comparison = self.generate_comparison_table()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_comparison_chart()

        print("\n" + "="*65)
        print(" EMPIRICAL MODEL COMPARISON SUMMARY TABLE")
        print("="*65)
        print(df_comparison[['Model', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC', 'PR-AUC']].to_string(index=False))
        print("="*65 + "\n")
        return df_comparison

if __name__ == '__main__':
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from imbalance_handler import ImbalanceHandler
    from model_trainer import ModelTrainer

    loader = DataLoader('data/transactions.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(loader.df)
    
    handler = ImbalanceHandler()
    X_train_res, y_train_res = handler.apply_smote(X_train, y_train)

    trainer = ModelTrainer()
    trainer.train_all(X_train_res, y_train_res)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_and_generate_all(trainer, X_test, y_test)
