"""
Module 6: Explainable AI (XAI) with SHAP Module
Provides model interpretability and transparency using SHapley Additive exPlanations (SHAP)
for the best-performing tree-based fraud classifier.

Generates:
1. SHAP Global Feature Importance Bar Plot
2. SHAP Summary (Beeswarm) Plot

Explains the best validation-ranked tree model; neural models are evaluated separately.
"""

import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ExplainabilityModule:
    """
    Computes game-theoretic SHAP feature attributions and generates interpretability visualizations.
    """

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.explainer = None
        self.shap_values = None
        self.best_model_name = ""

    def select_best_model(self, evaluator, trainer) -> tuple:
        """
        Selects the highest-validation-average-precision tree model.
        """
        best_score = -1.0
        best_name = None
        
        for data in trainer.validation_metrics:
            name = data['Model']
            if name not in ['Random Forest', 'XGBoost']:
                continue
            score = data['PR-AUC']
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            best_name = next(name for name in ['Random Forest', 'XGBoost'] if name in trainer.models)

        self.best_model_name = best_name
        best_model = trainer.models[best_name]
        return best_name, best_model

    def compute_shap_explanations(self, model, X_sample: pd.DataFrame, max_samples: int = 500):
        """
        Calculates SHAP values for the given tree model using TreeExplainer.
        """
        if len(X_sample) > max_samples:
            X_eval = X_sample.sample(n=max_samples, random_state=42)
        else:
            X_eval = X_sample

        print(f"[Explainability] Computing SHAP values using TreeExplainer for '{self.best_model_name}' on {len(X_eval)} samples...")
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(model)
        raw_shap = self.explainer.shap_values(X_eval)

        # Handle formatting differences across Scikit-learn Random Forest and XGBoost
        if isinstance(raw_shap, list):
            # RandomForest in older SHAP versions returns [array_class0, array_class1]
            self.shap_values = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
            # Shape: (n_samples, n_features, n_classes) -> extract class 1 (Fraud)
            self.shap_values = raw_shap[:, :, 1]
        else:
            self.shap_values = raw_shap

        self.X_eval = X_eval
        print("[Explainability] SHAP values successfully calculated.")
        return self.shap_values

    def plot_feature_importance(self, save_filename: str = 'shap_feature_importance.png'):
        """
        Generates and saves standard SHAP global feature importance bar plot.
        """
        save_path = os.path.join(self.results_dir, save_filename)
        
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(self.shap_values, self.X_eval, plot_type="bar", show=False)
        plt.title(f"SHAP Global Feature Importance ({self.best_model_name})\n[Mean Absolute SHAP Attribution]",
                  fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Explainability] Saved SHAP Feature Importance: '{save_path}'")

    def plot_summary_beeswarm(self, save_filename: str = 'shap_summary_plot.png'):
        """
        Generates and saves SHAP summary (beeswarm) plot showing feature value directionality.
        """
        save_path = os.path.join(self.results_dir, save_filename)
        
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(self.shap_values, self.X_eval, show=False)
        plt.title(f"SHAP Summary Beeswarm Plot ({self.best_model_name})\n[Feature Value Impact on Model Decision]",
                  fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Explainability] Saved SHAP Summary Beeswarm Plot: '{save_path}'")

    def explain_pipeline(self, evaluator, trainer, X_test: pd.DataFrame):
        """
        Executes complete XAI pipeline: model selection, attribution computation, and visualization export.
        """
        print("\n" + "-"*65)
        print(" MODULE 6: EXPLAINABLE AI (SHAP INTERPRETABILITY)")
        print("-"*65)

        best_name, best_model = self.select_best_model(evaluator, trainer)
        print(f"[Explainability] Best performing model identified for XAI: '{best_name}'")

        self.compute_shap_explanations(best_model, X_test)
        self.plot_feature_importance()
        self.plot_summary_beeswarm()

        print("-"*65 + "\n")


if __name__ == '__main__':
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    from imbalance_handler import ImbalanceHandler
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator

    loader = DataLoader('data/transactions.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(loader.df)
    
    handler = ImbalanceHandler()
    X_train_res, y_train_res = handler.apply_smote(X_train, y_train)

    trainer = ModelTrainer()
    trainer.train_all(X_train_res, y_train_res)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_and_generate_all(trainer, X_test, y_test)

    xai = ExplainabilityModule()
    xai.explain_pipeline(evaluator, trainer, X_test)
