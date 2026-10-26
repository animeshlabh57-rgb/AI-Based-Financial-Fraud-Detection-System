"""
Module 3: Class Imbalance & SMOTE Oversampling Module
Analyzes class imbalance ratios, applies Synthetic Minority Over-sampling Technique (SMOTE)
exclusively to the training dataset (to strictly prevent data leakage), and generates
before-and-after class distribution visualization plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler

class ImbalanceHandler:
    """
    Handles class imbalance analysis and SMOTE resampling strictly on the training partition.
    """

    def __init__(self, random_state: int = 42, sampling_strategy: float = 1.0):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        self.before_counts = {}
        self.after_counts = {}

    def analyze_imbalance(self, y_train: pd.Series, y_test: pd.Series = None) -> dict:
        """
        Computes exact counts and percentages of minority (Fraud) vs majority (Legitimate) classes.
        """
        train_counts = y_train.value_counts().to_dict()
        total_train = len(y_train)
        
        info = {
            'train_legit': train_counts.get(0, 0),
            'train_fraud': train_counts.get(1, 0),
            'train_legit_pct': round((train_counts.get(0, 0) / total_train) * 100, 2),
            'train_fraud_pct': round((train_counts.get(1, 0) / total_train) * 100, 2),
            'train_imbalance_ratio': f"{round(train_counts.get(0, 0) / max(1, train_counts.get(1, 0)), 1)}:1"
        }
        
        if y_test is not None:
            test_counts = y_test.value_counts().to_dict()
            total_test = len(y_test)
            info['test_legit'] = test_counts.get(0, 0)
            info['test_fraud'] = test_counts.get(1, 0)
            info['test_fraud_pct'] = round((test_counts.get(1, 0) / total_test) * 100, 2)
            
        return info

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series, results_dir: str = 'results'):
        """
        Applies SMOTE ONLY on (X_train, y_train).
        Returns resampled (X_train_resampled, y_train_resampled).
        """
        os.makedirs(results_dir, exist_ok=True)
        print("\n" + "-"*65)
        print(" MODULE 3: FEATURE PREPARATION & CLASS IMBALANCE (SMOTE)")
        print("-"*65)

        # Pre-SMOTE Stats
        self.before_counts = y_train.value_counts().to_dict()
        print(f"[ImbalanceHandler] Before SMOTE (Train Set):")
        print(f"                   Legitimate (Class 0): {self.before_counts.get(0, 0):,}")
        print(f"                   Fraudulent (Class 1): {self.before_counts.get(1, 0):,} ({y_train.mean()*100:.2f}%)")
        print(f"                   Imbalance Ratio:      {self.before_counts.get(0, 0) / max(1, self.before_counts.get(1, 0)):.1f}:1")

        # Apply SMOTE to Training Set ONLY
        counts = y_train.value_counts()
        if len(counts) != 2:
            raise ValueError('Training requires both legitimate and fraudulent examples.')
        dummy_cols = [c for c in X_train.columns if c.startswith('type_')]
        if int(counts.min()) < 2:
            sampler = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            X_resampled_arr, y_resampled_arr = sampler.fit_resample(X_train, y_train)
        elif dummy_cols:
            # SMOTE-NC treats the transaction type as ONE categorical variable.
            # Decode/re-encode its scaled dummies to avoid fractional categories.
            minima, maxima = X_train[dummy_cols].min(), X_train[dummy_cols].max()
            active = (X_train[dummy_cols] - minima) / (maxima - minima).replace(0, 1)
            numeric_cols = [c for c in X_train.columns if c not in dummy_cols]
            compact = X_train[numeric_cols].copy()
            compact['__transaction_type'] = np.argmax(active.to_numpy(), axis=1)
            sampler = SMOTENC(categorical_features=[len(numeric_cols)],
                               sampling_strategy=self.sampling_strategy, random_state=self.random_state,
                               k_neighbors=min(5, int(counts.min()) - 1))
            compact_resampled, y_resampled_arr = sampler.fit_resample(compact, y_train)
            X_resampled_arr = compact_resampled[numeric_cols].copy()
            for index, col in enumerate(dummy_cols):
                X_resampled_arr[col] = np.where(compact_resampled['__transaction_type'] == index,
                                                maxima[col], minima[col])
            X_resampled_arr = X_resampled_arr[X_train.columns]
        else:
            sampler = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state,
                            k_neighbors=min(5, int(counts.min()) - 1))
            X_resampled_arr, y_resampled_arr = sampler.fit_resample(X_train, y_train)

        # Convert back to DataFrame / Series
        X_train_resampled = pd.DataFrame(X_resampled_arr, columns=X_train.columns)
        y_train_resampled = pd.Series(y_resampled_arr, name=y_train.name)

        # Post-SMOTE Stats
        self.after_counts = y_train_resampled.value_counts().to_dict()
        print(f"[ImbalanceHandler] After SMOTE (Train Set ONLY - Test untouched):")
        print(f"                   Legitimate (Class 0): {self.after_counts.get(0, 0):,}")
        print(f"                   Fraudulent (Class 1): {self.after_counts.get(1, 0):,} ({y_train_resampled.mean()*100:.2f}%)")
        print(f"                   Balanced Ratio:       1:1")

        # Generate & Save Visualization Plot
        plot_path = os.path.join(results_dir, 'class_distribution_before_after_smote.png')
        self.generate_distribution_plot(plot_path)
        print(f"[ImbalanceHandler] Saved SMOTE comparison plot: '{plot_path}'")
        print("-"*65 + "\n")

        return X_train_resampled, y_train_resampled

    def generate_distribution_plot(self, save_path: str):
        """
        Generates publication-quality side-by-side bar chart showing class distributions.
        Provides sufficient y-limit headroom and padding to avoid overlapping titles and labels.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
        
        # Color palette
        colors_before = ['#1d4ed8', '#dc2626']
        colors_after = ['#1d4ed8', '#16a34a']
        
        # 1. Before SMOTE
        categories = ['Legitimate (0)', 'Fraudulent (1)']
        counts_before = [self.before_counts.get(0, 0), self.before_counts.get(1, 0)]
        bars1 = axes[0].bar(categories, counts_before, color=colors_before, width=0.45, edgecolor='black', linewidth=1.2)
        axes[0].set_title("Training Class Distribution: Before SMOTE\n(Severe Imbalance: ~1.5% Fraud)", fontsize=12, fontweight='bold', pad=16)
        axes[0].set_ylabel("Number of Transactions", fontsize=11, fontweight='semibold')
        axes[0].grid(axis='y', linestyle='--', alpha=0.4)
        axes[0].set_ylim(0, max(counts_before) * 1.25)
        
        # Annotate counts on bars
        for bar in bars1:
            height = bar.get_height()
            pct = (height / sum(counts_before) * 100) if sum(counts_before) else 0
            axes[0].annotate(f'{height:,}\n({pct:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 6), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. After SMOTE
        counts_after = [self.after_counts.get(0, 0), self.after_counts.get(1, 0)]
        bars2 = axes[1].bar(categories, counts_after, color=colors_after, width=0.45, edgecolor='black', linewidth=1.2)
        axes[1].set_title("Training Class Distribution: After SMOTE\n(Synthetic Oversampling: 50% Fraud)", fontsize=12, fontweight='bold', pad=16)
        axes[1].set_ylabel("Number of Transactions", fontsize=11, fontweight='semibold')
        axes[1].grid(axis='y', linestyle='--', alpha=0.4)
        axes[1].set_ylim(0, max(counts_after) * 1.25)
        
        for bar in bars2:
            height = bar.get_height()
            pct = (height / sum(counts_after) * 100) if sum(counts_after) else 0
            axes[1].annotate(f'{height:,}\n({pct:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 6), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle("AI Fraud Detection: Class Imbalance Mitigation (SMOTE Applied to Train Set Only)", fontsize=14, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.82, wspace=0.25)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    from data_loader import DataLoader
    from preprocessor import DataPreprocessor
    loader = DataLoader('data/transactions.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(loader.df)
    handler = ImbalanceHandler()
    X_train_res, y_train_res = handler.apply_smote(X_train, y_train)
    print("Resampled X_train shape:", X_train_res.shape)
