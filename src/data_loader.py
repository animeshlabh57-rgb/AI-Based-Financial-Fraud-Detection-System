"""
Module 1: Data Input & Ingestion Module
Handles loading financial transaction CSV datasets, exploratory inspection,
missing value auditing, duplicate record detection, and class distribution profiling.
"""

import os
import pandas as pd
import numpy as np

class DataLoader:
    """
    DataLoader encapsulates methods for reading financial transaction datasets
    and performing comprehensive data health checks.
    """
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.df = None
        if filepath:
            self.load_data(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads CSV dataset from disk and checks validity.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at path: {filepath}")
        
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        print(f"[DataLoader] Successfully loaded dataset: '{filepath}' ({len(self.df)} rows, {self.df.shape[1]} columns)")
        return self.df

    def get_dataset_summary(self, target_col: str = 'isFraud') -> dict:
        """
        Gathers comprehensive statistics: shape, preview, data types,
        missing value audit, duplicate counts, and class balance.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df
        total_records = len(df)
        total_cols = df.shape[1]
        
        # Missing values check
        missing_series = df.isnull().sum()
        missing_pct = (missing_series / total_records) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_series,
            'Missing_Percentage': missing_pct.round(2)
        })
        
        # Duplicate records check
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_count / total_records) * 100, 2)
        
        # Data types
        dtypes_dict = df.dtypes.astype(str).to_dict()
        
        # Class distribution check
        if target_col in df.columns:
            class_counts = df[target_col].value_counts().to_dict()
            class_pcts = df[target_col].value_counts(normalize=True).mul(100).round(3).to_dict()
            legit_count = class_counts.get(0, 0)
            fraud_count = class_counts.get(1, 0)
            legit_pct = class_pcts.get(0, 0.0)
            fraud_pct = class_pcts.get(1, 0.0)
            imbalance_ratio = round(legit_count / max(1, fraud_count), 2)
        else:
            class_counts = {}
            class_pcts = {}
            legit_count = fraud_count = 0
            legit_pct = fraud_pct = imbalance_ratio = 0.0

        summary = {
            'shape': (total_records, total_cols),
            'total_rows': total_records,
            'total_columns': total_cols,
            'column_names': list(df.columns),
            'dtypes': dtypes_dict,
            'missing_values': missing_df,
            'has_missing': bool(missing_series.sum() > 0),
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_pct,
            'target_col': target_col,
            'class_distribution': {
                'legitimate_count': legit_count,
                'fraud_count': fraud_count,
                'legitimate_pct': legit_pct,
                'fraud_pct': fraud_pct,
                'imbalance_ratio': f"{imbalance_ratio}:1"
            },
            'preview_head': df.head(5)
        }
        return summary

    def display_summary_report(self, target_col: str = 'isFraud'):
        """
        Prints a structured academic summary report to console.
        """
        summary = self.get_dataset_summary(target_col)
        print("\n" + "="*65)
        print(" MODULE 1: FINANCIAL TRANSACTION DATASET SUMMARY")
        print("="*65)
        print(f" Dataset Path:          {self.filepath}")
        print(f" Total Transactions:    {summary['total_rows']:,}")
        print(f" Total Features:        {summary['total_columns']}")
        print(f" Duplicate Records:     {summary['duplicate_count']} ({summary['duplicate_percentage']}%)")
        print(f" Total Missing Values:  {summary['missing_values']['Missing_Count'].sum()}")
        print("-" * 65)
        print(" Feature Data Types:")
        for col, dtype in summary['dtypes'].items():
            print(f"   • {col:<18}: {dtype}")
        print("-" * 65)
        print(" Class Distribution (Ground Truth):")
        dist = summary['class_distribution']
        print(f"   • Legitimate (Class 0): {dist['legitimate_count']:,} ({dist['legitimate_pct']}%)")
        print(f"   • Fraudulent (Class 1): {dist['fraud_count']:,} ({dist['fraud_pct']}%)")
        print(f"   • Imbalance Ratio:      {dist['imbalance_ratio']}")
        print("="*65)
        print(" Sample Data Preview (First 5 records):")
        print(summary['preview_head'])
        print("="*65 + "\n")

if __name__ == '__main__':
    # Direct test execution
    loader = DataLoader('data/transactions.csv')
    loader.display_summary_report()
