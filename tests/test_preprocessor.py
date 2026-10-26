"""
Unit Tests for Module 2: DataPreprocessor
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
import tempfile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, BASE_DIR)

from preprocessor import DataPreprocessor
from data_loader import DataLoader
import config

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.loader = DataLoader(config.DATASET_PATH)
        self.preprocessor = DataPreprocessor(
            target_col='isFraud',
            test_size=0.2,
            random_state=config.RANDOM_STATE
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_feature_engineering_calculations(self):
        """Verifies derived financial discrepancy features are correctly computed."""
        sample_df = pd.DataFrame([{
            'step': 100,
            'type': 'TRANSFER',
            'amount': 5000.0,
            'oldbalanceOrg': 10000.0,
            'newbalanceOrig': 5000.0,
            'oldbalanceDest': 2000.0,
            'newbalanceDest': 7000.0,
            'isFlaggedFraud': 0,
            'isFraud': 0
        }])
        feat_df = self.preprocessor.engineer_features(sample_df)
        
        self.assertIn('errorBalanceOrig', feat_df.columns)
        self.assertIn('errorBalanceDest', feat_df.columns)
        self.assertIn('amount_to_oldbalance_ratio', feat_df.columns)
        
        # In balanced case: errorBalanceOrig = 5000 + 5000 - 10000 = 0.0
        self.assertEqual(feat_df['errorBalanceOrig'].iloc[0], 0.0)

    def test_fit_transform_split_shapes(self):
        """Verifies train and test partitions adhere to 80:20 split and scaling."""
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform_split(self.loader.df, models_dir=self.temp_dir.name)
        total_samples = len(self.loader.df)
        
        expected_test_len = int(total_samples * 0.2)
        self.assertEqual(len(X_test), expected_test_len)
        self.assertEqual(len(X_train), total_samples - expected_test_len)
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertTrue(self.preprocessor.is_fitted)

    def test_transform_new_data_alignment(self):
        """Verifies transform_new_data aligns column dimensions with training features."""
        self.preprocessor.fit_transform_split(self.loader.df, models_dir=self.temp_dir.name)
        new_tx = pd.DataFrame([{
            'step': 120,
            'type': 'CASH_OUT',
            'amount': 1500.0,
            'oldbalanceOrg': 3000.0,
            'newbalanceOrig': 1500.0,
            'oldbalanceDest': 500.0,
            'newbalanceDest': 2000.0
        }])
        X_new = self.preprocessor.transform_new_data(new_tx)
        self.assertEqual(X_new.shape[1], len(self.preprocessor.feature_names))

if __name__ == '__main__':
    unittest.main()
