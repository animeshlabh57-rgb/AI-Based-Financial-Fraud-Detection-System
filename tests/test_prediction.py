"""
Unit Tests for Module 7: FraudPredictor
"""

import unittest
import os
import sys
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, BASE_DIR)

from predictor import FraudPredictor
import config

class TestFraudPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = FraudPredictor('XGBoost')

    def test_single_prediction_output_structure(self):
        """Verifies single transaction prediction outputs a valid alert object."""
        sample_txn = {
            'step': 150,
            'type': 'PAYMENT',
            'amount': 45.0,
            'oldbalanceOrg': 500.0,
            'newbalanceOrig': 455.0,
            'oldbalanceDest': 1200.0,
            'newbalanceDest': 1245.0
        }
        alert = self.predictor.predict_single(sample_txn)
        
        expected_keys = [
            'Transaction_ID', 'Model_Used', 'Prediction', 'Fraud_Probability',
            'Fraud_Probability_Pct', 'Risk_Level', 'Risk_Badge', 'Recommended_Action',
            'Top_Risk_Factors', 'Raw_Inputs'
        ]
        for key in expected_keys:
            self.assertIn(key, alert, f"Alert object must contain key '{key}'")
            
        self.assertIn(alert['Prediction'], ['LEGITIMATE', 'FRAUDULENT'])
        self.assertIn(alert['Risk_Level'], ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        self.assertGreaterEqual(alert['Fraud_Probability'], 0.0)
        self.assertLessEqual(alert['Fraud_Probability'], 1.0)

    def test_batch_prediction(self):
        """Verifies batch inference appends all risk and prediction columns."""
        sample_df = pd.read_csv(config.SAMPLE_TEST_PATH).head(5)
        res_df = self.predictor.predict_batch(sample_df)

        self.assertEqual(len(res_df), 5)
        self.assertIn('Prediction', res_df.columns)
        self.assertIn('Fraud_Probability', res_df.columns)
        self.assertIn('Risk_Level', res_df.columns)
        self.assertIn('Recommended_Action', res_df.columns)

    def test_missing_column_schema_validation(self):
        """Verifies that missing columns raise a clear descriptive ValueError."""
        invalid_tx = pd.DataFrame([{
            'amount': 500.0,
            'oldbalanceOrg': 1000.0
            # Missing 'step', 'type', 'newbalanceOrig', etc.
        }])
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(invalid_tx)

if __name__ == '__main__':
    unittest.main()
