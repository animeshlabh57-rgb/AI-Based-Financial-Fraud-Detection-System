"""
Unit Tests for Module 1: DataLoader
"""

import unittest
import os
import sys
import pandas as pd

# Add src and base to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, BASE_DIR)

from data_loader import DataLoader
import config

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        self.dataset_path = config.DATASET_PATH
        self.loader = DataLoader(self.dataset_path)

    def test_file_exists_and_loads(self):
        """Verifies dataset exists and loads into a non-empty DataFrame."""
        self.assertTrue(os.path.exists(self.dataset_path), "Dataset file should exist on disk.")
        self.assertIsInstance(self.loader.df, pd.DataFrame)
        self.assertGreater(len(self.loader.df), 1000, "Dataset should have more than 1000 records.")

    def test_dataset_summary_structure(self):
        """Verifies get_dataset_summary returns all required metadata keys."""
        summary = self.loader.get_dataset_summary(target_col='isFraud')
        required_keys = [
            'shape', 'total_rows', 'total_columns', 'column_names',
            'dtypes', 'missing_values', 'has_missing', 'duplicate_count',
            'duplicate_percentage', 'class_distribution', 'preview_head'
        ]
        for key in required_keys:
            self.assertIn(key, summary, f"Summary dictionary must contain '{key}' key.")

    def test_class_distribution_integrity(self):
        """Verifies that fraud and legitimate class counts sum to total rows."""
        summary = self.loader.get_dataset_summary(target_col='isFraud')
        dist = summary['class_distribution']
        total_counted = dist['legitimate_count'] + dist['fraud_count']
        self.assertEqual(total_counted, summary['total_rows'])
        self.assertGreater(dist['fraud_count'], 0, "There should be at least one fraud record in dataset.")

    def test_invalid_file_handling(self):
        """Verifies FileNotFoundError is raised for non-existent dataset paths."""
        with self.assertRaises(FileNotFoundError):
            DataLoader("non_existent_dataset.csv")

if __name__ == '__main__':
    unittest.main()
