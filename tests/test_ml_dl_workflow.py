"""Regression checks for leakage prevention, neural learning, and saved inference."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))
import config
from preprocessor import DataPreprocessor
from predictor import FraudPredictor
from imbalance_handler import ImbalanceHandler
from src.neural_models import DeepNeuralNetworkClassifier, AutoencoderAnomalyDetector


class TestPreprocessingRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = pd.read_csv(config.DATASET_PATH)
        self.prep = DataPreprocessor()
        self.parts = self.prep.fit_transform_partitions(self.data, self.temp.name)

    def test_partitions_and_training_only_statistics(self):
        train, val, test = (set(self.prep.partition_indices_[p]) for p in ['train', 'validation', 'test'])
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual((len(train), len(val), len(test)), (9000, 3000, 3000))
        self.assertAlmostEqual(self.prep.numeric_fill_['amount'], self.data.loc[list(train), 'amount'].median())
        self.assertNotIn('isFlaggedFraud', self.prep.feature_names)
        self.assertNotIn('isFraud', self.prep.feature_names)

    def test_duplicate_inference_rows_are_preserved(self):
        duplicate = pd.concat([self.data.iloc[[0]], self.data.iloc[[0]]])
        transformed = self.prep.transform_new_data(duplicate)
        self.assertEqual(len(transformed), 2)
        np.testing.assert_allclose(transformed.iloc[0], transformed.iloc[1])

    def test_missing_value_is_independent_of_batch_neighbours(self):
        single = self.data.iloc[[0]].copy()
        single['amount'] = np.nan
        single['type'] = pd.NA
        neighbour = self.data.iloc[[1]].copy()
        neighbour['amount'] = 1e8
        alone = self.prep.transform_new_data(single).iloc[0]
        in_batch = self.prep.transform_new_data(pd.concat([single, neighbour])).iloc[0]
        np.testing.assert_allclose(alone, in_batch)
        self.assertTrue(np.isfinite(alone).all())

    def test_bad_numeric_type_and_negative_values_are_rejected(self):
        for column, value in [('amount', 'not-a-number'), ('amount', -10),
                              ('oldbalanceDest', np.inf), ('type', 'UNSUPPORTED')]:
            invalid = self.data.iloc[[0]].copy()
            invalid[column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.prep.transform_new_data(invalid)

    def test_missing_labels_are_rejected(self):
        invalid = self.data.copy()
        invalid.loc[0, 'isFraud'] = np.nan
        with self.assertRaises(ValueError):
            self.prep.fit_transform_partitions(invalid, self.temp.name)

    def test_oversampling_preserves_categorical_transactions(self):
        train, _, _, y_train, _, _ = self.parts
        balanced, labels = ImbalanceHandler().apply_smote(train, y_train, self.temp.name)
        restored = pd.DataFrame(self.prep.scaler.inverse_transform(balanced), columns=balanced.columns)
        dummy = restored.filter(like='type_').to_numpy()
        self.assertTrue(np.isin(np.round(dummy, 8), [0, 1]).all())
        np.testing.assert_allclose(dummy.sum(axis=1), 1)
        self.assertEqual(int((labels == 0).sum()), int((labels == 1).sum()))


class TestDeepLearning(unittest.TestCase):
    def test_deep_classifier_learns_and_reloads(self):
        rng = np.random.default_rng(42)
        X = np.r_[rng.normal(-1, 0.3, (80, 6)), rng.normal(1, 0.3, (80, 6))]
        y = np.r_[np.zeros(80), np.ones(80)].astype(int)
        model = DeepNeuralNetworkClassifier(max_epochs=15, patience=5).fit(X, y)
        self.assertGreater(average_precision_score(y, model.predict_proba(X)[:, 1]), 0.95)
        self.assertEqual(len(model.network_.coefs_), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'network.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(model.predict_proba(X), joblib.load(path).predict_proba(X))

    def test_autoencoder_uses_normal_examples_and_detects_shift(self):
        rng = np.random.default_rng(43)
        normal = rng.normal(0, 0.1, (120, 6))
        anomaly = rng.normal(5, 0.1, (20, 6))
        model = AutoencoderAnomalyDetector(max_epochs=15, patience=5).fit(
            np.r_[normal, anomaly], np.r_[np.zeros(120), np.ones(20)],
            X_val=normal[:20], y_val=np.zeros(20))
        self.assertEqual(model.normal_training_count_, 120)
        self.assertEqual(len(model.network_.coefs_), 6)
        self.assertGreater(model.predict_proba(anomaly)[:, 1].mean(), model.predict_proba(normal)[:, 1].mean())


class TestPersistedWorkflow(unittest.TestCase):
    def test_all_four_saved_models_reproduce_final_test_predictions(self):
        expected = pd.read_csv(Path(config.RESULTS_DIR) / 'test_predictions.csv').head(20)
        source = pd.read_csv(config.DATASET_PATH).iloc[expected.source_row]
        for name in config.MODEL_FILES:
            with self.subTest(model=name):
                predictor = FraudPredictor(name)
                actual = predictor.predict_batch(source)
                # XGBoost returns float32; CSV stores its shortest decimal representation.
                np.testing.assert_allclose(actual.Fraud_Probability, expected[f'{name} score'], rtol=2e-7, atol=1e-9)
                np.testing.assert_array_equal(actual.Prediction == 'FRAUDULENT', expected[f'{name} prediction'] == 1)

    def test_default_model_matches_validation_selection(self):
        predictor = FraudPredictor()
        with open(config.METADATA_PATH, encoding='utf-8') as handle:
            self.assertEqual(predictor.model_name, json.load(handle)['best_model'])

    def test_custom_model_directory_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ['model_metadata.json', 'preprocessor.joblib', config.MODEL_FILES['XGBoost']]:
                shutil.copyfile(Path(config.MODELS_DIR) / filename, Path(directory) / filename)
            prediction = FraudPredictor('XGBoost', directory).predict_batch(pd.read_csv(config.SAMPLE_TEST_PATH).head(2))
            self.assertEqual(len(prediction), 2)

    def test_input_metadata_does_not_change_predictions(self):
        predictor = FraudPredictor()
        data = pd.read_csv(config.SAMPLE_TEST_PATH).head(5)
        changed = data.copy()
        changed['isFlaggedFraud'] = 1 - changed['isFlaggedFraud']
        changed['isFraud'] = 1 - changed['isFraud']
        np.testing.assert_allclose(predictor.predict_batch(data).Fraud_Probability,
                                   predictor.predict_batch(changed).Fraud_Probability)

    def test_unknown_model_and_empty_batch_are_rejected(self):
        with self.assertRaises(ValueError):
            FraudPredictor('missing-model')
        with self.assertRaises(ValueError):
            FraudPredictor().predict_batch(pd.DataFrame(columns=config.RAW_FEATURE_COLUMNS))


if __name__ == '__main__':
    unittest.main()
