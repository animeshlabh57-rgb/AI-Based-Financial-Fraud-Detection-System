"""Compare review flags against the actual saved ML and DL predictors."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.model_comparison import FraudModelComparison
from src.predictor import FraudPredictor


class TestFraudModelComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = FraudModelComparison()
        cls.samples = pd.read_csv(config.SAMPLE_TEST_PATH)
        cls.individual = {name: FraudPredictor(name) for name in config.MODEL_FILES}

    def test_all_four_models_match_saved_scores_and_threshold_decisions(self):
        self.assertEqual(self.comparison.model_names, list(config.MODEL_FILES))
        result = self.comparison.predict_batch(self.samples)
        flags = []
        for name, predictor in self.individual.items():
            expected = predictor.predict_batch(self.samples)
            np.testing.assert_allclose(result[f'{name} Score'], expected['Fraud_Probability'],
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(result[f'{name} Prediction'], expected['Prediction'])
            self.assertTrue((result[f'{name} Threshold'] == predictor.threshold).all())
            np.testing.assert_array_equal(result[f'{name} Prediction'] == 'FRAUDULENT',
                                          result[f'{name} Score'] >= predictor.threshold)
            flags.append(expected['Prediction'].to_numpy() == 'FRAUDULENT')
        votes = np.stack(flags).sum(axis=0)
        np.testing.assert_array_equal(result['Fraud_Votes'], votes)
        self.assertTrue((result['Models_Checked'] == 4).all())
        np.testing.assert_array_equal(result['Prediction'] == 'FLAGGED FOR REVIEW', votes > 0)
        self.assertNotIn('Fraud_Probability', result)
        self.assertNotIn('Average_Score', result)

    def test_preserves_rows_indices_metadata_and_duplicate_transaction_ids(self):
        source = self.samples.iloc[[4, 0, 4, 7]].copy()
        source.index = [9, 9, 2, 9]
        source['Transaction_ID'] = ['same', 'same', 'same', 'other']
        source['Reviewer_note'] = ['first', 'second', 'duplicate', 'fourth']
        before = source.copy(deep=True)
        result = self.comparison.predict_batch(source)
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result[source.columns], source)
        self.assertEqual(result.index.tolist(), [9, 9, 2, 9])
        self.assertEqual(result['Source_Row'].tolist(), [1, 2, 3, 4])
        for name in self.comparison.model_names:
            self.assertEqual(result.iloc[0][f'{name} Score'], result.iloc[2][f'{name} Score'])

    def test_missing_transaction_ids_are_deterministic_and_positional(self):
        source = self.samples.head(3)
        first = self.comparison.predict_batch(source)
        second = self.comparison.predict_batch(source)
        self.assertEqual(first['Transaction_ID'].tolist(), ['ROW-000001', 'ROW-000002', 'ROW-000003'])
        self.assertEqual(first['Transaction_ID'].tolist(), second['Transaction_ID'].tolist())
        self.assertEqual(first['Source_Row'].tolist(), [1, 2, 3])

    def test_ground_truth_and_flag_metadata_do_not_change_model_scores(self):
        changed = self.samples.copy()
        changed['isFraud'] = 1 - changed['isFraud']
        changed['isFlaggedFraud'] = 1 - changed['isFlaggedFraud']
        expected = self.comparison.predict_batch(self.samples)
        actual = self.comparison.predict_batch(changed)
        for name in self.comparison.model_names:
            np.testing.assert_allclose(actual[f'{name} Score'], expected[f'{name} Score'],
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(actual[f'{name} Prediction'], expected[f'{name} Prediction'])
        pd.testing.assert_series_equal(actual['isFraud'], changed['isFraud'])
        pd.testing.assert_series_equal(actual['isFlaggedFraud'], changed['isFlaggedFraud'])

    def test_single_model_details_match_each_saved_predictor(self):
        tx = self.samples.iloc[4].to_dict()
        result = self.comparison.predict_single(tx)
        flagged = []
        self.assertEqual(len(result['models']), 4)
        for model_result in result['models']:
            predictor = self.individual[model_result['Model']]
            expected = predictor.predict_single(tx)
            self.assertEqual(model_result['Prediction'], expected['Prediction'])
            self.assertAlmostEqual(model_result['Score'], expected['Fraud_Probability'], places=12)
            self.assertEqual(model_result['Threshold'], predictor.threshold)
            self.assertEqual(model_result['Score_Kind'], predictor.score_kind)
            self.assertEqual(model_result['Family'], config.MODEL_FAMILIES[model_result['Model']])
            if model_result['Prediction'] == 'FRAUDULENT':
                flagged.append(model_result['Model'])
        self.assertEqual(result['Flagged_By'], flagged)
        self.assertEqual(result['Fraud_Votes'], len(flagged))
        self.assertEqual(result['Models_Checked'], 4)
        self.assertIn('fraud not confirmed', result['Review_Status'])

    def test_disagreements_keep_individual_decisions_and_review_wording(self):
        result = self.comparison.predict_batch(self.samples)
        disagreements = result[result['Fraud_Votes'].between(1, 3)]
        if disagreements.empty:
            # The small demonstration CSV may contain only clear examples.
            # Replay disagreement rows from the saved held-out evaluation.
            held_out = pd.read_csv(os.path.join(config.RESULTS_DIR, 'test_predictions.csv'))
            decisions = [f'{name} prediction' for name in self.comparison.model_names]
            examples = held_out.loc[held_out[decisions].nunique(axis=1) > 1].head(10)
            self.assertFalse(examples.empty, 'Saved evaluation should include disagreement examples.')
            transactions = pd.read_csv(config.DATASET_PATH).iloc[examples['source_row'].astype(int).tolist()]
            replayed = self.comparison.predict_batch(transactions)
            for name in self.comparison.model_names:
                np.testing.assert_array_equal(
                    replayed[f'{name} Prediction'],
                    np.where(examples[f'{name} prediction'] == 1, 'FRAUDULENT', 'LEGITIMATE'),
                )
            result = pd.concat([result, replayed], ignore_index=True)
            disagreements = result[result['Fraud_Votes'].between(1, 3)]
        self.assertFalse(disagreements.empty, 'The comparison must preserve individual model disagreements.')
        for _, row in disagreements.iterrows():
            flagged = [name for name in self.comparison.model_names if row[f'{name} Prediction'] == 'FRAUDULENT']
            self.assertEqual(row['Flagged_By'], ', '.join(flagged))
            self.assertEqual(row['Prediction'], 'FLAGGED FOR REVIEW')
            self.assertIn('Models disagree', row['Review_Status'])
            self.assertIn('fraud not confirmed', row['Review_Status'])
        unflagged = result[result['Fraud_Votes'] == 0]
        self.assertTrue((unflagged['Flagged_By'] == 'No model flags').all())
        self.assertTrue((unflagged['Prediction'] == 'NO MODEL FLAG').all())

    def test_validation_errors_are_clear_and_source_metadata_is_not_overwritten(self):
        bad_type = self.samples.head(1).copy()
        bad_type['type'] = 'UNSUPPORTED'
        bad_amount = self.samples.head(1).copy()
        bad_amount['amount'] = -1
        for source in (None, [], pd.DataFrame(), self.samples.drop(columns='amount'), bad_type, bad_amount):
            with self.subTest(input_type=type(source).__name__), self.assertRaises(ValueError):
                self.comparison.predict_batch(source)
        with self.assertRaisesRegex(ValueError, 'dictionary'):
            self.comparison.predict_single([])
        for column in ('Source_Row', 'Prediction', 'Random Forest Score'):
            source = self.samples.head(1).assign(**{column: 'source metadata'})
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'Remove or rename'):
                self.comparison.predict_batch(source)

    def test_missing_models_fail_instead_of_silently_skipping_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                FraudModelComparison(models_dir=directory)


if __name__ == '__main__':
    unittest.main()
