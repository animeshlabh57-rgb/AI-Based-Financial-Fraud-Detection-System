"""Verify numeric row observations and the separation of predictions and labels."""

import math
import unittest

import config
from dashboard.transaction_explanation import build_transaction_explanation


class TestTransactionExplanation(unittest.TestCase):
    def transfer(self, **updates):
        tx = dict(step=1, type='TRANSFER', amount=900, oldbalanceOrg=1000,
                  newbalanceOrig=100, oldbalanceDest=200, newbalanceDest=1100)
        tx.update(updates)
        return tx

    def test_exact_balance_features_and_ratios(self):
        result = build_transaction_explanation(self.transfer(newbalanceOrig=150, newbalanceDest=1050))
        values = result['measurements']
        self.assertEqual(values['sender_debit'], 850)
        self.assertEqual(values['sender_debit_gap'], -50)
        self.assertEqual(values['errorBalanceOrig'], 50)
        self.assertEqual(values['recipient_credit'], 850)
        self.assertEqual(values['recipient_credit_gap'], 50)
        self.assertEqual(values['errorBalanceDest'], 50)
        self.assertAlmostEqual(values['amount_to_opening_balance'], 0.9)

    def test_balanced_transaction_is_not_declared_fraud_or_legitimate(self):
        result = build_transaction_explanation(self.transfer())
        self.assertEqual(result['measurements']['errorBalanceOrig'], 0)
        self.assertEqual(result['measurements']['errorBalanceDest'], 0)
        self.assertIn('cannot be determined', result['summary'])
        self.assertNotIn('→ FRAUDULENT', result['summary'])
        self.assertNotIn('→ LEGITIMATE', result['summary'])
        self.assertIsNone(result['label'])

    def test_model_threshold_equality_flags_and_supplied_label_is_separate(self):
        model_rows = [
            {'Model': 'XGBoost', 'Score': 0.6, 'Threshold': 0.6, 'Prediction': 'FRAUDULENT'},
            {'Model': 'Deep Neural Network', 'Score': 0.4, 'Threshold': 0.7},
        ]
        result = build_transaction_explanation(self.transfer(isFraud=0), model_rows)
        self.assertEqual([row['Prediction'] for row in result['model_decisions']], ['FRAUDULENT', 'LEGITIMATE'])
        self.assertIn('XGBoost', result['summary'])
        self.assertIn('Deep Neural Network', result['summary'])
        self.assertIn('≥ threshold 0.600000', result['summary'])
        self.assertIn('Provided dataset label: LEGITIMATE', result['label'])
        self.assertIn('separate', result['label'])

    def test_no_label_is_fabricated_from_model_prediction(self):
        result = build_transaction_explanation(self.transfer(), [
            {'Model': 'Random Forest', 'Score': 0.99, 'Threshold': 0.5},
        ])
        self.assertIsNone(result['label'])
        for invalid_label in [None, math.nan, 2, -1, 'fraud', True]:
            with self.subTest(label=invalid_label):
                self.assertIsNone(build_transaction_explanation(self.transfer(isFraud=invalid_label))['label'])
        self.assertIn('FRAUDULENT', build_transaction_explanation(self.transfer(isFraud=1))['label'])

    def test_null_and_empty_inputs_are_safe(self):
        for payload in [None, {}, [], 'not a transaction']:
            with self.subTest(payload=payload):
                result = build_transaction_explanation(payload)
                self.assertEqual(result['indicators'], [])
                self.assertIsNone(result['label'])
                self.assertEqual(result['model_decisions'], [])
                self.assertEqual(set(result['missing_inputs']), set(config.RAW_FEATURE_COLUMNS))
                self.assertIn('cannot be determined', result['summary'])

    def test_partial_and_nonfinite_inputs_do_not_become_zero(self):
        result = build_transaction_explanation({'amount': 500, 'oldbalanceOrg': math.nan,
                                               'newbalanceOrig': None, 'type': 'TRANSFER'})
        self.assertEqual(result['measurements']['amount'], 500)
        self.assertNotIn('sender_debit', result['measurements'])
        self.assertNotIn('amount_to_opening_balance', result['measurements'])
        self.assertNotIn('recipient_opening_zero', result['measurements'])
        self.assertIn('oldbalanceOrg', result['missing_inputs'])
        self.assertIn('newbalanceOrig', result['missing_inputs'])

    def test_cash_in_direction_is_explained(self):
        result = build_transaction_explanation(self.transfer(type='CASH_IN', newbalanceOrig=1900))
        sender = next(row for row in result['indicators'] if row['Indicator'] == 'Sender debit versus transaction amount')
        self.assertEqual(result['measurements']['sender_debit'], -900)
        self.assertIn('not a debit consistency test', sender['Meaning'])
        self.assertNotIn('Zero recorded recipient opening balance', [row['Indicator'] for row in result['indicators']])

    def test_merchant_and_cash_out_recipient_gaps_are_not_proof(self):
        for transaction_type in ['PAYMENT', 'DEBIT', 'CASH_OUT']:
            with self.subTest(transaction_type=transaction_type):
                result = build_transaction_explanation(self.transfer(type=transaction_type, oldbalanceDest=0, newbalanceDest=0))
                row = next(row for row in result['indicators'] if row['Indicator'] == 'Recipient credit gap / engineered feature')
                self.assertEqual(result['measurements']['errorBalanceDest'], 900)
                self.assertTrue('not a reliable fraud test' in row['Meaning'] or 'not interpret' in row['Meaning'])

    def test_zero_opening_balance_has_no_infinite_ratio(self):
        result = build_transaction_explanation(self.transfer(oldbalanceOrg=0, newbalanceOrig=0))
        self.assertNotIn('amount_to_opening_balance', result['measurements'])
        self.assertFalse(result['measurements']['sender_drained'])
        ratio = next(row for row in result['indicators'] if row['Indicator'] == 'Amount / sender opening balance')
        self.assertIn('Undefined', ratio['Observed'])

    def test_incomplete_and_conflicting_model_outputs_are_explicit(self):
        result = build_transaction_explanation(self.transfer(), [None,
            {'Model': 'Random Forest', 'Prediction': 'FRAUDULENT', 'Score': math.nan},
            {'Model': 'XGBoost', 'Prediction': 'FRAUDULENT', 'Score': 0.2, 'Threshold': 0.8},
        ])
        self.assertEqual(result['model_decisions'][0]['Prediction'], 'Unavailable')
        self.assertEqual(result['model_decisions'][1]['Prediction'], 'LEGITIMATE')
        self.assertEqual(result['model_decisions'][1]['Reported_Prediction'], 'FRAUDULENT')
        self.assertIn('could not be reproduced', result['summary'])

    def test_renderer_heading_matches_decision_and_can_hide_duplicate_table(self):
        from streamlit.testing.v1 import AppTest

        cases = [
            (0.9, 0.8, 'Why was this transaction flagged?'),
            (0.2, 0.8, 'Why was this transaction not flagged?'),
            (None, None, 'Understand this transaction'),
        ]
        for score, threshold, heading in cases:
            with self.subTest(score=score):
                rows = [{'Model': 'XGBoost', 'Score': score, 'Threshold': threshold}]
                source = ('from dashboard.transaction_explanation import render_transaction_explanation\n'
                          f'render_transaction_explanation({self.transfer()!r}, {rows!r}, show_model_table=False)')
                app = AppTest.from_string(source).run(timeout=20)
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(app.subheader[0].value, heading)
                self.assertEqual(len(app.dataframe), 1, 'Only the quantitative observations table should appear')


if __name__ == '__main__':
    unittest.main()
