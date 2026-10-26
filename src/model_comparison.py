"""Compare saved models at their own validation-selected thresholds.

The review rule is deliberately explicit: any model flag sends a transaction
for review. Votes are a comparison aid, not an ensemble probability or evidence
that fraud has been confirmed. No model is trained or modified here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

try:
    from .predictor import FraudPredictor
except ImportError:  # Support the project's existing src-on-sys.path entry point.
    from predictor import FraudPredictor


class FraudModelComparison:
    """Apply all four saved ML/DL models without combining their scores."""

    def __init__(self, models_dir=config.MODELS_DIR):
        self.model_names = list(config.MODEL_FILES)
        self._predictors = {
            name: FraudPredictor(name, models_dir=models_dir)
            for name in self.model_names
        }

    @staticmethod
    def _review_status(votes, checked):
        if votes == 0:
            return f'No model flagged ({votes}/{checked}); this does not establish legitimacy.'
        if votes == checked:
            return f'All models flagged ({votes}/{checked}); manual review required, fraud not confirmed.'
        return f'Models disagree ({votes}/{checked} flagged); manual review required, fraud not confirmed.'

    def _reserved_columns(self):
        columns = {'Source_Row', 'Fraud_Votes', 'Models_Checked', 'Flagged_By',
                   'Prediction', 'Review_Status'}
        for name in self.model_names:
            columns.update(f'{name} {suffix}' for suffix in ('Prediction', 'Score', 'Threshold'))
        return columns

    def predict_batch(self, df):
        """Return positional, row-preserving model decisions for a raw CSV frame.

        Existing Transaction_ID values, including duplicates, remain untouched.
        Otherwise deterministic positional IDs are generated. Source_Row always
        identifies a one-based position in this input, independently of its
        pandas index or supplied transaction identifier.

        Comparison output names are reserved so existing source metadata cannot
        be silently overwritten when a previously scored file is supplied.
        """
        next(iter(self._predictors.values())).validate_input_schema(df)
        collisions = sorted(self._reserved_columns().intersection(df.columns))
        if collisions:
            raise ValueError('Remove or rename existing comparison output columns before rescoring: '
                             + ', '.join(collisions))

        source = df.copy()
        if 'Transaction_ID' not in source:
            source.insert(0, 'Transaction_ID', [f'ROW-{position:06d}' for position in range(1, len(source) + 1)])
        output = source.copy()
        output['Source_Row'] = np.arange(1, len(source) + 1, dtype=int)
        flags = []
        for name, predictor in self._predictors.items():
            prediction = predictor.predict_batch(source)
            # Positional arrays avoid pandas alignment when source indices repeat.
            decisions = prediction['Prediction'].to_numpy(copy=True)
            output[f'{name} Prediction'] = decisions
            output[f'{name} Score'] = prediction['Fraud_Probability'].to_numpy(copy=True)
            output[f'{name} Threshold'] = predictor.threshold
            flags.append(decisions == 'FRAUDULENT')

        model_flags = np.stack(flags, axis=1)
        votes = model_flags.sum(axis=1).astype(int)
        output['Fraud_Votes'] = votes
        output['Models_Checked'] = len(self.model_names)
        output['Flagged_By'] = [
            ', '.join(name for name, flagged in zip(self.model_names, row) if flagged) or 'No model flags'
            for row in model_flags
        ]
        output['Prediction'] = np.where(votes > 0, 'FLAGGED FOR REVIEW', 'NO MODEL FLAG')
        output['Review_Status'] = [self._review_status(int(vote), len(self.model_names)) for vote in votes]
        return output

    def predict_single(self, tx):
        """Compare a transaction with its own model thresholds and score types."""
        if not isinstance(tx, dict):
            raise ValueError('A single transaction must be supplied as a dictionary.')
        result = self.predict_batch(pd.DataFrame([tx])).iloc[0]
        models = []
        flagged_by = []
        for name, predictor in self._predictors.items():
            prediction = result[f'{name} Prediction']
            models.append({
                'Model': name,
                'Family': predictor.metadata['models'][name]['family'],
                'Prediction': prediction,
                'Score': float(result[f'{name} Score']),
                'Threshold': float(predictor.threshold),
                'Score_Kind': predictor.score_kind,
            })
            if prediction == 'FRAUDULENT':
                flagged_by.append(name)
        return {
            'Transaction_ID': result['Transaction_ID'],
            'models': models,
            'Fraud_Votes': int(result['Fraud_Votes']),
            'Models_Checked': len(self.model_names),
            'Flagged_By': flagged_by,
            'Prediction': result['Prediction'],
            'Review_Status': result['Review_Status'],
        }
