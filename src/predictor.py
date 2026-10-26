"""Single and batch inference with the saved validation-selected operating point."""

import json
import os
import sys
import uuid
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
import config


class FraudPredictor:
    def __init__(self, model_name=None, models_dir=config.MODELS_DIR):
        self.models_dir = os.fspath(models_dir)
        manifest_path = os.path.join(self.models_dir, 'model_metadata.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError('Trained ML/DL artifacts are missing. Run main.py first.')
        with open(manifest_path, encoding='utf-8') as handle:
            self.metadata = json.load(handle)
        aliases = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'dnn': 'Deep Neural Network', 'ae': 'Deep Autoencoder'}
        aliases.update({name.lower(): name for name in config.MODEL_FILES})
        requested = aliases.get(str(model_name).lower(), model_name) if model_name else self.metadata['best_model']
        if requested not in self.metadata['models']:
            raise ValueError(f'Unknown model {requested!r}. Available: {list(self.metadata["models"])}')
        self.model_name = requested
        details = self.metadata['models'][requested]
        self.threshold = float(details['threshold'])
        self.score_kind = details['score_kind']
        self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.joblib'))
        self.model = joblib.load(os.path.join(self.models_dir, details['file']))

    def validate_input_schema(self, df):
        self.preprocessor.validate_raw_data(df)
        return list(df.columns)

    def calculate_risk_level(self, probability):
        if probability >= config.RISK_THRESHOLDS['HIGH']:
            level = 'CRITICAL'
        elif probability >= config.RISK_THRESHOLDS['MEDIUM']:
            level = 'HIGH'
        elif probability >= config.RISK_THRESHOLDS['LOW']:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        return level, f'[{level} RISK]', config.RISK_ACTIONS[level]

    def identify_contributing_risk_factors(self, tx_row):
        factors = []
        amount, old, new = (tx_row.get(c, 0) for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig'])
        if tx_row.get('type') in ['TRANSFER', 'CASH_OUT']:
            factors.append(f'Transaction type: {tx_row["type"]}')
        if old > 0 and new == 0:
            factors.append('Sender balance completely depleted')
        if amount > 100000:
            factors.append('Transaction amount exceeds 100,000 currency units')
        if old > 0 and amount / old > 0.85:
            factors.append('Amount exceeds 85% of sender opening balance')
        return factors or ['No predefined rule indicators triggered.']

    def predict_single(self, tx_dict):
        frame = pd.DataFrame([tx_dict])
        result = self.predict_batch(frame).iloc[0]
        return {
            'Transaction_ID': result['Transaction_ID'], 'Model_Used': self.model_name,
            'Prediction': result['Prediction'], 'Fraud_Probability': float(result['Fraud_Probability']),
            'Fraud_Probability_Pct': f'{result["Risk_Score_Pct"]:.2f}%',
            'Risk_Level': result['Risk_Level'], 'Risk_Badge': result['Risk_Badge'],
            'Recommended_Action': result['Recommended_Action'],
            'Top_Risk_Factors': self.identify_contributing_risk_factors(self.preprocessor.clean_data(frame).iloc[0]),
            'Raw_Inputs': tx_dict, 'Decision_Threshold': self.threshold, 'Score_Kind': self.score_kind,
        }

    def predict_batch(self, df):
        self.validate_input_schema(df)
        output = df.copy()
        if 'Transaction_ID' not in output:
            output.insert(0, 'Transaction_ID', [f'TXN-{uuid.uuid4().hex[:10].upper()}' for _ in range(len(df))])
        features = self.preprocessor.transform_new_data(df)
        scores = np.asarray(self.model.predict_proba(features)[:, 1])
        if not np.isfinite(scores).all():
            raise ValueError('Model produced non-finite scores. Check inputs and retrain.')
        risk = [self.calculate_risk_level(float(score)) for score in scores]
        output['Prediction'] = np.where(scores >= self.threshold, 'FRAUDULENT', 'LEGITIMATE')
        # Retain the legacy column for CSV compatibility; the UI calls it a score.
        output['Fraud_Probability'] = scores
        output['Risk_Score_Pct'] = scores * 100
        output['Risk_Level'] = [row[0] for row in risk]
        output['Risk_Badge'] = [row[1] for row in risk]
        output['Recommended_Action'] = [row[2] for row in risk]
        output['Model_Used'] = self.model_name
        output['Decision_Threshold'] = self.threshold
        output['Score_Kind'] = self.score_kind
        return output
