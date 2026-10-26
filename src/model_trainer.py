"""Train ML and DL models; select decision thresholds using validation only."""

import json
import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score
from xgboost import XGBClassifier

import config
from src.neural_models import DeepNeuralNetworkClassifier, AutoencoderAnomalyDetector


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.training_times = {}
        self.thresholds = {}
        self.validation_metrics = []
        self.best_model_name = None

    def _train(self, name, model, X, y, **fit_kwargs):
        start = time.perf_counter()
        print(f'[ModelTrainer] Training {name}...', flush=True)
        model.fit(X, y, **fit_kwargs)
        self.models[name] = model
        self.training_times[name] = time.perf_counter() - start
        self.thresholds[name] = 0.5
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=150, max_depth=12, **kwargs):
        return self._train('Random Forest', RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state,
            n_jobs=2, class_weight='balanced_subsample', **kwargs), X_train, y_train)

    def train_xgboost(self, X_train, y_train, n_estimators=150, max_depth=6,
                      learning_rate=0.08, **kwargs):
        return self._train('XGBoost', XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            eval_metric='logloss', random_state=self.random_state, n_jobs=2, **kwargs), X_train, y_train)

    def train_all(self, X_train, y_train, X_val=None, y_val=None, X_original=None, y_original=None):
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self._train('Deep Neural Network', DeepNeuralNetworkClassifier(self.random_state),
                    X_train, y_train, X_val=X_val, y_val=y_val)
        self._train('Deep Autoencoder', AutoencoderAnomalyDetector(self.random_state),
                    X_original if X_original is not None else X_train,
                    y_original if y_original is not None else y_train, X_val=X_val, y_val=y_val)
        return self.models

    def select_on_validation(self, X_val, y_val):
        self.validation_metrics = []
        for name, model in self.models.items():
            scores = model.predict_proba(X_val)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_val, scores)
            f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            # Resolve equal-F1 operating points with the higher threshold.
            best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
            threshold = float(thresholds[best])
            self.thresholds[name] = threshold
            self.validation_metrics.append({
                'Model': name, 'Family': config.MODEL_FAMILIES[name],
                'PR-AUC': float(average_precision_score(y_val, scores)),
                'F1-Score': float(f1[best]),
                'Recall (Sensitivity)': float(recall_score(y_val, scores >= threshold)),
                'Threshold': threshold,
            })
        winner = max(self.validation_metrics, key=lambda r: (r['PR-AUC'], r['F1-Score'], r['Recall (Sensitivity)']))
        self.best_model_name = winner['Model']
        return self.validation_metrics

    def predict_and_score(self, model_name, X_test):
        scores = self.models[model_name].predict_proba(X_test)[:, 1]
        return (scores >= self.thresholds.get(model_name, 0.5)).astype(int), scores

    def model_metadata(self):
        details = {}
        for name, model in self.models.items():
            details[name] = {
                'file': config.MODEL_FILES[name], 'family': config.MODEL_FAMILIES[name],
                'threshold': self.thresholds.get(name, 0.5), 'training_seconds': self.training_times[name],
                'score_kind': 'normalized reconstruction anomaly score' if name == 'Deep Autoencoder'
                              else 'uncalibrated classifier score',
            }
            if hasattr(model, 'history_'):
                details[name].update({'hidden_layers': list(model.hidden_layer_sizes),
                                     'best_epoch': model.best_epoch_, 'epochs_run': model.epochs_run_,
                                     'parameters': model.parameter_count_})
            if hasattr(model, 'normal_training_count_'):
                details[name]['legitimate_training_rows'] = model.normal_training_count_
        return {'best_model': self.best_model_name,
                'selection_rule': 'Validation average precision (PR-AUC), then F1, then recall',
                'threshold_rule': 'Maximum validation F1; tied F1 uses higher threshold',
                'validation_metrics': self.validation_metrics, 'models': details}

    def save_models(self, models_dir='models'):
        os.makedirs(models_dir, exist_ok=True)
        saved = {}
        for name, model in self.models.items():
            path = os.path.join(models_dir, config.MODEL_FILES[name])
            joblib.dump(model, path)
            saved[name] = path
        with open(os.path.join(models_dir, 'model_metadata.json'), 'w', encoding='utf-8') as handle:
            json.dump(self.model_metadata(), handle, indent=2)
        return saved

    def load_models(self, models_dir='models'):
        manifest = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(manifest):
            with open(manifest, encoding='utf-8') as handle:
                metadata = json.load(handle)
            self.best_model_name = metadata['best_model']
            self.thresholds = {name: row['threshold'] for name, row in metadata['models'].items()}
        for name, filename in config.MODEL_FILES.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
        return self.models
