"""Reproducible ML + DL fraud experiment and dashboard artifact generation."""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone

import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, BASE_DIR)

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from imbalance_handler import ImbalanceHandler
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from explainability import ExplainabilityModule
from predictor import FraudPredictor


def plot_training_history(trainer):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    for ax, name, metric, label in [
        (axes[0], 'Deep Neural Network', 'validation_ap', 'Validation average precision'),
        (axes[1], 'Deep Autoencoder', 'validation_mse', 'Normal validation reconstruction MSE'),
    ]:
        model = trainer.models[name]
        history = pd.DataFrame(model.history_)
        ax.plot(history['epoch'], history[metric], color='#2563eb')
        ax.axvline(model.best_epoch_, color='#d97706', linestyle='--', label=f'Checkpoint: epoch {model.best_epoch_}')
        ax.set(title=name, xlabel='Epoch', ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()
        history.to_csv(os.path.join(config.RESULTS_DIR, config.MODEL_FILES[name].replace('.joblib', '_history.csv')), index=False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, 'deep_learning_training.png'))
    plt.close(fig)


def run_pipeline():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                        handlers=[logging.FileHandler(config.LOG_FILE_PATH, mode='w', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)], force=True)
    logger = logging.getLogger('FraudGuard')
    start = time.perf_counter()
    logger.info('FraudGuard AI: Random Forest, XGBoost, deep classifier and deep autoencoder')
    if not os.path.exists(config.DATASET_PATH):
        from data.generate_data import main as generate
        generate()
    loader = DataLoader(config.DATASET_PATH)
    preprocessor = DataPreprocessor(test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform_partitions(
        loader.df, config.MODELS_DIR, validation_size=config.VALIDATION_SPLIT_SIZE)
    logger.info('Split: %s training, %s validation, %s final test', len(X_train), len(X_val), len(X_test))
    handler = ImbalanceHandler(random_state=config.RANDOM_STATE)
    X_balanced, y_balanced = handler.apply_smote(X_train, y_train, config.RESULTS_DIR)
    trainer = ModelTrainer(config.RANDOM_STATE)
    trainer.train_all(X_balanced, y_balanced, X_val, y_val, X_train, y_train)
    trainer.select_on_validation(X_val, y_val)
    trainer.save_models(config.MODELS_DIR)
    pd.DataFrame(trainer.validation_metrics).to_csv(os.path.join(config.RESULTS_DIR, 'validation_comparison.csv'), index=False)
    logger.info('Selected model on validation only: %s', trainer.best_model_name)
    evaluator = ModelEvaluator(config.RESULTS_DIR)
    comparison = evaluator.evaluate_and_generate_all(trainer, X_test, y_test)
    plot_training_history(trainer)
    xai = ExplainabilityModule(config.RESULTS_DIR)
    xai.explain_pipeline(evaluator, trainer, X_test)
    predictions = pd.DataFrame({'source_row': X_test.index, 'isFraud': y_test.to_numpy()})
    for name in trainer.models:
        labels, scores = trainer.predict_and_score(name, X_test)
        predictions[f'{name} score'] = scores
        predictions[f'{name} prediction'] = labels
    predictions.to_csv(os.path.join(config.RESULTS_DIR, 'test_predictions.csv'), index=False)
    with open(config.DATASET_PATH, 'rb') as handle:
        dataset_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = trainer.model_metadata()
    metadata.update({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'random_state': config.RANDOM_STATE,
        'dataset': {'path': 'data/transactions.csv', 'sha256': dataset_hash,
                    'provenance': 'Locally generated synthetic benchmark; not real bank transactions',
                    'rows': len(loader.df), 'fraud_count': int(loader.df[config.TARGET_COLUMN].sum())},
        'splits': {name: {'rows': len(y), 'fraud_count': int(y.sum())} for name, y in
                   [('train', y_train), ('validation', y_val), ('test', y_test)]},
        'feature_names': preprocessor.feature_names,
        'oversampling': 'SMOTE-NC on training only; autoencoder sees original legitimate training rows only',
        'shap_model': xai.best_model_name,
        'python': platform.python_version(),
        'versions': {package: importlib.metadata.version(package) for package in
                     ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'imbalanced-learn', 'shap', 'streamlit']},
        'elapsed_seconds': round(time.perf_counter() - start, 3),
    })
    with open(config.METADATA_PATH, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    with open(os.path.join(config.RESULTS_DIR, 'split_indices.json'), 'w', encoding='utf-8') as handle:
        json.dump(preprocessor.partition_indices_, handle)
    # A held-out demonstration CSV includes examples of both labels.
    held_out = loader.df.loc[X_test.index]
    demo = pd.concat([held_out[held_out.isFraud == 1].head(10), held_out[held_out.isFraud == 0].head(20)])
    demo.sample(frac=1, random_state=42).to_csv(config.SAMPLE_TEST_PATH, index=False)
    for name in trainer.models:
        prediction = FraudPredictor(name).predict_batch(demo[config.RAW_FEATURE_COLUMNS])
        assert len(prediction) == len(demo)
    logger.info('Pipeline completed in %.1fs. All four saved models passed inference checks.', time.perf_counter() - start)
    logger.info('Launch: .\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py')
    return comparison


if __name__ == '__main__':
    run_pipeline()
