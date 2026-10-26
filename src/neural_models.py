"""CPU-trained deep networks, using scikit-learn's backpropagation/Adam backend.

The classifier has three hidden layers. The autoencoder has five hidden layers
and learns to reconstruct legitimate training transactions only.
"""

from copy import deepcopy
import numpy as np
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


class DeepNeuralNetworkClassifier:
    def __init__(self, random_state=42, hidden_layer_sizes=(64, 32, 16),
                 max_epochs=120, patience=15):
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.patience = patience

    def _transform(self, X):
        return np.clip(self.scaler_.transform(np.asarray(X, dtype=float)), -12, 12)

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        train, validation = self._transform(X), self._transform(X_val)
        network = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                activation='relu', solver='adam', alpha=0.001,
                                batch_size=min(256, len(X)), learning_rate_init=0.001,
                                random_state=self.random_state, shuffle=True)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.history_ = []
        best_score, stale = -np.inf, 0
        with threadpool_limits(limits=2):
            for epoch in range(1, self.max_epochs + 1):
                network.partial_fit(train, y, classes=self.classes_)
                score = float(average_precision_score(y_val, network.predict_proba(validation)[:, 1]))
                self.history_.append({'epoch': epoch, 'loss': float(network.loss_), 'validation_ap': score})
                if score > best_score + 1e-5:
                    best_score, stale = score, 0
                    self.network_ = deepcopy(network)
                    self.best_epoch_ = epoch
                else:
                    stale += 1
                if epoch == 1 or epoch % 10 == 0:
                    print(f'[DNN] Epoch {epoch}: loss={network.loss_:.5f}, validation AP={score:.5f}', flush=True)
                if stale >= self.patience:
                    break
        self.epochs_run_ = len(self.history_)
        self.parameter_count_ = sum(w.size + b.size for w, b in zip(self.network_.coefs_, self.network_.intercepts_))
        return self

    def predict_proba(self, X):
        return self.network_.predict_proba(self._transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AutoencoderAnomalyDetector:
    def __init__(self, random_state=42, hidden_layer_sizes=(64, 32, 8, 32, 64),
                 max_epochs=100, patience=12):
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.patience = patience

    def _transform(self, X):
        return np.clip(self.scaler_.transform(np.asarray(X, dtype=float)), -20, 20)

    def fit(self, X, y, X_val=None, y_val=None):
        normal = np.asarray(X, dtype=float)[np.asarray(y) == 0]
        if len(normal) < 10:
            raise ValueError('Autoencoder requires at least 10 legitimate training examples.')
        if X_val is None:
            normal, normal_val = train_test_split(normal, test_size=0.2, random_state=self.random_state)
        else:
            normal_val = np.asarray(X_val, dtype=float)[np.asarray(y_val) == 0]
        if not len(normal_val):
            raise ValueError('Autoencoder requires legitimate validation examples.')
        self.scaler_ = StandardScaler().fit(normal)
        train, validation = self._transform(normal), self._transform(normal_val)
        network = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                               activation='relu', solver='adam', alpha=0.001,
                               batch_size=min(256, len(normal)), learning_rate_init=0.001,
                               random_state=self.random_state, shuffle=True)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = train.shape[1]
        self.normal_training_count_ = len(normal)
        self.history_ = []
        best_loss, stale = np.inf, 0
        with threadpool_limits(limits=2):
            for epoch in range(1, self.max_epochs + 1):
                network.partial_fit(train, train)
                loss = float(mean_squared_error(validation, network.predict(validation)))
                self.history_.append({'epoch': epoch, 'loss': float(network.loss_), 'validation_mse': loss})
                if loss < best_loss - 1e-6:
                    best_loss, stale = loss, 0
                    self.network_ = deepcopy(network)
                    self.best_epoch_ = epoch
                else:
                    stale += 1
                if epoch == 1 or epoch % 10 == 0:
                    print(f'[Autoencoder] Epoch {epoch}: validation reconstruction MSE={loss:.6f}', flush=True)
                if stale >= self.patience:
                    break
        self.error_scale_ = max(float(np.quantile(self.reconstruction_error(normal), 0.99)), 1e-12)
        self.epochs_run_ = len(self.history_)
        self.parameter_count_ = sum(w.size + b.size for w, b in zip(self.network_.coefs_, self.network_.intercepts_))
        return self

    def reconstruction_error(self, X):
        transformed = self._transform(X)
        return np.mean((transformed - self.network_.predict(transformed)) ** 2, axis=1)

    def predict_proba(self, X):
        """API-compatible anomaly scores, NOT calibrated fraud probabilities."""
        error = self.reconstruction_error(X)
        score = error / (error + self.error_scale_)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
