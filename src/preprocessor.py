"""Training-only imputation/scaling and a fixed schema shared with inference."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import config


class DataPreprocessor:
    def __init__(self, target_col='isFraud', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.feature_names = []
        self.is_fitted = False
        self.numeric_columns = [c for c in config.RAW_FEATURE_COLUMNS if c != 'type']

    def validate_raw_data(self, df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError('Provide at least one transaction.')
        if not df.columns.is_unique:
            raise ValueError('Transaction column names must be unique.')
        missing = sorted(set(config.RAW_FEATURE_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f'Missing required transaction columns: {missing}')
        data = df[config.RAW_FEATURE_COLUMNS].copy()
        for col in self.numeric_columns:
            numeric = pd.to_numeric(data[col], errors='coerce')
            if (data[col].notna() & numeric.isna()).any():
                raise ValueError(f"Column '{col}' must contain numeric values.")
            if np.isinf(numeric.to_numpy(dtype=float)).any():
                raise ValueError(f"Column '{col}' cannot contain infinity.")
            if (numeric.dropna() < 0).any():
                raise ValueError(f"Column '{col}' cannot contain negative values.")
            data[col] = numeric.astype(float)
        data['type'] = data['type'].astype('string').str.strip().str.upper()
        invalid = data['type'].notna() & ~data['type'].isin(config.TRANSACTION_TYPES)
        if invalid.any():
            raise ValueError(f'Unsupported transaction type. Use: {config.TRANSACTION_TYPES}')
        return data

    def clean_data(self, df):
        """Impute using stored training statistics; preserve every inference row."""
        data = self.validate_raw_data(df)
        for col in self.numeric_columns:
            data[col] = data[col].fillna(self.numeric_fill_[col])
        data['type'] = data['type'].fillna(self.type_fill_)
        return data

    def engineer_features(self, df):
        data = df[self.numeric_columns].copy()
        data['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        data['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        data['amount_to_oldbalance_ratio'] = np.divide(
            df['amount'].to_numpy(dtype=float),
            df['oldbalanceOrg'].to_numpy(dtype=float) + 1e-5,
            out=np.zeros(len(df)), where=df['oldbalanceOrg'].to_numpy(dtype=float) > 0,
        )
        for tx_type in config.TRANSACTION_TYPES:
            data[f'type_{tx_type}'] = (df['type'] == tx_type).astype(float)
        return data

    def _fit(self, raw_train, models_dir):
        train = self.validate_raw_data(raw_train)
        self.numeric_fill_ = train[self.numeric_columns].median().fillna(0).to_dict()
        modes = train['type'].mode()
        self.type_fill_ = str(modes.iloc[0]) if len(modes) else 'PAYMENT'
        features = self.engineer_features(self.clean_data(train))
        self.feature_names = list(features.columns)
        self.scaler.fit(features)
        self.is_fitted = True
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self, os.path.join(models_dir, 'preprocessor.joblib'))
        joblib.dump(self.scaler, os.path.join(models_dir, 'scaler.joblib'))
        joblib.dump(self.feature_names, os.path.join(models_dir, 'feature_names.joblib'))

    def _labelled_data(self, df):
        self.validate_raw_data(df)
        if self.target_col not in df:
            raise ValueError(f"Target column '{self.target_col}' not found.")
        if df[self.target_col].isna().any() or not df[self.target_col].isin([0, 1]).all():
            raise ValueError('Training labels must be 0 or 1, with no missing labels.')
        return df.drop_duplicates(subset=config.RAW_FEATURE_COLUMNS + [self.target_col]).copy()

    def fit_transform_split(self, df, models_dir='models'):
        """Compatibility entry point: a stratified train/test split."""
        data = self._labelled_data(df)
        train, test = train_test_split(data, test_size=self.test_size,
                                      random_state=self.random_state, stratify=data[self.target_col])
        self._fit(train, models_dir)
        return (self.transform_new_data(train), self.transform_new_data(test),
                train[self.target_col].astype(int), test[self.target_col].astype(int))

    def fit_transform_partitions(self, df, models_dir='models', validation_size=0.2):
        """60/20/20 by default; all fitted preprocessing uses training rows only."""
        if not 0 < validation_size < 1 - self.test_size:
            raise ValueError('Validation and test fractions must leave training data.')
        data = self._labelled_data(df)
        development, test = train_test_split(data, test_size=self.test_size,
                                            random_state=self.random_state, stratify=data[self.target_col])
        train, validation = train_test_split(
            development, test_size=validation_size / (1 - self.test_size),
            random_state=self.random_state, stratify=development[self.target_col])
        self.partition_indices_ = {name: part.index.tolist() for name, part in
                                   [('train', train), ('validation', validation), ('test', test)]}
        self._fit(train, models_dir)
        return (self.transform_new_data(train), self.transform_new_data(validation), self.transform_new_data(test),
                train[self.target_col].astype(int), validation[self.target_col].astype(int),
                test[self.target_col].astype(int))

    def transform_new_data(self, df):
        if not self.is_fitted:
            raise ValueError('Preprocessor has not been fitted. Run main.py first.')
        features = self.engineer_features(self.clean_data(df))[self.feature_names]
        return pd.DataFrame(self.scaler.transform(features), columns=self.feature_names, index=df.index)
