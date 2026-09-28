import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


class DataTransformation:
    def get_data_transformer_object(self):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer(transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ])

    @staticmethod
    def _split_features_and_target(dataframe):
        missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataframe.columns))
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
        features = dataframe.loc[:, FEATURE_COLUMNS].copy()
        for column in NUMERIC_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
        features["SeniorCitizen"] = features["SeniorCitizen"].astype(str)
        target_values = dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower()
        target = target_values.map({"no": 0, "yes": 1, "0": 0, "1": 1})
        if target.isna().any():
            raise ValueError("Churn values must be 'Yes' or 'No'.")
        return features, target.astype(int)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_dataframe = pd.read_csv(train_path)
            test_dataframe = pd.read_csv(test_path)
            X_train, y_train = self._split_features_and_target(train_dataframe)
            X_test, y_test = self._split_features_and_target(test_dataframe)
            preprocessor = self.get_data_transformer_object()
            logging.info("Loaded raw Telco features and created preprocessing pipeline")
            return X_train, y_train, X_test, y_test, preprocessor
        except Exception as error:
            raise CustomException(error, sys) from error
