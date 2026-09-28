import sys
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.schema import FEATURE_COLUMNS, NUMERIC_FEATURES
from src.utils import load_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictPipeline:
    def __init__(self, model_path=None, preprocessor_path=None):
        artifact_dir = PROJECT_ROOT / "artifacts"
        self.model_path = Path(model_path or artifact_dir / "model.pkl")
        self.preprocessor_path = Path(preprocessor_path or artifact_dir / "preprocessor.pkl")

    def predict(self, features):
        try:
            missing_columns = sorted(set(FEATURE_COLUMNS) - set(features.columns))
            if missing_columns:
                raise ValueError(f"Prediction input is missing columns: {missing_columns}")
            features = features.loc[:, FEATURE_COLUMNS].copy()
            for column in NUMERIC_FEATURES:
                features[column] = pd.to_numeric(features[column], errors="coerce")
            features["SeniorCitizen"] = features["SeniorCitizen"].astype(str)
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            transformed_features = preprocessor.transform(features)
            prediction = int(model.predict(transformed_features)[0])
            probability = float(model.predict_proba(transformed_features)[0, 1])
            risk_level = "LOW" if probability < 0.30 else "MEDIUM" if probability < 0.60 else "HIGH"
            logging.info(
                "Churn inference completed; prediction=%s risk_level=%s",
                prediction,
                risk_level,
            )
            return {
                "prediction": prediction,
                "churn_probability": probability,
                "risk_level": risk_level,
            }
        except Exception as error:
            raise CustomException(error, sys) from error


class TelcoCustomerData:
    def __init__(self, **values):
        missing_fields = sorted(set(FEATURE_COLUMNS) - set(values))
        if missing_fields:
            raise ValueError(f"Missing customer fields: {missing_fields}")
        self.values = {column: values[column] for column in FEATURE_COLUMNS}
        for column in NUMERIC_FEATURES:
            try:
                value = self.values[column]
                if column == "TotalCharges" and value in (None, ""):
                    self.values[column] = float("nan")
                else:
                    self.values[column] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{column} must be numeric") from error

    def get_data_as_data_frame(self):
        return pd.DataFrame([self.values], columns=FEATURE_COLUMNS)

