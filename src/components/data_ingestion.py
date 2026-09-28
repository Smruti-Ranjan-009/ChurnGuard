import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.schema import FEATURE_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    data_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "CHURN_DATA_PATH",
        str(PROJECT_ROOT / "notebook" / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    )))
    train_data_path: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    raw_data_path: Path = PROJECT_ROOT / "artifacts" / "data.csv"


class DataIngestion:
    def __init__(self, ingestion_config=None):
        self.ingestion_config = ingestion_config or DataIngestionConfig()

    def initiate_data_ingestion(self):
        config = self.ingestion_config
        try:
            data_path = Path(config.data_path)
            if not data_path.is_file():
                raise FileNotFoundError(
                    f"Telco dataset not found at {data_path}. Set CHURN_DATA_PATH to its location."
                )
            dataframe = pd.read_csv(data_path)
            required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
            missing_columns = sorted(required_columns - set(dataframe.columns))
            if missing_columns:
                raise ValueError(f"Dataset is missing required columns: {missing_columns}")
            target = dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower()
            if not target.isin(["yes", "no"]).all():
                raise ValueError("Churn values must be 'Yes' or 'No'.")
            dataframe[TARGET_COLUMN] = target.map({"yes": 1, "no": 0}).astype("int8")
            dataframe["TotalCharges"] = pd.to_numeric(
                dataframe["TotalCharges"], errors="coerce"
            )
            Path(config.raw_data_path).parent.mkdir(parents=True, exist_ok=True)
            Path(config.train_data_path).parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(config.raw_data_path, index=False)
            train_set, test_set = train_test_split(
                dataframe,
                test_size=0.2,
                random_state=42,
                stratify=dataframe[TARGET_COLUMN],
            )
            train_set.to_csv(config.train_data_path, index=False)
            test_set.to_csv(config.test_data_path, index=False)
            logging.info("Ingested %d Telco customer records", len(dataframe))
            return str(config.train_data_path), str(config.test_data_path)
        except Exception as error:
            raise CustomException(error, sys) from error



