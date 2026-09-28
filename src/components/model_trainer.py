import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import compare_classifiers, evaluate_classifier, save_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: Path = PROJECT_ROOT / "artifacts" / "model.pkl"
    preprocessor_file_path: Path = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"


class ModelTrainer:
    def __init__(self, model_trainer_config=None):
        self.model_trainer_config = model_trainer_config or ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, preprocessor):
        try:
            classifiers = {
                "LogisticRegression": LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    n_estimators=120,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
                "XGBClassifier": XGBClassifier(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                ),
            }
            candidate_pipelines = {
                name: Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("classifier", classifier),
                ])
                for name, classifier in classifiers.items()
            }
            cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            selection_scores = compare_classifiers(
                candidate_pipelines, X_train, y_train, cross_validation
            )
            selected_name = max(selection_scores, key=selection_scores.get)
            selected_pipeline = candidate_pipelines[selected_name].fit(X_train, y_train)
            metrics = evaluate_classifier(selected_pipeline, X_test, y_test)

            save_object(
                self.model_trainer_config.trained_model_file_path,
                selected_pipeline.named_steps["classifier"],
            )
            save_object(
                self.model_trainer_config.preprocessor_file_path,
                selected_pipeline.named_steps["preprocessor"],
            )
            result = {"model_name": selected_name, **metrics}
            logging.info(
                "Selected %s by cross-validated ROC-AUC %.4f; held-out metrics: %s",
                selected_name,
                selection_scores[selected_name],
                metrics,
            )
            return result
        except Exception as error:
            raise CustomException(error, sys) from error