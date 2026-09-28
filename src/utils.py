import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error


def load_object(file_path):
    try:
        with Path(file_path).open("rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error


def evaluate_classifier(model, features, target):
    try:
        predictions = model.predict(features)
        if hasattr(model, "predict_proba"):
            positive_class_index = list(model.classes_).index(1)
            positive_scores = model.predict_proba(features)[:, positive_class_index]
        else:
            positive_scores = model.decision_function(features)
        return {
            "roc_auc": float(roc_auc_score(target, positive_scores))
            if len(np.unique(target)) > 1 else 0.0,
            "f1": float(f1_score(target, predictions, zero_division=0)),
            "precision": float(precision_score(target, predictions, zero_division=0)),
            "recall": float(recall_score(target, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(target, predictions)),
        }
    except Exception as error:
        raise CustomException(error, sys) from error


def compare_classifiers(models, features, target, cross_validation):
    try:
        scores = {}
        for name, model in models.items():
            fold_scores = cross_val_score(
                model,
                features,
                target,
                scoring="roc_auc",
                cv=cross_validation,
                n_jobs=1,
                error_score="raise",
            )
            scores[name] = float(fold_scores.mean())
            logging.info("%s cross-validated ROC-AUC: %.4f", name, scores[name])
        return scores
    except Exception as error:
        raise CustomException(error, sys) from error
