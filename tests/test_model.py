from sklearn.linear_model import LogisticRegression

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.pipeline.predict_pipeline import PredictPipeline
from src.schema import FEATURE_COLUMNS
from src.utils import load_object, save_object


def test_model_artifacts_load_and_predict(tmp_path, telco_data):
    features, target = DataTransformation._split_features_and_target(telco_data)
    preprocessor = DataTransformation().get_data_transformer_object()
    transformed = preprocessor.fit_transform(features)
    model = LogisticRegression(max_iter=500, random_state=42).fit(transformed, target)
    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    save_object(model_path, model)
    save_object(preprocessor_path, preprocessor)

    prediction = PredictPipeline(model_path, preprocessor_path).predict(
        telco_data.loc[[0], FEATURE_COLUMNS]
    )

    assert load_object(model_path) is not None
    assert prediction["prediction"] in {0, 1}
    assert 0 <= prediction["churn_probability"] <= 1
    assert prediction["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_model_trainer_compares_classifiers_by_roc_auc(tmp_path, telco_data):
    train_data = telco_data.iloc[:32]
    test_data = telco_data.iloc[32:]
    feature_transformer = DataTransformation()
    X_train, y_train = feature_transformer._split_features_and_target(train_data)
    X_test, y_test = feature_transformer._split_features_and_target(test_data)
    metrics = ModelTrainer(
        ModelTrainerConfig(tmp_path / "model.pkl", tmp_path / "preprocessor.pkl")
    ).initiate_model_trainer(
        X_train, y_train, X_test, y_test,
        feature_transformer.get_data_transformer_object(),
    )

    assert metrics["model_name"] in {
        "LogisticRegression", "RandomForestClassifier", "XGBClassifier"
    }
    assert set(metrics) == {
        "model_name", "roc_auc", "f1", "precision", "recall", "accuracy"
    }
    assert 0 <= metrics["roc_auc"] <= 1