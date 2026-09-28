from src.components.data_transformation import DataTransformation
from src.schema import FEATURE_COLUMNS


def test_preprocessor_handles_valid_rows_and_unknown_categories(telco_data):
    transformation = DataTransformation()
    features, target = transformation._split_features_and_target(telco_data)
    preprocessor = transformation.get_data_transformer_object()
    transformed_train = preprocessor.fit_transform(features.iloc[:-1])
    unknown_features = features.iloc[[-1]].copy()
    unknown_features.loc[unknown_features.index[0], "InternetService"] = "Satellite"
    transformed_unknown = preprocessor.transform(unknown_features)

    assert transformed_train.shape[0] == len(target) - 1
    assert transformed_train.shape[1] == transformed_unknown.shape[1]
    assert transformed_unknown.shape[0] == 1
    assert list(features.columns) == FEATURE_COLUMNS