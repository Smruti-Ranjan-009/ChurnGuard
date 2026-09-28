import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


def test_ingestion_validates_and_splits_telco_data(tmp_path, telco_data):
    source_path = tmp_path / "telco.csv"
    train_path, test_path, raw_path = (
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        tmp_path / "data.csv",
    )
    telco_data.to_csv(source_path, index=False)
    config = DataIngestionConfig(source_path, train_path, test_path, raw_path)

    returned_train, returned_test = DataIngestion(config).initiate_data_ingestion()

    train_data = pd.read_csv(returned_train)
    assert train_data["Churn"].value_counts(normalize=True).to_dict() == {0: 0.5, 1: 0.5}
    assert train_data["TotalCharges"].dtype.kind in "fi"
    assert len(pd.read_csv(returned_test)) == 8
    assert raw_path.is_file()


def test_dataset_path_can_be_overridden_with_environment(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    monkeypatch.setenv("CHURN_DATA_PATH", str(dataset_path))

    config = DataIngestionConfig()

    assert config.data_path == dataset_path