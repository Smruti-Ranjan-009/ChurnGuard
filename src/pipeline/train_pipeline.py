from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
	def run(self):
		train_path, test_path = DataIngestion().initiate_data_ingestion()
		X_train, y_train, X_test, y_test, preprocessor = (
			DataTransformation().initiate_data_transformation(train_path, test_path)
		)
		return ModelTrainer().initiate_model_trainer(
			X_train, y_train, X_test, y_test, preprocessor
		)


if __name__ == "__main__":
	print(TrainingPipeline().run())
