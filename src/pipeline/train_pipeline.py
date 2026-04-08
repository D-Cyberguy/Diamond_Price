import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("="*50)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("="*50)

            # Stage 1: Data Ingestion
            logging.info("Stage 1: Data Ingestion")
            train_path, test_path = \
                self.data_ingestion.initiate_data_ingestion()

            # Stage 2: Data Transformation
            logging.info("Stage 2: Data Transformation")
            X_train, X_test, y_train, y_test, preprocessor_path = \
                self.data_transformation.initiate_data_transformation(
                    train_path, test_path
                )

            # Stage 3: Model Training
            logging.info("Stage 3: Model Training")
            r2_score = self.model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )

            logging.info("="*50)
            logging.info(f"TRAINING PIPELINE COMPLETE — R2: {r2_score:.4f}")
            logging.info("="*50)

            print("\n" + "="*50)
            print("TRAINING PIPELINE COMPLETE")
            print("="*50)
            print(f"Final R2 Score : {r2_score:.4f}")
            print(f"Model saved    : artifacts/model.pkl")
            print(f"Preprocessor   : artifacts/preprocessor.pkl")

            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
