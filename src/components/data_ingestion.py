import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read raw data
            df = pd.read_csv('data/Diamonds Prices2022.csv')
            logging.info(f"Dataset loaded — shape: {df.shape}")

            # Create artifacts/ directory
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )

            # Save raw data to artifacts
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved to artifacts/")

            # Train/test split — same as notebook (80/20, random_state=42)
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save splits to artifacts
            train_df.to_csv(
                self.ingestion_config.train_data_path, index=False
            )
            test_df.to_csv(
                self.ingestion_config.test_data_path, index=False
            )

            logging.info(f"Train set: {train_df.shape}")
            logging.info(f"Test set:  {test_df.shape}")
            logging.info("Data ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train path: {train_path}")
    print(f"Test path:  {test_path}")