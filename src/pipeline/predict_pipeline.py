import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DiamondFeatureEngineer


class PredictPipeline:
    def __init__(self):
        self.model_path        = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'

    def predict(self, features):
        try:
            logging.info("Prediction pipeline started")

            # Load artifacts
            model        = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            logging.info("Model and preprocessor loaded")

            # Feature engineering first
            # Replicate notebook: rename x,y,z + create volume
            engineer = DiamondFeatureEngineer()
            features_engineered = engineer.transform(features)
            logging.info("Feature engineering applied")

            # Drop price if exists
            if 'price' in features_engineered.columns:
                features_engineered = features_engineered.drop(columns=['price'])

            # Preprocess input
            data_transformed = preprocessor.transform(features_engineered)
            logging.info("Input data transformed")

            # Predict
            pred_log = model.predict(data_transformed)

            # Reverse log transform → actual price
            pred_actual = np.expm1(pred_log)
            logging.info(f"Prediction complete: ${pred_actual[0]:,.2f}")

            return pred_actual

        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class DiamondData:
    """Maps incoming request data to model features"""
    carat   : float
    cut     : str
    color   : str
    clarity : str
    depth   : float
    table   : float
    x       : float
    y       : float
    z       : float

    def get_data_as_dataframe(self):
        try:
            data = {
                'carat'  : [self.carat],
                'cut'    : [self.cut],
                'color'  : [self.color],
                'clarity': [self.clarity],
                'depth'  : [self.depth],
                'table'  : [self.table],
                'x'      : [self.x],
                'y'      : [self.y],
                'z'      : [self.z],
            }
            df = pd.DataFrame(data)
            logging.info(f"Input DataFrame created:\n{df}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample = DiamondData(
        carat   = 0.23,
        cut     = 'Ideal',
        color   = 'E',
        clarity = 'SI2',
        depth   = 61.5,
        table   = 55.0,
        x       = 3.95,
        y       = 3.98,
        z       = 2.43
    )

    df = sample.get_data_as_dataframe()
    print("Input:")
    print(df)

    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    print(f"\nPredicted Price: ${result[0]:,.2f}")
    print(f"Actual Price:    $326.00")