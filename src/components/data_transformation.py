import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl'
    )


# Custom transformer: replicate notebook feature engineering
class DiamondFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Replicates all feature engineering steps from the notebook:
    1. Flag zeros as NaN in x, y, z
    2. Impute using carat-grouped median
    3. Rename x, y, z → zirconia_length, zirconia_width, zirconia_height
    4. Create volume feature
    5. Drop duplicates
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            df = X.copy()

            # Step 1 — Flag zeros as NaN
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].replace(0, np.nan)

            # Step 2 — Carat-grouped median imputation
            df['carat_bin'] = pd.cut(df['carat'], bins=10)
            for col in ['x', 'y', 'z']:
                df[col] = df.groupby(
                    'carat_bin', observed=True
                )[col].transform(
                    lambda g: g.fillna(g.median())
                )
                # Fallback: global median
                df[col] = df[col].fillna(df[col].median())
            df.drop(columns='carat_bin', inplace=True)

            # Step 3 — Rename x, y, z
            df.rename(columns={
                'x': 'zirconia_length',
                'y': 'zirconia_width',
                'z': 'zirconia_height'
            }, inplace=True)

            # Step 4 — Create volume feature
            df['volume'] = (
                df['zirconia_length'] *
                df['zirconia_width']  *
                df['zirconia_height'] *
                0.0061
            )

            # Step 5 — Drop duplicates
            df = df.drop_duplicates()

            return df

        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        """
        Builds the full sklearn preprocessor:
        - Ordinal encoding for cut, color, clarity
        - Standard scaling for all numeric features
        """
        try:
            # Ordinal orders — same as notebook
            cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_order   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            categorical_cols = ['cut', 'color', 'clarity']

            numerical_cols = [
                'carat', 'depth', 'table',
                'zirconia_length', 'zirconia_width',
                'zirconia_height', 'volume'
            ]

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('ordinal_encoder', OrdinalEncoder(
                    categories=[cut_order, color_order, clarity_order],
                    dtype=int
                ))
            ])

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Combine both
            preprocessor = ColumnTransformer(transformers=[
                ('cat', cat_pipeline, categorical_cols),
                ('num', num_pipeline, numerical_cols)
            ])

            logging.info("Preprocessor built successfully")
            logging.info(f"Categorical cols: {categorical_cols}")
            logging.info(f"Numerical cols:   {numerical_cols}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info(f"Train loaded: {train_df.shape}")
            logging.info(f"Test loaded:  {test_df.shape}")

            # Feature engineering
            engineer = DiamondFeatureEngineer()
            train_df = engineer.transform(train_df)
            test_df  = engineer.transform(test_df)
            logging.info("Feature engineering complete")

            # Log-transform target
            target_col = 'price'
            y_train = np.log1p(train_df[target_col])
            y_test  = np.log1p(test_df[target_col])

            X_train = train_df.drop(columns=[target_col])
            X_test  = test_df.drop(columns=[target_col])

            # Fit preprocessor
            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)
            logging.info("Preprocessing complete")

            # Save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Run ingestion first
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Run transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = \
        transformation.initiate_data_transformation(train_path, test_path)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Preprocessor:  {preprocessor_path}")