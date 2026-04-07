import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training started")

            #  Define all models
            models = {
                'Linear Regression':  LinearRegression(),
                'Ridge':              Ridge(),
                'Lasso':              Lasso(),
                'Decision Tree':      DecisionTreeRegressor(random_state=42),
                'Random Forest':      RandomForestRegressor(
                                          n_estimators=100, random_state=42
                                      ),
                'Gradient Boosting':  GradientBoostingRegressor(
                                          random_state=42
                                      ),
                'XGBoost':            XGBRegressor(
                                          random_state=42, verbosity=0
                                      ),
            }

            #  Evaluate all models 
            logging.info("Evaluating all models...")
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            #  Print leaderboard
            print("\n" + "="*50)
            print("MODEL LEADERBOARD")
            print("="*50)
            sorted_models = sorted(
                model_report.items(), key=lambda x: x[1], reverse=True
            )
            for rank, (name, score) in enumerate(sorted_models, 1):
                print(f"{rank}. {name:<25} R2: {score:.4f}")

            #  Pick best baseline model
            best_model_name = sorted_models[0][0]
            best_model_score = sorted_models[0][1]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} — R2: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No model achieved R2 > 0.6", sys)

            # Train final tuned XGBoost
            # Best params from notebook RandomizedSearchCV
            logging.info("Training tuned XGBoost with best params from notebook...")

            tuned_xgb = XGBRegressor(
                subsample=0.7,
                n_estimators=500,
                min_child_weight=5,
                max_depth=8,
                learning_rate=0.1,
                gamma=0,
                colsample_bytree=0.7,
                random_state=42,
                verbosity=0
            )

            tuned_xgb.fit(X_train, y_train)
            y_pred_tuned = tuned_xgb.predict(X_test)

            # Evaluate tuned model
            r2_tuned   = r2_score(y_test, y_pred_tuned)
            rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
            mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)

            print("\n" + "="*50)
            print("TUNED XGBoost RESULTS")
            print("="*50)
            print(f"R2 Score : {r2_tuned:.4f}")
            print(f"RMSE     : {rmse_tuned:.4f}")
            print(f"MAE      : {mae_tuned:.4f}")

            logging.info(f"Tuned XGBoost — R2: {r2_tuned:.4f} RMSE: {rmse_tuned:.4f}")

            # Save tuned model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=tuned_xgb
            )

            print(f"\n✅ Model saved → {self.model_trainer_config.trained_model_file_path}")
            logging.info("Model saved to artifacts/model.pkl")

            return r2_tuned

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Step 1 — Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2 — Transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = \
        transformation.initiate_data_transformation(train_path, test_path)

    # Step 3 — Training
    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    print(f"\nFinal R2 Score: {r2:.4f}")