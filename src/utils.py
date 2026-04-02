import os
import pickle
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys


def save_object(file_path, obj):
    """Save any object (model, preprocessor) to a pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load any pickle object (model, preprocessor) from file"""
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models
    Returns a report dict with R2 scores
    """
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            report[name] = r2

            logging.info(f"{name} → R2: {r2:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)