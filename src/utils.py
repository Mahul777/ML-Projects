# 🔧 Purpose:
# Contains utility/helper functions that are:
# Reusable across multiple modules
# Perform small, common tasks that don’t belong to any specific component
# Keep your code DRY (Don't Repeat Yourself) and clean

# | File       | Role                                       |
# | ---------- | ------------------------------------------ |
# | `utils.py` | Helper functions shared across the project |


import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object to a specified file path using dill.

    Args:
        file_path (str): Destination file path (e.g., artifacts/preprocessor.pkl)
        obj (object): The object to save (e.g., preprocessor, model)

    Raises:
        CustomException: If saving fails
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file in write-binary mode and dump the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    """
    Train and evaluate multiple regression models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        models (dict): Dictionary of model_name: model_instance

    Returns:
        dict: Model name as key and R2 score as value
    """
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_pred_test)

            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
