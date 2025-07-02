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
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

# 🚀 Save any Python object to a file using pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ✅ Evaluate multiple models and return their R² scores
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params.get(model_name, {})

            gs = GridSearchCV(model, param, cv=3, verbose=1, n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score
            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)


# ✅ Load a saved object from file
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

