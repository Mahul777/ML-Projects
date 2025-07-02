# 🔧 Purpose:
# This file trains the machine learning model using the transformed data, evaluates its performance,
#  and saves the final model.

# | Task 🧩                 | Description 📌                                                            |
# | ----------------------- | ------------------------------------------------------------------------- |
# | Accept Transformed Data | Takes preprocessed features and target from `data_transformation.py`.     |
# | Train Model             | Trains an ML model (e.g., Linear Regression, RandomForest, XGBoost, etc.) |
# | Evaluate Model          | Uses metrics like accuracy, F1 score, RMSE, R², etc.                      |
# | Save Trained Model      | Dumps the model object for future prediction (`model.pkl`)                |

import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    # Path to save the best trained model
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Split into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define candidate models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            # Define hyperparameter grid for tuning
            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                }
            }

            # Evaluate models
            model_report, best_models = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Identify best model and score
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found (R2 < 0.6)", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model to file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final model evaluation on test set
            y_pred = best_model.predict(X_test)
            final_r2_score = r2_score(y_test, y_pred)

            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)



