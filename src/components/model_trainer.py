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
    # Stores the file path where the trained model will be saved
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        # intializing the config
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Split features and target
            
            #All columns except the last one (:-1) are features (input variables).
            # The last column (-1) is the target (what you want to predict, e.g., math_score).
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

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Find best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found (R2 < 0.6)", sys)

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and return final R2 score
            y_pred = best_model.predict(X_test)
            final_r2_score = r2_score(y_test, y_pred)

            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)
