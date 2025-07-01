# | 🧩 Step               | 🔍 Description                                                              |
# | --------------------- | --------------------------------------------------------------------------- |
# | Data Cleaning         | Handle missing values, remove duplicates, fix datatypes, etc.               |
# | Feature Engineering   | Create new features from existing ones (e.g., extracting date parts)        |
# | Encoding              | Convert categorical data to numerical (e.g., OneHotEncoding, LabelEncoding) |
# | Scaling/Normalization | Standardize numeric features (e.g., using `StandardScaler`, `MinMaxScaler`) |
# | Train-Test Split      | Split the dataset into training and testing sets                            |
# | Save Preprocessor     | Save the transformation pipeline (usually with `joblib` or `pickle`)        |

import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # utility function to save pickle

@dataclass
class DataTransformationConfig:
    # Define where the preprocessor object(pipeline) will saved
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # Defines which columns are numerical and which are categorical
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")), #Imputes missing values with the median
                ("scaler", StandardScaler())
            ])
            logging.info("Numerical columns scaling pipeline created.")

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), #Imputes missing values with the most frequent value
                ("onehot", OneHotEncoder()), #One-hot encodes categorical variables /convert text into number
                ("scaler", StandardScaler(with_mean=False))  # for sparse matrix
            ])
            logging.info("Categorical columns encoding pipeline created.")

            # Column transformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Read the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data successfully loaded.")

            # Define target column
            target_column_name = "math_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data.")

            # Get preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Fit and transform on training data, only transform on test data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine features and targets
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saving the preprocessor object as pickle.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# If you have a column color with values red, blue, green, 
# one-hot encoding will create three new columns: color_red, color_blue, color_green.
# A row with color = blue will be encoded as [0, 1, 0].