# 🔧 Purpose:
# Handles everything related to loading raw data into your project.

# This could include:
# Reading CSV/JSON/parquet files
# Downloading from cloud (e.g., S3, GCS)
# Cleaning or splitting the dataset
# Saving intermediate data to local storage


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# ⚙️ Configuration class for setting paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# 📦 Main class to handle data ingestion
class DataIngestion:
    def __init__(self):   # Initializes the config with default paths.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # 📥 Step 1: Read the raw dataset
            df = pd.read_csv("notebook/data/stud.csv") #this line need to change if data may come from mongodb/sql/mysql 
            logging.info("Read the dataset as dataframe")

            # 📁 Step 2: Create the artifacts directory(folder) if not present
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 💾 Step 3: Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # ✂️ Step 4: Perform train-test split
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 💾 Step 5: Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # 📤 Step 6: Return file paths for next steps
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# 🧪 Run this file independently to test ingestion
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
