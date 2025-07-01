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
