# 🔧 Purpose:
# Defines a custom exception class to improve error handling by:
# Capturing detailed error info such as file name and line number where the error occurred
# Providing a clearer, more informative error message
# Making debugging easier during development and production

# | File           | Role                                                |
# | -------------- | --------------------------------------------------- |
# | `exception.py` | Custom exception class with detailed traceback info |

import sys
import os

# Builds a detailed error message including:
#   - The file name where the error occurred
#   - The line number of the error
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script: [{0}] line: [{1}] message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

#Defines a custom exception that provides more informative error messages
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

