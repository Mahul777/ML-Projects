# 🔧 Purpose:
# Sets up a logging system for your entire project to:
# Track what happens when your code runs (info, errors, debug)
# Save logs to timestamped files inside a logs/ folder
# Help you debug problems and keep a history of runtime events

import logging
import os
from datetime import datetime

# 📌 Create log file with current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 📁 Create logs/ folder if not exists
# Constructs the full path to the log file inside a logs/ directory in the current 
# working directory.
# Creates the logs/ directory if it doesn't already exist 
# (exist_ok=True prevents errors if it does).
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok=True)

# 🧾 Configure logger
# All logs (INFO level ) are written to the specified file.
logging.basicConfig(
    filename=logs_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started.")
