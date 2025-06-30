# 🔧 Purpose:
# This file acts as the main controller that links together:
# Data Ingestion
# Data Transformation
# Model Training
# Think of it as the "director" calling all the "actors" (components) to perform their roles in order.

# | File                | Role                                                               |
# | ------------------- | ------------------------------------------------------------------ |
# | `train_pipeline.py` | Runs the full pipeline (ingest → transform → train → save)         |

# 📥 1. Ingest (Data Ingestion)
# 👉 What it means:
# Bringing raw data into your pipeline from a source.

# 🔧 2. Transform (Data Transformation)
# 👉 What it means:
# Prepare and clean the data for machine learning.
