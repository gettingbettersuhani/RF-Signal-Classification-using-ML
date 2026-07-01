"""
Project Configuration File

All paths used in the project are defined here.
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Results
RESULTS_PATH = PROJECT_ROOT / "results"
FIGURE_PATH = RESULTS_PATH / "figures"
REPORT_PATH = RESULTS_PATH / "reports"
FEATURE_PATH = RESULTS_PATH / "features"

# Models
MODEL_PATH = PROJECT_ROOT / "models"

# Dataset
DATASET_PATH = RAW_DATA_PATH / "RML2016.10a_dict.pkl"

# Create folders automatically
for folder in [
    RESULTS_PATH,
    FIGURE_PATH,
    REPORT_PATH,
    FEATURE_PATH,
    MODEL_PATH,
]:
    folder.mkdir(parents=True, exist_ok=True)