from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"

REPORT_DIR = PROJECT_ROOT / "report"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "Churn"


def create_project_directories() -> None:
    """
    Ensure all required project directories exist before running analysis.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        METRICS_DIR,
        REPORT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)