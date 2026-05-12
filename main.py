import logging
import subprocess
import sys

from src.config import create_project_directories
from src.utils import setup_logging


PIPELINE_STAGES = [
    ("Data loading and cleaning", "src/data_loader.py"),
    ("Exploratory data analysis", "src/eda.py"),
    ("Model training and tuning", "src/train.py"),
    ("Final model evaluation", "src/evaluate.py"),
    ("Model explainability", "src/explainability.py"),
]


def run_stage(stage_name: str, script_path: str) -> None:
    """
    Run one pipeline stage as a separate Python process.
    """
    logging.info("Starting stage: %s", stage_name)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline stage failed: {stage_name}")

    logging.info("Completed stage: %s", stage_name)


def run_pipeline() -> None:
    """
    Run the complete customer churn machine learning workflow.
    """
    setup_logging()
    create_project_directories()

    logging.info("Starting full customer churn ML pipeline.")

    for stage_name, script_path in PIPELINE_STAGES:
        run_stage(stage_name, script_path)

    logging.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()