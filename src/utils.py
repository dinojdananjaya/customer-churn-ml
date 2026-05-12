import json
import logging
from pathlib import Path

import pandas as pd


def setup_logging() -> None:
    """
    Configure a consistent logging format across project scripts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame as CSV and ensure the destination folder exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Saved table to %s", path)


def save_json(data: dict, path: Path) -> None:
    """
    Save dictionary data as a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    logging.info("Saved JSON output to %s", path)


def load_processed_data(path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by the data ingestion stage.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_loader.py first."
        )

    df = pd.read_csv(path)
    logging.info("Processed dataset loaded. Shape: %s", df.shape)

    return df