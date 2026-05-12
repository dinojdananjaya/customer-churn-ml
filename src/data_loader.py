from pathlib import Path
import logging
import pandas as pd

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place the CSV file in data/raw/."
        )

    df = pd.read_csv(path)
    logging.info("Raw dataset loaded successfully. Shape: %s", df.shape)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the raw dataset contains the expected columns.
    """
    missing_columns = EXPECTED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"The dataset is missing expected columns: {sorted(missing_columns)}"
        )

    logging.info("Schema validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset before exploratory analysis and modelling.
    """
    cleaned_df = df.copy()

    initial_shape = cleaned_df.shape

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors="coerce"
    )

    missing_total_charges = cleaned_df["TotalCharges"].isna().sum()
    cleaned_df = cleaned_df.dropna(subset=["TotalCharges"])

    duplicate_rows = cleaned_df.duplicated().sum()
    cleaned_df = cleaned_df.drop_duplicates()

    if "customerID" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["customerID"])

    cleaned_df["Churn"] = cleaned_df["Churn"].map({"Yes": 1, "No": 0})

    if cleaned_df["Churn"].isna().any():
        raise ValueError("Target variable contains unmapped values after encoding.")

    logging.info("Data cleaning completed.")
    logging.info("Initial shape: %s", initial_shape)
    logging.info("Rows removed due to missing TotalCharges: %s", missing_total_charges)
    logging.info("Duplicate rows removed: %s", duplicate_rows)
    logging.info("Final shape: %s", cleaned_df.shape)

    return cleaned_df


def save_processed_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH
) -> None:
    """
    Save the cleaned dataset to the processed data folder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    logging.info("Processed dataset saved to %s", path)


def prepare_dataset() -> pd.DataFrame:
    """
    Run the full data ingestion and cleaning stage.
    """
    raw_df = load_raw_data()
    validate_schema(raw_df)

    cleaned_df = clean_data(raw_df)
    save_processed_data(cleaned_df)

    return cleaned_df


if __name__ == "__main__":
    prepare_dataset()