import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, PROCESSED_DATA_PATH, TABLES_DIR, TARGET_COLUMN
from utils import load_processed_data, save_dataframe, setup_logging
from visualisation import save_figure, set_plot_style


def summarise_dataset(df: pd.DataFrame) -> None:
    """
    Save high-level dataset summary tables for the report appendix.
    """
    overview = pd.DataFrame({
        "metric": [
            "rows",
            "columns",
            "duplicate_rows",
            "missing_values_total",
            "churn_rate"
        ],
        "value": [
            df.shape[0],
            df.shape[1],
            df.duplicated().sum(),
            df.isna().sum().sum(),
            round(df[TARGET_COLUMN].mean(), 4)
        ]
    })

    data_types = (
        df.dtypes
        .reset_index()
        .rename(columns={"index": "column", 0: "data_type"})
    )

    missing_values = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )

    save_dataframe(overview, TABLES_DIR / "dataset_overview.csv")
    save_dataframe(data_types, TABLES_DIR / "data_types.csv")
    save_dataframe(missing_values, TABLES_DIR / "missing_values.csv")


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of the target variable.
    """
    churn_counts = df[TARGET_COLUMN].value_counts().rename(index={0: "No churn", 1: "Churn"})

    plt.figure()
    sns.barplot(x=churn_counts.index, y=churn_counts.values)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn status")
    plt.ylabel("Number of customers")
    save_figure(FIGURES_DIR / "churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Plot key numerical feature distributions.
    """
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    for column in numeric_columns:
        plt.figure()
        sns.histplot(data=df, x=column, hue=TARGET_COLUMN, kde=True, bins=30)
        plt.title(f"Distribution of {column} by Churn")
        plt.xlabel(column)
        plt.ylabel("Customer count")
        save_figure(FIGURES_DIR / f"{column.lower()}_distribution_by_churn.png")


def plot_categorical_churn_rates(df: pd.DataFrame) -> None:
    """
    Plot churn rate across selected categorical variables.
    """
    categorical_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling"
    ]

    rows = []

    for column in categorical_columns:
        churn_rate_table = (
            df.groupby(column)[TARGET_COLUMN]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        churn_rate_table["feature"] = column
        rows.append(churn_rate_table)

        plt.figure()
        sns.barplot(data=churn_rate_table, x=column, y=TARGET_COLUMN)
        plt.title(f"Churn Rate by {column}")
        plt.xlabel(column)
        plt.ylabel("Churn rate")
        plt.xticks(rotation=30, ha="right")
        save_figure(FIGURES_DIR / f"churn_rate_by_{column.lower()}.png")

    combined_table = pd.concat(rows, ignore_index=True)
    save_dataframe(combined_table, TABLES_DIR / "categorical_churn_rates.csv")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot correlation matrix for numerical features.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Correlation Matrix for Numerical Features")
    save_figure(FIGURES_DIR / "numeric_correlation_matrix.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Plot boxplots to inspect potential outliers in numerical features.
    """
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    for column in numeric_columns:
        plt.figure()
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column)
        plt.title(f"{column} by Churn Status")
        plt.xlabel("Churn")
        plt.ylabel(column)
        save_figure(FIGURES_DIR / f"{column.lower()}_boxplot_by_churn.png")


def run_eda() -> None:
    """
    Run the full exploratory data analysis stage.
    """
    setup_logging()
    set_plot_style()

    logging.info("Starting exploratory data analysis.")

    df = load_processed_data(PROCESSED_DATA_PATH)

    summarise_dataset(df)
    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(df)
    plot_correlation_matrix(df)
    plot_boxplots(df)

    logging.info("Exploratory data analysis completed successfully.")


if __name__ == "__main__":
    run_eda()