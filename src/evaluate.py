import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_PATH
from preprocessing import prepare_model_inputs
from utils import load_processed_data, save_dataframe, setup_logging
from visualisation import save_figure, set_plot_style


def load_best_model():
    """
    Load the best trained model selected during the training stage.
    """
    model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Best model not found. Run src/train.py before evaluation."
        )

    return joblib.load(model_path)


def save_classification_report(model, X_test, y_test) -> None:
    """
    Save precision, recall, F1-score, and support for each class.
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No churn", "Churn"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    save_dataframe(report_df, METRICS_DIR / "classification_report.csv")


def plot_confusion_matrix(model, X_test, y_test) -> None:
    """
    Save the confusion matrix for the final selected model.
    """
    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"]
    )
    plt.title("Confusion Matrix for Final Model")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    save_figure(FIGURES_DIR / "final_confusion_matrix.png")


def plot_roc_curve(model, X_test, y_test) -> None:
    """
    Save ROC curve for the final selected model.
    """
    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve for Final Model")
    save_figure(FIGURES_DIR / "final_roc_curve.png")


def plot_precision_recall_curve(model, X_test, y_test) -> None:
    """
    Save precision-recall curve for the final selected model.
    """
    plt.figure()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve for Final Model")
    save_figure(FIGURES_DIR / "final_precision_recall_curve.png")


def save_prediction_examples(model, X_test, y_test) -> None:
    """
    Save a small sample of prediction probabilities for report discussion.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    examples = X_test.copy()
    examples["actual_churn"] = y_test.values
    examples["predicted_churn"] = y_pred
    examples["predicted_churn_probability"] = y_proba

    examples = examples.sort_values(
        by="predicted_churn_probability",
        ascending=False
    )

    save_dataframe(
        examples.head(20),
        METRICS_DIR / "high_risk_prediction_examples.csv"
    )


def run_evaluation() -> None:
    """
    Run the final model evaluation workflow.
    """
    setup_logging()
    set_plot_style()

    logging.info("Starting final model evaluation.")

    df = load_processed_data(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test, _ = prepare_model_inputs(df)

    model = load_best_model()

    save_classification_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall_curve(model, X_test, y_test)
    save_prediction_examples(model, X_test, y_test)

    logging.info("Final model evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()