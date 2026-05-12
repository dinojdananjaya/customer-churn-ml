import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_PATH
from preprocessing import prepare_model_inputs
from utils import load_processed_data, save_dataframe, setup_logging
from visualisation import save_figure, set_plot_style


def load_best_model():
    model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Best model not found. Run src/train.py first.")

    return joblib.load(model_path)


def get_feature_names(model) -> list:
    """
    Extract transformed feature names from the preprocessing stage.
    """
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    return [name.replace("num__", "").replace("cat__", "") for name in feature_names]


def create_feature_importance_table(model) -> pd.DataFrame:
    """
    Create feature importance table for tree-based models.
    """
    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model)

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "The selected best model does not provide feature_importances_."
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """
    Save a feature importance plot for the final selected model.
    """
    top_features = importance_df.head(20).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_figure(FIGURES_DIR / "feature_importance_top_20.png")


def run_shap_analysis(model, X_test: pd.DataFrame) -> None:
    """
    Run SHAP analysis on a sample of the test set.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_test_sample = X_test.sample(
        n=min(500, len(X_test)),
        random_state=42
    )

    X_transformed = preprocessor.transform(X_test_sample)
    feature_names = get_feature_names(model)

    explainer = shap.Explainer(classifier, X_transformed, feature_names=feature_names)
    shap_values = explainer(X_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    save_figure(FIGURES_DIR / "shap_summary_plot.png")


def run_explainability() -> None:
    """
    Run final model interpretation workflow.
    """
    setup_logging()
    set_plot_style()

    logging.info("Starting model explainability analysis.")

    df = load_processed_data(PROCESSED_DATA_PATH)
    _, X_test, _, _, _ = prepare_model_inputs(df)

    model = load_best_model()

    try:
        importance_df = create_feature_importance_table(model)
        save_dataframe(importance_df, METRICS_DIR / "feature_importance.csv")
        plot_feature_importance(importance_df)
    except ValueError as error:
        logging.warning("Feature importance skipped: %s", error)

    run_shap_analysis(model, X_test)

    logging.info("Model explainability analysis completed successfully.")


if __name__ == "__main__":
    run_explainability()