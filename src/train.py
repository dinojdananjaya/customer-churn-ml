import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_PATH, RANDOM_STATE
from preprocessing import prepare_model_inputs
from utils import load_processed_data, save_dataframe, setup_logging


def build_model_candidates(preprocessor):
    """
    Define candidate machine learning pipelines.
    """
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def build_parameter_grids():
    """
    Define focused hyperparameter grids for model tuning.
    """
    return {
        "logistic_regression": {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [None, 8, 12],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": [None, "balanced"],
        },
        "xgboost": {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on the unseen test set.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_tune_models():
    """
    Train, tune, evaluate, and save candidate classification models.
    """
    setup_logging()

    df = load_processed_data(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = prepare_model_inputs(df)

    models = build_model_candidates(preprocessor)
    parameter_grids = build_parameter_grids()

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []
    trained_models = {}

    for model_name, pipeline in models.items():
        logging.info("Training model: %s", model_name)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[model_name],
            scoring="roc_auc",
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        test_metrics = evaluate_model(best_model, X_test, y_test)

        result_row = {
            "model": model_name,
            "best_cv_roc_auc": grid_search.best_score_,
            **test_metrics,
            "best_parameters": grid_search.best_params_,
        }

        results.append(result_row)
        trained_models[model_name] = best_model

        model_path = MODELS_DIR / f"{model_name}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)

        logging.info("Saved trained model to %s", model_path)
        logging.info("%s test ROC-AUC: %.4f", model_name, test_metrics["roc_auc"])

    results_df = pd.DataFrame(results).sort_values(
        by="roc_auc",
        ascending=False
    )

    save_dataframe(results_df, METRICS_DIR / "model_comparison.csv")

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    logging.info("Best model selected: %s", best_model_name)
    logging.info("Training and tuning completed successfully.")


if __name__ == "__main__":
    train_and_tune_models()