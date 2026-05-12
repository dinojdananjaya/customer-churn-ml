import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE, TARGET_COLUMN


def split_features_target(df: pd.DataFrame):
    """
    Separate input features from the target variable.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def identify_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical columns for preprocessing.
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build a preprocessing transformer for numerical and categorical features.

    Numerical features are standardised so models such as Logistic Regression
    can learn effectively. Categorical features are one-hot encoded to convert
    text categories into machine-readable numeric features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
):
    """
    Create a stratified train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )


def prepare_model_inputs(df: pd.DataFrame):
    """
    Prepare features, target, preprocessing pipeline, and train-test split.
    """
    X, y = split_features_target(df)

    numerical_features, categorical_features = identify_feature_types(X)

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    return X_train, X_test, y_train, y_test, preprocessor