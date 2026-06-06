import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, RANDOM_STATE, TEST_SIZE


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    # Drop any non-numeric columns that were not explicitly one-hot encoded.
    # Keeps the target column regardless of dtype.
    non_numeric = [
        col for col in df.columns
        if df[col].dtype == object and col != TARGET_COL
    ]
    if non_numeric:
        df = df.drop(columns=non_numeric)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def generate_synthetic_dataset(n_samples: int = 5000, imbalance_ratio: float = 0.12) -> pd.DataFrame:
    from sklearn.datasets import make_classification
    import numpy as np

    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0.01,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df[TARGET_COL] = y
    return df
