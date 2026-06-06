import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, RANDOM_STATE, TEST_SIZE, FEATURE_NAMES


def load_uci_credit_dataset() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    X = X.rename(columns=FEATURE_NAMES)

    df = X.copy()
    df[TARGET_COL] = y.iloc[:, 0].values
    return df


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    dupes_dropped = before - len(df)
    before = len(df)
    df = df.dropna()
    nulls_dropped = before - len(df)
    return df, dupes_dropped, nulls_dropped


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def compute_scale_pos_weight(y_train) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return round(n_neg / n_pos, 4)
