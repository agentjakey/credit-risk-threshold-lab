from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.config import RANDOM_STATE


def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def build_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def get_predicted_probabilities(model, X_test):
    return model.predict_proba(X_test)[:, 1]
