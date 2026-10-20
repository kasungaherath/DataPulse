"""Leakage-safe preprocessing and churn model comparison."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)])


def candidate_models(X, random_state=42):
    preprocessor = build_preprocessor(X)
    return {
        "Logistic Regression": Pipeline([("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))]),
        "Random Forest": Pipeline([("prep", preprocessor),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                class_weight="balanced", random_state=random_state, n_jobs=1))]),
        "HistGradientBoosting": Pipeline([("prep", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                l2_regularization=1.0, random_state=random_state))]),
    }

