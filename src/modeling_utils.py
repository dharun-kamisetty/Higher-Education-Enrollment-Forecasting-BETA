import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TEST_YEARS

TARGET = "higher_education_enrollment"
CATEGORICAL = ["State"]

NUMERIC_FEATURES = [
    "Year",
    "enrollment_lag_1",
    "enrollment_lag_2",
    "enrollment_growth",
    "rolling_mean",
    "university_enrolment_lag1",
    "foreign_students_lag1",
    "stand_alone_institutions_lag1",
    "teachers_lag1",
    "colleges_lag1",
    "universities_lag1",
    "GPI_lag1",
    "GER_lag1",
    "PTR_all_lag1",
    "PTR_university_colleges_lag1",
    "PTR_university_units_lag1",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor():
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL),
        ]
    )


def build_model_pipeline(model):
    return Pipeline(steps=[("prep", build_preprocessor()), ("model", model)])


def get_transformed_feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def split_chronologically(df):
    train = df[df["YearCode"] < min(TEST_YEARS)].copy()
    test = df[df["YearCode"].isin(TEST_YEARS)].copy()
    return train, test


def aggregate_state_importance(importance_df):
    result = importance_df.copy()
    state_cols = [c for c in result["feature"] if c.startswith("State_")]
    other = result[~result["feature"].isin(state_cols)]
    if state_cols:
        state_sum = result.loc[result["feature"].isin(state_cols), "importance"].sum()
        other = pd.concat(
            [
                other,
                pd.DataFrame([{"feature": "State", "importance": state_sum}]),
            ],
            ignore_index=True,
        )
    return other.sort_values("importance", ascending=False).reset_index(drop=True)
