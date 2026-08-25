import os

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import OUTPUTS_DIR
from modeling_utils import (
    FEATURES,
    TARGET,
    build_model_pipeline,
    compute_metrics,
    split_chronologically,
)

ML_DATASET_PATH = os.path.join(OUTPUTS_DIR, "ml_dataset.csv")

BASELINE_MODELS = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
}


def run_baseline():
    df = pd.read_csv(ML_DATASET_PATH)
    train, test = split_chronologically(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = {}
    predictions = {}
    for name, model in BASELINE_MODELS.items():
        pipe = build_model_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        results[name] = compute_metrics(y_test, pred)
        print(f"{name}: fitted")

    comparison = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    comparison = comparison[["Model", "MAE", "RMSE", "MAPE", "R2"]]
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    comparison.to_csv(os.path.join(OUTPUTS_DIR, "baseline_comparison.csv"), index=False)

    pred_df = test[["State", "YearCode", TARGET]].copy()
    pred_df = pred_df.rename(columns={"YearCode": "Year", TARGET: "Actual Enrollment"})
    for name in BASELINE_MODELS:
        pred_df[f"{name} Prediction"] = predictions[name]
    pred_df.to_csv(os.path.join(OUTPUTS_DIR, "baseline_predictions.csv"), index=False)

    print("\n=== Baseline comparison (test years {} - {}) ===".format(min(pred_df["Year"]), max(pred_df["Year"])))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    run_baseline()
