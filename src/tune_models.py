import json
import os
import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from config import MODELS_DIR, OUTPUTS_DIR
from modeling_utils import FEATURES, TARGET, build_model_pipeline, split_chronologically

BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")

SEARCH = {
    "SVR": {
        "model": SVR(),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.05, 0.1, 0.5],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale"],
        },
        "n_iter": 12,
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.5, "sqrt"],
        },
        "n_iter": 15,
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
        "n_iter": 20,
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [-1, 5, 10],
            "model__min_child_samples": [5, 10, 20],
        },
        "n_iter": 20,
    },
}


def tune():
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "ml_dataset.csv"))
    train, _ = split_chronologically(df)
    X, y = train[FEATURES], train[TARGET]
    years = sorted(train["YearCode"].unique())
    n_splits = max(2, len(years) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_params = {}
    for name, spec in SEARCH.items():
        pipe = build_model_pipeline(spec["model"])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=spec["params"],
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_iter=spec["n_iter"],
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        best_params[name] = search.best_params_
        print(f"{name}: best CV RMSE = {-search.best_score_:.2f}")
        print(f"  params: {search.best_params_}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\nBest params saved to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    tune()
