import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, TEST_YEARS
from modeling_utils import (
    FEATURES,
    TARGET,
    build_model_pipeline,
    compute_metrics,
    split_chronologically,
)

MODEL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "predictions.csv")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")

MODEL_ORDER = [
    "Linear Regression",
    "SVR",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]

DEFAULT_PARAMS = {
    "Linear Regression": {},
    "SVR": {},
    "Random Forest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "n_jobs": -1, "verbosity": 0},
    "LightGBM": {"n_estimators": 100, "random_state": 42, "verbose": -1, "n_jobs": -1},
}

MODEL_FACTORY = {
    "Linear Regression": LinearRegression,
    "SVR": SVR,
    "Random Forest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
}


def load_best_params():
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return {}


def strip_model_prefix(params):
    out = {}
    for k, v in params.items():
        key = k.replace("model__", "")
        if key.startswith("lgbmclassifier") or key.startswith("xgbclassifier"):
            continue
        out[key] = v
    return out


def build_models(best_params):
    models = {}
    for name in MODEL_ORDER:
        params = {**DEFAULT_PARAMS[name], **strip_model_prefix(best_params.get(name, {}))}
        models[name] = MODEL_FACTORY[name](**params)
    return models


def select_best_model(comparison):
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    rank_df = comparison.copy()
    rank_df["R2"] = -rank_df["R2"]
    for m in metrics:
        rank_df[f"{m}_rank"] = rank_df[m].rank(method="min")
    rank_df["avg_rank"] = rank_df[[f"{m}_rank" for m in metrics]].mean(axis=1)
    best = rank_df.loc[rank_df["avg_rank"].idxmin()]
    return best["Model"], rank_df


def plot_actual_trend(df):
    trend = df.groupby("YearCode")[TARGET].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(trend.index, trend.values, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Average higher education enrollment")
    plt.title("Actual enrollment trend over time (mean across states)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "actual_trend.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(test, predictions, pred_name_map):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, name in enumerate(MODEL_ORDER):
        ax = axes[i]
        actual = test[TARGET].values
        pred = predictions[name]
        lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.scatter(actual, pred, s=20, alpha=0.6)
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("Actual vs Predicted enrollment (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(comparison):
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        ax.bar(comparison["Model"], comparison[metric], color="steelblue")
        ax.set_title(f"{metric} by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metric_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "ml_dataset.csv"))
    train, test = split_chronologically(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    best_params = load_best_params()
    models = build_models(best_params)

    predictions = {}
    fitted_pipes = {}
    for name, model in models.items():
        pipe = build_model_pipeline(model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        fitted_pipes[name] = pipe
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.joblib"))
        print(f"{name}: trained, test {compute_metrics(y_test, pred)}")

    comparison = pd.DataFrame({name: compute_metrics(y_test, predictions[name]) for name in MODEL_ORDER}).T
    comparison = comparison[["MAE", "RMSE", "MAPE", "R2"]]
    comparison = comparison.reset_index().rename(columns={"index": "Model"})
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    pred_df = test[["State", "YearCode", TARGET]].copy()
    pred_df = pred_df.rename(columns={"YearCode": "Year", TARGET: "Actual Enrollment"})
    for name in MODEL_ORDER:
        pred_df[f"{name} Prediction"] = predictions[name]
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    best_model, ranked = select_best_model(comparison)
    reason_parts = []
    row = comparison[comparison["Model"] == best_model].iloc[0]
    reason_parts.append(f"MAE={row['MAE']:.0f}")
    reason_parts.append(f"RMSE={row['RMSE']:.0f}")
    reason_parts.append(f"MAPE={row['MAPE']:.2f}%")
    reason_parts.append(f"R2={row['R2']:.4f}")
    selection = (
        f"Best-performing model: {best_model}\n"
        f"Reason: Lowest average ranking across MAE, RMSE, MAPE and R2 on the "
        f"chronological test set (years {min(TEST_YEARS)}-{max(TEST_YEARS)}). "
        f"Metrics: {', '.join(reason_parts)}.\n"
    )
    with open(os.path.join(OUTPUTS_DIR, "model_selection.txt"), "w") as f:
        f.write(selection)
    print("\n" + selection)

    plot_actual_trend(df)
    plot_actual_vs_predicted(test, predictions, MODEL_ORDER)
    plot_metric_bars(comparison)

    print(f"\nModel comparison saved to {MODEL_COMPARISON_PATH}")
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    print(f"Trained models saved to {MODELS_DIR}")
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
