import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

import evaluate_models
import feature_engineering
import feature_importance
import forecast
import hybrid_models
import inspect_data
import preprocess
import time_series_models
import train_models
import tune_models


def write_reproducibility_summary():
    mld = pd.read_csv(os.path.join(BASE_DIR, "outputs", "ml_dataset.csv"))
    mc = pd.read_csv(os.path.join(BASE_DIR, "outputs", "model_comparison.csv"))
    with open(os.path.join(BASE_DIR, "models", "best_params.json")) as f:
        best_params = json.load(f)
    lines = [
        "=== ML Pipeline Reproducibility Summary ===",
        f"Dataset path: {os.path.join(BASE_DIR, 'NDAP_REPORT_9259.csv')}",
        f"ML dataset dimensions: {mld.shape[0]} rows x {mld.shape[1]} columns",
        f"Year range (predictions): {mld['YearCode'].min()} - {mld['YearCode'].max()}",
        f"States: {mld['State'].nunique()}",
        f"Features used: {', '.join(sorted(c for c in mld.columns if c not in ['State', 'StateCode', 'YearCode', 'Year', 'higher_education_enrollment']))}",
        f"Target variable: higher_education_enrollment",
        f"Train years: 2014-2017",
        f"Test years: 2018-2019",
        f"Missing-value strategy: median imputation for numeric, most_frequent for state, fitted on training data only inside sklearn Pipeline",
        f"Models: Linear Regression, SVR, Random Forest, XGBoost, LightGBM, WOASVR, GWOSVR, PSOSVR, ActualGRIDSVR, ARIMA, ETS, TBATS",
        f"Hyperparameters (best from temporal CV):",
    ]
    for name, params in best_params.items():
        clean = {k.replace("model__", ""): v for k, v in params.items()}
        lines.append(f"  {name}: {clean}")
    lines.append("Evaluation results (test set):")
    lines.append(mc.to_string(index=False))
    fs = os.path.join(BASE_DIR, "outputs", "enrollment_forecasts.csv")
    if os.path.exists(fs):
        fsum = pd.read_csv(fs)
        base = fsum[fsum["Scenario"] == "Base_2019_Ratio"]
        y2022 = base.loc[base["Year"] == 2022, "Forecast_Enrollment"].sum()
        y2019 = float(
            pd.read_csv(os.path.join(BASE_DIR, "outputs", "ml_dataset.csv"))[
                pd.read_csv(os.path.join(BASE_DIR, "outputs", "ml_dataset.csv"))["YearCode"] == 2019
            ]["higher_education_enrollment"].sum()
        )
        change = (y2022 / y2019 - 1) * 100
        n_scen = fsum["Scenario"].nunique()
        lines.append(
            f"Two-stage projection 2020-2024 (see outputs/enrollment_forecasts.csv, "
            f"outputs/future_indicator_forecasts.csv, outputs/forecast_methodology.txt): "
            f"{n_scen} ratio scenarios; primary projection 2020-2022; "
            f"aggregate base 2022 change vs 2019: {change:.1f}% "
            f"(Stage 1: per-State linear trends for 6 count indicators; "
            f"Stage 2: XGBoost retrained on all 2014-2019 data, recursive forecast; "
            f"GER/GPI/PTRs held at 2019 baseline as a scenario)"
        )
    with open(os.path.join(BASE_DIR, "outputs", "pipeline_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nReproducibility summary written to outputs/pipeline_summary.txt")


def main():
    print("=" * 60)
    print("STEP 1: Data understanding")
    print("=" * 60)
    inspect_data.main()

    print("\n" + "=" * 60)
    print("STEP 2: Data cleaning")
    print("=" * 60)
    preprocess.main()

    print("\n" + "=" * 60)
    print("STEP 3: Feature engineering (State-Year ML dataset)")
    print("=" * 60)
    feature_engineering.main()

    print("\n" + "=" * 60)
    print("STEP 4: Baseline models")
    print("=" * 60)
    train_models.run_baseline()

    print("\n" + "=" * 60)
    print("STEP 5: Hyperparameter tuning (temporal CV)")
    print("=" * 60)
    tune_models.tune()

    print("\n" + "=" * 60)
    print("STEP 6: Final evaluation and model selection")
    print("=" * 60)
    evaluate_models.main()

    print("\n" + "=" * 60)
    print("STEP 6.5: Hybrid models (metaheuristic + grid SVR tuning)")
    print("=" * 60)
    hybrid_models.main()

    print("\n" + "=" * 60)
    print("STEP 6.6: Time-series models (ARIMA, ETS, TBATS)")
    print("=" * 60)
    time_series_models.main()

    print("\n" + "=" * 60)
    print("STEP 7: Feature importance")
    print("=" * 60)
    feature_importance.main()

    print("\n" + "=" * 60)
    print("STEP 8: Two-stage 2020-2024 enrollment projection")
    print("=" * 60)
    forecast.main()

    write_reproducibility_summary()

    print("\nPipeline complete. Outputs are in outputs/ and models/.")


if __name__ == "__main__":
    main()
