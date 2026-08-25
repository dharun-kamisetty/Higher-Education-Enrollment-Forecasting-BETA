import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import CLEANED_PATH, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR
from evaluate_models import MODEL_COMPARISON_PATH, PREDICTIONS_PATH, select_best_model
from feature_engineering import EXCLUDED_STATES, pivot_features
from modeling_utils import TARGET, compute_metrics, split_chronologically

TS_MODELS = ["ARIMA", "ETS", "TBATS"]

TRAIN_MAX_YEAR = 2017
TEST_YEARS = [2018, 2019]

TS_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "time_series_comparison.csv")
TS_PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "time_series_predictions.csv")
TS_METHODS_PATH = os.path.join(OUTPUTS_DIR, "time_series_methods.csv")


def _patch_tbats_sklearn_compat():
    import sys

    import sklearn.utils.validation as suv

    if getattr(suv, "_force_all_finite_patched", False):
        return

    orig = suv.check_array

    def wrapper(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return orig(*args, **kwargs)

    wrapper.__name__ = orig.__name__
    suv.check_array = wrapper
    suv._force_all_finite_patched = True


_patch_tbats_sklearn_compat()


def forecast_series(values, model_name):
    values = np.asarray(values, dtype=float)
    last = float(values[-1])
    try:
        if model_name == "ARIMA":
            from pmdarima import auto_arima

            fit = auto_arima(
                values,
                start_p=0, max_p=2, d=1, max_d=1,
                start_q=0, max_q=2,
                seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                information_criterion="aic",
            )
            fc = np.asarray(fit.predict(n_periods=len(TEST_YEARS)), dtype=float)
        elif model_name == "ETS":
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel

            fit = ETSModel(values, error="add", trend="add", seasonal=None, damped_trend=False).fit(maxiter=500)
            fc = np.asarray(fit.forecast(len(TEST_YEARS)), dtype=float)
        elif model_name == "TBATS":
            from tbats import TBATS

            fit = TBATS(
                use_box_cox=None,
                use_trend=True,
                use_damped_trend=True,
                use_arma_errors=False,
                seasonal_periods=None,
                n_jobs=1,
            ).fit(values)
            fc = np.asarray(fit.forecast(steps=len(TEST_YEARS)), dtype=float)
        else:
            raise ValueError(f"unknown model {model_name}")
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return np.clip(fc, 0.0, None).tolist(), "fitted"
    except Exception:
        return [last, last], "last_observed_fallback"


def plot_comparison(comparison):
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
    fig.savefig(os.path.join(PLOTS_DIR, "time_series_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "ml_dataset.csv"))
    train, test = split_chronologically(df)
    test = test.sort_values(["State", "YearCode"]).reset_index(drop=True)
    y_test = test[TARGET].values

    hist = pivot_features(pd.read_csv(CLEANED_PATH))
    hist = hist[~hist["State"].str.upper().isin(EXCLUDED_STATES)]

    states = sorted(test["State"].unique())
    missing = [s for s in states if s not in set(hist["State"].unique())]
    if missing:
        raise ValueError(f"states missing from historical pivot: {missing}")

    predictions = {}
    metrics = {}
    for name in TS_MODELS:
        state_fc = {}
        method_rows = []
        for state in states:
            series = hist[hist["State"] == state].set_index("YearCode")["enrolment"].sort_index()
            series = series[(series.index >= 2013) & (series.index <= TRAIN_MAX_YEAR)].dropna()
            if len(series) < 2:
                fc, method = [float(series.iloc[-1])] * len(TEST_YEARS), "insufficient_history"
            else:
                fc, method = forecast_series(series.values, name)
            state_fc[state] = fc
            method_rows.append({"State": state, "Model": name, "Method": method})
        pred = np.array([state_fc[r.State][int(r.YearCode) - TEST_YEARS[0]] for r in test.itertuples()])
        predictions[name] = pred
        metrics[name] = compute_metrics(y_test, pred)
        pd.DataFrame(method_rows).to_csv(
            os.path.join(OUTPUTS_DIR, f"time_series_methods_{name.lower()}.csv"), index=False
        )
        print(f"{name}: test {metrics[name]}")
        print(f"  method counts: {pd.Series([r['Method'] for r in method_rows]).value_counts().to_dict()}")

    ts_comp = pd.DataFrame(metrics).T
    ts_comp = ts_comp[["MAE", "RMSE", "MAPE", "R2"]].reset_index().rename(columns={"index": "Model"})
    ts_comp.to_csv(TS_COMPARISON_PATH, index=False)

    ts_pred = test[["State", "YearCode", TARGET]].copy()
    ts_pred = ts_pred.rename(columns={"YearCode": "Year", TARGET: "Actual Enrollment"})
    for name in TS_MODELS:
        ts_pred[f"{name} Prediction"] = predictions[name]
    ts_pred.to_csv(TS_PREDICTIONS_PATH, index=False)

    methods = []
    for name in TS_MODELS:
        m = pd.read_csv(os.path.join(OUTPUTS_DIR, f"time_series_methods_{name.lower()}.csv"))
        methods.append(m)
    pd.concat(methods, ignore_index=True).to_csv(TS_METHODS_PATH, index=False)

    combined = ts_comp.copy()
    if os.path.exists(MODEL_COMPARISON_PATH):
        combined = pd.concat([pd.read_csv(MODEL_COMPARISON_PATH), ts_comp], ignore_index=True)
    combined.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nCombined model comparison ({len(combined)} models) saved to {MODEL_COMPARISON_PATH}")

    if os.path.exists(PREDICTIONS_PATH):
        all_pred = pd.read_csv(PREDICTIONS_PATH)
        for name in TS_MODELS:
            all_pred[f"{name} Prediction"] = predictions[name]
        all_pred.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Time-series predictions appended to {PREDICTIONS_PATH}")

    best_model, ranked = select_best_model(combined)
    row = combined[combined["Model"] == best_model].iloc[0]
    selection = (
        f"Best-performing model: {best_model}\n"
        f"Reason: Lowest average ranking across MAE, RMSE, MAPE and R2 on the "
        f"chronological test set (2018-2019). "
        f"Metrics: MAE={row['MAE']:.0f}, RMSE={row['RMSE']:.0f}, "
        f"MAPE={row['MAPE']:.2f}%, R2={row['R2']:.4f}.\n"
        f"Additional models: hybrid SVR (WOASVR, GWOSVR, PSOSVR metaheuristic tuning; "
        f"ActualGRIDSVR grid search) and per-state univariate time-series models "
        f"(ARIMA auto_arima, ETS additive trend, TBATS) trained on each State's "
        f"2013-2017 enrollment history and forecasting 2018-2019. See "
        f"outputs/hybrid_best_params.json, outputs/hybrid_optimization.csv and "
        f"outputs/time_series_methods.csv.\n"
    )
    with open(os.path.join(OUTPUTS_DIR, "model_selection.txt"), "w") as f:
        f.write(selection)
    print("\n" + selection)

    plot_comparison(combined)
    print(f"Plot saved: time_series_comparison.png")


if __name__ == "__main__":
    main()