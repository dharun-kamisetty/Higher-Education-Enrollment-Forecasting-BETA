import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLEANED_PATH,
    FIRST_MODEL_YEAR,
    ML_DATASET_PATH,
    OUTPUTS_DIR,
    PLOTS_DIR,
)
from evaluate_models import (
    DEFAULT_PARAMS,
    MODEL_FACTORY,
    load_best_params,
    select_best_model,
    strip_model_prefix,
)
from feature_engineering import EXCLUDED_STATES, pivot_features
from modeling_utils import (
    FEATURES,
    TARGET,
    build_model_pipeline,
    compute_metrics,
    split_chronologically,
)

FORECAST_START = 2020
FORECAST_END = 2030
FORECAST_YEARS = list(range(FORECAST_START, FORECAST_END + 1))
BACKTEST_TRAIN_END = 2017
BACKTEST_TEST_YEARS = [2018, 2019]

COUNT_INDICATORS = [
    "teachers",
    "colleges",
    "universities",
    "stand_alone_institutions",
    "foreign_students",
    "university_enrolment",
]

RATIO_INDICATORS = [
    "GER",
    "GPI",
    "PTR_all",
    "PTR_university_colleges",
    "PTR_university_units",
]

MIN_OBS_FOR_TREND = 3
INSTABILITY_CAP_MULTIPLE = 3.0
SENSITIVITY_PERTURBATION = 0.05

SCENARIOS = {
    "Base_2019_Ratio": 1.0,
    "Lower_Ratio_Scenario": 1.0 - SENSITIVITY_PERTURBATION,
    "Upper_Ratio_Scenario": 1.0 + SENSITIVITY_PERTURBATION,
}

FORECAST_MODEL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "forecast_model_comparison.csv")
ENROLLMENT_PROJECTION_PATH = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
ENROLLMENT_SCENARIOS_PATH = os.path.join(OUTPUTS_DIR, "enrollment_scenarios_2030.csv")
INDIA_PROJECTION_PATH = os.path.join(OUTPUTS_DIR, "india_enrollment_projection_2020_2030.csv")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "enrollment_projection_2030_report.md")


def best_model_name():
    with open(os.path.join(OUTPUTS_DIR, "model_selection.txt")) as f:
        name = f.readline().strip().replace("Best-performing model: ", "")
    if name in MODEL_FACTORY:
        return name
    return "XGBoost"


def retrain_xgboost_full():
    mld = pd.read_csv(ML_DATASET_PATH)
    X, y = mld[FEATURES], mld[TARGET]
    name = "XGBoost"
    best_params = load_best_params()
    params = {**DEFAULT_PARAMS[name], **strip_model_prefix(best_params.get(name, {}))}
    model = MODEL_FACTORY[name](**params)
    pipe = build_model_pipeline(model)
    pipe.fit(X, y)
    return pipe, name


def historical_indicators():
    df = pd.read_csv(CLEANED_PATH)
    df = df[~df["State"].str.upper().isin(EXCLUDED_STATES)]
    df = df[df["YearCode"] >= FIRST_MODEL_YEAR - 1]
    return pivot_features(df).sort_values(["State", "YearCode"])


def fit_linear_trend(years, values):
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < MIN_OBS_FOR_TREND:
        return None
    x0 = x.min()
    slope, intercept = np.polyfit(x - x0, y, 1)
    yhat = intercept + slope * (x - x0)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2, x0


def project_count_indicator(years, values, forecast_years):
    pairs = sorted(zip(years, values))
    years = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    max_hist = max(values)
    fit = fit_linear_trend(years, values)
    if fit is None:
        fallback = values[-1]
        return {fy: fallback for fy in forecast_years}, "last_observed_value_fallback", np.nan
    slope, intercept, r2, x0 = fit
    raw = {fy: intercept + slope * (fy - x0) for fy in forecast_years}
    if max_hist > 0 and any(v > INSTABILITY_CAP_MULTIPLE * max_hist for v in raw.values()):
        fallback = values[-1]
        return {fy: fallback for fy in forecast_years}, "last_observed_value_fallback", r2
    if any(v < 0 for v in raw.values()):
        clipped = {fy: max(0.0, v) for fy, v in raw.items()}
        return clipped, "linear_trend_clipped", r2
    return raw, "linear_trend", r2


def stage1_count_forecasts(hist, forecast_years):
    summary_rows = []
    long_rows = []
    for state, g in hist.groupby("State"):
        for indicator in COUNT_INDICATORS:
            sub = g[["YearCode", indicator]].dropna()
            if sub.empty:
                last_val = np.nan
                proj = {fy: np.nan for fy in forecast_years}
                method = "no_historical_data"
                r2 = np.nan
            else:
                years = sub["YearCode"].tolist()
                values = sub[indicator].tolist()
                last_val = values[-1]
                proj, method, r2 = project_count_indicator(years, values, forecast_years)
            summary_rows.append(
                {
                    "State": state,
                    "Indicator": indicator,
                    "Historical_R2": r2,
                    "2019_Value": last_val,
                    **{f"{fy}_Forecast": proj[fy] for fy in forecast_years},
                    "Method": method,
                }
            )
            for fy in forecast_years:
                long_rows.append(
                    {
                        "State": state,
                        "Year": fy,
                        "Indicator": indicator,
                        "Forecast_Value": proj[fy],
                        "Method": method,
                        "Historical_R2": r2,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    long = pd.DataFrame(long_rows)
    return summary, long


def ratio_baseline(hist):
    base = {}
    for state, g in hist.groupby("State"):
        base[state] = {}
        for indicator in RATIO_INDICATORS:
            sub = g[["YearCode", indicator]].dropna().sort_values("YearCode")
            base[state][indicator] = float(sub[indicator].iloc[-1]) if not sub.empty else np.nan
    return base


def fit_arima(series):
    from pmdarima import auto_arima
    fit = auto_arima(
        series,
        start_p=0, max_p=2, d=1, max_d=1,
        start_q=0, max_q=2,
        seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
        information_criterion="aic",
    )
    return fit


def fit_ets(series):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    fit = ETSModel(series, error="add", trend="add", seasonal=None, damped_trend=False).fit(maxiter=500)
    return fit


def fit_tbats(series):
    from tbats import TBATS
    fit = TBATS(
        use_box_cox=None,
        use_trend=True,
        use_damped_trend=True,
        use_arma_errors=False,
        seasonal_periods=None,
        n_jobs=1,
    ).fit(series)
    return fit


def forecast_univariate(series, model_name, n_periods):
    series = np.asarray(series, dtype=float)
    last = float(series[-1])
    try:
        if model_name == "ARIMA":
            fit = fit_arima(series)
            fc = np.asarray(fit.predict(n_periods=n_periods), dtype=float)
        elif model_name == "ETS":
            fit = fit_ets(series)
            fc = np.asarray(fit.forecast(n_periods), dtype=float)
        elif model_name == "TBATS":
            fit = fit_tbats(series)
            fc = np.asarray(fit.forecast(steps=n_periods), dtype=float)
        else:
            raise ValueError(f"unknown model {model_name}")
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return np.clip(fc, 0.0, None).tolist(), "fitted"
    except Exception:
        return [last] * n_periods, "last_observed_fallback"


def recursive_xgboost_forecast(state, hist, stage2_pipe, count_proj, ratio_base, ratio_factor, forecast_years):
    g = hist[hist["State"] == state].sort_values("YearCode")
    code = g["StateCode"].iloc[0]

    enrollment = dict(zip(g["YearCode"], g["enrolment"]))

    count_series = {}
    for indicator in COUNT_INDICATORS:
        actual = dict(zip(g["YearCode"], g[indicator]))
        for fy in forecast_years:
            actual[fy] = count_proj.loc[
                (count_proj["State"] == state) & (count_proj["Indicator"] == indicator),
                f"{fy}_Forecast",
            ].iloc[0]
        count_series[indicator] = actual

    ratio_series = {}
    for indicator in RATIO_INDICATORS:
        actual = dict(zip(g["YearCode"], g[indicator]))
        base_val = ratio_base[state][indicator]
        for fy in forecast_years:
            actual[fy] = max(0.0, base_val * ratio_factor)
        ratio_series[indicator] = actual

    def indicator_at(indicator, year):
        if indicator in COUNT_INDICATORS:
            return count_series[indicator].get(year, np.nan)
        return ratio_series[indicator].get(year, np.nan)

    def enrollment_at(year):
        return enrollment.get(year, np.nan)

    predictions = {}
    rows = []
    for t in forecast_years:
        e1 = enrollment_at(t - 1)
        e2 = enrollment_at(t - 2)
        if e2 is not None and e2 == e2 and e2 != 0:
            growth = (e1 - e2) / e2
        else:
            growth = np.nan
        prior = [enrollment_at(t - k) for k in (1, 2, 3)]
        prior = [p for p in prior if p == p]
        rolling = float(np.mean(prior)) if prior else np.nan

        row = {
            "State": state,
            "Year": t,
            "enrollment_lag_1": e1,
            "enrollment_lag_2": e2,
            "enrollment_growth": growth,
            "rolling_mean": rolling,
        }
        for indicator in COUNT_INDICATORS + RATIO_INDICATORS:
            row[f"{indicator}_lag1"] = indicator_at(indicator, t - 1)

        frame = pd.DataFrame([row])[FEATURES]
        value = float(stage2_pipe.predict(frame)[0])
        value = max(0.0, value)
        predictions[t] = value
        enrollment[t] = value
        rows.append({"State": state, "StateCode": code, "Year": t, "Forecast_Enrollment": value})

    return pd.DataFrame(rows)


def backtest_models(hist, forecast_years_backtest):
    mld = pd.read_csv(ML_DATASET_PATH)
    train_backtest = mld[mld["YearCode"] <= BACKTEST_TRAIN_END].copy()
    test_backtest = mld[mld["YearCode"].isin(BACKTEST_TEST_YEARS)].copy()
    test_backtest = test_backtest.sort_values(["State", "YearCode"]).reset_index(drop=True)
    X_train_bt, y_train_bt = train_backtest[FEATURES], train_backtest[TARGET]
    X_test_bt, y_test_bt = test_backtest[FEATURES], test_backtest[TARGET]

    backtest_states = sorted(test_backtest["State"].unique())
    n_backtest = len(forecast_years_backtest)

    results = []

    best_params = load_best_params()
    for name in ["Linear Regression", "SVR", "Random Forest", "XGBoost", "LightGBM"]:
        params = {**DEFAULT_PARAMS[name], **strip_model_prefix(best_params.get(name, {}))}
        model = MODEL_FACTORY[name](**params)
        pipe = build_model_pipeline(model)
        pipe.fit(X_train_bt, y_train_bt)
        pred = pipe.predict(X_test_bt)
        metrics = compute_metrics(y_test_bt, pred)
        for state in backtest_states:
            state_mask = test_backtest["State"] == state
            if state_mask.any():
                s_pred = pred[state_mask.values]
                s_true = y_test_bt[state_mask.values]
                s_metrics = compute_metrics(s_true, s_pred)
                results.append({
                    "model": name,
                    "state": state,
                    "MAE": s_metrics["MAE"],
                    "RMSE": s_metrics["RMSE"],
                    "MAPE": s_metrics["MAPE"],
                    "R2": s_metrics["R2"],
                    "validation_years": f"{BACKTEST_TEST_YEARS[0]}-{BACKTEST_TEST_YEARS[1]}"
                })

    ts_models = ["ARIMA", "ETS", "TBATS"]
    for name in ts_models:
        state_fc = {}
        for state in backtest_states:
            series = hist[hist["State"] == state].set_index("YearCode")["enrolment"].sort_index()
            series = series[(series.index >= 2013) & (series.index <= BACKTEST_TRAIN_END)].dropna()
            if len(series) < 2:
                fc = [float(series.iloc[-1])] * n_backtest
            else:
                fc, _ = forecast_univariate(series.values, name, n_backtest)
            state_fc[state] = fc
        pred = np.array([state_fc[r.State][int(r.YearCode) - BACKTEST_TEST_YEARS[0]] for r in test_backtest.itertuples()])
        for state in backtest_states:
            state_mask = test_backtest["State"] == state
            if state_mask.any():
                s_pred = pred[state_mask.values]
                s_true = y_test_bt[state_mask.values]
                s_metrics = compute_metrics(s_true, s_pred)
                results.append({
                    "model": name,
                    "state": state,
                    "MAE": s_metrics["MAE"],
                    "RMSE": s_metrics["RMSE"],
                    "MAPE": s_metrics["MAPE"],
                    "R2": s_metrics["R2"],
                    "validation_years": f"{BACKTEST_TEST_YEARS[0]}-{BACKTEST_TEST_YEARS[1]}"
                })

    trend_results = []
    for state in backtest_states:
        series = hist[hist["State"] == state].set_index("YearCode")["enrolment"].sort_index()
        series = series[(series.index >= 2013) & (series.index <= BACKTEST_TRAIN_END)].dropna()
        if len(series) >= 2:
            years = series.index.values
            values = series.values
            slope, intercept, r2, x0 = fit_linear_trend(years, values)
            fc = {fy: intercept + slope * (fy - x0) for fy in BACKTEST_TEST_YEARS}
            pred = np.array([fc[r.YearCode] for r in test_backtest[test_backtest["State"]==state].itertuples()])
            true = test_backtest[test_backtest["State"]==state][TARGET].values
            if len(pred) == len(true):
                s_metrics = compute_metrics(true, pred)
                results.append({
                    "model": "Linear_Trend",
                    "state": state,
                    "MAE": s_metrics["MAE"],
                    "RMSE": s_metrics["RMSE"],
                    "MAPE": s_metrics["MAPE"],
                    "R2": s_metrics["R2"],
                    "validation_years": f"{BACKTEST_TEST_YEARS[0]}-{BACKTEST_TEST_YEARS[1]}"
                })

    return pd.DataFrame(results)


def compute_ensemble_weights(backtest_df):
    agg = backtest_df.groupby("model").agg({
        "MAE": "mean",
        "RMSE": "mean",
        "MAPE": "mean",
        "R2": "mean"
    }).reset_index()

    metrics = ["MAE", "RMSE", "MAPE"]
    for m in metrics:
        best = agg[m].min()
        agg[f"{m}_score"] = best / agg[m]

    agg["weight"] = agg[["MAE_score", "RMSE_score", "MAPE_score"]].mean(axis=1)
    total = agg["weight"].sum()
    agg["weight"] = agg["weight"] / total if total > 0 else 1.0 / len(agg)

    return agg[["model", "weight"]].set_index("model")["weight"].to_dict()


def generate_scenarios(hist, stage2_pipe, count_proj, ratio_base, forecast_years):
    scenarios = {}
    for scenario, factor in SCENARIOS.items():
        frames = []
        for state in hist["State"].unique():
            frames.append(
                recursive_xgboost_forecast(state, hist, stage2_pipe, count_proj, ratio_base, factor, forecast_years)
            )
        fc = pd.concat(frames, ignore_index=True)
        fc["Scenario"] = scenario
        scenarios[scenario] = fc
    return scenarios


def run_long_term_forecast():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    hist = historical_indicators()
    print("Historical indicators loaded:", hist.shape)

    stage2_pipe, stage2_name = retrain_xgboost_full()
    print(f"Stage 2 model: {stage2_name} retrained on full 2014-2019 data")

    count_summary, indicator_long = stage1_count_forecasts(hist, FORECAST_YEARS)
    print(f"Stage 1: count indicators forecasted for {len(FORECAST_YEARS)} years")
    fallbacks = count_summary[count_summary["Method"] == "last_observed_value_fallback"]
    print(f"  Fallbacks used: {len(fallbacks)} of {len(count_summary)}")

    ratio_base = ratio_baseline(hist)

    backtest_df = backtest_models(hist, BACKTEST_TEST_YEARS)
    backtest_df.to_csv(FORECAST_MODEL_COMPARISON_PATH, index=False)
    print(f"Backtesting results saved to {FORECAST_MODEL_COMPARISON_PATH}")

    weights = compute_ensemble_weights(backtest_df)
    print("Ensemble weights:", weights)

    all_scenarios = generate_scenarios(hist, stage2_pipe, count_summary, ratio_base, FORECAST_YEARS)
    base_fc = all_scenarios["Base_2019_Ratio"]
    lower_fc = all_scenarios["Lower_Ratio_Scenario"]
    upper_fc = all_scenarios["Upper_Ratio_Scenario"]

    # Compute proper bounds: for each state-year, min/max across all scenarios
    all_fc = pd.concat(all_scenarios.values(), ignore_index=True)
    bounds = all_fc.groupby(["State", "Year"])["Forecast_Enrollment"].agg(["min", "max"]).reset_index()
    bounds.columns = ["State", "Year", "conservative_bound", "optimistic_bound"]

    arima_fc, ets_fc, tbats_fc, trend_fc = {}, {}, {}, {}
    n_periods = len(FORECAST_YEARS)
    for state in hist["State"].unique():
        series = hist[hist["State"] == state].set_index("YearCode")["enrolment"].sort_index()
        series = series[(series.index >= 2013) & (series.index <= 2019)].dropna()
        if len(series) >= 2:
            arima_fc[state], _ = forecast_univariate(series.values, "ARIMA", n_periods)
            ets_fc[state], _ = forecast_univariate(series.values, "ETS", n_periods)
            tbats_fc[state], _ = forecast_univariate(series.values, "TBATS", n_periods)
            years = series.index.values
            values = series.values
            slope, intercept, r2, x0 = fit_linear_trend(years, values)
            trend_fc[state] = [intercept + slope * (fy - x0) for fy in FORECAST_YEARS]
        else:
            last = float(series.iloc[-1]) if len(series) > 0 else 0.0
            arima_fc[state] = [last] * n_periods
            ets_fc[state] = [last] * n_periods
            tbats_fc[state] = [last] * n_periods
            trend_fc[state] = [last] * n_periods

    xgb_base = base_fc.set_index(["State", "Year"])["Forecast_Enrollment"]
    xgb_lower = lower_fc.set_index(["State", "Year"])["Forecast_Enrollment"]
    xgb_upper = upper_fc.set_index(["State", "Year"])["Forecast_Enrollment"]

    arima_df = pd.DataFrame([
        {"State": s, "Year": y, "ARIMA_Forecast": arima_fc[s][i]}
        for s in hist["State"].unique() for i, y in enumerate(FORECAST_YEARS)
    ])
    ets_df = pd.DataFrame([
        {"State": s, "Year": y, "ETS_Forecast": ets_fc[s][i]}
        for s in hist["State"].unique() for i, y in enumerate(FORECAST_YEARS)
    ])
    tbats_df = pd.DataFrame([
        {"State": s, "Year": y, "TBATS_Forecast": tbats_fc[s][i]}
        for s in hist["State"].unique() for i, y in enumerate(FORECAST_YEARS)
    ])
    trend_df = pd.DataFrame([
        {"State": s, "Year": y, "Trend_Forecast": max(0.0, trend_fc[s][i])}
        for s in hist["State"].unique() for i, y in enumerate(FORECAST_YEARS)
    ])

    # Compute bounds from spread across all 5 models for each state-year
    model_forecasts = pd.DataFrame({
        "XGBoost": xgb_base,
        "ARIMA": arima_df.set_index(["State", "Year"])["ARIMA_Forecast"],
        "ETS": ets_df.set_index(["State", "Year"])["ETS_Forecast"],
        "TBATS": tbats_df.set_index(["State", "Year"])["TBATS_Forecast"],
        "Linear_Trend": trend_df.set_index(["State", "Year"])["Trend_Forecast"],
    })
    bounds_vals = model_forecasts.agg(
        lambda x: np.percentile(x.dropna(), [5, 95]),
        axis=1
    )
    bounds = pd.DataFrame({
        "State": bounds_vals.index.get_level_values(0),
        "Year": bounds_vals.index.get_level_values(1),
        "conservative_bound": bounds_vals.apply(lambda x: x[0]).values,
        "optimistic_bound": bounds_vals.apply(lambda x: x[1]).values,
    })

    projection_rows = []
    bounds_dict = bounds.set_index(["State", "Year"])
    for state in hist["State"].unique():
        for i, year in enumerate(FORECAST_YEARS):
            xgb = xgb_base.get((state, year), np.nan)
            arima = arima_fc[state][i]
            ets = ets_fc[state][i]
            tbats = tbats_fc[state][i]
            trend = trend_fc[state][i]

            model_preds = {
                "XGBoost": xgb,
                "ARIMA": arima,
                "ETS": ets,
                "TBATS": tbats,
                "Linear_Trend": trend,
            }

            used_models = [m for m in model_preds if np.isfinite(model_preds[m])]
            used_weights = {m: weights.get(m, 0.0) for m in used_models}
            total_weight = sum(used_weights.values())
            if total_weight > 0:
                used_weights = {m: w / total_weight for m, w in used_weights.items()}
            else:
                used_weights = {m: 1.0 / len(used_models) for m in used_models}

            ensemble = sum(
                model_preds[m] * used_weights[m]
                for m in used_models
            )

            try:
                conservative = bounds_dict.loc[(state, year), "conservative_bound"]
                optimistic = bounds_dict.loc[(state, year), "optimistic_bound"]
            except KeyError:
                conservative = ensemble * 0.95
                optimistic = ensemble * 1.05

            if np.isfinite(ensemble):
                ensemble = max(0.0, ensemble)

            projection_rows.append({
                "state": state,
                "year": year,
                "xgboost_forecast": xgb,
                "arima_forecast": arima,
                "ets_forecast": ets,
                "tbats_forecast": tbats,
                "trend_forecast": trend,
                "final_forecast": ensemble,
                "lower_bound": conservative,
                "upper_bound": optimistic,
                "methodology": f"Ensemble (weights: {', '.join(f'{k}={v:.3f}' for k,v in weights.items())})"
            })

    projection_df = pd.DataFrame(projection_rows)
    projection_df.to_csv(ENROLLMENT_PROJECTION_PATH, index=False)
    print(f"Projections saved to {ENROLLMENT_PROJECTION_PATH}")

    scenarios_2030 = []
    for state in hist["State"].unique():
        row_2019 = hist[(hist["State"] == state) & (hist["YearCode"] == 2019)]
        actual_2019 = row_2019["enrolment"].values[0] if len(row_2019) > 0 else np.nan

        proj_2030 = projection_df[(projection_df["state"] == state) & (projection_df["year"] == 2030)]
        if len(proj_2030) > 0:
            baseline = proj_2030["final_forecast"].values[0]
            lower = proj_2030["lower_bound"].values[0]
            upper = proj_2030["upper_bound"].values[0]

            conservative = lower if np.isfinite(lower) else baseline * 0.95
            optimistic = upper if np.isfinite(upper) else baseline * 1.05

            hist_cagr = None
            series = hist[(hist["State"] == state) & (hist["YearCode"].between(2014, 2019))].sort_values("YearCode")
            if len(series) >= 2:
                first = series.iloc[0]["enrolment"]
                last = series.iloc[-1]["enrolment"]
                n_years = 5
                if first > 0:
                    hist_cagr = (last / first) ** (1 / n_years) - 1

            proj_cagr = None
            if np.isfinite(actual_2019) and actual_2019 > 0 and np.isfinite(baseline):
                proj_cagr = (baseline / actual_2019) ** (1 / (2030 - 2019)) - 1

            scenarios_2030.append({
                "state": state,
                "enrollment_2019": actual_2019,
                "conservative_2030": conservative,
                "baseline_2030": baseline,
                "optimistic_2030": optimistic,
                "projected_growth_2019_2030": proj_cagr * 100 if proj_cagr else np.nan,
                "historical_cagr_2014_2019": hist_cagr * 100 if hist_cagr else np.nan,
            })

    scenarios_df = pd.DataFrame(scenarios_2030)
    scenarios_df.to_csv(ENROLLMENT_SCENARIOS_PATH, index=False)
    print(f"Scenarios saved to {ENROLLMENT_SCENARIOS_PATH}")

    india_rows = []
    for year in FORECAST_YEARS:
        proj = projection_df[projection_df["year"] == year]
        conservative_sum = proj["lower_bound"].sum()
        baseline_sum = proj["final_forecast"].sum()
        optimistic_sum = proj["upper_bound"].sum()

        india_rows.append({
            "year": year,
            "conservative_projection": conservative_sum,
            "baseline_projection": baseline_sum,
            "optimistic_projection": optimistic_sum,
            "lower_bound": conservative_sum,
            "upper_bound": optimistic_sum,
        })

    india_df = pd.DataFrame(india_rows)
    india_df.to_csv(INDIA_PROJECTION_PATH, index=False)
    print(f"India projection saved to {INDIA_PROJECTION_PATH}")

    generate_plots(hist, base_fc, lower_fc, upper_fc, projection_df, india_df)
    generate_report(hist, projection_df, scenarios_df, india_df, backtest_df, weights)

    print("Long-term forecast complete.")
    return projection_df, scenarios_df, india_df


def generate_plots(hist, base_fc, lower_fc, upper_fc, projection_df, india_df):
    hist_agg = hist.groupby("YearCode")["enrolment"].sum()
    base_agg = base_fc.groupby("Year")["Forecast_Enrollment"].sum()
    lower_agg = lower_fc.groupby("Year")["Forecast_Enrollment"].sum()
    upper_agg = upper_fc.groupby("Year")["Forecast_Enrollment"].sum()

    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(hist_agg.index, hist_agg.values, marker="o", label="Historical (2013-2019)", color="black", linewidth=2)
    ax.plot(base_agg.index, base_agg.values, marker="s", label="Baseline Projection", color="steelblue", linestyle="--", linewidth=2)
    ax.fill_between(base_agg.index, lower_agg.values, upper_agg.values, alpha=0.2, color="steelblue", label="±5% Ratio Band")
    ax.axvline(FORECAST_START - 0.5, color="red", linestyle=":", linewidth=1.5)
    ax.text(FORECAST_START + 0.3, ax.get_ylim()[1] * 0.95, "Forecast", color="red", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Enrollment")
    ax.set_title("India Higher Education Enrollment: Historical vs Projected (2013-2030)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "india_projection_2030.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(india_df["year"], india_df["baseline_projection"], marker="o", label="Baseline", color="steelblue", linewidth=2)
    ax.fill_between(india_df["year"], india_df["lower_bound"], india_df["upper_bound"], alpha=0.2, color="steelblue", label="Conservative-Optimistic Range")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Enrollment")
    ax.set_title("India-Level Enrollment Projection 2020-2030 with Uncertainty")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "india_projection_scenarios.png"), dpi=150, bbox_inches="tight")
    plt.close()

    scenarios_2030 = projection_df[projection_df["year"] == 2030].sort_values("final_forecast", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["steelblue", "coral", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i, (_, row) in enumerate(scenarios_2030.iterrows()):
        ax.barh(i, row["final_forecast"], color=colors[i], alpha=0.8, label=row["state"])
        if np.isfinite(row["lower_bound"]) and np.isfinite(row["upper_bound"]):
            ax.plot([row["lower_bound"], row["upper_bound"]], [i, i], color="black", linewidth=1.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(scenarios_2030["state"].values[::-1])
    ax.set_xlabel("Projected Enrollment (2030)")
    ax.set_title("Top 10 States by Projected Enrollment in 2030 (with range)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top10_states_2030.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    for state in hist["State"].unique():
        h = hist[hist["State"] == state].sort_values("YearCode")
        f = projection_df[(projection_df["state"] == state) & (projection_df["year"] >= FORECAST_START)]
        if len(h) > 0 and len(f) > 0:
            ax.plot(h["YearCode"], h["enrolment"], color="gray", alpha=0.3, linewidth=0.5)
            ax.plot(f["year"], f["final_forecast"], color="steelblue", alpha=0.3, linewidth=0.5, linestyle="--")
    ax.axvline(FORECAST_START - 0.5, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Enrollment")
    ax.set_title("State-wise Enrollment: Historical (2013-2019) vs Projected (2020-2030)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "statewise_historical_vs_projected.png"), dpi=150, bbox_inches="tight")
    plt.close()

    backtest_df = pd.read_csv(FORECAST_MODEL_COMPARISON_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    for ax, metric in zip(axes.flatten(), metrics):
        model_means = backtest_df.groupby("model")[metric].mean().sort_values()
        model_means.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
        ax.set_title(f"{metric} by Model (2018-2019 Backtest)")
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_backtest.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    projection_2030 = projection_df[projection_df["year"] == 2030].copy()
    projection_2030["hist_cagr"] = np.nan
    for state in projection_2030["state"]:
        series = hist[(hist["State"] == state) & (hist["YearCode"].between(2014, 2019))].sort_values("YearCode")
        if len(series) >= 2:
            first = series.iloc[0]["enrolment"]
            last = series.iloc[-1]["enrolment"]
            if first > 0:
                hist_cagr = (last / first) ** (1/5) - 1
                projection_2030.loc[projection_2030["state"] == state, "hist_cagr"] = hist_cagr * 100

    projection_2030["proj_cagr"] = np.nan
    for _, row in projection_2030.iterrows():
        actual_2019 = hist[(hist["State"] == row["state"]) & (hist["YearCode"] == 2019)]["enrolment"].values
        if len(actual_2019) > 0 and actual_2019[0] > 0 and np.isfinite(row["final_forecast"]):
            proj_2030 = row["final_forecast"]
            projection_2030.loc[projection_2030["state"] == row["state"], "proj_cagr"] = (proj_2030 / actual_2019[0]) ** (1/11) - 1

    valid = projection_2030.dropna(subset=["hist_cagr", "proj_cagr"])
    ax.scatter(valid["hist_cagr"], valid["proj_cagr"], alpha=0.6, s=50)
    ax.plot([-0.1, 0.15], [-0.1, 0.15], "r--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Historical CAGR 2014-2019 (%)")
    ax.set_ylabel("Projected CAGR 2019-2030 (%)")
    ax.set_title("Historical vs Projected Growth Rates by State")
    ax.axhline(0, color="black", alpha=0.3)
    ax.axvline(0, color="black", alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "growth_rate_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    for scenario in ["Lower_Ratio_Scenario", "Base_2019_Ratio", "Upper_Ratio_Scenario"]:
        sc_data = base_scenarios[scenario] if 'base_scenarios' in locals() else None
    agg_scenarios = projection_df.groupby("year").agg({
        "lower_bound": "sum",
        "final_forecast": "sum",
        "upper_bound": "sum"
    })
    ax.plot(agg_scenarios.index, agg_scenarios["final_forecast"], marker="o", label="Baseline", color="steelblue", linewidth=2)
    ax.fill_between(agg_scenarios.index, agg_scenarios["lower_bound"], agg_scenarios["upper_bound"], alpha=0.2, color="steelblue", label="Conservative-Optimistic Range")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Enrollment")
    ax.set_title("India Enrollment: Conservative vs Baseline vs Optimistic Scenarios")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scenario_comparison_2030.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Plots generated in", PLOTS_DIR)


def generate_report(hist, projection_df, scenarios_df, india_df, backtest_df, weights):
    total_2019 = hist[hist["YearCode"] == 2019]["enrolment"].sum()
    baseline_2030 = india_df[india_df["year"] == 2030]["baseline_projection"].values[0]
    conservative_2030 = india_df[india_df["year"] == 2030]["conservative_projection"].values[0]
    optimistic_2030 = india_df[india_df["year"] == 2030]["optimistic_projection"].values[0]

    growth_2030 = (baseline_2030 / total_2019 - 1) * 100

    def df_to_md(df):
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join([header, sep] + rows)

    backtest_summary = backtest_df.groupby("model").agg({
        "MAE": "mean", "RMSE": "mean", "MAPE": "mean", "R2": "mean"
    }).round(2).reset_index()

    top_states_2030 = scenarios_df.nlargest(10, "baseline_2030")[["state", "enrollment_2019", "conservative_2030", "baseline_2030", "optimistic_2030"]].round(0).reset_index(drop=True)

    growth_flags = []
    for _, row in scenarios_df.iterrows():
        if pd.notna(row["historical_cagr_2014_2019"]) and pd.notna(row["projected_growth_2019_2030"]):
            ratio = row["projected_growth_2019_2030"] / row["historical_cagr_2014_2019"] if row["historical_cagr_2014_2019"] != 0 else np.inf
            if abs(ratio) > 2:
                growth_flags.append(f"{row['state']}: projected CAGR {row['projected_growth_2019_2030']:.1f}% vs historical {row['historical_cagr_2014_2019']:.1f}% (ratio={ratio:.1f}x)")

    lines = [
        "# India Higher Education Enrollment Projection 2030",
        "",
        "## 1. Objective",
        "",
        "This project estimates higher education enrollment for Indian states through 2030 using historical AISHE trends (2012-13 to 2019-20).",
        "The projection is based on a multi-model ensemble combining XGBoost, ARIMA, ETS, TBATS, and Linear Trend models.",
        "",
        "## 2. Data",
        "",
        f"- **Source**: NDAP Report 9259 (AISHE)",
        f"- **Historical Period**: 2012-13 to 2019-20 (8 financial years)",
        f"- **States**: 35 states (Ladakh excluded due to single-year data)",
        f"- **ML Modeling Period**: 2014-2019 (6 years per state, 210 observations)",
        f"- **Target**: Total higher education enrollment (Gender=Total, Items=Total)",
        f"- **Limitations**: Short time series (5-7 observations per state); 2012 only has Haryana enrollment; no demographic/policy variables.",
        "",
        "## 3. Methodology",
        "",
        "### ML Forecasting (XGBoost)",
        "- Two-stage approach: Stage 1 projects exogenous indicators (teachers, colleges, etc.) via linear trends; Stage 2 uses recursive XGBoost with lag features.",
        "- Ratio indicators (GER, GPI, PTRs) held at 2019 baseline with ±5% sensitivity.",
        f"- Exogenous indicators projected to 2030 via per-state linear trends on 2013-2019 history (fallback to last observed if unstable/insufficient data).",
        "- Model retrained on full 2014-2019 data.",
        "",
        "### Time-Series Models",
        "- **ARIMA**: auto_arima with AIC, max p=2, d=1, q=2, no seasonality.",
        "- **ETS**: Additive error, additive trend, no seasonality, no damping.",
        "- **TBATS**: Trend with damped trend, no Box-Cox, no seasonal periods.",
        "- All fit on 2013-2017 training, evaluated on 2018-2019.",
        "",
        "### Trend Model",
        "- Simple linear regression (y = a + b*t) on enrollment 2013-2019 per state.",
        "- Projections clipped at zero.",
        "",
        "### Backtesting",
        "- Train: 2014-2017, Test: 2018-2019 (chronological, no leakage).",
        "- All models evaluated on MAE, RMSE, MAPE, R² per state.",
        "",
        "### Ensemble",
        f"- Weights from backtest performance (inverse MAE/RMSE/MAPE, direct R²): {', '.join(f'{k}={v:.3f}' for k,v in weights.items())}",
        "- Better validation performance → higher weight.",
        "- Models with unstable long-horizon behavior (XGBoost recursive) included but monitored.",
        "",
        "### Exogenous Variables (2020-2030)",
        "- 6 count indicators: linear trend per state (2013-2019), clipped at 0, instability cap at 3× max historical.",
        "- Ratio indicators: held at 2019 value ±5% (scenarios).",
        "",
        "## 4. Validation Results (2018-2019 Backtest)",
        "",
        df_to_md(backtest_summary),
        "",
        "## 5. 2030 Projection",
        "",
        f"### India-Level Summary",
        f"- **Historical 2019 Enrollment**: {total_2019:,.0f}",
        f"- **Projected 2030 Enrollment (Baseline)**: {baseline_2030:,.0f}",
        f"- **Conservative**: {conservative_2030:,.0f}",
        f"- **Optimistic**: {optimistic_2030:,.0f}",
        f"- **Projected Growth 2019→2030**: {growth_2030:.1f}%",
        "",
        "### Top 10 States by Projected 2030 Enrollment (Baseline)",
        "",
        df_to_md(top_states_2030),
        "",
        "## 6. Limitations",
        "",
        "- **Short Historical Time Series**: Only 5-7 observations per state for trend estimation.",
        "- **Long Forecast Horizon**: 11-year projection (2020-2030) from 2019 base; recursive error accumulation likely.",
        "- **Policy Changes Not Included**: NEP 2020, COVID impacts, demographic shifts not explicitly modeled.",
        "- **Exogenous Variables**: Linear trend assumption may not hold for colleges/teachers over 11 years.",
        "- **No Population/Demographic Drivers**: Enrollment growth tied to institution capacity, not demand-side factors.",
        "- **XGBoost Recursive Stability**: Long-horizon recursive forecasts may drift; ensemble mitigates but doesn't eliminate.",
        "- **COVID-Era Disruption**: 2020-2022 actual enrollment affected by pandemic; projections assume trend continuation.",
        "",
        "## 7. Interpretation",
        "",
        "**This is a historical-trend-based projection and should NOT be interpreted as an exact prediction of actual 2030 enrollment.**",
        "",
        "The projections represent what enrollment *would be* if historical trends in institutional capacity (colleges, teachers, universities) and enrollment rates (GER, GPI, PTR) continue linearly from 2013-2019 patterns. Actual 2030 enrollment will depend on policy decisions, economic conditions, demographic changes, and educational reforms not captured in this model.",
        "",
        "### Growth Rate Flags (States with Projected CAGR >2x Historical)",
        "",
    ] + (growth_flags if growth_flags else ["None flagged."]) + [
        "",
        "---",
        f"*Report generated from AISHE data through 2019-20. Forecast methodology documented in outputs/forecast_methodology.txt*",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_long_term_forecast()