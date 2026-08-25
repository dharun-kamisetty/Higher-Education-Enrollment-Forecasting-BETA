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
    NUMERIC_COLUMNS,
    OUTPUTS_DIR,
    PLOTS_DIR,
)
from evaluate_models import (
    DEFAULT_PARAMS,
    MODEL_FACTORY,
    load_best_params,
    strip_model_prefix,
)
from feature_engineering import EXCLUDED_STATES, pivot_features
from modeling_utils import FEATURES, build_model_pipeline

FORECAST_START = 2020
PRIMARY_END = 2022
FORECAST_END = 2024
SENSITIVITY_PERTURBATION = 0.05
INSTABILITY_CAP_MULTIPLE = 3.0
MIN_OBS_FOR_TREND = 3

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

SCENARIOS = {
    "Base_2019_Ratio": 1.0,
    "Lower_Ratio_Scenario": 1.0 - SENSITIVITY_PERTURBATION,
    "Upper_Ratio_Scenario": 1.0 + SENSITIVITY_PERTURBATION,
}

FORECAST_YEARS = list(range(FORECAST_START, FORECAST_END + 1))

INDICATOR_FORECAST_PATH = os.path.join(OUTPUTS_DIR, "future_indicator_forecasts.csv")
INDICATOR_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "indicator_trend_summary.csv")
ENROLLMENT_FORECAST_PATH = os.path.join(OUTPUTS_DIR, "enrollment_forecasts.csv")
ENROLLMENT_SERIES_PATH = os.path.join(OUTPUTS_DIR, "enrollment_series.csv")
METHODOLOGY_PATH = os.path.join(OUTPUTS_DIR, "forecast_methodology.txt")


def best_model_name():
    with open(os.path.join(OUTPUTS_DIR, "model_selection.txt")) as f:
        name = f.readline().strip().replace("Best-performing model: ", "")
    if name in MODEL_FACTORY:
        return name
    return "XGBoost"


def retrain_stage2_model():
    mld = pd.read_csv(ML_DATASET_PATH)
    X, y = mld[FEATURES], mld["higher_education_enrollment"]
    name = best_model_name()
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


def stage1_count_forecasts(hist):
    summary_rows = []
    long_rows = []
    for state, g in hist.groupby("State"):
        for indicator in COUNT_INDICATORS:
            sub = g[["YearCode", indicator]].dropna()
            if sub.empty:
                last_val = np.nan
                proj = {fy: np.nan for fy in FORECAST_YEARS}
                method = "no_historical_data"
                r2 = np.nan
            else:
                years = sub["YearCode"].tolist()
                values = sub[indicator].tolist()
                last_val = values[-1]
                proj, method, r2 = project_count_indicator(years, values, FORECAST_YEARS)
            summary_rows.append(
                {
                    "State": state,
                    "Indicator": indicator,
                    "Historical_R2": r2,
                    "2019_Value": last_val,
                    **{f"{fy}_Forecast": proj[fy] for fy in FORECAST_YEARS},
                    "Method": method,
                }
            )
            for fy in FORECAST_YEARS:
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


def recursive_forecast_state(state, hist, stage2_pipe, count_proj, ratio_base, ratio_factor):
    g = hist[hist["State"] == state].sort_values("YearCode")
    code = g["StateCode"].iloc[0]

    enrollment = dict(zip(g["YearCode"], g["enrolment"]))

    count_series = {}
    for indicator in COUNT_INDICATORS:
        actual = dict(zip(g["YearCode"], g[indicator]))
        for fy in FORECAST_YEARS:
            actual[fy] = count_proj.loc[
                (count_proj["State"] == state) & (count_proj["Indicator"] == indicator),
                f"{fy}_Forecast",
            ].iloc[0]
        count_series[indicator] = actual

    ratio_series = {}
    for indicator in RATIO_INDICATORS:
        actual = dict(zip(g["YearCode"], g[indicator]))
        base_val = ratio_base[state][indicator]
        for fy in FORECAST_YEARS:
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
    for t in FORECAST_YEARS:
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
        predictions[t] = value
        enrollment[t] = value
        rows.append({"State": state, "StateCode": code, "Year": t, "Forecast_Enrollment": value})

    return pd.DataFrame(rows)


def build_enrollment_series(hist, base_forecasts):
    hist_series = hist[["State", "YearCode", "enrolment"]].copy()
    hist_series = hist_series.rename(columns={"YearCode": "Year", "enrolment": "Enrollment"})
    hist_series["is_forecast"] = False
    hist_series["Scenario"] = "Historical"
    fc = base_forecasts[["State", "Year", "Forecast_Enrollment"]].rename(
        columns={"Forecast_Enrollment": "Enrollment"}
    )
    fc["is_forecast"] = True
    fc["Scenario"] = "Base_2019_Ratio"
    return pd.concat([hist_series, fc], ignore_index=True)


def plot_forecasts(hist, base_fc, all_fc, summary):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    hist_agg = hist.groupby("YearCode")["enrolment"].sum()
    base_agg = base_fc.groupby("Year")["Forecast_Enrollment"].sum()
    agg = all_fc.groupby(["Year", "Scenario"])["Forecast_Enrollment"].sum().unstack()

    palette = plt.cm.tab10.colors

    def state_colors():
        top = summary.sort_values("2019_Value", ascending=False).head(5)["State"].tolist()
        return top, palette

    fig, ax = plt.subplots(figsize=(10, 6))
    top, palette = state_colors()
    for state in hist["State"].unique():
        h = hist[hist["State"] == state]
        f = base_fc[base_fc["State"] == state]
        color = palette[top.index(state)] if state in top else (0.7, 0.7, 0.7, 0.5)
        lw = 1.8 if state in top else 1.0
        ax.plot(h["YearCode"], h["enrolment"], color=color, alpha=0.9 if state in top else 0.5, linewidth=lw)
        ax.plot(f["Year"], f["Forecast_Enrollment"], color=color, alpha=0.7 if state in top else 0.4, linewidth=lw, linestyle="--")
    ax.axvline(FORECAST_START - 0.5, color="red", linestyle=":", linewidth=1)
    ax.text(FORECAST_START + 0.3, ax.get_ylim()[1] * 0.95, "forecast", color="red", fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Enrollment")
    ax.set_title("State-wise enrollment: history and 2020-2024 base forecast (top 5 highlighted)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "state_forecasts.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_agg.index, hist_agg.values, marker="o", label="Historical", color="black")
    ax.plot(base_agg.index, base_agg.values, marker="s", label="Base forecast", color="steelblue", linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total enrollment")
    ax.set_title("Aggregate enrollment: history and 2020-2024 base forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "aggregate_forecast.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_agg.index, hist_agg.values, marker="o", label="Historical", color="black")
    ax.plot(base_agg.index, base_agg.values, marker="s", label="Base_2019_Ratio", color="steelblue", linestyle="--")
    if "Lower_Ratio_Scenario" in agg.columns:
        ax.plot(agg.index, agg["Lower_Ratio_Scenario"], marker=".", label="Lower scenario", color="green", linestyle=":")
    if "Upper_Ratio_Scenario" in agg.columns:
        ax.plot(agg.index, agg["Upper_Ratio_Scenario"], marker=".", label="Upper scenario", color="red", linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total enrollment")
    ax.set_title("Aggregate enrollment scenario comparison (2020-2024)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scenario_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_agg.index, hist_agg.values, marker="o", label="Historical", color="black")
    base22 = base_fc[base_fc["Year"] <= PRIMARY_END].groupby("Year")["Forecast_Enrollment"].sum()
    ax.plot(base22.index, base22.values, marker="s", label=f"Primary projection (to {PRIMARY_END})", color="steelblue", linestyle="--")
    ax.axvline(FORECAST_START - 0.5, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total enrollment")
    ax.set_title(f"Aggregate enrollment: history and primary projection {FORECAST_START}-{PRIMARY_END}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_2022_primary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_agg.index, hist_agg.values, marker="o", label="Historical", color="black")
    ax.plot(base_agg.index, base_agg.values, marker="s", label=f"Extended base forecast (to {FORECAST_END})", color="steelblue", linestyle="--")
    if "Lower_Ratio_Scenario" in agg.columns and "Upper_Ratio_Scenario" in agg.columns:
        ax.fill_between(agg.index, agg["Lower_Ratio_Scenario"], agg["Upper_Ratio_Scenario"], alpha=0.2, color="gray", label=f"Lower-upper {int(SENSITIVITY_PERTURBATION*100)}% band")
    ax.axvline(FORECAST_START - 0.5, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total enrollment")
    ax.set_title(f"Aggregate enrollment: history and extended forecast {FORECAST_START}-{FORECAST_END}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_2024_extended.png"), dpi=150, bbox_inches="tight")
    plt.close()


def write_methodology(count_summary, aggregate_text, stage2_name):
    lines = [
        "=== Forecast Methodology ===",
        "",
        "1. Historical modeling period: 2014-2019 (6 years per state; 2013 used as lag input). "
        "Benchmark train/test split 2014-17 / 2018-19 is separate and unchanged.",
        f"2. Forecast horizon: primary projection {FORECAST_START}-{PRIMARY_END}; "
        f"extended sensitivity {FORECAST_START}-{FORECAST_END}. No 2030 point forecast is produced.",
        "3. Stage 1 method: per State, per indicator, a simple linear trend y = a + b*t (t = calendar "
        f"year) is fitted on available historical observations through 2019. If fewer than "
        f"{MIN_OBS_FOR_TREND} observations exist, or the projection exceeds "
        f"{INSTABILITY_CAP_MULTIPLE}x the historical maximum, the last observed value is used "
        "as a documented fallback.",
        f"4. Count indicators forecast: {', '.join(COUNT_INDICATORS)}. Negative projections are "
        "clipped to zero; fallbacks are recorded in outputs/future_indicator_forecasts.csv and "
        "outputs/indicator_trend_summary.csv.",
        "5. Ratio indicators treatment: GER, GPI, PTR_all, PTR_university_colleges, "
        "PTR_university_units are NOT extrapolated. They are held at their 2019 value as the "
        "'2019 ratio baseline scenario'. These are assumptions, not actual future observations.",
        f"6. Stage 2 model: {stage2_name}, retrained on the complete 2014-2019 modeling dataset "
        "with the same feature preprocessing as the validated benchmark pipeline.",
        "7. Recursive forecasting procedure: for each State and scenario, enrollment is forecast "
        "year-by-year. Each year uses the previous (predicted where applicable) enrollment for "
        "enrollment lag features, Stage-1 projected count indicators, and the ratio scenario. "
        "Predicted enrollment is fed back; actual future enrollment is never used.",
        f"8. Scenario assumptions: base = 2019 ratio baseline; lower = base x {1-SENSITIVITY_PERTURBATION:.2f}; "
        f"upper = base x {1+SENSITIVITY_PERTURBATION:.2f} (documented "
        f"{int(SENSITIVITY_PERTURBATION*100)}% perturbation). Count indicator projections are "
        "identical across scenarios.",
        "9. Limitations: only 6-7 historical observations per state; count-indicator projections "
        "assume linear continuation of short trends; ratio indicators are held constant by "
        "design; long-horizon projections are therefore indicative, not predictive. The 2018-19 "
        "benchmark validation measures one-step model accuracy and must be interpreted "
        "separately from the multi-year projection.",
        "",
        f"Aggregate base projection (sum across {len(count_summary['State'].unique())} states): "
        f"{aggregate_text}",
    ]
    with open(METHODOLOGY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    hist = historical_indicators()
    stage2_pipe, stage2_name = retrain_stage2_model()
    print(f"Stage 1: fitting per-State linear trends for {len(COUNT_INDICATORS)} count indicators.")
    count_summary, indicator_long = stage1_count_forecasts(hist)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    indicator_long.to_csv(INDICATOR_FORECAST_PATH, index=False)
    count_summary.to_csv(INDICATOR_SUMMARY_PATH, index=False)
    ratio_base = ratio_baseline(hist)

    fallbacks = count_summary[count_summary["Method"] == "last_observed_value_fallback"]
    print(f"  Count-indicator fallbacks used: {len(fallbacks)} of {len(count_summary)}")

    all_rows = []
    for scenario, factor in SCENARIOS.items():
        frames = []
        for state in hist["State"].unique():
            frames.append(
                recursive_forecast_state(state, hist, stage2_pipe, count_summary, ratio_base, factor)
            )
        fc = pd.concat(frames, ignore_index=True)
        fc["Scenario"] = scenario
        all_rows.append(fc)
    all_fc = pd.concat(all_rows, ignore_index=True)
    all_fc.to_csv(ENROLLMENT_FORECAST_PATH, index=False)

    base_fc = all_fc[all_fc["Scenario"] == "Base_2019_Ratio"]
    series = build_enrollment_series(hist, base_fc)
    series.to_csv(ENROLLMENT_SERIES_PATH, index=False)

    hist_agg = hist.groupby("YearCode")["enrolment"].sum()
    base_agg = base_fc.groupby("Year")["Forecast_Enrollment"].sum()
    agg_text = " | ".join(
        f"{y}: {base_agg.get(y, float('nan')):,.0f}" for y in FORECAST_YEARS
    )
    hist_text = " | ".join(f"{y}: {v:,.0f}" for y, v in hist_agg.items())
    aggregate_text = f"history ({hist_text}); base forecast ({agg_text})"

    plot_forecasts(hist, base_fc, all_fc, count_summary)
    write_methodology(count_summary, aggregate_text, stage2_name)

    primary = base_fc[base_fc["Year"] <= PRIMARY_END]
    print(f"Stage 2: recursive enrollment forecast using {stage2_name}.")
    print(f"Primary projection {FORECAST_START}-{PRIMARY_END} saved to {ENROLLMENT_FORECAST_PATH}")
    print(f"Extended projection to {FORECAST_END} saved.")
    print("Scenarios:", ", ".join(SCENARIOS))
    print(f"Aggregate base forecast (states):")
    print(base_agg.to_string())
    print(f"Methodology saved to {METHODOLOGY_PATH}")


if __name__ == "__main__":
    main()
