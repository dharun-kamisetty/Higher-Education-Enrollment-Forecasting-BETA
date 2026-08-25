import os

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODELS = [
    "Linear Regression",
    "SVR",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "WOASVR",
    "GWOSVR",
    "PSOSVR",
    "ActualGRIDSVR",
    "ARIMA",
    "ETS",
    "TBATS",
]

METRICS = ["MAE", "RMSE", "MAPE", "R2"]

METRIC_HELP = {
    "MAE": "Mean Absolute Error - lower is better",
    "RMSE": "Root Mean Squared Error - lower is better",
    "MAPE": "Mean Absolute Percentage Error (%) - lower is better",
    "R2": "Coefficient of Determination - higher is better",
}


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(OUTPUTS_DIR, name))


def require_outputs():
    required = [
        "ml_dataset.csv",
        "predictions.csv",
        "model_comparison.csv",
        "feature_importance.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(OUTPUTS_DIR, f))]
    return missing


st.set_page_config(page_title="Higher Education Enrollment Forecasting", layout="wide")

missing = require_outputs()
if missing:
    st.error(
        f"Pipeline outputs missing: {missing}. "
        "Run the pipeline first with: `python3 run_pipeline.py`"
    )
    st.stop()

mld = load_csv("ml_dataset.csv")
pred = load_csv("predictions.csv")
mc = load_csv("model_comparison.csv")
fi = load_csv("feature_importance.csv")

has_forecast = all(
    os.path.exists(os.path.join(OUTPUTS_DIR, f))
    for f in [
        "enrollment_forecasts.csv",
        "enrollment_series.csv",
        "indicator_trend_summary.csv",
        "forecast_methodology.txt",
    ]
)
if has_forecast:
    fc_all = load_csv("enrollment_forecasts.csv")
    fc_series = load_csv("enrollment_series.csv")
    fc_indicators = load_csv("indicator_trend_summary.csv")

with open(os.path.join(OUTPUTS_DIR, "model_selection.txt")) as f:
    selection = f.read().strip()

best_model = selection.splitlines()[0].replace("Best-performing model: ", "")

st.title("Higher Education Enrollment Forecasting")
st.caption(
    "AISHE enrollment forecasts for Indian states - State-Year level ML pipeline "
    f"(train 2014-17, test 2018-19)"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("States", mld["State"].nunique())
c2.metric("Years (predictions)", f"{mld['Year'].min()} - {mld['Year'].max()}")
c3.metric("ML observations", f"{len(mld)}")
c4.metric("Best model", best_model)
best_row = mc[mc["Model"] == best_model].iloc[0]
c5.metric("Test MAE", f"{best_row['MAE']:,.0f}")

tab_overview, tab_trends, tab_models, tab_predictions, tab_forecast, tab_importance = st.tabs(
    [
        "Overview",
        "Enrollment Trends",
        "Model Comparison",
        "Predictions",
        "Projected enrollment trend to 2024",
        "Feature Importance",
    ]
)

with tab_overview:
    st.subheader("What this dashboard shows")
    st.write(
        "A machine-learning forecast of higher-education enrollment for each Indian "
        "state and financial year, built from the AISHE report. Five regression models "
        "were trained on lagged indicators (previous year's enrollment, teachers, "
        "colleges, universities, GER, GPI, pupil-teacher ratios)."
    )

    st.subheader("Dataset snapshot (State-Year level)")
    show_cols = [
        "State",
        "Year",
        "higher_education_enrollment",
        "enrollment_lag_1",
        "teachers_lag1",
        "colleges_lag1",
        "GER_lag1",
        "GPI_lag1",
    ]
    st.dataframe(
        mld[show_cols].sort_values(["State", "Year"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Best model selection")
    st.info(selection)
    st.write(
        "Selection uses the average ranking across all four metrics on the "
        "chronological test set, not R-squared alone."
    )

with tab_trends:
    st.subheader("Actual higher-education enrollment over time")
    states = st.multiselect(
        "Select states (empty = national average)",
        options=sorted(mld["State"].unique()),
        default=[],
    )
    if states:
        sub = mld[mld["State"].isin(states)]
        pivot = sub.pivot_table(
            index="Year", columns="State", values="higher_education_enrollment"
        )
        st.line_chart(pivot, use_container_width=True)
    else:
        avg = mld.groupby("Year")["higher_education_enrollment"].mean()
        st.line_chart(avg, use_container_width=True)
        st.caption("National average enrollment across all 35 states.")

    top = (
        mld.groupby("State")["higher_education_enrollment"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    st.subheader("Top 10 states by average enrollment")
    st.bar_chart(top, use_container_width=True)

with tab_models:
    st.subheader("Test-set metrics by model")
    st.dataframe(
        mc.style.format(
            {
                "MAE": "{:,.0f}",
                "RMSE": "{:,.0f}",
                "MAPE": "{:.1f}",
                "R2": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    metric = st.radio(
        "Choose metric", METRICS, horizontal=True, help=METRIC_HELP[METRICS[0]]
    )
    if metric == "R2":
        plot_data = mc.set_index("Model")[metric]
        st.bar_chart(plot_data, use_container_width=True)
    else:
        plot_data = mc.set_index("Model")[metric]
        st.bar_chart(plot_data, use_container_width=True)
    st.caption(METRIC_HELP[metric])

    st.subheader("Actual vs predicted (test set)")
    model = st.selectbox("Select model", MODELS)
    scatter = pred[["Actual Enrollment", f"{model} Prediction"]].rename(
        columns={"Actual Enrollment": "Actual", f"{model} Prediction": "Predicted"}
    )
    st.scatter_chart(scatter, use_container_width=True)
    st.caption(
        f"{model}: MAE={best_row['MAE'] if model == best_model else mc[mc['Model'] == model].iloc[0]['MAE']:,.0f} | "
        f"R2={mc[mc['Model'] == model].iloc[0]['R2']:.4f}"
    )

    with st.expander("Hybrid and time-series model details"):
        st.markdown(
            "**Hybrid SVR** (WOASVR, GWOSVR, PSOSVR) tune SVR hyperparameters "
            "(C, epsilon, gamma) with the Whale, Grey-Wolf and Particle-Swarm "
            "metaheuristics; ActualGRIDSVR is the equivalent exhaustive grid search. "
            "All minimize TimeSeriesSplit CV RMSE on 2014-2017 training data. "
            "**Time-series models** (ARIMA, ETS, TBATS) are per-State univariate fits on "
            "each State's 2013-2017 enrollment history forecasting 2018-2019."
        )
        hopt = os.path.join(OUTPUTS_DIR, "hybrid_optimization.csv")
        hparam = os.path.join(OUTPUTS_DIR, "hybrid_best_params.json")
        tsm = os.path.join(OUTPUTS_DIR, "time_series_methods.csv")
        if os.path.exists(hparam):
            import json

            with open(hparam) as f:
                st.json(json.load(f))
        if os.path.exists(hopt):
            st.caption("Metaheuristic convergence (best CV RMSE per iteration)")
            st.line_chart(
                pd.read_csv(hopt).pivot(index="Iteration", columns="Optimizer", values="Best_CV_RMSE"),
                use_container_width=True,
            )
        if os.path.exists(tsm):
            st.caption("Time-series fit method per State (fitted vs fallback)")
            st.dataframe(pd.read_csv(tsm), use_container_width=True, hide_index=True)

with tab_predictions:
    st.subheader("Model predictions vs actual enrollment (test years 2018-19)")
    col_filter, year_filter = st.columns(2)
    sel_states = col_filter.multiselect(
        "Filter states", options=sorted(pred["State"].unique()), default=sorted(pred["State"].unique())[:5]
    )
    sel_years = year_filter.multiselect(
        "Filter years", options=sorted(pred["Year"].unique()), default=sorted(pred["Year"].unique())
    )
    filtered = pred[
        pred["State"].isin(sel_states) & pred["Year"].isin(sel_years)
    ].sort_values(["State", "Year"])
    fmt = {f"{m} Prediction": "{:,.0f}" for m in MODELS}
    fmt["Actual Enrollment"] = "{:,.0f}"
    st.dataframe(filtered.style.format(fmt), use_container_width=True, hide_index=True)

    show_all = st.checkbox("Show all states")
    if show_all:
        st.dataframe(
            pred.sort_values(["State", "Year"]).style.format(fmt),
            use_container_width=True,
            hide_index=True,
        )

with tab_forecast:
    if not has_forecast:
        st.info(
            "Projection outputs not found. Run `python3 run_pipeline.py` to generate "
            "`outputs/enrollment_forecasts.csv`."
        )
    else:
        st.subheader("Projected enrollment trend to 2024")
        st.info(
            "Forecasts from 2020 onward are model-based projections. Count indicators "
            "(teachers, colleges, universities, stand-alone institutions, foreign "
            "students, university enrolment) are extrapolated from historical State-level "
            "trends, while GER, GPI and PTR indicators use their 2019 values as the "
            "baseline scenario."
        )
        st.write(
            f"History (2013-19) plus two-stage projection. Primary horizon 2020-2022; "
            f"extended sensitivity 2020-2024. A 5% sensitivity band is applied to the "
            f"2019 GER/GPI/PTR ratio baseline (lower 0.95x, base 1.0x, upper 1.05x). "
            f"No 2030 point forecast is produced."
        )

        base = fc_all[fc_all["Scenario"] == "Base_2019_Ratio"]
        base_agg = base.groupby("Year")["Forecast_Enrollment"].sum()
        scenario_agg = fc_all.groupby(["Year", "Scenario"])["Forecast_Enrollment"].sum().unstack()
        hist_agg = fc_series[fc_series["is_forecast"] == False].groupby("Year")["Enrollment"].sum()

        up = int(
            (
                base.loc[base["Year"] == 2022, "Forecast_Enrollment"].to_numpy()
                > base.loc[base["Year"] == 2020, "Forecast_Enrollment"].to_numpy()
            ).sum()
        )
        down = 35 - up
        agg_pct = (base_agg[2022] / hist_agg[2019] - 1) * 100
        f1, f2, f3 = st.columns(3)
        f1.metric("States projected to increase (2020-22)", up)
        f2.metric("States projected to decrease (2020-22)", down)
        f3.metric("Aggregate change 2019 to 2022 (base)", f"{agg_pct:+.1f}%")

        st.subheader("Primary projection (2020-2022) and extended sensitivity (2020-2024)")
        df_chart = pd.DataFrame({"Historical": hist_agg})
        df_chart = df_chart.join(base_agg.rename("Base forecast"))
        df_chart = df_chart.join(scenario_agg[["Lower_Ratio_Scenario", "Upper_Ratio_Scenario"]])
        st.line_chart(df_chart, use_container_width=True)

        st.subheader("State-wise projection (base scenario)")
        default_states = (
            fc_series[fc_series["is_forecast"] == False]
            .sort_values("Year", ascending=False)
            .head(35)
            .sort_values("Enrollment", ascending=False)
            .head(4)["State"]
            .tolist()
        )
        fc_states = st.multiselect(
            "Select states for trend (empty = top 4 by 2019 enrollment)",
            options=sorted(fc_series["State"].unique()),
            default=default_states,
        )
        if not fc_states:
            fc_states = default_states
        series_sel = fc_series[fc_series["State"].isin(fc_states)]
        pivot = series_sel.pivot_table(index="Year", columns="State", values="Enrollment")
        st.line_chart(pivot, use_container_width=True)

        st.subheader("Count indicator projections (Stage 1, per state)")
        ind_state = st.selectbox(
            "Select state for indicator trends",
            options=sorted(fc_indicators["State"].unique()),
            key="ind_state",
        )
        ind_sub = fc_indicators[fc_indicators["State"] == ind_state]
        ind_pivot = ind_sub.pivot_table(
            index="Indicator",
            values=[f"{y}_Forecast" for y in range(2020, 2025)],
        ).reindex(
            ["teachers", "colleges", "universities", "stand_alone_institutions",
             "foreign_students", "university_enrolment"]
        )
        ind_pivot.columns = [f"{c.split('_')[0]}" for c in ind_pivot.columns]
        st.dataframe(
            ind_pivot.style.format("{:,.0f}"),
            use_container_width=True,
            hide_index=False,
        )
        st.caption(
            "Method: per-State linear trend y = a + b*t on 2013-2019 history, negative "
            "projections clipped to zero, last-observed-value fallback when the series is "
            "too short or unstable. See outputs/indicator_trend_summary.csv for R2 and "
            "method per State-Indicator."
        )

        st.subheader("Enrollment forecast table")
        st.dataframe(
            fc_all[fc_all["Year"] <= 2022].sort_values(
                ["Scenario", "State", "Year"]
            ).style.format({"Forecast_Enrollment": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

with tab_importance:
    st.subheader("Feature importance (tree-based models)")
    st.write(
        "How strongly each lagged feature contributed to predictions. This is not a "
        "causal interpretation."
    )
    imp_model = st.selectbox("Select model", fi["model"].unique(), key="imp")
    sub = fi[fi["model"] == imp_model].sort_values("importance", ascending=True)
    chart = sub.set_index("feature")["importance"]
    st.bar_chart(chart, use_container_width=True)
    st.dataframe(
        sub[["feature", "importance"]].sort_values("importance", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Key takeaway")
    st.write(
        "Previous-year enrollment is consistently the strongest predictor across all "
        "tree-based models."
    )
