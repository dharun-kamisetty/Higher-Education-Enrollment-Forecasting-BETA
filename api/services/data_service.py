"""
Data service layer — reads existing pipeline outputs.
No schema changes; read-only access to outputs/ and models/.
"""
import json
import os
from functools import lru_cache

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output file not found: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def get_ml_dataset() -> pd.DataFrame:
    return _read_csv("ml_dataset.csv")


@lru_cache(maxsize=1)
def get_model_comparison() -> pd.DataFrame:
    return _read_csv("model_comparison.csv")


@lru_cache(maxsize=1)
def get_predictions() -> pd.DataFrame:
    return _read_csv("predictions.csv")


@lru_cache(maxsize=1)
def get_feature_importance() -> pd.DataFrame:
    return _read_csv("feature_importance.csv")


@lru_cache(maxsize=1)
def get_enrollment_forecasts() -> pd.DataFrame:
    return _read_csv("enrollment_forecasts.csv")


@lru_cache(maxsize=1)
def get_enrollment_series() -> pd.DataFrame:
    return _read_csv("enrollment_series.csv")


@lru_cache(maxsize=1)
def get_indicator_trends() -> pd.DataFrame:
    return _read_csv("indicator_trend_summary.csv")


@lru_cache(maxsize=1)
def get_best_params() -> dict:
    path = os.path.join(MODELS_DIR, "best_params.json")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_hybrid_best_params() -> dict:
    path = os.path.join(OUTPUTS_DIR, "hybrid_best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_states() -> list[dict]:
    mld = get_ml_dataset()
    states = sorted(mld["State"].unique().tolist())
    return [{"name": s, "code": int(mld[mld["State"] == s]["StateCode"].iloc[0])} for s in states]


def get_state_data(state: str) -> dict:
    mld = get_ml_dataset()
    sub = mld[mld["State"] == state].sort_values("YearCode")
    if sub.empty:
        return {}
    return {
        "state": state,
        "years": sub["YearCode"].tolist(),
        "enrollment": sub["higher_education_enrollment"].tolist(),
        "enrollment_lag_1": sub["enrollment_lag_1"].tolist(),
        "teachers": sub["teachers_lag1"].tolist(),
        "colleges": sub["colleges_lag1"].tolist(),
        "GER": sub["GER_lag1"].tolist(),
        "GPI": sub["GPI_lag1"].tolist(),
    }


def get_latest_enrollment_by_state() -> list[dict]:
    mld = get_ml_dataset()
    latest = mld[mld["YearCode"] == mld["YearCode"].max()]
    return [
        {"state": row["State"], "code": int(row["StateCode"]), "enrollment": float(row["higher_education_enrollment"])}
        for _, row in latest.iterrows()
    ]


def get_forecast_by_state(state: str) -> dict:
    fc = get_enrollment_forecasts()
    sub = fc[fc["State"] == state]
    if sub.empty:
        return {}
    scenarios = {}
    for scen in sub["Scenario"].unique():
        s = sub[sub["Scenario"] == scen].sort_values("Year")
        scenarios[scen] = {
            "years": s["Year"].tolist(),
            "values": s["Forecast_Enrollment"].tolist(),
        }
    return {"state": state, "scenarios": scenarios}


def get_all_forecasts_latest() -> list[dict]:
    fc = get_enrollment_forecasts()
    base = fc[fc["Scenario"] == "Base_2019_Ratio"]
    latest_year = int(base["Year"].max())
    latest = base[base["Year"] == latest_year]
    return [
        {"state": row["State"], "forecast": float(row["Forecast_Enrollment"]), "year": int(row["Year"])}
        for _, row in latest.iterrows()
    ]


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.joblib")
    if not os.path.exists(path):
        alt = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(alt):
            path = alt
        else:
            return None
    return joblib.load(path)


def clear_cache():
    get_ml_dataset.cache_clear()
    get_model_comparison.cache_clear()
    get_predictions.cache_clear()
    get_feature_importance.cache_clear()
    get_enrollment_forecasts.cache_clear()
    get_enrollment_series.cache_clear()
    get_indicator_trends.cache_clear()
    get_best_params.cache_clear()
    get_hybrid_best_params.cache_clear()
