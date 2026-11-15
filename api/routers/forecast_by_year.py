from fastapi import APIRouter, Query
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

router = APIRouter()


@router.get("/forecast/by-year/{year}")
def forecast_by_year(year: int):
    """Return XGBoost forecasts for all states for a given year (2020-2030)."""
    path = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
    if not os.path.exists(path):
        return {"error": "Forecast data not available. Run forecast_2030.py first."}
    df = pd.read_csv(path)
    sub = df[df["year"] == year]
    if sub.empty:
        # If year not pre-computed, return empty with message
        available = sorted(df["year"].unique().tolist())
        return {"error": f"Year {year} not available. Available years: {available}"}
    return [
        {
            "state": row["state"],
            "year": int(row["year"]),
            "xgboost": float(row["xgboost_forecast"]),
            "ensemble": float(row["final_forecast"]),
            "lower": float(row["lower_bound"]),
            "upper": float(row["upper_bound"]),
        }
        for _, row in sub.iterrows()
    ]


@router.get("/forecast/available-years")
def available_years():
    path = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
    if not os.path.exists(path):
        return {"years": []}
    df = pd.read_csv(path)
    return {"years": sorted(df["year"].unique().tolist())}


@router.get("/forecast/state-timeline/{state}")
def state_timeline(state: str):
    """Return full 2020-2030 forecast timeline for a single state."""
    path = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    sub = df[df["state"] == state].sort_values("year")
    if sub.empty:
        return {}
    return {
        "state": state,
        "years": sub["year"].tolist(),
        "xgboost": sub["xgboost_forecast"].tolist(),
        "ensemble": sub["final_forecast"].tolist(),
        "lower": sub["lower_bound"].tolist(),
        "upper": sub["upper_bound"].tolist(),
    }
