from fastapi import APIRouter
import os
import pandas as pd

from api.services import data_service as ds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

router = APIRouter()


@router.get("/forecast/2030")
def forecast_2030():
    path = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    # Return XGBoost forecast for 2030 per state (beta)
    latest = df[df["year"] == 2030]
    return [
        {"state": row["state"], "year": 2030, "forecast": float(row["xgboost_forecast"]),
         "final_forecast": float(row["final_forecast"]),
         "lower_bound": float(row["lower_bound"]),
         "upper_bound": float(row["upper_bound"])}
        for _, row in latest.iterrows()
    ]


@router.get("/forecast/2030/{state}")
def forecast_2030_state(state: str):
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
        "final": sub["final_forecast"].tolist(),
        "lower": sub["lower_bound"].tolist(),
        "upper": sub["upper_bound"].tolist(),
    }


@router.get("/forecast/2030-all")
def forecast_2030_all():
    path = os.path.join(OUTPUTS_DIR, "enrollment_projection_2020_2030.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
