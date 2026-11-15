from fastapi import APIRouter
import os

from api.services import data_service as ds

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "outputs",
)

router = APIRouter()


@router.get("/forecast/all")
def all_forecasts():
    return ds.get_all_forecasts_latest()


@router.get("/forecast/methodology")
def methodology():
    p = os.path.join(OUTPUTS_DIR, "forecast_methodology.txt")
    if not os.path.exists(p):
        return {"text": "Methodology file not found."}
    with open(p) as f:
        return {"text": f.read()}


@router.get("/forecast/indicator-trends")
def indicator_trends():
    df = ds.get_indicator_trends()
    return df.to_dict(orient="records")
