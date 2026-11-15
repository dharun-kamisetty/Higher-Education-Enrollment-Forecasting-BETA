from fastapi import APIRouter, HTTPException
from api.services import data_service as ds

router = APIRouter()


@router.get("/states")
def list_states():
    return ds.get_states()


@router.get("/states/{state}/data")
def state_data(state: str):
    data = ds.get_state_data(state)
    if not data:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found")
    return data


@router.get("/states/enrollment/latest")
def latest_enrollment():
    return ds.get_latest_enrollment_by_state()


@router.get("/states/{state}/forecast")
def state_forecast(state: str):
    data = ds.get_forecast_by_state(state)
    if not data:
        raise HTTPException(status_code=404, detail=f"No forecast for '{state}'")
    return data
