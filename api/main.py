"""
FastAPI backend for the enrollment forecasting dashboard.
Serves existing pipeline outputs via REST. No DB schema changes.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import states, models, forecast, predict, forecast_2030, forecast_by_year, assistant

app = FastAPI(
    title="Enrollment Forecasting API",
    description="REST API serving AISHE higher-education enrollment forecasts",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(states.router, prefix="/api", tags=["states"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(forecast.router, prefix="/api", tags=["forecast"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(forecast_2030.router, prefix="/api", tags=["forecast-2030"])
app.include_router(forecast_by_year.router, prefix="/api", tags=["forecast-by-year"])
app.include_router(assistant.router, prefix="/api", tags=["assistant"])


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)