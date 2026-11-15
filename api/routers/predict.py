from fastapi import APIRouter
from pydantic import BaseModel
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

router = APIRouter()

# Feature columns used by the model
FEATURES = [
    "enrollment_lag_1", "enrollment_lag_2", "enrollment_growth", "rolling_mean",
    "university_enrolment_lag1", "foreign_students_lag1", "stand_alone_institutions_lag1",
    "teachers_lag1", "colleges_lag1", "universities_lag1",
    "GPI_lag1", "GER_lag1", "PTR_all_lag1", "PTR_university_colleges_lag1",
    "PTR_university_units_lag1", "Year", "State",
]


class PredictRequest(BaseModel):
    state: str
    year: int
    enrollment_lag_1: float
    enrollment_lag_2: float = 0
    teachers_lag1: float = 0
    colleges_lag1: float = 0
    universities_lag1: float = 0
    stand_alone_institutions_lag1: float = 0
    foreign_students_lag1: float = 0
    university_enrolment_lag1: float = 0
    GER_lag1: float = 0
    GPI_lag1: float = 0
    PTR_all_lag1: float = 0
    PTR_university_colleges_lag1: float = 0
    PTR_university_units_lag1: float = 0


@router.post("/predict")
def predict(req: PredictRequest):
    """Run XGBoost prediction with user-provided indicator values."""
    # Load the trained XGBoost model
    model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
    if not os.path.exists(model_path):
        return {"error": "Model not found. Run the pipeline first."}

    try:
        pipe = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    # Build feature vector
    e1 = req.enrollment_lag_1
    e2 = req.enrollment_lag_2 if req.enrollment_lag_2 > 0 else e1
    growth = (e1 - e2) / e2 if e2 > 0 else 0.0
    prior = [e1, e2]
    prior = [p for p in prior if p > 0]
    rolling = float(np.mean(prior)) if prior else e1

    row = {
        "enrollment_lag_1": e1,
        "enrollment_lag_2": e2,
        "enrollment_growth": growth,
        "rolling_mean": rolling,
        "university_enrolment_lag1": req.university_enrolment_lag1,
        "foreign_students_lag1": req.foreign_students_lag1,
        "stand_alone_institutions_lag1": req.stand_alone_institutions_lag1,
        "teachers_lag1": req.teachers_lag1,
        "colleges_lag1": req.colleges_lag1,
        "universities_lag1": req.universities_lag1,
        "GPI_lag1": req.GPI_lag1,
        "GER_lag1": req.GER_lag1,
        "PTR_all_lag1": req.PTR_all_lag1,
        "PTR_university_colleges_lag1": req.PTR_university_colleges_lag1,
        "PTR_university_units_lag1": req.PTR_university_units_lag1,
        "Year": req.year,
    }

    try:
        import pandas as pd
        row["State"] = req.state
        frame = pd.DataFrame([row])[FEATURES]
        prediction = float(pipe.predict(frame)[0])
        return {
            "state": req.state,
            "year": req.year,
            "predicted_enrollment": round(prediction, 0),
            "model": "XGBoost",
            "input_summary": {
                "enrollment_lag_1": e1,
                "teachers_lag1": req.teachers_lag1,
                "colleges_lag1": req.colleges_lag1,
                "GER_lag1": req.GER_lag1,
                "GPI_lag1": req.GPI_lag1,
            },
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
