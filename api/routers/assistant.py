import os
from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are EduCast, an AI assistant for the Higher Education Enrollment Forecasting project (AISHE).
You answer questions about this project concisely, precisely, and accurately.

PROJECT FACTS:
- Dataset: AISHE higher-education enrollment, 35 Indian states, 2012-2019
- ML dataset: 210 rows (35 states x 6 years: 2014-2019), 20 columns
- Target: higher_education_enrollment
- Train: 2014-2017, Test: 2018-2019
- Best model: XGBoost (MAE=43,467, RMSE=69,042, MAPE=62.38%, R2=0.9975)
- XGBoost params: subsample=0.8, n_estimators=300, max_depth=3, learning_rate=0.05, colsample_bytree=1.0
- Features (15): enrollment_lag_1, enrollment_lag_2, enrollment_growth, rolling_mean, teachers_lag1, colleges_lag1, universities_lag1, stand_alone_institutions_lag1, foreign_students_lag1, university_enrolment_lag1, GER_lag1, GPI_lag1, PTR_all_lag1, PTR_university_colleges_lag1, PTR_university_units_lag1, State (categorical)
- Feature importance (XGBoost): enrollment_lag_1=0.834, colleges_lag1=0.112, rolling_mean=0.031, teachers_lag1=0.008, GPI_lag1=0.004
- 12 models total: XGBoost, ARIMA, Linear Regression, TBATS, ETS, LightGBM, Random Forest, SVR, WOASVR, GWOSVR, PSOSVR, ActualGRIDSVR
- ARIMA: MAE=44,474, R2=0.9971 | Linear Regression: MAE=55,947, R2=0.9966
- SVR variants all have negative R2 (~-0.13)
- Forecasting: Two-stage (Stage1: linear trends for 6 count indicators; Stage2: recursive XGBoost)
- Forecast horizon: 2020-2030 (beta)
- 3 scenarios: Base (2019 ratio), Lower (0.95x), Upper (1.05x)
- National enrollment 2019: 38,228,258 | 2030 XGBoost forecast: ~44.5M
- Top 5 states by enrollment (2019): Uttar Pradesh (6,388,214), Maharashtra (4,265,472), Tamil Nadu (3,520,311), Rajasthan (2,206,517), Karnataka (2,187,892)
- Top 5 states by 2030 XGBoost forecast: Uttar Pradesh (6,284,544), Maharashtra (4,482,506), Tamil Nadu (3,657,194), Karnataka (2,498,536), Rajasthan (2,431,334)
- All 35 states enrollment 2019: Andaman And Nicobar Islands (10,131), Andhra Pradesh (1,897,149), Arunachal Pradesh (55,816), Assam (650,601), Bihar (1,738,432), Chandigarh (106,667), Chhattisgarh (586,595), Delhi (974,483), Goa (52,782), Gujarat (1,544,840), Haryana (933,541), Himachal Pradesh (289,488), Jammu And Kashmir (273,599), Jharkhand (817,560), Karnataka (2,187,892), Kerala (1,137,853), Lakshadweep (560), Madhya Pradesh (2,182,154), Maharashtra (4,265,472), Manipur (124,538), Meghalaya (87,541), Mizoram (33,236), Nagaland (44,561), Odisha (994,929), Puducherry (55,110), Punjab (869,463), Rajasthan (2,206,517), Sikkim (58,071), Tamil Nadu (3,520,311), Telangana (1,389,608), The Dadra And Nagar Haveli And Daman And Diu (10,325), Tripura (86,247), Uttar Pradesh (6,388,214), Uttarakhand (493,279), West Bengal (2,160,893)
- Model comparison: XGBoost (MAE=43467, R2=0.9975), ARIMA (MAE=44474, R2=0.9971), Linear Regression (MAE=55947, R2=0.9966), TBATS (MAE=51703, R2=0.9945), ETS (MAE=50192, R2=0.9956), LightGBM (MAE=114163, R2=0.9810), Random Forest (MAE=94583, R2=0.9729), SVR (MAE=927410, R2=-0.1344)
- Tech stack: Python, FastAPI, Next.js 14, Tailwind, Leaflet, XGBoost, scikit-learn
- Frontend: Dark mode, 6 tabs (Overview, Map, Models, Forecast, Predict, Features)
- Map: 35 Indian states choropleth, OpenStreetMap tiles

RULES:
- Be concise (2-4 sentences max unless asked for detail)
- Be accurate — only state facts from above
- If asked something not in the facts, say "I don't have that info"
- Never make up numbers or metrics
"""


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    model: str


@router.post("/assistant/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return ChatResponse(
            reply="Groq API key not configured. Set GROQ_API_KEY environment variable.",
            model="none",
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add conversation history (last 6 messages max)
        for h in req.history[-6:]:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": req.message})

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        reply = completion.choices[0].message.content
        return ChatResponse(reply=reply, model=GROQ_MODEL)
    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", model="error")
