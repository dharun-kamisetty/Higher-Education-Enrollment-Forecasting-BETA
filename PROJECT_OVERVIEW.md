# Higher Education Enrollment Forecasting — Project Overview

## 1. Project Title
**Higher Education Enrollment Forecasting for Indian States using Machine Learning**

## 2. Problem Statement
Higher education enrollment in India varies significantly across states and years. Accurate forecasting of enrollment numbers is critical for:
- Government policy planning and resource allocation
- University/college capacity planning
- Education infrastructure development
- Budget estimation for education sector

This project builds a machine learning system that forecasts higher-education enrollment for 35 Indian states/UTs using historical AISHE (All India Survey on Higher Education) data.

## 3. Objectives
1. Analyze historical higher-education enrollment patterns across Indian states
2. Build and compare multiple ML models for enrollment prediction
3. Identify the most accurate model for state-level enrollment forecasting
4. Generate multi-year forecasts (2020-2030) with scenario analysis
5. Deploy an interactive web application for visualization and exploration
6. Provide AI-assisted Q&A for project insights

## 4. Dataset

### Source
- **AISHE (All India Survey on Higher Education)** — NDAP_REPORT_9259.csv
- **Period:** Financial Year 2012-2019 (8 years)
- **Coverage:** 36 states/UTs, 12 item categories, 3 gender values

### Raw Data
| Property | Value |
|----------|-------|
| Rows | 11,232 |
| Columns | 18 |
| States/UTs | 36 |
| Years | 2012-2019 |
| Gender | Male, Female, Total |
| Items | Total, All Categories, SC, ST, Professor, Reader, Lecturer, etc. |

### ML Dataset (After Feature Engineering)
| Property | Value |
|----------|-------|
| Rows | 210 (35 states × 6 years) |
| Columns | 20 |
| Year range | 2014-2019 |
| States | 35 (1 excluded) |
| Target | higher_education_enrollment |
| Train period | 2014-2017 |
| Test period | 2018-2019 |

### Features (15 numeric + 1 categorical)
| # | Feature | Description |
|---|---------|-------------|
| 1 | enrollment_lag_1 | Previous year enrollment |
| 2 | enrollment_lag_2 | 2-year lag enrollment |
| 3 | enrollment_growth | Year-over-year growth rate |
| 4 | rolling_mean | 3-year rolling mean of enrollment |
| 5 | teachers_lag1 | Previous year teacher count |
| 6 | colleges_lag1 | Previous year college count |
| 7 | universities_lag1 | Previous year university count |
| 8 | stand_alone_institutions_lag1 | Previous year stand-alone institutions |
| 9 | foreign_students_lag1 | Previous year foreign students |
| 10 | university_enrolment_lag1 | Previous year university enrollment |
| 11 | GER_lag1 | Gross Enrolment Ratio (lag-1) |
| 12 | GPI_lag1 | Gender Parity Index (lag-1) |
| 13 | PTR_all_lag1 | Pupil-Teacher Ratio all (lag-1) |
| 14 | PTR_university_colleges_lag1 | PTR university/colleges (lag-1) |
| 15 | PTR_university_units_lag1 | PTR university units (lag-1) |
| 16 | State | Categorical (one-hot encoded, 35 states) |

## 5. Methodology

### 5.1 Data Pipeline
```
Raw AISHE Data (CSV)
    ↓
Data Inspection & Understanding
    ↓
Data Cleaning (filter, rename, impute)
    ↓
Feature Engineering (lag features, rolling means, growth rates)
    ↓
ML Dataset (210 × 20)
```

### 5.2 Model Training Pipeline
```
ML Dataset
    ↓
Chronological Split (Train: 2014-17, Test: 2018-19)
    ↓
Baseline Models (5 models)
    ↓
Hyperparameter Tuning (Temporal CV)
    ↓
Final Evaluation (12 models)
    ↓
Model Selection (XGBoost — lowest avg rank)
    ↓
Feature Importance Analysis
```

### 5.3 Forecasting Pipeline (Two-Stage)
```
Stage 1: Per-State Linear Trends
  - 6 count indicators (teachers, colleges, universities, etc.)
  - Linear trend: y = a + b*t
  - Fallback: last observed value if <3 obs or projection >3× max
  - Negative projections clipped to 0
  
Stage 2: Recursive XGBoost Forecast
  - Retrain XGBoost on all 2014-2019 data
  - Year-by-year recursive prediction
  - Predicted enrollment fed back as lag
  - Count indicators from Stage 1
  - Ratio indicators (GER, GPI, PTR) held at 2019 baseline
  
Scenarios:
  - Base: 2019 ratio baseline (1.0×)
  - Lower: 0.95× baseline
  - Upper: 1.05× baseline
```

### 5.4 2030 Forecasting (Beta)
- Ensemble approach: XGBoost + ARIMA + ETS + TBATS + Linear Trend
- Per-state forecasts with lower/upper bounds
- Horizon: 2020-2030

## 6. Models

### 6.1 Model Comparison (Test Set 2018-2019)
| Model | MAE | RMSE | MAPE (%) | R² |
|-------|-----|------|----------|-----|
| **XGBoost** | **43,467** | **69,042** | **62.38** | **0.9975** |
| ARIMA | 44,474 | 74,234 | 6.76 | 0.9971 |
| Linear Regression | 55,947 | 79,869 | 429.83 | 0.9966 |
| ETS | 50,192 | 90,757 | 7.33 | 0.9956 |
| TBATS | 51,703 | 101,668 | 6.50 | 0.9945 |
| LightGBM | 114,163 | 188,858 | 57.36 | 0.9810 |
| Random Forest | 94,583 | 225,782 | 186.50 | 0.9729 |
| SVR | 927,410 | 1,459,550 | 3,565.60 | -0.1344 |
| WOASVR | 920,433 | 1,451,195 | 3,555.24 | -0.1214 |
| GWOSVR | 927,013 | 1,459,194 | 3,568.15 | -0.1338 |
| PSOSVR | 920,438 | 1,451,199 | 3,555.19 | -0.1214 |
| ActualGRIDSVR | 921,197 | 1,451,881 | 3,549.84 | -0.1225 |

### 6.2 Best Model: XGBoost
| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 1.0 |

### 6.3 Feature Importance (XGBoost)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | enrollment_lag_1 | 0.8338 |
| 2 | colleges_lag1 | 0.1122 |
| 3 | rolling_mean | 0.0311 |
| 4 | teachers_lag1 | 0.0076 |
| 5 | GPI_lag1 | 0.0039 |

### 6.4 Model Selection Criteria
- Average ranking across all 4 metrics (MAE, RMSE, MAPE, R²)
- XGBoost has lowest average rank
- Not based on R² alone (avoids overfitting bias)

## 7. Forecasting Results

### 7.1 National Aggregate (Base Scenario)
| Year | Enrollment | Type |
|------|------------|------|
| 2013 | 32,064,257 | Historical |
| 2014 | 33,943,690 | Historical |
| 2015 | 34,320,373 | Historical |
| 2016 | 35,430,513 | Historical |
| 2017 | 36,350,701 | Historical |
| 2018 | 37,095,329 | Historical |
| 2019 | 38,228,258 | Historical |
| 2020 | 39,126,568 | Forecast |
| 2021 | 39,464,089 | Forecast |
| 2022 | 39,702,441 | Forecast |
| 2023 | 39,989,498 | Forecast |
| 2024 | 40,030,167 | Forecast |

### 7.2 Top 5 States by Enrollment (2019)
| Rank | State | Enrollment |
|------|-------|------------|
| 1 | Uttar Pradesh | 6,388,214 |
| 2 | Maharashtra | 4,265,472 |
| 3 | Tamil Nadu | 3,520,311 |
| 4 | Rajasthan | 2,206,517 |
| 5 | Karnataka | 2,187,892 |

### 7.3 Top 5 States by 2030 XGBoost Forecast
| Rank | State | Forecast |
|------|-------|----------|
| 1 | Uttar Pradesh | 6,284,544 |
| 2 | Maharashtra | 4,482,506 |
| 3 | Tamil Nadu | 3,657,194 |
| 4 | Karnataka | 2,498,536 |
| 5 | Rajasthan | 2,431,334 |

## 8. System Architecture

### 8.1 Overall Architecture
```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  Next.js 14 + React + TypeScript + Tailwind      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Dashboard │ │ Map View │ │ EduCast Chat     │ │
│  │ (6 tabs)  │ │ (Leaflet)│ │ (Groq AI)       │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │ REST API (HTTP)
┌────────────────────┴────────────────────────────┐
│                   Backend                        │
│  FastAPI (Python) + Uvicorn                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Data     │ │ Models   │ │ Forecast         │ │
│  │ Service  │ │ Service  │ │ Service          │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │ Read-only
┌────────────────────┴────────────────────────────┐
│                   ML Pipeline                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Trained  │ │ CSV      │ │ GeoJSON          │ │
│  │ Models   │ │ Outputs  │ │ (India States)   │ │
│  │ (.joblib)│ │ (.csv)   │ │                  │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 8.2 Technology Stack
| Layer | Technology |
|-------|------------|
| ML/Data | Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, SciPy |
| Backend | FastAPI, Uvicorn, python-dotenv, Groq SDK |
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS |
| Maps | Leaflet, react-leaflet, OpenStreetMap |
| Charts | Recharts |
| AI | Groq Cloud (openai/gpt-oss-120b) |
| Storage | CSV files, Joblib model artifacts |

### 8.3 API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /api/states | List 35 states |
| GET | /api/states/{state}/data | Historical data per state |
| GET | /api/states/enrollment/latest | Latest enrollment by state |
| GET | /api/states/{state}/forecast | Forecast by state |
| GET | /api/models/comparison | 12-model comparison |
| GET | /api/models/best | Best model (XGBoost) |
| GET | /api/feature-importance | Feature rankings |
| GET | /api/forecast/all | All forecasts |
| GET | /api/forecast/2030 | 2030 XGBoost forecasts |
| GET | /api/forecast/by-year/{year} | Forecasts for specific year |
| GET | /api/forecast/available-years | Available years (2020-2030) |
| GET | /api/forecast/state-timeline/{state} | Full timeline per state |
| POST | /api/predict | Custom XGBoost prediction |
| POST | /api/assistant/chat | EduCast AI chat |

## 9. Application Features

### 9.1 Dashboard (6 Tabs)
1. **Overview:** KPI cards (states, total enrollment, best model, MAE), top 5 states, model summary
2. **Map:** Interactive India choropleth with year selector (2020-2030), hover tooltips, click selection
3. **Models:** MAE bar chart + full metrics table for all 12 models
4. **Forecast:** 2030 projection table + national aggregate totals
5. **Predict:** Custom forecast form (state, year, indicators → XGBoost prediction)
6. **Features:** Feature importance by model (RF, XGBoost, LightGBM)

### 9.2 EduCast AI Assistant
- Floating chat widget (bottom-right corner)
- Answers project-specific questions concisely
- Powered by Groq Cloud (openai/gpt-oss-120b)
- System prompt contains all project facts and data
- Conversation history support

### 9.3 Indian State Map
- 35-state choropleth colored by enrollment/forecast
- OpenStreetMap tiles (free, no API key)
- Dark mode CSS filter
- Year selector dropdown
- Hover tooltips with state name + value
- Click to select state
- Legend with min/max values

## 10. Key Findings

1. **XGBoost is the best model** — lowest average rank across MAE, RMSE, MAPE, R²
2. **enrollment_lag_1 dominates** — 83.4% feature importance in XGBoost
3. **SVR variants fail** — negative R² (-0.13) due to scale sensitivity
4. **ARIMA is competitive** — MAE=44,474, R²=0.9971 (close to XGBoost)
5. **Enrollment is growing** — 3.9% aggregate increase projected 2019→2022
6. **Uttar Pradesh leads** — highest enrollment (6.39M) and 2030 forecast (6.28M)
7. **Count indicators trend linearly** — teachers, colleges show steady growth
8. **Ratio indicators stable** — GER, GPI, PTRs held at 2019 baseline

## 11. Limitations

1. Only 6-7 historical observations per state (limited training data)
2. Count-indicator projections assume linear continuation of short trends
3. Ratio indicators held constant by design (not actual future observations)
4. Long-horizon projections (2030) are indicative, not predictive
5. 2018-19 benchmark validation measures one-step accuracy only
6. No district-level data (state-level aggregation only)
7. No demographic/economic predictors (GDP, population, budget)

## 12. Future Work

1. **Longer horizon:** Extend to 2030 with demographic drivers and policy scenarios
2. **Better features:** GDP, education budget, infrastructure, digital adoption
3. **Advanced models:** Quantile regression, neural networks, ensemble stacking
4. **Hierarchical models:** Mixed-effects, Bayesian pooling across states
5. **Causal analysis:** Policy intervention evaluation, counterfactuals
6. **Deployment:** API + monitoring + automated retraining
7. **Explainability:** SHAP values, partial dependence plots
8. **Data enrichment:** District-level AISHE, NIRF rankings, placement data

## 13. Repository

**GitHub:** https://github.com/dharun-kamisetty/Higher-Education-Enrollment-Forecasting-BETA

### How to Run
```bash
# Clone
git clone https://github.com/dharun-kamisetty/Higher-Education-Enrollment-Forecasting-BETA.git
cd Higher-Education-Enrollment-Forecasting-BETA

# Install Python deps
pip install -r requirements.txt

# Install frontend deps
cd frontend && npm install && cd ..

# Set Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run full-stack
./run_fullstack.sh

# Or run ML pipeline (if needed)
python3 run_pipeline.py

# Or run original Streamlit dashboard
streamlit run dashboard.py
```

### Access Points
| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## 14. Conclusion

This project demonstrates a complete machine learning workflow from data collection to deployment:
- **Data:** AISHE higher-education enrollment (35 states, 2012-2019)
- **Models:** 12 models compared, XGBoost selected (R²=0.9975)
- **Forecasting:** Two-stage recursive forecast to 2030 with 3 scenarios
- **Deployment:** Full-stack web app with interactive dashboard, Indian map, and AI assistant
- **Documentation:** Comprehensive changelog, activity log, and API documentation

The system provides actionable insights for education planning and policy-making in India.
