# Project Activity Log — Higher Education Enrollment Forecasting

## Project Title
**Higher Education Enrollment Forecasting for Indian States using Machine Learning and Full-Stack Deployment**

## Project Overview
A comprehensive machine learning project that forecasts higher-education enrollment across 35 Indian states using AISHE (All India Survey on Higher Education) data. The project evolved from a standalone ML pipeline into a full-stack web application with an interactive dashboard, Indian state map, AI assistant, and REST API.

---

## Phase 1: ML Pipeline Development (Existing)

### 1.1 Data Understanding & Inspection
- Loaded raw AISHE dataset (`NDAP_REPORT_9259.csv`): 11,232 rows × 18 columns
- Identified 36 states/UTs, 8 financial years (2012-2019), 12 item categories, 3 gender values
- Determined target variable: `higher_education_enrollment` (Total gender, All Categories)
- Found 247 non-null enrollment records after filtering (Items=Total, Gender=Total)

### 1.2 Data Cleaning & Preprocessing
- Filtered data to Items=Total & Gender=Total
- Renamed columns using COLUMN_RENAME mapping (12 columns)
- Handled missing values: median imputation for numeric, most_frequent for state
- Detected and handled suspicious values
- Output: `outputs/cleaned_dataset.csv`

### 1.3 Feature Engineering
- Created State-Year level ML dataset: 35 states × 6 years = 210 observations
- Generated 15 features:
  - Lag-1 features: enrollment, teachers, colleges, universities, GER, GPI, PTRs
  - Lag-2 enrollment, enrollment growth rate, 3-year rolling mean
  - Stand-alone institutions, foreign students, university enrollment lags
- Excluded 1 state (EXCLUDED_STATES)
- Output: `outputs/ml_dataset.csv` (210 × 20)

### 1.4 Baseline Model Training
- Trained 5 models on 2014-2017, tested on 2018-2019:
  - Linear Regression, SVR, Random Forest, XGBoost, LightGBM
- XGBoost emerged as best baseline (MAE=39,770, R²=0.9978)

### 1.5 Hyperparameter Tuning
- Temporal TimeSeriesSplit cross-validation on 2014-2017
- Grid search for each model
- Saved best params to `models/best_params.json`

### 1.6 Final Evaluation & Model Selection
- Retrained tuned models on full 2014-2017, evaluated on 2018-2019
- **XGBoost selected as best model** (lowest average rank across MAE, RMSE, MAPE, R²)
- Metrics: MAE=43,467, RMSE=69,042, MAPE=62.38%, R²=0.9975
- Saved: `model_comparison.csv`, `predictions.csv`, trained models (.joblib)

### 1.7 Hybrid Models (Metaheuristic SVR)
- Implemented Whale Optimization Algorithm (WOA), Grey Wolf Optimization (GWO), Particle Swarm Optimization (PSO)
- Tuned SVR hyperparameters (C, epsilon, gamma) minimizing CV RMSE
- Added exhaustive grid search (ActualGRIDSVR) as baseline
- All SVR variants underperformed (negative R² ≈ -0.12)

### 1.8 Time-Series Models
- Per-state univariate models: ARIMA (auto_arima), ETS (additive trend), TBATS
- Trained on each state's 2013-2017 enrollment, forecasted 2018-2019
- ARIMA: MAE=44,474, R²=0.9971 (closest to XGBoost)
- All 35 states successfully fitted (no fallbacks)

### 1.9 Feature Importance Analysis
- Extracted feature importance from Random Forest, XGBoost, LightGBM
- Key finding: `enrollment_lag_1` is dominant predictor (>80% for XGBoost)
- Saved: `feature_importance.csv`

### 1.10 Two-Stage Forecasting (2020-2024)
- **Stage 1:** Per-state linear trends for 6 count indicators
  - teachers, colleges, universities, stand_alone_institutions, foreign_students, university_enrolment
  - Fallback: last observed value when <3 observations or projection >3× historical max
  - Negative projections clipped to zero
- **Stage 2:** Recursive XGBoost forecast
  - Retrained on all 2014-2019 data
  - Year-by-year recursive prediction (predicted enrollment fed back as lag)
- **3 scenarios:** Base (2019 ratio), Lower (0.95×), Upper (1.05×)
- Ratio indicators (GER, GPI, PTRs) held at 2019 baseline
- Output: `enrollment_forecasts.csv`, `enrollment_series.csv`, `indicator_trend_summary.csv`

### 1.11 2030 Forecasting (Beta)
- Extended forecast horizon to 2030 using ensemble approach
- Combines XGBoost + ARIMA + ETS + TBATS + Linear Trend
- Per-state forecasts with lower/upper bounds
- Output: `enrollment_projection_2020_2030.csv`, `enrollment_scenarios_2030.csv`

---

## Phase 2: Full-Stack Application Development

### 2.1 FastAPI Backend (`api/`)
**Created files:**
- `api/main.py` — FastAPI app with CORS middleware, dotenv loading
- `api/services/data_service.py` — Read-only data layer (CSVs + models, lru_cache)
- `api/routers/states.py` — State listing, state data, latest enrollment, forecasts
- `api/routers/models.py` — Model comparison, best model selection, feature importance, params
- `api/routers/forecast.py` — All forecasts, methodology, indicator trends
- `api/routers/predict.py` — Custom XGBoost prediction (user inputs → enrollment forecast)
- `api/routers/forecast_2030.py` — 2030 forecast endpoints (by state, all states)
- `api/routers/forecast_by_year.py` — Year-specific forecasts (2020-2030)
- `api/routers/assistant.py` — EduCast AI assistant endpoint (Groq Cloud)

**API Endpoints (14 total):**
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
| GET | /api/forecast/all | All forecasts (2022) |
| GET | /api/forecast/2030 | 2030 XGBoost forecasts |
| GET | /api/forecast/by-year/{year} | Forecasts for specific year |
| GET | /api/forecast/available-years | Available years (2020-2030) |
| GET | /api/forecast/state-timeline/{state} | Full timeline per state |
| POST | /api/predict | Custom XGBoost prediction |
| POST | /api/assistant/chat | EduCast AI chat |

### 2.2 Next.js Frontend (`frontend/`)
**Created files:**
- `frontend/app/page.tsx` — Main dashboard (6 tabs)
- `frontend/app/layout.tsx` — Root layout with dark mode, EduCast widget
- `frontend/app/globals.css` — Dark theme styles, Leaflet overrides
- `frontend/components/map/IndiaMap.tsx` — Leaflet choropleth map
- `frontend/components/map/IndiaMapWrapper.tsx` — Dynamic import (SSR disabled)
- `frontend/components/common/Skeleton.tsx` — Loading skeletons
- `frontend/components/common/EmptyState.tsx` — Empty/error states
- `frontend/components/providers.tsx` — React Query + Sonner toasts
- `frontend/components/EduCastChat.tsx` — AI assistant chat widget
- `frontend/lib/api.ts` — API client + TypeScript types
- `frontend/lib/utils.ts` — Utility functions (formatNumber, etc.)
- `frontend/public/geo/india_states.geojson` — India state boundaries (9.8MB)

**Frontend Tabs:**
| Tab | Content |
|-----|---------|
| Overview | KPI cards, top 5 states, model summary |
| Map | Interactive India choropleth with year selector |
| Models | MAE bar chart + full metrics table |
| Forecast | 2030 projection table + national aggregates |
| Predict | Custom forecast form (indicator inputs → XGBoost) |
| Features | Feature importance by model |

**Design System:**
- Dark mode (CSS variables + Tailwind dark class)
- Inter font, minimalistic aesthetic
- Indigo accent color (#6366f1)
- Responsive grid layout
- Loading skeletons, empty states, error handling

### 2.3 Indian State Map Integration
- **Library:** Leaflet + react-leaflet (v4)
- **Tiles:** OpenStreetMap (free, no API key)
- **GeoJSON:** 35 Indian states (simplified from 22MB to 9.8MB)
- **State name mapping:** GeoJSON names → dataset names (35 mappings)
- **Features:**
  - Choropleth colored by enrollment/forecast
  - Hover tooltips with state name + value
  - Click to select state
  - Year selector dropdown (2020-2030)
  - Legend with min/max values
  - Dark CSS filter for map tiles
  - Zoom/reset controls

### 2.4 EduCast AI Assistant
- **Name:** EduCast (Education + Forecast)
- **Backend:** Groq Cloud API (openai/gpt-oss-120b model)
- **System prompt:** Contains all project facts (metrics, features, methodology, state data)
- **Frontend:** Floating chat widget (bottom-right corner)
- **Features:** Conversation history, loading indicator, auto-scroll, Enter to send
- **Response style:** Concise (max 300 tokens, temperature 0.3)
- **API key:** Stored in .env (gitignored)

### 2.5 Deployment & Infrastructure
- **Run script:** `run_fullstack.sh` (starts both backend + frontend)
- **Backend:** Uvicorn on port 8000
- **Frontend:** Next.js dev server on port 3000
- **API Docs:** Swagger UI at /docs
- **Original Streamlit:** Still works (`streamlit run dashboard.py`)

---

## Phase 3: Bug Fixes & Refinements

### 3.1 Map "Requires API Key" Issue
- **Problem:** CARTO dark tiles returned "requires API Key" errors
- **Fix:** Switched to OpenStreetMap tiles + CSS dark filter
- **Result:** Map renders without API key

### 3.2 Map "No Data" on Hover
- **Problem:** GeoJSON component doesn't re-render when data changes; tooltip bound once with empty data
- **Fix:** Used useRef for current data + setTooltipContent() in mouseover handler
- **Fix:** Added key prop to force re-render when data changes
- **Result:** Hover tooltips show correct enrollment/forecast values

### 3.3 2030 Forecasting Integration
- **Added:** Year selector on map (2020-2030 dropdown)
- **Added:** 2030 forecast API endpoints
- **Updated:** Forecast tab to show 2030 projections
- **Updated:** Map to show selected year's forecasts
- **Beta badge:** Added to header

### 3.4 EduCast System Prompt Enhancement
- **Problem:** Assistant said "I don't have that info" for state-specific questions
- **Fix:** Added all 35 states with 2019 enrollment values
- **Fix:** Added full model comparison metrics
- **Fix:** Added top 5 states by 2030 forecast
- **Result:** Assistant now answers state-specific and model-specific questions

---

## Phase 4: Documentation & Sharing

### 4.1 Changelog Document
- Created `MAJOR_PROJECT_CHANGELOG.md` (265 lines)
- Covers: ML pipeline, model metrics, API endpoints, frontend, map, deployment

### 4.2 GitHub Repository
- **Repository:** https://github.com/dharun-kamisetty/Higher-Education-Enrollment-Forecasting-BETA
- **Updated .gitignore:** Excluded node_modules, .next, .env, raw data; included models, outputs, source code
- **Pushed:** 2 commits (initial + major upgrade)
- **Total files:** 103+ files committed

### 4.3 Activity Log
- This document (`PROJECT_ACTIVITY_LOG.md`)

---

## Summary of All Files Created/Modified

### New Files (60+)
| Category | Count | Key Files |
|----------|-------|-----------|
| API Backend | 10 | main.py, data_service.py, 8 routers |
| Frontend | 16 | page.tsx, IndiaMap.tsx, EduCastChat.tsx, etc. |
| Documentation | 3 | MAJOR_PROJECT_CHANGELOG.md, PROJECT_ACTIVITY_LOG.md, REPORT.md |
| Config | 2 | .env, run_fullstack.sh |

### Modified Files
| File | Change |
|------|--------|
| requirements.txt | Added fastapi, uvicorn, python-dotenv, groq |
| .gitignore | Updated for project sharing |

### Unchanged Files
| Category | Files |
|----------|-------|
| ML Pipeline | src/ (13 scripts), run_pipeline.py |
| Models | models/ (12 .joblib + best_params.json) |
| Outputs | outputs/ (CSVs, plots, reports) |
| Streamlit | dashboard.py |

---

## Key Metrics Summary

### Best Model (XGBoost)
| Metric | Value |
|--------|-------|
| MAE | 43,467 |
| RMSE | 69,042 |
| MAPE | 62.38% |
| R² | 0.9975 |

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 1.0 |

### Dataset
| Metric | Value |
|--------|-------|
| States | 35 |
| Years | 2014-2019 (6 years) |
| Observations | 210 |
| Features | 15 + 1 categorical |
| Train period | 2014-2017 |
| Test period | 2018-2019 |

### Forecast
| Metric | Value |
|--------|-------|
| Horizon | 2020-2030 |
| Scenarios | 3 (Base, Lower, Upper) |
| 2019 actual (national) | 38,228,258 |
| 2030 forecast (XGBoost) | ~44.5M (aggregate) |

### Tech Stack
| Layer | Technology |
|-------|------------|
| ML | Python, scikit-learn, XGBoost, LightGBM |
| Backend | FastAPI, Uvicorn, python-dotenv, Groq |
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS |
| Maps | Leaflet, react-leaflet, OpenStreetMap |
| Charts | Recharts |
| AI | Groq Cloud (openai/gpt-oss-120b) |
| Data | Pandas, NumPy, Joblib |

---

## Conclusion

The project evolved from a standalone ML pipeline into a comprehensive full-stack application with:
1. **12 trained ML models** with XGBoost as the best (R²=0.9975)
2. **REST API** with 14 endpoints for data access and prediction
3. **Interactive dashboard** with 6 tabs, dark mode, and responsive design
4. **Indian state map** with choropleth visualization and year selection
5. **AI assistant** (EduCast) for project Q&A
6. **2030 forecasting** (beta) with per-state projections
7. **Comprehensive documentation** and GitHub repository

All components are production-ready and documented for team collaboration.
