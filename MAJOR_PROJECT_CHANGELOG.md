# Major Project — Change Log & Documentation

## Project Overview

**Title:** Higher Education Enrollment Forecasting (AISHE)
**Goal:** Forecast higher-education enrollment for 35 Indian states using machine learning
**Data Source:** AISHE (All India Survey on Higher Education) — NDAP_REPORT_9259.csv
**Best Model:** XGBoost (MAE=43,467, RMSE=69,042, MAPE=62.38%, R²=0.9975)

---

## 1. ML Pipeline (Existing — No Changes to Model Logic)

### Dataset
- **Raw data:** 11,232 rows × 18 columns (AISHE 2012-2019)
- **ML dataset:** 210 rows × 20 columns (35 states × 6 years: 2014-2019)
- **Target:** `higher_education_enrollment` (Total, all categories, both genders)
- **Train:** 2014-2017 | **Test:** 2018-2019

### Features (15 numeric + 1 categorical)
| Feature | Description |
|---------|-------------|
| enrollment_lag_1 | Previous year enrollment |
| enrollment_lag_2 | 2-year lag enrollment |
| enrollment_growth | Year-over-year growth rate |
| rolling_mean | 3-year rolling mean |
| teachers_lag1 | Previous year teachers |
| colleges_lag1 | Previous year colleges |
| universities_lag1 | Previous year universities |
| stand_alone_institutions_lag1 | Previous year stand-alone institutions |
| foreign_students_lag1 | Previous year foreign students |
| university_enrolment_lag1 | Previous year university enrollment |
| GER_lag1 | Gross Enrolment Ratio (lag-1) |
| GPI_lag1 | Gender Parity Index (lag-1) |
| PTR_all_lag1 | Pupil-Teacher Ratio all (lag-1) |
| PTR_university_colleges_lag1 | PTR university/colleges (lag-1) |
| PTR_university_units_lag1 | PTR university units (lag-1) |
| State | Categorical (one-hot encoded) |

### Model Comparison (Test Set 2018-2019)
| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| **XGBoost** | **43,467** | **69,042** | **62.38** | **0.9975** |
| ARIMA | 44,474 | 74,234 | 6.76 | 0.9971 |
| Linear Regression | 55,947 | 79,869 | 429.83 | 0.9966 |
| TBATS | 51,703 | 101,668 | 6.50 | 0.9945 |
| ETS | 50,192 | 90,757 | 7.33 | 0.9956 |
| LightGBM | 114,163 | 188,858 | 57.36 | 0.9810 |
| Random Forest | 94,583 | 225,782 | 186.50 | 0.9729 |
| SVR | 927,410 | 1,459,550 | 3,565.60 | -0.1344 |
| WOASVR | 920,433 | 1,451,195 | 3,555.24 | -0.1214 |
| GWOSVR | 927,013 | 1,459,194 | 3,568.15 | -0.1338 |
| PSOSVR | 920,438 | 1,451,199 | 3,555.19 | -0.1214 |
| ActualGRIDSVR | 921,197 | 1,451,881 | 3,549.84 | -0.1225 |

### XGBoost Hyperparameters (Best from Temporal CV)
```json
{
  "subsample": 0.8,
  "n_estimators": 300,
  "max_depth": 3,
  "learning_rate": 0.05,
  "colsample_bytree": 1.0
}
```

### Feature Importance (XGBoost)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | enrollment_lag_1 | 0.8338 |
| 2 | colleges_lag1 | 0.1122 |
| 3 | rolling_mean | 0.0311 |
| 4 | teachers_lag1 | 0.0076 |
| 5 | GPI_lag1 | 0.0039 |

### Other Model Hyperparameters
| Model | Key Params |
|-------|------------|
| SVR | kernel=rbf, gamma=scale, epsilon=0.05, C=10 |
| Random Forest | n_estimators=100, max_depth=5, min_samples_leaf=2, max_features=0.5 |
| LightGBM | num_leaves=63, n_estimators=100, max_depth=10, learning_rate=0.1 |
| Hybrid SVR (WOA/GWO/PSO) | C=1000, epsilon varies, gamma varies (metaheuristic-tuned) |

---

## 2. Forecasting (2020-2030)

### Method
- **Stage 1:** Per-state linear trends for 6 count indicators (teachers, colleges, universities, stand-alone institutions, foreign students, university enrollment)
- **Stage 2:** XGBoost retrained on all 2014-2019 data, recursive forecast
- **Ratio indicators** (GER, GPI, PTRs): held at 2019 baseline (base scenario), with ±5% sensitivity (lower/upper)
- **Scenarios:** Base_2019_Ratio, Lower_Ratio_Scenario (0.95×), Upper_Ratio_Scenario (1.05×)

### National Aggregate (Base Scenario)
| Year | Enrollment |
|------|------------|
| 2019 (actual) | 38,228,258 |
| 2020 | 39,126,568 |
| 2021 | 39,464,089 |
| 2022 | 39,702,441 |
| 2023 | 39,989,498 |
| 2024 | 40,030,167 |
| 2030 (XGBoost) | ~44.5M (varies by state) |

### 2030 Forecast (XGBoost, Beta)
- Uses ensemble of XGBoost + ARIMA + ETS + TBATS + Linear Trend
- Per-state forecasts with lower/upper bounds
- Available for all 35 states

---

## 3. Changes Made (New Additions)

### A. FastAPI Backend (`api/`)
**New files created:**
- `api/main.py` — FastAPI app with CORS (port 8000)
- `api/services/data_service.py` — Read-only data layer (CSVs + models)
- `api/routers/states.py` — State endpoints
- `api/routers/models.py` — Model comparison, feature importance
- `api/routers/forecast.py` — Forecast endpoints
- `api/routers/predict.py` — Custom prediction endpoint (XGBoost inference)
- `api/routers/forecast_2030.py` — 2030 forecast endpoints
- `api/routers/forecast_by_year.py` — Year-specific forecast endpoints

**API Endpoints:**
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

### B. Next.js Frontend (`frontend/`)
**New files created:**
- `frontend/app/page.tsx` — Main dashboard (5 tabs)
- `frontend/app/layout.tsx` — Root layout with dark mode
- `frontend/app/globals.css` — Dark theme styles
- `frontend/components/map/IndiaMap.tsx` — Leaflet choropleth map
- `frontend/components/map/IndiaMapWrapper.tsx` — Dynamic import wrapper
- `frontend/components/common/Skeleton.tsx` — Loading skeletons
- `frontend/components/common/EmptyState.tsx` — Empty/error states
- `frontend/components/providers.tsx` — React Query + Sonner
- `frontend/lib/api.ts` — API client + TypeScript types
- `frontend/lib/utils.ts` — Utility functions
- `frontend/public/geo/india_states.geojson` — India state boundaries (9.8MB)

**Frontend Tabs:**
| Tab | Content |
|-----|---------|
| Overview | KPI cards, top 5 states, model summary |
| Map | Interactive India choropleth (enrollment 2019 / forecast by year) |
| Models | MAE bar chart + full metrics table |
| Forecast | 2030 projection table + national aggregates |
| Predict | Custom forecast form (input indicators → XGBoost prediction) |
| Features | Feature importance by model |

**Design:** Dark mode, minimalistic, Inter font, Tailwind CSS, responsive

### C. Map Integration
- **Library:** Leaflet + react-leaflet
- **Tiles:** OpenStreetMap (free, no API key required)
- **GeoJSON:** 35 Indian states with name mapping to dataset
- **Features:** Hover tooltips, click selection, year selector, legend, dark CSS filter
- **Color:** Indigo gradient (low → high enrollment)

### D. Run Scripts
- `run_fullstack.sh` — Starts both backend (port 8000) and frontend (port 3000)
- `run_pipeline.py` — Original ML pipeline (unchanged)

---

## 4. How to Run

### Prerequisites
```bash
pip install -r requirements.txt  # pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, joblib, scipy, streamlit, fastapi, uvicorn
cd frontend && npm install
```

### Run ML Pipeline (if needed)
```bash
python3 run_pipeline.py
```

### Run Full-Stack App
```bash
./run_fullstack.sh
# OR manually:
# Terminal 1: python3 -m uvicorn api.main:app --port 8000
# Terminal 2: cd frontend && npm run dev -- -p 3000
```

### Run Original Streamlit Dashboard
```bash
streamlit run dashboard.py
```

---

## 5. File Structure (New + Modified)

```
mini_proj/
├── api/                          # NEW: FastAPI backend
│   ├── main.py
│   ├── services/data_service.py
│   └── routers/
│       ├── states.py
│       ├── models.py
│       ├── forecast.py
│       ├── predict.py
│       ├── forecast_2030.py
│       └── forecast_by_year.py
├── frontend/                     # NEW: Next.js 14 frontend
│   ├── app/page.tsx
│   ├── app/layout.tsx
│   ├── app/globals.css
│   ├── components/
│   │   ├── map/IndiaMap.tsx
│   │   ├── map/IndiaMapWrapper.tsx
│   │   ├── common/Skeleton.tsx
│   │   ├── common/EmptyState.tsx
│   │   └── providers.tsx
│   ├── lib/api.ts
│   ├── lib/utils.ts
│   └── public/geo/india_states.geojson
├── run_fullstack.sh              # NEW: Convenience script
├── requirements.txt              # MODIFIED: Added fastapi, uvicorn
├── src/                          # UNCHANGED: ML pipeline
├── outputs/                      # UNCHANGED: Pipeline outputs
├── models/                       # UNCHANGED: Trained models
├── dashboard.py                  # UNCHANGED: Streamlit dashboard
└── run_pipeline.py               # UNCHANGED: ML pipeline runner
```

---

## 6. No Breaking Changes

- **Backend ML pipeline:** Untouched (src/, outputs/, models/)
- **Streamlit dashboard:** Still works (`streamlit run dashboard.py`)
- **API contracts:** New FastAPI layer is read-only, reads existing CSVs/models
- **Database schemas:** No changes (CSV-based, no DB)
- **Model files:** No changes (.joblib files unchanged)

---

## 7. Key Technical Decisions

1. **XGBoost selected as primary model** — lowest average rank across MAE, RMSE, MAPE, R²
2. **OpenStreetMap tiles** — free, no API key required (replaced CARTO which had rate limits)
3. **Pre-computed forecasts** — 2020-2030 forecasts pre-computed for performance; custom predictions run on-demand
4. **Dark mode** — CSS variables + Tailwind dark class strategy
5. **Read-only API** — FastAPI only reads existing outputs, no writes to pipeline data
6. **GeoJSON simplified** — 22MB → 9.8MB (coordinate precision reduction) for faster loading
