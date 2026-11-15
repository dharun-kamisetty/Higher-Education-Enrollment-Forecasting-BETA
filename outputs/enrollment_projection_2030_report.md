# India Higher Education Enrollment Projection 2030

## 1. Objective

This project estimates higher education enrollment for Indian states through 2030 using historical AISHE trends (2012-13 to 2019-20).
The projection is based on a multi-model ensemble combining XGBoost, ARIMA, ETS, TBATS, and Linear Trend models.

## 2. Data

- **Source**: NDAP Report 9259 (AISHE)
- **Historical Period**: 2012-13 to 2019-20 (8 financial years)
- **States**: 35 states (Ladakh excluded due to single-year data)
- **ML Modeling Period**: 2014-2019 (6 years per state, 210 observations)
- **Target**: Total higher education enrollment (Gender=Total, Items=Total)
- **Limitations**: Short time series (5-7 observations per state); 2012 only has Haryana enrollment; no demographic/policy variables.

## 3. Methodology

### ML Forecasting (XGBoost)
- Two-stage approach: Stage 1 projects exogenous indicators (teachers, colleges, etc.) via linear trends; Stage 2 uses recursive XGBoost with lag features.
- Ratio indicators (GER, GPI, PTRs) held at 2019 baseline with ±5% sensitivity.
- Exogenous indicators projected to 2030 via per-state linear trends on 2013-2019 history (fallback to last observed if unstable/insufficient data).
- Model retrained on full 2014-2019 data.

### Time-Series Models
- **ARIMA**: auto_arima with AIC, max p=2, d=1, q=2, no seasonality.
- **ETS**: Additive error, additive trend, no seasonality, no damping.
- **TBATS**: Trend with damped trend, no Box-Cox, no seasonal periods.
- All fit on 2013-2017 training, evaluated on 2018-2019.

### Trend Model
- Simple linear regression (y = a + b*t) on enrollment 2013-2019 per state.
- Projections clipped at zero.

### Backtesting
- Train: 2014-2017, Test: 2018-2019 (chronological, no leakage).
- All models evaluated on MAE, RMSE, MAPE, R² per state.

### Ensemble
- Weights from backtest performance (inverse MAE/RMSE/MAPE, direct R²): ARIMA=0.176, ETS=0.158, LightGBM=0.054, Linear Regression=0.095, Linear_Trend=0.154, Random Forest=0.059, SVR=0.006, TBATS=0.172, XGBoost=0.127
- Better validation performance → higher weight.
- Models with unstable long-horizon behavior (XGBoost recursive) included but monitored.

### Exogenous Variables (2020-2030)
- 6 count indicators: linear trend per state (2013-2019), clipped at 0, instability cap at 3× max historical.
- Ratio indicators: held at 2019 value ±5% (scenarios).

## 4. Validation Results (2018-2019 Backtest)

| model | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- |
| ARIMA | 44474.21 | 50146.37 | 6.76 | -1029.98 |
| ETS | 50192.21 | 56900.91 | 7.33 | -290.94 |
| LightGBM | 114162.94 | 121700.74 | 57.36 | -26203.18 |
| Linear Regression | 55947.13 | 60893.88 | 429.83 | -2774976.87 |
| Linear_Trend | 54052.61 | 59051.72 | 7.07 | -386.38 |
| Random Forest | 94582.75 | 99223.57 | 186.5 | -556082.06 |
| SVR | 927410.36 | 928142.02 | 3565.6 | -167219521.13 |
| TBATS | 46723.7 | 51959.07 | 6.67 | -134.69 |
| XGBoost | 43467.13 | 47719.53 | 62.38 | -61233.15 |

## 5. 2030 Projection

### India-Level Summary
- **Historical 2019 Enrollment**: 38,228,258
- **Projected 2030 Enrollment (Baseline)**: 44,340,281
- **Conservative**: 37,413,713
- **Optimistic**: 52,249,100
- **Projected Growth 2019→2030**: 16.0%

### Top 10 States by Projected 2030 Enrollment (Baseline)

| state | enrollment_2019 | conservative_2030 | baseline_2030 | optimistic_2030 |
| --- | --- | --- | --- | --- |
| Uttar Pradesh | 6388214.0 | 6290198.0 | 6900628.0 | 8194698.0 |
| Maharashtra | 4265472.0 | 4308879.0 | 5035648.0 | 5661644.0 |
| Tamil Nadu | 3520311.0 | 3520311.0 | 3736369.0 | 4043044.0 |
| Madhya Pradesh | 2182154.0 | 2216442.0 | 2947143.0 | 3974817.0 |
| Karnataka | 2187892.0 | 2250021.0 | 2789326.0 | 3770792.0 |
| Rajasthan | 2206517.0 | 2251480.0 | 2753017.0 | 3175509.0 |
| West Bengal | 2160893.0 | 2198344.0 | 2595091.0 | 2839985.0 |
| Bihar | 1738432.0 | 1801307.0 | 2151950.0 | 2466813.0 |
| Andhra Pradesh | 1897149.0 | 1876065.0 | 1939175.0 | 2085448.0 |
| Gujarat | 1544840.0 | 1573662.0 | 1736631.0 | 1884626.0 |

## 6. Limitations

- **Short Historical Time Series**: Only 5-7 observations per state for trend estimation.
- **Long Forecast Horizon**: 11-year projection (2020-2030) from 2019 base; recursive error accumulation likely.
- **Policy Changes Not Included**: NEP 2020, COVID impacts, demographic shifts not explicitly modeled.
- **Exogenous Variables**: Linear trend assumption may not hold for colleges/teachers over 11 years.
- **No Population/Demographic Drivers**: Enrollment growth tied to institution capacity, not demand-side factors.
- **XGBoost Recursive Stability**: Long-horizon recursive forecasts may drift; ensemble mitigates but doesn't eliminate.
- **COVID-Era Disruption**: 2020-2022 actual enrollment affected by pandemic; projections assume trend continuation.

## 7. Interpretation

**This is a historical-trend-based projection and should NOT be interpreted as an exact prediction of actual 2030 enrollment.**

The projections represent what enrollment *would be* if historical trends in institutional capacity (colleges, teachers, universities) and enrollment rates (GER, GPI, PTR) continue linearly from 2013-2019 patterns. Actual 2030 enrollment will depend on policy decisions, economic conditions, demographic changes, and educational reforms not captured in this model.

### Growth Rate Flags (States with Projected CAGR >2x Historical)

None flagged.

---
*Report generated from AISHE data through 2019-20. Forecast methodology documented in outputs/forecast_methodology.txt*