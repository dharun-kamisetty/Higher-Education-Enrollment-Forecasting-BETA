# Comparative Machine Learning and Hybrid Forecasting Framework for Higher Education Enrollment Prediction Using State-Level Educational Indicators from India

## Abstract

This study develops a comparative machine learning and hybrid forecasting framework for higher education enrollment across Indian states and Union Territories using Government of India data from the National Data and Analytics Platform (NDAP Report 9259). The dataset spans financial years 2012–13 to 2019–20, yielding six usable modeling years (2014–2019) across 35 states (210 state–year observations). Five standalone regression models—Linear Regression, Support Vector Regression (SVR), Random Forest, XGBoost, and LightGBM—are evaluated alongside four hybrid SVR variants optimized via Whale Optimization Algorithm (WOA-SVR), Grey Wolf Optimizer (GWO-SVR), Particle Swarm Optimization (PSO-SVR), and an exhaustive grid-search baseline (ActualGRIDSVR). Three univariate time-series models (ARIMA, ETS, TBATS) are benchmarked per state. Models are validated chronologically (train: 2014–2017; test: 2018–2019) using MAE, RMSE, MAPE, and R². XGBoost achieves the lowest average rank across all metrics (MAE=43,467; RMSE=69,042; MAPE=62.4%; R²=0.9975), though the high MAPE reflects substantial relative errors for smaller states despite strong overall fit. All hybrid SVR variants underperform standalone tree-based models, yielding negative R² on the test period. A two-stage forecasting framework generates short-to-medium-term data-driven projections (2020–2022 primary; 2020–2024 extended): Stage 1 extrapolates count-based indicators via state-level linear trends with documented fallbacks; Stage 2 recursively forecasts enrollment using XGBoost with scenario-based ratio indicators held at 2019 values. Aggregate base projections estimate national enrollment at 39.1M (2020), 39.5M (2021), and 39.7M (2022). Key limitations include sparse historical observations (only six modeling years per state), high MAPE despite strong R² due to cross-state scale heterogeneity, frozen ratio-indicator assumptions, and recursive error accumulation in multi-step forecasts.

**Keywords**—Higher education enrollment forecasting, machine learning, hybrid optimization, metaheuristics, chronological validation, educational planning, NDAP India

---

## 1. Introduction

Higher education enrollment forecasting is a critical input for educational planning at national and state levels. Reliable projections inform decisions on infrastructure expansion, faculty recruitment, institutional capacity, budget allocation, and long-term policy design. In India, where higher education spans diverse states with varying demographic and institutional landscapes, state-level forecasting enables targeted resource planning.

Traditional enrollment forecasting has relied on statistical time-series methods or cohort-survival models. Recent literature demonstrates growing adoption of machine learning (ML) techniques, including support vector regression, random forests, and gradient boosting, often combined with metaheuristic hyperparameter optimization. However, many studies use institution-specific data or limited predictor sets, and few evaluate hybrid approaches against standalone ML models on the same chronological benchmark.

This study constructs a state–year panel dataset from NDAP Report 9259, integrating multiple educational indicators—enrollment history, teacher statistics, institutional counts, university enrollment, foreign students, Gender Parity Index (GPI), Gross Enrolment Ratio (GER), and Pupil-Teacher Ratios (PTR)—into a unified forecasting framework. We compare five standalone ML regressors, four hybrid SVR variants, and three univariate time-series models under a rigorous chronological split. The best model drives a two-stage forecasting framework that explicitly addresses the limitation of freezing non-target indicators during recursive prediction.

**Contributions:**

1. A state-level higher education enrollment dataset constructed from NDAP Report 9259, integrating 14 educational indicators across 35 states and 6 modeling years (210 observations).
2. Comparative evaluation of Linear Regression, SVR, Random Forest, XGBoost, and LightGBM using chronological validation and multi-metric ranking.
3. Analysis of four hybrid SVR variants (WOA-SVR, GWO-SVR, PSO-SVR, ActualGRIDSVR) with metaheuristic and grid-search hyperparameter optimization.
4. Benchmarking against per-state univariate time-series models (ARIMA, ETS, TBATS).
5. Identification and correction of the "frozen indicator" problem in recursive forecasting via a two-stage framework separating input-feature projection from enrollment prediction.
6. Short-to-medium-term enrollment projections (2020–2024) with scenario analysis, acknowledging data limitations that preclude reliable long-horizon forecasts.

---

## 2. Literature Survey

**Enrollment Forecasting Methods:** Yang et al. (2020) proposed a Whale Optimization Algorithm–Support Vector Regression (WOA-SVR) hybrid for student enrollment and teacher statistics forecasting, demonstrating improved accuracy over standalone SVR. Other studies have applied ARIMA, neural networks, and ensemble methods to enrollment prediction, typically at institutional or national aggregate levels.

**Machine Learning in Educational Forecasting:** Random Forest and gradient boosting (XGBoost, LightGBM) have shown strong performance in tabular regression tasks with mixed feature types. Metaheuristic optimization (WOA, GWO, PSO) has been applied to SVR hyperparameter tuning in various domains, often reporting gains over default or grid-search configurations.

**Indian Context:** Government of India educational data via NDAP provides state-level annual indicators but has not been widely used for ML-based enrollment forecasting with multi-indicator feature sets and hybrid model comparison.

**Research Gap:** Existing work lacks (a) a comprehensive multi-indicator state-level dataset from NDAP, (b) side-by-side comparison of standalone ML, hybrid metaheuristic-SVR, and univariate time-series models under identical chronological validation, and (c) a forecasting methodology that decouples future input-feature generation from recursive enrollment prediction to avoid the frozen-indicator artifact.

---

## 3. Dataset Description and Preprocessing

### 3.1 Data Source

**NDAP Report 9259** (National Data and Analytics Platform, Government of India) provides annual state-level higher education statistics for financial years 2012–13 through 2019–20. The raw file contains 11,233 rows with columns: Country, State LGD Code, State, Financial Year, Gender, Items, and 13 indicator columns.

### 3.2 Data Characteristics and Limitations

- **Temporal coverage:** 8 financial years (2012–13 to 2019–20). Year 2012 contains data for only one state (Haryana); 2013 onward has broad state coverage.
- **State coverage:** ~34 states/UTs have continuous sequences 2013–2019. Ladakh excluded (single-year observation).
- **Modeling period:** First usable year is 2014 (lag features require 2013 history). Final ML dataset: 2014–2019 (6 years × 35 states = 210 observations).
- **Train/Test split:** Chronological—Training: 2014–2017 (140 obs); Testing: 2018–2019 (70 obs).

### 3.3 Preprocessing Pipeline

1. **Gender selection:** Only `Gender = Total` rows retained to avoid double-counting male/female enrollment.
2. **Pivot restructuring:** The `Items` column encodes indicator categories. Each indicator is extracted from its corresponding `Items` value (e.g., `Total` for enrollment, `All Categories` for GER/GPI, `Regular Mode` for PTR) and merged into a single state–year row.
3. **Teacher aggregation:** Teacher count constructed by summing five disjoint academic ranks: Professor & Equivalent, Reader & Associate Professor, Lecturer/Assistant Professor, Demonstrator/Tutor, Temporary Teacher etc.
4. **Cleaning:** State-name standardization, financial-year parsing to numeric `YearCode`, missing-value handling (markers: `<br>`, empty, `NA`, `N/A`, `null`, `NaN`, `nan`), numeric coercion, duplicate removal, state-consistency checks.
5. **Exclusions:** Ladakh removed; states with insufficient history dropped.

### 3.4 Feature Engineering

**Target:** `higher_education_enrollment` (from `Items = Total` enrollment column).

**Features (all lagged by 1 year within state to prevent leakage):**

| Category | Features |
|----------|----------|
| Categorical | State (one-hot encoded) |
| Temporal | Year |
| Enrollment lags | `enrollment_lag_1`, `enrollment_lag_2`, `enrollment_growth`, `rolling_mean` (3-year) |
| Indicator lags | `university_enrolment_lag1`, `foreign_students_lag1`, `stand_alone_institutions_lag1`, `teachers_lag1`, `colleges_lag1`, `universities_lag1`, `GPI_lag1`, `GER_lag1`, `PTR_all_lag1`, `PTR_university_colleges_lag1`, `PTR_university_units_lag1` |

Lagged construction uses state-wise chronological shifting (`groupby("State").shift(1)`), ensuring only pre-prediction-year information is available.

---

## 4. Proposed Methodology

### 4.1 Standalone Machine Learning Models

Five regression algorithms representing diverse approaches:

| Model | Type | Key Configuration |
|-------|------|-------------------|
| Linear Regression | Linear statistical | Default |
| SVR | Kernel-based | RBF kernel, hyperparameters tuned |
| Random Forest | Bagging ensemble | 100 trees, tuned via RandomizedSearchCV |
| XGBoost | Gradient boosting | 100 estimators, max_depth=6, tuned |
| LightGBM | Light gradient boosting | 100 estimators, tuned |

**Hyperparameter tuning:** RandomizedSearchCV with TimeSeriesSplit (n_splits = years−1 = 3) on training data (2014–2017), optimizing negative RMSE. Test set (2018–2019) held out entirely.

### 4.2 Hybrid Model Analysis

Four SVR variants with optimized hyperparameters (C, ε, γ):

1. **WOA-SVR:** Whale Optimization Algorithm (10 agents, 15 iterations).
2. **GWO-SVR:** Grey Wolf Optimizer (10 agents, 15 iterations).
3. **PSO-SVR:** Particle Swarm Optimization (10 agents, 15 iterations).
4. **ActualGRIDSVR:** Exhaustive grid search (C ∈ {0.1,1,10,100,1000}, ε ∈ {0.001,0.01,0.1,0.5}, γ ∈ {0.001,0.01,0.1,1}).

All metaheuristics minimize TimeSeriesSplit CV RMSE on 2014–2017 training data. Search bounds: log₁₀(C) ∈ [−2,3], log₁₀(ε) ∈ [−3,−0.301], log₁₀(γ) ∈ [−3,0].

### 4.3 Univariate Time-Series Models (Per-State)

Trained on each state's 2013–2017 enrollment history; forecast 2018–2019:

- **ARIMA:** `auto_arima` (p,d,q ∈ [0,2], d=1, seasonal=False).
- **ETS:** Additive error, additive trend, no seasonality, damped=False.
- **TBATS:** Trend + damped trend, no seasonality, Box-Cox optional.

Fallback: last observed value if fitting fails or history < 2 points.

### 4.4 Model Selection Criterion

Models ranked by average rank across four metrics (MAE, RMSE, MAPE, R²). For R², higher is better, so rank is computed on −R². Lowest average rank wins. This avoids over-reliance on R², which is inflated by large-state dominance.

### 4.5 Two-Stage Forecasting Framework

**Stage 1 — Future Input Features (2020–2024):**

- **Count indicators** (7): teachers, colleges, universities, stand-alone institutions, foreign students, university enrollment. Per state, per indicator, fit linear trend y = a + b·t on available history through 2019. If n < 3 observations or projection > 3× historical max → fallback to last observed value. Negative projections clipped to 0. Method recorded per state–indicator.
- **Ratio indicators** (5): GER, GPI, PTR_all, PTR_university_colleges, PTR_university_units. NOT extrapolated. Held at 2019 value ("2019 ratio baseline scenario"). Sensitivity scenarios: Lower = 0.95×, Upper = 1.05× (5% perturbation). Documented as assumptions.

**Stage 2 — Recursive Enrollment Forecasting:**

- Retrain best model (XGBoost) on full 2014–2019 data with identical preprocessing.
- For each state and scenario, forecast year-by-year (2020→2024):
  - Enrollment lags use previous predictions recursively.
  - Count indicators from Stage 1 projections.
  - Ratio indicators from scenario assumption.
  - Predict → feed back as next year's lag input.

**Horizons:** Primary = 2020–2022 (3 years); Extended sensitivity = 2020–2024 (5 years). No 2030 point forecast produced.

---

## 5. Experimental Results and Discussion

### 5.1 Standalone Model Comparison (Chronological Test: 2018–2019)

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Linear Regression | 55,947 | 79,869 | 429.8% | 0.9966 |
| SVR (default) | 927,410 | 1,459,550 | 3,565.6% | −0.1344 |
| Random Forest | 94,583 | 225,782 | 186.5% | 0.9729 |
| **XGBoost** | **43,467** | **69,042** | **62.4%** | **0.9975** |
| LightGBM | 114,163 | 188,858 | 57.4% | 0.9810 |

**Selection:** XGBoost achieves lowest average rank (1.0) across MAE, RMSE, MAPE, R². LightGBM has lower MAPE (57.4% vs 62.4%) but higher MAE/RMSE and lower R².

**Observation:** High R² coexists with high MAPE. Large states (Maharashtra, Uttar Pradesh, Tamil Nadu) dominate R²; smaller states exhibit high percentage errors. MAPE should not be ignored.

### 5.2 Hybrid SVR Variants Comparison

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| WOA-SVR | 920,433 | 1,451,195 | 3,555.2% | −0.1214 |
| GWO-SVR | 927,014 | 1,459,194 | 3,568.2% | −0.1338 |
| PSO-SVR | 920,438 | 1,451,199 | 3,555.2% | −0.1214 |
| ActualGRIDSVR | 921,197 | 1,451,881 | 3,549.8% | −0.1225 |

**Result:** All hybrid SVR variants severely underperform standalone tree-based models (R² < 0). Metaheuristic optimization converges to similar CV RMSE (~1.4M) but test generalization remains poor. Grid search (ActualGRIDSVR) finds C=1000, ε=0.001, γ=0.01; metaheuristics find comparable regions. SVR with RBF kernel is fundamentally unsuited to this feature space and sample size.

### 5.3 Univariate Time-Series Models (Per-State)

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| ARIMA | 44,474 | 74,234 | 6.76% | 0.9971 |
| ETS | 50,192 | 90,757 | 7.33% | 0.9956 |
| TBATS | 51,703 | 101,668 | 6.50% | 0.9945 |

**Notable:** ARIMA achieves remarkably low MAPE (6.76%) vs. XGBoost (62.4%). This is because ARIMA forecasts each state independently on its own enrollment history, avoiding cross-state scale heterogeneity that inflates MAPE in pooled ML models. However, ARIMA cannot incorporate exogenous educational indicators.

### 5.4 Feature Importance (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | enrollment_lag_1 | 0.834 |
| 2 | colleges_lag1 | 0.112 |
| 3 | rolling_mean | 0.031 |
| 4 | teachers_lag1 | 0.008 |
| 5 | GPI_lag1 | 0.004 |
| ... | ... | ... |

Enrollment lag-1 dominates; institutional count (colleges) is the strongest exogenous predictor.

### 5.5 State-Level Forecasting Results (2020–2024 Base Scenario)

Aggregate base projections (sum across 35 states):

| Year | Historical | Base Forecast |
|------|------------|---------------|
| 2019 | 38,228,258 | — |
| 2020 | — | 39,126,568 |
| 2021 | — | 39,464,089 |
| 2022 | — | 39,702,441 |
| 2023 | — | 39,989,498 |
| 2024 | — | 40,030,167 |

Top-5 states by 2019 enrollment (Maharashtra, Uttar Pradesh, Tamil Nadu, Karnataka, Rajasthan) drive aggregate trends. State-level trajectories vary: some grow (Uttar Pradesh, Rajasthan), some plateau (Tamil Nadu, Karnataka), some decline slightly (Andhra Pradesh, Maharashtra).

### 5.6 Scenario Analysis

Three ratio-indicator scenarios (count indicators identical):

| Scenario | 2020 | 2021 | 2022 | 2023 | 2024 |
|----------|------|------|------|------|------|
| Lower (−5%) | 38.7M | 38.9M | 39.1M | 39.3M | 39.3M |
| Base (2019) | 39.1M | 39.5M | 39.7M | 40.0M | 40.0M |
| Upper (+5%) | 39.5M | 40.0M | 40.3M | 40.7M | 40.7M |

Sensitivity band (±5% on ratio indicators) yields ~±1.5% enrollment variation at aggregate level.

### 5.7 Indicator Projection Diagnostics

Of 7 count indicators × 35 states = 245 projections:
- Linear trend fitted: majority.
- Fallbacks (last observed value): 52 cases (21.2%)—due to insufficient history or instability cap.
- Clipped (negative → 0): 4 cases (stand-alone institutions in small states).

Ratio indicators held constant by design; no trend extrapolation attempted given only 6–7 observations.

---

## 6. Limitations

1. **Sparse historical data:** Only 6 modeling years per state (2014–2019); 2013 used solely for lag generation. Limits model complexity and forecast horizon.
2. **Scale heterogeneity:** R² inflated by large states; MAPE reveals high relative errors for small states. Metrics must be interpreted jointly.
3. **Hybrid SVR failure:** All metaheuristic-SVR variants produce negative R² on test data, indicating poor generalization despite CV optimization.
4. **Frozen ratio indicators:** GER, GPI, PTR held at 2019 values—strong assumption. Sensitivity scenarios (±5%) partially address this but remain arbitrary.
5. **Recursive error accumulation:** Multi-step forecasts compound errors; 2022–2024 projections less reliable than 2020–2021.
6. **Foreign student data:** Weak coverage; many states have missing or constant values.
7. **No causal claims:** Correlational ML framework; projections are data-driven, not policy-simulative.
8. **Excluded states/UTs:** Ladakh dropped; some indicators missing for smaller UTs.

---

## 7. Conclusion and Future Work

This study presents a comparative ML and hybrid forecasting framework for Indian higher education enrollment using NDAP state-level data. Key findings:

- **XGBoost** is the best standalone model under chronological multi-metric evaluation (MAE=43,467; MAPE=62.4%; R²=0.9975).
- **Hybrid SVR variants (WOA, GWO, PSO, Grid)** fail to outperform even Linear Regression, yielding negative R² on the test period.
- **Per-state ARIMA** achieves excellent MAPE (6.8%) but cannot use exogenous indicators.
- A **two-stage forecasting framework** mitigates the frozen-indicator artifact: Stage 1 projects count indicators via transparent linear trends with fallbacks; Stage 2 recursively forecasts enrollment using XGBoost with scenario-based ratio assumptions.
- **Primary projections (2020–2022):** National enrollment ~39.1M → 39.7M. Extended sensitivity (2020–2024) shows gradual growth to ~40.0M under base scenario.

**Future Work:**

1. Incorporate demographic covariates (age-cohort population, GDP, literacy) to enrich feature set.
2. Explore hierarchical/multi-task models that share strength across states while respecting heterogeneity.
3. Extend historical depth via older statistical yearbooks to enable longer-horizon validation.
4. Develop policy-simulation layer (e.g., "what-if" GER targets, institution openings) on top of ML core.
5. Investigate conformal prediction for calibrated prediction intervals.

---

## References

1. Y.-C. Yang, C.-H. Hsu, and C.-C. Chen, "Student Enrollment and Teacher Statistics Forecasting Based on Time-Series Analysis," *Computational Intelligence and Neuroscience*, vol. 2020, Art. no. 1246920, 2020.

2. National Data and Analytics Platform (NDAP), "Report 9259: Higher Education Statistics," Government of India. Available: https://ndap.niti.gov.in/

3. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. KDD*, 2016, pp. 785–794.

4. S. Mirjalili and A. Lewis, "The Whale Optimization Algorithm," *Advances in Engineering Software*, vol. 95, pp. 51–67, 2016.

5. S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey Wolf Optimizer," *Advances in Engineering Software*, vol. 69, pp. 46–61, 2014.

6. J. Kennedy and R. Eberhart, "Particle Swarm Optimization," in *Proc. ICNN*, 1995, pp. 1942–1948.

7. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.

8. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. NIPS*, 2017, pp. 3146–3154.

---

## Appendix A: Dataset Summary

| Attribute | Value |
|-----------|-------|
| Source | NDAP Report 9259 |
| Raw years | 2012–13 to 2019–20 |
| Modeling years | 2014–2019 |
| States | 35 |
| Observations | 210 |
| Train (2014–2017) | 140 |
| Test (2018–2019) | 70 |
| Features (numeric) | 16 |
| Features (categorical) | 1 (State, one-hot) |

## Appendix B: Hybrid Optimizer Best Parameters

| Model | C | ε | γ |
|-------|---|---|---|
| WOA-SVR | 1000.0 | 0.500 | 0.0130 |
| GWO-SVR | 1000.0 | 0.0580 | 0.470 |
| PSO-SVR | 1000.0 | 0.00464 | 0.0130 |
| ActualGRIDSVR | 1000.0 | 0.001 | 0.01 |

## Appendix C: Forecast Methodology Summary

Detailed methodology documented in `outputs/forecast_methodology.txt`. Key parameters:
- Stage 1: Linear trend per state–indicator, min 3 observations, instability cap 3× historical max.
- Stage 2: XGBoost retrained on full 2014–2019 data.
- Scenarios: Base (2019 ratios), Lower (−5%), Upper (+5%).
- Horizons: Primary 2020–2022, Extended 2020–2024.