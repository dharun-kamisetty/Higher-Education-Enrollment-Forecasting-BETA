# ML Pipeline Instructions

Build only the machine learning pipeline for the educational enrollment forecasting project. Do not build the dashboard or write the academic report at this stage.

## 1. Dataset

Use this exact dataset:

```text
/Users/dharunkamisetty/Desktop/01_Active/mini_proj/NDAP_REPORT_9259.csv
```

First inspect the dataset and determine its actual schema. Do not assume column names, year range, or available indicators.

The raw CSV must never be modified.

---

## 2. Data Understanding

Load the dataset using pandas and report:

* Number of rows and columns
* Column names
* Data types
* Missing values
* Duplicate rows
* Unique years
* Unique states
* Unique gender values
* Unique `Items` values
* Numeric columns
* Target candidate columns

The dataset contains dimensions such as `State`, `Year`, `Gender`, and `Items`. Multiple rows may belong to the same State-Year because of these dimensions.

**Do not treat every raw row as an independent ML observation.**

---

## 3. Data Cleaning

Perform the following:

1. Strip whitespace from column names.
2. Convert `<br>`, empty strings, and equivalent missing markers to `NaN`.
3. Convert appropriate numeric columns to numeric types.
4. Parse the `Year` field into a usable year value.
5. Check duplicate records.
6. Check inconsistent state names/codes.
7. Identify impossible or suspicious numeric values.
8. Generate a cleaning summary.

Never convert missing values to zero unless the dataset documentation explicitly establishes that zero is the correct meaning.

---

## 4. Construct the ML Dataset

Create a clean analytical dataset at the **State-Year level**.

The raw dataset contains `Gender` and `Items`, so determine which rows represent:

* Total enrollment
* Gender-specific enrollment
* SC/ST enrollment
* Teacher categories
* Other aggregate indicators

Do not add together overlapping categories.

For example, do not calculate:

```text
Male + Female + Male+Female
```

if an aggregate value already exists.

The final dataset should contain one meaningful observation per:

```text
State + Year
```

where possible.

---

## 5. Target Variable

The target is:

```text
Number of enrolment in higher education
```

Use the appropriate aggregate enrollment value available in the dataset.

Internally rename it:

```text
higher_education_enrollment
```

Verify exactly how this value is represented before building the model.

The target must represent enrollment for the prediction year.

---

## 6. Features

Use only features that are genuinely available in the dataset.

Potential features include:

```text
GER
GPI
Number of teachers
Number of colleges
Number of universities
Number of foreign students
Number of stand-alone institutions
University/college enrollment indicators
Year
State
```

Create historical features where appropriate:

```text
enrollment_lag_1
enrollment_lag_2
enrollment_growth
rolling_mean
```

Lag features must be calculated separately for each State.

For example:

```text
State A, 2021 → enrollment_lag_1 = State A enrollment in 2020
State A, 2022 → enrollment_lag_1 = State A enrollment in 2021
```

Never allow one state's historical value to become another state's lag.

---

## 7. Time-Series Split

This is a forecasting problem.

**Never randomly shuffle the dataset before splitting.**

Sort observations chronologically and use a temporal split.

For example:

```text
Older years → Training
Latest available year(s) → Testing
```

Determine the exact split from the actual dataset.

If sufficient years are available, use:

```text
TimeSeriesSplit
```

for cross-validation and hyperparameter tuning.

Do not use ordinary random K-fold validation.

---

## 8. Prevent Data Leakage

This is mandatory.

When predicting enrollment for year `t`, only information available before or at the appropriate prediction point may be used.

For example:

Valid:

```text
Enrollment(t-1)
Enrollment(t-2)
GER(t-1)
Teachers(t-1)
GPI(t-1)
```

Invalid:

```text
Enrollment(t)
```

when enrollment at `t` is the target.

Any preprocessing operation must be fitted only on the training data:

* Imputation
* Scaling
* Feature selection
* Hyperparameter tuning

Use sklearn `Pipeline` and `ColumnTransformer` wherever appropriate.

---

# 9. Models

Implement exactly these five regression models:

### 1. Linear Regression

Use as the baseline.

### 2. Support Vector Regression

Use:

```text
SVR
```

Scaling is required.

### 3. Random Forest

Use:

```text
RandomForestRegressor
```

### 4. XGBoost

Use:

```text
XGBRegressor
```

### 5. LightGBM

Use:

```text
LGBMRegressor
```

Do not assume beforehand which model will perform best.

---

## 10. Baseline Experiment

First train all five models using reasonable baseline parameters.

Do not tune them immediately.

For every model generate:

```text
Predictions
MAE
RMSE
MAPE
R²
```

Create:

| Model             | MAE | RMSE | MAPE | R² |
| ----------------- | --: | ---: | ---: | -: |
| Linear Regression |     |      |      |    |
| SVR               |     |      |      |    |
| Random Forest     |     |      |      |    |
| XGBoost           |     |      |      |    |
| LightGBM          |     |      |      |    |

Populate this table only from actual experimental results.

---

## 11. Hyperparameter Tuning

After the baseline experiment works, tune the models.

Use temporal validation.

Suitable methods:

```text
GridSearchCV
RandomizedSearchCV
```

with:

```text
TimeSeriesSplit
```

Do not use random K-fold validation.

Tune sensible parameters such as:

### SVR

```text
C
epsilon
gamma
kernel
```

### Random Forest

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

### XGBoost

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

### LightGBM

```text
n_estimators
num_leaves
learning_rate
max_depth
min_child_samples
```

Do not perform excessively large searches that make the project unnecessarily slow.

---

## 12. Evaluation Metrics

Calculate:

### MAE

Lower is better.

### RMSE

Lower is better.

### MAPE

Lower is better.

Handle zero actual enrollment values correctly. Do not allow division-by-zero to produce meaningless MAPE values.

### R²

Higher is better.

Use the same test dataset for all five models.

---

## 13. Model Selection

Determine the best model from the actual test results.

Do not simply select the model with the highest R².

Consider all four metrics:

```text
MAE
RMSE
MAPE
R²
```

The final output should clearly state:

```text
Best-performing model: ______
Reason: ______
```

The reason must be based on the actual results.

---

## 14. Feature Importance

For:

```text
Random Forest
XGBoost
LightGBM
```

calculate feature importance.

Save the results in:

```text
outputs/feature_importance.csv
```

Also create a feature-importance plot.

Do not interpret feature importance as causal influence.

For example, say:

> GER was an important predictor in the trained model.

Do not say:

> GER causes enrollment to increase.

---

## 15. Predictions

Save a prediction dataset containing:

```text
State
Year
Actual Enrollment
Linear Regression Prediction
SVR Prediction
Random Forest Prediction
XGBoost Prediction
LightGBM Prediction
```

Save as:

```text
outputs/predictions.csv
```

---

## 16. Required Plots

Generate:

1. Actual enrollment trend over time.
2. Actual vs predicted enrollment for every model.
3. MAE comparison.
4. RMSE comparison.
5. MAPE comparison.
6. R² comparison.
7. Feature importance for tree models.

Save all plots under:

```text
outputs/plots/
```

---

## 17. Project Files

Create a clean modular implementation:

```text
src/
├── inspect_data.py
├── preprocess.py
├── feature_engineering.py
├── train_models.py
├── evaluate_models.py
├── tune_models.py
└── feature_importance.py
```

Use a single entry point if useful:

```text
run_pipeline.py
```

The pipeline should be executable from the terminal.

Example:

```bash
python run_pipeline.py
```

---

## 18. Required Outputs

The completed ML pipeline must produce:

```text
outputs/
├── cleaned_dataset.csv
├── ml_dataset.csv
├── predictions.csv
├── model_comparison.csv
├── feature_importance.csv
└── plots/
```

Save trained models under:

```text
models/
```

---

## 19. Reproducibility

Set random seeds for models that use randomness.

Record:

* Dataset path
* Dataset dimensions
* Year range
* Features used
* Target variable
* Train/test years
* Missing-value strategy
* Models
* Hyperparameters
* Evaluation results

Create:

```text
requirements.txt
```

containing the exact Python packages required.

---

## 20. Final Validation

Before considering the pipeline complete, verify:

```text
[ ] Raw dataset was not modified
[ ] Missing values were handled correctly
[ ] <br> values were not treated as zero
[ ] State-Year observations are correctly constructed
[ ] Gender categories are not double-counted
[ ] Target is correctly defined
[ ] No target leakage exists
[ ] Train/test split is chronological
[ ] Scaling is fitted only on training data
[ ] Imputation is fitted only on training data
[ ] All five models run successfully
[ ] MAE is calculated
[ ] RMSE is calculated
[ ] MAPE is calculated
[ ] R² is calculated
[ ] Predictions are saved
[ ] Model comparison is saved
[ ] Feature importance is generated
[ ] Required plots are generated
```

If any of these checks fail, fix the issue before proceeding.

---

## 21. Important Rule

Do not fabricate or manually enter any experimental result.

The following must come exclusively from the actual execution of the pipeline:

* Model accuracy
* MAE
* RMSE
* MAPE
* R²
* Best model
* Feature importance
* Predictions
* Forecast values

The purpose of this stage is to produce a **correct, reproducible ML pipeline**, not to make the results look better.

Only after this pipeline is working should the project proceed to future forecasting and dashboard development.

