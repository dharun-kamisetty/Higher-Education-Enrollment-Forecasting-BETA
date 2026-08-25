import os

import numpy as np
import pandas as pd

from config import (
    CLEANED_PATH,
    FIRST_MODEL_YEAR,
    ML_DATASET_PATH,
    NUMERIC_COLUMNS,
    OUTPUTS_DIR,
    TEACHER_RANK_ITEMS,
)

EXCLUDED_STATES = ["LADAKH", "LADAK"]

PIVOT_MAP = {
    "enrolment": "Total",
    "university_enrolment": "Total",
    "foreign_students": "Total",
    "colleges": None,
    "universities": None,
    "stand_alone_institutions": "Institutions",
    "GER": "All Categories",
    "GPI": "All Categories",
    "PTR_all": "Regular Mode",
    "PTR_university_colleges": "Regular Mode",
    "PTR_university_units": "Regular Mode",
}


def select_items_rows(df, items):
    if items is None:
        return df[df["Items"].isna()]
    return df[df["Items"] == items]


def pivot_features(df):
    df = df[df["Gender"] == "Total"].copy()
    base = df[["State", "StateCode", "YearCode"]].drop_duplicates().reset_index(drop=True)

    for col, items in PIVOT_MAP.items():
        sub = select_items_rows(df, items)[["State", "YearCode", col]].drop_duplicates(
            subset=["State", "YearCode"]
        )
        base = base.merge(sub, on=["State", "YearCode"], how="left")

    rank_teachers = (
        df[df["Items"].isin(TEACHER_RANK_ITEMS)]
        .groupby(["State", "YearCode"])["teachers"]
        .sum()
        .reset_index()
        .rename(columns={"teachers": "teachers"})
    )
    base = base.merge(rank_teachers, on=["State", "YearCode"], how="left")

    return base


def add_lags(df):
    df = df.sort_values(["State", "YearCode"]).reset_index(drop=True)
    grouped = df.groupby("State", group_keys=False)

    df["enrollment_lag_1"] = grouped["enrolment"].shift(1)
    df["enrollment_lag_2"] = grouped["enrolment"].shift(2)
    prior = grouped["enrolment"].shift(1)
    df["enrollment_growth"] = np.where(
        df["enrollment_lag_2"].ne(0),
        (df["enrollment_lag_1"] - df["enrollment_lag_2"]) / df["enrollment_lag_2"],
        np.nan,
    )
    df["rolling_mean"] = df.groupby("State")["enrolment"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    lagged_cols = [c for c in NUMERIC_COLUMNS if c != "enrolment"]
    for col in lagged_cols:
        df[f"{col}_lag1"] = grouped[col].shift(1)

    return df


def build_ml_dataset():
    df = pd.read_csv(CLEANED_PATH)
    df = df[~df["State"].str.upper().isin(EXCLUDED_STATES)]
    df = df[df["YearCode"] >= FIRST_MODEL_YEAR - 1]
    df = pivot_features(df)
    df = add_lags(df)
    df = df[df["YearCode"] >= FIRST_MODEL_YEAR].copy()
    df = df.rename(columns={"enrolment": "higher_education_enrollment"})
    raw_feature_cols = [c for c in NUMERIC_COLUMNS if c != "enrolment"]
    df = df.drop(columns=[c for c in raw_feature_cols if c in df.columns])
    df = df.sort_values(["State", "YearCode"]).reset_index(drop=True)
    df["Year"] = df["YearCode"]
    return df


def main():
    df = build_ml_dataset()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_csv(ML_DATASET_PATH, index=False)
    print(f"ML dataset saved to {ML_DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Year range: {df['YearCode'].min()} - {df['YearCode'].max()}")
    print(f"States: {df['State'].nunique()}")
    print(f"Missing values:\n{df.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
