import os

import numpy as np
import pandas as pd

from config import (
    CLEANED_PATH,
    COLUMN_RENAME,
    MISSING_MARKERS,
    NUMERIC_COLUMNS,
    OUTPUTS_DIR,
    RAW_DATA,
)


def load_raw():
    df = pd.read_csv(RAW_DATA)
    return df


def strip_and_rename(df):
    df.columns = [c.split(" (UOM")[0].strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        for key, new_name in COLUMN_RENAME.items():
            if key in col:
                rename_map[col] = new_name
                break
    rename_map["State lgd code"] = "StateCode"
    df = df.rename(columns=rename_map)
    return df


def parse_year(df):
    df["YearCode"] = (
        df["Year"].astype(str).str.extract(r"(20\d{2})").iloc[:, 0].astype(float).astype("Int64")
    )
    df = df.rename(columns={"Year": "YearLabel"})
    return df


def to_missing(df):
    for marker in MISSING_MARKERS:
        if marker == "":
            df = df.replace(r"^\s*$", np.nan, regex=True)
        else:
            df = df.replace(marker, np.nan)
    return df


def coerce_numeric(df):
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_duplicates(df):
    return int(df.duplicated().sum())


def check_state_consistency(df):
    bad_name = df.groupby("State")["StateCode"].nunique()
    bad_code = df.groupby("StateCode")["State"].nunique()
    return {
        "states_with_multiple_codes": int((bad_name > 1).sum()),
        "codes_with_multiple_states": int((bad_code > 1).sum()),
    }


def suspicious_values(df):
    findings = []
    for col in NUMERIC_COLUMNS:
        neg = (df[col] < 0).sum()
        if neg:
            findings.append(f"{col}: {neg} negative values")
    all_null = [c for c in NUMERIC_COLUMNS if df[c].isna().all()]
    if all_null:
        findings.append(f"columns entirely missing: {all_null}")
    return findings


def cleaning_summary(df, dups, state_check, findings):
    lines = [
        f"Raw rows: {len(df)}",
        f"Duplicate rows: {dups}",
        f"States with multiple LGD codes: {state_check['states_with_multiple_codes']}",
        f"Codes with multiple state names: {state_check['codes_with_multiple_states']}",
        f"Missing values after cleaning:\n{df[NUMERIC_COLUMNS].isna().sum().to_string()}",
    ]
    if findings:
        lines.append("Suspicious values found:")
        lines.extend(f"  - {f}" for f in findings)
    else:
        lines.append("Suspicious values found: none")
    return "\n".join(lines)


def main():
    df = load_raw()
    df = strip_and_rename(df)
    df = parse_year(df)
    df = to_missing(df)
    df = coerce_numeric(df)
    dups = check_duplicates(df)
    state_check = check_state_consistency(df)
    findings = suspicious_values(df)
    summary = cleaning_summary(df, dups, state_check, findings)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print(summary)
    print(f"\nCleaned dataset saved to {CLEANED_PATH}")


if __name__ == "__main__":
    main()
