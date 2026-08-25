import pandas as pd

from config import RAW_DATA


def main():
    df = pd.read_csv(RAW_DATA)
    report = []
    report.append(f"Number of rows: {df.shape[0]}")
    report.append(f"Number of columns: {df.shape[1]}")
    report.append("\nColumn names:")
    for c in df.columns:
        report.append(f"  {c}")
    report.append("\nData types:")
    report.append(df.dtypes.to_string())
    report.append(f"\nMissing values:\n{df.isna().sum().to_string()}")
    report.append(f"\nDuplicate rows: {df.duplicated().sum()}")
    report.append(f"\nUnique years:\n{df['Year'].value_counts(dropna=False).to_string()}")
    report.append(f"\nUnique states: {df['State'].nunique()}")
    report.append(f"\nUnique gender values:\n{df['Gender'].value_counts(dropna=False).to_string()}")
    report.append(f"\nUnique Items values:\n{df['Items'].value_counts(dropna=False).to_string()}")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    report.append(f"\nNumeric columns: {numeric_cols}")
    target_candidates = [
        c
        for c in df.columns
        if "enrolment" in c.lower() and "universities" not in c.lower()
    ]
    report.append(f"\nTarget candidate columns: {target_candidates}")
    report.append("\nTarget non-null by dimension combination (Items=Total, Gender):")
    report.append(
        df[df["Items"] == "Total"]
        .groupby("Gender")[target_candidates[0]]
        .count()
        .to_string()
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
