
import os
import pandas as pd
import numpy as np

RAW_FILE = "data/raw/transport_data.csv"
OUTPUT_DIR = "data/processed"


def validate_transport_data():

    print("\n===== TRANSPORT DATA VALIDATION =====")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    original_rows = len(df)

    required_columns = [
        "bus_id", "route", "bus_type", "depot", "date",
        "capacity", "passengers", "occupancy_rate",
        "distance_km", "fare_per_passenger", "revenue",
        "fuel_consumed_liters", "month", "day_of_week"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Convert data types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "capacity", "passengers", "occupancy_rate",
        "distance_km", "fare_per_passenger", "revenue",
        "fuel_consumed_liters"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check data quality
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    invalid_dates = int(df["date"].isnull().sum())

    invalid_capacity = int(
        (df["capacity"].notna() & (df["capacity"] <= 0)).sum()
    )

    invalid_passengers = int(
        (df["passengers"].notna() & (df["passengers"] < 0)).sum()
    )

    passengers_over_capacity = int(
        (
            df["capacity"].notna()
            & df["passengers"].notna()
            & (df["passengers"] > df["capacity"])
        ).sum()
    )

    # Recalculate occupancy
    df["occupancy_rate"] = np.where(
        df["capacity"] > 0,
        (df["passengers"] / df["capacity"]) * 100,
        np.nan
    )

    # Remove invalid records
    df = df.dropna(subset=[
        "bus_id", "route", "bus_type", "depot", "date",
        "capacity", "passengers", "distance_km",
        "fare_per_passenger", "revenue", "fuel_consumed_liters"
    ])

    df = df[
        (df["capacity"] > 0)
        & (df["passengers"] >= 0)
        & (df["passengers"] <= df["capacity"])
        & (df["distance_km"] >= 0)
        & (df["fare_per_passenger"] >= 0)
        & (df["revenue"] >= 0)
        & (df["fuel_consumed_liters"] >= 0)
    ].copy()

    df = df.drop_duplicates()

    # Quality report
    report = {
        "Original Rows": original_rows,
        "Valid Rows": len(df),
        "Removed Rows": original_rows - len(df),
        "Missing Values Initially": missing_values,
        "Duplicate Rows Initially": duplicate_rows,
        "Invalid Dates": invalid_dates,
        "Invalid Capacity": invalid_capacity,
        "Invalid Passengers": invalid_passengers,
        "Passengers Over Capacity": passengers_over_capacity,
        "Unique Routes": df["route"].nunique(),
        "Unique Bus Types": df["bus_type"].nunique()
    }

    report_df = pd.DataFrame(
        list(report.items()),
        columns=["Metric", "Value"]
    )

    # Save outputs
    df.to_csv(
        os.path.join(OUTPUT_DIR, "validated_transport_data.csv"),
        index=False
    )

    report_df.to_csv(
        os.path.join(OUTPUT_DIR, "data_quality_report.csv"),
        index=False
    )

    print("\nDATA QUALITY REPORT")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nValidated dataset saved successfully.")
    print("Data quality report saved successfully.")

    return df, report


if __name__ == "__main__":
    validated_df, report = validate_transport_data()

    print("\nValidated Dataset Shape:", validated_df.shape)
    print("\nFirst 5 Records:")
    print(validated_df.head())
