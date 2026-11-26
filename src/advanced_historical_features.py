
import os
import pandas as pd

INPUT_FILE = "data/processed/feature_engineered_data.csv"
OUTPUT_FILE = "data/processed/advanced_historical_features.csv"

def create_advanced_historical_features():

    print("\n===== ADVANCED HISTORICAL FEATURE ENGINEERING =====")

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate multiple buses operating on the same route,
    # bus type and date.
    daily = (
        df.groupby(
            ["route", "bus_type", "date"],
            as_index=False
        )
        .agg(
            passengers=("passengers", "sum"),
            capacity=("capacity", "sum"),
            distance_km=("distance_km", "mean"),
            fare_per_passenger=("fare_per_passenger", "mean"),
            revenue=("revenue", "sum"),
            fuel_consumed_liters=("fuel_consumed_liters", "sum")
        )
    )

    # Recalculate daily occupancy
    daily["occupancy_rate"] = (
        daily["passengers"] / daily["capacity"] * 100
    )

    # Time-based features
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["day_of_week"] = daily["date"].dt.day_of_week
    daily["week_of_year"] = (
        daily["date"].dt.isocalendar().week.astype(int)
    )
    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)
    daily["quarter"] = daily["date"].dt.quarter

    # Sort before creating historical features
    daily = daily.sort_values(
        ["route", "bus_type", "date"]
    ).reset_index(drop=True)

    grouped = daily.groupby(
        ["route", "bus_type"]
    )["passengers"]

    # Previous observed dates only
    daily["lag_1"] = grouped.shift(1)
    daily["lag_2"] = grouped.shift(2)
    daily["lag_3"] = grouped.shift(3)

    # Rolling averages using previous observations only
    daily["rolling_mean_3"] = (
        daily.groupby(["route", "bus_type"])["passengers"]
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    daily["rolling_mean_7"] = (
        daily.groupby(["route", "bus_type"])["passengers"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    # Keep records with sufficient historical information
    daily = daily.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_3",
            "rolling_mean_7"
        ]
    ).copy()

    # Save without modifying existing datasets
    os.makedirs("data/processed", exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)

    print("\nOriginal records:", len(df))
    print(
        "Unique route-date-bus type combinations:",
        df.groupby(["route", "bus_type", "date"]).ngroups
    )
    print("Aggregated records:", daily.shape[0])
    print("Final columns:", daily.columns.tolist())
    print("\nSaved:", OUTPUT_FILE)
    print("\nDataset sample:")
    print(daily.head())

    return daily


if __name__ == "__main__":
    result = create_advanced_historical_features()
    print("\nFinal dataset shape:", result.shape)
