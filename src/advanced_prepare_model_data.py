
import os
import pandas as pd

INPUT_FILE = "data/processed/advanced_historical_features.csv"
OUTPUT_DIR = "data/processed"

def prepare_advanced_model_data():

    print("\n===== ADVANCED MODEL DATA PREPARATION =====")

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    features = [
        "route",
        "bus_type",
        "capacity",
        "distance_km",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_mean_7"
    ]

    target = "passengers"

    required_columns = ["date"] + features + [target]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = df[required_columns].copy()

    if model_df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in model dataset.")

    # Split using unique dates to prevent same-date leakage
    unique_dates = sorted(model_df["date"].unique())

    split_index = int(len(unique_dates) * 0.80)

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_df = model_df[
        model_df["date"].isin(train_dates)
    ].copy()

    test_df = model_df[
        model_df["date"].isin(test_dates)
    ].copy()

    # Verify chronological separation
    if train_df["date"].max() >= test_df["date"].min():
        raise ValueError("Date leakage detected!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(
        f"{OUTPUT_DIR}/advanced_train_data.csv",
        index=False
    )

    test_df.to_csv(
        f"{OUTPUT_DIR}/advanced_test_data.csv",
        index=False
    )

    print("\nTOTAL RECORDS:", len(model_df))
    print("TRAINING RECORDS:", len(train_df))
    print("TESTING RECORDS:", len(test_df))

    print("\nTRAINING PERIOD:")
    print(train_df["date"].min(), "to", train_df["date"].max())

    print("\nTESTING PERIOD:")
    print(test_df["date"].min(), "to", test_df["date"].max())

    print("\nTRAINING UNIQUE DATES:", train_df["date"].nunique())
    print("TESTING UNIQUE DATES:", test_df["date"].nunique())

    print("\nNo same-date leakage detected.")
    print("Advanced model datasets saved successfully!")

if __name__ == "__main__":
    prepare_advanced_model_data()
