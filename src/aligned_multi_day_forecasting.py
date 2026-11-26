import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# CONFIGURATION
# ------------------------------------

MODEL_FILE = "models/advanced/gradient_boosting.pkl"
HISTORY_FILE = "data/processed/advanced_historical_features.csv"
RAW_FILE = "data/raw/transport_data.csv"

OUTPUT_DIR = "outputs/advanced"
FORECAST_DAYS = 7

FEATURES = [
    "route", "bus_type", "capacity", "distance_km",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------
# LOAD DATA AND MODEL
# ------------------------------------

def load_resources():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(f"Historical features file not found: {HISTORY_FILE}")
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    model = joblib.load(MODEL_FILE)
    history = pd.read_csv(HISTORY_FILE)
    raw = pd.read_csv(RAW_FILE)

    required_history = [
        "route", "bus_type", "date", "passengers",
        "capacity", "distance_km"
    ]
    required_raw = ["route", "bus_type", "capacity", "date"]

    missing_history = [c for c in required_history if c not in history.columns]
    missing_raw = [c for c in required_raw if c not in raw.columns]

    if missing_history:
        raise ValueError(f"Missing columns in historical data: {missing_history}")
    if missing_raw:
        raise ValueError(f"Missing columns in raw data: {missing_raw}")

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

    for column in ["passengers", "capacity", "distance_km"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    raw["capacity"] = pd.to_numeric(raw["capacity"], errors="coerce")

    history = history.dropna(
        subset=["route", "bus_type", "date", "passengers", "capacity", "distance_km"]
    ).copy()

    raw = raw.dropna(subset=["route", "bus_type", "capacity"]).copy()

    if history.empty:
        raise ValueError("No valid rows remain in the historical features dataset.")
    if raw.empty:
        raise ValueError("No valid capacity records remain in the raw dataset.")

    history = history.sort_values(
        ["route", "bus_type", "date"]
    ).reset_index(drop=True)

    return model, history, raw


# ------------------------------------
# CAPACITY INFORMATION
# ------------------------------------

def prepare_capacity(raw):
    capacity_table = (
        raw.groupby(["route", "bus_type"])["capacity"]
        .mean()
        .to_dict()
    )

    type_capacity = (
        raw.groupby("bus_type")["capacity"]
        .mean()
        .to_dict()
    )

    overall_capacity = float(raw["capacity"].mean())

    if not np.isfinite(overall_capacity) or overall_capacity <= 0:
        raise ValueError("Could not calculate a valid average bus capacity.")

    return capacity_table, type_capacity, overall_capacity


def get_capacity(
    route,
    bus_type,
    capacity_table,
    type_capacity,
    overall_capacity
):
    capacity = capacity_table.get(
        (route, bus_type),
        type_capacity.get(bus_type, overall_capacity)
    )

    capacity = float(capacity)

    if not np.isfinite(capacity) or capacity <= 0:
        capacity = float(overall_capacity)

    return capacity


# ------------------------------------
# DATE FEATURES
# ------------------------------------

def date_features(date):
    date = pd.Timestamp(date)

    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.day_of_week,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.day_of_week >= 5),
        "quarter": date.quarter
    }


# ------------------------------------
# RECURSIVE SINGLE-DAY PREDICTION
# ------------------------------------

def predict_one_day(
    model,
    route,
    bus_type,
    date,
    passenger_history,
    latest_capacity,
    latest_distance
):
    # Keep only valid numeric history values.
    history_values = [
        float(value)
        for value in passenger_history
        if pd.notna(value) and np.isfinite(float(value))
    ]

    if not history_values:
        raise ValueError(
            f"No passenger history available for {route} | {bus_type}"
        )

    # Safe fallback for groups with fewer than 3 observations.
    lag_1 = history_values[-1]
    lag_2 = history_values[-2] if len(history_values) >= 2 else lag_1
    lag_3 = history_values[-3] if len(history_values) >= 3 else lag_2

    rolling_mean_3 = float(np.mean(history_values[-3:]))
    rolling_mean_7 = float(np.mean(history_values[-7:]))

    features = {
        "route": route,
        "bus_type": bus_type,
        "capacity": float(latest_capacity),
        "distance_km": float(latest_distance),
        **date_features(date),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_7": rolling_mean_7
    }

    X = pd.DataFrame([features], columns=FEATURES)
    prediction = float(model.predict(X)[0])

    if not np.isfinite(prediction):
        raise ValueError(
            f"Model returned an invalid prediction for {route} | {bus_type}"
        )

    return max(0.0, prediction)


# ------------------------------------
# ALIGN EVERY GROUP TO COMMON DATE
# ------------------------------------

def align_group(model, group, global_latest_date):
    group = group.sort_values("date").reset_index(drop=True)

    if group.empty:
        raise ValueError("Cannot align an empty route/type group.")

    route = str(group.iloc[-1]["route"])
    bus_type = str(group.iloc[-1]["bus_type"])

    passenger_history = group["passengers"].astype(float).tolist()
    latest_capacity = float(group.iloc[-1]["capacity"])
    latest_distance = float(group.iloc[-1]["distance_km"])
    group_latest_date = pd.Timestamp(group.iloc[-1]["date"])

    # Fill missing dates using recursive predictions until the
    # common latest historical date for the entire network.
    next_date = group_latest_date + pd.Timedelta(days=1)

    while next_date <= global_latest_date:
        prediction = predict_one_day(
            model,
            route,
            bus_type,
            next_date,
            passenger_history,
            latest_capacity,
            latest_distance
        )

        passenger_history.append(prediction)
        next_date += pd.Timedelta(days=1)

    return {
        "route": route,
        "bus_type": bus_type,
        "passenger_history": passenger_history,
        "capacity": latest_capacity,
        "distance": latest_distance
    }


# ------------------------------------
# MAIN FORECASTING
# ------------------------------------

def run_aligned_forecast():
    print("\n===== ALIGNED 7-DAY FORECASTING =====")

    model, history, raw = load_resources()

    capacity_table, type_capacity, overall_capacity = prepare_capacity(raw)

    global_latest_date = pd.Timestamp(history["date"].max()).normalize()

    print("Latest historical date:", global_latest_date.date())

    forecast_start = global_latest_date + pd.Timedelta(days=1)
    forecast_end = global_latest_date + pd.Timedelta(days=FORECAST_DAYS)

    print("Forecast start:", forecast_start.date())
    print("Forecast end:", forecast_end.date())

    all_forecasts = []

    grouped = history.groupby(["route", "bus_type"], sort=True)
    total_groups = len(grouped)

    print("Total route/type groups:", total_groups)

    for (route, bus_type), group in grouped:
        print(f"Aligning: {route} | {bus_type}")

        aligned = align_group(
            model,
            group,
            global_latest_date
        )

        passenger_history = aligned["passenger_history"]

        individual_capacity = get_capacity(
            route,
            bus_type,
            capacity_table,
            type_capacity,
            overall_capacity
        )

        for day in range(1, FORECAST_DAYS + 1):
            forecast_date = global_latest_date + pd.Timedelta(days=day)

            prediction = predict_one_day(
                model,
                route,
                bus_type,
                forecast_date,
                passenger_history,
                aligned["capacity"],
                aligned["distance"]
            )

            # Estimated number of buses for this route/type demand.
            buses_required = int(np.ceil(prediction / individual_capacity))

            all_forecasts.append({
                "date": forecast_date,
                "forecast_day": day,
                "route": route,
                "bus_type": bus_type,
                "predicted_passengers": round(prediction, 2),
                "average_bus_capacity": round(individual_capacity, 2),
                "estimated_buses_required": buses_required
            })

            # Use today's prediction as history for the next forecast day.
            passenger_history.append(prediction)

    # --------------------------------
    # SAVE FORECAST RESULTS
    # --------------------------------

    if not all_forecasts:
        raise ValueError("No forecasts were generated. Check the input datasets.")

    forecast_df = pd.DataFrame(all_forecasts)

    forecast_df = forecast_df.sort_values(
        ["date", "route", "bus_type"]
    ).reset_index(drop=True)

    expected_dates = pd.date_range(
        start=forecast_start,
        end=forecast_end,
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        pd.to_datetime(forecast_df["date"].unique())
    ).sort_values()

    if not actual_dates.equals(expected_dates):
        raise ValueError(
            "Forecast dates are not aligned correctly. "
            f"Expected: {list(expected_dates.date)}; "
            f"actual: {list(actual_dates.date)}"
        )

    expected_records = total_groups * FORECAST_DAYS
    if len(forecast_df) != expected_records:
        raise ValueError(
            f"Expected {expected_records} forecast rows "
            f"({total_groups} groups x {FORECAST_DAYS} days), "
            f"but generated {len(forecast_df)}."
        )

    forecast_df.to_csv(
        f"{OUTPUT_DIR}/aligned_seven_day_forecast.csv",
        index=False
    )

    # --------------------------------
    # DAILY NETWORK SUMMARY
    # --------------------------------

    daily_summary = (
        forecast_df.groupby("date")
        .agg(
            total_predicted_passengers=("predicted_passengers", "sum"),
            total_estimated_bus_requirements=("estimated_buses_required", "sum")
        )
        .reset_index()
    )

    daily_summary.to_csv(
        f"{OUTPUT_DIR}/aligned_daily_summary.csv",
        index=False
    )

    # --------------------------------
    # ROUTE SUMMARY
    # --------------------------------

    route_summary = (
        forecast_df.groupby(["route", "bus_type"])
        .agg(
            average_daily_demand=("predicted_passengers", "mean"),
            total_seven_day_demand=("predicted_passengers", "sum"),
            total_bus_days=("estimated_buses_required", "sum")
        )
        .reset_index()
    )

    route_summary.to_csv(
        f"{OUTPUT_DIR}/aligned_route_summary.csv",
        index=False
    )

    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        daily_summary["date"],
        daily_summary["total_predicted_passengers"],
        marker="o"
    )

    plt.xlabel("Forecast Date")
    plt.ylabel("Predicted Passengers")
    plt.title("Synchronized 7-Day Transport Demand Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/aligned_seven_day_forecast.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------
    # FINAL REPORT
    # --------------------------------

    print("\n===== ALIGNED FORECAST SUMMARY =====")
    print(daily_summary.to_string(index=False))

    print("\nTotal forecast records:", len(forecast_df))
    print("Expected forecast records:", expected_records)
    print("Unique forecast dates:", forecast_df["date"].nunique())

    print(
        "Route/type combinations:",
        forecast_df.groupby(["route", "bus_type"]).ngroups
    )

    print(
        "Total estimated bus-days:",
        int(forecast_df["estimated_buses_required"].sum())
    )

    print("\nGenerated files:")
    print("1. aligned_seven_day_forecast.csv")
    print("2. aligned_daily_summary.csv")
    print("3. aligned_route_summary.csv")
    print("4. aligned_seven_day_forecast.png")

    print("\nAligned forecasting completed successfully!")


if __name__ == "__main__":
    run_aligned_forecast()
