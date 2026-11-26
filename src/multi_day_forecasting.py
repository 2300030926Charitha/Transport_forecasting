
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# CONFIGURATION
# ------------------------------------

MODEL_FILE = "models/advanced/gradient_boosting.pkl"

HISTORY_FILE = (
    "data/processed/advanced_historical_features.csv"
)

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
# LOAD MODEL AND DATA
# ------------------------------------

def load_resources():

    print("\nLoading forecasting model and datasets...")

    model = joblib.load(MODEL_FILE)

    history = pd.read_csv(HISTORY_FILE)
    raw = pd.read_csv(RAW_FILE)

    history["date"] = pd.to_datetime(history["date"])
    raw["date"] = pd.to_datetime(raw["date"])

    history = history.sort_values(
        ["route", "bus_type", "date"]
    )

    return model, history, raw


# ------------------------------------
# BUS CAPACITY ESTIMATION
# ------------------------------------

def calculate_bus_capacity(raw):

    # Raw data contains individual bus capacities.
    capacity_table = (
        raw.groupby(["route", "bus_type"])["capacity"]
        .mean()
        .reset_index()
        .rename(columns={
            "capacity": "average_bus_capacity"
        })
    )

    type_capacity = (
        raw.groupby("bus_type")["capacity"]
        .mean()
        .to_dict()
    )

    overall_capacity = raw["capacity"].mean()

    return capacity_table, type_capacity, overall_capacity


def get_average_capacity(
    route,
    bus_type,
    capacity_table,
    type_capacity,
    overall_capacity
):

    match = capacity_table[
        (capacity_table["route"] == route)
        & (capacity_table["bus_type"] == bus_type)
    ]

    if not match.empty:
        return float(match.iloc[0]["average_bus_capacity"])

    return float(
        type_capacity.get(bus_type, overall_capacity)
    )


# ------------------------------------
# FUTURE DATE FEATURES
# ------------------------------------

def create_date_features(date):

    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "quarter": date.quarter
    }


# ------------------------------------
# RECURSIVE FORECASTING
# ------------------------------------

def forecast_route(
    model,
    group,
    capacity_table,
    type_capacity,
    overall_capacity
):

    group = group.sort_values("date").reset_index(drop=True)

    if len(group) < 7:
        return []

    route = group.iloc[-1]["route"]
    bus_type = group.iloc[-1]["bus_type"]

    # Historical passenger observations
    passenger_history = (
        group["passengers"].astype(float).tolist()
    )

    # Use latest available aggregate route capacity
    latest_capacity = float(group.iloc[-1]["capacity"])

    latest_distance = float(group.iloc[-1]["distance_km"])

    # First forecast date follows this group's latest observation
    forecast_date = (
        group.iloc[-1]["date"] + pd.Timedelta(days=1)
    )

    average_bus_capacity = get_average_capacity(
        route,
        bus_type,
        capacity_table,
        type_capacity,
        overall_capacity
    )

    forecasts = []

    for day in range(1, FORECAST_DAYS + 1):

        # Build future date features
        date_features = create_date_features(
            forecast_date
        )

        # Build recursive historical features
        features = {
            "route": route,
            "bus_type": bus_type,
            "capacity": latest_capacity,
            "distance_km": latest_distance,

            **date_features,

            "lag_1": passenger_history[-1],
            "lag_2": passenger_history[-2],
            "lag_3": passenger_history[-3],

            "rolling_mean_3": np.mean(
                passenger_history[-3:]
            ),

            "rolling_mean_7": np.mean(
                passenger_history[-7:]
            )
        }

        input_df = pd.DataFrame(
            [features],
            columns=FEATURES
        )

        # Predict passenger demand
        prediction = float(model.predict(input_df)[0])

        # Passenger count cannot be negative
        prediction = max(0, prediction)

        # Estimate individual buses required
        buses_required = int(
            np.ceil(
                prediction / average_bus_capacity
            )
        )

        forecasts.append({
            "date": forecast_date,
            "forecast_day": day,
            "route": route,
            "bus_type": bus_type,
            "predicted_passengers": round(
                prediction, 2
            ),
            "average_bus_capacity": round(
                average_bus_capacity, 2
            ),
            "estimated_buses_required": buses_required
        })

        # Feed prediction into history for next day
        passenger_history.append(prediction)

        forecast_date += pd.Timedelta(days=1)

    return forecasts


# ------------------------------------
# MAIN FUNCTION
# ------------------------------------

def run_forecasting():

    print("\n===== MULTI-DAY DEMAND FORECASTING =====")

    model, history, raw = load_resources()

    capacity_table, type_capacity, overall_capacity = (
        calculate_bus_capacity(raw)
    )

    all_forecasts = []

    # Forecast every route and bus type separately
    grouped = history.groupby(
        ["route", "bus_type"]
    )

    print(f"\nRoute and bus-type groups: {len(grouped)}")

    for (route, bus_type), group in grouped:

        print(f"Forecasting: {route} | {bus_type}")

        forecasts = forecast_route(
            model,
            group,
            capacity_table,
            type_capacity,
            overall_capacity
        )

        all_forecasts.extend(forecasts)

    if not all_forecasts:
        raise ValueError(
            "No forecasts generated. Check historical data."
        )

    # Create output DataFrame
    forecast_df = pd.DataFrame(all_forecasts)

    forecast_df = forecast_df.sort_values(
        ["date", "route", "bus_type"]
    )

    # Save detailed forecasts
    forecast_path = (
        f"{OUTPUT_DIR}/seven_day_demand_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False
    )

    # --------------------------------
    # DAILY SUMMARY
    # --------------------------------

    daily_summary = (
        forecast_df.groupby("date")
        .agg(
            total_predicted_passengers=(
                "predicted_passengers", "sum"
            ),
            estimated_bus_requirements=(
                "estimated_buses_required", "sum"
            )
        )
        .reset_index()
    )

    daily_summary.to_csv(
        f"{OUTPUT_DIR}/seven_day_forecast_summary.csv",
        index=False
    )

    # --------------------------------
    # ROUTE SUMMARY
    # --------------------------------

    route_summary = (
        forecast_df.groupby(["route", "bus_type"])
        .agg(
            average_predicted_demand=(
                "predicted_passengers", "mean"
            ),
            total_predicted_demand=(
                "predicted_passengers", "sum"
            ),
            total_estimated_bus_days=(
                "estimated_buses_required", "sum"
            )
        )
        .reset_index()
    )

    route_summary.to_csv(
        f"{OUTPUT_DIR}/seven_day_route_summary.csv",
        index=False
    )

    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        daily_summary["date"],
        daily_summary["total_predicted_passengers"],
        marker="o",
        label="Predicted Passengers"
    )

    plt.xlabel("Forecast Date")
    plt.ylabel("Total Predicted Passengers")
    plt.title("Seven-Day Public Transport Demand Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/seven_day_forecast.png"
    )

    plt.close()

    # --------------------------------
    # RESULTS
    # --------------------------------

    print("\n===== FORECAST SUMMARY =====")

    print(
        daily_summary.to_string(index=False)
    )

    print("\nTotal forecast records:", len(forecast_df))

    print(
        "Total estimated bus-days:",
        int(forecast_df["estimated_buses_required"].sum())
    )

    print("\nGenerated files:")
    print("1. seven_day_demand_forecast.csv")
    print("2. seven_day_forecast_summary.csv")
    print("3. seven_day_route_summary.csv")
    print("4. seven_day_forecast.png")

    print("\nMulti-day forecasting completed successfully!")


if __name__ == "__main__":
    run_forecasting()
