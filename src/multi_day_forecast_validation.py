
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------
# CONFIGURATION
# ------------------------------------

MODEL_FILE = "models/advanced/gradient_boosting.pkl"
HISTORY_FILE = "data/processed/advanced_historical_features.csv"

OUTPUT_DIR = "outputs/advanced/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_DAYS = 7

# Model was trained using data through 18 November 2024.
# These forecast origins are all after the training period.
FORECAST_ORIGINS = [
    "2024-11-23",
    "2024-11-30",
    "2024-12-07",
    "2024-12-14",
    "2024-12-21",
    "2024-12-23"
]

FEATURES = [
    "route", "bus_type", "capacity", "distance_km",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7"
]


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
# PREDICTION
# ------------------------------------

def predict_one_day(
    model,
    route,
    bus_type,
    date,
    passenger_history,
    capacity,
    distance
):
    history = [
        float(x) for x in passenger_history
        if pd.notna(x) and np.isfinite(float(x))
    ]

    if not history:
        return None

    lag_1 = history[-1]
    lag_2 = history[-2] if len(history) >= 2 else lag_1
    lag_3 = history[-3] if len(history) >= 3 else lag_2

    features = {
        "route": route,
        "bus_type": bus_type,
        "capacity": capacity,
        "distance_km": distance,
        **date_features(date),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": float(np.mean(history[-3:])),
        "rolling_mean_7": float(np.mean(history[-7:]))
    }

    X = pd.DataFrame([features], columns=FEATURES)

    prediction = float(model.predict(X)[0])

    return max(0.0, prediction)


# ------------------------------------
# LOAD RESOURCES
# ------------------------------------

def load_resources():

    model = joblib.load(MODEL_FILE)
    data = pd.read_csv(HISTORY_FILE)

    data["date"] = pd.to_datetime(data["date"])

    required = [
        "route", "bus_type", "date",
        "passengers", "capacity", "distance_km"
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required).copy()

    data = data.sort_values(
        ["route", "bus_type", "date"]
    ).reset_index(drop=True)

    return model, data


# ------------------------------------
# SEVEN-DAY BACKTEST
# ------------------------------------

def run_validation():

    print("\n===== MULTI-DAY FORECAST VALIDATION =====")

    model, data = load_resources()

    all_results = []

    for origin_text in FORECAST_ORIGINS:

        origin = pd.Timestamp(origin_text)
        forecast_end = origin + pd.Timedelta(days=FORECAST_DAYS)

        print(f"\nForecast origin: {origin.date()}")

        # Only observations available on or before the origin
        past_data = data[data["date"] <= origin].copy()

        # Actual observations used only for final comparison
        future_actual = data[
            (data["date"] > origin) &
            (data["date"] <= forecast_end)
        ].copy()

        if past_data.empty or future_actual.empty:
            print("Skipping: insufficient historical or future data.")
            continue

        for (route, bus_type), group in past_data.groupby(
            ["route", "bus_type"]
        ):

            group = group.sort_values("date")

            passenger_history = (
                group["passengers"].astype(float).tolist()
            )

            capacity = float(group.iloc[-1]["capacity"])
            distance = float(group.iloc[-1]["distance_km"])

            for day in range(1, FORECAST_DAYS + 1):

                forecast_date = origin + pd.Timedelta(days=day)

                prediction = predict_one_day(
                    model,
                    route,
                    bus_type,
                    forecast_date,
                    passenger_history,
                    capacity,
                    distance
                )

                if prediction is None:
                    continue

                # Recursive forecasting:
                # use prediction, not future actual demand.
                previous_demand_baseline = passenger_history[-1]

                passenger_history.append(prediction)

                all_results.append({
                    "forecast_origin": origin,
                    "forecast_date": forecast_date,
                    "forecast_day": day,
                    "route": route,
                    "bus_type": bus_type,
                    "predicted_passengers": prediction,
                    "baseline_prediction": previous_demand_baseline
                })

    if not all_results:
        raise ValueError("No backtest predictions were generated.")

    predictions = pd.DataFrame(all_results)

    # ------------------------------------
    # COMPARE WITH ACTUAL VALUES
    # ------------------------------------

    actual = data[
        ["date", "route", "bus_type", "passengers"]
    ].rename(columns={
        "date": "forecast_date",
        "passengers": "actual_passengers"
    })

    results = predictions.merge(
        actual,
        on=["forecast_date", "route", "bus_type"],
        how="inner"
    )

    if results.empty:
        raise ValueError(
            "No matching actual observations found for backtesting."
        )

    results["model_absolute_error"] = (
        results["actual_passengers"] -
        results["predicted_passengers"]
    ).abs()

    results["baseline_absolute_error"] = (
        results["actual_passengers"] -
        results["baseline_prediction"]
    ).abs()

    results.to_csv(
        f"{OUTPUT_DIR}/multi_day_validation_details.csv",
        index=False
    )

    # ------------------------------------
    # OVERALL METRICS
    # ------------------------------------

    actual_values = results["actual_passengers"]
    predicted_values = results["predicted_passengers"]

    model_mae = mean_absolute_error(
        actual_values, predicted_values
    )

    model_rmse = np.sqrt(
        mean_squared_error(actual_values, predicted_values)
    )

    model_r2 = (
        r2_score(actual_values, predicted_values)
        if len(results) >= 2 else np.nan
    )

    baseline_mae = mean_absolute_error(
        actual_values,
        results["baseline_prediction"]
    )

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0 else np.nan
    )

    print("\n===== OVERALL RESULTS =====")
    print("Matched actual records:", len(results))
    print(f"Model MAE: {model_mae:.4f}")
    print(f"Model RMSE: {model_rmse:.4f}")
    print(f"Model R2: {model_r2:.4f}")
    print(f"Previous-demand baseline MAE: {baseline_mae:.4f}")
    print(f"MAE improvement over baseline: {improvement:.2f}%")

    # ------------------------------------
    # FORECAST HORIZON METRICS
    # ------------------------------------

    horizon_summary = (
        results.groupby("forecast_day")
        .agg(
            observations=("actual_passengers", "count"),
            model_MAE=("model_absolute_error", "mean"),
            baseline_MAE=("baseline_absolute_error", "mean")
        )
        .reset_index()
    )

    horizon_summary.to_csv(
        f"{OUTPUT_DIR}/forecast_horizon_metrics.csv",
        index=False
    )

    # ------------------------------------
    # ROUTE PERFORMANCE
    # ------------------------------------

    route_summary = (
        results.groupby(["route", "bus_type"])
        .agg(
            observations=("actual_passengers", "count"),
            model_MAE=("model_absolute_error", "mean"),
            baseline_MAE=("baseline_absolute_error", "mean")
        )
        .reset_index()
    )

    route_summary.to_csv(
        f"{OUTPUT_DIR}/multi_day_route_metrics.csv",
        index=False
    )

    # ------------------------------------
    # SAVE MODEL SUMMARY
    # ------------------------------------

    metrics = pd.DataFrame([{
        "model": "Gradient Boosting",
        "observations": len(results),
        "MAE": model_mae,
        "RMSE": model_rmse,
        "R2": model_r2,
        "baseline_MAE": baseline_mae,
        "improvement_percent": improvement
    }])

    metrics.to_csv(
        f"{OUTPUT_DIR}/multi_day_validation_summary.csv",
        index=False
    )

    print("\n===== FORECAST HORIZON PERFORMANCE =====")
    print(horizon_summary.to_string(index=False))

    print("\n===== ROUTE PERFORMANCE =====")
    print(route_summary.to_string(index=False))

    print("\nGenerated files:")
    print("1. multi_day_validation_details.csv")
    print("2. forecast_horizon_metrics.csv")
    print("3. multi_day_route_metrics.csv")
    print("4. multi_day_validation_summary.csv")

    print("\nMulti-day validation completed successfully!")


if __name__ == "__main__":
    run_validation()
