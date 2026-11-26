
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------
# CONFIGURATION
# --------------------------------

TEST_FILE = "data/processed/advanced_test_data.csv"

MODEL_FILE = (
    "models/advanced/gradient_boosting.pkl"
)

OUTPUT_DIR = "outputs/advanced"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "route", "bus_type", "capacity", "distance_km",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7"
]


# --------------------------------
# METRIC CALCULATION
# --------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


# --------------------------------
# MAIN VALIDATION
# --------------------------------

def validate_forecasting():

    print("\n===== FORECASTING RELIABILITY VALIDATION =====")

    # Load data
    df = pd.read_csv(TEST_FILE)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "route", "bus_type"]
    ).reset_index(drop=True)

    # Load original advanced Gradient Boosting model
    model = joblib.load(MODEL_FILE)

    X_test = df[FEATURES]
    actual = df["passengers"]

    # Model predictions
    predicted = model.predict(X_test)

    # Previous observation baseline
    baseline = df["lag_1"]

    # --------------------------------
    # MODEL COMPARISON
    # --------------------------------

    model_metrics = calculate_metrics(
        actual, predicted
    )

    baseline_metrics = calculate_metrics(
        actual, baseline
    )

    comparison = pd.DataFrame([
        {
            "Model": "Gradient Boosting",
            **model_metrics
        },
        {
            "Model": "Previous Demand Baseline",
            **baseline_metrics
        }
    ])

    comparison.to_csv(
        f"{OUTPUT_DIR}/forecast_baseline_comparison.csv",
        index=False
    )

    print("\nMODEL VS BASELINE")
    print(comparison.to_string(index=False))

    improvement = (
        (baseline_metrics["MAE"] - model_metrics["MAE"])
        / baseline_metrics["MAE"]
    ) * 100

    print(
        f"\nMAE Improvement over baseline: {improvement:.2f}%"
    )

    if improvement > 0:
        print("Gradient Boosting performs better than baseline.")
    elif improvement < 0:
        print("Baseline performs better than Gradient Boosting.")
    else:
        print("Both methods have equal MAE.")

    # --------------------------------
    # DETAILED FORECAST RESULTS
    # --------------------------------

    results = df[
        ["date", "route", "bus_type", "passengers", "lag_1"]
    ].copy()

    results["predicted_passengers"] = predicted

    results["baseline_prediction"] = baseline

    results["model_error"] = (
        results["passengers"]
        - results["predicted_passengers"]
    )

    results["baseline_error"] = (
        results["passengers"]
        - results["baseline_prediction"]
    )

    results["absolute_error"] = (
        results["model_error"].abs()
    )

    results["forecast_status"] = np.where(
        results["absolute_error"] <= 5,
        "Good",
        np.where(
            results["absolute_error"] <= 10,
            "Moderate",
            "Needs Improvement"
        )
    )

    results.to_csv(
        f"{OUTPUT_DIR}/forecast_reliability_details.csv",
        index=False
    )

    # --------------------------------
    # ROUTE-WISE VALIDATION
    # --------------------------------

    route_analysis = (
        results.groupby("route")
        .agg(
            Actual_Mean=("passengers", "mean"),
            Predicted_Mean=("predicted_passengers", "mean"),
            Model_MAE=("absolute_error", "mean"),
            Baseline_MAE=(
                "baseline_error",
                lambda x: x.abs().mean()
            ),
            Records=("passengers", "count")
        )
        .reset_index()
    )

    route_analysis["MAE_Improvement_Percent"] = (
        (
            route_analysis["Baseline_MAE"]
            - route_analysis["Model_MAE"]
        )
        / route_analysis["Baseline_MAE"].replace(0, np.nan)
    ) * 100

    route_analysis.to_csv(
        f"{OUTPUT_DIR}/forecast_route_validation.csv",
        index=False
    )

    print("\nROUTE-WISE VALIDATION")
    print(route_analysis.to_string(index=False))

    # --------------------------------
    # FORECAST STATUS SUMMARY
    # --------------------------------

    print("\nFORECAST STATUS SUMMARY")

    print(
        results["forecast_status"]
        .value_counts()
        .to_string()
    )

    # --------------------------------
    # FINAL SUMMARY
    # --------------------------------

    print("\n===== VALIDATION COMPLETED =====")

    print(f"Total test records: {len(results)}")
    print(f"Baseline MAE: {baseline_metrics['MAE']:.4f}")
    print(f"Model MAE: {model_metrics['MAE']:.4f}")

    print("\nGenerated files:")
    print("1. forecast_baseline_comparison.csv")
    print("2. forecast_reliability_details.csv")
    print("3. forecast_route_validation.csv")


if __name__ == "__main__":
    validate_forecasting()
