
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# FILE PATHS
# -----------------------------------

INPUT_DIR = "outputs/advanced/validation"
OUTPUT_DIR = "outputs/advanced/validation/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAILS_FILE = os.path.join(
    INPUT_DIR, "multi_day_validation_details.csv"
)

HORIZON_FILE = os.path.join(
    INPUT_DIR, "forecast_horizon_metrics.csv"
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

def load_data():

    if not os.path.exists(DETAILS_FILE):
        raise FileNotFoundError(
            f"Validation file not found: {DETAILS_FILE}"
        )

    if not os.path.exists(HORIZON_FILE):
        raise FileNotFoundError(
            f"Horizon file not found: {HORIZON_FILE}"
        )

    details = pd.read_csv(DETAILS_FILE)
    horizon = pd.read_csv(HORIZON_FILE)

    required = [
        "actual_passengers",
        "predicted_passengers",
        "baseline_prediction",
        "forecast_origin",
        "forecast_date",
        "forecast_day"
    ]

    missing = [c for c in required if c not in details.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    details["error"] = (
        details["actual_passengers"] -
        details["predicted_passengers"]
    )

    details["absolute_error"] = details["error"].abs()

    details["baseline_absolute_error"] = (
        details["actual_passengers"] -
        details["baseline_prediction"]
    ).abs()

    return details, horizon


# -----------------------------------
# CHART 1: HORIZON COMPARISON
# -----------------------------------

def plot_horizon_comparison(horizon):

    plt.figure(figsize=(10, 6))

    plt.plot(
        horizon["forecast_day"],
        horizon["model_MAE"],
        marker="o",
        label="Gradient Boosting"
    )

    plt.plot(
        horizon["forecast_day"],
        horizon["baseline_MAE"],
        marker="s",
        label="Previous-demand baseline"
    )

    plt.title("MAE by Forecast Horizon")
    plt.xlabel("Forecast Day")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(range(1, 8))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, "horizon_mae_comparison.png"
        )
    )

    plt.close()


# -----------------------------------
# CHART 2: ACTUAL VS PREDICTED
# -----------------------------------

def plot_actual_vs_predicted(details):

    actual = details["actual_passengers"]
    predicted = details["predicted_passengers"]

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.figure(figsize=(8, 6))

    plt.scatter(actual, predicted, alpha=0.7)

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.title("Actual vs Predicted Passengers")
    plt.xlabel("Actual Passengers")
    plt.ylabel("Predicted Passengers")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, "actual_vs_predicted.png"
        )
    )

    plt.close()


# -----------------------------------
# CHART 3: ERROR DISTRIBUTION
# -----------------------------------

def plot_error_distribution(details):

    plt.figure(figsize=(8, 6))

    plt.hist(
        details["error"],
        bins=15,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero error"
    )

    plt.title("Forecast Error Distribution")
    plt.xlabel("Actual - Predicted Passengers")
    plt.ylabel("Number of Observations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR, "forecast_error_distribution.png"
        )
    )

    plt.close()


# -----------------------------------
# CHART 4: MODEL ERROR BY ORIGIN
# -----------------------------------

def analyze_forecast_origins(details):

    origin_summary = (
        details.groupby("forecast_origin")
        .agg(
            observations=("actual_passengers", "count"),
            model_MAE=("absolute_error", "mean"),
            baseline_MAE=("baseline_absolute_error", "mean")
        )
        .reset_index()
    )

    origin_summary.to_csv(
        os.path.join(
            OUTPUT_DIR, "forecast_origin_analysis.csv"
        ),
        index=False
    )

    return origin_summary


# -----------------------------------
# MAIN
# -----------------------------------

def main():

    print("\n===== FORECAST PERFORMANCE ANALYSIS =====")

    details, horizon = load_data()

    plot_horizon_comparison(horizon)
    plot_actual_vs_predicted(details)
    plot_error_distribution(details)

    origin_summary = analyze_forecast_origins(details)

    print("\n===== FORECAST ORIGIN ANALYSIS =====")
    print(origin_summary.to_string(index=False))

    print("\n===== ANALYSIS SUMMARY =====")

    model_mae = details["absolute_error"].mean()
    baseline_mae = details["baseline_absolute_error"].mean()

    print(f"Model MAE: {model_mae:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")

    if model_mae < baseline_mae:
        print("Model performs better overall than baseline.")
    elif model_mae > baseline_mae:
        print("Baseline performs better overall than model.")
    else:
        print("Model and baseline have equal MAE.")

    print("\nGenerated charts and reports:")
    print(OUTPUT_DIR)

    print("\nPhase 9 analysis completed successfully!")


if __name__ == "__main__":
    main()
