
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

MODEL_FILE = "models/advanced/gradient_boosting.pkl"
TRAIN_FILE = "data/processed/advanced_train_data.csv"
TEST_FILE = "data/processed/advanced_test_data.csv"

OUTPUT_DIR = "outputs/advanced"

FEATURES = [
    "route", "bus_type", "capacity", "distance_km",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7"
]


def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2_Score": r2_score(actual, predicted)
    }


def evaluate_model():

    print("\n===== ADVANCED MODEL EVALUATION =====")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model and datasets
    model = joblib.load(MODEL_FILE)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df[FEATURES]
    y_train = train_df["passengers"]

    X_test = test_df[FEATURES]
    y_test = test_df["passengers"]

    # Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(
        y_train, train_predictions
    )

    test_metrics = calculate_metrics(
        y_test, test_predictions
    )

    print("\nTRAINING PERFORMANCE")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTESTING PERFORMANCE")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # Save evaluation report
    report = pd.DataFrame([
        {"Dataset": "Training", **train_metrics},
        {"Dataset": "Testing", **test_metrics}
    ])

    report.to_csv(
        f"{OUTPUT_DIR}/evaluation_report.csv",
        index=False
    )

    # Save detailed predictions and residuals
    results = pd.DataFrame({
        "date": test_df["date"],
        "route": test_df["route"],
        "bus_type": test_df["bus_type"],
        "actual_passengers": y_test,
        "predicted_passengers": test_predictions
    })

    results["residual"] = (
        results["actual_passengers"]
        - results["predicted_passengers"]
    )

    results["absolute_error"] = results["residual"].abs()

    results.to_csv(
        f"{OUTPUT_DIR}/detailed_predictions.csv",
        index=False
    )

    # Actual vs predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_predictions, alpha=0.7)

    minimum = min(y_test.min(), test_predictions.min())
    maximum = max(y_test.max(), test_predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Passengers")
    plt.ylabel("Predicted Passengers")
    plt.title("Actual vs Predicted - Gradient Boosting")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/actual_vs_predicted.png"
    )
    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 6))
    plt.hist(results["residual"], bins=20)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Distribution")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/residual_distribution.png"
    )
    plt.close()

    # Route-wise performance
    route_results = (
        results.groupby("route")
        .agg(
            Actual_Mean=("actual_passengers", "mean"),
            Predicted_Mean=("predicted_passengers", "mean"),
            MAE=("absolute_error", "mean"),
            Records=("actual_passengers", "count")
        )
        .reset_index()
    )

    route_results.to_csv(
        f"{OUTPUT_DIR}/route_performance.csv",
        index=False
    )

    print("\nROUTE-WISE PERFORMANCE")
    print(route_results.to_string(index=False))

    print("\nEvaluation files saved successfully!")


if __name__ == "__main__":
    evaluate_model()
