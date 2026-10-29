import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error


# Load data
df = pd.read_csv(
    "data/processed/historical_features_data.csv"
)

df["date"] = pd.to_datetime(df["date"])

# Create time features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["quarter"] = df["date"].dt.quarter

# Sort
df = df.sort_values("date").reset_index(drop=True)

# Encode categorical features
for column in ["route", "bus_type"]:
    df[column] = df[column].astype("category").cat.codes


# Features
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

# Time-based test split
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

X_test = test_df[features]
y_test = test_df["passengers"]

# Load trained model
model = joblib.load(
    "models/improved_xgboost_model.pkl"
)

# Predict
predictions = model.predict(X_test)

# Create result table
results = pd.DataFrame({
    "date": test_df["date"],
    "route": test_df["route"],
    "actual_passengers": y_test.values,
    "predicted_passengers": predictions
})

# Error
results["error"] = (
    results["actual_passengers"]
    - results["predicted_passengers"]
)

results["absolute_error"] = results["error"].abs()

print("\n===== PREDICTION RESULTS =====")
print(results.head(15))

print("\nAverage Absolute Error:")
print(results["absolute_error"].mean())

# Save predictions
results.to_csv(
    "outputs/predictions.csv",
    index=False
)

# -------------------------------
# Actual vs Predicted graph
# -------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    results["actual_passengers"].values,
    label="Actual Demand"
)

plt.plot(
    results["predicted_passengers"].values,
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Passenger Demand")
plt.xlabel("Test Records")
plt.ylabel("Passenger Demand")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.show()

print("\nGraph saved successfully!")