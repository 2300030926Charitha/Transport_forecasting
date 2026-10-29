import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

# Calculate historical average demand
route_bus_avg = (
    train_df
    .groupby(["route", "bus_type"])["passengers"]
    .mean()
    .reset_index()
    .rename(columns={"passengers": "predicted_passengers"})
)

# Merge predictions with test data
test_pred = test_df.merge(
    route_bus_avg,
    on=["route", "bus_type"],
    how="left"
)

# Overall average as fallback for unseen combinations
overall_average = train_df["passengers"].mean()

test_pred["predicted_passengers"] = (
    test_pred["predicted_passengers"]
    .fillna(overall_average)
)

# Actual and predicted values
y_actual = test_pred["passengers"]
y_pred = test_pred["predicted_passengers"]

# Evaluation
mae = mean_absolute_error(y_actual, y_pred)
rmse = mean_squared_error(y_actual, y_pred) ** 0.5
r2 = r2_score(y_actual, y_pred)

print("\n===== BASELINE RESULTS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")