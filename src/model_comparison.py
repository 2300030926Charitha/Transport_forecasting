import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

# -----------------------------
# 1. BASELINE PREDICTION
# -----------------------------

route_bus_avg = (
    train_df
    .groupby(["route", "bus_type"])["passengers"]
    .mean()
    .reset_index()
    .rename(columns={"passengers": "baseline_prediction"})
)

baseline_test = test_df.merge(
    route_bus_avg,
    on=["route", "bus_type"],
    how="left"
)

overall_average = train_df["passengers"].mean()

baseline_test["baseline_prediction"] = (
    baseline_test["baseline_prediction"]
    .fillna(overall_average)
)

y_actual = baseline_test["passengers"]
baseline_pred = baseline_test["baseline_prediction"]

baseline_mae = mean_absolute_error(y_actual, baseline_pred)
baseline_rmse = mean_squared_error(y_actual, baseline_pred) ** 0.5
baseline_r2 = r2_score(y_actual, baseline_pred)

# -----------------------------
# 2. XGBOOST PREDICTION
# -----------------------------

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
    "quarter"
]

# Convert categories exactly as during training
for column in ["route", "bus_type"]:
    combined = pd.concat(
        [train_df[column], test_df[column]]
    ).astype("category")

    train_df[column] = combined.iloc[:len(train_df)].cat.codes
    test_df[column] = combined.iloc[len(train_df):].cat.codes

model = joblib.load("models/xgboost_demand_model.pkl")

x_test = test_df[features]
xgboost_pred = model.predict(x_test)

xgb_mae = mean_absolute_error(y_actual, xgboost_pred)
xgb_rmse = mean_squared_error(y_actual, xgboost_pred) ** 0.5
xgb_r2 = r2_score(y_actual, xgboost_pred)

# -----------------------------
# 3. DISPLAY COMPARISON
# -----------------------------

comparison = pd.DataFrame({
    "Model": ["Baseline", "XGBoost"],
    "MAE": [baseline_mae, xgb_mae],
    "RMSE": [baseline_rmse, xgb_rmse],
    "R2": [baseline_r2, xgb_r2]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nComparison saved to outputs/model_comparison.csv")