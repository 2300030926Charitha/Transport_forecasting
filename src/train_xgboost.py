import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load train and test data
train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

# Convert categorical columns to category codes
for column in ["route", "bus_type"]:
    combined = pd.concat([train_df[column], test_df[column]]).astype("category")

    train_df[column] = combined.iloc[:len(train_df)].cat.codes
    test_df[column] = combined.iloc[len(train_df):].cat.codes

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
    "quarter"
]

X_train = train_df[features]
y_train = train_df["passengers"]

X_test = test_df[features]
y_test = test_df["passengers"]

print("Training XGBoost model...")

# Create model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== XGBOOST RESULTS =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# Save model
joblib.dump(model, "models/xgboost_demand_model.pkl")

print("\nModel saved successfully!")