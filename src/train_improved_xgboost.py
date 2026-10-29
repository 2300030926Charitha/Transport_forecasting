import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/historical_features_data.csv"
)

# Convert date
df["date"] = pd.to_datetime(df["date"])

# ==========================================
# 2. CREATE TIME FEATURES
# ==========================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["quarter"] = df["date"].dt.quarter


# ==========================================
# 3. SORT DATA
# ==========================================

df = df.sort_values("date").reset_index(drop=True)


# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

for column in ["route", "bus_type"]:
    df[column] = df[column].astype("category").cat.codes


# ==========================================
# 5. SELECT FEATURES
# ==========================================

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

target = "passengers"


# ==========================================
# 6. CREATE X AND Y
# ==========================================

X = df[features]
y = df[target]


# ==========================================
# 7. TIME-BASED TRAIN/TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 8. CREATE XGBOOST MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nTraining improved XGBoost model...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# 10. PREDICT
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 11. EVALUATE
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n===== IMPROVED XGBOOST RESULTS =====")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/improved_xgboost_model.pkl"
)

print("\nImproved model saved successfully!")