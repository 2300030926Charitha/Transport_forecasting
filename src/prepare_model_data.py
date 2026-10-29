import pandas as pd

# Load feature-engineered data
df = pd.read_csv("data/processed/feature_engineered_data.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# Display date range
print("Date range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# Select features and target
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

target = "passengers"

# Check required columns
print("\nRequired columns:")
for column in features + [target]:
    print(column, "->", "OK" if column in df.columns else "MISSING")

# Create model dataset
model_df = df[["date"] + features + [target]].copy()

# Time-based 80/20 split
split_index = int(len(model_df) * 0.80)

train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining period:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nTesting period:")
print(test_df["date"].min(), "to", test_df["date"].max())

# Save datasets
train_df.to_csv(
    "data/processed/train_data.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test_data.csv",
    index=False
)

print("\nTraining and testing datasets saved successfully!")