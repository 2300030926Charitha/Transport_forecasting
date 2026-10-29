import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_transport_data.csv")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Time-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Weekend feature
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Quarter
df["quarter"] = df["date"].dt.quarter

# Calculate occupancy again for consistency
df["occupancy_rate"] = (
    df["passengers"] / df["capacity"] * 100
)

# Save feature-engineered dataset
df.to_csv(
    "data/processed/feature_engineered_data.csv",
    index=False
)

print("Feature engineering completed successfully!")

print("\nNew columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())