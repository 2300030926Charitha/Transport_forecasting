import pandas as pd

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_transport_data.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Sort by route, bus type and date
df = df.sort_values(
    ["route", "bus_type", "date"]
).reset_index(drop=True)

# Create historical demand features
group = df.groupby(["route", "bus_type"])["passengers"]

df["lag_1"] = group.shift(1)
df["lag_2"] = group.shift(2)
df["lag_3"] = group.shift(3)

# Rolling averages
df["rolling_mean_3"] = (
    df.groupby(["route", "bus_type"])["passengers"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

df["rolling_mean_7"] = (
    df.groupby(["route", "bus_type"])["passengers"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

# Remove rows where historical features don't exist
df = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_mean_7"
    ]
)

# Save
df.to_csv(
    "data/processed/historical_features_data.csv",
    index=False
)

print("Historical features created successfully!")

print("\nNew features:")
print([
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_7"
])

print("\nDataset shape:", df.shape)

print("\nSample:")
print(
    df[
        [
            "route",
            "bus_type",
            "date",
            "passengers",
            "lag_1",
            "lag_2",
            "rolling_mean_3",
            "rolling_mean_7"
        ]
    ].head(10)
)