import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/transport_data.csv")

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where important values are missing
df = df.dropna(subset=[
    "route",
    "bus_type",
    "date",
    "passengers",
    "capacity"
])

# Make sure numerical columns are numeric
df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

# Remove invalid passenger/capacity values
df = df[
    (df["passengers"] >= 0) &
    (df["capacity"] > 0)
]

# Sort by date
df = df.sort_values("date")

print("\nCleaned shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_transport_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")