import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/transport_data.csv")

# Display first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Display dataset size
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Display data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check duplicate rows
print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

# Basic statistics
print("\n--- STATISTICS ---")
print(df.describe())