import pandas as pd
import math

# ==========================================
# 1. LOAD PREDICTIONS
# ==========================================

predictions = pd.read_csv(
    "outputs/predictions.csv"
)

# ==========================================
# 2. LOAD ORIGINAL DATA
# ==========================================

original = pd.read_csv(
    "data/raw/transport_data.csv"
)

# ==========================================
# 3. GET CAPACITY INFORMATION
# ==========================================

bus_info = (
    original[
        ["route", "bus_type", "capacity"]
    ]
    .drop_duplicates()
)

# Make route columns strings
predictions["route"] = predictions["route"].astype(str)
bus_info["route"] = bus_info["route"].astype(str)

# ==========================================
# IMPORTANT:
# predictions currently contains encoded route
# numbers, so we need to recover route names.
# ==========================================

route_mapping = (
    original["route"]
    .astype("category")
)

mapping_df = pd.DataFrame({
    "route_code": route_mapping.cat.codes,
    "route": route_mapping.astype(str)
})

mapping_df = mapping_df.drop_duplicates()

# Convert prediction route code to integer
predictions["route_code"] = predictions["route"].astype(int)

# Recover original route name
predictions = predictions.drop(
    columns=["route"]
)

predictions = predictions.merge(
    mapping_df,
    on="route_code",
    how="left"
)

# ==========================================
# 4. MERGE CAPACITY INFORMATION
# ==========================================

predictions = predictions.merge(
    bus_info,
    on="route",
    how="left"
)

# ==========================================
# 5. CHECK CAPACITY
# ==========================================

print("\n===== CAPACITY CHECK =====")

print(
    predictions[
        ["route", "capacity"]
    ].head(10)
)

# ==========================================
# 6. CALCULATE REQUIRED BUSES
# ==========================================

predictions["required_buses"] = (
    predictions["predicted_passengers"]
    / predictions["capacity"]
).apply(math.ceil)

# At least one bus
predictions["required_buses"] = (
    predictions["required_buses"]
    .clip(lower=1)
)

# ==========================================
# 7. TOTAL CAPACITY
# ==========================================

predictions["total_capacity"] = (
    predictions["required_buses"]
    * predictions["capacity"]
)

# ==========================================
# 8. UNUSED CAPACITY
# ==========================================

predictions["unused_capacity"] = (
    predictions["total_capacity"]
    - predictions["predicted_passengers"]
)

# ==========================================
# 9. DEPLOYMENT STATUS
# ==========================================

predictions["deployment_status"] = predictions.apply(
    lambda row:
    "Sufficient Capacity"
    if row["total_capacity"]
    >= row["predicted_passengers"]
    else "Insufficient Capacity",
    axis=1
)

# ==========================================
# 10. SELECT FINAL COLUMNS
# ==========================================

deployment = predictions[
    [
        "date",
        "route",
        "predicted_passengers",
        "capacity",
        "required_buses",
        "total_capacity",
        "unused_capacity",
        "deployment_status"
    ]
]

# ==========================================
# 11. SAVE RESULTS
# ==========================================

deployment.to_csv(
    "outputs/vehicle_deployment_recommendations.csv",
    index=False
)

# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n===== VEHICLE DEPLOYMENT RECOMMENDATIONS =====")

print(
    deployment.head(20).to_string(
        index=False
    )
)

print(
    "\nTotal test trips:",
    len(deployment)
)

print(
    "Total buses required:",
    deployment["required_buses"].sum()
)

print(
    "\nDeployment recommendations saved successfully!"
)