import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_transport_data.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== ROUTES =====")
print(df["route"].value_counts())

print("\n===== BUS TYPES =====")
print(df["bus_type"].value_counts())

print("\n===== AVERAGE PASSENGERS BY ROUTE =====")
route_demand = df.groupby("route")["passengers"].mean().sort_values(ascending=False)
print(route_demand)

print("\n===== AVERAGE PASSENGERS BY BUS TYPE =====")
bus_demand = df.groupby("bus_type")["passengers"].mean().sort_values(ascending=False)
print(bus_demand)

print("\n===== DATE RANGE =====")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print("\n===== PASSENGER STATISTICS =====")
print(df["passengers"].describe())

# -------------------------------
# Route-wise demand graph
# -------------------------------

route_demand.plot(kind="bar", figsize=(10, 6))

plt.title("Average Passenger Demand by Route")
plt.xlabel("Route")
plt.ylabel("Average Passengers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/route_demand.png")
plt.show()

# -------------------------------
# Bus-type demand graph
# -------------------------------

bus_demand.plot(kind="bar", figsize=(10, 6))

plt.title("Average Passenger Demand by Bus Type")
plt.xlabel("Bus Type")
plt.ylabel("Average Passengers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/bus_type_demand.png")
plt.show()