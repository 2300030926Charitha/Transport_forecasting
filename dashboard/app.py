import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from math import ceil

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Transport Demand Forecasting",
    page_icon="🚌",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "historical_features_data.csv"
MODEL_FILE = BASE_DIR / "models" / "improved_xgboost_model.pkl"
PREDICTION_FILE = BASE_DIR / "outputs" / "predictions.csv"
DEPLOYMENT_FILE = BASE_DIR / "outputs" / "vehicle_deployment_recommendations.csv"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_predictions():
    if PREDICTION_FILE.exists():
        df = pd.read_csv(PREDICTION_FILE)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_deployment():
    if DEPLOYMENT_FILE.exists():
        df = pd.read_csv(DEPLOYMENT_FILE)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# =========================================================
# LOAD FILES
# =========================================================

try:
    df = load_data()
    predictions = load_predictions()
    deployment = load_deployment()
    model = load_model()

except Exception as e:
    st.error("Error loading project files.")
    st.code(str(e))
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.title("🚌 Transport Demand Forecasting Dashboard")

st.markdown(
    """
    ### Urban Public Transit Deployment

    This dashboard analyzes passenger demand, compares actual and
    predicted demand, and recommends the required number of buses.
    """
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔎 Filters")

routes = sorted(df["route"].dropna().unique())
bus_types = sorted(df["bus_type"].dropna().unique())

selected_route = st.sidebar.selectbox(
    "Select Route",
    ["All Routes"] + routes
)

selected_bus = st.sidebar.selectbox(
    "Select Bus Type",
    ["All Bus Types"] + bus_types
)

# Apply filters
filtered = df.copy()

if selected_route != "All Routes":
    filtered = filtered[
        filtered["route"] == selected_route
    ]

if selected_bus != "All Bus Types":
    filtered = filtered[
        filtered["bus_type"] == selected_bus
    ]

# =========================================================
# KPI SECTION
# =========================================================

st.header("📌 Key Performance Indicators")

total_trips = len(filtered)

average_passengers = filtered["passengers"].mean()

total_passengers = filtered["passengers"].sum()

average_occupancy = filtered["occupancy_rate"].mean()

if not deployment.empty:

    deployment_filtered = deployment.copy()

    if selected_route != "All Routes":
        deployment_filtered = deployment_filtered[
            deployment_filtered["route"] == selected_route
        ]

    if "required_buses" in deployment_filtered.columns:
        buses_required = deployment_filtered[
            "required_buses"
        ].sum()
    else:
        buses_required = 0

else:
    buses_required = 0


c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Trips",
    f"{total_trips:,}"
)

c2.metric(
    "Avg Passengers",
    f"{average_passengers:.2f}"
)

c3.metric(
    "Total Passengers",
    f"{total_passengers:,.0f}"
)

c4.metric(
    "Buses Required",
    f"{buses_required:,.0f}"
)

c5.metric(
    "Avg Occupancy",
    f"{average_occupancy:.1%}"
)

st.divider()

# =========================================================
# DEMAND ANALYSIS
# =========================================================

st.header("📊 Passenger Demand Analysis")

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# ROUTE DEMAND
# ---------------------------------------------------------

with col1:

    route_demand = (
        filtered.groupby("route")["passengers"]
        .mean()
        .reset_index()
        .sort_values("passengers", ascending=False)
    )

    fig1 = px.bar(
        route_demand,
        x="route",
        y="passengers",
        title="Average Passenger Demand by Route",
        labels={
            "route": "Route",
            "passengers": "Average Passengers"
        }
    )

    fig1.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# ---------------------------------------------------------
# BUS TYPE DEMAND
# ---------------------------------------------------------

with col2:

    bus_demand = (
        filtered.groupby("bus_type")["passengers"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        bus_demand,
        x="bus_type",
        y="passengers",
        title="Average Demand by Bus Type",
        labels={
            "bus_type": "Bus Type",
            "passengers": "Average Passengers"
        }
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# DEMAND OVER TIME
# =========================================================

st.subheader("📅 Passenger Demand Over Time")

time_data = (
    filtered.groupby("date")["passengers"]
    .sum()
    .reset_index()
)

fig3 = px.line(
    time_data,
    x="date",
    y="passengers",
    markers=True,
    title="Passenger Demand Trend"
)

fig3.update_layout(
    xaxis_title="Date",
    yaxis_title="Passengers"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.header("🎯 Actual vs Predicted Demand")

if not predictions.empty:

    pred_filtered = predictions.copy()

    if selected_route != "All Routes":
        pred_filtered = pred_filtered[
            pred_filtered["route"] == selected_route
        ]

    if {
        "date",
        "actual_passengers",
        "predicted_passengers"
    }.issubset(pred_filtered.columns):

        fig4 = px.line(
            pred_filtered.sort_values("date"),
            x="date",
            y=[
                "actual_passengers",
                "predicted_passengers"
            ],
            markers=True,
            title="Actual vs Predicted Passenger Demand"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.dataframe(
            pred_filtered,
            use_container_width=True
        )

else:

    st.warning(
        "predictions.csv was not found."
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header("🤖 Model Performance")

metrics = pd.DataFrame({
    "Model": [
        "Baseline",
        "Original XGBoost",
        "Improved XGBoost"
    ],
    "MAE": [
        8.39,
        8.89,
        8.29
    ],
    "RMSE": [
        10.45,
        10.99,
        10.34
    ],
    "R²": [
        0.1332,
        0.0415,
        0.1740
    ]
})

st.dataframe(
    metrics,
    use_container_width=True
)

fig5 = px.bar(
    metrics,
    x="Model",
    y=["MAE", "RMSE"],
    barmode="group",
    title="Model Error Comparison"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# =========================================================
# VEHICLE DEPLOYMENT
# =========================================================

st.header("🚌 Vehicle Deployment Recommendations")

if not deployment.empty:

    dep_filtered = deployment.copy()

    if selected_route != "All Routes":
        dep_filtered = dep_filtered[
            dep_filtered["route"] == selected_route
        ]

    # -----------------------------------------------------
    # Required buses
    # -----------------------------------------------------

    if "required_buses" in dep_filtered.columns:

        bus_route = (
            dep_filtered.groupby("route")["required_buses"]
            .sum()
            .reset_index()
        )

        fig6 = px.bar(
            bus_route,
            x="route",
            y="required_buses",
            title="Required Buses by Route"
        )

        fig6.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Deployment table
    # -----------------------------------------------------

    st.subheader("Deployment Details")

    st.dataframe(
        dep_filtered,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    csv = dep_filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Deployment Recommendations",
        csv,
        "vehicle_deployment_recommendations.csv",
        "text/csv"
    )

else:

    st.warning(
        "Vehicle deployment file was not found."
    )

# =========================================================
# FORECAST SIMULATOR
# =========================================================

st.header("🔮 Demand Forecast Simulator")

st.write(
    "Enter route and vehicle information to estimate passenger "
    "demand and required buses."
)

sim1, sim2, sim3 = st.columns(3)

with sim1:

    forecast_route = st.selectbox(
        "Forecast Route",
        routes
    )

    forecast_bus = st.selectbox(
        "Bus Type",
        bus_types
    )

with sim2:

    route_history = df[
        df["route"] == forecast_route
    ]

    default_capacity = int(
        route_history["capacity"].median()
    )

    default_distance = float(
        route_history["distance_km"].median()
    )

    capacity = st.number_input(
        "Bus Capacity",
        min_value=1,
        value=default_capacity
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=1.0,
        value=default_distance
    )

with sim3:

    forecast_date = st.date_input(
        "Forecast Date",
        value=df["date"].max().date()
    )

    forecast_date = pd.Timestamp(
        forecast_date
    )

# =========================================================
# FORECAST BUTTON
# =========================================================

if st.button(
    "🚀 Predict Passenger Demand",
    type="primary"
):

    history = df[
        (df["route"] == forecast_route)
        & (df["date"] < forecast_date)
    ].sort_values("date")

    if history.empty:

        st.error(
            "Not enough historical data for this route."
        )

    else:

        passengers = history[
            "passengers"
        ].tolist()

        lag1 = passengers[-1]

        lag2 = (
            passengers[-2]
            if len(passengers) >= 2
            else lag1
        )

        lag3 = (
            passengers[-3]
            if len(passengers) >= 3
            else lag2
        )

        rolling3 = np.mean(
            passengers[-3:]
        )

        rolling7 = np.mean(
            passengers[-7:]
        )

        # Same category encoding used during training
        route_categories = pd.Categorical(
            df["route"]
        ).categories

        bus_categories = pd.Categorical(
            df["bus_type"]
        ).categories

        route_code = list(
            route_categories
        ).index(
            forecast_route
        )

        bus_code = list(
            bus_categories
        ).index(
            forecast_bus
        )

        model_input = pd.DataFrame([{

            "route": route_code,

            "bus_type": bus_code,

            "capacity": capacity,

            "distance_km": distance,

            "year": forecast_date.year,

            "month": forecast_date.month,

            "day": forecast_date.day,

            "day_of_week":
                forecast_date.dayofweek,

            "week_of_year":
                forecast_date.isocalendar().week,

            "is_weekend":
                int(forecast_date.dayofweek >= 5),

            "quarter":
                forecast_date.quarter,

            "lag_1": lag1,

            "lag_2": lag2,

            "lag_3": lag3,

            "rolling_mean_3":
                rolling3,

            "rolling_mean_7":
                rolling7

        }])

        feature_order = [

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

        model_input = model_input[
            feature_order
        ]

        predicted = float(
            model.predict(
                model_input
            )[0]
        )

        predicted = max(
            0,
            round(predicted, 2)
        )

        required_buses = ceil(
            predicted / capacity
        )

        total_capacity = (
            required_buses * capacity
        )

        unused_capacity = (
            total_capacity - predicted
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.success(
            "Passenger demand prediction completed!"
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Predicted Passengers",
            f"{predicted:.2f}"
        )

        r2.metric(
            "Required Buses",
            f"{required_buses}"
        )

        r3.metric(
            "Total Capacity",
            f"{total_capacity}"
        )

        r4.metric(
            "Unused Capacity",
            f"{unused_capacity:.2f}"
        )

        if unused_capacity <= capacity * 0.25:

            st.success(
                "Deployment Status: Efficient Capacity"
            )

        else:

            st.info(
                "Deployment Status: Sufficient Capacity"
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Transport Demand Forecasting for Urban Public Transit Deployment"
)