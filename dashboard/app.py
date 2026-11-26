import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from pathlib import Path
from math import ceil
from itertools import count


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transport Demand Forecasting",
    page_icon="🚌",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ADV_DIR = OUTPUT_DIR / "advanced"
VALIDATION_DIR = ADV_DIR / "validation"

HISTORICAL_FILE = DATA_DIR / "historical_features_data.csv"
ADV_HISTORICAL_FILE = DATA_DIR / "advanced_historical_features.csv"

ADV_MODEL_FILE = MODEL_DIR / "advanced" / "gradient_boosting.pkl"

PREDICTION_FILE = OUTPUT_DIR / "predictions.csv"
DEPLOYMENT_FILE = OUTPUT_DIR / "vehicle_deployment_recommendations.csv"

MODEL_COMPARISON_FILE = ADV_DIR / "model_comparison.csv"
EVALUATION_FILE = ADV_DIR / "evaluation_report.csv"
ADV_PREDICTION_FILE = ADV_DIR / "detailed_predictions.csv"

ALIGNED_FORECAST_FILE = ADV_DIR / "aligned_seven_day_forecast.csv"
ALIGNED_DAILY_FILE = ADV_DIR / "aligned_daily_summary.csv"
ALIGNED_ROUTE_FILE = ADV_DIR / "aligned_route_summary.csv"

VALIDATION_SUMMARY_FILE = VALIDATION_DIR / "multi_day_validation_summary.csv"
VALIDATION_DETAILS_FILE = VALIDATION_DIR / "multi_day_validation_details.csv"
HORIZON_FILE = VALIDATION_DIR / "forecast_horizon_metrics.csv"
ROUTE_VALIDATION_FILE = VALIDATION_DIR / "multi_day_route_metrics.csv"


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

_download_counter = count(1)


@st.cache_data(show_spinner=False)
def load_csv(path_string):
    """Load a CSV safely and parse known date columns."""

    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        data = pd.read_csv(path)

        for col in ["date", "forecast_date", "forecast_origin"]:
            if col in data.columns:
                data[col] = pd.to_datetime(
                    data[col],
                    errors="coerce"
                )

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path_string):
    """Load the saved machine learning model."""

    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def convert_numeric(data, columns):
    data = data.copy()

    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    return data


def find_column(data, names):
    if data is None or data.empty:
        return None

    for name in names:
        if name in data.columns:
            return name

    return None


def safe_median(series, fallback=0):
    if series is None:
        return fallback

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    values = values[np.isfinite(values)]

    if values.empty:
        return fallback

    result = float(values.median())

    return result if np.isfinite(result) else fallback


def safe_sum(series):
    if series is None:
        return 0

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    return float(values.sum())


def safe_metric(value, digits=2):
    try:
        value = float(value)

        if np.isfinite(value):
            return f"{value:,.{digits}f}"

    except (TypeError, ValueError):
        pass

    return "N/A"


def filter_data(data, route, bus_type):
    result = data.copy()

    if result.empty:
        return result

    if route != "All Routes" and "route" in result.columns:
        result = result[
            result["route"].astype(str) == route
        ]

    if bus_type != "All Bus Types" and "bus_type" in result.columns:
        result = result[
            result["bus_type"].astype(str) == bus_type
        ]

    return result.copy()


def download_csv(data, filename, label):
    """Create a download button with a unique key every time."""

    if data is None or data.empty:
        return

    unique_id = next(_download_counter)

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"csv_download_{unique_id}_{filename}"
    )


def show_missing(message):
    st.info(message)


def show_dataframe(data):
    if data is not None and not data.empty:
        st.dataframe(
            data,
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No data available to display.")


def show_chart(fig):
    st.plotly_chart(
        fig,
        width="stretch"
    )


def metric_value(row, column):
    if column is None:
        return np.nan

    return pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce"
    ).iloc[0]


# =========================================================
# 4. LOAD PROJECT DATA
# =========================================================

df = load_csv(str(HISTORICAL_FILE))
advanced_df = load_csv(str(ADV_HISTORICAL_FILE))

predictions = load_csv(str(PREDICTION_FILE))
deployment = load_csv(str(DEPLOYMENT_FILE))

model_comparison = load_csv(str(MODEL_COMPARISON_FILE))
evaluation = load_csv(str(EVALUATION_FILE))
advanced_predictions = load_csv(str(ADV_PREDICTION_FILE))

aligned_forecast = load_csv(str(ALIGNED_FORECAST_FILE))
aligned_daily = load_csv(str(ALIGNED_DAILY_FILE))
aligned_route = load_csv(str(ALIGNED_ROUTE_FILE))

validation_summary = load_csv(str(VALIDATION_SUMMARY_FILE))
validation_details = load_csv(str(VALIDATION_DETAILS_FILE))
horizon_metrics = load_csv(str(HORIZON_FILE))
route_validation = load_csv(str(ROUTE_VALIDATION_FILE))

advanced_model = load_model(str(ADV_MODEL_FILE))


# =========================================================
# 5. VALIDATE HISTORICAL DATA
# =========================================================

if df.empty:
    st.error(
        "Historical dataset not found or could not be read:\n\n"
        "data/processed/historical_features_data.csv"
    )
    st.stop()

required_columns = {
    "date",
    "route",
    "bus_type",
    "passengers"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Historical dataset is missing these columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

df = df.copy()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "route", "bus_type"]
).copy()

df["route"] = df["route"].astype(str)
df["bus_type"] = df["bus_type"].astype(str)

df = convert_numeric(
    df,
    [
        "passengers",
        "capacity",
        "occupancy_rate",
        "distance_km",
        "revenue",
        "fuel_consumed_liters"
    ]
)

df = df.dropna(subset=["passengers"]).copy()

if df.empty:
    st.error("No valid historical passenger records are available.")
    st.stop()


# Prepare advanced dataset

if not advanced_df.empty:

    advanced_df = advanced_df.copy()

    if "date" in advanced_df.columns:
        advanced_df["date"] = pd.to_datetime(
            advanced_df["date"],
            errors="coerce"
        )

    advanced_df = convert_numeric(
        advanced_df,
        ["passengers", "capacity", "distance_km"]
    )


# =========================================================
# 6. HEADER
# =========================================================

st.title("🚌 Transport Demand Forecasting Dashboard")

st.markdown("""
### Urban Public Transit Deployment

An intelligent transportation analytics system that analyzes
historical passenger demand, forecasts future demand, evaluates
machine learning models and recommends bus deployment.
""")

st.caption(
    "Forecasts are generated using historical project data, "
    "not live passenger counts."
)

st.divider()


# =========================================================
# 7. SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🔎 Dashboard Filters")

routes = sorted(
    df["route"].dropna().astype(str).unique().tolist()
)

bus_types = sorted(
    df["bus_type"].dropna().astype(str).unique().tolist()
)

selected_route = st.sidebar.selectbox(
    "Select Route",
    ["All Routes"] + routes
)

selected_bus = st.sidebar.selectbox(
    "Select Bus Type",
    ["All Bus Types"] + bus_types
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Historical Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = filter_data(
    df,
    selected_route,
    selected_bus
)

# Support both a date range and a single selected date.

if isinstance(date_range, (tuple, list)):

    if len(date_range) == 2:

        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])

        filtered = filtered[
            (filtered["date"] >= start_date) &
            (filtered["date"] < end_date + pd.Timedelta(days=1))
        ].copy()

    elif len(date_range) == 1:

        selected_date = pd.Timestamp(date_range[0])

        filtered = filtered[
            filtered["date"].dt.date == selected_date.date()
        ].copy()

else:

    selected_date = pd.Timestamp(date_range)

    filtered = filtered[
        filtered["date"].dt.date == selected_date.date()
    ].copy()


if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# =========================================================
# 8. KPI SECTION
# =========================================================

st.header("📌 Key Performance Indicators")

total_trips = len(filtered)

total_passengers = safe_sum(
    filtered["passengers"]
)

average_passengers = (
    filtered["passengers"].mean()
    if not filtered.empty
    else np.nan
)

total_capacity = (
    safe_sum(filtered["capacity"])
    if "capacity" in filtered.columns
    else 0
)

if total_capacity > 0:
    average_occupancy = (
        total_passengers / total_capacity
    ) * 100
else:
    average_occupancy = np.nan


deployment_filtered = filter_data(
    deployment,
    selected_route,
    selected_bus
)

required_col = find_column(
    deployment_filtered,
    [
        "required_buses",
        "estimated_bus_requirements",
        "total_estimated_bus_requirements",
        "total_required_buses"
    ]
)

if required_col and not deployment_filtered.empty:
    buses_required = safe_sum(
        deployment_filtered[required_col]
    )
else:
    buses_required = np.nan


k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Trips",
    f"{total_trips:,}"
)

k2.metric(
    "Average Passengers",
    safe_metric(average_passengers)
)

k3.metric(
    "Total Passengers",
    safe_metric(total_passengers, 0)
)

k4.metric(
    "Buses Required",
    safe_metric(buses_required, 0)
)

k5.metric(
    "Average Occupancy",
    f"{average_occupancy:.1f}%"
    if pd.notna(average_occupancy)
    else "N/A"
)

st.caption(
    f"Historical data available: {min_date} to {max_date}"
)

st.divider()


# =========================================================
# 9. DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Demand Analysis",
    "🤖 Model Performance",
    "🔮 7-Day Forecast",
    "🧪 Validation",
    "🚌 Deployment",
    "⬇️ Reports & Simulator"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab1:

    st.header("📊 Passenger Demand Overview")

    if filtered.empty:

        st.warning("No records match the selected filters.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            route_demand = (
                filtered.groupby("route", as_index=False)
                ["passengers"].mean()
                .sort_values("passengers", ascending=False)
            )

            if not route_demand.empty:

                fig = px.bar(
                    route_demand,
                    x="route",
                    y="passengers",
                    title="Average Passenger Demand by Route",
                    color="passengers"
                )

                fig.update_layout(xaxis_tickangle=-45)
                show_chart(fig)

        with col2:

            bus_demand = (
                filtered.groupby("bus_type", as_index=False)
                ["passengers"].mean()
            )

            if not bus_demand.empty:

                fig = px.bar(
                    bus_demand,
                    x="bus_type",
                    y="passengers",
                    title="Average Demand by Bus Type",
                    color="passengers"
                )

                show_chart(fig)

        st.subheader("📅 Passenger Demand Trend")

        daily_data = (
            filtered.groupby("date", as_index=False)
            ["passengers"].sum()
        )

        if not daily_data.empty:

            fig = px.line(
                daily_data,
                x="date",
                y="passengers",
                markers=True,
                title="Historical Passenger Demand Over Time"
            )

            show_chart(fig)

        st.subheader("Historical Data Preview")
        show_dataframe(filtered.head(20))


# =========================================================
# TAB 2: DEMAND ANALYSIS
# =========================================================

with tab2:

    st.header("📈 Detailed Passenger Demand Analysis")

    if filtered.empty:

        st.warning("No records available for demand analysis.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            route_stats = (
                filtered.groupby("route")
                .agg(
                    Total_Passengers=("passengers", "sum"),
                    Average_Passengers=("passengers", "mean"),
                    Total_Trips=("passengers", "count")
                )
                .reset_index()
            )

            if not route_stats.empty:

                fig = px.bar(
                    route_stats,
                    x="route",
                    y="Total_Passengers",
                    title="Total Passenger Demand by Route"
                )

                fig.update_layout(xaxis_tickangle=-45)
                show_chart(fig)

                show_dataframe(route_stats)

        with col2:

            type_stats = (
                filtered.groupby("bus_type")
                .agg(
                    Total_Passengers=("passengers", "sum"),
                    Average_Passengers=("passengers", "mean")
                )
                .reset_index()
            )

            pie_data = type_stats[
                type_stats["Total_Passengers"] > 0
            ]

            if not pie_data.empty:

                fig = px.pie(
                    pie_data,
                    names="bus_type",
                    values="Total_Passengers",
                    title="Passenger Distribution by Bus Type"
                )

                show_chart(fig)

            else:

                st.info("No positive passenger totals available for the pie chart.")

        st.subheader("Monthly Passenger Demand")

        monthly = filtered.copy()

        monthly["month"] = (
            monthly["date"].dt.to_period("M").astype(str)
        )

        monthly_data = (
            monthly.groupby("month", as_index=False)
            ["passengers"].sum()
        )

        if not monthly_data.empty:

            fig = px.bar(
                monthly_data,
                x="month",
                y="passengers",
                title="Monthly Passenger Demand"
            )

            show_chart(fig)


# =========================================================
# TAB 3: MODEL PERFORMANCE
# =========================================================

with tab3:

    st.header("🤖 Advanced Machine Learning Model Performance")

    if model_comparison.empty:

        show_missing(
            "Model comparison unavailable. Run "
            "python -m src.advanced_model_comparison"
        )

    else:

        st.subheader("Model Comparison Results")
        show_dataframe(model_comparison)

        model_name_col = find_column(
            model_comparison,
            ["Model", "model", "model_name", "Model_Name", "algorithm"]
        )

        mae_col = find_column(
            model_comparison,
            ["MAE", "mae", "test_MAE", "Test MAE", "model_MAE"]
        )

        rmse_col = find_column(
            model_comparison,
            ["RMSE", "rmse", "test_RMSE", "Test RMSE", "model_RMSE"]
        )

        if model_name_col and (mae_col or rmse_col):

            metric_columns = [
                col for col in [mae_col, rmse_col] if col
            ]

            chart_data = model_comparison[
                [model_name_col] + metric_columns
            ].copy()

            for col in metric_columns:
                chart_data[col] = pd.to_numeric(
                    chart_data[col],
                    errors="coerce"
                )

            chart_data = chart_data.melt(
                id_vars=[model_name_col],
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Error"
            ).dropna(subset=["Error"])

            if not chart_data.empty:

                fig = px.bar(
                    chart_data,
                    x=model_name_col,
                    y="Error",
                    color="Metric",
                    barmode="group",
                    title="MAE and RMSE Comparison (Lower is Better)"
                )

                show_chart(fig)

        else:

            st.info(
                "Model comparison chart needs a model-name column "
                "and at least one MAE or RMSE column."
            )

    st.subheader("Detailed Model Evaluation")

    if not evaluation.empty:
        show_dataframe(evaluation)
    else:
        show_missing("Detailed evaluation report is unavailable.")

    if not advanced_predictions.empty:

        actual_col = find_column(
            advanced_predictions,
            ["actual_passengers", "actual", "passengers"]
        )

        predicted_col = find_column(
            advanced_predictions,
            ["predicted_passengers", "predicted"]
        )

        if actual_col and predicted_col:

            plot_data = convert_numeric(
                advanced_predictions,
                [actual_col, predicted_col]
            ).dropna(
                subset=[actual_col, predicted_col]
            )

            if not plot_data.empty:

                fig = px.scatter(
                    plot_data,
                    x=actual_col,
                    y=predicted_col,
                    title="Actual vs Predicted Passenger Demand",
                    labels={
                        actual_col: "Actual Passengers",
                        predicted_col: "Predicted Passengers"
                    }
                )

                show_chart(fig)

            else:

                st.info("No valid actual/predicted values are available.")

        else:

            st.info(
                "Actual vs predicted chart requires actual and predicted "
                "passenger columns."
            )


# =========================================================
# TAB 4: ALIGNED SEVEN-DAY FORECAST
# =========================================================

with tab4:

    st.header("🔮 Aligned Seven-Day Demand Forecast")

    if aligned_daily.empty:

        st.warning(
            "Aligned forecast unavailable. Run "
            "python -m src.aligned_multi_day_forecasting"
        )

    else:

        date_col = find_column(
            aligned_daily,
            ["date", "forecast_date"]
        )

        passenger_col = find_column(
            aligned_daily,
            [
                "total_predicted_passengers",
                "predicted_passengers",
                "total_passengers"
            ]
        )

        buses_col = find_column(
            aligned_daily,
            [
                "total_estimated_bus_requirements",
                "estimated_bus_requirements",
                "required_buses",
                "total_required_buses"
            ]
        )

        forecast_data = aligned_daily.copy()

        if date_col:
            forecast_data[date_col] = pd.to_datetime(
                forecast_data[date_col],
                errors="coerce"
            )

        for col in [passenger_col, buses_col]:
            if col:
                forecast_data[col] = pd.to_numeric(
                    forecast_data[col],
                    errors="coerce"
                )

        a, b, c = st.columns(3)

        a.metric(
            "Forecast Days",
            str(forecast_data[date_col].nunique())
            if date_col else "N/A"
        )

        b.metric(
            "Total Forecast Passenger-Days",
            safe_metric(safe_sum(forecast_data[passenger_col]), 0)
            if passenger_col else "N/A"
        )

        c.metric(
            "Estimated Bus-Days",
            safe_metric(safe_sum(forecast_data[buses_col]), 0)
            if buses_col else "N/A"
        )

        if date_col and passenger_col:

            plot_data = forecast_data.dropna(
                subset=[date_col, passenger_col]
            ).sort_values(date_col)

            if not plot_data.empty:

                fig = px.line(
                    plot_data,
                    x=date_col,
                    y=passenger_col,
                    markers=True,
                    title="Seven-Day Predicted Passenger Demand"
                )

                show_chart(fig)

        if date_col and buses_col:

            plot_data = forecast_data.dropna(
                subset=[date_col, buses_col]
            ).sort_values(date_col)

            if not plot_data.empty:

                fig = px.bar(
                    plot_data,
                    x=date_col,
                    y=buses_col,
                    title="Estimated Bus Requirements by Day"
                )

                show_chart(fig)

        st.subheader("Daily Forecast Summary")
        show_dataframe(forecast_data)

        download_csv(
            forecast_data,
            "aligned_daily_summary.csv",
            "⬇️ Download Daily Forecast"
        )

    st.divider()

    st.subheader("Route-Wise Forecast")

    if not aligned_forecast.empty:

        route_forecast_filtered = filter_data(
            aligned_forecast,
            selected_route,
            selected_bus
        )

        show_dataframe(route_forecast_filtered)

        download_csv(
            route_forecast_filtered,
            "aligned_seven_day_forecast.csv",
            "⬇️ Download Route Forecast"
        )

    else:

        show_missing("Aligned route forecast details are unavailable.")

    if not aligned_route.empty:

        st.subheader("Route Forecast Summary")
        show_dataframe(aligned_route)

        download_csv(
            aligned_route,
            "aligned_route_summary.csv",
            "⬇️ Download Route Summary"
        )


# =========================================================
# TAB 5: MULTI-DAY FORECAST VALIDATION
# =========================================================

with tab5:

    st.header("🧪 Recursive Multi-Day Forecast Validation")

    if validation_summary.empty:

        st.warning(
            "Validation results unavailable. Run "
            "python -m src.multi_day_validation_analysis"
        )

    else:

        st.subheader("Overall Validation Results")
        show_dataframe(validation_summary)

        result = validation_summary.iloc[0]

        model_mae_col = find_column(
            validation_summary,
            ["model_MAE", "Model_MAE", "MAE", "model_mae"]
        )

        baseline_mae_col = find_column(
            validation_summary,
            ["baseline_MAE", "Baseline_MAE", "baseline_mae"]
        )

        improvement_col = find_column(
            validation_summary,
            [
                "improvement_percent",
                "MAE_improvement_percent",
                "improvement",
                "mae_improvement_percent"
            ]
        )

        model_value = metric_value(
            result,
            model_mae_col
        )

        baseline_value = metric_value(
            result,
            baseline_mae_col
        )

        improvement_value = metric_value(
            result,
            improvement_col
        )

        if (
            pd.isna(improvement_value)
            and pd.notna(model_value)
            and pd.notna(baseline_value)
            and baseline_value != 0
        ):

            improvement_value = (
                (baseline_value - model_value)
                / baseline_value
            ) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model MAE",
            safe_metric(model_value, 4)
        )

        c2.metric(
            "Baseline MAE",
            safe_metric(baseline_value, 4)
        )

        c3.metric(
            "MAE Improvement",
            f"{improvement_value:.2f}%"
            if pd.notna(improvement_value)
            else "N/A"
        )

        if pd.notna(model_value) and pd.notna(baseline_value):

            if model_value < baseline_value:

                st.success(
                    "The model achieved lower overall MAE than the baseline."
                )

            elif model_value > baseline_value:

                st.warning(
                    "The baseline achieved lower overall MAE than the model."
                )

            else:

                st.info("The model and baseline have equal MAE.")

    st.divider()

    st.subheader("Forecast Horizon Analysis")

    if horizon_metrics.empty:

        show_missing("Forecast horizon metrics are unavailable.")

    else:

        show_dataframe(horizon_metrics)

        horizon_col = find_column(
            horizon_metrics,
            ["forecast_day", "horizon", "day"]
        )

        model_col = find_column(
            horizon_metrics,
            ["model_MAE", "Model_MAE", "mae", "model_mae"]
        )

        baseline_col = find_column(
            horizon_metrics,
            ["baseline_MAE", "Baseline_MAE", "baseline_mae"]
        )

        if horizon_col and (model_col or baseline_col):

            metric_columns = [
                col for col in [model_col, baseline_col] if col
            ]

            chart_data = horizon_metrics[
                [horizon_col] + metric_columns
            ].copy()

            for col in metric_columns:
                chart_data[col] = pd.to_numeric(
                    chart_data[col],
                    errors="coerce"
                )

            chart_data = chart_data.melt(
                id_vars=[horizon_col],
                value_vars=metric_columns,
                var_name="Method",
                value_name="MAE"
            ).dropna(subset=["MAE"])

            if not chart_data.empty:

                fig = px.line(
                    chart_data,
                    x=horizon_col,
                    y="MAE",
                    color="Method",
                    markers=True,
                    title="Model vs Baseline MAE by Forecast Day"
                )

                show_chart(fig)

    st.divider()

    st.subheader("Route-Wise Validation")

    if not route_validation.empty:

        show_dataframe(route_validation)

        download_csv(
            route_validation,
            "multi_day_route_metrics.csv",
            "⬇️ Download Route Validation"
        )

    else:

        show_missing("Route-wise validation report is unavailable.")

    if not validation_details.empty:

        with st.expander("View Individual Forecast Predictions"):

            show_dataframe(validation_details)

            download_csv(
                validation_details,
                "multi_day_validation_details.csv",
                "⬇️ Download Validation Details"
            )


# =========================================================
# TAB 6: VEHICLE DEPLOYMENT
# =========================================================

with tab6:

    st.header("🚌 Vehicle Deployment Recommendations")

    if deployment.empty:

        st.warning("Vehicle deployment file is unavailable.")

    else:

        dep_filtered = filter_data(
            deployment,
            selected_route,
            selected_bus
        )

        dep_bus_col = find_column(
            dep_filtered,
            [
                "required_buses",
                "estimated_bus_requirements",
                "total_estimated_bus_requirements",
                "total_required_buses"
            ]
        )

        if dep_filtered.empty:

            st.info("No deployment records match the selected filters.")

        else:

            if dep_bus_col and "route" in dep_filtered.columns:

                plot_data = dep_filtered.copy()

                plot_data[dep_bus_col] = pd.to_numeric(
                    plot_data[dep_bus_col],
                    errors="coerce"
                )

                route_buses = (
                    plot_data.groupby("route", as_index=False)
                    [dep_bus_col].sum()
                ).dropna(subset=[dep_bus_col])

                if not route_buses.empty:

                    fig = px.bar(
                        route_buses,
                        x="route",
                        y=dep_bus_col,
                        title="Estimated Bus Requirements by Route"
                    )

                    fig.update_layout(xaxis_tickangle=-45)
                    show_chart(fig)

            st.subheader("Deployment Details")
            show_dataframe(dep_filtered)

            download_csv(
                dep_filtered,
                "vehicle_deployment_recommendations.csv",
                "⬇️ Download Deployment Recommendations"
            )

    st.caption(
        "Bus requirements are simplified capacity-based estimates. "
        "They are not a complete timetable or physical fleet scheduling plan."
    )


# =========================================================
# TAB 7: FORECAST SIMULATOR
# =========================================================

with tab7:

    st.header("🔮 Demand Forecast Simulator")

    if advanced_model is None:

        st.warning(
            "Advanced Gradient Boosting model could not be loaded. "
            "Check models/advanced/gradient_boosting.pkl."
        )

    elif advanced_df.empty:

        st.warning(
            "Advanced historical dataset is unavailable. "
            "Check data/processed/advanced_historical_features.csv."
        )

    else:

        required_sim_columns = {
            "route",
            "bus_type",
            "date",
            "passengers",
            "capacity",
            "distance_km"
        }

        missing_sim_columns = (
            required_sim_columns - set(advanced_df.columns)
        )

        if missing_sim_columns:

            st.error(
                "Advanced historical data is missing: "
                + ", ".join(sorted(missing_sim_columns))
            )

        else:

            simulator_data = advanced_df.copy()

            simulator_data["route"] = (
                simulator_data["route"].astype(str)
            )

            simulator_data["bus_type"] = (
                simulator_data["bus_type"].astype(str)
            )

            simulator_data["date"] = pd.to_datetime(
                simulator_data["date"],
                errors="coerce"
            )

            simulator_data = convert_numeric(
                simulator_data,
                ["passengers", "capacity", "distance_km"]
            )

            simulator_data = simulator_data.dropna(
                subset=["date", "route", "bus_type"]
            ).copy()

            valid_combinations = (
                simulator_data[
                    ["route", "bus_type"]
                ]
                .drop_duplicates()
                .sort_values(["route", "bus_type"])
            )

            if valid_combinations.empty:

                st.error(
                    "No valid route and bus-type combinations are available."
                )

            else:

                sim1, sim2, sim3 = st.columns(3)

                with sim1:

                    sim_route = st.selectbox(
                        "Forecast Route",
                        sorted(
                            valid_combinations["route"].unique()
                        ),
                        key="sim_route"
                    )

                    available_types = sorted(
                        valid_combinations.loc[
                            valid_combinations["route"] == sim_route,
                            "bus_type"
                        ].unique()
                    )

                    sim_bus_type = st.selectbox(
                        "Bus Type",
                        available_types,
                        key=f"sim_bus_type_{sim_route}"
                    )

                history_all = simulator_data[
                    (simulator_data["route"] == sim_route) &
                    (simulator_data["bus_type"] == sim_bus_type)
                ].sort_values("date").copy()

                route_history = simulator_data[
                    simulator_data["route"] == sim_route
                ].copy()

                # Safe capacity default

                capacity_default = safe_median(
                    history_all["capacity"],
                    safe_median(
                        route_history["capacity"],
                        safe_median(
                            simulator_data["capacity"],
                            50
                        )
                    )
                )

                capacity_default = max(
                    1,
                    int(round(capacity_default))
                )

                # Safe distance default

                distance_default = safe_median(
                    history_all["distance_km"],
                    safe_median(
                        route_history["distance_km"],
                        safe_median(
                            simulator_data["distance_km"],
                            100.0
                        )
                    )
                )

                distance_default = max(
                    0.1,
                    float(distance_default)
                )

                with sim2:

                    capacity = st.number_input(
                        "Bus Capacity",
                        min_value=1,
                        value=capacity_default,
                        step=1,
                        key="sim_capacity"
                    )

                    distance = st.number_input(
                        "Distance (km)",
                        min_value=0.1,
                        value=float(distance_default),
                        step=1.0,
                        key="sim_distance"
                    )

                latest_history_date = history_all["date"].max()

                if pd.isna(latest_history_date):
                    latest_history_date = simulator_data["date"].max()

                default_forecast_date = (
                    latest_history_date + pd.Timedelta(days=1)
                ).date()

                with sim3:

                    forecast_date_input = st.date_input(
                        "Forecast Date",
                        value=default_forecast_date,
                        key="sim_forecast_date"
                    )

                st.caption(
                    "The simulator uses records dated before the forecast "
                    "date to calculate lag and rolling-average features."
                )

                if st.button(
                    "🚀 Predict Passenger Demand",
                    type="primary",
                    key="predict_demand"
                ):

                    forecast_timestamp = pd.Timestamp(
                        forecast_date_input
                    )

                    # Important: use only records before forecast date.
                    valid_history = history_all[
                        (history_all["date"] < forecast_timestamp)
                    ].dropna(
                        subset=["passengers"]
                    ).sort_values("date").copy()

                    if valid_history.empty:

                        st.error(
                            "No passenger history exists before the selected "
                            "forecast date. Select a later forecast date."
                        )

                    elif capacity <= 0:

                        st.error("Bus capacity must be greater than zero.")

                    elif distance <= 0:

                        st.error("Distance must be greater than zero.")

                    else:

                        passenger_history = (
                            valid_history["passengers"]
                            .astype(float)
                            .tolist()
                        )

                        lag1 = passenger_history[-1]

                        lag2 = (
                            passenger_history[-2]
                            if len(passenger_history) >= 2
                            else lag1
                        )

                        lag3 = (
                            passenger_history[-3]
                            if len(passenger_history) >= 3
                            else lag2
                        )

                        rolling3 = float(
                            np.mean(passenger_history[-3:])
                        )

                        rolling7 = float(
                            np.mean(passenger_history[-7:])
                        )

                        # Keep the exact feature names and order
                        # used during Gradient Boosting model training.

                        expected_features = [
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

                        model_input = pd.DataFrame([{

                            "route": sim_route,
                            "bus_type": sim_bus_type,

                            "capacity": float(capacity),
                            "distance_km": float(distance),

                            "year": forecast_timestamp.year,
                            "month": forecast_timestamp.month,
                            "day": forecast_timestamp.day,

                            "day_of_week": forecast_timestamp.dayofweek,

                            "week_of_year": int(
                                forecast_timestamp.isocalendar().week
                            ),

                            "is_weekend": int(
                                forecast_timestamp.dayofweek >= 5
                            ),

                            "quarter": forecast_timestamp.quarter,

                            "lag_1": lag1,
                            "lag_2": lag2,
                            "lag_3": lag3,

                            "rolling_mean_3": rolling3,
                            "rolling_mean_7": rolling7

                        }])

                        model_input = model_input[
                            expected_features
                        ]

                        try:

                            prediction = float(
                                advanced_model.predict(
                                    model_input
                                )[0]
                            )

                            if not np.isfinite(prediction):
                                raise ValueError(
                                    "The model returned an invalid prediction."
                                )

                            predicted = max(
                                0.0,
                                prediction
                            )

                            required_buses = ceil(
                                predicted / float(capacity)
                            )

                            total_bus_capacity = (
                                required_buses * capacity
                            )

                            unused_capacity = (
                                total_bus_capacity - predicted
                            )

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
                                f"{total_bus_capacity}"
                            )

                            r4.metric(
                                "Unused Capacity",
                                f"{unused_capacity:.2f}"
                            )

                            st.caption(
                                "Bus requirement = ceiling of predicted "
                                "passengers divided by bus capacity."
                            )

                            result = model_input.copy()

                            result["forecast_date"] = forecast_timestamp
                            result["predicted_passengers"] = predicted
                            result["required_buses"] = required_buses
                            result["total_capacity"] = total_bus_capacity
                            result["unused_capacity"] = unused_capacity

                            st.subheader("Prediction Details")
                            show_dataframe(result)

                            download_csv(
                                result,
                                "demand_prediction.csv",
                                "⬇️ Download Prediction"
                            )

                        except Exception as e:

                            st.error(
                                "Prediction failed. Check that the saved "
                                "Gradient Boosting model uses the expected "
                                "features and preprocessing pipeline."
                            )

                            st.exception(e)


    # =====================================================
    # PROJECT REPORT DOWNLOADS
    # =====================================================

    st.divider()

    st.header("⬇️ Download Project Reports")

    report_files = [
        ("Historical Data", df, "historical_data.csv"),
        ("Advanced Historical Features", advanced_df, "advanced_historical_features.csv"),
        ("Model Comparison", model_comparison, "model_comparison.csv"),
        ("Model Evaluation", evaluation, "evaluation_report.csv"),
        ("Advanced Predictions", advanced_predictions, "detailed_predictions.csv"),
        ("Seven-Day Forecast", aligned_forecast, "aligned_seven_day_forecast.csv"),
        ("Daily Forecast Summary", aligned_daily, "aligned_daily_summary.csv"),
        ("Route Forecast Summary", aligned_route, "aligned_route_summary.csv"),
        ("Validation Summary", validation_summary, "multi_day_validation_summary.csv"),
        ("Validation Details", validation_details, "multi_day_validation_details.csv"),
        ("Forecast Horizon Metrics", horizon_metrics, "forecast_horizon_metrics.csv"),
        ("Route Validation", route_validation, "multi_day_route_metrics.csv"),
        ("Vehicle Deployment", deployment, "vehicle_deployment_recommendations.csv")
    ]

    for label, data, filename in report_files:

        if data is not None and not data.empty:

            download_csv(
                data,
                filename,
                f"⬇️ Download {label}"
            )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    "Transport Demand Forecasting for Urban Public Transit Deployment | "
    "Machine Learning | Advanced Forecasting | Bus Deployment"
)

st.caption(
    "The historical dataset ends on 2024-12-30. "
    "Forecasts are historical-model estimates, not live 2026 demand."
)