import streamlit as st
import polars as pl
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import sys
import time
import requests
import datetime

import dashboard_utils


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from donations.setup_and_validation import download_data  # noqa

# Page configuration
st.set_page_config(
    page_title="Malaysia Blood Donations Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩸 Malaysia Blood Donations Dashboard")

# Constants
DATA_URL = "https://storage.data.gov.my/healthcare/blood_donations_state.parquet"
PKL_PATH = Path(__file__).parent.parent / "tmp" / "blood_donations.pkl"
PKL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Data Loading & Caching
@st.cache_resource
def load_or_download_data():
    """Download latest blood donation data and save to cache."""

    try:
        df = download_data(DATA_URL)
        # Save to pickle for future use
        with open(PKL_PATH, 'wb') as f:
            pickle.dump(df, f)
        return df
    except Exception as e:
        st.sidebar.error(f"Error downloading data: {e}")
        st.stop()


# Data Preparation
df = load_or_download_data()
df = df.with_columns(
    pl.col("date").cast(pl.Date)
)

# Get unique values for filters
states = sorted(df.select("state").unique().to_series().to_list())
blood_types = sorted(df.select("blood_type").unique().to_series().to_list())
date_min = df.select("date").min()[0, 0]
date_max = df.select("date").max()[0, 0]

# Sidebar Filters
st.sidebar.header("🔍 Filters")

selected_states = st.sidebar.multiselect(
    "Select States",
    options=states,
    help="Choose one or more states to display"
)

selected_blood_types = st.sidebar.multiselect(
    "Select Blood Types",
    options=blood_types,
    help="Choose blood types to display"
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
    help="Choose the date range for analysis"
)

# Validate date range
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_max

# Apply Filters
effective_blood_types = selected_blood_types if selected_blood_types else blood_types

filtered_df = df.filter(
    (pl.col("state").is_in(selected_states if selected_states else states)) &
    (pl.col("blood_type").is_in(effective_blood_types)) &
    (pl.col("date") >= start_date) &
    (pl.col("date") <= end_date)
)

if filtered_df.height == 0:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Key Metrics
st.sidebar.header("📊 Summary Statistics")
# If 'all' is present in selected blood types, use only 'all' for metrics
# to avoid double-counting against a/b/ab/o.
metrics_source_df = (
    filtered_df.filter(pl.col("blood_type") == "all")
    if "all" in effective_blood_types
    else filtered_df
)

daily_for_metrics = (
    metrics_source_df
    .group_by("date")
    .agg(pl.col("donations").sum().alias("total_donations"))
)
total_donations = daily_for_metrics.select("total_donations").sum()[0, 0]
avg_daily_donations = daily_for_metrics.select("total_donations").mean()[0, 0]

# Find the date with maximum daily donations
max_day = daily_for_metrics.sort("total_donations", descending=True).head(1)
if max_day.height > 0:
    peak_date = max_day.select("date")[0, 0]
    peak_donations = max_day.select("total_donations")[0, 0]
    peak_day_label = f"{peak_date.strftime('%b %d, %Y')} ({peak_donations:,.0f})"
else:
    peak_day_label = "N/A"

st.sidebar.metric("Total Donations", f"{total_donations:,.0f}")
st.sidebar.metric("Avg Daily", f"{avg_daily_donations:,.0f}")
st.sidebar.metric("Peak Day", peak_day_label)


@st.cache_resource
def check_api_running():
    """Check if the FastAPI server is running."""
    for _ in range(3):
        try:
            response = requests.get(f"{dashboard_utils.API_URL}/health", timeout=2)
            if response.status_code == 200:
                return True, "API is running"
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False, "FastAPI server is not running. Please start it with: python -m uvicorn src.donations.api:app --reload --host 0.0.0.0 --port 8001"  # noqa


@st.cache_resource
def ensure_api_running():
    """Check if the FastAPI server is running."""
    success, message = check_api_running()
    return success, message


# Main Dashboard Content

# Tab 1: Time Series Trends
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🗺️ State Comparison",
    "🔴 Blood Type Analysis",
    "📅 Patterns & Trends",
    "🔮 Predictions"
])

with tab1:
    st.subheader("Daily Donations Over Time")

    # Aggregation option: Week or Month
    aggregation = st.radio(
        "Aggregation Level",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
        help="Choose how to group the data"
    )

    # Aggregate by selected level
    if aggregation == "Daily":
        daily_agg = (
            filtered_df
            .group_by("date")
            .agg(pl.col("donations").sum().alias("total_donations"))
            .sort("date")
        )
        dates = daily_agg.select("date").to_series().to_list()
        donations = daily_agg.select("total_donations").to_series().to_list()
        x_label = "Date"
    elif aggregation == "Weekly":
        daily_agg = (
            filtered_df
            .with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
            .group_by("week_start")
            .agg(pl.col("donations").sum().alias("total_donations"))
            .sort("week_start")
        )
        dates = daily_agg.select("week_start").to_series().to_list()
        donations = daily_agg.select("total_donations").to_series().to_list()
        x_label = "Week Starting (Monday)"
    else:  # Monthly
        daily_agg = (
            filtered_df
            .with_columns(pl.col("date").dt.year().alias("year"))
            .with_columns(pl.col("date").dt.month().alias("month"))
            .group_by("year", "month")
            .agg(pl.col("donations").sum().alias("total_donations"), pl.col("date").min().alias("month_date"))
            .sort("year", "month")
        )
        dates = daily_agg.select("month_date").to_series().to_list()
        donations = daily_agg.select("total_donations").to_series().to_list()
        x_label = "Month"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, donations, linewidth=1.5, color="#e74c3c", alpha=0.8)
    ax.fill_between(dates, donations, alpha=0.2, color="#e74c3c")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Number of Donations", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Donations by State")

    # Aggregate by state
    state_agg = (
        filtered_df
        .group_by("state")
        .agg(pl.col("donations").sum().alias("total_donations"))
        .sort("total_donations", descending=True)
    )

    states_list = state_agg.select("state").to_series().to_list()
    state_totals = state_agg.select("total_donations").to_series().to_list()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(states_list, state_totals, color="#3498db")
    ax.set_xlabel("Total Donations", fontsize=11)
    ax.set_title("Total Donations by State", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='x')

    # Format x-axis to use comma notation instead of scientific notation
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add value labels on bars with padding
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 0.98, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}', ha='right', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Donations by Blood Type")

    # Aggregate by blood type (exclude 'all' as it's the sum of others)
    blood_agg = (
        filtered_df
        .filter(pl.col("blood_type") != "all")
        .group_by("blood_type")
        .agg(pl.col("donations").sum().alias("total_donations"))
        .sort("total_donations", descending=True)
    )

    blood_types_list = blood_agg.select("blood_type").to_series().to_list()
    blood_totals = blood_agg.select("total_donations").to_series().to_list()

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        wedges, texts, autotexts = ax.pie(
            blood_totals, labels=blood_types_list, autopct='%1.1f%%',
            colors=colors[:len(blood_types_list)], startangle=90
        )
        ax.set_title("Blood Type Distribution", fontsize=13, fontweight="bold")
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.dataframe(
            blood_agg.with_columns(
                percentage=(pl.col("total_donations") * 100 / pl.col("total_donations").sum()).round(2)
            ),
            use_container_width=True,
            hide_index=True
        )

with tab4:
    st.subheader("Average Donations By Day of the Week")

    # First aggregate to daily totals
    daily_total_dow = (
        filtered_df
        .group_by("date")
        .agg(pl.col("donations").sum().alias("daily_donations"))
    )

    # Add day of week to daily totals
    daily_total_dow = daily_total_dow.with_columns(
        day_of_week=pl.col("date").dt.weekday()
    )

    # Aggregate by day of week
    dow_agg = (
        daily_total_dow
        .group_by("day_of_week")
        .agg(pl.col("daily_donations").mean().alias("avg_donations"))
        .sort("day_of_week")
    )

    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    dow_values = dow_agg.select("day_of_week").to_series().to_list()
    dow_donations = dow_agg.select("avg_donations").to_series().to_list()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar([day_names[d % 7] for d in dow_values], dow_donations, color="#2ecc71")
    ax.set_ylabel("Average Daily Donations", fontsize=11)  # noqa
    ax.set_title("Average Donations by Day of Week", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Average donations by month
    st.subheader("Average Donations by Month")

    # First aggregate to daily totals, then average by month
    daily_total = (
        filtered_df
        .group_by("date")
        .agg(pl.col("donations").sum().alias("daily_donations"))
    )

    month_agg = (
        daily_total
        .with_columns(pl.col("date").dt.month().alias("month"))
        .group_by("month")
        .agg(pl.col("daily_donations").mean().alias("avg_donations"))
        .sort("month")
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_values = month_agg.select("month").to_series().to_list()
    month_donations = month_agg.select("avg_donations").to_series().to_list()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar([month_names[m-1] for m in month_values], month_donations, color="#e67e22")
    ax.set_ylabel("Average Daily Donations", fontsize=11)
    ax.set_title("Average Donations by Month", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # Average donations by week
    st.subheader("Average Donations by Week of Year")

    # First aggregate to daily totals, then average by week
    week_agg = (
        daily_total
        .with_columns(pl.col("date").dt.week().alias("week"))
        .group_by("week")
        .agg(pl.col("daily_donations").mean().alias("avg_donations"))
        .sort("week")
    )

    week_values = week_agg.select("week").to_series().to_list()
    week_donations = week_agg.select("avg_donations").to_series().to_list()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week_values, week_donations, linewidth=1.5, color="#9b59b6", marker='o', markersize=4, alpha=0.8)
    ax.fill_between(week_values, week_donations, alpha=0.2, color="#9b59b6")
    ax.set_xlabel("Week of Year", fontsize=11)
    ax.set_ylabel("Average Daily Donations", fontsize=11)
    ax.set_title("Average Donations by Week of Year", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    st.subheader("🔮 Predict Next Day Donations")

    # Check if API is running
    api_status, api_message = ensure_api_running()

    if not api_status:
        st.error(f"⚠️ {api_message}")
        st.stop()

    st.success("✅ API is online")

    # Prediction date selector
    st.subheader("Prediction Date")
    tomorrow = date_max + datetime.timedelta(days=1)

    # Restrict to tomorrow only
    st.info(f"📅 Predictions for tomorrow: **{tomorrow.strftime('%A, %B %d, %Y')}**")
    prediction_date = tomorrow

    # Infer lag values from full dataset
    inferred_lags = dashboard_utils.infer_lag_values(df, prediction_date)

    # Allow user to modify lag values
    st.subheader("Historical Donations (Last 7 Days)")

    with st.expander("View/Modify Lag Values", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        lag_values = {}
        day_offsets = [1, 2, 3, 4]

        for idx, day_offset in enumerate(day_offsets):
            with [col1, col2, col3, col4][idx]:
                lag_date = prediction_date - datetime.timedelta(days=day_offset)
                lag_values[f"lag{day_offset}"] = st.number_input(
                    f"Lag {day_offset} ({lag_date.strftime('%Y-%m-%d')})",
                    value=inferred_lags[f"lag{day_offset}"],
                    min_value=0,
                    max_value=10000,
                    step=100
                )

        col1, col2, col3, col4 = st.columns(4)
        day_offsets = [5, 6, 7]
        for idx, day_offset in enumerate(day_offsets):
            with [col1, col2, col3][idx]:
                lag_date = prediction_date - datetime.timedelta(days=day_offset)
                lag_values[f"lag{day_offset}"] = st.number_input(
                    f"Lag {day_offset} ({lag_date.strftime('%Y-%m-%d')})",
                    value=inferred_lags[f"lag{day_offset}"],
                    min_value=0,
                    max_value=10000,
                    step=100
                )

    # Display inferred values as summary if not expanded
    if st.checkbox("Show inferred values summary", value=False):
        summary_data = []
        for i in range(1, 8):
            lag_date = prediction_date - datetime.timedelta(days=i)
            summary_data.append({
                "Lag": i,
                "Date": lag_date.strftime('%Y-%m-%d'),
                "Donations": f"{inferred_lags[f'lag{i}']:,}"
            })
        st.dataframe(pl.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Holiday flags
    st.subheader("Holiday Flags")
    st.warning("⚠️ Only select ONE holiday type if applicable. Selecting multiple types may cause unexpected behavior.")

    col1, col2 = st.columns(2)

    with col1:
        high_donation_holiday = st.checkbox(
            "High Donation Holiday",
            value=False,
            help="Hari Malaysia, Hari Kebangsaan, Hari Pekerja, or Hari Wesak"
        )
        low_donation_holiday = st.checkbox(
            "Low Donation Holiday",
            value=False,
            help="Hari Peristiwa, Hari Raya Puasa or Hari Raya Qurban"
        )

    with col2:
        religion_or_culture_holiday = st.checkbox(
            "Religious/Cultural Holiday",
            value=False,
            help="Any religious or cultural holiday"
        )
        other_holiday = st.checkbox(
            "Other Holiday",
            value=False,
            help="Any other national public holiday"
        )

    # Validate holiday flags
    holiday_flags = [
        high_donation_holiday,
        low_donation_holiday,
        religion_or_culture_holiday,
        other_holiday
    ]
    if sum(holiday_flags) > 1:
        st.error("❌ Please select only ONE holiday type.")
        st.stop()

    # Prediction button
    st.subheader("Generate Prediction")

    if st.button("🔮 Predict Donations", type="primary", use_container_width=True):
        # Make sure lag_values exist (in case expander wasn't expanded)
        if 'lag_values' not in locals() or not lag_values:
            lag_values = inferred_lags

        # Prepare request
        request_data = {
            **lag_values,
            "nextday": prediction_date.strftime('%Y%m%d'),
            "high_donation_holiday": int(high_donation_holiday),
            "low_donation_holiday": int(low_donation_holiday),
            "religion_or_culture_holiday": int(religion_or_culture_holiday),
            "other_holiday": int(other_holiday)
        }

        # Call API
        with st.spinner("Making prediction..."):
            result = dashboard_utils.call_prediction_api(request_data)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            # Display prediction results
            st.success("✅ Prediction successful!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Donations",
                    f"{result['prediction']:,.0f}",
                    delta=None,
                    help="Predicted number of donations for the selected date"
                )

            with col2:
                avg_180day = dashboard_utils.calculate_180day_average(df, prediction_date, avg_daily_donations)
                diff = result['prediction'] - avg_180day
                pct_diff = (diff / avg_180day) * 100 if avg_180day > 0 else 0
                st.metric(
                    "vs. 180-Day Average",
                    f"{pct_diff:+.1f}%",
                    delta=f"{diff:+,.0f} donations",
                    help=f"Compared to 180-day average of {avg_180day:,.0f} donations/day"
                )

            # Show input features used
            with st.expander("View Input Features", expanded=False):
                st.write("**Lags (Last 7 Days):**")
                cols = st.columns(7)
                for i in range(1, 8):
                    with cols[i-1]:
                        st.metric(f"Lag {i}", f"{request_data[f'lag{i}']:,}")

                st.write("**Holiday Flags:**")
                holiday_info = {
                    "High Donation Holiday": request_data["high_donation_holiday"],
                    "Low Donation Holiday": request_data["low_donation_holiday"],
                    "Religious/Cultural Holiday": request_data["religion_or_culture_holiday"],
                    "Other Holiday": request_data["other_holiday"]
                }
                for key, value in holiday_info.items():
                    st.write(f"- {key}: {'✅' if value else '❌'}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Data Updated:** {date_max.strftime('%Y-%m-%d')}\n\n"
    f"**Data Points:** {filtered_df.height:,}"
)
