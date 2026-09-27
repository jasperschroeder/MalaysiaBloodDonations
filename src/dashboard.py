import streamlit as st
import polars as pl
import pickle
import plotly.express as px
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

# Custom styling for a more distinctive, "hero banner" look and consistent
# Plotly-friendly metric cards, on top of the base red/white Streamlit theme.
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #ec7063 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(192, 57, 43, 0.25);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    .hero-banner p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #eceef1;
        border-radius: 10px;
        padding: 0.75rem 1rem 0.5rem 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    </style>
    <div class="hero-banner">
        <h1>🩸 Malaysia Blood Donations Dashboard</h1>
        <p>Explore national donation trends, compare states, and forecast tomorrow's turnout.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Constants
DATA_URL = "https://storage.data.gov.my/healthcare/blood_donations_state.parquet"
PKL_PATH = Path(__file__).parent.parent / "tmp" / "blood_donations.pkl"
PKL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Data Loading & Caching
@st.cache_resource
def load_or_download_data(force_refresh: bool = False):
    """
    Load blood donation data from the local pickle cache when available, downloading
    a fresh copy from data.gov.my only on first run or when `force_refresh` is True.
    """
    if not force_refresh and PKL_PATH.exists():
        try:
            with open(PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            st.sidebar.warning(f"Cached data was unreadable ({e}); re-downloading.")

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

if st.sidebar.button("🔄 Refresh Data", use_container_width=True, help="Re-download the latest data from data.gov.my"):
    load_or_download_data.clear()
    df = load_or_download_data(force_refresh=True).with_columns(pl.col("date").cast(pl.Date))
    st.rerun()

selected_states = st.sidebar.multiselect(
    "Select States",
    options=states,
    help="Choose one or more states to display"
)

blood_type_order = ["all", "a", "b", "ab", "o"]
blood_type_options = sorted(
    blood_types,
    key=lambda bt: blood_type_order.index(bt) if bt in blood_type_order else len(blood_type_order)
)
selected_blood_types = st.sidebar.multiselect(
    "Select Blood Types",
    options=blood_type_options,
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

# Period-over-period comparison (uses full date history, filtered only by
# state/blood-type, so the "previous period" isn't clipped by the date filter).
state_filtered_df = df.filter(pl.col("state").is_in(selected_states if selected_states else states))
period_comparison_source_df = (
    state_filtered_df.filter(pl.col("blood_type") == "all")
    if "all" in effective_blood_types
    else state_filtered_df.filter(pl.col("blood_type").is_in(effective_blood_types))
)
period_delta = dashboard_utils.compute_period_over_period_delta(
    period_comparison_source_df, start_date, end_date
)

# Headline KPI row in the main content area
st.subheader("📊 Overview")
state_totals = (
    metrics_source_df
    .group_by("state")
    .agg(pl.col("donations").sum().alias("total_donations"))
    .sort("total_donations", descending=True)
)
leading_state = state_totals.row(0, named=True)
export_df = (
    filtered_df.filter(pl.col("blood_type") == "all")
    if "all" in effective_blood_types
    else filtered_df
)
state_scope = f"{len(selected_states) if selected_states else len(states)} states"
blood_type_scope = (
    "all blood types"
    if "all" in effective_blood_types
    else ", ".join(blood_type.upper() for blood_type in effective_blood_types)
)
scope_columns = st.columns([3, 1])
scope_columns[0].caption(
    f"{start_date:%b %d, %Y} to {end_date:%b %d, %Y} · "
    f"{state_scope} · {blood_type_scope} · "
    f"Latest observation: {date_max:%b %d, %Y}"
)
scope_columns[1].download_button(
    "Download filtered CSV",
    data=export_df.write_csv().encode("utf-8"),
    file_name=f"malaysia_blood_donations_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
    mime="text/csv",
    use_container_width=True,
    help="Export the rows matching the current state, blood type, and date filters",
)
st.caption(
    f"Leading state in this selection: **{leading_state['state']}** "
    f"({leading_state['total_donations']:,.0f} donations)"
)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Donations", f"{total_donations:,.0f}")
kpi2.metric("Avg Daily Donations", f"{avg_daily_donations:,.0f}")
kpi3.metric("Peak Day", peak_day_label)
if period_delta["pct_change"] is not None:
    kpi4.metric(
        "vs. Previous Period",
        f"{period_delta['pct_change']:+.1f}%",
        delta=f"{period_delta['absolute_change']:+,.0f} donations",
        help="Compared to the immediately preceding period of equal length",
    )
else:
    kpi4.metric("vs. Previous Period", "N/A", help="Not enough historical data before the selected range")


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

PLOTLY_TEMPLATE = "plotly_white"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Time Series",
    "🗺️ State Comparison",
    "📊 State Trends",
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

    # Use only blood_type='all' if selected, else use filtered data to avoid double counting
    chart_df = (
        filtered_df.filter(pl.col("blood_type") == "all")
        if "all" in effective_blood_types
        else filtered_df
    )

    # Aggregate by selected level
    if aggregation == "Daily":
        daily_agg = (
            chart_df
            .group_by("date")
            .agg(pl.col("donations").sum().alias("total_donations"))
            .sort("date")
        )
        x_col, x_label = "date", "Date"
    elif aggregation == "Weekly":
        daily_agg = (
            chart_df
            .with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
            .group_by("week_start")
            .agg(pl.col("donations").sum().alias("total_donations"))
            .sort("week_start")
        )
        x_col, x_label = "week_start", "Week Starting (Monday)"
    else:  # Monthly
        daily_agg = (
            chart_df
            .with_columns(pl.col("date").dt.year().alias("year"))
            .with_columns(pl.col("date").dt.month().alias("month"))
            .group_by("year", "month")
            .agg(pl.col("donations").sum().alias("total_donations"), pl.col("date").min().alias("month_date"))
            .sort("year", "month")
        )
        x_col, x_label = "month_date", "Month"

    fig = px.area(
        daily_agg.to_pandas(),
        x=x_col,
        y="total_donations",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_traces(
        hovertemplate=f"{x_label}: %{{x}}<br>Donations: %{{y:,.0f}}<extra></extra>",
        line_width=2,
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Number of Donations",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Donations by State")

    # Use only blood_type='all' if selected, else use filtered data to avoid double counting
    chart_df = (
        filtered_df.filter(pl.col("blood_type") == "all")
        if "all" in effective_blood_types
        else filtered_df
    )

    # Aggregate by state
    state_agg = (
        chart_df
        .group_by("state")
        .agg(pl.col("donations").sum().alias("total_donations"))
        .sort("total_donations")
    )

    fig = px.bar(
        state_agg.to_pandas(),
        x="total_donations",
        y="state",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color="total_donations",
        color_continuous_scale="Reds",
        text="total_donations",
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate="State: %{y}<br>Donations: %{x:,.0f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Total Donations",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=max(400, 28 * state_agg.height),
    )
    st.plotly_chart(fig, use_container_width=True)

    heatmap_types = [
        blood_type
        for blood_type in (selected_blood_types or blood_types)
        if blood_type != "all"
    ]
    if heatmap_types:
        heatmap_df = (
            filtered_df
            .filter(pl.col("blood_type").is_in(heatmap_types))
            .group_by(["state", "blood_type"])
            .agg(pl.col("donations").sum().alias("total_donations"))
        )
        st.subheader("Blood Type Mix by State")
        st.caption(
            "Component blood groups only; the overlapping all-types total is excluded."
        )
        heatmap = px.density_heatmap(
            heatmap_df.to_pandas(),
            x="blood_type",
            y="state",
            z="total_donations",
            histfunc="sum",
            category_orders={"blood_type": heatmap_types},
            color_continuous_scale="Reds",
            template=PLOTLY_TEMPLATE,
        )
        heatmap.update_traces(
            hovertemplate=(
                "State: %{y}<br>Blood type: %{x}<br>Donations: %{z:,.0f}<extra></extra>"
            )
        )
        heatmap.update_layout(
            xaxis_title="Blood Type",
            yaxis_title="",
            coloraxis_colorbar_title="Donations",
            margin=dict(l=10, r=10, t=30, b=10),
            height=max(400, 28 * heatmap_df.get_column("state").n_unique()),
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Select one or more component blood types to view the state heatmap.")

with tab3:
    st.subheader("Historical Trends by State")
    st.caption(
        "Compare how selected states trend over time, with an optional rolling average "
        "to smooth day-to-day noise."
    )

    # Use only blood_type='all' if selected, else use filtered data to avoid double counting
    chart_df = (
        filtered_df.filter(pl.col("blood_type") == "all")
        if "all" in effective_blood_types
        else filtered_df
    )

    trend_states = st.multiselect(
        "States to plot",
        options=states,
        default=(selected_states if selected_states else states)[: min(5, len(states))],
        help="Choose which states to overlay on the trend chart",
    )
    rolling_window = st.radio(
        "Rolling average window (days)",
        options=[1, 7, 14, 30],
        index=1,
        format_func=lambda days: "1 (raw)" if days == 1 else str(days),
        horizontal=True,
        help="1 = raw daily values, higher values smooth out noise",
    )

    if not trend_states:
        st.info("Select at least one state to view its trend.")
    else:
        state_daily = (
            chart_df
            .filter(pl.col("state").is_in(trend_states))
            .group_by(["state", "date"])
            .agg(pl.col("donations").sum().alias("donations"))
        )
        state_daily = dashboard_utils.add_rolling_average(
            state_daily, value_col="donations", window=rolling_window, group_col="state"
        )

        fig = px.line(
            state_daily.sort(["state", "date"]).to_pandas(),
            x="date",
            y="donations_rolling_avg",
            color="state",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(
            hovertemplate="%{fullData.name}<br>%{x}<br>Donations: %{y:,.0f}<extra></extra>",
            line_width=2,
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Donations ({rolling_window}-day avg)" if rolling_window > 1 else "Donations",
            hovermode="x unified",
            legend_title="State",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Year-over-year style comparison: total donations per state, per year
        st.subheader("Yearly Totals by State")
        yearly_state = (
            chart_df
            .filter(pl.col("state").is_in(trend_states))
            .with_columns(pl.col("date").dt.year().alias("year"))
            .group_by(["state", "year"])
            .agg(pl.col("donations").sum().alias("total_donations"))
            .sort(["year", "state"])
        )
        fig_year = px.bar(
            yearly_state.to_pandas(),
            x="year",
            y="total_donations",
            color="state",
            barmode="group",
            template=PLOTLY_TEMPLATE,
        )
        fig_year.update_traces(hovertemplate="%{fullData.name}<br>Year: %{x}<br>Donations: %{y:,.0f}<extra></extra>")
        fig_year.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Donations",
            legend_title="State",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_year, use_container_width=True)

with tab4:
    st.subheader("Donations by Blood Type")

    # Aggregate by blood type (exclude 'all' as it's the sum of others)
    blood_agg = (
        filtered_df
        .filter(pl.col("blood_type") != "all")
        .group_by("blood_type")
        .agg(pl.col("donations").sum().alias("total_donations"))
        .sort("total_donations", descending=True)
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.pie(
            blood_agg.to_pandas(),
            names="blood_type",
            values="total_donations",
            hole=0.4,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
        )
        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="%{label}<br>Donations: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            blood_agg.with_columns(
                percentage=(pl.col("total_donations") * 100 / pl.col("total_donations").sum()).round(2)
            ),
            use_container_width=True,
            hide_index=True
        )

with tab5:
    st.subheader("Average Donations By Day of the Week")

    # Use only blood_type='all' if selected, else use filtered data to avoid double counting
    chart_df = (
        filtered_df.filter(pl.col("blood_type") == "all")
        if "all" in effective_blood_types
        else filtered_df
    )

    # First aggregate to daily totals
    daily_total_dow = (
        chart_df
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
    dow_agg = dow_agg.with_columns(
        pl.col("day_of_week").map_elements(lambda d: day_names[d % 7], return_dtype=pl.Utf8).alias("day_name")
    )

    fig = px.bar(
        dow_agg.to_pandas(),
        x="day_name",
        y="avg_donations",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#2ecc71"],
        text_auto=",.0f",
    )
    fig.update_traces(hovertemplate="%{x}<br>Avg Donations: %{y:,.0f}<extra></extra>")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Average Daily Donations",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Average donations by month
    st.subheader("Average Donations by Month")

    # First aggregate to daily totals, then average by month
    daily_total = (
        chart_df
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
    month_agg = month_agg.with_columns(
        pl.col("month").map_elements(lambda m: month_names[m - 1], return_dtype=pl.Utf8).alias("month_name")
    )

    fig = px.bar(
        month_agg.to_pandas(),
        x="month_name",
        y="avg_donations",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#e67e22"],
        text_auto=",.0f",
    )
    fig.update_traces(hovertemplate="%{x}<br>Avg Donations: %{y:,.0f}<extra></extra>")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Average Daily Donations",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

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

    fig = px.area(
        week_agg.to_pandas(),
        x="week",
        y="avg_donations",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#9b59b6"],
        markers=True,
    )
    fig.update_traces(hovertemplate="Week %{x}<br>Avg Donations: %{y:,.0f}<extra></extra>")
    fig.update_layout(
        xaxis_title="Week of Year",
        yaxis_title="Average Daily Donations",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab6:
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
