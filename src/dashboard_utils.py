import datetime
import os
import polars as pl
import requests


# API & Prediction Helpers
API_URL = os.getenv("API_URL", "http://localhost:8001")
API_HEALTH_CHECK_RETRIES = 10
API_HEALTH_CHECK_DELAY = 1


def aggregate_state_blood_type_donations(
    data: pl.DataFrame, blood_types: list[str]
) -> pl.DataFrame:
    """Aggregate component blood types by state without the overlapping `all` rows."""
    component_types = [blood_type for blood_type in blood_types if blood_type != "all"]
    return (
        data
        .filter(pl.col("blood_type").is_in(component_types))
        .group_by(["state", "blood_type"])
        .agg(pl.col("donations").sum().alias("total_donations"))
    )


def infer_lag_values(data: pl.DataFrame, prediction_date: datetime.date) -> dict:
    """
    Infer the last 7 days of donations (lags) from the dataset.
    Uses all data regardless of dashboard filters for more representative values.
    """
    # Load full dataset for lag inference (not filtered)
    # Use only blood_type == 'all' to avoid double counting blood groups.
    full_df = (
        data
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("blood_type") == "all")
    )

    # Aggregate by date to get daily totals
    daily_agg = (
        full_df
        .group_by("date")
        .agg(pl.col("donations").sum().alias("total_donations"))
        .sort("date")
    )

    lags = {}
    for i in range(1, 8):
        target_date = prediction_date - datetime.timedelta(days=i)
        row = daily_agg.filter(pl.col("date") == target_date).select("total_donations")

        if row.height > 0:
            lags[f"lag{i}"] = int(row[0, 0])
        else:
            # Use the most recent available value before target_date
            recent = (
                daily_agg
                .filter(pl.col("date") < target_date)
                .sort("date", descending=True)
                .limit(1)
                .select("total_donations")
            )
            if recent.height > 0:
                lags[f"lag{i}"] = int(recent[0, 0])
            else:
                raise ValueError(f"No historical data available before {target_date}")

    return lags


def call_prediction_api(request_data: dict) -> dict:
    """Call the prediction API endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=request_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def compute_period_over_period_delta(
    state_and_blood_type_filtered_df: pl.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict:
    """
    Compare total donations in the selected [start_date, end_date] window against the
    immediately preceding window of equal length, using a dataframe already filtered by
    state/blood-type selections (but not yet by date).

    Returns a dict with current_total, previous_total, absolute_change and pct_change.
    `previous_total`/`pct_change` are None when there is no prior data to compare against.
    """
    window_length = (end_date - start_date).days + 1
    previous_end = start_date - datetime.timedelta(days=1)
    previous_start = previous_end - datetime.timedelta(days=window_length - 1)

    current_total = (
        state_and_blood_type_filtered_df
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        .select("donations")
        .sum()[0, 0]
    ) or 0

    previous_df = state_and_blood_type_filtered_df.filter(
        (pl.col("date") >= previous_start) & (pl.col("date") <= previous_end)
    )

    if previous_df.height == 0:
        return {
            "current_total": current_total,
            "previous_total": None,
            "absolute_change": None,
            "pct_change": None,
        }

    previous_total = previous_df.select("donations").sum()[0, 0] or 0
    absolute_change = current_total - previous_total
    pct_change = (absolute_change / previous_total * 100) if previous_total > 0 else None

    return {
        "current_total": current_total,
        "previous_total": previous_total,
        "absolute_change": absolute_change,
        "pct_change": pct_change,
    }


def add_rolling_average(
    df: pl.DataFrame, value_col: str, window: int, group_col: str | None = None
) -> pl.DataFrame:
    """
    Add a rolling average column (`{value_col}_rolling_avg`) computed over `window` periods,
    sorted by date. When `group_col` is provided, the rolling average is computed per group
    (e.g. per state) instead of across the whole dataset.
    """
    sort_cols = [group_col, "date"] if group_col else ["date"]
    rolling_expr = pl.col(value_col).rolling_mean(window_size=window, min_samples=1)
    if group_col:
        rolling_expr = rolling_expr.over(group_col)

    return (
        df
        .sort(sort_cols)
        .with_columns(rolling_expr.alias(f"{value_col}_rolling_avg"))
    )


def calculate_180day_average(
    df: pl.DataFrame, prediction_date: datetime.date, avg_daily_donations: float
) -> float:
    """
    Calculate average daily donations over the past 180 days before the prediction date.
    """
    cutoff_date = prediction_date - datetime.timedelta(days=180)
    recent_data = (
        df
        .filter(
            (pl.col("blood_type") == "all") &
            (pl.col("date") >= cutoff_date) &
            (pl.col("date") < prediction_date)
        )
        .group_by("date")
        .agg(pl.col("donations").sum().alias("total_donations"))
    )

    if recent_data.height > 0:
        return float(recent_data.select("total_donations").mean()[0, 0])
    else:
        return avg_daily_donations  # Fallback to overall average if no data
