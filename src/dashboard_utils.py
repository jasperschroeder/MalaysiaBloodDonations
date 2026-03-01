import datetime
import polars as pl
import requests


# API & Prediction Helpers
API_URL = "http://localhost:8001"
API_HEALTH_CHECK_RETRIES = 10
API_HEALTH_CHECK_DELAY = 1


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
            recent = daily_agg.filter(pl.col("date") < target_date).sort("date", descending=True).limit(1)
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
