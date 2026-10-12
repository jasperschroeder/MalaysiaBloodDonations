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


def _get_weekly_observations(
    pair_data: pl.DataFrame, period_start: datetime.date
) -> tuple[int, int | None]:
    period_end = period_start + datetime.timedelta(days=7)
    period = pair_data.filter(
        (pl.col("date") >= period_start) & (pl.col("date") < period_end)
    )
    if period.height == 0:
        return 0, None
    return (
        period.get_column("date").n_unique(),
        int(period.get_column("donations").sum()),
    )


def _classify_weekly_change(
    observed_days: int,
    current_total: int | None,
    baseline_mean: float | None,
    min_change_pct: float,
) -> tuple[float | None, str]:
    if observed_days < 7:
        return None, "Incomplete current week"
    if baseline_mean is None:
        return None, "Insufficient complete baseline"
    if baseline_mean <= 0:
        return None, "Baseline unavailable"

    change_pct = (current_total - baseline_mean) / baseline_mean * 100
    if change_pct <= -min_change_pct:
        return change_pct, "Review: below baseline"
    if change_pct >= min_change_pct:
        return change_pct, "Review: above baseline"
    return change_pct, "Within threshold"


def _build_weekly_surveillance_row(
    component_data: pl.DataFrame,
    state: str,
    blood_type: str,
    week_start: datetime.date,
    min_change_pct: float,
    baseline_weeks: int,
) -> dict:
    pair_data = component_data.filter(
        (pl.col("state") == state) & (pl.col("blood_type") == blood_type)
    )
    observations = [
        _get_weekly_observations(
            pair_data,
            week_start - datetime.timedelta(days=7 * offset),
        )
        for offset in range(baseline_weeks, -1, -1)
    ]
    current_days, current_total = observations[-1]
    baseline_totals = [
        total for days, total in observations[:-1]
        if days == 7 and total is not None
    ]
    baseline_mean = (
        sum(baseline_totals) / baseline_weeks
        if len(baseline_totals) == baseline_weeks
        else None
    )
    change_pct, signal = _classify_weekly_change(
        current_days, current_total, baseline_mean, min_change_pct
    )

    return {
        "state": state,
        "blood_type": blood_type,
        "week_start": week_start,
        "observed_days": current_days,
        "coverage_pct": current_days / 7 * 100,
        "donations": current_total,
        "baseline_mean": baseline_mean,
        "change_pct": change_pct,
        "signal": signal,
    }


def build_weekly_surveillance_table(
    data: pl.DataFrame,
    week_start: datetime.date,
    min_change_pct: float = 20.0,
    baseline_weeks: int = 4,
) -> pl.DataFrame:
    """Compare complete state/blood-type weeks with the preceding complete weeks."""
    if week_start.weekday() != 0:
        raise ValueError("week_start must be a Monday")
    if min_change_pct < 0:
        raise ValueError("min_change_pct must be non-negative")
    if baseline_weeks < 1:
        raise ValueError("baseline_weeks must be at least 1")

    columns = {
        "state": pl.String,
        "blood_type": pl.String,
        "week_start": pl.Date,
        "observed_days": pl.Int64,
        "coverage_pct": pl.Float64,
        "donations": pl.Int64,
        "baseline_mean": pl.Float64,
        "change_pct": pl.Float64,
        "signal": pl.String,
    }
    component_data = (
        data
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("blood_type") != "all")
        .group_by(["state", "blood_type", "date"])
        .agg(pl.col("donations").sum().alias("donations"))
    )
    pairs = component_data.select("state", "blood_type").unique().sort(
        ["state", "blood_type"]
    )
    if pairs.height == 0:
        return pl.DataFrame(schema=columns)

    rows = [
        _build_weekly_surveillance_row(
            component_data,
            state,
            blood_type,
            week_start,
            min_change_pct,
            baseline_weeks,
        )
        for state, blood_type in pairs.iter_rows()
    ]
    return pl.DataFrame(rows, schema=columns)


def get_upcoming_holidays(
    holiday_calendar: pl.DataFrame,
    today: datetime.date,
    days_ahead: int = 90,
) -> pl.DataFrame:
    """Return national holidays in the requested future window."""
    if days_ahead < 0:
        raise ValueError("days_ahead must be non-negative")
    if holiday_calendar.height == 0:
        return pl.DataFrame(schema={
            "date": pl.Date,
            "name": pl.String,
            "estimated": pl.Boolean,
        })

    return (
        holiday_calendar
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(
            (pl.col("date") >= today)
            & (pl.col("date") <= today + datetime.timedelta(days=days_ahead))
        )
        .with_columns(
            pl.col("name").str.to_lowercase().str.contains("estimated").alias("estimated")
        )
        .sort("date")
    )


def build_holiday_impact_table(
    daily_donations: pl.DataFrame,
    holiday_calendar: pl.DataFrame,
    comparison_window_days: int = 28,
    min_comparison_days: int = 3,
) -> pl.DataFrame:
    """Compare holiday donations with nearby non-holidays on the same weekday."""
    if comparison_window_days < 7:
        raise ValueError("comparison_window_days must be at least 7")
    if min_comparison_days < 1:
        raise ValueError("min_comparison_days must be at least 1")

    schema = {
        "date": pl.Date,
        "name": pl.String,
        "estimated": pl.Boolean,
        "donations": pl.Int64,
        "baseline_mean": pl.Float64,
        "change_pct": pl.Float64,
        "comparison_days": pl.Int64,
    }
    daily = (
        daily_donations
        .with_columns(pl.col("date").cast(pl.Date))
        .group_by("date")
        .agg(pl.col("donations").sum().alias("donations"))
    )
    if daily.height == 0 or holiday_calendar.height == 0:
        return pl.DataFrame(schema=schema)

    donation_by_date = dict(daily.select("date", "donations").iter_rows())
    observed_dates = set(donation_by_date)
    holiday_rows = holiday_calendar.with_columns(
        pl.col("date").cast(pl.Date)
    ).sort("date").to_dicts()
    holiday_dates = {row["date"] for row in holiday_rows}
    rows = []

    for holiday in holiday_rows:
        holiday_date = holiday["date"]
        if holiday_date not in observed_dates:
            continue

        comparison_dates = [
            holiday_date + datetime.timedelta(days=offset)
            for offset in range(-comparison_window_days, comparison_window_days + 1)
            if offset != 0
            and (holiday_date + datetime.timedelta(days=offset)) in observed_dates
            and (holiday_date + datetime.timedelta(days=offset)) not in holiday_dates
            and (holiday_date + datetime.timedelta(days=offset)).weekday() == holiday_date.weekday()
        ]
        comparison_days = len(comparison_dates)
        baseline_mean = None
        change_pct = None

        if comparison_days >= min_comparison_days:
            baseline_mean = sum(donation_by_date[date] for date in comparison_dates) / comparison_days
            if baseline_mean > 0:
                change_pct = (
                    donation_by_date[holiday_date] - baseline_mean
                ) / baseline_mean * 100

        rows.append({
            "date": holiday_date,
            "name": holiday["name"],
            "estimated": "estimated" in holiday["name"].lower(),
            "donations": donation_by_date[holiday_date],
            "baseline_mean": baseline_mean,
            "change_pct": change_pct,
            "comparison_days": comparison_days,
        })

    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def calculate_upcoming_festive_risk(
    upcoming_holidays: pl.DataFrame,
    holiday_impact: pl.DataFrame,
    recent_daily_baseline: float,
    high_risk_threshold_pct: float = -20.0,
    moderate_risk_threshold_pct: float = -5.0,
) -> pl.DataFrame:
    """
    Project expected festive impact and shortage risks for upcoming holidays.

    Matches each upcoming holiday by name (normalized) against observed median
    change percentages from historical holiday occurrences. Projects the expected
    daily donation volume, projected unit change (deficit/surplus), and risk tier.

    Risk tiers:
    - 🔴 High Risk: Historical median drop >= 20% (change_pct <= -20%)
    - 🟡 Moderate Risk: Historical median drop between 5% and 20% (-20% < change_pct <= -5%)
    - 🟢 Low / Stable Risk: Small drop or surge (change_pct > -5%)
    - ⚪ Insufficient History: No reliable historical observations for this holiday name
    """
    schema = {
        "date": pl.Date,
        "name": pl.String,
        "days_away": pl.Int64,
        "historical_occurrences": pl.Int64,
        "median_change_pct": pl.Float64,
        "baseline_daily_donations": pl.Float64,
        "projected_donations": pl.Float64,
        "projected_unit_diff": pl.Float64,
        "risk_level": pl.String,
    }

    if upcoming_holidays.height == 0:
        return pl.DataFrame(schema=schema)

    # Calculate median historical change per holiday name from usable impact records
    usable = holiday_impact.filter(pl.col("change_pct").is_not_null())
    impact_stats = {}
    if usable.height > 0:
        stats_df = (
            usable
            .group_by("name")
            .agg(
                pl.len().alias("count"),
                pl.col("change_pct").median().alias("median_change_pct"),
            )
        )
        for row in stats_df.iter_rows(named=True):
            impact_stats[row["name"].strip().lower()] = (
                row["count"],
                float(row["median_change_pct"]),
            )

    today = datetime.date.today()
    rows = []
    for hol in upcoming_holidays.to_dicts():
        hol_date = hol["date"]
        hol_name = hol["name"]
        days_away = (hol_date - today).days

        # Match holiday name (case-insensitive substring match or exact match)
        hol_clean = hol_name.strip().lower()
        matched_stat = impact_stats.get(hol_clean)
        if not matched_stat:
            # Fallback substring match
            for k, v in impact_stats.items():
                if k in hol_clean or hol_clean in k:
                    matched_stat = v
                    break

        if matched_stat:
            count, median_pct = matched_stat
            projected = max(0.0, recent_daily_baseline * (1.0 + (median_pct / 100.0)))
            unit_diff = projected - recent_daily_baseline

            if median_pct <= high_risk_threshold_pct:
                risk_level = "🔴 High Shortage Risk"
            elif median_pct <= moderate_risk_threshold_pct:
                risk_level = "🟡 Moderate Shortage Risk"
            else:
                risk_level = "🟢 Low Risk / Stable"
        else:
            count = 0
            median_pct = None
            projected = None
            unit_diff = None
            risk_level = "⚪ Insufficient History"

        rows.append({
            "date": hol_date,
            "name": hol_name,
            "days_away": days_away,
            "historical_occurrences": count,
            "median_change_pct": median_pct,
            "baseline_daily_donations": round(recent_daily_baseline, 1),
            "projected_donations": round(projected, 1) if projected is not None else None,
            "projected_unit_diff": round(unit_diff, 1) if unit_diff is not None else None,
            "risk_level": risk_level,
        })

    return pl.DataFrame(rows, schema=schema)


def simulate_multi_day_donations(
    start_date: datetime.date,
    days: int,
    recent_lags: dict[str, int],
    holiday_calendar: pl.DataFrame,
    day_of_week_factors: dict[int, float] | None = None,
    recent_baseline_daily: float = 1000.0,
) -> pl.DataFrame:
    """
    Simulate expected multi-day donation trajectory over a planning horizon (e.g. 7-14 days).

    Combines day-of-week seasonality, upcoming holiday flags, and autoregressive lag decay
    to project daily and cumulative collections against baseline requirements.

    Returns a pl.DataFrame with:
      - date, day_name, is_holiday, holiday_name, day_factor
      - projected_donations, baseline_donations, daily_deficit, cumulative_deficit
    """
    if days < 1:
        raise ValueError("Simulation horizon days must be at least 1")

    # Default day of week seasonality if not provided (Monday=0 ... Sunday=6)
    # Weekends and Fridays often have different collection profiles in Malaysia
    if not day_of_week_factors:
        day_of_week_factors = {0: 1.0, 1: 0.95, 2: 0.95, 3: 0.95, 4: 1.05, 5: 1.15, 6: 1.10}

    holidays_by_date = {}
    if holiday_calendar.height > 0:
        for r in holiday_calendar.with_columns(pl.col("date").cast(pl.Date)).iter_rows(named=True):
            holidays_by_date[r["date"]] = r["name"]

    # Recent 7 lag values
    lag_series = [recent_lags.get(f"lag{i}", int(recent_baseline_daily)) for i in range(1, 8)]
    current_moving_avg = sum(lag_series) / len(lag_series) if lag_series else recent_baseline_daily

    rows = []
    cum_projected = 0.0
    cum_baseline = 0.0

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for step in range(days):
        current_date = start_date + datetime.timedelta(days=step)
        weekday = current_date.weekday()
        day_name = day_names[weekday]
        dow_factor = day_of_week_factors.get(weekday, 1.0)

        hol_name = holidays_by_date.get(current_date)
        is_holiday = hol_name is not None

        # Base expectation: combination of moving average and baseline
        base_exp = 0.6 * current_moving_avg + 0.4 * recent_baseline_daily

        # Apply holiday shock if applicable
        holiday_factor = 1.0
        if is_holiday:
            hol_lower = hol_name.lower()
            if any(term in hol_lower for term in ["raya", "peristiwa", "qurban", "aidilfitri", "aidiladha"]):
                holiday_factor = 0.65  # Major 35% festive drop
            elif any(term in hol_lower for term in ["merdeka", "kebangsaan", "malaysia", "pekerja", "wesak"]):
                holiday_factor = 1.10  # Civic blood drive boost
            else:
                holiday_factor = 0.85  # General holiday drop

        daily_proj = base_exp * dow_factor * holiday_factor
        daily_base = recent_baseline_daily * dow_factor

        cum_projected += daily_proj
        cum_baseline += daily_base
        cum_deficit = cum_projected - cum_baseline

        rows.append({
            "date": current_date,
            "day_name": day_name,
            "is_holiday": is_holiday,
            "holiday_name": hol_name or "-",
            "projected_donations": round(daily_proj, 1),
            "baseline_donations": round(daily_base, 1),
            "daily_net": round(daily_proj - daily_base, 1),
            "cumulative_projected": round(cum_projected, 1),
            "cumulative_baseline": round(cum_baseline, 1),
            "cumulative_net": round(cum_deficit, 1),
        })

        # Autoregressive update of moving average
        current_moving_avg = 0.7 * current_moving_avg + 0.3 * daily_proj

    return pl.DataFrame(rows)


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
