"""Reusable data preparation for blood-donation forecasting.

Extracted from the exploratory notebooks so that training, evaluation,
and (later) inference share exactly one feature-building implementation.

Pipeline:
    raw long-format frame -> pivot blood types -> daily aggregation
    (nationwide or per state) -> validation -> holiday flags ->
    calendar features -> lag features -> model-ready frame.
"""

import datetime

import holidays
import numpy as np
import polars as pl

BLOOD_TYPE_COLUMNS = ["a", "b", "ab", "o"]
TARGET_COLUMN = "all"

CALENDAR_FEATURE_COLUMNS = ["weekday", "month", "day_of_year"]

# Canonical order matches the API request contract:
# [weekday, month, day_of_year, high, low, religion_or_culture, other]
HOLIDAY_FLAG_COLUMNS = [
    "is_high_donation_holiday",
    "is_low_donation_holiday",
    "is_religion_or_culture_holiday",
    "is_other_holiday",
]

FEATURE_COLUMNS = CALENDAR_FEATURE_COLUMNS + HOLIDAY_FLAG_COLUMNS

EXTENDED_CALENDAR_COLUMNS = [
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "year", "trend_days",
]

# --- Holiday definitions (from the EDA notebook) ---

EXCLUDED_HOLIDAY_NAMES = [
    "Cuti tambahan sempena memperingati SAT 2017",
]

HOLIDAY_NAME_CLEANUP_PATTERNS = [
    r"\s*\(Hari Kedua\)",
    r"\s*\(pergantian hari\)",
    r"\s*\(pilihan raya umum\)",
    r"\s+ke-\d+$",  # e.g. "Hari Pertabalan Yang di-Pertuan Agong ke-15"
]

HOLIDAY_CATEGORIES = {
    "is_religion_or_culture_holiday": [
        "Awal Muharam",
        "Hari Keputeraan Nabi Muhammad S.A.W.",
        "Hari Krismas",
        "Tahun Baharu Cina",
        "Hari Raya Puasa",
        "Hari Raya Qurban",
        "Hari Wesak",
    ],
    "is_other_holiday": [
        "Hari Pekerja",
        "Hari Keputeraan Rasmi Seri Paduka Baginda Yang di-Pertuan Agong",
        "Hari Pertabalan Yang di-Pertuan Agong",
        "Peristiwa",
        "Hari Malaysia",
        "Hari Kebangsaan",
    ],
    "is_low_donation_holiday": [
        "Hari Raya Puasa",
        "Hari Raya Qurban",
        "Peristiwa",
    ],
    "is_high_donation_holiday": [
        "Hari Pekerja",
        "Hari Wesak",
        "Hari Malaysia",
        "Hari Kebangsaan",
    ],
}

# name -> set of flag column names
_HOLIDAY_NAME_TO_FLAGS = {
    name: {flag for flag, names in HOLIDAY_CATEGORIES.items() if name in names}
    for names in HOLIDAY_CATEGORIES.values()
    for name in names
}


def _clean_holiday_name(name: str) -> str:
    """Apply the same name normalisation as the EDA notebook."""
    import re

    cleaned = name.replace("Cuti ", "", 1) if name.startswith("Cuti ") else name
    for pattern in HOLIDAY_NAME_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def categorize_holiday_name(name: str) -> set:
    """Map a (possibly combined) holiday name to its flag columns.

    Handles multi-holiday days such as "Hari Pekerja; Hari Wesak" by
    splitting on ';' and unioning the categories of each part.
    """
    flags = set()
    for part in str(name).split(";"):
        flags |= _HOLIDAY_NAME_TO_FLAGS.get(_clean_holiday_name(part), set())
    return flags


def get_malaysia_holidays(start_year: int, end_year: int) -> pl.DataFrame:
    """Return a (date, name) frame of Malaysian public holidays."""
    holiday_rows = [
        {"date": date, "name": name}
        for date, name in holidays.country_holidays(
            "MY", years=range(start_year, end_year + 1)
        ).items()
    ]
    df = pl.DataFrame(holiday_rows)
    if df.height == 0:
        return df
    return df.filter(~pl.col("name").is_in(EXCLUDED_HOLIDAY_NAMES))


def pivot_blood_types(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot long-format data to one row per (date, state).

    Recalculates the aggregate 'all' column as the sum of the four blood
    types to correct the small source-data mismatches found during EDA.
    """
    if "blood_type" not in df.columns:
        return df  # already pivoted

    pivoted = df.pivot(
        values="donations",
        index=["date", "state"],
        on="blood_type",
    ).drop(TARGET_COLUMN, strict=False)

    return pivoted.with_columns(
        pl.sum_horizontal(BLOOD_TYPE_COLUMNS).alias(TARGET_COLUMN)
    )


def aggregate_daily_donations(df: pl.DataFrame, state: str = None) -> pl.DataFrame:
    """Aggregate a pivoted frame to one row per date.

    Args:
        df: Pivoted frame with columns date, state, a, b, ab, o, all.
        state: If given, restrict to a single state; otherwise nationwide.
    """
    if state is not None:
        df = df.filter(pl.col("state") == state)
        if df.height == 0:
            raise ValueError(f"No rows found for state '{state}'.")

    return (
        df.group_by("date")
        .agg([pl.col(c).sum() for c in BLOOD_TYPE_COLUMNS + [TARGET_COLUMN]])
        .sort("date")
    )


def validate_daily_frame(
    df: pl.DataFrame,
    date_col: str = "date",
    target_col: str = TARGET_COLUMN,
    allow_missing_dates: bool = False,
) -> dict:
    """Validate a daily aggregated frame before feature engineering.

    Raises ValueError on hard failures (duplicates, negatives, nulls).
    Returns a report dict, including any missing dates in the range.
    """
    for col in (date_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    if df.select(pl.col(date_col).is_null().any()).item():
        raise ValueError(f"Column '{date_col}' contains nulls.")

    n_duplicates = df.select(pl.col(date_col).is_duplicated().sum()).item()
    if n_duplicates:
        raise ValueError(f"{n_duplicates} duplicated dates detected.")

    if df.select((pl.col(target_col) < 0).any()).item():
        raise ValueError(f"Column '{target_col}' contains negative donations.")

    df = df.sort(date_col)
    dates = df[date_col].to_list()
    full_range = pl.date_range(dates[0], dates[-1], interval="1d", eager=True)
    missing = sorted(set(full_range.to_list()) - set(dates))

    if missing and not allow_missing_dates:
        preview = ", ".join(str(d) for d in missing[:5])
        raise ValueError(
            f"{len(missing)} missing dates in the series (e.g. {preview})."
        )

    return {
        "n_rows": df.height,
        "start_date": dates[0],
        "end_date": dates[-1],
        "missing_dates": missing,
    }


def add_holiday_flags(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Add the four holiday flag columns to a daily frame."""
    dates = df[date_col].to_list()
    years = range(dates[0].year, dates[-1].year + 1)
    holiday_map = {}
    for row in get_malaysia_holidays(years.start, years.stop - 1).iter_rows(named=True):
        holiday_map.setdefault(row["date"], set()).update(
            categorize_holiday_name(row["name"])
        )

    flag_values = {
        flag: [int(flag in holiday_map.get(d, set())) for d in dates]
        for flag in HOLIDAY_FLAG_COLUMNS
    }
    return df.with_columns(
        [pl.Series(flag, values, dtype=pl.Int64) for flag, values in flag_values.items()]
    )


def add_calendar_features(
    df: pl.DataFrame, date_col: str = "date", extended: bool = False
) -> pl.DataFrame:
    """Add calendar features. `extended=True` adds cyclical and trend terms."""
    df = df.with_columns(
        pl.col(date_col).dt.weekday().alias("weekday"),
        pl.col(date_col).dt.month().alias("month"),
        pl.col(date_col).dt.ordinal_day().alias("day_of_year"),
    )

    if not extended:
        return df

    min_date = df.select(pl.col(date_col).min()).item()
    return df.with_columns(
        (2 * np.pi * pl.col("weekday") / 7).sin().alias("weekday_sin"),
        (2 * np.pi * pl.col("weekday") / 7).cos().alias("weekday_cos"),
        (2 * np.pi * (pl.col("month") - 1) / 12).sin().alias("month_sin"),
        (2 * np.pi * (pl.col("month") - 1) / 12).cos().alias("month_cos"),
        (2 * np.pi * (pl.col("day_of_year") - 1) / 365.25).sin().alias("day_of_year_sin"),
        (2 * np.pi * (pl.col("day_of_year") - 1) / 365.25).cos().alias("day_of_year_cos"),
        pl.col(date_col).dt.year().alias("year"),
        (pl.col(date_col) - pl.lit(min_date)).dt.total_days().alias("trend_days"),
    )


def add_lag_features(
    df: pl.DataFrame,
    target_col: str = TARGET_COLUMN,
    window_size: int = 7,
    drop_nulls: bool = True,
) -> pl.DataFrame:
    """Add lag columns target_col_lag_1 .. target_col_lag_N."""
    lag_col = f"{target_col}_lag"
    df = df.with_columns(
        [pl.col(target_col).shift(lag).alias(f"{lag_col}_{lag}") for lag in range(1, window_size + 1)]
    )
    return df.drop_nulls() if drop_nulls else df


def lag_column_names(window_size: int = 7, target_col: str = TARGET_COLUMN) -> list:
    return [f"{target_col}_lag_{lag}" for lag in range(1, window_size + 1)]


def get_feature_columns(extended: bool = False) -> list:
    """Feature columns in the canonical (API-compatible) order."""
    if extended:
        return CALENDAR_FEATURE_COLUMNS + EXTENDED_CALENDAR_COLUMNS + HOLIDAY_FLAG_COLUMNS
    return list(FEATURE_COLUMNS)


def prepare_training_frame(
    raw_df: pl.DataFrame,
    state: str = None,
    min_date: datetime.date = None,
    window_size: int = 7,
    extended_features: bool = False,
    allow_missing_dates: bool = False,
) -> pl.DataFrame:
    """Build a model-ready daily frame from raw or pivoted donation data.

    Args:
        raw_df: Long-format (date, state, blood_type, donations) or pivoted
            (date, state, a, b, ab, o, all) frame.
        state: Restrict to a single state; None means nationwide aggregate.
        min_date: Drop rows before this date (applied after lag creation so
            early lags are still computed from full history).
        window_size: Number of lag columns.
        extended_features: Include cyclical/trend calendar features.
        allow_missing_dates: Tolerate gaps in the daily series.

    Returns:
        A date-sorted frame with target, holiday, calendar and lag columns.
    """
    df = pivot_blood_types(raw_df)
    df = aggregate_daily_donations(df, state=state)
    report = validate_daily_frame(df, allow_missing_dates=allow_missing_dates)
    if report["missing_dates"]:
        import warnings
        warnings.warn(f"Series has {len(report['missing_dates'])} missing dates.")

    df = add_holiday_flags(df)
    df = add_calendar_features(df, extended=extended_features)
    df = add_lag_features(df, window_size=window_size)

    if min_date is not None:
        df = df.filter(pl.col("date") >= min_date)

    return df.sort("date")


def frame_to_arrays(
    df: pl.DataFrame,
    window_size: int = 7,
    extended_features: bool = False,
    target_col: str = TARGET_COLUMN,
):
    """Convert a prepared frame to (X_seq, X_features, y) numpy arrays."""
    x_seq = (
        df.select(lag_column_names(window_size, target_col))
        .to_numpy()
        .reshape(-1, window_size, 1)
    )
    x_features = df.select(get_feature_columns(extended_features)).to_numpy()
    y = df[target_col].to_numpy()
    return x_seq, x_features, y


def list_available_states(raw_df: pl.DataFrame) -> list:
    """Sorted list of states present in the raw data."""
    return sorted(raw_df.select("state").unique().to_series().to_list())
