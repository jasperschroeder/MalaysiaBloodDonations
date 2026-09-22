import datetime

import numpy as np
import polars as pl
import pytest

from src.donations import data_preparation as dp


def _long_frame(states, start, n_days, base=100):
    """Build a synthetic long-format donation frame."""
    rows = []
    for s_idx, state in enumerate(states):
        for d in range(n_days):
            date = start + datetime.timedelta(days=d)
            for bt_idx, bt in enumerate(["a", "b", "ab", "o"]):
                rows.append({
                    "date": date,
                    "state": state,
                    "blood_type": bt,
                    "donations": base + 10 * s_idx + bt_idx + d,
                })
    return pl.DataFrame(rows)


class TestCategorizeHolidayName:
    def test_known_names(self):
        assert dp.categorize_holiday_name("Hari Wesak") == {
            "is_high_donation_holiday", "is_religion_or_culture_holiday"
        }
        assert dp.categorize_holiday_name("Peristiwa") == {
            "is_low_donation_holiday", "is_other_holiday"
        }

    def test_combined_names(self):
        flags = dp.categorize_holiday_name("Hari Pekerja; Hari Wesak")
        assert "is_high_donation_holiday" in flags
        assert "is_other_holiday" in flags
        assert "is_religion_or_culture_holiday" in flags

    def test_cleanup_patterns(self):
        assert dp.categorize_holiday_name("Hari Raya Puasa (Hari Kedua)") == {
            "is_low_donation_holiday", "is_religion_or_culture_holiday"
        }
        flags = dp.categorize_holiday_name("Hari Pertabalan Yang di-Pertuan Agong ke-16")
        assert flags == {"is_other_holiday"}

    def test_unknown_name(self):
        assert dp.categorize_holiday_name("Some Random Day") == set()


class TestPivotBloodTypes:
    def test_all_recalculated(self):
        df = _long_frame(["Perlis"], datetime.date(2024, 1, 1), 3)
        pivoted = dp.pivot_blood_types(df)
        assert "blood_type" not in pivoted.columns
        expected = df.filter(
            (pl.col("state") == "Perlis") & (pl.col("date") == datetime.date(2024, 1, 1))
        )["donations"].sum()
        actual = pivoted.filter(pl.col("date") == datetime.date(2024, 1, 1))["all"].item()
        assert actual == expected

    def test_already_pivoted_passthrough(self):
        df = pl.DataFrame({"date": [datetime.date(2024, 1, 1)], "state": ["Perlis"],
                           "a": [1], "b": [2], "ab": [3], "o": [4], "all": [10]})
        assert dp.pivot_blood_types(df).equals(df)


class TestAggregateDaily:
    def test_nationwide_sum(self):
        df = dp.pivot_blood_types(_long_frame(["Perlis", "Selangor"], datetime.date(2024, 1, 1), 2))
        daily = dp.aggregate_daily_donations(df)
        assert daily.height == 2
        assert "state" not in daily.columns

    def test_state_filter(self):
        df = dp.pivot_blood_types(_long_frame(["Perlis", "Selangor"], datetime.date(2024, 1, 1), 2))
        daily = dp.aggregate_daily_donations(df, state="Perlis")
        total = dp.aggregate_daily_donations(df)
        assert daily["all"].sum() < total["all"].sum()

    def test_unknown_state_raises(self):
        df = dp.pivot_blood_types(_long_frame(["Perlis"], datetime.date(2024, 1, 1), 1))
        with pytest.raises(ValueError, match="No rows found for state"):
            dp.aggregate_daily_donations(df, state="Atlantis")


class TestValidateDailyFrame:
    def _daily(self, n=10, start=datetime.date(2024, 1, 1)):
        return pl.DataFrame({
            "date": [start + datetime.timedelta(days=i) for i in range(n)],
            "all": list(range(100, 100 + n)),
        })

    def test_happy_path(self):
        report = dp.validate_daily_frame(self._daily())
        assert report["n_rows"] == 10
        assert report["missing_dates"] == []

    def test_duplicate_dates_raise(self):
        df = pl.concat([self._daily(3), self._daily(3)])
        with pytest.raises(ValueError, match="duplicated dates"):
            dp.validate_daily_frame(df)

    def test_negative_donations_raise(self):
        df = self._daily().with_columns(
            pl.when(pl.col("all") == 103).then(-1).otherwise(pl.col("all")).alias("all")
        )
        with pytest.raises(ValueError, match="negative donations"):
            dp.validate_daily_frame(df)

    def test_missing_dates_raise(self):
        df = self._daily(5).filter(pl.col("date") != datetime.date(2024, 1, 3))
        with pytest.raises(ValueError, match="missing dates"):
            dp.validate_daily_frame(df)
        report = dp.validate_daily_frame(df, allow_missing_dates=True)
        assert report["missing_dates"] == [datetime.date(2024, 1, 3)]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="Required column"):
            dp.validate_daily_frame(pl.DataFrame({"date": [datetime.date(2024, 1, 1)]}))


class TestHolidayFlags:
    def test_known_holiday_flagged(self):
        # 31 August is Hari Kebangsaan (Merdeka) - high donation + other holiday.
        df = pl.DataFrame({
            "date": [datetime.date(2024, 8, 31), datetime.date(2024, 9, 1)],
            "all": [500, 300],
        })
        out = dp.add_holiday_flags(df)
        row = out.filter(pl.col("date") == datetime.date(2024, 8, 31)).row(0, named=True)
        assert row["is_high_donation_holiday"] == 1
        assert row["is_other_holiday"] == 1
        normal = out.filter(pl.col("date") == datetime.date(2024, 9, 1)).row(0, named=True)
        assert all(normal[c] == 0 for c in dp.HOLIDAY_FLAG_COLUMNS)

    def test_hari_raya_puasa_is_low_donation(self):
        import holidays as holidays_lib
        my = holidays_lib.country_holidays("MY", years=[2024])
        raya = next(d for d, n in my.items() if "Hari Raya Puasa" in n)
        df = pl.DataFrame({"date": [raya], "all": [100]})
        row = dp.add_holiday_flags(df).row(0, named=True)
        assert row["is_low_donation_holiday"] == 1
        assert row["is_religion_or_culture_holiday"] == 1


class TestCalendarAndLags:
    def test_calendar_base(self):
        df = pl.DataFrame({"date": [datetime.date(2024, 1, 1)], "all": [1]})  # Monday
        row = dp.add_calendar_features(df).row(0, named=True)
        assert row["weekday"] == 1
        assert row["month"] == 1
        assert row["day_of_year"] == 1

    def test_calendar_extended(self):
        df = pl.DataFrame({
            "date": [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(3)],
            "all": [1, 2, 3],
        })
        out = dp.add_calendar_features(df, extended=True)
        for col in dp.EXTENDED_CALENDAR_COLUMNS:
            assert col in out.columns
        assert out["trend_days"].to_list() == [0, 1, 2]
        assert np.isclose(out["weekday_sin"][0], np.sin(2 * np.pi * 1 / 7))

    def test_lag_features(self):
        df = pl.DataFrame({
            "date": [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(10)],
            "all": list(range(10)),
        })
        out = dp.add_lag_features(df, window_size=3)
        assert out.height == 7  # first 3 rows dropped
        first = out.row(0, named=True)
        assert first["all_lag_1"] == 2 and first["all_lag_3"] == 0


class TestPrepareTrainingFrame:
    def test_end_to_end_nationwide(self):
        raw = _long_frame(["Perlis", "Selangor"], datetime.date(2024, 1, 1), 40)
        df = dp.prepare_training_frame(raw, window_size=7)
        assert df.height == 40 - 7
        for col in dp.FEATURE_COLUMNS + dp.lag_column_names(7):
            assert col in df.columns
        assert df["date"].is_sorted()

    def test_state_level_and_min_date(self):
        raw = _long_frame(["Perlis", "Selangor"], datetime.date(2024, 1, 1), 40)
        df = dp.prepare_training_frame(
            raw, state="Perlis", min_date=datetime.date(2024, 1, 20), window_size=3
        )
        assert df["date"].min() >= datetime.date(2024, 1, 20)
        # Lags still computed from full history (no nulls at the boundary).
        assert df.null_count().sum_horizontal().item() == 0

    def test_frame_to_arrays(self):
        raw = _long_frame(["Perlis"], datetime.date(2024, 1, 1), 30)
        df = dp.prepare_training_frame(raw, window_size=7)
        x_seq, x_features, y = dp.frame_to_arrays(df, window_size=7)
        assert x_seq.shape == (23, 7, 1)
        assert x_features.shape == (23, len(dp.FEATURE_COLUMNS))
        assert y.shape == (23,)
