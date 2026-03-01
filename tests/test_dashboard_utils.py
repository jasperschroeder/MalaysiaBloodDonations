"""
Test module for dashboard utility functions.
Tests infer_lag_values, call_prediction_api, and calculate_180day_average.
"""

import sys
import os
import pytest
import polars as pl
import datetime
from unittest.mock import Mock, patch
import requests
from src.dashboard_utils import (
    infer_lag_values,
    call_prediction_api,
    calculate_180day_average,
    API_URL
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_donations_df():
    """
    Generate sample donations DataFrame for testing.
    Creates data for 365 days with varying donation counts.
    """
    base_date = datetime.date(2024, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(365)]
    
    # Create sample data with blood_type='all' and varying donations
    data = {
        "date": dates * 3,  # Repeat for different blood types
        "blood_type": ["all"] * 365 + ["A+"] * 365 + ["B+"] * 365,
        "donations": [1000 + (i % 100) for i in range(365)] * 3,
        "state": ["Selangor"] * 365 * 3,
    }
    
    return pl.DataFrame(data)


@pytest.fixture
def sparse_donations_df():
    """
    Generate DataFrame with sparse/missing dates for testing fallback logic.
    """
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 3),  # Skip day 2
        datetime.date(2024, 1, 5),  # Skip day 4
        datetime.date(2024, 1, 10),  # Skip days 6-9
    ]
    
    data = {
        "date": dates,
        "blood_type": ["all"] * 4,
        "donations": [1000, 1100, 1200, 1300],
        "state": ["Selangor"] * 4,
    }
    
    return pl.DataFrame(data)


@pytest.fixture
def empty_donations_df():
    """Generate empty DataFrame with correct schema."""
    return pl.DataFrame({
        "date": [],
        "blood_type": [],
        "donations": [],
        "state": [],
    }).with_columns([
        pl.col("date").cast(pl.Date),
        pl.col("blood_type").cast(pl.Utf8),
        pl.col("donations").cast(pl.Int64),
        pl.col("state").cast(pl.Utf8),
    ])


@pytest.fixture
def mock_api_response():
    """Standard successful API response."""
    return {
        "prediction": 1234.56,
        "input_features": {
            "lag1": 1000,
            "lag2": 1050,
            "lag3": 980,
            "lag4": 1020,
            "lag5": 1100,
            "lag6": 1030,
            "lag7": 990,
            "nextday": "20240115",
            "high_donation_holiday": 0,
            "low_donation_holiday": 0,
            "religion_or_culture_holiday": 0,
            "other_holiday": 0
        }
    }


# ============================================================================
# Test infer_lag_values
# ============================================================================

class TestInferLagValues:
    """Test suite for infer_lag_values function."""
    
    def test_infer_lags_normal_case(self, sample_donations_df):
        """Test lag inference with complete data for all 7 days."""
        prediction_date = datetime.date(2024, 1, 15)
        lags = infer_lag_values(sample_donations_df, prediction_date)
        
        # Should have all 7 lags
        assert len(lags) == 7
        assert all(f"lag{i}" in lags for i in range(1, 8))
        
        # All lag values should be integers
        assert all(isinstance(lags[f"lag{i}"], int) for i in range(1, 8))
        
        # Lag values should be positive
        assert all(lags[f"lag{i}"] > 0 for i in range(1, 8))
        
        # lag1 should correspond to prediction_date - 1 day
        # From our fixture, day 14 (index 13) has donations = 1000 + 13 = 1013
        assert lags["lag1"] == 1013
        assert lags["lag2"] == 1012
        assert lags["lag7"] == 1007
    
    def test_infer_lags_filters_blood_type_all(self, sample_donations_df):
        """Test that only blood_type=='all' is used to avoid double counting."""
        prediction_date = datetime.date(2024, 1, 15)
        lags = infer_lag_values(sample_donations_df, prediction_date)
        
        # With blood_type filtering, we should only sum 'all' records
        # Not A+ and B+ records (which would triple the value)
        assert lags["lag1"] == 1013  # Not 3039 (if all blood types were summed)
    
    def test_infer_lags_with_missing_dates(self, sparse_donations_df):
        """Test lag inference when some dates are missing (uses fallback to recent value)."""
        prediction_date = datetime.date(2024, 1, 11)
        lags = infer_lag_values(sparse_donations_df, prediction_date)
        
        # Should still have all 7 lags
        assert len(lags) == 7
        
        # lag1 = Jan 10 (exists) = 1300
        assert lags["lag1"] == 1300
        
        # lag2 = Jan 9 (missing, should use most recent before Jan 9 = Jan 5 = 1200)
        assert lags["lag2"] == 1200
        
        # lag3-lag6 = Jan 8, 7, 6, 5 (only Jan 5 exists = 1200)
        assert lags["lag3"] == 1200
        assert lags["lag4"] == 1200
        assert lags["lag5"] == 1200
        assert lags["lag6"] == 1200
        
        # lag7 = Jan 4 (missing, should use Jan 3 = 1100)
        assert lags["lag7"] == 1100
    
    def test_infer_lags_consecutive_missing_dates(self, sparse_donations_df):
        """Test when multiple consecutive dates are missing."""
        # Use prediction date Jan 11 instead of Jan 6 to ensure enough history
        # With earliest data on Jan 1, predicting for Jan 11 requires lags back to Jan 4
        prediction_date = datetime.date(2024, 1, 11)
        lags = infer_lag_values(sparse_donations_df, prediction_date)
        
        # lag1 = Jan 10 (exists) = 1300
        assert lags["lag1"] == 1300
        
        # lag2 = Jan 9 (missing, use Jan 10 [most recent before Jan 9] - wait no, Jan 5 is most recent before Jan 9)
        # Actually sparse_donations_df has: Jan 1, 3, 5, 10
        # For lag2 (Jan 9): most recent before Jan 9 is Jan 5 = 1200
        assert lags["lag2"] == 1200
        
        # lag3 = Jan 8 (missing, use Jan 5 = 1200)
        assert lags["lag3"] == 1200
        
        # lag4 = Jan 7 (missing, use Jan 5 = 1200)
        assert lags["lag4"] == 1200
        
        # lag5 = Jan 6 (missing, use Jan 5 = 1200)
        assert lags["lag5"] == 1200
        
        # lag6 = Jan 5 (exists) = 1200
        assert lags["lag6"] == 1200
        
        # lag7 = Jan 4 (missing, use Jan 3 = 1100)
        assert lags["lag7"] == 1100
    
    def test_infer_lags_no_historical_data_raises_error(self, empty_donations_df):
        """Test that ValueError is raised when no historical data exists."""
        prediction_date = datetime.date(2024, 1, 15)
        
        with pytest.raises(ValueError, match="No historical data available before"):
            infer_lag_values(empty_donations_df, prediction_date)
    
    def test_infer_lags_insufficient_history_raises_error(self, sparse_donations_df):
        """Test error when prediction date is before all available data."""
        # All data starts from Jan 1, 2024, so predicting for Jan 1 should fail
        prediction_date = datetime.date(2024, 1, 1)
        
        with pytest.raises(ValueError, match="No historical data available before"):
            infer_lag_values(sparse_donations_df, prediction_date)
    
    def test_infer_lags_date_casting(self):
        """Test that date column is properly cast to pl.Date type."""
        # DataFrame with date as string - needs sufficient history
        base_date = datetime.date(2024, 1, 1)
        dates = [(base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 30,
            "donations": [1000 + (i * 10) for i in range(30)],
            "state": ["Selangor"] * 30,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 1, 20)
        lags = infer_lag_values(df, prediction_date)
        
        assert len(lags) == 7
        # lag1 = Jan 19 (day 18, index from 0) = 1000 + 18*10 = 1180
        assert lags["lag1"] == 1180
        # lag2 = Jan 18 = 1000 + 17*10 = 1170
        assert lags["lag2"] == 1170
        # lag3 = Jan 17 = 1000 + 16*10 = 1160
        assert lags["lag3"] == 1160
    
    def test_infer_lags_aggregates_multiple_entries_same_date(self):
        """Test that donations on the same date are summed correctly."""
        base_date = datetime.date(2024, 1, 1)
        # Create data for 10 days, each day appears 3 times
        dates = []
        donations = []
        states = []
        for i in range(10):
            for _ in range(3):
                dates.append(base_date + datetime.timedelta(days=i))
                donations.append(400)  # Each entry is 400, so total per day = 1200
                states.append("Selangor")
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 30,
            "donations": donations,
            "state": states,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 1, 10)
        lags = infer_lag_values(df, prediction_date)
        
        # lag1 should be sum of all donations on Jan 9 (day 8)
        assert lags["lag1"] == 1200  # 3 entries * 400 each


# ============================================================================
# Test call_prediction_api
# ============================================================================

class TestCallPredictionAPI:
    """Test suite for call_prediction_api function."""
    
    def test_successful_api_call(self, mock_api_response):
        """Test successful API call returns expected response."""
        request_data = {
            "lag1": 1000,
            "lag2": 1050,
            "lag3": 980,
            "lag4": 1020,
            "lag5": 1100,
            "lag6": 1030,
            "lag7": 990,
            "nextday": "20240115",
            "high_donation_holiday": 0,
            "low_donation_holiday": 0,
            "religion_or_culture_holiday": 0,
            "other_holiday": 0
        }
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            result = call_prediction_api(request_data)
            
            # Verify correct endpoint was called
            mock_post.assert_called_once_with(
                f"{API_URL}/predict",
                json=request_data,
                timeout=10
            )
            
            # Verify response structure
            assert "prediction" in result
            assert result["prediction"] == 1234.56
            assert "input_features" in result
    
    def test_api_call_timeout(self):
        """Test that timeout error is caught and returned as error dict."""
        request_data = {"lag1": 1000}
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            result = call_prediction_api(request_data)
            
            assert "error" in result
            assert "API request failed" in result["error"]
            assert "timed out" in result["error"].lower()
    
    def test_api_call_connection_error(self):
        """Test that connection error is caught and returned as error dict."""
        request_data = {"lag1": 1000}
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            result = call_prediction_api(request_data)
            
            assert "error" in result
            assert "API request failed" in result["error"]
    
    def test_api_call_http_error(self):
        """Test that HTTP errors (4xx, 5xx) are caught and returned as error dict."""
        request_data = {"lag1": 1000}
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_post.return_value = mock_response
            
            result = call_prediction_api(request_data)
            
            assert "error" in result
            assert "API request failed" in result["error"]
    
    def test_api_call_invalid_json_response(self):
        """Test handling of malformed JSON response."""
        request_data = {"lag1": 1000}
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            # Should raise the ValueError (not caught by RequestException)
            with pytest.raises(ValueError):
                call_prediction_api(request_data)
    
    def test_api_call_uses_correct_timeout(self):
        """Test that timeout parameter is set correctly."""
        request_data = {"lag1": 1000}
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"prediction": 1000}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            call_prediction_api(request_data)
            
            # Verify timeout=10 was passed
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["timeout"] == 10


# ============================================================================
# Test calculate_180day_average
# ============================================================================

class TestCalculate180DayAverage:
    """Test suite for calculate_180day_average function."""
    
    def test_sufficient_data_180_days(self, sample_donations_df):
        """Test average calculation with sufficient data (≥180 days)."""
        prediction_date = datetime.date(2024, 7, 1)  # Day 183 of year
        avg_daily_donations = 999.0  # Fallback value (should not be used)
        
        result = calculate_180day_average(
            sample_donations_df,
            prediction_date,
            avg_daily_donations
        )
        
        # Should calculate actual average, not use fallback
        assert result != avg_daily_donations
        assert isinstance(result, float)
        assert result > 0
        
        # With our fixture pattern (1000 + i % 100), average should be around 1050
        assert 1000 <= result <= 1100
    
    def test_exactly_180_days_of_data(self):
        """Test with exactly 180 days of data."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i) for i in range(180)]
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 180,
            "donations": [1000] * 180,  # Constant donations
            "state": ["Selangor"] * 180,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 6, 30)  # Day 181
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # Average should be exactly 1000
        assert result == 1000.0
    
    def test_insufficient_data_uses_fallback(self):
        """Test that fallback is used when less than 180 days of data in the window."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i) for i in range(30)]  # Only 30 days
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 30,
            "donations": [1000] * 30,
            "state": ["Selangor"] * 30,
        }
        df = pl.DataFrame(data)
        
        # Prediction date is way in the future, so no data in 180-day window
        prediction_date = datetime.date(2024, 12, 31)  # Dec 31
        # 180 days before Dec 31 = July 4, so window is July 4 - Dec 30
        # Our data is only Jan 1-30, so no data in window
        fallback_value = 1234.56
        
        result = calculate_180day_average(df, prediction_date, fallback_value)
        
        # Should use fallback since no data in 180-day window
        assert result == fallback_value
    
    def test_empty_dataframe_uses_fallback(self, empty_donations_df):
        """Test that fallback is used with empty DataFrame."""
        prediction_date = datetime.date(2024, 7, 1)
        fallback_value = 1500.0
        
        result = calculate_180day_average(
            empty_donations_df,
            prediction_date,
            fallback_value
        )
        
        assert result == fallback_value
    
    def test_filters_blood_type_all_only(self):
        """Test that only blood_type=='all' records are used."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i) for i in range(200)]
        
        data = {
            "date": dates * 3,  # Repeat for different blood types
            "blood_type": ["all"] * 200 + ["A+"] * 200 + ["B+"] * 200,
            "donations": [1000] * 200 + [500] * 200 + [300] * 200,
            "state": ["Selangor"] * 600,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 7, 20)
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # Should average only 'all' records (1000), not include A+ (500) or B+ (300)
        assert result == 1000.0
    
    def test_date_range_filtering(self):
        """Test that only data in [cutoff_date, prediction_date) is used."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i) for i in range(365)]
        
        # First 100 days: 500 donations, next 265 days: 2000 donations
        donations = [500] * 100 + [2000] * 265
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 365,
            "donations": donations,
            "state": ["Selangor"] * 365,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 12, 31)  # End of year
        # 180 days before = July 4 (day 185), so days 185-364 are in range
        # Days 185-364 = 180 days, all with 2000 donations
        
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # Should average only the last 180 days (all 2000)
        assert result == 2000.0
    
    def test_excludes_prediction_date(self):
        """Test that prediction_date itself is excluded from calculation."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i) for i in range(200)]
        
        # All days have 1000 donations except prediction date which has 9999
        donations = [1000] * 199 + [9999]
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 200,
            "donations": donations,
            "state": ["Selangor"] * 200,
        }
        df = pl.DataFrame(data)
        
        prediction_date = dates[-1]  # Last date with 9999 donations
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # 9999 should be excluded, average should be 1000
        assert result == 1000.0
    
    def test_partial_data_in_window(self):
        """Test with some data in 180-day window but not complete."""
        base_date = datetime.date(2024, 6, 1)
        # Only 50 days of data starting from June 1
        dates = [base_date + datetime.timedelta(days=i) for i in range(50)]
        
        data = {
            "date": dates,
            "blood_type": ["all"] * 50,
            "donations": [1200] * 50,
            "state": ["Selangor"] * 50,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 7, 25)  # July 25
        # 180 days before July 25 = Jan 27, so window is Jan 27 - July 24
        # Our data is June 1 - July 20 (50 days), which overlaps with window (June 1 - July 24)
        # So we should get the average of that overlapping data
        
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # Should average the 50 days that exist in window (all 1200)
        assert result == 1200.0
    
    def test_aggregates_multiple_entries_same_date(self):
        """Test that multiple entries on same date are aggregated properly."""
        base_date = datetime.date(2024, 1, 1)
        dates = [base_date + datetime.timedelta(days=i // 3) for i in range(600)]
        
        # Each date appears 3 times with 400 donations each (total 1200 per day)
        data = {
            "date": dates,
            "blood_type": ["all"] * 600,
            "donations": [400] * 600,
            "state": ["Selangor"] * 600,
        }
        df = pl.DataFrame(data)
        
        prediction_date = datetime.date(2024, 7, 20)
        result = calculate_180day_average(df, prediction_date, 999.0)
        
        # Each day has 1200 total (3 * 400), so average should be 1200
        assert result == 1200.0
