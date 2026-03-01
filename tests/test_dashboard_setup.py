"""
Integration test to verify the dashboard setup works correctly.
Tests data loading, caching, and basic data validation.
"""

import sys
import os
import pytest
import pickle
from src.donations.setup_and_validation import download_data
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_URL = "https://storage.data.gov.my/healthcare/blood_donations_state.parquet"


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test cache files."""
    return tmp_path


@pytest.fixture
def cache_file_path(tmp_dir):
    """Provide path for cache file in temporary directory."""
    return tmp_dir / "blood_donations_test.pkl"


class TestDashboardSetup:
    """Integration tests for dashboard data setup and caching."""

    def test_data_download(self):
        """Test that data can be downloaded from data.gov.my."""
        df = download_data(DATA_URL)
        
        # Verify DataFrame is returned
        assert isinstance(df, pl.DataFrame)
        
        # Verify data has expected structure
        assert df.shape[0] > 0, "Downloaded data should have rows"
        assert df.shape[1] > 0, "Downloaded data should have columns"
        
        # Verify expected columns exist
        expected_columns = {'date', 'state', 'blood_type', 'donations'}
        actual_columns = set(df.columns)
        assert expected_columns.issubset(actual_columns), \
            f"Missing columns: {expected_columns - actual_columns}"
    
    def test_pickle_save_and_load(self, cache_file_path):
        """Test that data can be saved to and loaded from pickle cache."""
        # Download fresh data
        df = download_data(DATA_URL)
        original_shape = df.shape
        
        # Save to pickle
        with open(cache_file_path, 'wb') as f:
            pickle.dump(df, f)
        
        assert cache_file_path.exists(), "Cache file should be created"
        
        # Load from pickle
        with open(cache_file_path, 'rb') as f:
            df_cached = pickle.load(f)
        
        # Verify loaded data matches original
        assert isinstance(df_cached, pl.DataFrame)
        assert df_cached.shape == original_shape
        assert df_cached.columns == df.columns
    
    def test_data_validation_states(self):
        """Test that downloaded data contains expected states."""
        df = download_data(DATA_URL)
        
        states = df.select('state').unique()
        num_states = states.shape[0]
        
        # Malaysia has 13 states + 3 federal territories
        assert num_states > 0, "Should have at least one state"
        assert num_states <= 20, "Should not have more than 20 regions"
    
    def test_data_validation_blood_types(self):
        """Test that downloaded data contains expected blood types."""
        df = download_data(DATA_URL)
        
        blood_types = df.select('blood_type').unique()
        num_blood_types = blood_types.shape[0]
        
        # Should have multiple blood types (A+, A-, B+, B-, AB+, AB-, O+, O-, all, etc.)
        assert num_blood_types > 0, "Should have at least one blood type"
        
        # Verify 'all' blood type exists (used for aggregation)
        blood_type_values = set(blood_types.to_series().to_list())
        assert 'all' in blood_type_values, "Should contain 'all' blood type for aggregation"
    
    def test_data_validation_date_range(self):
        """Test that downloaded data has valid date range."""
        df = download_data(DATA_URL)
        
        min_date = df.select('date').min()[0, 0]
        max_date = df.select('date').max()[0, 0]
        
        # Verify dates are not null
        assert min_date is not None, "Minimum date should not be null"
        assert max_date is not None, "Maximum date should not be null"
        
        # Verify date range makes sense
        assert min_date < max_date, "Minimum date should be before maximum date"
    
    def test_data_validation_donations(self):
        """Test that donation values are valid."""
        df = download_data(DATA_URL)
        
        total_donations = df.select('donations').sum()[0, 0]
        min_donations = df.select('donations').min()[0, 0]
        max_donations = df.select('donations').max()[0, 0]
        
        # Verify donations are positive
        assert total_donations > 0, "Total donations should be positive"
        assert min_donations >= 0, "Minimum donations should be non-negative"
        assert max_donations > 0, "Maximum donations should be positive"
        assert max_donations < 1_000_000, "Maximum donations per record seems unrealistic"
    
    def test_data_validation_no_nulls_in_key_columns(self):
        """Test that key columns have no null values."""
        df = download_data(DATA_URL)
        
        key_columns = ['date', 'state', 'blood_type', 'donations']
        
        for col in key_columns:
            null_count = df.select(pl.col(col).is_null().sum())[0, 0]
            assert null_count == 0, f"Column '{col}' should have no null values"
    
    def test_cache_is_reusable(self, cache_file_path):
        """Test that cached data can be reused multiple times."""
        # Download and cache data
        df_original = download_data(DATA_URL)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(df_original, f)
        
        # Load cache multiple times
        for _ in range(3):
            with open(cache_file_path, 'rb') as f:
                df_cached = pickle.load(f)
            
            assert df_cached.shape == df_original.shape
            assert df_cached.columns == df_original.columns
