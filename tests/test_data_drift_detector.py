"""
Test module for data drift detection functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from src.donations.data_drift_detector import DataDriftDetector


def generate_sample_datasets(n_samples: int = 100, random_seed: int = 42) -> dict:
    """
    Generate three generic datasets for testing drift detection.
    
    Two datasets from the same distribution and one from a different distribution
    to demonstrate drift detection capabilities.
    
    Args:
        n_samples: Number of samples to generate for each dataset
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing three datasets:
        - 'reference': Normal distribution (mean=100, std=15)
        - 'same_distribution': Same distribution as reference
        - 'different_distribution': Different distribution (mean=120, std=20)
    """
    rng = np.random.Generator(np.random.PCG64(random_seed))
    
    # Dataset 1: Reference distribution (Normal: mean=100, std=15)
    reference_data = rng.normal(100, 15, n_samples)
    
    # Dataset 2: Same distribution as reference (Normal: mean=100, std=15)
    same_distribution_data = rng.normal(100, 15, n_samples)
    
    # Dataset 3: Different distribution (Normal: mean=120, std=20) - should show drift
    different_distribution_data = rng.normal(120, 20, n_samples)
    
    return {
        'reference': reference_data,
        'same_distribution': same_distribution_data,
        'different_distribution': different_distribution_data
    }


class TestDataDriftDetector:
    
    def test_init_valid_parameters(self):
        """Test detector initialization with valid parameters."""
        detector = DataDriftDetector(
            reference_window_size=20,
            rolling_window_size=15,
            significance_level=0.01,
            test_method='ks'
        )
        
        assert detector.reference_window_size == 20
        assert detector.rolling_window_size == 15
        assert detector.significance_level == 0.01
        assert detector.test_method == 'ks'
        assert detector.min_samples == 15  # default value
    
    def test_init_invalid_test_method(self):
        """Test detector initialization with invalid test method."""
        with pytest.raises(ValueError, match="test_method must be one of"):
            DataDriftDetector(test_method='invalid_method')
    
    def test_set_reference_data_numpy(self):
        """Test setting reference data with numpy array."""
        detector = DataDriftDetector(reference_window_size=10)
        data = np.random.normal(0, 1, 20)
        
        detector.set_reference_data(data)
        
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 10  # Should take only reference_window_size
        np.testing.assert_array_equal(detector.reference_data, data[:10])
    
    def test_set_reference_data_dataframe(self):
        """Test setting reference data with polars DataFrame."""
        detector = DataDriftDetector(reference_window_size=15)
        
        # Create test DataFrame
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(25)]
        df = pl.DataFrame({
            'date': dates,
            'donations': np.random.normal(100, 20, 25)
        })
        
        detector.set_reference_data(df, column='donations')
        
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 15
    
    def test_set_reference_data_no_column_specified(self):
        """Test error when column not specified for DataFrame."""
        detector = DataDriftDetector()
        df = pl.DataFrame({'values': [1, 2, 3, 4, 5]})
        
        with pytest.raises(ValueError, match="Column name must be specified"):
            detector.set_reference_data(df)
    
    def test_set_reference_data_invalid_type(self):
        """Test error with invalid data type."""
        detector = DataDriftDetector()
        
        with pytest.raises(ValueError, match="Provided data must be numpy array or polars dataframe"):
            detector.set_reference_data([1, 2, 3, 4, 5])  # list is not supported
    
    def test_set_reference_data_insufficient_samples(self):
        """Test warning when reference data has fewer samples than window size."""
        detector = DataDriftDetector(reference_window_size=20)
        data = np.random.normal(0, 1, 10)  # Less than reference_window_size
        
        with pytest.warns(UserWarning, match="Reference data has 10 samples"):
            detector.set_reference_data(data)
        
        assert len(detector.reference_data) == 10
    
    def test_kolmogorov_smirnov_test_same_distribution(self):
        """Test KS test with samples from same distribution."""
        detector = DataDriftDetector(test_method='ks')
        
        # Create samples from same distribution
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 50)
        sample2 = np.random.normal(100, 15, 50)
        
        result = detector._kolmogorov_smirnov_test(sample1, sample2)
        
        assert result['test'] == 'kolmogorov_smirnov'
        assert 'statistic' in result
        assert 'p_value' in result
        assert isinstance(result['drift_detected'], np.bool_)
        # With same distribution, p-value should typically be > 0.05
    
    def test_kolmogorov_smirnov_test_different_distribution(self):
        """Test KS test with samples from different distributions."""
        detector = DataDriftDetector(test_method='ks', significance_level=0.05)
        
        # Create samples from different distributions
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 100)  # mean=100
        sample2 = np.random.normal(150, 15, 100)  # mean=150, should be significantly different
        
        result = detector._kolmogorov_smirnov_test(sample1, sample2)
        
        assert result['test'] == 'kolmogorov_smirnov'
        assert result['drift_detected'] is np.True_  # Should detect drift
        assert result['p_value'] < 0.05
    
    def test_mann_whitney_test(self):
        """Test Mann-Whitney U test."""
        detector = DataDriftDetector(test_method='mannwhitney')
        
        # Create samples from different distributions
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 50)
        sample2 = np.random.normal(120, 15, 50)
        
        result = detector._mann_whitney_test(sample1, sample2)
        
        assert result['test'] == 'mann_whitney'
        assert 'statistic' in result
        assert 'p_value' in result
        assert isinstance(result['drift_detected'], np.bool_)
    
    def test_anderson_darling_test(self):
        """Test Anderson-Darling test."""
        detector = DataDriftDetector(test_method='anderson')
        
        # Create samples
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 50)
        sample2 = np.random.normal(120, 15, 50)
        
        result = detector._anderson_darling_test(sample1, sample2)
        
        assert result['test'] == 'anderson_darling'
        assert 'statistic' in result
        assert isinstance(result['drift_detected'], np.bool_)

    def test_combined_test(self):
        """Test combined statistical tests."""
        detector = DataDriftDetector(test_method='combined')
        
        # Create samples from different distributions
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 50)
        sample2 = np.random.normal(130, 15, 50)
        
        result = detector._combined_test(sample1, sample2)
        
        assert result['test'] == 'combined'
        assert 'ks_statistic' in result
        assert 'ks_p_value' in result
        assert 'mw_statistic' in result
        assert 'mw_p_value' in result
        assert 'min_p_value' in result
        assert isinstance(result['drift_detected'], np.bool_)
    
    def test_detect_drift_single_window_no_reference_data(self):
        """Test error when no reference data is set."""
        detector = DataDriftDetector()
        test_data = np.random.normal(0, 1, 30)
        
        with pytest.raises(ValueError, match="Reference data must be set before detecting drift"):
            detector.detect_drift_single_window(test_data)
    
    def test_detect_drift_single_window_insufficient_samples(self):
        """Test handling of insufficient samples."""
        detector = DataDriftDetector(min_samples=20)
        
        # Set reference data
        reference = np.random.normal(0, 1, 30)
        detector.set_reference_data(reference)
        
        # Test data with insufficient samples
        test_data = np.random.normal(0, 1, 5)
        
        result = detector.detect_drift_single_window(test_data, window_start_idx=10)
        
        assert result['drift_detected'] is False
        assert 'Insufficient samples' in result['reason']
        assert result['window_start_idx'] == 10
    
    def test_detect_drift_single_window_ks_test(self):
        """Test single window drift detection with KS test."""
        detector = DataDriftDetector(test_method='ks', significance_level=0.05)
        
        # Set reference data (normal distribution mean=0)
        np.random.seed(42)
        reference = np.random.normal(0, 1, 50)
        detector.set_reference_data(reference)
        
        # Test data from different distribution (mean=3)
        test_data = np.random.normal(3, 1, 50)
        
        result = detector.detect_drift_single_window(test_data, window_start_idx=50)
        
        assert result['test_method'] == 'ks'
        assert result['window_start_idx'] == 50
        assert result['reference_size'] == 30
        assert result['test_size'] == 50
        assert 'drift_detected' in result
        assert 'p_value' in result
        assert 'statistic' in result
    
    def test_detect_drift_single_window_combined_test(self):
        """Test single window drift detection with combined tests."""
        detector = DataDriftDetector(test_method='combined', significance_level=0.05)
        
        # Set reference data
        np.random.seed(42)
        reference = np.random.normal(100, 15, 40)
        detector.set_reference_data(reference)
        
        # Test data from different distribution
        test_data = np.random.normal(120, 15, 40)
        
        result = detector.detect_drift_single_window(test_data)
        
        assert result['test_method'] == 'combined'
        assert 'ks_p_value' in result
        assert 'mw_p_value' in result
        assert 'min_p_value' in result
        assert isinstance(result['drift_detected'], np.bool_)
    
    def test_all_test_methods(self):
        """Test that all test methods work correctly."""
        test_methods = ['ks', 'mannwhitney', 'anderson', 'combined']
        
        # Create test data
        np.random.seed(42)
        reference = np.random.normal(100, 15, 50)
        test_data = np.random.normal(120, 15, 50)  # Different distribution
        
        for method in test_methods:
            detector = DataDriftDetector(test_method=method)
            detector.set_reference_data(reference)
            
            result = detector.detect_drift_single_window(test_data)
            
            assert result['test_method'] == method
            assert 'drift_detected' in result
            assert isinstance(result['drift_detected'], np.bool_)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        detector = DataDriftDetector()
        
        # Test with very small arrays
        small_ref = np.array([1.0, 2.0])
        small_test = np.array([1.1, 2.1])
        
        detector.set_reference_data(small_ref)
        result = detector.detect_drift_single_window(small_test)
        
        # Should handle small samples gracefully
        assert 'drift_detected' in result or 'reason' in result
    
    def test_reproducibility(self):
        """Test that results are reproducible with same data."""
        detector1 = DataDriftDetector(test_method='ks', significance_level=0.05)
        detector2 = DataDriftDetector(test_method='ks', significance_level=0.05)
        
        # Same data for both detectors (length > 15)
        reference = np.arange(1, 21)  # 20 elements: 1 to 20
        test_data = np.arange(2, 22)  # 20 elements: 2 to 21
        
        detector1.set_reference_data(reference)
        detector2.set_reference_data(reference)
        
        result1 = detector1.detect_drift_single_window(test_data)
        result2 = detector2.detect_drift_single_window(test_data)
        
        print(result1)
        print(result2)
        
        # Results should be identical
        assert result1['drift_detected'] == result2['drift_detected']
        assert result1['p_value'] == result2['p_value']
        assert result1['statistic'] == result2['statistic']


def test_sample_datasets_functionality():
    """Test the sample datasets generation and usage."""
    # Test using the generated sample datasets
    detector = DataDriftDetector(test_method='ks')
    
    # Generate test datasets
    datasets = generate_sample_datasets(n_samples=50, random_seed=42)
    
    detector.set_reference_data(datasets['reference'])
    
    # Test with same distribution
    result_same = detector.detect_drift_single_window(datasets['same_distribution'])
    
    # Test with different distribution
    result_diff = detector.detect_drift_single_window(datasets['different_distribution'])
    
    # Assertions
    assert isinstance(result_same['drift_detected'], np.bool_)
    assert isinstance(result_diff['drift_detected'], np.bool_)
    assert 'p_value' in result_same
    assert 'p_value' in result_diff
    
    # Different distribution should have lower p-value than same distribution
    assert result_diff['p_value'] <= result_same['p_value']
    
    # Test dataset characteristics
    assert len(datasets['reference']) == 50
    assert len(datasets['same_distribution']) == 50
    assert len(datasets['different_distribution']) == 50
    
    # Verify different distribution has different mean
    assert abs(datasets['reference'].mean() - datasets['different_distribution'].mean()) > 10


if __name__ == "__main__":
    pytest.main([__file__])
