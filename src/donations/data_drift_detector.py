import numpy as np
import polars as pl
from scipy import stats
from typing import Union, Dict
import warnings

class DataDriftDetector:
    """
    A class for detecting data drift in time series data.

    This detector uses statistical tests to compare the distribution of a reference period
    with rolling windows of subsequent data to identify potential distribution changes.
    """

    def __init__(
        self,
        reference_window_size: int = 30,
        rolling_window_size: int = 30,
        significance_level: float = 0.05,
        test_method: str = "ks",
        min_samples: int = 15
    ):
        self.reference_window_size=reference_window_size
        self.rolling_window_size=rolling_window_size
        self.significance_level=significance_level
        self.test_method=test_method.lower()
        self.min_samples=min_samples

        # Validate test method
        valid_methods = ['ks', 'mannwhitney', 'anderson', 'combined']
        if self.test_method not in valid_methods:
            raise ValueError(f"test_method must be one of {valid_methods}.")

        self.reference_data=None
        self.drift_results=[]

    def set_reference_data(self, data: Union[np.ndarray, pl.DataFrame], column: str = None) -> None:
        """
        Set the reference data for drift detection.

        Inputs:
            data: Reference data (numpy array or polars DataFrame)
            column: Column name if data is a DataFrame
        """

        if isinstance(data, pl.DataFrame):
            if column is None:
                raise ValueError("Column name must be specified for DataFrame input.")
            reference_values = data[column].to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Provided data must be numpy array or polars dataframe.")
        else:
            reference_values = np.array(data)

        # Take only the specified window size from the beginning
        if len(reference_values) < self.reference_window_size:
            warnings.warn(f"Reference data has {len(reference_values)} samples, "
                          f"but reference_window_size is {self.reference_window_size}")
            self.reference_data=reference_values
        else:
            self.reference_data=reference_values[:self.reference_window_size]

    def _kolmogorov_smirnov_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict:
        """Perform KS test"""
        try:
            statistic, p_value = stats.ks_2samp(sample1, sample2)
            return {
                'test': 'kolmogorov_smirnov',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.significance_level
            }
        except Exception as e:
            return {
                'test': 'kolmogorov_smirnov',
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'error': str(e)
            }
            
    def _mann_whitney_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict:
        """Perform Mann-Whitney U test."""
        try:
            statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            return {
                'test': 'mann_whitney',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.significance_level
            }
        except Exception as e:
            return {
                'test': 'mann_whitney',
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'error': str(e)
            }


    def _anderson_darling_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict:
        """Perform Anderson-Darling test."""
        try:
            # Use Anderson-Darling k-sample test
            result = stats.anderson_ksamp([sample1, sample2])
            return {
                'test': 'anderson_darling',
                'statistic': result.statistic,
                'p_value': result.pvalue if hasattr(result, 'pvalue') else np.nan,
                'drift_detected': result.pvalue < self.significance_level if hasattr(result, 'pvalue') else False
            }
        except Exception as e:
            return {
                'test': 'anderson_darling',
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'error': str(e)
            }


    def _combined_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Dict:
        """Perform combined statistical tests."""
        ks_result = self._kolmogorov_smirnov_test(sample1, sample2)
        mw_result = self._mann_whitney_test(sample1, sample2)
        
        # Drift detected if either test shows significance
        drift_detected = ks_result['drift_detected'] or mw_result['drift_detected']
        
        return {
            'test': 'combined',
            'ks_statistic': ks_result['statistic'],
            'ks_p_value': ks_result['p_value'],
            'mw_statistic': mw_result['statistic'],
            'mw_p_value': mw_result['p_value'],
            'drift_detected': drift_detected,
            'min_p_value': min(ks_result['p_value'], mw_result['p_value'])
        }


    def detect_drift_single_window(
        self,
        test_data: np.ndarray,
        window_start_idx: int = None
    ) -> Dict:
        """
        Detect drift for a single test window.
        """

        if self.reference_data is None:
            raise ValueError("Reference data must be set before detecting drift.")

        if len(test_data) < self.min_samples:
            return {
                'window_start_idx': window_start_idx,
                'test_method': self.test_method,
                'drift_detected': False,
                'reason': f'Insufficient samples: {len(test_data)} < {self.min_samples}'
            }
            
        # Perform the appropriate statistical test
        if self.test_method == 'ks':
            result = self._kolmogorov_smirnov_test(self.reference_data, test_data)
        elif self.test_method == 'mannwhitney':
            result = self._mann_whitney_test(self.reference_data, test_data)
        elif self.test_method == 'anderson':
            result = self._anderson_darling_test(self.reference_data, test_data)
        elif self.test_method == 'combined':
            result = self._combined_test(self.reference_data, test_data)
        
        # Add metadata
        result['window_start_idx']=window_start_idx
        result['reference_size']=len(self.reference_data)
        result['test_size']=len(test_data)
        result['test_method']=self.test_method

        return result
    
