"""
Quick test to verify the dashboard setup works correctly.
This script tests data loading and basic functionality.
"""

import sys
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from donations.setup_and_validation import download_data  # noqa
import polars as pl  # noqa  

DATA_URL = "https://storage.data.gov.my/healthcare/blood_donations_state.parquet"
PKL_PATH = Path(__file__).parent.parent / "tmp" / "blood_donations.pkl"


def test_data_download_and_cache():
    """Test downloading and caching data."""
    print("Testing data download and caching...")

    # Create tmp directory if needed
    PKL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove cache to test fresh download
    if PKL_PATH.exists():
        PKL_PATH.unlink()
        print("  ✓ Cleared existing cache")

    # Download data
    print("  ↓ Downloading data from data.gov.my...")
    try:
        df = download_data(DATA_URL)
        print(f"  ✓ Downloaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

    # Save to pickle
    print("  💾 Saving to pickle cache...")
    try:
        with open(PKL_PATH, 'wb') as f:
            pickle.dump(df, f)
        print(f"  ✓ Saved to {PKL_PATH}")
    except Exception as e:
        print(f"  ✗ Cache save failed: {e}")
        return False

    # Load from pickle
    print("  📂 Loading from cache...")
    try:
        with open(PKL_PATH, 'rb') as f:
            df_cached = pickle.load(f)
        print(f"  ✓ Loaded {df_cached.shape[0]:,} rows from cache")
    except Exception as e:
        print(f"  ✗ Cache load failed: {e}")
        return False

    # Basic validation
    print("  ✓ Data Validation:")
    print(f"    - States: {df_cached.select('state').unique().shape[0]}")
    print(f"    - Blood Types: {df_cached.select('blood_type').unique().shape[0]}")
    print(f"    - Date Range: {df_cached.select('date').min()[0, 0]} to {df_cached.select('date').max()[0, 0]}")
    print(f"    - Total Donations: {df_cached.select('donations').sum()[0, 0]:,.0f}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Malaysia Blood Donations Dashboard - Setup Test")
    print("="*60 + "\n")

    success = test_data_download_and_cache()

    print("\n" + "="*60)
    if success:
        print("✓ All tests passed! Dashboard is ready to run.")
        print("\nStart the dashboard with:")
        print("  streamlit run src/dashboard.py")
    else:
        print("✗ Some tests failed. Check the errors above.")
    print("="*60 + "\n")
