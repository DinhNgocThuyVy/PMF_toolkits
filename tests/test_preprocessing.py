import pytest
import pandas as pd
import numpy as np
from PMF_toolkits import PMFPreprocessor
import os

@pytest.fixture
def test_data():
    """Create test DataFrame with real data structure."""
    # Create a simplified version matching your real data structure
    data = {
        'Date': pd.date_range(start='2020-11-12', periods=10, freq='3D'),
        'PM10recons': [23.31, 25.51, 12.98, 17.57, 10.61, 43.11, 6.79, 12.72, 7.97, 9.21],
        'OC*': [6.38, 8.27, 4.69, 6.53, 3.14, 16.30, 1.89, 3.57, 2.60, 0.88],
        'EC': [0.90, 1.17, 0.83, 1.32, 0.43, 2.34, 0.23, 0.32, 0.29, 0.14]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@pytest.fixture
def test_ql_values():
    """Define quantification limits based on real data."""
    # Define realistic QL values for the columns in test_data
    return {
        'PM10recons': 0.1,
        'OC*': 0.05,
        'EC': 0.02
    }

def test_preprocessor_initialization(test_data, test_ql_values):
    """Test PMFPreprocessor initialization."""
    preprocessor = PMFPreprocessor(test_data, test_ql_values)
    assert preprocessor.data is not None
    assert preprocessor.ql_values is not None
    assert isinstance(preprocessor.data, pd.DataFrame)
    assert isinstance(preprocessor.ql_values, dict)

def test_track_quantification_limits(test_data, test_ql_values):
    """Test detection of below-QL values."""
    # Create a preprocessor with our test data
    preprocessor = PMFPreprocessor(test_data, test_ql_values)
    
    # Add some values below QL to test data
    test_data_with_ql = test_data.copy()
    test_data_with_ql.loc[test_data_with_ql.index[0], 'OC*'] = '<QL'
    test_data_with_ql.loc[test_data_with_ql.index[1], 'EC'] = -999  # Common QL marker
    
    # Create new preprocessor with this modified data
    preprocessor_with_ql = PMFPreprocessor(test_data_with_ql, test_ql_values)
    
    # Track QL values
    ql_mask = preprocessor_with_ql.track_quantification_limits()
    
    # Verify mask shape matches data
    assert isinstance(ql_mask, pd.DataFrame)
    assert ql_mask.shape == test_data_with_ql.shape
    
    # Verify QL markers were detected
    assert ql_mask.loc[test_data_with_ql.index[0], 'OC*'] == True
    assert ql_mask.loc[test_data_with_ql.index[1], 'EC'] == True

def test_convert_to_numeric(test_data, test_ql_values):
    """Test conversion to numeric values."""
    preprocessor = PMFPreprocessor(test_data, test_ql_values)
    preprocessor.track_quantification_limits()  # Needed to identify <QL values
    numeric_data = preprocessor.convert_to_numeric()
    assert all(pd.api.types.is_numeric_dtype(numeric_data[col]) for col in numeric_data.columns)
    assert not numeric_data.isnull().all().all()
    
    # Check that values below detection limit are replaced with QL/2
    # Example with known below QL values
    below_ql_values = preprocessor.ql_mask.sum(axis=0)
    if below_ql_values.any():
        col_with_ql = below_ql_values[below_ql_values > 0].index[0]
        row_with_ql = preprocessor.ql_mask[col_with_ql][preprocessor.ql_mask[col_with_ql]].index[0]
        assert numeric_data.loc[row_with_ql, col_with_ql] == test_ql_values.get(col_with_ql, 0.001) / 2
    
def test_compute_uncertainties(test_data, test_ql_values):
    """Test uncertainty calculation methods."""
    preprocessor = PMFPreprocessor(test_data, test_ql_values)
    
    # Test percentage method
    unc_pct = preprocessor.compute_uncertainties(
        method="percentage", 
        params={"percentage": 0.1}
    )
    assert isinstance(unc_pct, pd.DataFrame)
    assert unc_pct.shape == test_data.shape
    
    # Verify specific uncertainty calculation
    numeric_data = preprocessor.convert_to_numeric()
    # Check a known value's uncertainty
    col = 'PM10recons'
    row_idx = numeric_data.index[0]
    expected_unc = numeric_data.loc[row_idx, col] * 0.1
    assert np.isclose(unc_pct.loc[row_idx, col], expected_unc)
    
    # Test DL method
    unc_dl = preprocessor.compute_uncertainties(
        method="DL", 
        params={"DL": test_ql_values}
    )
    assert isinstance(unc_dl, pd.DataFrame)
    assert unc_dl.shape == test_data.shape

def test_handle_missing_values(test_data, test_ql_values):
    """Test missing value handling methods."""
    # Create data with missing values
    data_with_na = test_data.copy()
    data_with_na.loc[data_with_na.index[0], 'OC*'] = np.nan
    data_with_na.loc[data_with_na.index[1:3], 'EC'] = np.nan
    
    preprocessor = PMFPreprocessor(data_with_na, test_ql_values)
    
    # Test interpolation
    filled_interp = preprocessor.handle_missing_values(method="interpolate", 
                                                     data=data_with_na)
    assert isinstance(filled_interp, pd.DataFrame)
    assert filled_interp.isnull().sum().sum() < data_with_na.isnull().sum().sum()
    
    # Test mean filling
    filled_mean = preprocessor.handle_missing_values(method="mean",
                                                   data=data_with_na)
    assert isinstance(filled_mean, pd.DataFrame)
    assert filled_mean.isnull().sum().sum() == 0

def test_data_summary(test_data, test_ql_values):
    """Test data quality summary generation."""
    # Create data with both missing values and QL values
    data_with_issues = test_data.copy()
    data_with_issues.loc[data_with_issues.index[0], 'OC*'] = '<QL'
    data_with_issues.loc[data_with_issues.index[1], 'EC'] = np.nan
    
    preprocessor = PMFPreprocessor(data_with_issues, test_ql_values)
    
    # Force tracking of QL values first
    preprocessor.track_quantification_limits()
    
    # Get data quality summary
    summary = preprocessor.summarize_data_quality()
    
    assert isinstance(summary, pd.DataFrame)
    assert "Missing" in summary.index
    assert "Below QL" in summary.index
    
    # Verify counts
    assert summary.loc["Missing", "EC"] > 0
    assert summary.loc["Below QL", "OC*"] > 0

def test_normalize_to_total(test_data, test_ql_values):
    """Test normalization to a total variable."""
    preprocessor = PMFPreprocessor(test_data, test_ql_values)
    normalized = preprocessor.normalize_to_total("PM10recons")
    assert isinstance(normalized, pd.DataFrame)
    
    # Check that values are properly normalized
    row_idx = test_data.index[0]
    col = 'OC*'
    expected_ratio = test_data.loc[row_idx, col] / test_data.loc[row_idx, 'PM10recons']
    assert np.isclose(normalized.loc[row_idx, col], expected_ratio)
    
    # All normalized PM10recons values should be 1
    assert all(np.isclose(normalized['PM10recons'], 1.0))
