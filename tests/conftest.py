import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_dir)

from PMF_toolkits import PMF

# Use non-interactive backend for testing
matplotlib.use('Agg')

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    df = pd.DataFrame({
        'PM10': np.random.normal(20, 5, len(dates)),
        'OC': np.random.normal(5, 1, len(dates)),
        'EC': np.random.normal(2, 0.5, len(dates))
    }, index=dates)
    
    return df

@pytest.fixture
def test_pmf():
    """Create a test PMF instance"""
    pmf = PMF(site="test_site", reader="xlsx", BDIR="tests/data/test_site/")
    return pmf

@pytest.fixture
def mock_pmf(tmp_path):
    """Create a mock PMF directory structure with sample files"""
    # Create directory structure
    pmf_dir = tmp_path / "pmf_outputs"
    pmf_dir.mkdir()
    
    # Create dummy files
    (pmf_dir / "site_base.xlsx").touch()
    (pmf_dir / "site_Constrained.xlsx").touch()
    (pmf_dir / "site_boot.xlsx").touch()
    
    return str(pmf_dir)

@pytest.fixture
def pmf_fixture(tmp_path):
    """
    Provides a PMF object with mock data for testing.
    Debug: If error, check that 'tmp_path' is valid and PMF is initialized.
    """
    pmf_obj = PMF(
        site="test_site",
        reader=None,  # Not reading from file system in this fixture
        BDIR=str(tmp_path)
    )
    # Mock vital attributes to avoid reading actual data
    pmf_obj.totalVar = "PM2.5"
    pmf_obj.profiles = ["Factor1", "Factor2"]
    pmf_obj.species = ["PM2.5", "NO3-", "SO4--"]
    return pmf_obj

@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    return pd.DataFrame({
        'Date': dates,
        'Value': np.random.rand(len(dates))
    })

@pytest.fixture
def pmf_fixture():
    """Create a simplified PMF object for testing"""
    # Create a dummy PMF object
    pmf = PMF(site="test_site", savedir="./")
    
    # Set up basic attributes
    pmf.totalVar = "PM2.5"
    pmf.profiles = ["Factor1", "Factor2", "Factor3"]
    pmf.species = ["PM2.5", "OC", "EC"]
    
    # Create mock data for profiles and contributions
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    
    # Mock profiles - 3 species x 3 factors
    pmf.dfprofiles_b = pd.DataFrame(
        np.random.rand(3, 3),
        index=["PM2.5", "OC", "EC"], 
        columns=["Factor1", "Factor2", "Factor3"]
    )
    
    # Make sure PM2.5 has reasonable values
    pmf.dfprofiles_b.loc["PM2.5"] = np.array([0.8, 0.7, 0.6])
    
    # Mock contribution - 10 days x 3 factors
    pmf.dfcontrib_b = pd.DataFrame(
        np.random.rand(10, 3),
        index=dates,
        columns=["Factor1", "Factor2", "Factor3"]
    )
    
    # Copy base data to constrained
    pmf.dfprofiles_c = pmf.dfprofiles_b.copy()
    pmf.dfcontrib_c = pmf.dfcontrib_b.copy()
    
    return pmf

@pytest.fixture
def test_data_single_site(tmp_path):
    """Create test data files for single site testing."""
    site_dir = tmp_path / "single_site"
    site_dir.mkdir()
    
    # Create profiles data
    profiles_data = pd.DataFrame({
        'Factor1': [1.0, 0.5, 0.3],
        'Factor2': [0.8, 0.4, 0.2]
    }, index=['PM10', 'OC', 'EC'])
    
    # Create contributions data
    dates = pd.date_range('2023-01-01', periods=10)
    contrib_data = pd.DataFrame({
        'Factor1': np.random.rand(10),
        'Factor2': np.random.rand(10)
    }, index=dates)
    
    # Save test files
    profiles_data.to_excel(site_dir / "GRE-fr_base.xlsx")
    contrib_data.to_excel(site_dir / "GRE-fr_Constrained.xlsx")
    
    return str(site_dir)
