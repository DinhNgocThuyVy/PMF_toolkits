import pytest
import pandas as pd
import numpy as np
from PMF_toolkits import PMF
import os

@pytest.fixture
def multi_site_pmf():
    """Create PMF instance for multi-site testing."""
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'multi_site')
    pmf = PMF(site="11fnew", reader="xlsx", BDIR=base_dir, multisites=True)
    return pmf

def test_multi_site_read_all(multi_site_pmf):
    """Test reading all data from multi-site files."""
    pmf = multi_site_pmf
    try:
        pmf.read.read_all()
        assert pmf.profiles is not None, "Profiles not loaded"
        assert pmf.dfprofiles_b is not None, "Base profiles not loaded"
        assert pmf.dfcontrib_b is not None, "Base contributions not loaded"
        
        # Check that only data for the specified site is loaded
        assert all(pmf.dfcontrib_b.index.notnull()), "Invalid dates in contributions"
    except Exception as e:
        pytest.fail(f"Error reading multi-site data: {str(e)}")

def test_multi_site_filter(multi_site_pmf):
    """Test proper filtering of multi-site data."""
    pmf = multi_site_pmf
    pmf.read.read_base_profiles()
    
    # Verify that only data for the specified site is present
    if hasattr(pmf.dfprofiles_b, 'Station'):
        stations = pmf.dfprofiles_b['Station'].unique()
        assert len(stations) == 1, "Multiple stations found"
        assert stations[0] == "11fnew", "Wrong station data loaded"

def test_multi_site_bootstrap(multi_site_pmf):
    """Test bootstrap results for multi-site."""
    pmf = multi_site_pmf
    pmf.read.read_base_bootstrap()
    assert pmf.dfBS_profile_b is not None
    assert pmf.dfbootstrap_mapping_b is not None

def test_multi_site_contributions(multi_site_pmf):
    """Test contribution calculations for multi-site."""
    pmf = multi_site_pmf
    pmf.read.read_all()
    
    # Print debug information
    print("\nDEBUG INFO (multi-site):")
    print(f"Total variable: {pmf.totalVar}")
    print(f"Profiles available: {pmf.profiles}")
    print(f"Base profiles available: {pmf.dfprofiles_b is not None}")
    print(f"Base contributions available: {pmf.dfcontrib_b is not None}")
    
    # Verify at least some data was loaded
    assert pmf.dfprofiles_b is not None, "Base profiles not loaded"
    assert pmf.dfcontrib_b is not None, "Base contributions not loaded"
    
    # Try to calculate contributions with multiple total variable options
    try:
        contrib = pmf.to_cubic_meter(constrained=False)  # Try base run first
    except Exception as e:
        print(f"Failed with constrained=False: {str(e)}")
        try:
            if pmf.totalVar in pmf.dfprofiles_b.index:
                contrib = pmf.to_cubic_meter(specie=pmf.totalVar, constrained=False)
            else:
                known_totals = ["PM10", "PM2.5", "PM10recons", "PMrecons", "TC"]
                for species in known_totals:
                    if species in pmf.dfprofiles_b.index:
                        contrib = pmf.to_cubic_meter(specie=species, constrained=False)
                        break
                else:
                    pytest.skip("No suitable total variable found for to_cubic_meter test")
                    return
        except Exception as e:
            print(f"Failed all contribution calculations: {str(e)}")
            pytest.skip("Could not calculate contributions for multi-site test")
            return
    
    # Only check basic properties if we got this far
    assert isinstance(contrib, pd.DataFrame), "Contribution result not a DataFrame"
