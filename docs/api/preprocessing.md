# Core Module Documentation

This section provides detailed documentation for the core PMF module.

## Mathematical Background

### PMF Model

The PMF model is based on the following equation:

::: PMF_toolkits.PMF
    options:
      show_root_heading: true
      show_source: true

# Preprocessing Module Documentation

## Overview

The Preprocessing module provides tools for preparing environmental data for PMF analysis, including uncertainty calculation, missing value handling, and data quality assessment.

## Mathematical Background

### Uncertainty Calculation Methods

1. **Percentage Method**
   - σᵢⱼ = p × xᵢⱼ
   - p is the relative uncertainty (e.g., 0.1 for 10%)
   - Used when measurement uncertainty is proportional to concentration

2. **Detection Limit Method**
   - For values > MDL: σᵢⱼ = √((p × xᵢⱼ)² + (MDL/3)²)
   - For values ≤ MDL: σᵢⱼ = 5/6 × MDL
   - MDL is the Method Detection Limit
   - p is the relative uncertainty

3. **Combined Method**
   - σᵢⱼ = √((p × xᵢⱼ)² + q²)
   - p is the multiplicative component
   - q is the additive component

### Signal-to-Noise Ratio

The signal-to-noise ratio (S/N) helps identify species quality:
- S/N = √(Σ(xᵢⱼ)²/Σ(σᵢⱼ)²)
- Strong variables: S/N > 2
- Weak variables: 0.2 < S/N ≤ 2
- Bad variables: S/N ≤ 0.2

## PMFPreprocessor Class

### Methods

#### compute_uncertainties
```python
def compute_uncertainties(self,
                        data: pd.DataFrame,
                        method: str = "percentage",
                        params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate uncertainties for PMF input data.

    Parameters
    ----------
    data : pd.DataFrame
        Concentration data matrix
    method : str
        Calculation method:
        - "percentage": Simple percentage of concentration
        - "dl": Detection limit based
        - "combined": Combined percentage and additive
    params : dict, optional
        Method-specific parameters:
        - percentage: {"p": relative_uncertainty}
        - dl: {"mdl": detection_limits, "p": relative_uncertainty}
        - combined: {"p": mult_component, "q": add_component}

    Returns
    -------
    pd.DataFrame
        Matrix of calculated uncertainties
    """
```

#### handle_missing_values
```python
def handle_missing_values(self,
                        data: pd.DataFrame,
                        method: str = "interpolate",
                        max_gap: int = 3) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Input data matrix
    method : str
        Handling method:
        - "interpolate": Linear interpolation
        - "mean": Replace with species mean
        - "median": Replace with species median
        - "remove": Remove samples with missing values
    max_gap : int
        Maximum gap size for interpolation

    Returns
    -------
    pd.DataFrame
        Data with handled missing values
    """
```

#### filter_species
```python
def filter_species(self,
                  data: pd.DataFrame,
                  uncertainties: pd.DataFrame,
                  min_sn_ratio: float = 0.2,
                  min_valid: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter species based on quality criteria.

    Parameters
    ----------
    data : pd.DataFrame
        Concentration data
    uncertainties : pd.DataFrame
        Uncertainty estimates
    min_sn_ratio : float
        Minimum signal-to-noise ratio
    min_valid : float
        Minimum fraction of valid observations

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered data and uncertainties
    """
```

#### compute_signal_to_noise
```python
def compute_signal_to_noise(self,
                          data: pd.DataFrame,
                          uncertainties: pd.DataFrame) -> pd.Series:
    """
    Calculate signal-to-noise ratio for each species.

    Parameters
    ----------
    data : pd.DataFrame
        Concentration data
    uncertainties : pd.DataFrame
        Uncertainty estimates

    Returns
    -------
    pd.Series
        Signal-to-noise ratios by species
    """
```

## Examples

### Basic Data Preparation
```python
import pandas as pd
from PMF_toolkits import PMFPreprocessor

# Initialize preprocessor
prep = PMFPreprocessor()

# Load data
data = pd.read_csv("concentrations.csv")

# Calculate uncertainties (10% of concentration)
uncertainties = prep.compute_uncertainties(
    data,
    method="percentage",
    params={"p": 0.1}
)

# Handle missing values
data_clean = prep.handle_missing_values(
    data,
    method="interpolate",
    max_gap=3
)

# Filter species
data_filtered, unc_filtered = prep.filter_species(
    data_clean,
    uncertainties,
    min_sn_ratio=0.2,
    min_valid=0.75
)
```

### Advanced Usage with Detection Limits
```python
# Define detection limits
dl_values = {
    "As": 0.1,
    "Cd": 0.05,
    "Pb": 0.2
}

# Calculate uncertainties using DL method
uncertainties = prep.compute_uncertainties(
    data,
    method="dl",
    params={
        "mdl": dl_values,
        "p": 0.1
    }
)

# Check data quality
sn_ratios = prep.compute_signal_to_noise(data, uncertainties)
print("\nSignal-to-noise ratios:")
print(sn_ratios.sort_values(ascending=False))
```

## Best Practices

1. **Data Quality Assessment**
   - Check data completeness
   - Identify outliers
   - Calculate signal-to-noise ratios
   - Document all quality control steps

2. **Uncertainty Calculation**
   - Choose appropriate method based on data quality
   - Consider detection limits when available
   - Document assumptions and parameters

3. **Missing Value Treatment**
   - Use interpolation for short gaps
   - Consider removing samples with too many missing values
   - Document handling strategy

4. **Species Selection**
   - Remove species with poor S/N ratios
   - Consider down-weighting weak variables
   - Maintain sufficient variables for source identification

## References

1. Polissar, A.V., et al., 1998. The aerosol at Barrow, Alaska: long-term trends
   and source locations. Atmospheric Environment 32, 2441-2458.

2. Reff, A., Eberly, S.I., Bhave, P.V., 2007. Receptor modeling of ambient
   particulate matter data using positive matrix factorization: Review of
   existing methods. Journal of the Air & Waste Management Association 57,
   146-154.

::: PMF_toolkits.preprocessing.PMFPreprocessor
    handler: python
    selection:
      members:
        - compute_uncertainties
        - handle_missing_values
        - filter_species
        - compute_signal_to_noise
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2
