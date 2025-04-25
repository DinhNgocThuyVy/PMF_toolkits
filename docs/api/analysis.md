# Analysis Module Documentation

## Overview

The Analysis module provides statistical analysis tools for PMF results, including factor profile analysis, uncertainty estimation, and similarity metrics calculation.

## Mathematical Background

### Similarity Metrics

The module implements several similarity metrics for comparing factor profiles:

1. **Pearson Distance (PD)**
   - PD = 1 - r²
   - r is the Pearson correlation coefficient
   - Ranges from 0 (identical) to 1 (uncorrelated)

2. **Standardized Identity Distance (SID)**
   - SID = √2/n * Σ|xᵢ-yᵢ|/(xᵢ+yᵢ)
   - n is the number of species
   - Ranges from 0 (identical) to 1 (completely different)

3. **Coefficient of Divergence (COD)**
   - COD = √(1/n * Σ((xᵢ-yᵢ)/(xᵢ+yᵢ))²)
   - Measures the degree of divergence between profiles

### Uncertainty Estimation

The module supports multiple approaches to uncertainty estimation:

1. **Bootstrap Analysis**
   - Resamples the data with replacement
   - Provides confidence intervals
   - Assesses factor stability

2. **Displacement (DISP)**
   - Tests factor stability by perturbing the solution
   - Identifies rotational ambiguity

3. **BS-DISP Combined**
   - Combines bootstrap and displacement results
   - Provides comprehensive uncertainty estimates

## PMFAnalysis Class

### Methods

#### analyze_factor_profiles
```python
def analyze_factor_profiles(self,
                          method: str = "correlation",
                          external_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Analyze factor profiles using various methods.

    Parameters
    ----------
    method : str
        Analysis method:
        - "correlation": Pearson correlation between profiles
        - "similarity": Cosine similarity between profiles 
        - "comparison": Compare with external reference profiles
    external_data : pd.DataFrame, optional
        Reference profiles for comparison

    Returns
    -------
    dict
        Analysis results containing matrices and statistics
    """
```

#### estimate_uncertainties
```python
def estimate_uncertainties(self,
                         method: str = "bootstrap",
                         n_iterations: int = 1000) -> Dict:
    """
    Estimate uncertainties in PMF results.

    Parameters
    ----------
    method : str
        Method to use:
        - "bootstrap": Bootstrap resampling
        - "displacement": DISP analysis
        - "combined": BS-DISP combined
    n_iterations : int, default=1000
        Number of bootstrap iterations
    
    Returns
    -------
    dict
        Uncertainty estimates containing:
        - Standard errors
        - Confidence intervals
        - Method-specific statistics
    """
```

#### compute_profile_similarity
```python
def compute_profile_similarity(self,
                            df1: pd.DataFrame,
                            df2: pd.DataFrame,
                            factor1: Optional[str] = None,
                            factor2: Optional[str] = None,
                            method: str = "correlation",
                            normalize_to_mass: bool = True) -> Union[float, Tuple[float, float]]:
    """
    Compute similarity between factor profiles using various metrics.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Profile matrices to compare
    factor1, factor2 : str, optional
        Specific factors to compare
    method : str
        Similarity metric to use ("correlation", "SID", "both")
    normalize_to_mass : bool, default=True
        Whether to normalize profiles to total mass fraction
    
    Returns
    -------
    float or tuple
        Similarity metric(s) between profiles
    """
```

## Module-Level Functions

### compute_Q_values
```python
def compute_Q_values(X: np.ndarray,
                    G: np.ndarray,
                    F: np.ndarray,
                    S: np.ndarray) -> Dict[str, float]:
    """
    Compute Q values (scaled residuals) for PMF solution.
    
    Parameters
    ----------
    X : np.ndarray
        Original data matrix
    G : np.ndarray
        Factor contributions matrix
    F : np.ndarray
        Factor profiles matrix
    S : np.ndarray
        Uncertainties matrix
    
    Returns
    -------
    Dict[str, float]
        Q, Q_expected, and Q/Q_expected values
    """
```

### compute_distance_matrix
```python
def compute_distance_matrix(df: pd.DataFrame,
                          metric: str = 'SID',
                          normalize: bool = True) -> pd.DataFrame:
    """
    Compute distance matrix between profiles using selected metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing profiles as columns
    metric : str
        Distance metric to use ('SID', 'PD', 'COD')
    normalize : bool
        Whether to normalize profiles before computing distances
    
    Returns
    -------
    pd.DataFrame
        Distance matrix between profiles
    """
```

## Examples

### Basic Profile Analysis
```python
pmf = PMF(site="urban_site", reader="xlsx")
pmf.read.read_all()

# Analyze correlations between factors
results = pmf.analysis.analyze_factor_profiles(method="correlation")

# Print correlation matrix
print(results['correlation_matrix'])
```

### Uncertainty Estimation
```python
# Get bootstrap uncertainties
uncertainties = pmf.analysis.estimate_uncertainties(
    method="bootstrap",
    n_iterations=1000
)

# Print confidence intervals
print(uncertainties['bootstrap_stats'])
```

### Profile Comparison
```python
# Compare two factor profiles
similarity = pmf.analysis.compute_profile_similarity(
    df1=pmf.dfprofiles_b,
    df2=reference_profiles,
    factor1="Factor 1",
    factor2="Reference A",
    method="both"
)

print(f"PD: {similarity[0]:.3f}, SID: {similarity[1]:.3f}")
```

::: PMF_toolkits.analysis.PMFAnalysis
    handler: python
    selection:
      members:
        - analyze_factor_profiles
        - compute_similarity_metrics
        - estimate_uncertainties
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2
