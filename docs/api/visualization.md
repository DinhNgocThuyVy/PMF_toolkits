# Visualization Module Documentation

## Overview

The Visualization module provides comprehensive plotting capabilities for PMF results, including factor profiles, time series, seasonal patterns, and diagnostic plots. It uses matplotlib and seaborn for high-quality visualizations.

## PMFVisualization Class

### Methods

#### plot_factor_profiles

```python
def plot_factor_profiles(self,
                        constrained: bool = True,
                        normalize: bool = True,
                        uncertainties: bool = True,
                        style: str = "bar") -> plt.Figure:
    """
    Plot factor profiles with optional uncertainties.

    Parameters
    ----------
    constrained : bool, default=True
        Use constrained or base run profiles
    normalize : bool, default=True
        Normalize profiles to 100%
    uncertainties : bool, default=True
        Show uncertainty bars from bootstrap analysis
    style : str
        Plot style: "bar" or "box" for bootstrap results

    Returns
    -------
    plt.Figure
        Figure containing the profile plots
    """
```

#### plot_time_series

```python
def plot_time_series(self,
                    factors: Optional[List[str]] = None,
                    normalize: bool = False,
                    rolling_mean: Optional[int] = None) -> plt.Figure:
    """
    Plot factor contribution time series.

    Parameters
    ----------
    factors : list of str, optional
        Specific factors to plot, all if None
    normalize : bool, default=False
        Show relative contributions
    rolling_mean : int, optional
        Window size for rolling average

    Returns
    -------
    plt.Figure
        Figure containing the time series plots
    """
```

#### plot_seasonal_patterns

```python
def plot_seasonal_patterns(self,
                         factors: Optional[List[str]] = None,
                         style: str = "box") -> plt.Figure:
    """
    Plot seasonal patterns of factor contributions.

    Parameters
    ----------
    factors : list of str, optional
        Specific factors to plot
    style : str
        Plot style: "box" or "violin"

    Returns
    -------
    plt.Figure
        Figure containing the seasonal pattern plots
    """
```

#### plot_diagnostics

```python
def plot_diagnostics(self,
                    plot_type: str = "residuals") -> plt.Figure:
    """
    Create diagnostic plots for PMF solution.

    Parameters
    ----------
    plot_type : str
        Type of diagnostic plot:
        - "residuals": Scaled residuals distribution
        - "obs_pred": Observed vs predicted
        - "q_contribution": Q value contributions
        - "species_fit": R² by species

    Returns
    -------
    plt.Figure
        Figure containing diagnostic plots
    """
```

#### plot_factor_fingerprints

```python
def plot_factor_fingerprints(self,
                           constrained: bool = True,
                           normalize: bool = True) -> plt.Figure:
    """
    Plot characteristic fingerprints for each factor.

    Parameters
    ----------
    constrained : bool, default=True
        Use constrained or base profiles
    normalize : bool, default=True
        Normalize to total mass

    Returns
    -------
    plt.Figure
        Figure with factor fingerprints
    """
```

## Examples

### Basic Factor Profile Plot

```python
from PMF_toolkits import PMF

# Initialize PMF
pmf = PMF(site="urban_site", reader="xlsx")
pmf.read.read_all()

# Create basic profile plot
fig = pmf.visualization.plot_factor_profiles(
    normalize=True,
    uncertainties=True
)
plt.show()
```

### Time Series with Rolling Mean

```python
# Plot time series with 7-day rolling mean
fig = pmf.visualization.plot_time_series(
    factors=["Traffic", "Industry", "Dust"],
    rolling_mean=7
)
plt.show()
```

### Seasonal Analysis

```python
# Plot seasonal patterns using violin plots
fig = pmf.visualization.plot_seasonal_patterns(
    style="violin"
)
plt.show()
```

### Model Diagnostics

```python
# Plot residuals distribution
fig = pmf.visualization.plot_diagnostics(
    plot_type="residuals"
)
plt.show()

# Plot R² by species
fig = pmf.visualization.plot_diagnostics(
    plot_type="species_fit"
)
plt.show()
```

## Plot Customization

### Style Guide

```python
# Import visualization module
from PMF_toolkits import PMFVisualization

# Set default style
viz = PMFVisualization(pmf, style="seaborn")

# Custom colors for factors
viz.set_factor_colors({
    "Traffic": "#1f77b4",
    "Industry": "#ff7f0e",
    "Dust": "#2ca02c"
})

# Create plot with custom style
fig = viz.plot_factor_profiles(
    figsize=(12, 6),
    dpi=300
)
```

### Advanced Layout

```python
# Create multi-panel figure
fig = viz.plot_combined_analysis(
    panels=["profiles", "timeseries", "seasonal"],
    layout=(2, 2),
    suptitle="PMF Analysis Overview"
)

# Adjust layout
plt.tight_layout()
plt.show()
```

## Plot Types

1. **Factor Profiles**
   - Bar plots with uncertainties
   - Stacked profiles
   - Radar plots for fingerprints

2. **Time Series**
   - Individual factor contributions
   - Stacked contributions
   - Rolling averages

3. **Seasonal Patterns**
   - Box plots by season
   - Violin plots
   - Monthly averages

4. **Diagnostics**
   - Residuals distribution
   - Q/Qexp contributions
   - R² by species
   - Obs vs. predicted

## Best Practices

1. **Data Presentation**
   - Use consistent scales
   - Include uncertainty estimates
   - Label axes clearly
   - Use colorblind-friendly palettes

2. **Figure Layout**
   - Consider publication requirements
   - Use consistent font sizes
   - Include legends
   - Add descriptive titles

3. **Export Settings**
   - Use high DPI for publication
   - Save in vector formats (PDF/SVG)
   - Maintain aspect ratios
   - Consider file size limits

::: PMF_toolkits.visualization.PMFVisualization
    handler: python
    selection:
      members:
        - plot_factor_profiles
        - plot_time_series
        - plot_seasonal_patterns
        - plot_diagnostics
        - plot_factor_fingerprints
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2
