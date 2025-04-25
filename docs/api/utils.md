# Utils Module Documentation

## Overview

The Utils module provides helper functions for PMF analysis, including season calculation, color mapping, and axis formatting. These utilities support the core functionality of the package.

## Functions

### add_season

```python
def add_season(dates: Union[pd.DatetimeIndex, pd.Series],
              scheme: str = "standard") -> pd.Series:
    """
    Add season information to datetime data.

    Parameters
    ----------
    dates : DatetimeIndex or Series
        Dates to classify into seasons
    scheme : str, default="standard"
        Season classification scheme:
        - "standard": Meteorological seasons
        - "two": Summer/Winter split
        - "monthly": Using month names

    Returns
    -------
    pd.Series
        Season labels for each date
    """
```

### get_sourcesCategories

```python
def get_sourcesCategories() -> Dict[str, str]:
    """
    Get standard source category mappings.

    Returns
    -------
    dict
        Mapping of source names to categories
        Example: {"Traffic": "Mobile", "Industry": "Industrial"}
    """
```

### get_sourceColor
```python
def get_sourceColor(sources: List[str], palette: str = "Set3") -> Dict[str, str]:
    """
    Get color mapping for source categories.

    Parameters
    ----------
    sources : list of str
        Source names to map
    palette : str, default="Set3"
        Color palette name from seaborn

    Returns
    -------
    dict
        Mapping of sources to hex colors
    """
```

### format_xaxis_timeseries

```python
def format_xaxis_timeseries(ax: plt.Axes,
                          date_format: str = "%Y-%m",
                          rotation: int = 45) -> None:
    """
    Format time series x-axis for better readability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format
    date_format : str, default="%Y-%m"
        Date format string
    rotation : int, default=45
        Tick label rotation
    """
```

### pretty_specie

```python
def pretty_specie(specie: str) -> str:
    """
    Format species names with proper subscripts/superscripts.

    Parameters
    ----------
    specie : str
        Species name to format

    Returns
    -------
    str
        Formatted species name for plotting
    """
```

### compute_similarity_metrics

```python
def compute_similarity_metrics(profile1: pd.Series,
                            profile2: pd.Series,
                            method: str = "all",
                            normalize: bool = True) -> Dict[str, float]:
    """
    Compute similarity metrics between profiles.

    Parameters
    ----------
    profile1, profile2 : pd.Series
        Profiles to compare
    method : str, default="all"
        Metrics to compute: "PD", "SID", "COD", or "all"
    normalize : bool, default=True
        Whether to normalize profiles before comparison

    Returns
    -------
    dict
        Computed similarity metrics
    """
```

## Examples

### Season Classification

```python
import pandas as pd
from PMF_toolkits.utils import add_season

# Create date range
dates = pd.date_range("2020-01-01", "2020-12-31")

# Add seasons
seasons = add_season(dates, scheme="standard")
print(seasons.value_counts())

# Using two-season scheme
seasons_binary = add_season(dates, scheme="two")
print(seasons_binary.value_counts())
```

### Source Colors

```python
from PMF_toolkits.utils import get_sourceColor

# Define sources
sources = ["Traffic", "Industry", "Dust", "Sea Salt"]

# Get color mapping
colors = get_sourceColor(sources, palette="Set2")

# Use in plotting
for source, color in colors.items():
    plt.plot(data[source], color=color, label=source)
```

### Time Series Formatting

```python
import matplotlib.pyplot as plt
from PMF_toolkits.utils import format_xaxis_timeseries

# Create time series plot
fig, ax = plt.subplots()
ax.plot(dates, values)

# Format x-axis
format_xaxis_timeseries(ax, date_format="%Y-%m-%d", rotation=30)
plt.show()
```

### Species Name Formatting

```python
from PMF_toolkits.utils import pretty_specie

# Format species names
print(pretty_specie("SO4-2"))  # Returns "SO₄²⁻"
print(pretty_specie("NH4+"))   # Returns "NH₄⁺"
```

## Best Practices

1. **Season Classification**
   - Choose appropriate scheme for analysis
   - Consider local climate patterns
   - Document scheme choice in results

2. **Color Selection**
   - Use colorblind-friendly palettes
   - Maintain consistent colors
   - Consider print/digital requirements

3. **Time Series Presentation**
   - Use appropriate date formats
   - Ensure readable tick labels
   - Consider data density

4. **Species Naming**
   - Use consistent formatting
   - Include units where appropriate
   - Follow domain conventions

## Common Patterns

### Working with Seasons

```python
# Add seasons and group data
data["Season"] = add_season(data.index)
seasonal_means = data.groupby("Season").mean()

# Custom season order
season_order = ["Winter", "Spring", "Summer", "Fall"]
seasonal_means = seasonal_means.reindex(season_order)
```

### Source Category Management

```python
# Get standard categories
categories = get_sourcesCategories()

# Map sources to categories
data["Category"] = data["Source"].map(categories)

# Get colors by category
category_colors = get_sourceColor(data["Category"].unique())
```

### Time Series Plotting

```python
# Create multi-panel time series
fig, axes = plt.subplots(3, 1, sharex=True)

for ax, var in zip(axes, variables):
    ax.plot(dates, data[var])
    format_xaxis_timeseries(ax)

plt.tight_layout()
```

::: PMF_toolkits.utils
    handler: python
    selection:
      members:
        - add_season
        - get_sourcesCategories
        - get_sourceColor
        - format_xaxis_timeseries
        - pretty_specie
        - compute_similarity_metrics
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2
