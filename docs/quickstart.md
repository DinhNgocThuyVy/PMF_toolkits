# Quickstart Guide

## Prerequisites

Before starting, ensure you have:
- EPA PMF 5.0 output files (Excel format)
- Python 3.8 or higher
- The `PMF_toolkits` package installed

```bash
pip install git+https://github.com/DinhNgocThuyVy/PMF_toolkits.git
```

## Step 1: Initialize PMF Object

The first step is to create a PMF object by specifying your site name and the location of your PMF output files.

```python
from PMF_toolkits import PMF

# Initialize PMF object
pmf = PMF(
    site="your_site_name",  # Name of your site/dataset
    reader="xlsx",          # Type of reader ("xlsx" or "sql")
    BDIR="path/to/files",   # Directory containing PMF output files
    multisites=False        # Set to True for multi-site analysis
)
```

## Step 2: Load Data

Load the data from your PMF output files:

```python
# Load all available data
pmf.read.read_all()

# Alternatively, load specific components
pmf.read.read_base_profiles()           # Load base run profiles
pmf.read.read_constrained_profiles()    # Load constrained run profiles
pmf.read.read_base_contributions()      # Load base run contributions
pmf.read.read_constrained_contributions()  # Load constrained run contributions
pmf.read.read_base_bootstrap()          # Load bootstrap results
pmf.read.read_base_uncertainties_summary()  # Load uncertainty summary
```

After loading, verify that the data was loaded correctly:

```python
# Check if data was loaded successfully
pmf.ensure_data_loaded()

# Examine the available data
print(f"Profiles: {pmf.profiles}")
print(f"Number of factors: {pmf.nprofiles}")
print(f"Species: {pmf.species[:5]}...")  # First 5 species
print(f"Total variable: {pmf.totalVar}")
```

## Step 3: Basic Data Access

Access the loaded data directly:

```python
# Access factor profiles (constrained run)
print(pmf.dfprofiles_c.head())

# Access factor contributions (constrained run)
print(pmf.dfcontrib_c.head())

# Access bootstrap results (if available)
if pmf.dfBS_profile_c is not None:
    print(pmf.dfBS_profile_c.head())
```

## Step 4: Data Analysis

### Convert Contributions to μg/m³

```python
# Convert contributions to μg/m³ for the total variable
contributions = pmf.to_cubic_meter()
print(contributions.head())

# For a specific species
sulfate_contrib = pmf.to_cubic_meter(specie="SO4")
print(sulfate_contrib.head())
```

### Calculate Seasonal Contributions

```python
# Get seasonal contributions
seasonal_contrib = pmf.get_seasonal_contribution(
    annual=True,        # Include annual average
    normalize=True,     # Normalize to relative contribution
    constrained=True    # Use constrained run results
)
print(seasonal_contrib)
```

### Calculate Explained Variation

```python
# Calculate explained variation by each factor for each species
explained_var = pmf.analysis.explained_variation()
print(explained_var.head())
```

### Analyze Factor Temporal Correlation

```python
# Calculate correlation between factor time series
temporal_corr = pmf.analysis.factor_temporal_correlation()
print(temporal_corr)

# Detect potentially mixed factors
mixed_factors = pmf.analysis.detect_mixed_factors(threshold=0.6)
print(f"Potentially mixed factors: {mixed_factors}")
```

### Analyze Factor Profiles

```python
# Analyze factor profiles using correlation
factor_analysis = pmf.analysis.analyze_factor_profiles(method="correlation")

# Print correlation matrix
print(factor_analysis['correlation_matrix'])
```

### Calculate Similarity Metrics

```python
# Compare two factor profiles
profile1 = pmf.dfprofiles_c["Factor1"]
profile2 = pmf.dfprofiles_c["Factor2"]
similarity = pmf.analysis.compute_similarity_metrics(profile1, profile2)
print(f"Similarity metrics: {similarity}")
```

## Step 5: Visualization

PMF_toolkits provides extensive visualization capabilities through the `visualization` module (the `plot` attribute is maintained for backward compatibility).

### Plot Factor Profiles

```python
# Basic profile plot
fig = pmf.visualization.plot_factor_profiles(
    normalize=True,         # Normalize profiles
    uncertainties=True,     # Show uncertainty bars
    log_scale=True,         # Use logarithmic scale
    sort_species=True       # Sort species by value
)
plt.show()
```

### Plot Time Series Contributions

```python
# Plot time series with rolling mean
fig = pmf.visualization.plot_time_series(
    stacked=False,          # Line plot instead of stacked area
    rolling_mean=7          # 7-day rolling mean
)
plt.show()

# Plot stacked time series
fig = pmf.visualization.plot_time_series(stacked=True)
plt.show()
```

### Plot Seasonal Contribution Pattern

```python
# Plot seasonal contributions
fig = pmf.visualization.plot_seasonal_contribution(
    normalize=True,         # Show relative contributions
    annual=True             # Include annual average
)
plt.show()
```

### Plot Stacked Profiles

```python
# Plot species distribution across factors
fig = pmf.visualization.plot_stacked_profiles(constrained=True)
plt.show()
```

### Plot Sample Contributions

```python
# Plot contribution per sample over time
fig = pmf.visualization.plot_samples_sources_contribution(constrained=True)
plt.show()
```

### Plot Per Microgram Profile

```python
# Plot per microgram profile
fig = pmf.visualization.plot_per_microgram(constrained=True)
plt.show()
```

### Plot Individual Factor with Uncertainties

```python
# Select a factor to plot in detail
factor_to_plot = pmf.profiles[0]

# Plot profile uncertainty
fig = pmf.visualization.plot_profile_uncertainty(
    profile=factor_to_plot, 
    constrained=True
)
plt.show()
```

### Plot Source Profile Details

```python
# Plot detailed source profile
pmf.visualization.plot_source_profile(
    profile=factor_to_plot, 
    constrained=True
)
```

### Plot Comprehensive Profile Information

```python
# Plot comprehensive information for each profile
fig_dict = pmf.visualization.plot_all_profiles(
    constrained=True,
    profiles=pmf.profiles[:2]  # Limit to first two profiles
)

# Display each figure
for profile, fig in fig_dict.items():
    plt.figure(fig.number)
    plt.show()
```

## Step 6: Advanced Preprocessing (For New Datasets)

If you want to prepare raw data for PMF analysis:

```python
import pandas as pd
from PMF_toolkits.preprocessing import PMFPreprocessor

# Load your raw data
raw_data = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Initialize preprocessor
preprocessor = PMFPreprocessor(raw_data)

# Track values below detection limit
dl_mask = preprocessor.track_quantification_limits()

# Summarize data quality
quality_summary = preprocessor.summarize_data_quality()
print(quality_summary)

# Convert to numeric and handle detection limits
data_numeric = preprocessor.convert_to_numeric()

# Handle missing values
data_clean = preprocessor.handle_missing_values(
    method="interpolate",
    data=data_numeric
)

# Compute uncertainties
uncertainties = preprocessor.compute_uncertainties(
    method="polissar",
    params={"error_fraction": 0.1}
)

# Filter species with sufficient valid data
data_filtered = preprocessor.filter_species(min_valid=0.75)

# Compute signal-to-noise ratio
sn_ratio = preprocessor.compute_signal_to_noise()
print(sn_ratio)

# Create correlation matrix
corr_matrix = preprocessor.compute_correlation_matrix()

# Select good tracer species
tracers = preprocessor.select_tracers(
    correlation_threshold=0.3,
    sn_threshold=3.0
)
print(f"Good tracer species: {tracers}")

# Prepare final data for PMF
pmf_object = PMF.from_data(
    data=data_filtered,
    uncertainties=uncertainties,
    site="my_custom_site"
)
```

## Step 7: Working with Multiple Sites

For multi-site analysis:

```python
# Initialize PMF with multi-site data
pmf_multi = PMF(
    site="site_name",
    reader="xlsx",
    BDIR="path/to/multisite_files",
    multisites=True
)

# Load data
pmf_multi.read.read_all()

# Analysis and visualization works the same way as for single sites
# Example: Compare profiles between sites
site1_profile = pmf.dfprofiles_c["Factor1"]
site2_profile = pmf_multi.dfprofiles_c["Factor1"]

similarity = pmf.analysis.compute_similarity_metrics(
    site1_profile, 
    site2_profile
)
print(similarity)
```

## Step 8: Saving Results

To save your plots and analysis results:

```python
# Create a plot and save it
fig = pmf.visualization.plot_factor_profiles()
pmf.visualization.save_plot(
    formats=["png", "pdf"],
    name="factor_profiles",
    directory="output_directory/"
)

# Export numerical results to CSV
contributions = pmf.to_cubic_meter()
contributions.to_csv("source_contributions.csv")

explained_var = pmf.analysis.explained_variation()
explained_var.to_csv("explained_variation.csv")
```

## Complete Example

Here's a complete workflow example:

```python
import matplotlib.pyplot as plt
from PMF_toolkits import PMF

# Initialize PMF object
pmf = PMF(site="urban_site", reader="xlsx", BDIR="pmf_outputs/")

# Load data
pmf.read.read_all()

# Verify data was loaded
if not pmf.ensure_data_loaded():
    print("Error loading data!")
    exit()

# Basic information
print(f"Analyzing {len(pmf.profiles)} factors:")
for i, factor in enumerate(pmf.profiles):
    print(f"  {i+1}. {factor}")

# Analysis
print("\nAnalyzing contributions...")
contributions = pmf.to_cubic_meter()
seasonal = pmf.get_seasonal_contribution()
explained = pmf.analysis.explained_variation()

# Detect potentially mixed factors
mixed = pmf.analysis.detect_mixed_factors(threshold=0.6)
if mixed:
    print(f"\nWarning: Potentially mixed factors detected: {mixed}")

# Create visualizations
print("\nCreating visualizations...")

# Factor profiles
plt.figure(figsize=(12, 8))
pmf.visualization.plot_factor_profiles(normalize=True)
plt.tight_layout()
plt.savefig("factor_profiles.png", dpi=300)

# Time series
plt.figure(figsize=(12, 6))
pmf.visualization.plot_time_series(stacked=True)
plt.tight_layout()
plt.savefig("time_series.png", dpi=300)

# Seasonal contributions
plt.figure(figsize=(10, 6))
pmf.visualization.plot_seasonal_contribution()
plt.tight_layout()
plt.savefig("seasonal_contributions.png", dpi=300)

print("\nAnalysis complete!")
```

## Further Resources

- [Theory Documentation](index.md) - Detailed explanation of PMF theory
- [EPA PMF 5.0 User Guide](https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses) - Official documentation from EPA

## Common Issues and Solutions

### Missing Data

If you encounter issues with missing data:

```python
# Check what data is available
print(f"Base profiles available: {pmf.dfprofiles_b is not None}")
print(f"Constrained profiles available: {pmf.dfprofiles_c is not None}")
print(f"Base contributions available: {pmf.dfcontrib_b is not None}")
print(f"Constrained contributions available: {pmf.dfcontrib_c is not None}")
print(f"Bootstrap results available: {pmf.dfBS_profile_c is not None}")

# Try to load specific components
try:
    pmf.read.read_constrained_profiles()
    print("Successfully loaded constrained profiles")
except Exception as e:
    print(f"Error loading constrained profiles: {str(e)}")
```

### Handling Non-Standard Factor Names

If your factors have non-standard names:

```python
# Rename factors to more informative names
pmf.rename_factors({
    "Factor1": "Traffic",
    "Factor2": "Industrial",
    "Factor3": "Biomass Burning",
    "Factor4": "Secondary Sulfate",
    "Factor5": "Dust"
})

# Alternatively, use automatic categorization
pmf.rename_factors_to_factors_category()
```

### Custom Handling of Total Variable

If you need to change the total variable:

```python
# Check current total variable
print(f"Current total variable: {pmf.totalVar}")

# Replace with a different one
pmf.replace_totalVar("PM2.5")
```