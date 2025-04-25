# Validation Module Documentation

## Introduction

The `validation.py` module in the `PMF_toolkits` package provides tools to validate Positive Matrix Factorization (PMF) outputs against reference profiles and species ratios. This module is essential for ensuring the physical interpretability and reliability of PMF results. By comparing PMF factor profiles with reference data and assessing species ratios, users can identify potential sources and validate the quality of their PMF solutions.

## Theoretical Background

### PMF Validation

Positive Matrix Factorization (PMF) decomposes a data matrix into source contributions and profiles. Validation is a critical step to ensure that the results are meaningful and consistent with known source characteristics. The `validation.py` module focuses on three key aspects of validation:

1. **Species Ratios**: Ratios like OC/EC (Organic Carbon to Elemental Carbon) are used to identify specific source types, such as biomass burning or vehicular emissions.
2. **Similarity Metrics**: Metrics like Pearson Distance (PD) and Standardized Identity Distance (SID) quantify the similarity between PMF factors and reference profiles.
3. **Regression Diagnostics**: R² values for each species indicate how well the PMF model explains the observed data.

### Similarity Metrics

- **Pearson Distance (PD)**: Measures the correlation between two profiles. A lower PD indicates higher similarity.
- **Standardized Identity Distance (SID)**: Quantifies the overlap between two profiles. A lower SID indicates higher similarity.
- **Coefficient of Divergence (COD)**: Measures the divergence between two profiles.

### Species Ratios

Species ratios are calculated for specific factors to compare with reference values. For example, the OC/EC ratio can differentiate between fossil fuel combustion and biomass burning.

## Module Features

The `validation.py` module provides the following key functionalities:

### 1. Species Ratio Calculation
- **`calculate_ratio`**: Computes the ratio of two species for a given factor.
- **`compare_ratio`**: Compares a calculated ratio with reference values to identify potential source types.

### 2. Similarity Metrics
- **`calculate_similarity_metrics`**: Calculates PD, SID, and COD for a PMF factor against reference profiles.
- **`find_similar_sources`**: Identifies reference sources similar to a given PMF factor based on similarity thresholds.

### 3. Regression Diagnostics
- **`read_regression_diagnostics`**: Extracts R² values for each species from PMF output files.

### 4. Visualization
- **`plot_similarity_diagram`**: Creates a scatter plot of PD vs. SID for reference sources.

## Tutorial

### 1. Initializing the Validator

```python
from PMF_toolkits.validation import OutputValidator

# Define data directory and site name
data_dir = "single_site"
site_name = "GRE-fr"

# Create a validator
validator = OutputValidator(
    site=site_name, 
    data_dir=data_dir,
    lib_dir="reference_data"  # Path to reference data files
)

print(f"Validator initialized for site: {site_name}")
```

### 2. Checking Species Ratios

```python
# Calculate OC/EC ratio for a specific factor
factor_name = 'Biomass burning'
ratio = validator.calculate_ratio(factor_name, "OC", "EC")
print(f"OC/EC ratio for {factor_name}: {ratio:.2f}")

# Compare with reference values
ratio_result = validator.compare_ratio(factor_name, "OC", "EC")
if ratio_result.get("success", False):
    print("Matching source types:")
    for match in ratio_result["matches"]:
        print(f"- {match['Source']} (expected range: {match['Min']:.2f}-{match['Max']:.2f})")
```

### 3. Comparing with Reference Profiles

```python
# Calculate similarity metrics for a factor
metrics = validator.calculate_similarity_metrics(factor_name)
print(metrics.sort_values('PD').head(5))

# Find similar sources using thresholds
similar_sources = validator.find_similar_sources(
    factor_name,
    pd_threshold=0.4,
    sid_threshold=0.8,
    min_species=5
)
print(similar_sources)
```

### 4. Visualizing Similarity

```python
# Plot similarity diagram
fig = validator.plot_similarity_diagram(factor_name, max_sources=10)
fig.show()
```

### 5. Extracting Regression Diagnostics

```python
# Extract regression diagnostics
r2_data = validator.read_regression_diagnostics()
print(r2_data)
```

## Conclusion

The `validation.py` module is a powerful tool for validating PMF outputs. By leveraging species ratios, similarity metrics, and regression diagnostics, users can ensure the reliability and interpretability of their PMF results. This module is essential for environmental scientists and researchers working with PMF data.