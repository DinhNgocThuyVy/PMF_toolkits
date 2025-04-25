# Changelog

## 0.2.1 (25/4/2025)

### Publish source code

### Removed
- Removed redundant similarity calculation methods in `PMFAnalysis` class
- Removed unused bootstrap_analysis stub method
- Streamlined metric calculation methods

### Changed
- Improved code maintainability
- Reduced code duplication

- Minor bug fixes and improvements
- Updated documentation

## Version 0.2.0 (2023-2024)

### Core Architecture Changes

#### Class Hierarchy & Structure
- **Added**: Abstract base class `BaseReader` with a formal interface for all reader operations
- **Added**: Class inheritance structure for readers (`XlsxReader`, `SqlReader`, `MultisitesReader`)
- **Added**: New specialized classes (`PMFAnalysis`, `PMFVisualization`, `PMFPreprocessor`) to encapsulate functionality
- **Added**: Type hints throughout the codebase to improve code robustness
- **Changed**: Main `PMF` class from `class PMF(object)` in original code to modern Python class
- **Changed**: Class structure from accessor-style to object-oriented design
- **Changed**: Moved plotting functions from `PlotterAccessor` (in original) to dedicated `PMFVisualization` class

#### Module Organization
- **Added**: Modular structure with separate files for different components:
  - core.py: Main PMF class (replaces the original PMF.py)
  - readers.py: Data readers (expanded from original)
  - analysis.py: Statistical analysis (entirely new)
  - `visualization.py`: Plotting functions (replaces the original plotter.py)
  - preprocessing.py: Data preparation (entirely new)
  - utils.py: Helper functions (expanded from original)
  - validation.py: Validation against reference data (entirely new)
  - `reference_data/`: Reference data directory with additional profiles

### Reader Module (readers.py)

#### BaseReader Class
- **Borrowed**: Abstract base class structure from weber-s but significantly expanded.
- **Added**: Abstract methods with clear signatures for all reader operations.
- **Added**: Common data processing functionality for all reader types.
- **Added**: `_handle_non_convergente_bootstrap()` method for handling non-convergent bootstrap runs.
- **Added**: `read_all()` convenience method to load all data at once.
- **Enhanced**: Error handling and logging for better debugging.

#### XlsxReader Class
- **Borrowed**: Core logic for reading Excel files from weber-s's implementation.
- **Added**: `_split_df_by_nan()` method for better handling of irregular Excel layouts.
- **Added**: `_detect_multisite_format()` method for improved format detection.
- **Added**: `_process_contributions()` method for standardized processing of contributions data.
- **Enhanced**: Error handling for missing or malformed data.
- **Enhanced**: Support for EPA PMF 5.0 file formats, including constrained and base runs.
- **Enhanced**: Handling of below-detection-limit values and missing data.
- **Renamed**: `read_base_profiles` → `read_base_profiles()` (kept name but enhanced functionality).
- **Renamed**: `read_constrained_profiles` → `read_constrained_profiles()` (kept name but enhanced functionality).

#### SqlReader Class
- **Borrowed**: Basic SQL reading logic from original code.
- **Added**: New SQL database storage support with expanded capabilities.
- **Added**: Configurable SQL table naming via `SQL_table_names` parameter.
- **Added**: Compatibility with different SQL engines.
- **Enhanced**: Error handling and data validation for SQL operations.

#### MultisitesReader Class
- **Added**: Completely new class for multi-site analysis.
- **Added**: Auto-detection of multi-site formats.
- **Added**: Methods for combining data from multiple sites.
- **Added**: Support for cross-site comparisons.
- **Enhanced**: Error handling for multi-site data inconsistencies.

### General Improvements
- **Enhanced**: Type hints and docstrings for better code readability and IDE support.
- **Enhanced**: Modular design for easier maintenance and extension.
- **Enhanced**: Logging for better traceability of errors and warnings.

### Removed Features
- **Removed**: Hardcoded file paths in favor of dynamic path handling.
- **Removed**: Deprecated methods that were no longer relevant to EPA PMF 5.0 outputs.

### Breaking Changes
- **Changed**: Method signatures for several reader methods to improve consistency.
- **Changed**: Error handling now raises exceptions instead of silent failures.

### Backward Compatibility
- **Maintained**: Core functionality from weber-s's implementation for basic use cases.
- **Added**: Compatibility layers for older method names where possible.

### New Features
- **Added**: Support for handling complex Excel layouts with empty rows and columns.
- **Added**: Automatic detection of EPA PMF 5.0 file formats.
- **Added**: Multi-site analysis capabilities.
- **Added**: SQL database support for large-scale data storage and retrieval.

### Performance Improvements
- **Optimized**: File reading operations for large Excel files.
- **Enhanced**: Memory management for handling large datasets.
- **Improved**: Speed of data processing and validation.

### Core Module (core.py)

#### PMF Class
- **Borrowed**: Basic approach to PMF data model from weber-s's PMF_toolkits
- **Changed**: Constructor parameters to support multiple reader types and multi-site analysis
- **Changed**: Initialization flow with `_init_reader()` and `_init_data_containers()` methods
- **Added**: Factory method `from_data()` for creating PMF objects directly from data
- **Added**: Integration with analysis and visualization components through `.analysis` and `.visualization` properties
- **Added**: `ensure_data_loaded()` method with improved diagnostics for data validation
- **Added**: `preprocess_data()` method for integrated preprocessing
- **Added**: Error handling for loading operations with detailed messages
- **Enhanced**: `to_cubic_meter()` method with better diagnostics and error handling

#### Data Handling Methods
- **Borrowed**: Basic approach to data conversion from weber-s's PMF_toolkits/py4pm
- **Added**: `get_seasonal_contribution()` method with improved handling of seasonal patterns
- **Added**: `replace_totalVar()` method for changing the total variable name in all dataframes
- **Added**: `rename_factors()` method (replacing `rename_profile()` from original)
- **Added**: `rename_factors_to_factors_category()` method (replacing `rename_profile_to_profile_category()` from original)
- **Added**: `explained_variation()` method for calculating explained variation by factors
- **Added**: `print_uncertainties_summary()` method with better formatting
- **Added**: `recompute_new_species()` method with expanded species support
- **Added**: Data validation and error handling throughout all methods

### Analysis Module (analysis.py)

#### PMFAnalysis Class
- **Added**: Entirely new class not present in original code
- **Added**: `analyze_factor_profiles()` method with multiple analysis approaches
- **Added**: `estimate_uncertainties()` method for comprehensive uncertainty estimation
- **Added**: `compute_profile_similarity()` method with multiple similarity metrics
- **Added**: `compute_bootstrap_similarity()` method for bootstrap analysis
- **Added**: `compare_runs()` method for comparing different PMF runs
- **Added**: `bootstrap_analysis()` method for detailed bootstrap statistics
- **Added**: `explained_variation()` method with improved calculations
- **Added**: `factor_temporal_correlation()` method for correlation analysis
- **Added**: `detect_mixed_factors()` method for identifying potentially mixed factors
- **Added**: `compute_model_diagnostics()` method for model quality assessment
- **Added**: Various helper methods for data analysis and statistical testing

#### Similarity Metrics
- **Borrowed**: Basic similarity calculations from weber-s's py4pm deltaTool module
- **Enhanced**: Added robust error handling and input validation
- **Enhanced**: Added support for different normalization options
- **Added**: Multiple similarity metrics (PD, SID, COD) with standardized implementation
- **Added**: `compute_similarity_metrics()` method for comparing profiles
- **Added**: Functions for calculating specific metrics: `compute_SID()`, `compute_PD()`, `compute_COD()`

#### Module-Level Functions
- **Added**: `compute_Q_values()` function for model diagnostics
- **Added**: `compute_r2_matrix()` function for correlation analysis
- **Added**: `compute_scaled_residuals()` function for residual analysis
- **Added**: `assess_rotational_ambiguity()` function for DISP analysis
- **Added**: `compute_distance_matrix()` function for profile comparison

### Visualization Module (`visualization.py`)

#### PMFVisualization Class
- **Borrowed**: Basic plotting approach from weber-s's PMF_toolkits's `PlotterAccessor` class
- **Changed**: Complete rewrite with enhanced functionality and object-oriented design
- **Changed**: Parameter names standardized (e.g., `plot_per_microgramm()` → `plot_per_microgram()`)
- **Added**: `save_plot()` method for saving figures in multiple formats
- **Added**: Over 15 different plotting functions, including:
  - `plot_factor_profiles()`: Enhanced version of original functionality
  - `plot_time_series()`: Time series visualization with multiple options
  - `plot_seasonal_patterns()`: Seasonal analysis visualization
  - `plot_diagnostics()`: Diagnostic plots for model assessment
  - `plot_factor_fingerprints()`: Characteristic profiles visualization
  - `plot_contribution_summary()`: Summary visualizations
  - `plot_similarity_matrix()`: Similarity matrix visualization
  - `plot_delta_tool()`: Delta Tool style visualization
  - `plot_seasonal_contribution()`: Enhanced version of original
  - `plot_stacked_contribution()`: Improved stacked area charts
  - `plot_source_profile()`: Detailed factor profile visualization
  - `plot_samples_sources_contribution()`: Enhanced sample-wise visualization
  - `plot_polluted_contribution()`: Improved pollution threshold analysis
  - `plot_per_microgram()`: Fixed spelling from original `plot_per_microgramm()`
  - `plot_stacked_profiles()`: Enhanced visualization of stacked profiles
  - `plot_profile_uncertainty()`: New visualization of uncertainties
  - `plot_all_profiles_with_uncertainties()`: Comprehensive visualization
- **Added**: Helper methods for consistent styling and formatting
- **Added**: Multiple output format support for saved figures
- **Enhanced**: Customization options for all plots with expanded parameters
- **Enhanced**: Better handling of date axes with `format_xaxis_timeseries()`

#### Backward Compatibility
- **Added**: Legacy method support through `_convert_plotter_to_new_api` decorator
- **Added**: Compatibility methods for old function names:
  - `plot_contrib()` → `plot_factor_contributions()`
  - `plot_per_microgramm()` → `plot_per_microgram()`
  - `plot_totalspeciesum()` → `plot_species_distribution()`
  - `Contrib_uncertainty_BS()` → `contribution_uncertainty_bootstrap()`

### Preprocessing Module (preprocessing.py)

#### PMFPreprocessor Class
- **Added**: Completely new preprocessing functionality not present in original code
- **Added**: `track_quantification_limits()` method for handling detection limits
- **Added**: `summarize_data_quality()` method for data quality assessment
- **Added**: `convert_to_numeric()` method for data type conversion
- **Added**: `compute_uncertainties()` method with multiple calculation approaches
- **Added**: `handle_missing_values()` method with multiple imputation strategies
- **Added**: `filter_species()` method for removing low-quality variables
- **Added**: `compute_species_statistics()` method for descriptive statistics
- **Added**: `normalize_to_total()` method for data normalization
- **Added**: `prepare_pmf_input()` method for creating EPA PMF input files
- **Added**: `prepare_data()` method for comprehensive data preparation
- **Added**: `compute_signal_to_noise()` method for variable selection
- **Added**: `compute_correlation_matrix()` method for correlation analysis
- **Added**: `select_tracers()` method for identifying tracer species
- **Added**: `count_below_ql_values()` method for data quality metrics
- **Added**: Visualization methods:
  - `plot_timeseries()`: Time series visualization with customization
  - `plot_timeseries_2axis()`: Dual-axis time series visualization
  - `plot_heatmap()`: Correlation heatmap visualization
  - `regression_plot()`: Regression analysis visualization
  - `cluster_analysis()`: Hierarchical clustering visualization

#### Module-Level Functions
- **Added**: `load_concentration_data()` function for data loading
- **Added**: `summarize_dataset()` function for overview statistics

### Utils Module (utils.py)

#### Helper Functions
- **Borrowed**: `add_season()` function from weber-s's code
- **Borrowed**: Basic approach for source categories from weber-s's `get_sourcesCategories()`
- **Enhanced**: `get_sourcesCategories()` with expanded categories and improved mapping
- **Added**: `get_sourceColor()` function with standardized color schemes
- **Added**: `format_xaxis_timeseries()` function for consistent date formatting
- **Added**: `pretty_specie()` function for formatting chemical species names
- **Added**: `compute_similarity_metrics()` function for profile comparison
- **Enhanced**: Robust error handling throughout all utility functions

### Validation Module (validation.py)

#### OutputValidator Class
- **Added**: Completely new validation functionality not present in original code
- **Added**: Reference profile comparison methods
- **Added**: Ratio-based validation approaches
- **Added**: Visualization of validation results against reference data
- **Added**: Statistical assessment of similarities between profiles
- **Added**: Methods for comparing with established source profiles

### Dependencies and Requirements
- **Updated**: Modernized package dependencies for latest Python versions
- **Added**: Support for newer pandas and numpy versions
- **Added**: Type annotations for better IDE integration
- **Updated**: Documentation with more comprehensive examples
- **Changed**: Package structure to follow modern Python packaging practices

### Documentation
- **Enhanced**: Function docstrings with full parameter descriptions and types
- **Added**: Examples in docstrings for common use cases
- **Added**: Theory explanations in module docstrings
- **Added**: Comprehensive README.md with installation and usage instructions
- **Added**: Example notebooks in docs/examples
- **Added**: Detailed API documentation in docs/api
- **Added**: Quickstart guide and extended tutorials
- **Enhanced**: Detailed error messages throughout the code

### Removed/Deprecated Features
- **Removed**: Direct manipulation of dataframes in favor of encapsulated methods
- **Deprecated**: Old plot function names (with backward compatibility maintained)
- **Removed**: Hardcoded file paths from original code
- **Changed**: Simplified configuration approach with explicit parameters
- **Removed**: Some redundant utility functions from original codebase

### API Changes
- **Changed**: Method signatures for consistency and clarity
- **Added**: Proper error handling with descriptive messages
- **Added**: Return type annotations for all functions
- **Enhanced**: Method chaining support for fluent interface
- **Added**: Factory methods for easier instantiation
- **Renamed**: Many methods for clarity (e.g., `rename_profile` → `rename_factors`)

### Performance Improvements
- **Enhanced**: More efficient Excel file parsing with robustness improvements
- **Added**: Optimized data structure handling for large datasets
- **Enhanced**: Better memory management for large datasets
- **Added**: Progress reporting for long-running operations
- **Enhanced**: File reading optimization with smarter data structures

## Version 0.2.0

- Refactory code based on the old version https://github.com/weber-s/PMF_toolkits/releases/tag/v0.1.10
- Basic PMF analysis functionality
- Visualization tools
- Documentation
- Original implementations by weber-s (https://github.com/weber-s/PMF_toolkits/ and https://github.com/weber-s/py4pm)