# Changes from Draft to Final PMF_toolkits Package

This document summarizes the major changes and improvements made to the `PMF_toolkits` package, transforming it from the initial draft version (functional scripts in the `draft` folder) into the current stable release. The draft code, while functional in isolation for specific tasks, lacked robustness, structure, and consistent data handling compared to the final package, leading to potential differences in results, including unexpected empty DataFrames.

## General Architecture & Structure

- **Major Refactoring**: Transitioned from script-like draft files (`draft_*.py`) to a modular package structure (`core.py`, `readers.py`, `analysis.py`, `visualization.py`, `preprocessing.py`, `utils.py`, `validation.py`).
- **Object-Oriented Design**:
    - Implemented a robust class hierarchy (e.g., `BaseReader` inheritance in `readers.py` evolved from `draft_readers.py`/`draft_Read_file_Multisites.py`).
    - Separated concerns into specialized classes like `PMFAnalysis`, `PMFVisualization`, `PMFPreprocessor`, replacing the combined `PMF` and `Plotter` classes in `draft_PMF.py` and `draft_plotter.py`.
- **Improved Naming & Consistency**: Standardized function and class names (e.g., `draft_plotter.Plotter` -> `visualization.PMFVisualization`, `draft_PMF.py.PMF` -> `core.PMF`).
- **Enhanced Code Quality**: Added type hints, improved error handling (e.g., specific `FileNotFoundError` catches in `readers.py` absent in drafts), and increased adherence to PEP8 standards.

## Core Functionality & Data Handling

- **Bug Fixes & Robustness**: Addressed potential issues and edge cases present in the draft's simpler data handling, ensuring more reliable calculations. For example, `readers.XlsxReader` includes more checks for NaN columns/rows and specific EPA PMF 5.0 Excel structures compared to `draft_Read_file_Multisites.XlsxReader`.
- **Reader Enhancements**:
    - Formalized `BaseReader` abstract class (`readers.py`) for consistent data loading API, inheriting from the concepts in both `draft_readers.py` and `draft_Read_file_Multisites.py`.
    - `draft_Read_file_Multisites.py` appears to be a more advanced draft reader, parts of which were likely refined and integrated into `readers.py`. Specific parsing logic (e.g., identifying headers, handling NaNs) differs significantly between draft versions and the final package (See Function Mapping below).
- **PMF Core (`core.py`)**:
    - Redesigned `PMF` class focusing on data management, integrating analysis/visualization via composition rather than direct methods as in `draft_PMF.py`.
    - Refined methods like `get_seasonal_contribution` (moved logic partly to `utils.add_season`), `replace_totalVar`, `rename_factors`.
    - Implemented robust data validation checks (`ensure_data_loaded`) absent in the draft.
- **Data Preprocessing (`preprocessing.py`)**: **Crucially**, added a dedicated module for preprocessing steps (handling LOQ, missing values, uncertainty calculation, variable filtering) which were largely absent or implicit in the draft files. **This is a major source of potential result differences.**

## API Changes

- **Intuitive Signatures**: Refined function parameters (e.g., using keyword arguments, adding defaults) across the API compared to the draft's simpler functions.
- **New Capabilities**:
    - Added comprehensive analysis features in `analysis.PMFAnalysis` (e.g., similarity metrics like SID/PD from `draft_extractSOURCES.py` are now in `utils.compute_similarity_metrics`, bootstrap analysis, diagnostics).
    - Introduced extensive visualization options in `visualization.PMFVisualization` (many more plot types and customization options compared to `draft_plotter.py`).
    - Added data validation capabilities (`validation.py`) - completely new compared to the draft.
- **Utility Functions (`utils.py`)**: Expanded `utils.py` with more helper functions (e.g., `get_sourcesCategories`, `get_sourceColor`, `pretty_specie` refined from drafts, added `compute_similarity_metrics`).
- **Deprecated/Refactored Functions**: Cleaned up or refactored elements from the draft (e.g., plotting functions in `draft_plotter.py` are reorganized and enhanced in `visualization.py`).

## Implementation Details

- **Algorithm Improvements**: While the core PMF algorithm isn't fully shown, the package structure implies potentially more robust or optimized numerical methods compared to any simplified versions in the draft. Error handling during calculations is likely improved.
- **Data Structure Changes**: More organized data storage using class attributes in `core.PMF` compared to potentially looser handling in `draft_PMF.py`. Metadata handling is more explicit.

## Testing & Validation

- **Validation Module**: Added `validation.py` for comparing results, a critical step missing in the draft.
- **Improved Reliability**: Implicitly fixed issues through refactoring, added error handling, and dedicated preprocessing. The package aims for greater reliability than the draft scripts.

## Documentation

- **Comprehensive Updates**: Significantly expanded docstrings, added examples, and improved overall documentation compared to the minimal comments in the draft files.

## Known Discrepancies Between Draft and Package (Potential Reasons for Different Results)

- **Preprocessing Differences**: **This is the most likely cause.** The final package includes explicit preprocessing steps (`preprocessing.py`) for handling uncertainties, missing values, LOQ, etc., which were absent or handled differently/implicitly in the draft. Different data weighting or input matrices will lead to different PMF solutions or intermediate dataframes used by analysis/visualization functions. If preprocessing filters data heavily, subsequent steps might receive empty DataFrames.

- **Reader Implementation Differences**: The package's `readers.py` likely implements stricter parsing and validation logic compared to `draft_readers.py` and `draft_Read_file_Multisites.py`.
    - **Impact**: The package might fail to read files the draft could, or it might drop rows/columns (e.g., due to unexpected NaN patterns, slightly different header locations) that the draft retained. This can lead to empty or differently shaped DataFrames being stored in the `PMF` object (e.g., `pmf.dfprofiles_b`, `pmf.dfcontrib_c`). Functions relying on this data will then produce different or empty results. *Example: `read_base_contributions` in `draft_Read_file_Multisites.py` has a specific way of identifying and setting the header row which might differ from the package's logic.*

- **Error Handling Philosophy**:
    - **Draft**: More lenient, often printing warnings but attempting to continue execution even with potentially problematic data (e.g., unexpected NaNs, type issues).
    - **Package**: Likely more rigorous, raising exceptions or returning empty DataFrames/`None` when encountering invalid data or failed validation checks (e.g., in `core.PMF` methods or readers).
    - **Impact**: The package may stop execution or return empty results at points where the draft would have continued, potentially yielding misleading results in the draft's case.

### Specific Implementation Differences

1.  **Data Handling Differences**:
    - **Draft**: In `draft_PMF.py`, methods like `to_cubic_meter` do minimal error checking and assume data exists and is correctly formatted.
    - **Package**: The `core.py` version likely adds extensive checks (e.g., `ensure_data_loaded`) and validates inputs before processing.
    - **Impact**: The package may return empty results or raise errors for inputs that the draft would process despite potential issues (e.g., required DataFrames being `None`).

2.  **Reader Implementation (Details)**:
    - **Draft**: `draft_readers.py` and `draft_Read_file_Multisites.py` contain simpler parsing logic (e.g., finding headers like "Factor Profiles", dropping initial columns, handling `-999`). Specific implementations differ even between these two draft files.
    - **Package**: The `readers.py` likely implements more complex validation (e.g., checking exact sheet names, expected number of columns/rows after certain steps, more robust NaN handling).
    - **Impact**: The package may not read the same files in the exact same way, especially with edge cases or slight variations from the expected EPA PMF Excel format. *Example: The logic for finding the start/end rows for profiles/contributions based on text like "Factor Profiles" or "Concentrations for" might be more sensitive in the package.*

3.  **Data Transformation Changes**:
    - **Draft**: `draft_PMF.py` applies transformations directly with minimal validation (e.g., `to_relative_mass`, `recompute_new_species`).
    - **Package**: In `core.py`, methods likely include additional checks (e.g., for zero division in `to_relative_mass`, presence of required species in `recompute_new_species`).
    - **Impact**: Even with the same input data *structure*, different validation or calculation logic (especially around edge cases like zeros or NaNs) can lead to different numerical results or empty outputs if checks fail.

4.  **Bootstrap Analysis**:
    - **Draft**: In `draft_readers.py` and `draft_Read_file_Multisites.py`, the `_handle_non_convergente_bootstrap` method uses a specific approach (checking `totalVar > 100`).
    - **Package**: The bootstrap handling in the package might use different thresholds, additional filtering criteria, or handle NaNs differently during this process.
    - **Impact**: Different filtering of bootstrap results can lead to different uncertainty estimates or affect functions relying on `dfBS_profile_*` DataFrames.

5.  **Uncertainty Summary Reading**:
    - **Draft**: `read_*_uncertainties_summary` methods parse specific structures in the summary Excel files. Logic exists for finding "Swaps" and "Concentrations for" sections.
    - **Package**: The package reader might expect a slightly different layout or have stricter parsing rules for these sections. Column naming and indexing (`set_index(["Profile", "Specie"])`) must match exactly.
    - **Impact**: Differences in parsing can lead to incorrect assignment or empty `df_uncertainties_summary_*` or `df_disp_swap_*` DataFrames.

6.  **Method Parameter Changes**:
    - **Draft**: Methods often use positional arguments and fewer default values.
    - **Package**: Methods have been redesigned with keyword arguments and potentially different defaults.
    - **Impact**: Function behavior may change if called with the same positional parameters, or if default behaviors differ.

7.  **Utility Function Logic**:
    - **Draft**: Functions like `add_season` (`draft_utils.py`) or `compute_SID`/`compute_PD` (`draft_extractSOURCES.py`) have specific implementations.
    - **Package**: Corresponding functions in `utils.py` or `analysis.py` might have different logic, default parameters, or error handling (e.g., `utils.add_season` might use different month-to-season mappings, `utils.compute_similarity_metrics` might handle NaNs differently).
    - **Impact**: Calculations relying on these utilities may produce different results.

### Troubleshooting Steps

1.  **Input Data Verification**: Ensure input data (Excel files or SQL tables) provided to both versions is *exactly* the same.
2.  **Parameter Consistency**: Check that method parameters (and their values) match between draft and package calls, paying attention to new or changed defaults in the package.
3.  **Verbose Debugging**: Add `print(df.info())`, `print(df.head())`, `print(df.shape)` statements within the package code (especially in `readers.py` and `core.py` methods) to track the state and shape of DataFrames at critical steps. Compare these outputs to similar print statements added to the draft code.
4.  **Step-by-Step Execution**: Run both the draft code and the package code side-by-side on a *single*, simple input dataset (e.g., one site's base run). Print outputs after each major step (reading profiles, reading contributions, calculating `to_cubic_meter`, etc.) to pinpoint exactly where the divergence occurs.
5.  **Intermediate Data Inspection**: Directly compare the intermediate DataFrames stored in the `PMF` object (`pmf.dfprofiles_b`, `pmf.dfcontrib_b`, etc.) between the draft run and the package run *immediately after they are read/created*. Use `df.equals(other_df)` or compare `df.head()`, `df.shape`, `df.columns`, `df.index`. This is crucial for identifying reader discrepancies.
6.  **Focus on Preprocessing**: If using the package's preprocessing module, temporarily disable it or compare its output DataFrame directly with the raw data used by the draft to understand its impact.
7.  **Check Reader Logic**: Carefully compare the parsing logic in the relevant `read_*` methods between the draft files (`draft_readers.py`, `draft_Read_file_Multisites.py`) and the package's `readers.py`. Look for differences in how headers are found, rows/columns are dropped, NaNs are handled, and indices/columns are set.

### Mitigation Strategies

1.  **Adapt Method Calls**: Update calls to package methods to account for parameter changes or new required parameters.
2.  **Review Preprocessing**: Understand exactly what the package's preprocessing module does and ensure it aligns with the desired analysis. Adjust its configuration or bypass it if its effects are unintended for the comparison.
3.  **Refine Package Readers**: If the package readers are too strict or parse incorrectly compared to the draft (and the draft's parsing was correct for the specific file format), adjust the package's reader logic to be more robust or flexible, while still maintaining correctness.
4.  **Check Data Assumptions**: Ensure the package functions are not making implicit assumptions about data (e.g., specific column names, index types, absence of NaNs) that were met in the draft but not after package refactoring/preprocessing.
5.  **Documentation**: Document specific known differences in behavior and expected outputs for user awareness, especially regarding preprocessing and reader validation.

## Function Mapping Between Draft and Final Package

This section provides a mapping between functions/methods in the draft files and their corresponding implementations in the final package. The "Key Differences" column highlights changes relevant to potential result discrepancies.

### Core PMF Functionality

| Draft File     | Draft Function/Method                  | Package File | Package Function/Method                | Key Differences                                                                                                                               |
| -------------- | -------------------------------------- | ------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `draft_PMF.py` | `PMF.__init__`                         | `core.py`    | `PMF.__init__`                         | Package adds more robust parameter checking, component initialization (Analysis, Viz, Preprocessor), and initializes data containers to `None`. |
| `draft_PMF.py` | `to_cubic_meter`                       | `core.py`    | `to_cubic_meter`                       | Package likely adds checks for `None` dataframes (`dfcontrib_*`, `dfprofiles_*`), potentially returning empty DF if data not loaded.        |
| `draft_PMF.py` | `to_relative_mass`                     | `core.py`    | `to_relative_mass`                     | Package likely adds zero-division protection, checks for `None` dataframes, and presence of `totalVar`.                                       |
| `draft_PMF.py` | `get_total_specie_sum`                 | `core.py`    | `get_total_specie_sum`                 | Package likely includes null checks for profile dataframes.                                                                                   |
| `draft_PMF.py` | `get_seasonal_contribution`            | `core.py`    | `get_seasonal_contribution`            | Package version calls `utils.add_season`; relies on `to_cubic_meter` internally, inheriting its data checks.                                  |
| `draft_PMF.py` | `replace_totalVar`                     | `core.py`    | `replace_totalVar`                     | Package adds validation (e.g., if `newTotalVar` exists).                                                                                      |
| `draft_PMF.py` | `rename_factors`                       | `core.py`    | `rename_factors`                       | Package adds type checking and checks if dataframes exist before renaming.                                                                    |
| `draft_PMF.py` | `rename_factors_to_factors_category` | `core.py`    | `rename_factors_to_factors_category` | Relies on `rename_factors` and `utils.get_sourcesCategories`.                                                                                 |
| `draft_PMF.py` | `recompute_new_species`                | `core.py`    | `recompute_new_species`                | Package includes more validation checks (e.g., if required species like `OC*` exist).                                                         |
| `draft_PMF.py` | `print_uncertainties_summary`          | `core.py`    | `get_uncertainties_summary` (likely) | Package likely renamed; checks if uncertainty dataframes are loaded.                                                                          |
| N/A            | N/A                                    | `core.py`    | `preprocess_data`                      | New in package, delegates to `PMFPreprocessor` - **Major source of difference**.                                                              |
| N/A            | N/A                                    | `core.py`    | `ensure_data_loaded`                   | New validation method used internally, can prevent execution if data is `None`.                                                               |
| N/A            | N/A                                    | `core.py`    | `from_data`                            | New factory method.                                                                                                                           |

### Readers

| Draft File (`_readers` or `_Multisites`) | Draft Function/Method                  | Package File | Package Function/Method                | Key Differences                                                                                                                                                              |
| ---------------------------------------- | -------------------------------------- | ------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Both                                     | `BaseReader` (abstract class)          | `readers.py` | `BaseReader` (abstract class)          | Package version ensures consistent interface; draft versions differ slightly.                                                                                                |
| Both                                     | `__init__`                             | `readers.py` | `__init__`                             | Package likely standardizes initialization.                                                                                                                                  |
| Both                                     | `_handle_non_convergente_bootstrap`    | `readers.py` | `_handle_non_convergente_bootstrap`    | Package might have different threshold (`>100`) or NaN handling logic. Draft versions seem identical here.                                                                     |
| Both                                     | `read_metadata`                        | `readers.py` | `read_metadata`                        | Package version might have more robust guessing logic for `totalVar` or stricter checks. Draft versions seem identical.                                                      |
| Both                                     | `read_all`                             | `readers.py` | `read_all`                             | Package improves exception handling, might stop sooner on error. Draft versions seem identical.                                                                                |
| Both                                     | `XlsxReader.__init__`                  | `readers.py` | `XlsxReader.__init__`                  | Package standardizes.                                                                                                                                                        |
| Both                                     | `XlsxReader._split_df_by_nan`          | `readers.py` | `XlsxReader._split_df_by_nan`          | Package adds error handling for unexpected data formats; core logic seems similar.                                                                                           |
| Both                                     | `XlsxReader.read_base_profiles`        | `readers.py` | `XlsxReader.read_base_profiles`        | Package adds more validation (NaN checks, header finding robustness, type inference). Draft versions differ slightly in finding end row. **Potential source of difference.** |
| Both                                     | `XlsxReader.read_constrained_profiles` | `readers.py` | `XlsxReader.read_constrained_profiles` | Similar to base profiles, more validation in package. **Potential source of difference.**                                                                                    |
| Both                                     | `XlsxReader.read_base_contributions`   | `readers.py` | `XlsxReader.read_base_contributions`   | Package adds more validation. Draft versions differ significantly in header/index setting logic. **High potential source of difference.**                                    |
| Both                                     | `XlsxReader.read_constrained_contributions` | `readers.py` | `XlsxReader.read_constrained_contributions` | Similar to base contributions, more validation in package. Draft versions differ in header/index setting. **High potential source of difference.**                         |
| Both                                     | `XlsxReader.read_base_bootstrap`       | `readers.py` | `XlsxReader.read_base_bootstrap`       | Package adds more validation for bootstrap data structure. Parsing logic (finding start row, slicing) might be stricter. Draft versions seem identical.                      |
| Both                                     | `XlsxReader.read_constrained_bootstrap` | `readers.py` | `XlsxReader.read_constrained_bootstrap` | Similar to base bootstrap. Draft versions seem identical.                                                                                                                    |
| Both                                     | `XlsxReader.read_base_uncertainties_summary` | `readers.py` | `XlsxReader.read_base_uncertainties_summary` | Package adds more validation. Parsing logic for sections ("Swaps", "Concentrations for") and column assignment might be stricter. Draft versions seem identical. **Potential source of difference.** |
| Both                                     | `XlsxReader.read_constrained_uncertainties_summary` | `readers.py` | `XlsxReader.read_constrained_uncertainties_summary` | Similar to base uncertainties. Draft versions seem identical. **Potential source of difference.**                                                              |
| Both                                     | `SqlReader` class                      | `readers.py` | `SqlReader` class                      | Package improves SQL query construction, error checks, data cleaning (dropping 'index' column). Draft versions seem identical.                                               |
| `_Multisites.py`                         | `XlsxReader` specific logic            | `readers.py` | `MultisitesReader` (likely)            | Package standardizes the multisites reader if applicable.                                                                                                                    |

### Visualization

| Draft File         | Draft Function/Method        | Package File       | Package Function/Method             | Key Differences                                                                                                                                                           |
| ------------------ | ---------------------------- | ------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `draft_plotter.py` | `Plotter.__init__`           | `visualization.py` | `PMFVisualization.__init__`         | Package adds more configuration options.                                                                                                                                  |
| `draft_plotter.py` | `_save_plot`                 | `visualization.py` | `_save_plot`                        | Package adds error handling for saving.                                                                                                                                   |
| `draft_plotter.py` | `_plot_per_microgramm`       | `visualization.py` | `plot_factor_profiles` (likely)     | Relies on `PMF` object data; differences stem from data loading/preprocessing. Plotting logic itself might be refactored.                                                 |
| `draft_plotter.py` | `_plot_totalspeciesum`       | `visualization.py` | `plot_species_percentage` (likely)  | Relies on `PMF` object data; differences stem from data loading/preprocessing.                                                                                            |
| `draft_plotter.py` | `_plot_contrib`              | `visualization.py` | `plot_factor_contributions` (likely) | Relies on `PMF` object data; differences stem from data loading/preprocessing.                                                                                            |
| `draft_plotter.py` | `Contrib_uncertainty_BS`     | `visualization.py` | `plot_bootstrap_contributions` (likely) | Draft version calculates BS contrib; Package likely separates calculation (analysis?) and plotting. Relies heavily on BS/uncertainty data being loaded correctly. |
| `draft_plotter.py` | `_plot_profile`              | `visualization.py` | `plot_factor_profiles_combined` (likely) | Combines other plots; differences accumulate from underlying plot functions/data.                                                                                         |
| `draft_plotter.py` | `_plot_ts_stackedbarplot`    | `visualization.py` | `plot_stacked_timeseries` (likely)  | Core stacking logic might be similar, but relies on contribution data.                                                                                                    |
| `draft_plotter.py` | `_get_polluted_days_mean`    | `analysis.py`      | `analyze_polluted_days` (likely)    | Logic moved to analysis; plotting part in visualization.                                                                                                                  |
| `draft_plotter.py` | `plot_per_microgramm`        | `visualization.py` | `plot_factor_profiles`              | Wrapper function; differences stem from underlying `_plot_per_microgramm`.                                                                                                |
| `draft_plotter.py` | `plot_totalspeciesum`        | `visualization.py` | `plot_species_percentage`           | Wrapper function; differences stem from underlying `_plot_totalspeciesum`.                                                                                                |
| `draft_plotter.py` | `plot_contrib`               | `visualization.py` | `plot_factor_contributions`         | Wrapper function; differences stem from underlying `_plot_contrib`.                                                                                                       |
| `draft_plotter.py` | `plot_all_profiles`          | `visualization.py` | `plot_all_factor_profiles`          | Wrapper function; differences accumulate.                                                                                                                                 |
| `draft_plotter.py` | `plot_stacked_contribution`  | `visualization.py` | `plot_stacked_contributions`        | Relies on `to_cubic_meter`; differences stem from data loading.                                                                                                           |
| `draft_plotter.py` | `plot_seasonal_contribution` | `visualization.py` | `plot_seasonal_contributions`       | Relies on `get_seasonal_contribution`; differences stem from data loading/calculation.                                                                                    |
| `draft_plotter.py` | `plot_stacked_profiles`      | `visualization.py` | `plot_stacked_species`              | Relies on `get_total_specie_sum`; differences stem from data loading.                                                                                                     |
| `draft_plotter.py` | `plot_polluted_contribution` | `visualization.py` | `plot_polluted_days_comparison`     | Relies on analysis function (`_get_polluted_days_mean` equivalent).                                                                                                       |
| `draft_plotter.py` | `plot_samples_sources_contribution` | `visualization.py` | `plot_samples_contributions` | Relies on `to_cubic_meter`; uses `_plot_ts_stackedbarplot`.                                                                                                               |
| `draft_plotter.py` | `_pretty_specie` / `pretty_specie` | `utils.py`     | `pretty_specie`                     | Moved to utils, potentially enhanced.                                                                                                                                     |

### Analysis

| Draft File                | Draft Function/Method       | Package File  | Package Function/Method                | Key Differences                                                                                                                            |
| ------------------------- | --------------------------- | ------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `draft_extractSOURCES.py` | `compute_SID`               | `utils.py`    | `compute_similarity_metrics`           | Generalized into a single function with method parameter; NaN handling or `to_relativeMass` logic might differ slightly.                   |
| `draft_extractSOURCES.py` | `compute_PD`                | `utils.py`    | `compute_similarity_metrics`           | Generalized into a single function with method parameter; NaN handling or correlation calculation might differ slightly.                 |
| `draft_extractSOURCES.py` | `to_relativeMass`           | `core.py`     | `to_relative_mass`                     | Moved to core.py with improved error handling (zero division, None checks).                                                                |
| `draft_extractSOURCES.py` | `plot_deltatool_pretty`     | `visualization.py` | `plot_similarity_diagram` (likely) | Plotting function, enhanced options.                                                                                                       |
| `draft_extractSOURCES.py` | `plot_similarityplot`       | `analysis.py` | `PMFAnalysis.compute_profile_similarity` (likely) | Core calculation likely moved to analysis, plotting to visualization. Relies on `compute_similarity_metrics`.                           |
| `draft_extractSOURCES.py` | `plot_similarityplot_all`   | `analysis.py` | `PMFAnalysis.compare_runs` (related)   | Draft function calculates and plots mean similarity; Package likely separates calculation (analysis) and plotting (visualization).           |
| `draft_extractSOURCES.py` | `get_all_SID_PD`            | `analysis.py` | `PMFAnalysis.compute_bootstrap_similarity` (related) | Draft calculates pairwise SID/PD; Package likely has more structured analysis methods.                                                     |
| `draft_extractSOURCES.py` | `plot_similarity_profile`   | `analysis.py`/`visualization.py` | Split functionality | Draft calculates mean SID/PD and plots; Package likely splits calculation (analysis) and plotting (visualization). Relies on similarity data. |
| `draft_extractSOURCES.py` | `plot_relativeMass`         | `visualization.py` | `plot_factor_profiles` (related)   | Draft plots relative mass profiles for multiple sites; Package likely uses `plot_factor_profiles` with specific parameters.                |
| `draft_extractSOURCES.py` | `save4deltaTool`            | N/A           | N/A                                    | Export function, might be added to package later if needed.                                                                                |
| N/A                       | N/A                         | `analysis.py` | `PMFAnalysis` methods                  | Many new analysis functions in package (uncertainty estimation, bootstrap analysis, diagnostics, correlations, etc.) absent in draft.      |

### Utilities

| Draft File         | Draft Function/Method     | Package File | Package Function/Method       | Key Differences                                                                                             |
| ------------------ | ------------------------- | ------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `draft_utils.py`   | `add_season`              | `utils.py`   | `add_season`                  | Package adds better error handling, potentially different season schemes/boundaries.                          |
| `draft_utils.py`   | `get_sourcesCategories`   | `utils.py`   | `get_sourcesCategories`       | Package likely extended with more categories/mappings.                                                      |
| `draft_utils.py`   | `get_sourceColor`         | `utils.py`   | `get_sourceColor`             | Package likely extended with more color mappings/palettes.                                                  |
| `draft_utils.py`   | `format_xaxis_timeseries` | `utils.py`   | `format_xaxis_timeseries`     | Enhanced formatting options.                                                                                |
| `draft_plotter.py` | `pretty_specie`           | `utils.py`   | `pretty_specie`               | Moved to utils module with better error checking.                                                           |
| `draft_extractSOURCES.py` | `sourcesColor`     | `utils.py`   | `get_sourceColor` (related)   | Draft had specific color dict; Package uses more general `get_sourceColor`.                                 |
| N/A                | N/A                       | `utils.py`   | `compute_similarity_metrics`  | New function combining `draft_extractSOURCES.py` SID/PD logic. **Potential source of difference.**          |
| N/A                | N/A                       | `utils.py`   | Various helper functions      | New utility functions added in package.                                                                     |

### Preprocessing

| Draft File | Draft Function/Method | Package File       | Package Function/Method          | Key Differences                                                                                                                                                              |
| ---------- | --------------------- | ------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| N/A        | N/A                   | `preprocessing.py` | `PMFPreprocessor.__init__`       | New in package. **Major source of difference.**                                                                                                                              |
| N/A        | N/A                   | `preprocessing.py` | `_validate_input`                | New in package.                                                                                                                                                              |
| N/A        | N/A                   | `preprocessing.py` | `track_quantification_limits`    | New in package.                                                                                                                                                              |
| N/A        | N/A                   | `preprocessing.py` | `summarize_data_quality`         | New in package.                                                                                                                                                              |
| N/A        | N/A                   | `preprocessing.py` | `load_concentration_data`        | New in package (though reading happens in `readers.py`, this might refer to initial loading/validation within the preprocessing context).                                    |
| N/A        | N/A                   | `preprocessing.py` | `summarize_dataset`              | New in package.                                                                                                                                                              |
| N/A        | N/A                   | `preprocessing.py` | (Other preprocessing functions)  | Functions for handling missing values, uncertainty calculation, species filtering, etc., are new and fundamentally change the data compared to the draft's direct usage. |

### Validation

| Draft File | Draft Function/Method | Package File    | Package Function/Method       | Key Differences                 |
| ---------- | --------------------- | --------------- | ----------------------------- | ------------------------------- |
| N/A        | N/A                   | `validation.py` | `OutputValidator`             | New in package, no draft equivalent |
| N/A        | N/A                   | `validation.py` | Various validation functions  | New in package                  |

This table provides a guide to understanding how your draft code evolved. Focus on the "Key Differences" for Readers, Core PMF Functionality, and the introduction of Preprocessing to diagnose why the package might yield different or empty results compared to the draft scripts.

## Academic Soundness Review

Based on the provided draft code (`draft_readers.py`, `draft_Read_file_Multisites.py`, `draft_PMF.py`, `draft_plotter.py`, `draft_utils.py`, `draft_extractSOURCES.py`):

1.  **Data Reading (`readers`)**:
    *   **Excel Parsing**: The `XlsxReader` implementations attempt to parse standard EPA PMF 5.0 Excel output structures. They rely on finding specific text markers ("Factor Profiles", "Factor Contributions", "Concentrations for", "Columns are:") and relative positioning. This is generally sound but can be brittle if the Excel layout varies slightly (e.g., extra rows/columns, merged cells not handled by `openpyxl`). The logic to find start/end rows seems reasonable. Dropping initial columns and setting headers based on parsed rows is standard.
    *   **NaN/Missing Value Handling**: Replacing `-999` with `np.nan` (`read_*_contributions`) is correct for EPA PMF output. Dropping NaN rows/columns (`dropna`) is common but needs care – ensuring essential data isn't accidentally dropped. The logic in `_split_df_by_nan` for bootstrap seems appropriate for the described format. Setting very small values (< 10e-6) to 0 in profiles is a minor data alteration, potentially justifiable to handle numerical noise but should be documented.
    *   **SQL Reading**: The `SqlReader` uses standard SQL queries based on provided table names, site, and optional program. This is sound, assuming the database schema correctly stores the PMF output components. Dropping specific columns like "Program", "Station", "index" after filtering is appropriate.
    *   **Metadata (`read_metadata`)**: Identifying profiles/species from `dfprofiles_b` is correct. The logic to guess `totalVar` based on common names ("PM10", "PM2.5", etc.) or names containing "PM" is a reasonable heuristic when not explicitly provided, but the warning about multiple possibilities is important.
    *   **Bootstrap Handling (`_handle_non_convergente_bootstrap`)**: Filtering runs where `totalVar > 100` seems like a heuristic to catch potentially non-physical/non-converged runs. While plausible, the threshold `100` might need justification or could be parameterizable. The commented-out section filtering runs with very low `totalVar` mass (`< 10**-3`) is another plausible heuristic to avoid division-by-zero issues, but again, the threshold is arbitrary. Standard PMF practice often relies on the mapping percentage reported by EPA PMF itself rather than post-hoc filtering based on profile values, though such filtering can be a pragmatic choice.
    *   **Potential Issues**: The main risk is the brittleness of Excel parsing. If the package `readers.py` implements stricter validation or slightly different parsing logic, it could correctly reject files that the draft processed (perhaps incorrectly) or vice-versa. The difference in handling contribution headers between the two draft readers highlights this sensitivity.

2.  **Data Transformation (`PMF` methods)**:
    *   **`to_cubic_meter`**: Correctly multiplies contribution factor loadings (`dfcontrib`) by the species concentration in the profile (`dfprofiles`). This is standard.
    *   **`to_relative_mass`**: Correctly divides species concentrations in profiles by the `totalVar` concentration in the profile. Dropping `totalVar` afterwards is appropriate for relative analysis. Standard.
    *   **`get_total_specie_sum`**: Correctly calculates the percentage contribution of each profile to the total concentration of each species (summing across profiles for a given species). Standard.
    *   **`get_seasonal_contribution`**: Correctly calculates the contribution in µg/m³ per season by multiplying profiles and contributions, then groups by season (using `add_season`). Normalization logic (dividing by total seasonal sum) is standard for relative seasonal contributions. Calculating the annual mean/sum is also standard. Relies on `add_season` utility.
    *   **`recompute_new_species`**: The logic for `OC` recalculation using `OC*` and specific organic species with fixed carbon content (`equivC`) is a common approach in atmospheric science when detailed speciation is available. The `equivC` values themselves should be based on literature/known stoichiometry. This modifies the input data based on scientific assumptions, which is acceptable but needs clear documentation.
    *   **Potential Issues**: These methods seem academically sound, performing standard PMF post-processing calculations. Errors or empty results in the package are likely due to the input DataFrames (`dfprofiles_*`, `dfcontrib_*`, etc.) being `None` or incorrectly formatted *before* these methods are called (i.e., issues in the reader or preprocessing steps).

3.  **Analysis (`extractSOURCES`, `plotter`)**:
    *   **SID/PD (`compute_SID`, `compute_PD`)**: These implement standard formulas for Scaled Identity Distance and Pearson Distance (1-R²). The use of `to_relativeMass` before calculation (if `isRelativeMass=False`) is appropriate. Checking for a minimum number of species (`len(sp)>3`) before calculating is a reasonable precaution. Sound.
    *   **Plotting Data Preparation**: Functions like `_get_polluted_days_mean` correctly group data based on a threshold and calculate means. Plotting functions generally prepare data correctly for `seaborn` or `matplotlib` (e.g., melting, pivoting, calculating means/std devs).
    *   **Potential Issues**: No major academic soundness issues found in the calculation logic itself within the draft. Discrepancies in analysis/plots using these will likely stem from differences in the input data provided by the `PMF` object.

**Conclusion on Soundness**: The core calculations and data transformations shown in the draft files appear to follow standard practices for interpreting and analyzing EPA PMF 5.0 output. The readers attempt to parse the standard output formats correctly, though Excel parsing can be fragile. The heuristics used in bootstrap filtering and `totalVar` guessing are common but should be used with awareness and documentation. The primary risk for discrepancies between the draft and the package lies in:
1.  The **new preprocessing module** in the package fundamentally altering the data.
2.  **Stricter validation and potentially different parsing logic in the package's readers** leading to different or empty DataFrames being loaded compared to the more lenient draft readers.
3.  **Stricter error handling in the package** causing functions to return empty results or raise errors where the draft might have continued with potentially invalid intermediate data.

Focusing on comparing the reader outputs and understanding the preprocessing steps is key to resolving the discrepancies.