"""PMF preprocessing module for data preparation and validation.

This module provides tools for preparing data for PMF analysis, including:
- Detection limit handling
- Missing value imputation
- Uncertainty calculation
- Data quality assessment
- Signal-to-noise calculation
- Data validation

The PMFPreprocessor class is the primary interface for data preparation.
PMF class methods like preprocess_data(), track_detection_limits(), and 
compute_uncertainties() are convenient wrappers that delegate to this class.

Example
-------
>>> from PMF_toolkits import PMFPreprocessor
>>> preprocessor = PMFPreprocessor(data)
>>> # Handle missing values and detection limits
>>> data_cleaned = preprocessor.handle_missing_values()
>>> dl_mask = preprocessor.track_quantification_limits()
>>> # Calculate uncertainties
>>> uncertainties = preprocessor.compute_uncertainties(method="polissar")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import KMeans
import warnings

# Configure logging
logger = logging.getLogger('PMF_toolkits.preprocessing')

class PMFPreprocessor:
    """
    Prepares and validates input data for PMF analysis.
    
    This class handles:
    - Conversion of data to numeric format
    - Treatment of below-detection-limit values
    - Calculation of uncertainties
    - Data quality checks
    - Signal-to-noise calculations
    - Correlation analysis
    - Tracer selection
    
    Parameters
    ----------
    data : pd.DataFrame
        Input concentration data matrix with samples as rows and species as columns
    ql_values : dict, optional
        Quantification limits for each species
    """
    
    def __init__(self, data: pd.DataFrame, ql_values: Optional[Dict[str, float]] = None):
        """Initialize with concentration data."""
        self.data = data.copy()
        self.ql_values = ql_values or {}
        self.ql_mask = None  # Track which values were quantification limit markers
        self._validate_input()
        
    def _validate_input(self) -> None:
        """Validate input data format and content."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if self.data.empty:
            raise ValueError("Input data cannot be empty")
        if self.data.isnull().all().any():
            logger.warning("Input data contains columns with all missing values")

    def track_quantification_limits(self) -> pd.DataFrame:
        """
        Identify values that were originally quantification limit markers.
        
        Returns
        -------
        pd.DataFrame
            Boolean mask of cells that were below quantification limit
        """
        self.ql_mask = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)
            
        for col in self.data.columns:
            # String-based QL markers
            string_mask = self.data[col].isin(["<QL", "<LQ", "<LD", "<DL"])
            
            # Numeric QL markers (e.g., negative values)
            numeric_mask = pd.Series(False, index=self.data.index)
            numeric_values = pd.to_numeric(self.data[col], errors='coerce')
            valid_numeric = ~numeric_values.isna()
            
            if valid_numeric.any():
                numeric_mask[valid_numeric] = ((numeric_values[valid_numeric] < 0) | 
                                             (numeric_values[valid_numeric] == -2))
                
            self.ql_mask[col] = string_mask | numeric_mask
            
        return self.ql_mask

    def summarize_data_quality(self) -> pd.DataFrame:
        """
        Summarize data quality issues (missing values, below QL).
        
        Returns
        -------
        pd.DataFrame
            Summary statistics about data quality
        """
        if self.ql_mask is None:
            self.track_quantification_limits()
            
        table = pd.DataFrame(columns=self.data.columns, index=["Missing", "Below QL"])
        
        for col in self.data.columns:
            missing_pct = self.data[col].isna().mean() * 100
            ql_pct = self.ql_mask[col].mean() * 100
            
            table.loc["Missing", col] = missing_pct
            table.loc["Below QL", col] = ql_pct
            
        return table

    def convert_to_numeric(self) -> pd.DataFrame:
        """
        Convert all data to numeric, replacing QL markers with appropriate values.
        
        Returns
        -------
        pd.DataFrame
            Data with all values converted to numeric
        """
        if self.ql_mask is None:
            self.track_quantification_limits()
            
        data_numeric = self.data.copy()
        
        for col in self.data.columns:
            # Get QL value for this column or estimate one
            ql = self.ql_values.get(col, None)
            if ql is None:
                numeric_vals = pd.to_numeric(self.data[col], errors='coerce')
                positive_vals = numeric_vals[numeric_vals > 0]
                ql = positive_vals.min() / 2 if len(positive_vals) > 0 else 0.001
                    
            # Replace values below QL with QL/2
            data_numeric.loc[self.ql_mask[col], col] = ql / 2
            
        return data_numeric.apply(pd.to_numeric, errors='coerce')

    def compute_uncertainties(self, method: str = "percentage", 
                            params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute uncertainties for each measurement using selected method.
        
        Parameters
        ----------
        method : str
            Uncertainty calculation method:
            - "percentage": Fixed percentage of concentration
            - "DL": Detection limit based uncertainty
            - "polissar": Polissar method using DL
            - "custom": Custom calculation function
        params : dict, optional
            Parameters specific to the chosen method
        
        Returns
        -------
        pd.DataFrame
            Calculated uncertainties for each measurement
        """
        data_numeric = self.convert_to_numeric()
        
        if params is None:
            params = {"percentage": 0.1}
            
        # Ensure QL mask exists
        if self.ql_mask is None:
            self.track_quantification_limits()

        # Select appropriate method
        method_map = {
            "percentage": self._compute_percentage_uncertainty,
            "DL": self._compute_dl_uncertainty,
            "polissar": self._compute_polissar_uncertainty
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown uncertainty method: {method}. "
                           f"Supported methods: {', '.join(method_map.keys())}")
            
        return method_map[method](data_numeric, params)

    def _compute_percentage_uncertainty(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate percentage-based uncertainties."""
        percentage = params.get("percentage", 0.1)
        return data * percentage
        
    def _compute_dl_uncertainty(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate detection limit based uncertainties."""
        dl_values = params.get("DL", {})
        uncertainties = pd.DataFrame(0, index=data.index, columns=data.columns)
        
        for col in data.columns:
            # Get detection limit for this column or use default
            dl = dl_values.get(col, data[col].min() / 3)
            
            # Calculate DL-based uncertainty
            mask_below_dl = self.ql_mask[col]
            uncertainties.loc[mask_below_dl, col] = dl * 5/6
            uncertainties.loc[~mask_below_dl, col] = np.sqrt(
                (data.loc[~mask_below_dl, col] * 0.1)**2 + (dl/3)**2
            )
        
        return uncertainties
        
    def _compute_polissar_uncertainty(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate uncertainties using Polissar method."""
        dl_values = params.get("DL", {})
        error_fraction = params.get("error_fraction", 0.1)
        uncertainties = pd.DataFrame(0, index=data.index, columns=data.columns)
        
        for col in data.columns:
            # Get detection limit for this column or use default
            dl = dl_values.get(col, data[col].min() / 3)
            
            # Polissar method for uncertainty calculation
            mask_below_dl = self.ql_mask[col]
            uncertainties.loc[mask_below_dl, col] = 5/6 * dl
            uncertainties.loc[~mask_below_dl, col] = np.sqrt(
                np.square(error_fraction * data.loc[~mask_below_dl, col]) +
                np.square(dl/3)
            )
            
        return uncertainties

    def handle_missing_values(self, method: str = "interpolate", data: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters
        ----------
        method : str
            Method to handle missing values:
            - "interpolate": Interpolate missing values
            - "mean": Fill with column means
            - "median": Fill with column medians
            - "remove": Remove rows with missing values
            - "min_fraction": Fill with minimum value fraction
        data : pd.DataFrame, optional
            Data to process, uses self.data if None
        **kwargs : dict
            Additional parameters specific to each method
            
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        data = data if data is not None else self.convert_to_numeric()
        
        if method == "interpolate":
            return data.interpolate(method=kwargs.get('interpolation_method', 'linear'))
        elif method == "mean":
            return data.fillna(data.mean())
        elif method == "median":
            return data.fillna(data.median())
        elif method == "remove":
            return data.dropna(**kwargs)
        elif method == "min_fraction":
            fraction = kwargs.get('fraction', 0.5)
            for col in data.columns:
                min_val = data[col].min()
                if pd.notnull(min_val) and min_val > 0:
                    data[col] = data[col].fillna(min_val * fraction)
            return data
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")

    def filter_species(self, min_valid: float = 0.75,
                     include_always: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter species based on percentage of valid measurements.
        
        Parameters
        ----------
        min_valid : float, default=0.75
            Minimum fraction of valid values required to keep a species
        include_always : list of str, optional
            Species to always include regardless of valid fraction
            
        Returns
        -------
        pd.DataFrame
            Filtered data with only valid species
        """
        data_numeric = self.convert_to_numeric()
        valid_frac = data_numeric.count() / len(data_numeric)
        valid_species = valid_frac[valid_frac >= min_valid].index.tolist()
        
        if include_always:
            for species in include_always:
                if species in data_numeric.columns and species not in valid_species:
                    valid_species.append(species)
        
        return data_numeric[valid_species]

    def compute_species_statistics(self) -> pd.DataFrame:
        """
        Compute basic statistics for each species.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with statistics (mean, std, min, max) for each species
        """
        data_numeric = self.convert_to_numeric()
        stats = data_numeric.describe().T
        return stats[['mean', 'std', 'min', 'max']]

    def normalize_to_total(self, total_var: str) -> pd.DataFrame:
        """
        Normalize species concentrations to the total concentration of a given variable.
        
        Parameters
        ----------
        total_var : str
            The variable to normalize to (e.g., "PM10")
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        data_numeric = self.convert_to_numeric()
        if total_var not in data_numeric.columns:
            raise ValueError(f"Total variable '{total_var}' not found in data columns")
        
        total_values = data_numeric[total_var]
        normalized_data = data_numeric.div(total_values, axis=0)
        return normalized_data

    def prepare_pmf_input(self, total_var: str = "PM10",
                         uncertainty_method: str = "percentage",
                         uncertainty_params: Optional[Dict] = None,
                         min_valid: float = 0.75,
                         handling_method: str = "interpolate") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Prepare input data for PMF analysis.
        
        Parameters
        ----------
        total_var : str, default="PM10"
            The variable to normalize to
        uncertainty_method : str, default="percentage"
            Method to compute uncertainties
        uncertainty_params : dict, optional
            Parameters for the uncertainty computation method
        min_valid : float, default=0.75
            Minimum fraction of valid values required to keep a species
        handling_method : str, default="interpolate"
            Method to handle missing values
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, Dict]
            Normalized data, uncertainties, and metadata
        """
        # Handle missing values
        data_handled = self.handle_missing_values(method=handling_method)
        
        # Filter species
        data_filtered = self.filter_species(min_valid=min_valid)
        
        # Normalize to total
        data_normalized = self.normalize_to_total(total_var=total_var)
        
        # Compute uncertainties
        uncertainties = self.compute_uncertainties(method=uncertainty_method, params=uncertainty_params)
        
        # Collect metadata
        metadata = {
            "total_var": total_var,
            "uncertainty_method": uncertainty_method,
            "uncertainty_params": uncertainty_params,
            "min_valid": min_valid,
            "handling_method": handling_method
        }
        
        return data_normalized, uncertainties, metadata

    def prepare_data(self, handle_missing: bool = True, convert_numeric: bool = True) -> pd.DataFrame:
        """
        Prepare data for PMF analysis in one step.
        
        Parameters
        ----------
        handle_missing : bool, default=True
            Whether to handle missing values
        convert_numeric : bool, default=True
            Whether to convert data to numeric
            
        Returns
        -------
        pd.DataFrame
            Prepared data
        """
        if self.ql_mask is None:
            self.track_quantification_limits()
            
        data = self.data.copy()
        
        if convert_numeric:
            data = self.convert_to_numeric()
            
        if handle_missing:
            data = self.handle_missing_values(data=data)
            
        return data

    def compute_signal_to_noise(self) -> pd.Series:
        """
        Compute signal-to-noise ratio for each species.
        
        Returns
        -------
        pd.Series
            Signal-to-noise ratios for each species
        """
        try:
            # Convert to numeric values if needed
            data_numeric = self.convert_to_numeric()
            
            # Calculate uncertainties using percentage method
            uncertainties = self.compute_uncertainties(method="percentage", 
                                                    params={"percentage": 0.1})
            
            # Calculate signal-to-noise ratio for each species
            signal_means = data_numeric.mean()
            noise_means = uncertainties.mean()
            
            # Calculate S/N ratio
            sn_ratio = signal_means / noise_means
            
            # Handle infinite values that might occur if noise is zero
            sn_ratio = sn_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Convert to Series with species as index
            sn_ratio = pd.Series(sn_ratio, index=data_numeric.columns)
            
            # Log results for debugging
            logger.debug(f"Signal-to-noise ratios calculated: {len(sn_ratio)} species")
            
            return sn_ratio
            
        except Exception as e:
            logger.error(f"Error calculating signal-to-noise: {str(e)}")
            # Return a Series with all species but zero values to avoid plotting errors
            return pd.Series(0, index=self.data.columns)
    
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix between species.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        try:
            data_numeric = self.convert_to_numeric()
            return data_numeric.corr()
        except Exception as e:
            logger.error(f"Error computing correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def select_tracers(self, correlation_threshold: float = 0.3, 
                      sn_threshold: float = 3.0) -> List[str]:
        """
        Select good tracer species based on correlation and signal-to-noise.
        
        Parameters
        ----------
        correlation_threshold : float, default=0.3
            Maximum correlation allowed for a good tracer
        sn_threshold : float, default=3.0
            Minimum signal-to-noise ratio required
            
        Returns
        -------
        List[str]
            List of selected tracer species
        """
        try:
            corr = self.compute_correlation_matrix()
            sn = self.compute_signal_to_noise()
            
            # Find species with low correlation with others
            low_corr_species = corr[corr < correlation_threshold].index
            
            # Find species with high signal-to-noise
            high_sn_species = sn[sn > sn_threshold].index
            
            # Intersection of both criteria
            tracers = pd.Index(low_corr_species).intersection(pd.Index(high_sn_species))
            
            return tracers.tolist()
        except Exception as e:
            logger.error(f"Error selecting tracers: {str(e)}")
            return []
    
    def count_below_ql_values(self) -> pd.DataFrame:
        """
        Count NaN and below quantification limit values for each species.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with percentage of NaN and <QL values for each species
        """
        try:
            if self.ql_mask is None:
                self.track_quantification_limits()
                
            table = pd.DataFrame(columns=self.data.columns, index=["NaN", "<QL"])
            
            for col in self.data.columns:
                nan_pct = self.data[col].isna().mean() * 100
                ql_pct = self.ql_mask[col].mean() * 100
                
                table.loc["NaN", col] = nan_pct
                table.loc["<QL", col] = ql_pct
                
            return table
        except Exception as e:
            logger.error(f"Error counting <QL values: {str(e)}")
            return pd.DataFrame()

    def plot_timeseries(self, species: Union[str, List[str]], unit: str = 'µg/m³', 
                       figsize: Tuple[int, int] = (8, 4), show_equation: bool = False) -> plt.Figure:
        """
        Plot time series of selected species.
        
        Parameters
        ----------
        species : str or list of str
            Species to plot
        unit : str, default='µg/m³'
            Unit for y-axis
        figsize : tuple, default=(8, 4)
            Figure size
        show_equation : bool, default=False
            Whether to show regression equation (for 2 species)
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=100)
            data_numeric = self.convert_to_numeric()
            
            if isinstance(species, str):
                data_numeric[species].plot(ax=ax)
                ax.set_ylabel(f"Concentration ({unit})")
                ax.set_title(species)
            elif isinstance(species, list) and len(species) == 2 and show_equation:
                data_numeric[species].plot(ax=ax)
                
                # Calculate regression
                x = data_numeric[species[0]].dropna()
                y = data_numeric[species[1]].dropna()
                idx = x.index.intersection(y.index)
                
                if len(idx) > 1:  # Need at least two points for regression
                    x_vals = x.loc[idx]
                    y_vals = y.loc[idx]
                    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                    
                    # Add annotation
                    ax.annotate(f'$R^2 = {r_value ** 2:.2f}$\ny= ${slope:.2f}x{intercept:+.2f}$',
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              fontsize=12, color='black', 
                              backgroundcolor='#FFFFFF99', ha='left', va='top')
            else:
                data_numeric[species].plot(ax=ax)
                
            ax.set_ylabel(f"Concentration ({unit})")
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting timeseries: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def plot_timeseries_2axis(self, species1: str, species2: str, unit: str = 'µg/m³',
                             figsize: Tuple[int, int] = (10, 6), 
                             show_equation: bool = False) -> plt.Figure:
        """
        Plot time series with two y-axes for comparing two species.
        
        Parameters
        ----------
        species1 : str
            First species (left y-axis)
        species2 : str
            Second species (right y-axis)
        unit : str, default='µg/m³'
            Unit for y-axes
        figsize : tuple, default=(10, 6)
            Figure size
        show_equation : bool, default=False
            Whether to show regression equation
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=100)
            data_numeric = self.convert_to_numeric()
            
            # Plot first species on left y-axis
            data_numeric[species1].dropna().plot(color="red", alpha=0.8, ax=ax)
            
            # Create second y-axis and plot second species
            ax2 = ax.twinx()
            data_numeric[species2].dropna().plot(alpha=0.8, ax=ax2)
            
            # Set labels
            ax.set_ylabel(f"{species1} ({unit})")
            ax2.set_ylabel(f"{species2} ({unit})")
            
            # Create combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper left')
            
            # Add regression equation if requested
            if show_equation:
                x = data_numeric[species1].dropna()
                y = data_numeric[species2].dropna()
                idx = x.index.intersection(y.index)
                
                if len(idx) > 1:
                    x_vals = x.loc[idx]
                    y_vals = y.loc[idx]
                    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                    
                    ax2.annotate(f'$R^2 = {r_value ** 2:.2f}$\ny= ${slope:.2f}x{intercept:+.2f}$',
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=12, color='black', 
                               backgroundcolor='#FFFFFF99', ha='left', va='top')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting dual axis timeseries: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def plot_heatmap(self, species: Optional[List[str]] = None, 
                    cluster: bool = False, figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
        """
        Plot correlation heatmap for selected species.
        
        Parameters
        ----------
        species : list of str, optional
            Species to include in heatmap (all by default)
        cluster : bool, default=False
            Whether to use clustermap instead of heatmap
        figsize : tuple, default=(10, 10)
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            data_numeric = self.convert_to_numeric()
            
            if species:
                data = data_numeric[species]
            else:
                data = data_numeric
                
            corr_matrix = data.corr()
            
            if cluster:
                # Use clustermap for hierarchical clustering
                g = sns.clustermap(
                    corr_matrix,
                    cmap = "rocket_r",
                    square=True,
                    linewidths=0.1,
                    cbar_kws={'shrink': 0.6},
                    figsize=figsize
                )
                return g.fig
            else:
                fig, ax = plt.subplots(figsize=figsize)
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Draw heatmap
                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    cmap="rocket_r",
                    square=True,
                    linewidths=0.1,
                    cbar_kws={'shrink': 0.6},
                    ax=ax
                )
                
                plt.tight_layout()
                return fig
        except Exception as e:
            logger.error(f"Error plotting heatmap: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def regression_plot(self, species1: str, species2: str, 
                       figsize: Tuple[int, int] = (8, 4)) -> plt.Figure:
        """
        Create regression scatter plot between two species.
        
        Parameters
        ----------
        species1 : str
            X-axis species
        species2 : str
            Y-axis species
        figsize : tuple, default=(8, 4)
            Figure size
            
        Returns
        -------

        matplotlib.figure.Figure
            The created figure
        """
        try:
            data_numeric = self.convert_to_numeric()
            
            # Get data for both species
            x = data_numeric[species1].dropna()
            y = data_numeric[species2].dropna()
            
            # Get common indices
            idx = x.index.intersection(y.index)
            x_vals = x.loc[idx]
            y_vals = y.loc[idx]
            
            if len(idx) < 2:
                raise ValueError(f"Not enough common data points between {species1} and {species2}")
            
            # Calculate regression
            slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
            line = slope * x_vals + intercept
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            
            # Plot regression line
            plt.plot(x_vals, line, linestyle='dotted', c='r', alpha=0.5)
            
            # Plot scatter points
            plt.scatter(x_vals, y_vals, color="blue", s=50, marker="o", alpha=0.8)
            
            # Add equation
            plt.annotate(f'$R^2 = {r_value ** 2:.2f}$\ny= ${slope:.2f}x{intercept:+.2f}$',
                       xy=(min(x_vals), max(y_vals)), fontsize=12,
                       color='black', backgroundcolor='#FFFFFF99', ha='left', va='top')
            
            # Add grid
            plt.grid(which='major', alpha=0.4, linestyle='--', zorder=-1.0)
            
            # Add labels
            plt.xlabel(species1)
            plt.ylabel(species2)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error creating regression plot: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def cluster_analysis(self, species1: str, species2: str, 
                        n_clusters: Union[int, List[int]] = 2,
                        figsize: Tuple[int, int] = (8, 4)) -> Tuple[plt.Figure, Dict]:
        """
        Perform k-means clustering on two species and plot results.
        
        Parameters
        ----------
        species1 : str
            X-axis species
        species2 : str
            Y-axis species
        n_clusters : int or list of int, default=2
            Number of clusters to create
        figsize : tuple, default=(8, 4)
            Figure size
            
        Returns
        -------
        tuple
            Figure and dictionary with cluster results
        """
        try:
            data_numeric = self.convert_to_numeric()
            
            # Get data for both species
            x = data_numeric[species1].dropna()
            y = data_numeric[species2].dropna()
            
            # Get common indices
            idx = x.index.intersection(y.index)
            x_vals = x.loc[idx].values
            y_vals = y.loc[idx].values
            
            if len(idx) < 2:
                raise ValueError(f"Not enough common data points between {species1} and {species2}")
            
            # Convert to list if single int
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=100)
            
            # Plot original data in gray
            plt.scatter(x_vals, y_vals, color='gray', label='Data')
            
            results = {}
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            
            # Process each number of clusters
            for i, k in enumerate(n_clusters):
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=k, random_state=42)
                data = np.column_stack((x_vals, y_vals))
                cluster_labels = kmeans.fit_predict(data)
                
                results[k] = {
                    'kmeans': kmeans,
                    'labels': cluster_labels,
                    'cluster_stats': {}
                }
                
                # Plot each cluster with regression line
                for cluster in range(k):
                    # Get points in this cluster
                    mask = cluster_labels == cluster
                    cluster_x = x_vals[mask]
                    cluster_y = y_vals[mask]
                    
                    if len(cluster_x) > 1:  # Need at least 2 points for regression
                        # Plot cluster points
                        color = colors[cluster % len(colors)]
                        plt.scatter(cluster_x, cluster_y, 
                                  label=f'Cluster {cluster+1} (k={k})',
                                  color=color, alpha=0.7)
                        
                        # Calculate regression for cluster
                        slope, intercept, r_value, p_value, std_err = linregress(cluster_x, cluster_y)
                        line_x = np.array([min(cluster_x), max(cluster_x)])
                        line_y = slope * line_x + intercept
                        
                        # Plot regression line
                        plt.plot(line_x, line_y, '--', color=color)
                        
                        # Add stats to results
                        results[k]['cluster_stats'][cluster] = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'std_err': std_err,
                            'count': len(cluster_x)
                        }
                        
                        # Add equation annotation near cluster center
                        plt.annotate(f'$R^2 = {r_value**2:.2f}$\ny= ${slope:.2f}x{intercept:+.2f}$',
                                   xy=(np.mean(cluster_x), np.mean(cluster_y)),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=9, color=color, 
                                   backgroundcolor='#FFFFFF99')
            
            # Add labels and legend
            plt.xlabel(species1)
            plt.ylabel(species2)
            plt.grid(True, alpha=0.3)
            
            # Create legend with smaller font size
            plt.legend(fontsize=8, loc='best')
            
            plt.tight_layout()
            return fig, results
        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig, {}

    def calculate_signal_to_noise(self) -> pd.DataFrame:
        """
        Calculate signal-to-noise ratio for each species.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with S/N values for each species
        """
        if not hasattr(self, 'data') or not hasattr(self, 'uncertainties'):
            raise ValueError("Data and uncertainties must be available")
            
        data = self.data
        uncertainties = self.uncertainties
        
        # Calculate S/N for each point
        sn_values = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            # For each data point where concentration > uncertainty
            mask = data[col] > uncertainties[col]
            
            if mask.any():
                # S/N = (concentration - uncertainty) / uncertainty
                sn_values.loc[mask, col] = (data.loc[mask, col] - 
                                         uncertainties.loc[mask, col]) / uncertainties.loc[mask, col]
            
            # Where concentration <= uncertainty, S/N = 0
            sn_values.loc[~mask, col] = 0
            
        # Calculate mean S/N for each species
        mean_sn = sn_values.mean()
        
        return pd.DataFrame(mean_sn, columns=['S/N'])
    
    def analyze_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between species.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        return self.data.corr()
        
    def identify_tracer_species(self, correlation_threshold=0.3, sn_threshold=3.0) -> pd.Index:
        """
        Identify potential tracer species based on correlation and S/N ratio.
        
        Parameters
        ----------
        correlation_threshold : float, default=0.3
            Maximum correlation for a tracer species
        sn_threshold : float, default=3.0
            Minimum S/N ratio for a tracer species
            
        Returns
        -------

        pd.Index
            Index of potential tracer species
        """
        # Calculate correlation matrix
        corr_matrix = self.analyze_correlation_matrix()
        
        # Calculate S/N ratio
        sn_ratios = self.calculate_signal_to_noise()
        
        # Find species with low correlation
        # We take the max absolute correlation excluding self-correlation
        max_corr = pd.Series(index=corr_matrix.columns)
        
        for col in corr_matrix.columns:
            # Exclude self-correlation (which is always 1)
            other_cols = [c for c in corr_matrix.columns if c != col]
            max_corr[col] = corr_matrix.loc[col, other_cols].abs().max()
        
        # Find species with low correlation and high S/N
        low_corr_species = max_corr[max_corr < correlation_threshold].index
        high_sn_species = sn_ratios[sn_ratios['S/N'] > sn_threshold].index
        
        # Return the intersection
        tracers = low_corr_species.intersection(high_sn_species)
        
        return tracers

"""
Convenience functions for preparing data for PMF analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from .preprocessing import PMFPreprocessor

logger = logging.getLogger('PMF_toolkits.data_preparation')

def load_concentration_data(file_path: Union[str, Path], 
                           date_column: str = 'Date',
                           date_format: Optional[str] = None,
                           sep: str = ',') -> pd.DataFrame:
    """
    Load concentration data from a CSV or Excel file with proper date parsing.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the data file (CSV or Excel)
    date_column : str, default='Date'
        Name of the column containing dates
    date_format : str, optional
        Date format string for parsing, if None, tries to infer
    sep : str, default=','
        Separator for CSV files
        
    Returns
    -------

    pd.DataFrame
        DataFrame with samples as rows and species as columns, with a DatetimeIndex
        
    Examples
    --------
    >>> data = load_concentration_data("pmf_data.csv", date_column="SampleDate", date_format="%Y-%m-%d")
    >>> data.head()
    """
    file_path = Path(file_path)
    
    try:
        # Load data based on file type
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, sep=sep)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Convert date column to datetime
        if date_column in df.columns:
            if date_format:
                df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            else:
                df[date_column] = pd.to_datetime(df[date_column])
            
            # Set date as index
            df = df.set_index(date_column)
        else:
            logger.warning(f"Date column '{date_column}' not found in data")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def summarize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive summary of the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with samples as rows and species as columns
        
    Returns
    -------

    pd.DataFrame
        Summary statistics for each species
        
    Examples
    --------
    >>> data = load_concentration_data("pmf_data.csv")
    >>> summary = summarize_dataset(data)
    >>> print(summary)
    """
    try:
        # Basic statistics
        summary = data.describe().T
        
        # Add missing value percentage
        summary['missing_pct'] = data.isna().mean() * 100
        
        # Add coefficient of variation
        summary['cv'] = summary['std'] / summary['mean']
        
        # Add detection frequency
        preprocessor = PMFPreprocessor(data)
        dl_mask = preprocessor.track_quantification_limits()
        summary['detection_pct'] = 100 - dl_mask.mean() * 100
        
        # Reorder columns for better readability
        cols = ['count', 'mean', 'std', 'cv', 'min', '25%', '50%', '75%', 'max', 
                'missing_pct', 'detection_pct']
        summary = summary[cols]
        
        return summary.round(4)
        
    except Exception as e:
        logger.error(f"Error summarizing dataset: {str(e)}")
        raise