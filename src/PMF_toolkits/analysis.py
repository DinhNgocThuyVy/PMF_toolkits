import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple, Any
from zipfile import ZipFile
import os
import warnings
import logging

# Configure logger
logger = logging.getLogger('PMF_toolkits.analysis')

from .utils import get_sourcesCategories, to_relative_mass

def compute_similarity_metrics(p1: pd.Series, p2: pd.Series) -> Dict[str, float]:
    """
    Compute similarity metrics between two profiles.
    
    Parameters
    ----------
    p1 : pd.Series
        First profile
    p2 : pd.Series
        Second profile
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing similarity metrics:
        - SID: Source Identification Distance
        - PD: Pearson Distance
        - COD: Coefficient of Divergence
    """
    p1 = p1.dropna()
    p2 = p2.dropna()
    common_species = p1.index.intersection(p2.index)
    n_species = len(common_species)
    
    if n_species == 0:
        return {"SID": np.nan, "PD": np.nan, "COD": np.nan, "n_species": 0}
    
    p1_common = p1[common_species]
    p2_common = p2[common_species]
    
    # Calculate SID
    diff_square = np.square((p1_common - p2_common) / (p1_common + p2_common))
    sid = np.sqrt(diff_square.sum() / n_species)
    
    # Calculate PD (Pearson Distance)
    corr, _ = stats.pearsonr(p1_common, p2_common)
    pd_value = 1 - np.power(corr, 2)
    
    # Calculate COD (Coefficient of Divergence)
    cod = np.sqrt(np.mean(diff_square))
    
    return {
        "SID": sid, 
        "PD": pd_value, 
        "COD": cod, 
        "n_species": n_species
    }

def compute_distance_matrix(profiles: pd.DataFrame, metric: str = "SID", normalize: bool = True) -> pd.DataFrame:
    """
    Compute distance matrix between all profiles using specified similarity metric.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        DataFrame with profiles as columns and species as rows
    metric : str, default="SID"
        Similarity metric to use ("SID", "PD", or "COD")
    normalize : bool, default=True
        Whether to normalize profiles before computing distances
    
    Returns
    -------
    pd.DataFrame
        Distance matrix
    """
    if normalize:
        # Normalize profiles to sum to 1
        profile_sums = profiles.sum()
        profiles_norm = profiles / profile_sums
    else:
        profiles_norm = profiles
    
    n_profiles = profiles.shape[1]
    profile_names = profiles.columns
    distance_matrix = pd.DataFrame(index=profile_names, columns=profile_names, dtype=float)
    
    for i in range(n_profiles):
        profile1 = profiles_norm.iloc[:, i]
        for j in range(n_profiles):
            profile2 = profiles_norm.iloc[:, j]
            metrics = compute_similarity_metrics(profile1, profile2)
            distance_matrix.iloc[i, j] = metrics[metric]
    
    return distance_matrix

# =====================================================
# Main analysis class
# =====================================================

class PMFAnalysis:
    """
    Statistical analysis and interpretation of PMF results.
    
    Provides methods for:
    - Factor profile analysis
    - Uncertainty estimation
    - Statistical testing
    - Source contribution analysis
    """

    def __init__(self, pmf):
        self.pmf = pmf

    def analyze_factor_profiles(self, method: str = "correlation",
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
            Reference profiles for comparison. Must have same species as columns.

        Returns
        -------
        dict
            Analysis results containing matrices and statistics
        """
        # Validate data availability
        if not self.pmf.ensure_data_loaded():
            raise ValueError("Required profile data not loaded. Call read_base_profiles() or read_constrained_profiles() first.")

        if method == "correlation":
            # Compute correlation matrix between profiles
            corr_matrix = self.pmf.dfprofiles_c.corr()
            
            # Get significant correlations
            p_values = pd.DataFrame(np.zeros_like(corr_matrix),
                                 index=corr_matrix.index,
                                 columns=corr_matrix.columns)
            
            for i in corr_matrix.index:
                for j in corr_matrix.columns:
                    _, p_value = stats.pearsonr(self.pmf.dfprofiles_c[i], 
                                              self.pmf.dfprofiles_c[j])
                    p_values.loc[i,j] = p_value
                    
            results = {
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'significant_correlations': corr_matrix[p_values < 0.05]
            }
            
        elif method == "similarity":
            # Compute cosine similarity between profiles
            normalized = self.pmf.dfprofiles_c / np.sqrt((self.pmf.dfprofiles_c**2).sum())
            similarity = normalized.T @ normalized
            
            results = {
                'similarity_matrix': similarity
            }
            
        elif method == "comparison" and external_data is not None:
            # Align profiles with external data
            common_species = self.pmf.dfprofiles_c.index.intersection(external_data.columns)
            if len(common_species) == 0:
                raise ValueError("No common species found between PMF profiles and external data")
                
            pmf_profiles = self.pmf.dfprofiles_c.loc[common_species]
            ref_profiles = external_data[common_species].T
            
            # Compute correlations
            correlations = pd.DataFrame(
                index=pmf_profiles.columns,
                columns=ref_profiles.columns
            )
            
            for pmf_factor in pmf_profiles.columns:
                for ref_factor in ref_profiles.columns:
                    corr, _ = stats.pearsonr(
                        pmf_profiles[pmf_factor],
                        ref_profiles[ref_factor]
                    )
                    correlations.loc[pmf_factor, ref_factor] = corr
                    
            results = {
                'profile_correlations': correlations,
                'best_matches': correlations.idxmax(axis=1)
            }
        else:
            raise ValueError(f"Invalid method '{method}' or missing external_data for comparison")
            
        return results

    def estimate_uncertainties(self, 
                             method: str = "bootstrap",
                             n_iterations: int = 1000) -> Dict:
        """
        Estimate uncertainties in PMF results using various methods.

        Parameters
        ----------
        method : str
            Method to use:
            - "bootstrap": Bootstrap resampling
            - "displacement": DISP analysis results
            - "combined": BS-DISP combined results
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
        # Validate data availability based on method
        if method == "bootstrap":
            if not hasattr(self.pmf, 'dfBS_profile_c') or self.pmf.dfBS_profile_c is None:
                raise ValueError(
                    "Bootstrap results not available. "
                    "Run bootstrap analysis first or check that bootstrap files exist."
                )
                
            bs_profiles = self.pmf.dfBS_profile_c
                    
            # Calculate statistics for each factor/species
            stats_dict = {}
            for factor in self.pmf.profiles:
                factor_data = bs_profiles[factor]
                stats_dict[factor] = {
                    'mean': factor_data.mean(),
                    'std': factor_data.std(),
                    'ci_lower': factor_data.quantile(0.025),
                    'ci_upper': factor_data.quantile(0.975),
                    'cv': factor_data.std() / factor_data.mean()
                }
                    
            results = {
                'bootstrap_stats': pd.DataFrame(stats_dict),
                'n_iterations': n_iterations
            }
                
        elif method == "displacement":
            if not self.pmf.ensure_data_loaded() or self.pmf.df_disp_swap_c is None:
                raise ValueError(
                    "DISP results not available. "
                    "Ensure DISP analysis was run and results are loaded."
                )
                    
            results = {
                'disp_swaps': self.pmf.df_disp_swap_c,
                'uncertainties': self.pmf.df_uncertainties_summary_c
            }
                
        elif method == "combined":
            if (self.pmf.dfBS_profile_c is None or 
                self.pmf.df_disp_swap_c is None):
                raise ValueError(
                    "BS-DISP results not available. "
                    "Both bootstrap and DISP results must be loaded."
                )
                    
            # Combine bootstrap and displacement uncertainties
            combined_uncertainty = np.sqrt(
                self.pmf.dfBS_profile_c.std()**2 + 
                self.pmf.df_uncertainties_summary_c**2
            )
                
            results = {
                'combined_uncertainty': combined_uncertainty,
                'bootstrap_results': self.pmf.dfBS_profile_c,
                'disp_results': self.pmf.df_disp_swap_c
            }
        else:
            raise ValueError(f"Unknown uncertainty estimation method: {method}")
                
        return results

    def compute_profile_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                factor1: Optional[str] = None, 
                                factor2: Optional[str] = None, 
                                method: str = "correlation",
                                normalize_to_mass: bool = True) -> Union[float, Dict[str, float]]:
        """
        Compute similarity between factor profiles using various metrics.
        
        Parameters
        ----------
        df1, df2 : pd.DataFrame
            Profile matrices to compare
        factor1, factor2 : str, optional
            Specific factors to compare
        method : str
            Similarity metric to use:
            - "correlation": Pearson correlation (PD = 1-r²)
            - "SID": Standardized Identity Distance
            - "both": Returns both metrics
        normalize_to_mass : bool, default=True
            Whether to normalize profiles to total mass fraction first
        """
        # Validate input data
        if df1 is None or df2 is None:
            raise ValueError("Both profile matrices must be provided")
            
        if factor1 and factor1 not in df1.columns:
            available_factors = df1.columns.tolist()
            raise ValueError(
                f"Factor '{factor1}' not found in first profile matrix. "
                f"Available factors: {available_factors}"
            )
            
        if factor2 and factor2 not in df2.columns:
            available_factors = df2.columns.tolist()
            raise ValueError(
                f"Factor '{factor2}' not found in second profile matrix. "
                f"Available factors: {available_factors}"
            )
        
        p1, p2 = self._prepare_profile_for_comparison(df1, df2, factor1, factor2, 
                                                    normalize_to_mass)
        
        if p1 is None or p2 is None:
            return np.nan
        
        common_species = p1.index.intersection(p2.index)
        if len(common_species) <= 3:
            logger.warning(
                f"Insufficient common species ({len(common_species)}) for reliable comparison. "
                "At least 4 common species are recommended."
            )
            return np.nan
        
        return compute_similarity_metrics(p1, p2)

    def compute_bootstrap_similarity(self, base_profiles: pd.DataFrame, bs_profiles: pd.DataFrame, 
                                    method: str = "correlation") -> pd.DataFrame:
        """
        Compute similarity metrics between base run and bootstrap profiles.
        
        Parameters
        ----------
        base_profiles : pd.DataFrame
            Base run factor profiles
        bs_profiles : pd.DataFrame  
            Bootstrap run factor profiles
        method : str
            Similarity metric to use
            
        Returns
        -------
        pd.DataFrame
            Matrix of similarity values between base and BS profiles
        """
        similarity_matrix = pd.DataFrame(
            index=base_profiles.columns,
            columns=bs_profiles.columns
        )
        
        for base_factor in base_profiles.columns:
            for bs_factor in bs_profiles.columns:
                sim = self.compute_profile_similarity(
                    base_profiles, 
                    bs_profiles,
                    base_factor,
                    bs_factor,
                    method=method
                )
                similarity_matrix.loc[base_factor, bs_factor] = sim
                
        return similarity_matrix


    def compare_runs(self, other_run: pd.DataFrame, metric: str = 'SID',
                    normalize: bool = True) -> pd.DataFrame:
        """
        Compare current PMF run with another one using specified metric.
        
        Parameters
        ----------
        other_run : pd.DataFrame
            Factor profiles from another PMF run
        metric : str
            Distance metric to use
        normalize : bool, default=True
            Whether to normalize profiles first
            
        Returns
        -------
        pd.DataFrame
            Distance matrix between current and other run factors
        """
        current_profiles = self.pmf.dfprofiles_c if self.pmf.dfprofiles_c is not None \
                         else self.pmf.dfprofiles_b
        return compute_distance_matrix(
            pd.concat([current_profiles, other_run], axis=1),
            metric=metric,
            normalize=normalize
        )

    def bootstrap_analysis(self, n_bootstrap: int = 100) -> dict:
        """
        Perform bootstrap analysis on the current solution using random resampling.

        Parameters
        ----------
        n_bootstrap : int, default=100
            Number of bootstrap resamples to generate.

        Returns
        -------

        dict
            Dictionary containing bootstrap statistics, such as confidence intervals
            for factor loadings and contributions.

        Examples
        --------
        >>> analysis = PMFAnalysis(pmf_obj)
        >>> results = analysis.bootstrap_analysis(n_bootstrap=500)
        >>> print(results["bootstrap_CIs"])
        """
        # Implementation of bootstrap analysis
        pass

    def explained_variation(self, constrained: bool = True) -> pd.DataFrame:
        """
        Calculate explained variation by each factor for each species.
        
        Parameters
        ----------
        constrained : bool, default=True
            Use constrained or base run
            
        Returns
        -------
        pd.DataFrame
            Explained variation matrix
        """
        profiles = self.pmf.dfprofiles_c if constrained else self.pmf.dfprofiles_b
        contrib = self.pmf.dfcontrib_c if constrained else self.pmf.dfcontrib_b
        
        explained_var = pd.DataFrame(index=profiles.index, 
                                   columns=profiles.columns,
                                   dtype=float)
        
        for species in profiles.index:
            total_var = 0
            for factor in profiles.columns:
                var = np.var(contrib[factor] * profiles.loc[species, factor])
                explained_var.loc[species, factor] = var
                total_var += var
            explained_var.loc[species] /= total_var
            
        return explained_var

    def factor_temporal_correlation(self, constrained: bool = True) -> pd.DataFrame:
        """
        Compute temporal correlation between factors.
        
        Parameters
        ----------
        constrained : bool, default=True
            Use constrained or base run
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix between factor time series
        """
        contrib = self.pmf.dfcontrib_c if constrained else self.pmf.dfcontrib_b
        return contrib.corr()

    def detect_mixed_factors(self, threshold: float = 0.6) -> List[Tuple[str, str]]:
        """
        Detect potentially mixed factors based on temporal correlation.
        
        Parameters
        ----------
        threshold : float, default=0.6
            Correlation threshold for considering factors as mixed
            
        Returns
        -------
        list of tuple
            Pairs of potentially mixed factors
        """
        try:
            corr = self.factor_temporal_correlation()
            mixed_pairs = []
            
            # Find pairs with correlation above threshold
            for i, col1 in enumerate(corr.columns):
                for j, col2 in enumerate(corr.columns[i+1:], i+1):
                    if abs(corr.loc[col1, col2]) > threshold:
                        mixed_pairs.append((col1, col2, float(corr.loc[col1, col2])))
                        
            # Sort by correlation strength
            mixed_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                        
            return [(f1, f2) for f1, f2, _ in mixed_pairs]
            
        except Exception as e:
            print(f"Error detecting mixed factors: {str(e)}")
            return []

    def compute_model_diagnostics(self, constrained: bool = True) -> dict:
        """
        Compute comprehensive model diagnostics.
        
        Parameters
        ----------
        constrained : bool, default=True
            Use constrained or base run
            
        Returns
        -------
        dict
            Model diagnostic metrics
        """
        # Get appropriate matrices
        profiles = self.pmf.dfprofiles_c if constrained else self.pmf.dfprofiles_b
        contrib = self.pmf.dfcontrib_c if constrained else self.pmf.dfcontrib_b
        
        # Convert to numpy arrays
        X = np.array(self.pmf.data)  # Original data
        G = np.array(contrib)  # Factor contributions
        F = np.array(profiles).T  # Factor profiles
        S = np.array(self.pmf.uncertainties)  # Uncertainties
        
        # Compute various diagnostics
        diagnostics = {}
        
        # Q values
        diagnostics.update(
            compute_Q_values(X, G, F, S)
        )
        
        # R² values
        diagnostics['R2_by_species'] = compute_r2_matrix(X, G, F)
        
        # Scaled residuals
        residuals = compute_scaled_residuals(X, G, F, S)
        diagnostics.update({
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'residuals_skew': stats.skew(residuals.flatten()),
            'residuals_kurtosis': stats.kurtosis(residuals.flatten())
        })
        
        # Rotational diagnostics
        rot_stats = assess_rotational_ambiguity(G, F)
        diagnostics['rotational_stats'] = rot_stats
        
        return diagnostics

    def _prepare_profile_for_comparison(self, df1, df2, factor1=None, factor2=None, normalize_to_mass=True):
        """Helper method to prepare profiles for comparison analysis."""
        if isinstance(df1, pd.Series) and isinstance(df2, pd.Series):
            p1 = df1
            p2 = df2
        else:
            if not(factor2):
                factor2 = factor1

            if factor1 not in df1.dropna(axis=1, how="all").columns:
                return (None, None)

            if factor2 not in df2.dropna(axis=1, how="all").columns:
                return (None, None)

            p1 = df1.loc[:, factor1]
            p2 = df2.loc[:, factor2]

        if not normalize_to_mass:
            p1 = self.pmf.to_relative_mass(p1)
            p2 = self.pmf.to_relative_mass(p2)

        # Remove PM specie because we compare normalized to PM, so everytime it will be a
        # perfect 1:1 for this specie.
        if p1.index.str.contains("PM").any():
            p1 = p1.loc[~p1.index.str.contains("PM")]
        if p2.index.str.contains("PM").any():
            p2 = p2.loc[~p2.index.str.contains("PM")]

        return (p1, p2)

    def compute_bootstrap_statistics(self, 
                                   profiles: Optional[List[str]] = None,
                                   confidence_level: float = 0.95) -> Dict[str, pd.DataFrame]:
        """
        Compute comprehensive bootstrap statistics with error handling.
        
        Parameters
        ----------
        profiles : list of str, optional
            Specific profiles to analyze
        confidence_level : float, default=0.95
            Confidence level for intervals (0-1)
            
        Returns
        -------
        dict
            Dictionary of DataFrames with bootstrap statistics for each profile
        
        Raises
        ------
        ValueError
            If bootstrap results are not available
        """
        if self.pmf.dfBS_profile_c is None:
            raise ValueError("Bootstrap results not available")
            
        bs_data = self.pmf.dfBS_profile_c
        
        # Use all profiles if none specified
        if profiles is None:
            profiles = self.pmf.profiles
            
        results = {}
        alpha = (1 - confidence_level) / 2
        
        for profile in profiles:
            try:
                profile_data = bs_data.xs(profile, level="Profile")
                
                # Calculate statistics
                stats = {
                    'mean': profile_data.mean(axis=1),
                    'median': profile_data.median(axis=1),
                    'std': profile_data.std(axis=1),
                    'cv': profile_data.std(axis=1) / profile_data.mean(axis=1),
                    f'ci_lower_{int(confidence_level*100)}': profile_data.quantile(alpha, axis=1),
                    f'ci_upper_{int(confidence_level*100)}': profile_data.quantile(1-alpha, axis=1)
                }
                
                results[profile] = pd.DataFrame(stats)
                
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process bootstrap data for {profile}: {str(e)}")
        
        if not results:
            raise ValueError("No valid bootstrap results found for any profile")
            
        return results

    def compute_Q_values(self, constrained: bool = True) -> Dict[str, float]:
        """
        Compute Q (robust and true) and Qexpected values.

        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run results.

        Returns
        -------
        Dict[str, float]
            Dictionary containing 'Q', 'Q_expected', and 'Q_over_Qexp'.
            Returns NaNs if required data is missing.
        """
        if not hasattr(self.pmf, 'ensure_data_loaded'):
            # Fallback if ensure_data_loaded doesn't exist
            if (constrained and (self.pmf.dfcontrib_c is None or self.pmf.dfprofiles_c is None) or
                not constrained and (self.pmf.dfcontrib_b is None or self.pmf.dfprofiles_b is None)):
                return {"Q": np.nan, "Q_expected": np.nan, "Q_over_Qexp": np.nan}
        elif not self.pmf.ensure_data_loaded(check_g=True, check_f=True, check_x=True, check_s=True, constrained=constrained):
            return {"Q": np.nan, "Q_expected": np.nan, "Q_over_Qexp": np.nan}

        # Get data matrices based on constrained flag
        if not hasattr(self.pmf, 'data') or not hasattr(self.pmf, 'uncertainties'):
            return {"Q": np.nan, "Q_expected": np.nan, "Q_over_Qexp": np.nan}
            
        X = self.pmf.data.values
        S = self.pmf.uncertainties.values
        G = self.pmf.dfcontrib_c.values if constrained else self.pmf.dfcontrib_b.values
        F = self.pmf.dfprofiles_c.values if constrained else self.pmf.dfprofiles_b.values

        residuals = X - G @ F.T
        scaled_residuals = residuals / S
        
        # Q value (robust Q, sum of squares of scaled residuals)
        Q = np.sum(scaled_residuals**2)
        
        # Qexpected calculation
        n, m = X.shape
        p = G.shape[1]
        # Degrees of freedom: nm - np - mp = nm - p(n+m) ? Check EPA PMF docs for exact formula.
        # Assuming degrees of freedom is roughly nm for large datasets, or more precisely nm - p(n+m) if non-zero constraints exist.
        # A simpler approximation often used is just n * m.
        # Let's use n * m for now, but this might need refinement based on PMF theory/constraints.
        Q_expected = n * m 
        # More refined Qexp might consider constraints or factor count: Qexp = n*m - p*(n+m)
        # Q_expected = max(1, n * m - p * (n + m)) # Ensure non-negative

        return {
            "Q": Q,
            "Q_expected": Q_expected,
            "Q_over_Qexp": Q / max(Q_expected, 1)  # Avoid division by zero
        }

    def compute_r2_matrix(self, constrained: bool = True) -> Optional[pd.DataFrame]:
        """
        Compute R² for each species across all factors.

        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run results.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with R² values for each species, or None if data is missing.
        """
        if not self.pmf.ensure_data_loaded(check_g=True, check_f=True, check_x=True, constrained=constrained):
            return None

        X = self.pmf.data.values
        G = self.pmf.dfcontrib_c.values if constrained else self.pmf.dfcontrib_b.values
        F = self.pmf.dfprofiles_c.values if constrained else self.pmf.dfprofiles_b.values

        # Reconstructed matrix
        X_reconstructed = G @ F.T

        # Calculate R² for each column (species)
        r2_values = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            # Calculate correlation coefficient, handle potential NaNs or zero variance
            valid_mask = ~np.isnan(X[:, i]) & ~np.isnan(X_reconstructed[:, i])
            if np.sum(valid_mask) < 2 or np.std(X[valid_mask, i]) < 1e-9 or np.std(X_reconstructed[valid_mask, i]) < 1e-9:
                 r2_values[i] = np.nan # Not enough data or no variance
            else:
                corr = np.corrcoef(X[valid_mask, i], X_reconstructed[valid_mask, i])[0, 1]
                r2_values[i] = corr**2

        return pd.DataFrame(r2_values, index=self.pmf.species, columns=["R2"])

    def compute_scaled_residuals(self, constrained: bool = True) -> Optional[pd.DataFrame]:
        """
        Compute scaled residuals for PMF solution.

        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run results.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame of scaled residuals, or None if data is missing.
        """
        if not self.pmf.ensure_data_loaded(check_g=True, check_f=True, check_x=True, check_s=True, constrained=constrained):
            return None

        X = self.pmf.data
        S = self.pmf.uncertainties
        G = self.pmf.dfcontrib_c if constrained else self.pmf.dfcontrib_b
        F = self.pmf.dfprofiles_c if constrained else self.pmf.dfprofiles_b

        X_reconstructed = pd.DataFrame(G.values @ F.T.values, index=X.index, columns=X.columns)
        
        # Ensure alignment and handle potential missing values if needed
        residuals = X - X_reconstructed
        scaled_residuals = residuals / S
        
        return scaled_residuals

    def compute_signal_to_noise(self) -> Optional[pd.DataFrame]:
        """
        Calculate Signal-to-Noise (S/N) ratio for each species.
        Based on draft_utils.Input_preparation.Input_SN_cal logic.

        S/N = mean( (Concentration - Uncertainty) / Uncertainty ) for Conc > Unc, else 0

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with S/N ratio for each species, or None if data missing.
        """
        if not self.pmf.ensure_data_loaded(check_x=True, check_s=True):
            return None
            
        conc = self.pmf.data
        unc = self.pmf.uncertainties
        
        # Calculate S/N for each data point
        sn_pointwise = (conc - unc) / unc
        sn_pointwise[conc <= unc] = 0
        sn_pointwise[unc < 1e-10] = np.nan # Avoid division by zero if uncertainty is zero

        # Calculate mean S/N for each species
        sn_mean = sn_pointwise.mean(axis=0)
        
        return pd.DataFrame(sn_mean, columns=['S/N'])

    def compute_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Calculate the correlation matrix for the input species data.
        Based on draft_utils.Input_preparation.Input_correlation.

        Returns
        -------
        Optional[pd.DataFrame]
            Correlation matrix, or None if data is missing.
        """
        if not self.pmf.ensure_data_loaded(check_x=True):
            return None
        
        # Ensure data is numeric, handle potential non-numeric entries if necessary
        # Assuming self.pmf.data is already numeric after preprocessing/loading
        try:
            corr_matrix = self.pmf.data.corr()
            return corr_matrix
        except TypeError:
            warnings.warn("Could not compute correlation matrix. Data might contain non-numeric values.")
            return None


    def save4deltaTool(self, output_dir: str = ".", constrained: bool = True, zip_output: bool = True):
        """
        Save PMF results (Profiles, Contributions) in a format compatible
        with the DeltaTool software (zipped CSV files).
        
        DeltaTool is specialized software for comparing source profiles from 
        different studies and receptor models. This method exports the PMF 
        results in the specific format required by DeltaTool.

        Parameters
        ----------
        output_dir : str, default="."
            Directory where the output zip file or CSVs will be saved.
        constrained : bool, default=True
            Whether to use constrained or base run results.
        zip_output : bool, default=True
            If True, creates a zip file containing the CSVs. Otherwise, saves individual CSVs.
            
        Notes
        -----
        Creates three files:
        - CONC.csv: Factor profiles (F matrix)
        - TREND.csv: Factor contributions over time 
        - CONTR.csv: Relative contribution of each factor to each species
        
        References
        ----------
        Belis, C.A., et al., 2015. European Guide on Air Pollution Source
        Apportionment with Receptor Models. JRC Technical Reports, EUR 26080.
        https://doi.org/10.2788/9307
        """
        if not self.pmf.ensure_data_loaded(check_g=True, check_f=True, constrained=constrained):
            warnings.warn("Cannot save for DeltaTool: Missing G or F matrices.")
            return

        site = self.pmf._site
        total_var = self.pmf.totalVar or 'PM_Guess' # Use totalVar or a placeholder

        F_df = self.pmf.dfprofiles_c if constrained else self.pmf.dfprofiles_b
        G_df = self.pmf.dfcontrib_c if constrained else self.pmf.dfcontrib_b

        # Prepare F matrix (CONC.csv) - Rename species if needed
        F_delta = F_df.rename(index={"OC*": "OC", "SO42-": "SO4="}, errors='ignore')
        F_delta.index.name = "CON (µg/m3)" # DeltaTool expects this header

        # Prepare G*F[totalVar] matrix (TREND.csv)
        if total_var not in F_df.index:
             warnings.warn(f"Total variable '{total_var}' not found in profiles. Cannot calculate TREND correctly.")
             trend_df = pd.DataFrame(index=G_df.index) # Empty df
        else:
             trend_df = G_df.multiply(F_df.loc[total_var], axis=1)
        trend_df.index.name = "TREND (µg/m3)" # DeltaTool expects this header

        # Prepare contribution to species matrix (CONTR.csv) - F normalized by column
        contr_specie_df = (F_df / F_df.sum(axis=0) * 100).copy() 
        contr_specie_df = contr_specie_df.rename(index={"OC*": "OC", "SO42-": "SO4="}, errors='ignore')
        contr_specie_df.index.name = "CONTR" # DeltaTool expects this header

        # Define filenames
        conc_file = "CONC.csv"
        trend_file = "TREND.csv"
        contr_file = "CONTR.csv"
        output_prefix = os.path.join(output_dir, f"{site}_constrained" if constrained else f"{site}_base")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        if zip_output:
            zip_filename = f"{output_prefix}_4deltaTool.zip"
            list_files = {conc_file: F_delta, trend_file: trend_df, contr_file: contr_specie_df}
            
            # Write CSVs temporarily then add to zip
            temp_files = []
            try:
                for fname, df_to_save in list_files.items():
                    temp_path = os.path.join(output_dir, f"_{fname}") # Temp prefix
                    df_to_save.to_csv(temp_path, date_format='%m/%d/%Y')
                    temp_files.append((temp_path, fname)) # Store temp path and final name in zip

                with ZipFile(zip_filename, "w") as zipf:
                    for temp_path, final_name in temp_files:
                        zipf.write(temp_path, arcname=final_name)
                print(f"DeltaTool files saved to {zip_filename}")

            finally:
                # Clean up temporary files
                for temp_path, _ in temp_files:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            # Save individual CSVs
            F_delta.to_csv(f"{output_prefix}_CONC.csv", date_format='%m/%d/%Y')
            trend_df.to_csv(f"{output_prefix}_TREND.csv", date_format='%m/%d/%Y')
            contr_specie_df.to_csv(f"{output_prefix}_CONTR.csv", date_format='%m/%d/%Y')
            print(f"DeltaTool files saved to {output_dir} with prefix {output_prefix}")

    # Add other analysis methods derived from Input_preparation if needed
    # e.g., tracer identification
    def identify_tracers(self, corr_threshold: float = 0.3, sn_threshold: float = 3.0) -> Optional[pd.Index]:
        """
        Identify potential tracer species based on low correlation with others
        and high signal-to-noise ratio.
        Based on draft_utils.Input_preparation.Input_select_tracer.

        Parameters
        ----------
        corr_threshold : float, default=0.3
            Maximum absolute correlation with other species.
        sn_threshold : float, default=3.0
            Minimum signal-to-noise ratio.

        Returns
        -------
        Optional[pd.Index]
            Index of potential tracer species, or None if data is missing.
        """
        corr_matrix = self.compute_correlation_matrix()
        sn_ratios = self.compute_signal_to_noise()

        if corr_matrix is None or sn_ratios is None:
            return None

        # Find species with max correlation below threshold
        # We check the maximum *absolute* correlation, excluding self-correlation (which is 1)
        max_abs_corr = corr_matrix.abs().mask(np.equal(*np.indices(corr_matrix.shape))).max()
        low_corr_species = max_abs_corr[max_abs_corr < corr_threshold].index

        # Find species with S/N above threshold
        high_sn_species = sn_ratios[sn_ratios['S/N'] > sn_threshold].index

        # Intersection of the two sets
        tracers = low_corr_species.intersection(high_sn_species)
        return tracers

# Analysis utility functions - simplified
def compute_Q_values(X: np.ndarray, G: np.ndarray, F: np.ndarray, S: np.ndarray) -> Dict[str, float]:
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
        Dictionary containing Q, Qexpected, and Q/Qexpected
    """
    # Reconstructed matrix
    X_reconstructed = G @ F.T
    
    # Residuals
    R = X - X_reconstructed
    
    # Scaled residuals
    scaled_residuals = R / S
    
    # Q value is sum of squares of scaled residuals
    Q = np.nansum(scaled_residuals**2)
    
    # Expected Q: (rows * cols - factors * (rows + cols))
    rows, cols = X.shape
    factors = G.shape[1]
    Q_expected = rows * cols - factors * (rows + cols)
    
    return {
        "Q": Q,
        "Q_expected": Q_expected,
        "Q_over_Qexp": Q / max(Q_expected, 1)  # Avoid division by zero
    }

def compute_r2_matrix(X: np.ndarray, G: np.ndarray, F: np.ndarray) -> pd.DataFrame:
    """
    Compute R² for each species across all factors.
    
    Parameters
    ----------
    X : np.ndarray
        Original data matrix
    G : np.ndarray
        Factor contributions matrix
    F : np.ndarray
        Factor profiles matrix
        
    Returns
    -------
    pd.DataFrame
        R² values for each species
    """
    # Reconstructed matrix
    X_reconstructed = G @ F.T
    
    # Calculate R² for each column (species)
    r2_values = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        # Calculate correlation coefficient
        corr = np.corrcoef(X[:, i], X_reconstructed[:, i])[0, 1]
        r2_values[i] = corr**2
    
    return pd.DataFrame(r2_values, columns=["R2"])

def compute_scaled_residuals(X: np.ndarray, G: np.ndarray, F: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Compute scaled residuals for PMF solution.
    
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
    np.ndarray
        Matrix of scaled residuals
    """
    # Reconstructed matrix
    X_reconstructed = G @ F.T
    
    # Residuals
    R = X - X_reconstructed
    
    # Scaled residuals
    scaled_residuals = R / S
    
    return scaled_residuals

def assess_rotational_ambiguity(G: np.ndarray, F: np.ndarray) -> Dict[str, float]:
    """
    Assess rotational ambiguity in PMF solution.
    
    Parameters
    ----------
    G : np.ndarray
        Factor contributions matrix
    F : np.ndarray
        Factor profiles matrix
        
    Returns
    -------
    Dict[str, float]
        Statistics about rotational ambiguity
    """
    # Calculate correlations between factors
    G_corr = np.corrcoef(G, rowvar=False)
    
    # Get maximum absolute correlation (excluding diagonal)
    n = G_corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    max_corr = np.max(np.abs(G_corr[mask]))
    
    # Calculate maximum dQ/dC from EPA PMF (simplified approach)
    # This is a heuristic calculation
    rotmat = G.T @ G
    dQdC_values = []
    
    for i in range(n):
        for j in range(i+1, n):
            if i != j:
                g_i = G[:, i]
                g_j = G[:, j]
                f_i = F[:, i]
                f_j = F[:, j]
                
                # Calculate dQ/dC using the formula from Paatero et al.
                dQdC = 4 * abs(np.sum(g_i * g_j) * np.sum(f_i * f_j))
                dQdC_values.append(dQdC)
    
    max_dQdC = max(dQdC_values) if dQdC_values else 0
    
    return {
        'max_factor_correlation': max_corr,
        'max_dQdC': max_dQdC,
        'rotational_stability': 1 / (1 + max_dQdC)  # Higher values indicate better stability
    }

def compute_distance_matrix(df: pd.DataFrame, metric: str = 'SID', normalize: bool = True) -> pd.DataFrame:
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
        Distance matrix
    """
    n_profiles = len(df.columns)
    distance_matrix = pd.DataFrame(np.zeros((n_profiles, n_profiles)), 
                                  index=df.columns, columns=df.columns)
    
    if normalize:
        norm_factor = df.sum(axis=0)
        df_norm = df.div(norm_factor, axis=1)
    else:
        df_norm = df
    
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i == j:
                continue
                
            distance_matrix.loc[col1, col2] = compute_similarity_metrics(df_norm[col1], df_norm[col2])[metric]
    
    return distance_matrix

def compute_SID(df1, df2, source1, source2=None, isRelativeMass=True):
    """
    Compute the Scaled Identity Distance (SID) between two source profiles.
    
    SID measures the dissimilarity between two profiles, with values ranging from 0 (identical) to 2 (completely different).
    Values below 1 generally indicate good similarity.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        First profile dataframe with species as index and sources as columns
    df2 : pd.DataFrame
        Second profile dataframe with species as index and sources as columns
    source1 : str
        Source name in df1
    source2 : str, optional
        Source name in df2. If None, uses the same name as source1
    isRelativeMass : bool, default True
        Whether the profiles are already normalized to relative mass.
        If False, they will be normalized first.
        
    Returns
    -------
    float
        The SID value between the two profiles
    """
    if not source2:
        source2 = source1
    if source1 not in df1.dropna(axis=1, how="all").columns:
        return np.nan
    if source2 not in df2.dropna(axis=1, how="all").columns:
        return np.nan
    
    if not isRelativeMass:
        # Find total variable - typically PM10 or PM2.5
        totalVar = None
        for var in ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons", "Total_1", "Total_2", "Total", "TC"]:
            if var in df1.index and var in df2.index:
                totalVar = var
                break
                
        if totalVar is None:
            # Try to guess total variable
            potential_vars = [x for x in df1.index if "PM" in x]
            if potential_vars:
                totalVar = potential_vars[0]
            else:
                raise ValueError("Could not identify a total variable for normalization")
        
        # Normalize to relative mass
        p1 = df1.loc[:, source1] / df1.loc[totalVar, source1]
        p2 = df2.loc[:, source2] / df2.loc[totalVar, source2]
    else:
        p1 = df1.loc[:, source1]
        p2 = df2.loc[:, source2]
    
    # Find common species
    sp = p1.index.intersection(p2.index)
    
    if len(sp) > 3:  # Need at least 4 common species for meaningful comparison
        ID = np.abs(p1[sp] - p2[sp])  # Identity difference
        MAD = p1[sp] + p2[sp]  # Sum of the profiles
        SID = np.sqrt(2) / len(sp) * (ID / MAD).sum()
    else:
        SID = np.nan
        
    return SID

def compute_PD(df1, df2, source1, source2=None, isRelativeMass=True):
    """
    Compute the Pearson Distance (PD) between two source profiles.
    
    PD is defined as 1-R², where R is the Pearson correlation coefficient.
    Values range from 0 (perfect correlation) to 1 (no correlation).
    
    Parameters
    ----------
    df1 : pd.DataFrame
        First profile dataframe with species as index and sources as columns
    df2 : pd.DataFrame
        Second profile dataframe with species as index and sources as columns
    source1 : str
        Source name in df1
    source2 : str, optional
        Source name in df2. If None, uses the same name as source1
    isRelativeMass : bool, default True
        Whether the profiles are already normalized to relative mass.
        If False, they will be normalized first.
        
    Returns
    -------
    float
        The PD value between the two profiles
    """
    if not source2:
        source2 = source1
    if source1 not in df1.dropna(axis=1, how="all").columns:
        return np.nan
    if source2 not in df2.dropna(axis=1, how="all").columns:
        return np.nan
    
    if not isRelativeMass:
        # Find total variable - typically PM10 or PM2.5
        totalVar = None
        for var in ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons", "Total_1", "Total_2", "Total", "TC"]:
            if var in df1.index and var in df2.index:
                totalVar = var
                break
                
        if totalVar is None:
            # Try to guess total variable
            potential_vars = [x for x in df1.index if "PM" in x]
            if potential_vars:
                totalVar = potential_vars[0]
            else:
                raise ValueError("Could not identify a total variable for normalization")
                
        # Normalize to relative mass
        p1 = df1.loc[:, source1] / df1.loc[totalVar, source1]
        p2 = df2.loc[:, source2] / df2.loc[totalVar, source2]
    else:
        p1 = df1.loc[:, source1]
        p2 = df2.loc[:, source2]
    
    # Drop NaN values
    p1 = p1.dropna()
    p2 = p2.dropna()
    
    # Find common species
    sp = p1.index.intersection(p2.index)
    
    if len(sp) > 3:  # Need at least 4 common species for meaningful comparison
        # Calculate Pearson correlation coefficient and convert to distance
        PD = 1 - np.corrcoef(p1[sp], p2[sp])[1, 0] ** 2
    else:
        PD = np.nan
        
    return PD

def compute_COD(df1, df2, source1, source2=None, isRelativeMass=True):
    """
    Compute the Coefficient of Divergence (COD) between two source profiles.
    
    COD measures the degree of divergence between two profiles, with values
    ranging from 0 (identical) to 1 (completely different).
    
    Parameters
    ----------
    df1 : pd.DataFrame
        First profile dataframe with species as index and sources as columns
    df2 : pd.DataFrame
        Second profile dataframe with species as index and sources as columns
    source1 : str
        Source name in df1
    source2 : str, optional
        Source name in df2. If None, uses the same name as source1
    isRelativeMass : bool, default True
        Whether the profiles are already normalized to relative mass.
        If False, they will be normalized first.
        
    Returns
    -------
    float
        The COD value between the two profiles
    """
    if not source2:
        source2 = source1
    if source1 not in df1.dropna(axis=1, how="all").columns:
        return np.nan
    if source2 not in df2.dropna(axis=1, how="all").columns:
        return np.nan
    
    if not isRelativeMass:
        # Find total variable - typically PM10 or PM2.5
        totalVar = None
        for var in ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons", "Total_1", "Total_2", "Total", "TC"]:
            if var in df1.index and var in df2.index:
                totalVar = var
                break
                
        if totalVar is None:
            # Try to guess total variable
            potential_vars = [x for x in df1.index if "PM" in x]
            if potential_vars:
                totalVar = potential_vars[0]
            else:
                raise ValueError("Could not identify a total variable for normalization")
                
        # Normalize to relative mass
        p1 = df1.loc[:, source1] / df1.loc[totalVar, source1]
        p2 = df2.loc[:, source2] / df2.loc[totalVar, source2]
    else:
        p1 = df1.loc[:, source1]
        p2 = df2.loc[:, source2]
    
    # Find common species
    sp = p1.index.intersection(p2.index)
    
    if len(sp) > 3:  # Need at least 4 common species for meaningful comparison
        # Calculate Coefficient of Divergence
        p1_sp = p1[sp]
        p2_sp = p2[sp]
        
        # Filter out zeros to avoid division by zero
        non_zero = (p1_sp != 0) & (p2_sp != 0)
        p1_sp = p1_sp[non_zero]
        p2_sp = p2_sp[non_zero]
        
        if len(p1_sp) < 3:  # Ensure we still have enough data points
            return np.nan
            
        # Calculate COD
        COD = np.sqrt(np.mean(((p1_sp - p2_sp) / (p1_sp + p2_sp)) ** 2))
    else:
        COD = np.nan
        
    return COD
