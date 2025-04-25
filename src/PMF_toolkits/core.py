import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

# Import MultisitesReader explicitly if it's used here
from .readers import XlsxReader, MultisitesReader 
from .utils import get_sourcesCategories, add_season
from .visualization import PMFVisualization
from .analysis import PMFAnalysis
from .preprocessing import PMFPreprocessor

class PMF:
    """
    PMF output handler with integrated visualization and analysis capabilities.

    This class provides a comprehensive interface for working with PMF (Positive Matrix 
    Factorization) results, including data loading, visualization, and statistical analysis.

    Key Features:
    ------------
    - Automatic handling of EPA PMF5.0 output files
    - Flexible plotting interface through .plot attribute
    - Statistical analysis and validation tools
    - Multi-site data support
    - Bootstrap and error estimation

    Attributes
    ----------
    totalVar : str
        Total variable name (e.g., "PM10", "PM2.5"). When not explicitly set,
        the system attempts to infer it using this priority:
        1. Look for common names: PM10, PM2.5, PMrecons, PM10rec, etc.
        2. Search for any variable containing "PM" in the name
        3. Raise an error if no suitable variable is found
        
        You can always explicitly set this using:
        >>> pmf.totalVar = "your_total_variable"

    plot : PMFVisualization
        Primary plotting interface providing all visualization methods.
        This is the recommended way to create plots:
        >>> pmf.plot.plot_factor_profiles()
        >>> pmf.plot.plot_contributions_timeseries()

    visualization : PMFVisualization
        Same as plot, provided for compatibility.

    analysis : PMFAnalysis
        Statistical analysis and validation tools.

    Examples
    --------
    >>> from PMF_toolkits import PMF
    >>> # Load PMF results from Excel files
    >>> pmf = PMF(site="urban_site", reader="xlsx", BDIR="pmf_outputs/")
    >>> pmf.read.read_all()
    >>> 
    >>> # Plot factor profiles
    >>> pmf.plot.plot_factor_profiles()
    >>> 
    >>> # Get contributions in μg/m³
    >>> contributions = pmf.to_cubic_meter()
    """
    def __init__(self, site: str, reader: Optional[str] = None, savedir: str = "./", 
                BDIR: Optional[str] = None, multisites: bool = False):
        """PMF object from output of EPAPMF5.

        Parameters
        ----------

        site : str, the name of the site (prefix of each files if outputed in xlsx)
        reader : str, 'xlsx' or None
            Format of the saved output of the PMF
            
            - xlsx : saved as xlsx output. Need to specify also BDIR. 
                     Use `multisites=True` for multi-site Excel files.
            - None : don't initialize a reader (for custom readers)

        savedir : str, default current path
            Path to directory to save the figures

        BDIR : str, the directory where the xlsx files live, if outputed in xlsx
        multisites : bool, default False
            Indicates if the Excel files contain data for multiple sites 
            (affects how 'xlsx' reader behaves).
        """
        self._site = site
        self.read = None # Initialize read attribute

        if reader == "xlsx":
            if BDIR is None:
                 raise ValueError("BDIR must be specified when reader='xlsx'")
            self.read = XlsxReader(BDIR=BDIR, site=site, pmf=self, multisites=multisites)
            print(f"Initialized XlsxReader for site '{site}' with multisites={multisites}")
        elif reader is not None:
             raise ValueError(f"Unsupported reader type: {reader}")

        self.analysis = PMFAnalysis(self)
        self.visualization = PMFVisualization(self, savedir=savedir)
        # Use visualization as the plotting backend but expose it through plot
        self.plot = self.visualization
        
        self._init_data_containers()

    def _init_data_containers(self):
        """Initialize data containers with None values."""
        self.profiles = self.nprofiles = self.species = self.nspecies = self.totalVar = None
        self.dfprofiles_b = self.dfcontrib_b = self.dfprofiles_c = self.dfcontrib_c = None
        self.dfBS_profile_b = self.dfBS_profile_c = self.dfbootstrap_mapping_b = self.dfbootstrap_mapping_c = None
        self.df_disp_swap_b = self.df_disp_swap_c = self.df_uncertainties_summary_b = self.df_uncertainties_summary_c = None
        self.data = self.uncertainties = None

    @classmethod
    def from_data(cls, data: pd.DataFrame, uncertainties: Optional[pd.DataFrame] = None, 
                 site: str = "custom", **kwargs):
        """
        Create a PMF object directly from data and uncertainties.

        Parameters
        ----------
        data : pd.DataFrame
            Measured data with samples as rows and species as columns.
        uncertainties : pd.DataFrame, optional
            Uncertainties associated with the data.
        site : str, default="custom"
            Label for the dataset, used in storing results.
        **kwargs
            Additional PMF configuration arguments.

        Returns
        -------
        PMF
            An instance of PMF initialized with the provided data.

        Examples
        --------
        >>> df_data = pd.read_csv("measurements.csv")
        >>> df_unc = pd.read_csv("uncertainty.csv")
        >>> pmf = PMF.from_data(df_data, df_unc, site="urban")
        """
        pmf = cls(site=site, **kwargs)
        pmf.data = data
        pmf.uncertainties = uncertainties if uncertainties is not None else pd.DataFrame(0, index=data.index, columns=data.columns)
        pmf.species = data.columns.tolist()
        pmf.nspecies = len(pmf.species)
        return pmf

    def to_cubic_meter(self, specie=None, constrained=True, profiles=None):
        """Convert the contribution in cubic meter for the given specie

        Parameters
        ----------
        constrained : Boolean, default True
            Use constrained results instead of base results
        specie : str, the specie, default totalVar
            The species to convert to concentration
        profiles : list of profile, default all profiles
            The specific profiles to include

        Returns
        -------
        df : pd.DataFrame
            DataFrame with contributions in μg/m³
        """
        if not self.ensure_data_loaded():
            raise ValueError("Required data not loaded. Call read_base_profiles() or read_constrained_profiles() first.")

        if specie is None:
            specie = self.totalVar
            if specie is None:
                # Try to infer totalVar if not set
                if self.species:
                    TOTALVAR_CANDIDATES = ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons","Total_1","Total_2","Total","TC"]
                    found_tv = [tv for tv in TOTALVAR_CANDIDATES if tv in self.species]
                    if found_tv:
                        self.totalVar = found_tv[0]
                        print(f"Inferred total variable as '{self.totalVar}' based on common naming patterns.")
                        print(f"Available candidates were: {', '.join(TOTALVAR_CANDIDATES)}")
                        specie = self.totalVar
                    else:
                        raise ValueError(
                            "No total variable could be inferred from standard names. "
                            f"Expected one of: {', '.join(TOTALVAR_CANDIDATES)}. "
                            "Please set totalVar explicitly or ensure profiles are loaded."
                        )
                else:
                    raise ValueError("No species list available to infer total variable. Load data first.")

        # Use self.profiles if profiles argument is None AND self.profiles exists
        if profiles is None:
            if self.profiles:
                profiles = self.profiles
            else:
                # Try to get profiles from loaded data
                df_prof_check = self.dfprofiles_c if constrained and self.dfprofiles_c is not None else self.dfprofiles_b
                if df_prof_check is not None:
                    profiles = df_prof_check.columns.tolist()
                    print(f"Using profiles from {'constrained' if constrained else 'base'} data: {profiles}")
                else:
                    raise ValueError(
                        "No profiles available (not explicitly provided and not found in data). "
                        "Please load profile data or specify profiles explicitly."
                    )

        # Select appropriate datasets based on constrained flag
        df_contrib = self.dfcontrib_c if constrained else self.dfcontrib_b
        dfprofiles = self.dfprofiles_c if constrained else self.dfprofiles_b

        # Important fallback: If constrained profiles aren't available, use base profiles
        if constrained and dfprofiles is None and self.dfprofiles_b is not None:
            print("Warning: Constrained profiles not available, falling back to base profiles")
            dfprofiles = self.dfprofiles_b

        # Verify that required components are available
        if df_contrib is None:
            data_type = 'constrained' if constrained else 'base'
            raise ValueError(
                f"No contribution data available for {data_type} run. "
                f"Ensure {data_type} contributions are loaded using read_{data_type}_contributions()."
            )

        if dfprofiles is None:
            data_type = 'constrained' if constrained else 'base'
            raise ValueError(
                f"No profile data available for {data_type} run. "
                f"Ensure {data_type} profiles are loaded using read_{data_type}_profiles()."
            )

        # Check if the specie exists in the profiles
        if specie not in dfprofiles.index:
            available_species = dfprofiles.index.tolist()
            raise ValueError(
                f"Species '{specie}' not found in profiles. "
                f"Available species: {', '.join(available_species)}"
            )

        # Check if contribution columns match profiles
        missing_contrib_cols = [p for p in profiles if p not in df_contrib.columns]
        if missing_contrib_cols:
            print(f"Warning: The following profiles are missing from contribution data: {missing_contrib_cols}")
            profiles = [p for p in profiles if p in df_contrib.columns]
            if not profiles:
                raise ValueError("None of the specified profiles exist in the contribution data.")

        missing_profile_cols = [p for p in profiles if p not in dfprofiles.columns]
        if missing_profile_cols:
            print(f"Warning: The following profiles are missing from profile data: {missing_profile_cols}")
            profiles = [p for p in profiles if p in dfprofiles.columns]
            if not profiles:
                raise ValueError("None of the specified profiles exist in the profile data.")

        # Calculate contributions with index alignment safety
        df_contrib_aligned, dfprofiles_aligned = df_contrib.align(dfprofiles, join='inner', axis=1)

        # Reindex dfprofiles_aligned to ensure specie exists after alignment
        if specie not in dfprofiles_aligned.index:
            raise ValueError(
                f"Species '{specie}' lost during data alignment. "
                "This may indicate an inconsistency between contribution and profile data."
            )

        # Select only the profiles we want after alignment
        profiles_to_use = [p for p in profiles if p in df_contrib_aligned.columns]

        contrib = pd.DataFrame(index=df_contrib_aligned.index, columns=profiles_to_use)

        for profile in profiles_to_use:
            # Multiply contribution series by the scalar profile value for the specie
            contrib[profile] = df_contrib_aligned[profile] * dfprofiles_aligned.loc[specie, profile]

        return contrib

    def to_cubic_meter_multisite(self, specie=None, constrained=True, profiles=None):
        """
        Convert profile x contribution to μg/m3 for multi-site data.
        
        This method converts factor contributions to concentration units (μg/m3)
        while preserving the multi-site structure with Station information.
        
        Parameters
        ----------
        specie : str, optional
            Species name to calculate contributions for. If None, uses totalVar
        constrained : bool, default True
            Whether to use constrained (True) or base (False) results
        profiles : list of str, optional
            Specific profiles to include. If None, uses all profiles.
        
        Returns
        -------
        pd.DataFrame
            Contributions in μg/m3 with same format as dfcontrib_c, including Station column
            
        Raises
        ------
        ValueError
            If data is not in multi-site format
        """
        # Check if we're using a multi-site reader
        if not hasattr(self.read, 'multisites') or not self.read.multisites:
            raise ValueError("This method is only for multi-site data. Use to_cubic_meter() instead.")
        
        # If not already loaded, attempt to load the data first
        if not self.ensure_data_loaded():
            print("Attempting to load required data...")
            if constrained:
                if hasattr(self.read, 'read_constrained_profiles'):
                    self.read.read_constrained_profiles()
                if hasattr(self.read, 'read_constrained_contributions'):
                    self.read.read_constrained_contributions()
            else:
                if hasattr(self.read, 'read_base_profiles'):
                    self.read.read_base_profiles()
                if hasattr(self.read, 'read_base_contributions'):
                    self.read.read_base_contributions()
        
        # Determine which species to use (same logic as to_cubic_meter)
        if specie is None:
            specie = self.totalVar
            if specie is None:
                # Try to infer totalVar if not set
                if self.species:
                    TOTALVAR_CANDIDATES = ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons","Total_1","Total_2","Total","TC"]
                    found_tv = [tv for tv in TOTALVAR_CANDIDATES if tv in self.species]
                    if found_tv:
                        self.totalVar = found_tv[0]
                        print(f"Inferred total variable as '{self.totalVar}' based on common naming patterns.")
                        specie = self.totalVar
                    else:
                        raise ValueError(
                            "No total variable could be inferred from standard names. "
                            f"Expected one of: {', '.join(TOTALVAR_CANDIDATES)}. "
                            "Please set totalVar explicitly or ensure profiles are loaded."
                        )
                else:
                    raise ValueError("No species list available to infer total variable. Load data first.")
                    
        # Use self.profiles if profiles argument is None AND self.profiles exists
        if profiles is None:
            if self.profiles:
                profiles = self.profiles
            else:
                # Try to get profiles from loaded data
                df_prof_check = self.dfprofiles_c if constrained and self.dfprofiles_c is not None else self.dfprofiles_b
                if df_prof_check is not None:
                    profiles = df_prof_check.columns.tolist()
                    print(f"Using profiles from {'constrained' if constrained else 'base'} data: {profiles}")
                else:
                    raise ValueError(
                        "No profiles available (not explicitly provided and not found in data). "
                        "Please load profile data or specify profiles explicitly."
                    )
        
        # Select appropriate dataframes based on constrained flag
        if constrained:
            dfprofiles = self.dfprofiles_c
            dfcontrib = self.dfcontrib_c
        else:
            dfprofiles = self.dfprofiles_b
            dfcontrib = self.dfcontrib_b
        
        # Important fallback: If constrained profiles aren't available, use base profiles
        if constrained and dfprofiles is None and self.dfprofiles_b is not None:
            print("Warning: Constrained profiles not available, falling back to base profiles")
            dfprofiles = self.dfprofiles_b
        
        # Check if data is available
        if dfcontrib is None:
            data_type = 'constrained' if constrained else 'base'
            raise ValueError(
                f"No contribution data available for {data_type} run. "
                f"Ensure {data_type} contributions are loaded using read_{data_type}_contributions()."
            )

        if dfprofiles is None:
            data_type = 'constrained' if constrained else 'base'
            raise ValueError(
                f"No profile data available for {data_type} run. "
                f"Ensure {data_type} profiles are loaded using read_{data_type}_profiles()."
            )
        
        # Check if the specie exists in the profiles
        if specie not in dfprofiles.index:
            available_species = dfprofiles.index.tolist()
            raise ValueError(
                f"Species '{specie}' not found in profiles. "
                f"Available species: {', '.join(available_species[:20])}..."
                if len(available_species) > 20 else f"Available species: {', '.join(available_species)}"
            )
        
        # Verify multi-site structure - capture Station column for later
        dfcontrib_reset = dfcontrib.reset_index()
        if "Station" not in dfcontrib_reset.columns:
            raise ValueError(
                "Multi-site data structure is invalid. 'Station' column is missing. "
                "Ensure data was loaded with a MultisitesReader."
            )
        
        station_column = dfcontrib_reset["Station"]
        date_column = dfcontrib_reset["Date"] if "Date" in dfcontrib_reset.columns else None
        
        print(f"Converting to μg/m3 for {len(station_column.unique())} stations using {specie}")
        
        # Check which profiles to use (filter by availability)
        profiles_to_use = [p for p in profiles if p in dfcontrib.columns and p in dfprofiles.columns]
        
        if not profiles_to_use:
            raise ValueError("No common factors found between profiles and contributions.")
        
        # Calculate μg/m3 contributions: multiply each factor by its profile value for the selected species
        df_cubic_meter = pd.DataFrame(index=dfcontrib.index)
        
        for factor in profiles_to_use:
            df_cubic_meter[factor] = dfcontrib[factor] * dfprofiles.loc[specie, factor]
        
        # Preserve the multi-site structure
        if isinstance(dfcontrib.index, pd.MultiIndex):
            # For MultiIndex, we already have the Station info in the index
            return df_cubic_meter
        else:
            # For flat index, add Station back as a column
            df_cubic_meter_reset = df_cubic_meter.reset_index()
            
            # Add Station column
            if "Station" not in df_cubic_meter_reset.columns:
                df_cubic_meter_reset["Station"] = station_column
            
            # Handle Date column
            if date_column is not None and "Date" not in df_cubic_meter_reset.columns:
                df_cubic_meter_reset["Date"] = date_column
                
            # Try to match the original index structure
            if isinstance(dfcontrib.index, pd.DatetimeIndex):
                return df_cubic_meter_reset.set_index("Date")
            else:
                return df_cubic_meter_reset.set_index(['Date'])


    def to_relative_mass(self, constrained: bool = True, species: Optional[List[str]] = None, 
                        profiles: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute factor profile relative mass by species normalization.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run
        species : list of str, optional
            Specific species to include
        profiles : list of str, optional
            Specific profiles to include
            
        Returns
        -------

        pd.DataFrame
            Profile data normalized by total mass
        """
        df = self.dfprofiles_c if constrained else self.dfprofiles_b
        if df is None:
            raise ValueError("Profile data not available")
        profiles = profiles or self.profiles
        species = species or self.species
        if self.totalVar not in df.index:
            raise ValueError(f"Total variable '{self.totalVar}' not found in profiles")
        total_vars = df.loc[self.totalVar, profiles]
        zero_totals = total_vars <= 1e-10
        if zero_totals.any():
            print(f"Warning: Near-zero total mass in factors: {profiles[zero_totals]}")
        d = pd.DataFrame(index=species, columns=profiles)
        for p in profiles:
            denominator = max(df.loc[self.totalVar, p], 1e-10)
            d[p] = df.loc[species, p] / denominator
        return d

    def get_total_species_sum(self, constrained: bool = True) -> pd.DataFrame:
        """
        Calculate percentage contribution of each factor to each species.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run
            
        Returns
        -------

        pd.DataFrame
            Matrix of percentage contributions
        """
        df = self.dfprofiles_c.copy() if constrained else self.dfprofiles_b.copy()
        if df is None:
            raise ValueError("Profile data not available")
        df_sum = df.sum(axis=1) + 1e-10
        return (df.T / df_sum).T * 100

    def get_seasonal_contribution(self, specie: Optional[str] = None, annual: bool = True, 
                                normalize: bool = True, constrained: bool = True) -> pd.DataFrame:
        """
        Calculate seasonal contribution of factors.
        
        Parameters
        ----------
        specie : str, optional
            Species to analyze, defaults to totalVar
        annual : bool, default=True
            Whether to include annual average
        normalize : bool, default=True
            Whether to normalize to 100%
        constrained : bool, default=True
            Whether to use constrained or base run
            
        Returns
        -------

        pd.DataFrame
            Seasonal contributions
        """
        dfprofiles, dfcontrib = (self.dfprofiles_c, self.dfcontrib_c) if constrained else (self.dfprofiles_b, self.dfcontrib_b)
        if dfprofiles is None or dfcontrib is None:
            raise ValueError("Profile or contribution data not available")
        specie = specie or self.totalVar
        if specie not in dfprofiles.index:
            raise ValueError(f"Species '{specie}' not found in profiles")
        dfcontribSeason = (dfprofiles.loc[specie] * dfcontrib).sort_index(axis=1)
        dfcontribSeason = add_season(dfcontribSeason, month=False).groupby("season")
        ordered_season = ["Winter", "Spring", "Summer", "Fall"]
        if annual:
            ordered_season.append("Annual")
        df = (dfcontribSeason.sum().T / dfcontribSeason.sum().sum(axis=1)).T if normalize else dfcontribSeason.mean()
        if annual:
            df.loc["Annual", :] = df.mean()
        return df.reindex(ordered_season)

    def replace_totalVar(self, newTotalVar: str) -> None:
        """
        Replace the current total variable name (e.g., PM10) with a new one.

        Parameters
        ----------
        newTotalVar : str
            New total variable name to be set.

        Returns
        -------

        None

        Examples
        --------
        >>> pmf.replace_totalVar("PM2.5_total")
        """
        if not self.totalVar:
            raise ValueError("No total variable currently defined")
        dataframes = [self.dfprofiles_b, self.dfprofiles_c, self.dfBS_profile_b, self.dfBS_profile_c, self.df_uncertainties_summary_b, self.df_uncertainties_summary_c]
        for df in dataframes:
            if df is not None and self.totalVar in df.index:
                df.rename({self.totalVar: newTotalVar}, inplace=True, axis=0)
        if self.species:
            self.species = [newTotalVar if x == self.totalVar else x for x in self.species]
        self.totalVar = newTotalVar

    def rename_factors(self, mapper: Dict[str, str]) -> None:
        """
        Rename factors based on a dictionary mapper.

        Parameters
        ----------
        mapper : dict of {str : str}
            Mapping from old factor names to new factor names.

        Returns
        -------

        None
        
        Examples
        --------
        >>> pmf.rename_factors({"Factor1": "Traffic", "Factor2": "Industrial"})
        """
        if not isinstance(mapper, dict):
            raise TypeError("Mapper must be a dictionary")
        dataframes = [self.dfprofiles_b, self.dfprofiles_c, self.dfcontrib_b, self.dfcontrib_c, self.dfBS_profile_b, self.dfBS_profile_c, self.df_uncertainties_summary_b, self.df_uncertainties_summary_c]
        for df in dataframes:
            if df is None:
                continue
            if hasattr(df, 'index') and isinstance(df.index, pd.Index) and df.index.dtype == 'O':
                df.rename(mapper, inplace=True, axis="index")
            if hasattr(df, 'columns') and df.columns.dtype == 'O':
                df.rename(mapper, inplace=True, axis="columns")
        if self.profiles:
            self.profiles = [mapper.get(p, p) for p in self.profiles]

    def rename_factors_to_factors_category(self) -> None:
        """
        Automatically rename factors to their category names, if available.

        Returns
        -------

        None

        Examples
        --------
        >>> pmf.rename_factors_to_factors_category()
        """
        if not self.profiles:
            raise ValueError("No profiles available to rename")
        possible_sources = {p: get_sourcesCategories([p])[0] for p in self.profiles}
        self.rename_factors(possible_sources)

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
        profiles = self.dfprofiles_c if constrained else self.dfprofiles_b
        contrib = self.dfcontrib_c if constrained else self.dfcontrib_b
        if profiles is None or contrib is None:
            raise ValueError("Profile or contribution data not available")
        explained_var = pd.DataFrame(index=profiles.index, columns=profiles.columns, dtype=float)
        for species in profiles.index:
            total_var = 0
            for factor in profiles.columns:
                var = np.var(contrib[factor] * profiles.loc[species, factor])
                explained_var.loc[species, factor] = var
                total_var += var
            if total_var > 0:
                explained_var.loc[species] = explained_var.loc[species] / total_var * 100
        return explained_var

    def print_uncertainties_summary(self, constrained: bool = True, profiles: Optional[List[str]] = None, 
                                   species: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return uncertainties summary (BS, DISP, BS-DISP).
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained or base run
        profiles : list of str, optional
            Specific profiles to include
        species : list of str, optional
            Specific species to include
            
        Returns
        -------

        pd.DataFrame
            Uncertainty summary data
        """
        df = self.df_uncertainties_summary_c if constrained else self.df_uncertainties_summary_b
        if df is None:
            raise ValueError("Uncertainty data not available")
        profiles = profiles or self.profiles
        species = species or self.species
        try:
            return df.T.loc[:, (profiles, species)]
        except KeyError:
            available_profiles = df.index.get_level_values("Profile").unique()
            available_species = df.index.get_level_values("Specie").unique()
            valid_profiles = [p for p in profiles if p in available_profiles]
            valid_species = [s for s in species if s in available_species]
            if not valid_profiles or not valid_species:
                raise ValueError("No valid profiles or species found in uncertainty data")
            return df.T.loc[:, (valid_profiles, valid_species)]

    def recompute_new_species(self, specie: str) -> None:
        """
        Recompute a species given the other species.
        
        For example, recompute OC from OC* and a list of organic species.
        Modifies dfprofile_b and dfprofile_c in-place.

        Parameters
        ----------
        specie : str
            Species to recompute, currently supported: ["OC"]
        """
        knownSpecies = ["OC"]
        if specie not in knownSpecies:
            print(f"Warning: Recomputing {specie} is not supported")
            return

        equivC = {
            'Oxalate': 0.27,
            'Arabitol': 0.40,
            'Mannitol': 0.40,
            'Sorbitol': 0.40,
            'Polyols': 0.40,
            'Levoglucosan': 0.44,
            'Mannosan': 0.44,
            'Galactosan': 0.44,
            'MSA': 0.12,
            'Glucose': 0.44,
            'Cellulose': 0.44,
            'Maleic': 0.41,
            'Succinic': 0.41,
            'Citraconic': 0.46,
            'Glutaric': 0.45,
            'Oxoheptanedioic': 0.48,
            'MethylSuccinic': 0.53,
            'Adipic': 0.49,
            'Methylglutaric': 0.49,
            '3-MBTCA': 0.47,
            'Phtalic': 0.58,
            'Pinic': 0.58,
            'Suberic': 0.55,
            'Azelaic': 0.57,
            'Sebacic': 0.59,
        }

        if specie == "OC":
            if specie not in self.species:
                self.species.append(specie)
            
            if "OC*" not in self.dfprofiles_b.index or "OC*" not in self.dfprofiles_c.index:
                print("Warning: OC* not found in profiles, cannot recompute OC")
                return
                
            OCb = self.dfprofiles_b.loc["OC*"].copy()
            OCc = self.dfprofiles_c.loc["OC*"].copy()
            
            for sp in equivC.keys():
                if sp in self.species:
                    OCb += (self.dfprofiles_b.loc[sp] * equivC[sp]).infer_objects()
                    OCc += (self.dfprofiles_c.loc[sp] * equivC[sp]).infer_objects()
                    
            self.dfprofiles_b.loc[specie] = OCb.infer_objects()
            self.dfprofiles_c.loc[specie] = OCc.infer_objects()

    def preprocess_data(self, data: pd.DataFrame, 
                       uncertainty_method: str = "percentage",
                       uncertainty_params: Optional[Dict] = None,
                       min_valid: float = 0.75,
                       handling_method: str = "interpolate",
                       total_var: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for PMF analysis with integrated error handling.
        
        This method handles data cleaning, uncertainty calculation,
        and data transformation in a single step.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with samples as rows and species as columns
        uncertainty_method : str, default="percentage"
            Method for calculating uncertainties:
            - "percentage": Fixed percentage of concentration
            - "DL": Detection limit based uncertainty
            - "polissar": Polissar method using DL
        uncertainty_params : dict, optional
            Parameters for uncertainty calculation method
        min_valid : float, default=0.75
            Minimum fraction of valid data required for each species
        handling_method : str, default="interpolate"
            Method for handling missing values
        total_var : str, optional
            Total variable to use for normalization
            
        Returns
        -------

        Tuple[pd.DataFrame, pd.DataFrame]
            Processed data and uncertainties
            
        Examples
        --------
        >>> raw_data = pd.read_csv("raw_data.csv", index_col=0)
        >>> processed_data, uncertainties = pmf.preprocess_data(
        ...     raw_data, 
        ...     uncertainty_method="polissar",
        ...     uncertainty_params={"error_fraction": 0.15}
        ... )
        """
        try:
            preprocessor = PMFPreprocessor(data)
            
            # Summarize data quality issues
            quality_summary = preprocessor.summarize_data_quality()
            print("Data quality summary (% of values):")
            print(quality_summary.round(2))
            
            # Handle missing values
            data_handled = preprocessor.handle_missing_values(method=handling_method)
            
            # Filter species based on data quality
            data_filtered = preprocessor.filter_species(min_valid=min_valid)
            
            # Normalize if total_var is specified
            data_normalized = data_filtered
            if total_var and total_var in data_filtered.columns:
                data_normalized = preprocessor.normalize_to_total(total_var=total_var)
            
            # Compute uncertainties
            uncertainties = preprocessor.compute_uncertainties(
                method=uncertainty_method, 
                params=uncertainty_params
            )
            
            # Store in the object for later use
            self.data = data_normalized
            self.uncertainties = uncertainties
            self.species = data_normalized.columns.tolist() 
            self.nspecies = len(self.species)
            
            if total_var:
                self.totalVar = total_var
                
            print(f"Preprocessing complete: {self.data.shape[1]} species, {self.data.shape[0]} samples")
            return self.data, self.uncertainties
            
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise
            
    def track_detection_limits(self, data: pd.DataFrame, dl_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Track values below detection limits in the dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data matrix
        dl_values : dict, optional
            Dictionary mapping species to detection limit values
            
        Returns
        -------

        pd.DataFrame
            Boolean mask where True indicates value below detection limit
            
        Examples
        --------
        >>> data = pd.read_csv("data.csv", index_col=0)
        >>> dl_values = {"SO4": 0.01, "NO3": 0.005}
        >>> dl_mask = pmf.track_detection_limits(data, dl_values)
        >>> print(f"Values below detection limit: {dl_mask.sum().sum()}")
        """
        preprocessor = PMFPreprocessor(data, ql_values=dl_values)
        return preprocessor.track_quantification_limits()
    
    def compute_uncertainties(self, data: pd.DataFrame, 
                            method: str = "percentage", 
                            params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate uncertainties for PMF analysis using various methods.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data matrix
        method : str
            Method for calculating uncertainties:
            - "percentage": Fixed percentage of concentration
            - "DL": Detection limit based uncertainty
            - "polissar": Polissar method using DL
        params : dict, optional
            Parameters specific to the chosen method
            
        Returns
        -------

        pd.DataFrame
            Calculated uncertainties
            
        Examples
        --------
        >>> data = pd.read_csv("data.csv", index_col=0)
        >>> # Calculate uncertainties using Polissar method
        >>> uncertainties = pmf.compute_uncertainties(
        ...     data, 
        ...     method="polissar", 
        ...     params={"DL": {"SO4": 0.01, "NO3": 0.005}, "error_fraction": 0.1}
        ... )
        """
        preprocessor = PMFPreprocessor(data)
        return preprocessor.compute_uncertainties(method=method, params=params)

    def ensure_data_loaded(self) -> bool:
        """
        Verify that essential data is loaded, with detailed diagnostics.
        
        Returns
        -------

        bool
            True if all essential data is loaded, False otherwise
        """
        missing_data = []
        
        # Check for essential attributes
        if self.profiles is None:
            missing_data.append("profiles")
        
        # First try to get constrained profiles
        if self.dfprofiles_c is None and hasattr(self.read, 'read_constrained_profiles'):
            try:
                print("Attempting to load constrained profiles...")
                self.read.read_constrained_profiles()
                if self.dfprofiles_c is not None:
                    print("Successfully loaded constrained profiles.")
                else:
                    missing_data.append("dfprofiles_c")
            except Exception as e:
                print(f"Couldn't load constrained profiles: {str(e)}")
                missing_data.append("dfprofiles_c")
        
        # Then try constrained contributions
        if self.dfcontrib_c is None and hasattr(self.read, 'read_constrained_contributions'):
            try:
                print("Attempting to load constrained contributions...")
                self.read.read_constrained_contributions()
                if self.dfcontrib_c is not None:
                    print("Successfully loaded constrained contributions.")
                else:
                    missing_data.append("dfcontrib_c")
            except Exception as e:
                print(f"Couldn't load constrained contributions: {str(e)}")
                missing_data.append("dfcontrib_c")
        
        # Fall back to base profiles if needed
        if self.dfprofiles_c is None and self.dfprofiles_b is None and hasattr(self.read, 'read_base_profiles'):
            try:
                print("Constrained profiles not available. Attempting to load base profiles...")
                self.read.read_base_profiles()
                if self.dfprofiles_b is not None:
                    print("Successfully loaded base profiles.")
                else:
                    missing_data.append("dfprofiles_b")
            except Exception as e:
                print(f"Couldn't load base profiles: {str(e)}")
                missing_data.append("dfprofiles_b")
        
        if missing_data:
            print(f"WARNING: The following essential data is missing: {', '.join(missing_data)}")
            # Print debug info about what is loaded
            print("\nCurrent data status:")
            print(f"- profiles: {self.profiles if hasattr(self, 'profiles') and self.profiles else 'None'}")
            print(f"- totalVar: {self.totalVar if hasattr(self, 'totalVar') and self.totalVar else 'None'}")
            print(f"- dfprofiles_b shape: {self.dfprofiles_b.shape if hasattr(self, 'dfprofiles_b') and self.dfprofiles_b is not None else 'None'}")
            print(f"- dfprofiles_c shape: {self.dfprofiles_c.shape if hasattr(self, 'dfprofiles_c') and self.dfprofiles_c is not None else 'None'}")
            print(f"- dfcontrib_b shape: {self.dfcontrib_b.shape if hasattr(self, 'dfcontrib_b') and self.dfcontrib_b is not None else 'None'}")
            print(f"- dfcontrib_c shape: {self.dfcontrib_c.shape if hasattr(self, 'dfcontrib_c') and self.dfcontrib_c is not None else 'None'}")
            
            # If this is a multisite reader, provide additional diagnostics
            if hasattr(self.read, 'multisites') and self.read.multisites:
                print("NOTE: Using multi-site reader. Site filtering is active.")
                print(f"- Current site: '{self._site}'")
            
            return False
        
        return True
