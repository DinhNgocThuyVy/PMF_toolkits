from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import logging

# Configure logger
logger = logging.getLogger('PMF_toolkits')

XLSX_ENGINE = 'openpyxl'

# Base reader class
class BaseReader(ABC):
    """
    Abstract base class for PMF result readers.

    Each reader must handle:
    - Base and constrained PMF run outputs
    - Bootstrap, DISP, and BS-DISP results
    - Metadata about factors and species

    See Also
    --------
    :class:`XlsxReader`
    :class:`MultisitesReader`
    """

    def __init__(self, site: str, pmf: Any):
        """
        Initialize the reader with site name and PMF object.
        
        Parameters
        ----------
        site : str
            The name of the site or location
        pmf : Any
            The parent PMF object that will store the read data
        """
        self.site = site
        self.pmf = pmf

    @abstractmethod
    def read_base_profiles(self) -> None:
        """
        Read the "base" profiles result from the file: '_base.xlsx',
        sheet "Profiles", and add :

        - self.dfprofiles_b: base factors profile
        """
        pass

    @abstractmethod
    def read_constrained_profiles(self) -> None:
        """
        Read the "constrained" profiles result from the file: '_Constrained.xlsx',
        sheet "Profiles", and add :

        - self.dfprofiles_c: constrained factors profile
        """
        pass

    @abstractmethod
    def read_base_contributions(self) -> None:
        """
        Read the "base" contributions result from the file: '_base.xlsx',
        sheet "Contributions", and add :

        - self.dfcontrib_b: base factors contribution
        """
        pass

    @abstractmethod
    def read_constrained_contributions(self) -> None:
        """
        Read the "constrained" contributions result from the file: '_Constrained.xlsx',
        sheet "Contributions", and add :

        - self.dfcontrib_c: constrained factors contribution
        """
        pass

    @abstractmethod
    def read_base_bootstrap(self) -> None:
        """
        Read the "base" bootstrap result from the file: '_boot.xlsx'
        and add :

        - self.dfBS_profile_b: all mapped profile
        - self.dfbootstrap_mapping_b: table of mapped profiles
        """
        pass

    @abstractmethod
    def read_constrained_bootstrap(self) -> None:
        """
        Read the "base" bootstrap result from the file: '_Gcon_profile_boot.xlsx'
        and add :

        - self.dfBS_profile_c: all mapped profile
        - self.dfbootstrap_mapping_c: table of mapped profiles
        """
        pass

    @abstractmethod
    def read_base_uncertainties_summary(self) -> None:
        """
        Read the _BaseErrorEstimationSummary.xlsx file and add:

        - self.df_uncertainties_summary_b : uncertainties from BS, DISP and BS-DISP
        """
        pass

    @abstractmethod
    def read_constrained_uncertainties_summary(self) -> None:
        """
        Read the _ConstrainedErrorEstimationSummary.xlsx file and add :

        - self.df_uncertainties_summary_b : uncertainties from BS, DISP and BS-DISP
        """
        pass

    def _handle_non_convergente_bootstrap(self, dfBS_profile: pd.DataFrame, 
                                         dfbootstrap_mapping: pd.DataFrame) -> None:
        """
        Get rid of bootstrap run that did not converge.
        Note that this is different than unmapped BS!

        Also, sometimes BS ends up with concentration of total variable close to 0, that may
        cause DividedByZero error...

        Parameters
        ----------
        dfBS_profile : pd.DataFrame
            Bootstrap profile (all of them)
        dfbootstrap_mapping : pd.DataFrame
            Mapping between base factor and BS factor (use to determine the number of non
            convergent BS (≠ than unmapped))
        """
        # Check if the totalVar is set
        if self.pmf.totalVar is None:
            print("Warning: totalVar is None, skipping non-convergent bootstrap check")
            return
            
        # handle nonconvergente BS
        try:
            nBSconverged = dfbootstrap_mapping.sum(axis=1)[0]
            nBSnotconverged = len(dfBS_profile.columns)-1-nBSconverged
            
            if nBSnotconverged <= 0:
                return
                
            print(f"Warning: trying to exclude {nBSnotconverged} non-convergent bootstrap runs")
            
            # Handling for both MultiIndex and regular index
            if isinstance(dfBS_profile.index, pd.MultiIndex):
                # Extract all rows where the first level is the totalVar
                try:
                    idx_totalVar = dfBS_profile.index.get_level_values(0) == self.pmf.totalVar
                    # Get the rows for the total variable
                    total_var_rows = dfBS_profile[idx_totalVar]
                    
                    # Find columns with values > 100 in any row
                    idxStrange = (total_var_rows.apply(pd.to_numeric, errors='coerce') > 100).any()
                    colStrange = idxStrange[idxStrange].index.tolist()
                    
                    if len(colStrange) > 0:
                        print(f"Found {len(colStrange)} BS runs to eliminate with values > 100 for {self.pmf.totalVar}")
                        dfBS_profile.drop(colStrange, axis=1, inplace=True)
                except Exception as e:
                    print(f"Error handling MultiIndex bootstrap: {str(e)}")
            else:
                # Direct access for standard index
                try:
                    if self.pmf.totalVar in dfBS_profile.index:
                        # Convert to numeric and check for values > 100
                        totalVarValues = pd.to_numeric(dfBS_profile.loc[self.pmf.totalVar], errors='coerce')
                        idxStrange = totalVarValues > 100
                        colStrange = idxStrange[idxStrange].index.tolist()
                        
                        if len(colStrange) > 0:
                            print(f"Found {len(colStrange)} BS runs to eliminate with values > 100 for {self.pmf.totalVar}")
                            dfBS_profile.drop(colStrange, axis=1, inplace=True)
                except Exception as e:
                    print(f"Error handling standard index bootstrap: {str(e)}")
                    
        except Exception as e:
            print(f"Warning: Error in _handle_non_convergente_bootstrap: {str(e)}")
            print("Bootstrap filtering skipped - proceed with caution")

    def read_metadata(self) -> None:
        """
        Read metadata such as factor names, species, and totalVar from the data source.

        Typically sets self.pmf.profiles, self.pmf.species, self.pmf.totalVar, etc.
        """
        pmf = self.pmf

        if pmf.dfprofiles_b is None:
            self.read_base_profiles()

        pmf.profiles = pmf.dfprofiles_b.columns.tolist()
        pmf.nprofiles = len(pmf.profiles)
        pmf.species = pmf.dfprofiles_b.index.tolist()
        pmf.nspecies = len(pmf.species)

        TOTALVAR = ["PM10", "PM2.5", "PMrecons", "PM10rec", "PM10recons","Total_1","Total_2","Total","TC"]
        pmf.totalVar = None
        for x in TOTALVAR:
            if x in pmf.species:
                pmf.totalVar = x
                break

        if pmf.totalVar is None:
            print("Warning: trying to guess total variable.")
            # Only pick string species containing "PM"
            totalvar_candidates = [x for x in pmf.species if isinstance(x, str) and "PM" in x]
            if totalvar_candidates and len(totalvar_candidates) >= 1:
                print(f"Warning: several possible total variables: {totalvar_candidates}")
                print(f"Warning: taking the first one {totalvar_candidates[0]}")
                pmf.totalVar = totalvar_candidates[0]
            else:
                print("Warning: No suitable total variable found, setting to None")
                pmf.totalVar = None
        print(f"Total variable set to: {pmf.totalVar}")

    @staticmethod 
    def _validate_data(df: pd.DataFrame, expected_cols: List[str]) -> None:
        """Validate dataframe format and content."""
        if df is None or df.empty:
            raise ValueError("Empty or invalid data")
            
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def read_all(self) -> None:
        """Read all data with error handling."""
        required_methods = [
            "read_base_profiles",
            "read_base_contributions",
            "read_base_bootstrap",
            "read_base_uncertainties_summary",
            "read_constrained_profiles",
            "read_constrained_contributions",
            "read_constrained_bootstrap",
            "read_constrained_uncertainties_summary"
        ]
        
        successful = []
        failed = []
        
        for method in required_methods:
            try:
                getattr(self, method)()
                successful.append(method)
            except FileNotFoundError:
                print(f"Warning: File not found for {method}")
                failed.append(method)
            except Exception as e:
                print(f"Error in {method}: {str(e)}")
                failed.append(method)
                
        print(f"Successfully read: {len(successful)}/{len(required_methods)} components")

# Standard Excel reader
class XlsxReader(BaseReader):
    """Reader for PMF outputs in Excel (XLSX) format.

    It automatically infers file names based on the site name. The user
    should place relevant Excel files (base, constrained, bootstrap, etc.) in BDIR.

    Examples
    --------
    >>> r = XlsxReader(BDIR="path/to/excel_files", site="mysite", pmf=pmfobj)
    >>> r.read_base_profiles()
    
    See Also
    --------
    :class:`MultisitesReader`
    :meth:`BaseReader.read_all`
    """
    def __init__(self, BDIR: str, site: str, pmf: Any, multisites: bool = False):
        """
        Initialize the Excel reader.

        Parameters
        ----------
        BDIR : str
            Directory containing the PMF output files
        site : str
            The name of the site
        pmf : Any
            The parent PMF object that will store the read data
        multisites : bool, default False
            Whether the files contain data for multiple sites
        """
        super().__init__(site=site, pmf=pmf)
        self.BDIR = BDIR
        self.basename = Path(BDIR) / site
        self.multisites = multisites

    def _split_df_by_nan(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Internet method to read the bootstrap file format:
        1 block of N lines (1 per factor) for each species, separated by an empty line.

        Parameter
        ---------
        df : pd.DataFrame
            The bootstrap data from the xlsx files. The header should be already removed.

        Return
        ------
        Dict[str, pd.DataFrame]
            Dictionary where keys are species names and values are DataFrames
            containing bootstrap results for that species, indexed by profile.
        """
        pmf = self.pmf
        d = {}
        
        try:
            dftmp = df.dropna(how='all')
            
            for i, sp in enumerate(pmf.species):
                # Select the block of rows for the current species
                start_row = pmf.nprofiles * i
                end_row = pmf.nprofiles * (i + 1)
                
                # Check if we've reached the end of the data
                if start_row >= len(dftmp):
                    break
                    
                # Extract the block
                block = dftmp.iloc[start_row:end_row, :]
                
                if len(block) < pmf.nprofiles:
                    print(f"Warning: Incomplete block for species {sp}: {len(block)}/{pmf.nprofiles} rows")
                    continue
                
                # Set proper index and column names
                block.index = pmf.profiles
                block.index.name = "Profile"
                block.columns = [f"Boot{j}" for j in range(len(block.columns))]
                d[sp] = block
                
        except Exception as e:
            print(f"Error in _split_df_by_nan: {str(e)}")
            return {}
            
        return d

    def _process_contributions(self, dfcontrib: pd.DataFrame) -> pd.DataFrame:
        """
        Process EPA PMF 5.0 contribution data for both single and multi-site formats.
        
        EPA PMF 5.0 Excel files have a complex structure with headers and data sections.
        This method identifies the correct data section using "Factor Contributions"
        markers and processes it according to whether it's a single or multi-site format.
        """
        pmf = self.pmf
        dfcontrib = dfcontrib.copy()
        
        try:
            # First, find data section using "Factor Contributions" marker
            idx = dfcontrib.iloc[:, 0].str.contains("Factor Contributions").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                print("Warning: Could not find 'Factor Contributions' marker in the sheet")
                # If marker not found, try to use the data as-is
            elif len(idx) > 1:
                # Multiple sections exist, use just the first section
                print(f"Found multiple 'Factor Contributions' sections, using first section")
                dfcontrib = dfcontrib.iloc[idx[0]:idx[1], :]
            else:
                # Just one section exists
                dfcontrib = dfcontrib.iloc[idx[0]+1:, :]
        except AttributeError:
            print("WARNING: Could not parse contribution data structure")
        
        # Clean up the data
        dfcontrib.dropna(axis=1, how="all", inplace=True)
        dfcontrib.dropna(how="all", inplace=True)
        
        # For multi-site format
        if self.multisites:
            # Drop the first column (usually contains row labels)
            if dfcontrib.shape[1] > 0:
                dfcontrib.drop(columns=dfcontrib.columns[0], inplace=True)
                
            # Reset index and set up column names based on first row
            dfcontrib = dfcontrib.dropna(how="all").reset_index(drop=True)
            
            # Set column names properly - make sure Station and Date columns exist
            if dfcontrib.shape[0] > 0:
                dfcontrib.loc[0, 1] = "Station"
                dfcontrib.loc[0, 2] = "Date"
                
                # Use first row as column names
                dfcontrib.columns = dfcontrib.iloc[0]
                
                # Drop the header row and set Date as index
                dfcontrib = dfcontrib.iloc[1:].set_index("Date")
                
                # Drop the Date row if it exists in the index
                if "Date" in dfcontrib.index:
                    dfcontrib = dfcontrib.drop(index="Date")
            
        # For single-site format
        else:
            # Check if we have enough columns
            nancolumns = dfcontrib.isna().all()
            if nancolumns.any():
                dfcontrib = dfcontrib.loc[:, :nancolumns.idxmax()]
                
            dfcontrib.dropna(axis=0, how="all", inplace=True)
            dfcontrib.dropna(axis=1, how="all", inplace=True)
            
            # Extract data starting from index 0, drop the first column if it's not needed
            if 1 in dfcontrib.columns and dfcontrib.shape[1] >= 3:
                # Set column names based on profile information
                dfcontrib.columns = ["Ignored", "Date"] + pmf.profiles
                if "Ignored" in dfcontrib.columns:
                    dfcontrib = dfcontrib.drop(columns=["Ignored"])
            else:
                print("Warning: Unexpected column structure in single site format")
                
            # Convert Date column to datetime and set as index
            try:
                dfcontrib["Date"] = pd.to_datetime(dfcontrib["Date"], errors="coerce")
                dfcontrib.set_index("Date", inplace=True)
                dfcontrib = dfcontrib[dfcontrib.index.notnull()]
            except KeyError:
                print("Warning: Could not set Date as index")
        
        # Replace -999 with NaN (common missing value code in PMF)
        dfcontrib.replace({-999: np.nan}, inplace=True)
        
        # Convert data to appropriate types
        dfcontrib = dfcontrib.infer_objects()
        
        return dfcontrib

    def read_base_profiles(self) -> None:
        """
        Read the "base" profiles result from the file: '_base.xlsx',
        sheet "Profiles", and add:

        - self.dfprofiles_b: base factors profile
        """
        pmf = self.pmf
        
        try:
            file_path = f"{self.basename}_base.xlsx"
            dfbase = pd.read_excel(
                file_path,
                sheet_name=['Profiles'],
                header=None,
                engine=XLSX_ENGINE
            )["Profiles"]
            
            # Find the row with "Factor Profiles"
            idx = dfbase.iloc[:, 0].str.contains("Factor Profiles").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                raise ValueError("Could not find 'Factor Profiles' in the sheet")
                
            # Extract the profile data
            dfbase = dfbase.iloc[idx[0]:, 1:]
            if len(idx) > 1:
                dfbase = dfbase.iloc[:idx[1]-idx[0], :]
                
            dfbase.dropna(axis=0, how="all", inplace=True)
            factor_names = list(dfbase.iloc[0, 1:])
            dfbase.columns = ["Specie"] + factor_names
            dfbase = dfbase.drop(dfbase.index[0]).set_index("Specie")
            
            # Clean up the data
            # Check correct number of column
            idx = dfbase.columns.isna().argmax()
            if idx > 0:
                dfbase = dfbase.iloc[:, :idx]
                dfbase.dropna(how="all", inplace=True)
                
            # avoid 10**-12 possible concentration values
            dfbase = dfbase.infer_objects()
            dfbase[dfbase < 10e-6] = 0
            
            pmf.dfprofiles_b = dfbase
            
            # Read metadata (profiles, species, totalVar)
            super().read_metadata()
            
            print(f"Successfully read base profiles with shape {dfbase.shape}")
            
        except FileNotFoundError:
            print(f"Base profiles file not found: {self.basename}_base.xlsx")
            raise
        except Exception as e:
            print(f"Error reading base profiles: {str(e)}")
            raise

    def read_constrained_profiles(self) -> None:
        """
        Read the "constrained" profiles result from the file: '_Constrained.xlsx',
        sheet "Profiles", and add:

        - self.dfprofiles_c: constrained factors profile
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles are loaded first
            if pmf.profiles is None:
                self.read_base_profiles()
                
            file_path = f"{self.basename}_Constrained.xlsx"
            dfcons = pd.read_excel(
                file_path,
                sheet_name=['Profiles'],
                header=None,
                engine=XLSX_ENGINE
            )["Profiles"]
            
            # Find the row with "Factor Profiles"
            idx = dfcons.iloc[:, 0].str.contains("Factor Profiles").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                raise ValueError("Could not find 'Factor Profiles' in the constrained sheet")
                
            # Extract the profile data
            dfcons = dfcons.iloc[idx[0]:, 1:]
            if len(idx) > 1:
                dfcons = dfcons.iloc[:idx[1]-idx[0], :]
                
            dfcons.dropna(axis=0, how="all", inplace=True)
            
            # Check correct number of columns
            idx = dfcons.columns.isna().argmax()
            if idx > 0:
                dfcons = dfcons.iloc[:, :idx]
                dfcons.dropna(how="all", inplace=True)
                
            nancolumns = dfcons.isna().all()
            if nancolumns.any():
                dfcons = dfcons.loc[:, :nancolumns.idxmax()]
                dfcons.dropna(axis=1, how="all", inplace=True)
                
            # Set column names using factor names from base profiles
            dfcons.columns = ["Specie"] + pmf.profiles
            dfcons = dfcons.set_index("Specie")
            dfcons = dfcons[dfcons.index.notnull()]
            
            # Clean up the data
            dfcons = dfcons.infer_objects()
            dfcons[dfcons < 10e-6] = 0
            
            pmf.dfprofiles_c = dfcons
            print(f"Successfully read constrained profiles with shape {dfcons.shape}")
            
        except FileNotFoundError:
            print(f"Constrained profiles file not found: {self.basename}_Constrained.xlsx")
            raise
        except Exception as e:
            print(f"Error reading constrained profiles: {str(e)}")
            raise

    def read_base_contributions(self) -> None:
        """
        Read the "base" contributions result from the file: '_base.xlsx',
        sheet "Contributions", and add:

        - self.dfcontrib_b: base factors contribution
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles are loaded first
            if pmf.profiles is None:
                self.read_base_profiles()
                
            file_path = f"{self.basename}_base.xlsx"
            dfcontrib_raw = pd.read_excel(
                file_path,
                sheet_name=['Contributions'],
                header=None,
                engine=XLSX_ENGINE
            )["Contributions"]
            
            # Process contributions using the helper method
            dfcontrib = self._process_contributions(dfcontrib_raw)
            
            # Store the processed DataFrame
            pmf.dfcontrib_b = dfcontrib
            print(f"Successfully read base contributions with shape {dfcontrib.shape}")
            
        except FileNotFoundError:
            print(f"Base contributions file not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error reading base contributions: {str(e)}")
            raise

    def read_constrained_contributions(self) -> None:
        """
        Read the "constrained" contributions result from the file: '_Constrained.xlsx',
        sheet "Contributions", and add:

        - self.dfcontrib_c: constrained factors contribution
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles are loaded first
            if pmf.profiles is None:
                try:
                    self.read_constrained_profiles()
                except:
                    self.read_base_profiles()
                
            file_path = f"{self.basename}_Constrained.xlsx"
            dfcontrib_raw = pd.read_excel(
                file_path,
                sheet_name=['Contributions'],
                header=None,
                engine=XLSX_ENGINE
            )["Contributions"]
            
            # Process contributions using the helper method
            dfcontrib = self._process_contributions(dfcontrib_raw)
            
            # Store the processed DataFrame
            pmf.dfcontrib_c = dfcontrib
            print(f"Successfully read constrained contributions with shape {dfcontrib.shape}")
            
        except FileNotFoundError:
            print(f"Constrained contributions file not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error reading constrained contributions: {str(e)}")
            raise

    def read_base_bootstrap(self) -> None:
        """
        Read the "base" bootstrap result from the file: '_boot.xlsx'
        and add:

        - self.dfBS_profile_b: all mapped profile
        - self.dfbootstrap_mapping_b: table of mapped profiles
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles are loaded first
            if pmf.profiles is None:
                self.read_base_profiles()
                
            file_path = f"{self.basename}_boot.xlsx"
            dfprofile_boot = pd.read_excel(
                file_path,
                sheet_name=['Profiles'],
                header=None,
                engine=XLSX_ENGINE
            )["Profiles"]
            
            # Extract bootstrap mapping table (showing how base factors map to bootstrap factors)
            dfbootstrap_mapping_b = dfprofile_boot.iloc[2:2+pmf.nprofiles, 0:pmf.nprofiles+2]
            dfbootstrap_mapping_b.columns = ["mapped"] + pmf.profiles + ["unmapped"]
            dfbootstrap_mapping_b.set_index("mapped", inplace=True)
            dfbootstrap_mapping_b.index = ["BF-"+f for f in pmf.profiles]
            
            # Find the bootstrap data section
            idx = dfprofile_boot.iloc[:, 0].str.contains("Columns are:").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                raise ValueError("Could not find bootstrap data section (missing 'Columns are:' marker)")
                
            # 13 is typically the first column for BS result
            dfprofile_boot = dfprofile_boot.iloc[idx[0]+1:, 13:]
            
            # Process bootstrap blocks by species
            bs_blocks = self._split_df_by_nan(dfprofile_boot)
            
            # Combine into a single DataFrame with multi-index
            dfBS_profile_b = pd.concat(bs_blocks)
            dfBS_profile_b.index.names = ["Specie", "Profile"]
            
            # Handle non-convergent bootstrap runs
            self._handle_non_convergente_bootstrap(dfBS_profile_b, dfbootstrap_mapping_b)
            
            # Store the processed data
            pmf.dfBS_profile_b = dfBS_profile_b
            pmf.dfbootstrap_mapping_b = dfbootstrap_mapping_b
            
            print(f"Successfully read base bootstrap with {len(dfBS_profile_b.columns)} runs")
            
        except FileNotFoundError:
            print(f"Base bootstrap file not found: {self.basename}_boot.xlsx")
            raise
        except Exception as e:
            print(f"Error reading base bootstrap: {str(e)}")
            raise

    def read_constrained_bootstrap(self) -> None:
        """
        Read the "constrained" bootstrap result from the file: '_Gcon_profile_boot.xlsx'
        and add:

        - self.dfBS_profile_c: all mapped profile
        - self.dfbootstrap_mapping_c: table of mapped profiles
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles are loaded first
            if pmf.profiles is None:
                try:
                    self.read_constrained_profiles()
                except:
                    self.read_base_profiles()
                
            file_path = f"{self.basename}_Gcon_profile_boot.xlsx"
            dfprofile_boot = pd.read_excel(
                file_path,
                sheet_name=['Profiles'],
                header=None,
                engine=XLSX_ENGINE
            )["Profiles"]
            
            # Extract bootstrap mapping table
            dfbootstrap_mapping_c = dfprofile_boot.iloc[2:2+pmf.nprofiles, 0:pmf.nprofiles+2]
            dfbootstrap_mapping_c.columns = ["mapped"] + pmf.profiles + ["unmapped"]
            dfbootstrap_mapping_c.set_index("mapped", inplace=True)
            dfbootstrap_mapping_c.index = ["BF-"+f for f in pmf.profiles]
            
            # Find the bootstrap data section
            idx = dfprofile_boot.iloc[:, 0].str.contains("Columns are:").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                raise ValueError("Could not find constrained bootstrap data section (missing 'Columns are:' marker)")
                
            # 13 is typically the first column for BS result
            dfprofile_boot = dfprofile_boot.iloc[idx[0]+1:, 13:]
            
            # Process bootstrap blocks by species
            bs_blocks = self._split_df_by_nan(dfprofile_boot)
            
            # Combine into a single DataFrame with multi-index
            dfBS_profile_c = pd.concat(bs_blocks)
            dfBS_profile_c.index.names = ["Specie", "Profile"]
            
            # Handle non-convergent bootstrap runs
            self._handle_non_convergente_bootstrap(dfBS_profile_c, dfbootstrap_mapping_c)
            
            # Store the processed data
            pmf.dfBS_profile_c = dfBS_profile_c
            pmf.dfbootstrap_mapping_c = dfbootstrap_mapping_c
            
            print(f"Successfully read constrained bootstrap with {len(dfBS_profile_c.columns)} runs")
            
        except FileNotFoundError:
            print(f"Constrained bootstrap file not found: {self.basename}_Gcon_profile_boot.xlsx")
            raise
        except Exception as e:
            print(f"Error reading constrained bootstrap: {str(e)}")
            raise

    def _get_uncertainty_sheet_name(self, file_path: str, is_constrained: bool = False) -> str:
        """
        Get the appropriate sheet name for uncertainty files.
        
        This is an extension point that can be overridden by subclasses.
        
        Parameters
        ----------
        file_path : str
            Path to the Excel file
        is_constrained : bool, default=False
            Whether this is for constrained run (vs base)
            
        Returns
        -------
        str
            Sheet name to use
        """
        # Default sheet names based on EPA PMF 5.0 structure
        return "Constrained Error Est. Summary" if is_constrained else "Error Estimation Summary"
    
    def _process_disp_swap_data(self, rawdf: pd.DataFrame, pmf: Any) -> pd.DataFrame:
        """
        Process DISP swap information from uncertainty sheets.
        
        This is an extension point that can be overridden by subclasses.
        
        Parameters
        ----------
        rawdf : pd.DataFrame
            Raw data from Excel file
        pmf : Any
            The PMF object to store data in
            
        Returns
        -------
        pd.DataFrame or None
            Processed DISP swap data
        """
        idx = rawdf.iloc[:, 1].str.contains("Swaps", na=False)
        if idx.any():
            df_swap = rawdf.loc[idx, :].dropna(axis=1).iloc[:, 1:].reset_index(drop=True)
            df_swap.columns = pmf.profiles
            df_swap.index = ["swap count"]
            return df_swap
        return None
    
    def read_base_uncertainties_summary(self) -> None:
        """
        Read the _BaseErrorEstimationSummary.xlsx file and add:

        - self.df_uncertainties_summary_b : uncertainties from BS, DISP and BS-DISP
        - self.df_disp_swap_b : DISP swap information
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles and species are loaded first
            if pmf.profiles is None or pmf.species is None:
                self.read_base_profiles()
                
            file_path = f"{self.basename}_BaseErrorEstimationSummary.xlsx"
            print(f"Loading base uncertainty summary from: {file_path}")
            
            try:
                # Get appropriate sheet name (extension point)
                sheet_name = self._get_uncertainty_sheet_name(file_path, is_constrained=False)
                
                # Try to read the sheet
                rawdf = pd.read_excel(
                    file_path,
                    sheet_name=[sheet_name],
                    header=None,
                    engine=XLSX_ENGINE
                )[sheet_name]
            except ValueError as e:
                print(f"Could not read sheet '{sheet_name}': {str(e)}")
                try:
                    # Try alternative approach - list all sheets and pick one with "Error" in name
                    available_sheets = pd.ExcelFile(file_path, engine=XLSX_ENGINE).sheet_names
                    print(f"Available sheets in file: {available_sheets}")
                    
                    # Try to find a sheet with "Error" in the name
                    error_sheets = [s for s in available_sheets if "Error" in s]
                    if error_sheets:
                        sheet_name = error_sheets[0]
                        print(f"Using alternative sheet name: {sheet_name}")
                        rawdf = pd.read_excel(
                            file_path,
                            sheet_name=[sheet_name],
                            header=None,
                            engine=XLSX_ENGINE
                        )[sheet_name]
                    else:
                        # If no error sheet found, try the first sheet
                        sheet_name = available_sheets[0]
                        print(f"Using first sheet: {sheet_name}")
                        rawdf = pd.read_excel(
                            file_path,
                            sheet_name=[sheet_name],
                            header=None,
                            engine=XLSX_ENGINE
                        )[sheet_name]
                except Exception as inner_e:
                    print(f"Could not determine sheet name: {str(inner_e)}")
                    raise ValueError(f"No suitable sheet found in {file_path}")
            
            # Clean up the data
            rawdf = rawdf.dropna(axis=0, how="all").reset_index()
            if "index" in rawdf.columns:
                rawdf = rawdf.drop("index", axis=1)
                
            # Process DISP swap information (extension point)
            df_swap = self._process_disp_swap_data(rawdf, pmf)
            if df_swap is not None:
                pmf.df_disp_swap_b = df_swap
                print(f"Read base DISP swap data: {pmf.df_disp_swap_b.iloc[0].to_dict()}")
            else:
                print("No DISP swap data found in base uncertainty summary")
                # Create empty swap data with zeros to maintain consistency
                pmf.df_disp_swap_b = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
                
            # Process uncertainties summary
            # Find rows with factor concentration data
            idx = rawdf.iloc[:, 0].str.contains("Concentrations for", na=False)
            if not idx.any():
                print("Warning: Could not find concentration data in base uncertainties file")
                pmf.df_uncertainties_summary_b = pd.DataFrame()
                return
                
            idx = rawdf.loc[idx].iloc[:, 0].dropna().index
            if len(idx) == 0:
                print("Warning: No concentration sections found in file")
                pmf.df_uncertainties_summary_b = pd.DataFrame()
                return
                
            # Extract data for all factors
            df = rawdf.loc[idx[0]+1:idx[-1]+1+pmf.nspecies, :]
            
            # Remove header rows that contain "Specie" or "Concentration"
            idx_headers = df.iloc[:, 0].str.contains("Specie|Concentration", na=False)
            df = df.drop(idx_headers[idx_headers].index)
            df = df.dropna(axis=0, how='all')
            
            # Add Profile column by repeating profile names for each species
            df["Profile"] = np.repeat(pmf.profiles, len(pmf.species)).tolist()
            
            # Set proper column names based on the type of data
            col_names = [
                "Specie", "Base run", 
                "BS 5th", "BS 25th", "BS median", "BS 75th", "BS 95th", "tmp1",
                "BS-DISP 5th", "BS-DISP average", "BS-DISP 95th", "tmp2",
                "DISP Min", "DISP average", "DISP Max",
                "Profile"
            ]
            
            # Adjust column names if we have a different number of columns
            if df.shape[1] != len(col_names):
                print(f"Warning: Expected {len(col_names)} columns but found {df.shape[1]}. Using generic column names.")
                # Create generic column names
                df.columns = [f"Col_{i}" for i in range(df.shape[1]-1)] + ["Profile"]
            else:
                df.columns = col_names
            
            # Add species names by repeating for each profile
            df["Specie"] = pmf.species * len(pmf.profiles)
            
            # Set multi-index and clean up temporary columns
            df.set_index(["Profile", "Specie"], inplace=True)
            if "tmp1" in df.columns:
                df.drop(["tmp1"], axis=1, inplace=True, errors='ignore')
            if "tmp2" in df.columns:
                df.drop(["tmp2"], axis=1, inplace=True, errors='ignore')
            
            # Convert to appropriate data types
            pmf.df_uncertainties_summary_b = df.infer_objects()
            
            print(f"Successfully read base uncertainties summary with shape {df.shape}")
            
        except FileNotFoundError:
            print(f"Base uncertainties file not found: {file_path}")
            # Create empty swap data with zeros to maintain consistency
            if hasattr(pmf, 'profiles') and pmf.profiles:
                pmf.df_disp_swap_b = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
            pmf.df_uncertainties_summary_b = pd.DataFrame()
        except Exception as e:
            print(f"Error reading base uncertainties summary: {str(e)}")
            # Create empty swap data with zeros to maintain consistency
            if hasattr(pmf, 'profiles') and pmf.profiles:
                pmf.df_disp_swap_b = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
            pmf.df_uncertainties_summary_b = pd.DataFrame()

    def read_constrained_uncertainties_summary(self) -> None:
        """
        Read the _ConstrainedErrorEstimationSummary.xlsx file and add:

        - self.df_uncertainties_summary_c : uncertainties from BS, DISP and BS-DISP
        - self.df_disp_swap_c : DISP swap information
        """
        pmf = self.pmf
        
        try:
            # Ensure profiles and species are loaded first
            if pmf.profiles is None or pmf.species is None:
                try:
                    self.read_constrained_profiles()
                except:
                    self.read_base_profiles()
                    
            file_path = f"{self.basename}_ConstrainedErrorEstimationSummary.xlsx"
            print(f"Loading constrained uncertainty summary from: {file_path}")
            
            try:
                # Get appropriate sheet name (extension point)
                sheet_name = self._get_uncertainty_sheet_name(file_path, is_constrained=True)
                
                # Try to read the sheet
                rawdf = pd.read_excel(
                    file_path,
                    sheet_name=[sheet_name],
                    header=None,
                    engine=XLSX_ENGINE
                )[sheet_name]
            except ValueError as e:
                print(f"Could not read sheet '{sheet_name}': {str(e)}")
                try:
                    # Try alternative approach - list all sheets and pick one with "Error" in name
                    available_sheets = pd.ExcelFile(file_path, engine=XLSX_ENGINE).sheet_names
                    print(f"Available sheets in file: {available_sheets}")
                    
                    # Try to find a sheet with "Error" in the name
                    error_sheets = [s for s in available_sheets if "Error" in s]
                    if error_sheets:
                        sheet_name = error_sheets[0]
                        print(f"Using alternative sheet name: {sheet_name}")
                        rawdf = pd.read_excel(
                            file_path,
                            sheet_name=[sheet_name],
                            header=None,
                            engine=XLSX_ENGINE
                        )[sheet_name]
                    else:
                        # If no error sheet found, try the first sheet
                        sheet_name = available_sheets[0]
                        print(f"Using first sheet: {sheet_name}")
                        rawdf = pd.read_excel(
                            file_path,
                            sheet_name=[sheet_name],
                            header=None,
                            engine=XLSX_ENGINE
                        )[sheet_name]
                except Exception as inner_e:
                    print(f"Could not determine sheet name: {str(inner_e)}")
                    raise ValueError(f"No suitable sheet found in {file_path}")
            
            # Clean up the data
            rawdf = rawdf.dropna(axis=0, how="all").reset_index()
            if "index" in rawdf.columns:
                rawdf = rawdf.drop("index", axis=1)
                
            # Process DISP swap information (extension point)
            df_swap = self._process_disp_swap_data(rawdf, pmf)
            if df_swap is not None:
                pmf.df_disp_swap_c = df_swap
                print(f"Read constrained DISP swap data: {pmf.df_disp_swap_c.iloc[0].to_dict()}")
            else:
                print("No DISP swap data found in constrained uncertainty summary")
                # Create empty swap data with zeros to maintain consistency
                pmf.df_disp_swap_c = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
                
            # Process uncertainties summary - almost identical to base but with different column names
            idx = rawdf.iloc[:, 0].str.contains("Concentrations for", na=False)
            if not idx.any():
                print("Warning: Could not find concentration data in constrained uncertainties file")
                pmf.df_uncertainties_summary_c = pd.DataFrame()
                return
                
            idx = rawdf.loc[idx].iloc[:, 0].dropna().index
            if len(idx) == 0:
                print("Warning: No concentration sections found in file")
                pmf.df_uncertainties_summary_c = pd.DataFrame()
                return
                
            # Extract data for all factors
            df = rawdf.loc[idx[0]+1:idx[-1]+1+pmf.nspecies, :]
            
            # Remove header rows that contain "Specie" or "Concentration"
            idx_headers = df.iloc[:, 0].str.contains("Specie|Concentration", na=False)
            df = df.drop(idx_headers[idx_headers].index)
            df = df.dropna(axis=0, how='all')
            
            # Add Profile column by repeating profile names for each species
            df["Profile"] = np.repeat(pmf.profiles, len(pmf.species)).tolist()
            
            # Set proper column names based on the type of data (constrained format is slightly different)
            col_names = [
                "Specie", "Constrained base run",
                "BS 5th", "BS median", "BS 95th", "tmp1",
                "BS-DISP 5th", "BS-DISP average", "BS-DISP 95th", "tmp2",
                "DISP Min", "DISP average", "DISP Max",
                "Profile"
            ]
            
            # Adjust column names if we have a different number of columns
            if df.shape[1] != len(col_names):
                print(f"Warning: Expected {len(col_names)} columns but found {df.shape[1]}. Using generic column names.")
                # Create generic column names
                df.columns = [f"Col_{i}" for i in range(df.shape[1]-1)] + ["Profile"]
            else:
                df.columns = col_names
            
            # Add species names by repeating for each profile
            df["Specie"] = pmf.species * len(pmf.profiles)
            
            # Set multi-index and clean up temporary columns
            df.set_index(["Profile", "Specie"], inplace=True)
            if "tmp1" in df.columns:
                df.drop(["tmp1"], axis=1, inplace=True, errors='ignore')
            if "tmp2" in df.columns:
                df.drop(["tmp2"], axis=1, inplace=True, errors='ignore')
            
            # Convert to appropriate data types
            pmf.df_uncertainties_summary_c = df.infer_objects()
            
            print(f"Successfully read constrained uncertainties summary with shape {df.shape}")
            
        except FileNotFoundError:
            print(f"Constrained uncertainties file not found: {file_path}")
            # Create empty swap data with zeros to maintain consistency
            if hasattr(pmf, 'profiles') and pmf.profiles:
                pmf.df_disp_swap_c = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
            pmf.df_uncertainties_summary_c = pd.DataFrame()
        except Exception as e:
            print(f"Error reading constrained uncertainties summary: {str(e)}")
            # Create empty swap data with zeros to maintain consistency
            if hasattr(pmf, 'profiles') and pmf.profiles:
                pmf.df_disp_swap_c = pd.DataFrame(
                    np.zeros((1, len(pmf.profiles))), 
                    columns=pmf.profiles,
                    index=["swap count"]
                )
            pmf.df_uncertainties_summary_c = pd.DataFrame()

# Fully implement the MultisitesReader class
class MultisitesReader(XlsxReader):
    """
    Reader for multi-site PMF outputs in Excel format.
    
    In multi-site mode, the 'site' parameter refers to the filename prefix 
    (like "11fnew"), NOT to a specific site within the file. The data for
    all sites is kept, with a 'Station' column identifying each site.
    
    Parameters
    ----------
    BDIR : str
        Directory containing the PMF output files
    site : str
        The filename prefix (not a specific site name)
    pmf : Any
        The parent PMF object that will store the read data
        
    Examples
    --------
    >>> # Load data from "11fnew_base.xlsx" containing multiple sites
    >>> r = MultisitesReader(BDIR="path/to/excel_files", site="11fnew", pmf=pmfobj)
    >>> r.read_all()
    """
    def __init__(self, BDIR: str, site: str, pmf: Any):
        """
        Initialize multi-site reader.
        
        Parameters
        ----------
        BDIR : str
            Directory containing the PMF output files
        site : str
            The filename prefix (like "11fnew"), NOT a specific site
        pmf : Any
            The parent PMF object that will store the read data
        """
        super().__init__(BDIR, site, pmf, multisites=True)
        print(f"MultisitesReader initialized for file prefix: {site} (will keep all sites in data)")
    
    def _process_contributions(self, dfcontrib: pd.DataFrame) -> pd.DataFrame:
        """
        Process EPA PMF 5.0 contribution data for multi-site format.
        Always preserves the Station column and doesn't filter by site.
        
        Implements the original logic from the draft code which correctly handles
        the Excel file format with Factor Contributions marker.
        """
        dfcontrib = dfcontrib.copy()
        pmf = self.pmf

        try:
            # First, find data section using "Factor Contributions" marker - from draft code
            idx = dfcontrib.iloc[:, 0].str.contains("Factor Contributions").fillna(False)
            idx = idx[idx].index.tolist()
            
            if not idx:
                print("Warning: Could not find 'Factor Contributions' marker in the sheet")
            elif len(idx) > 1:
                # Multiple sections exist, use section between first two markers
                print(f"Found multiple 'Factor Contributions' sections, using first section")
                dfcontrib = dfcontrib.iloc[idx[0]:idx[1], :]
            else:
                # Just one section exists, use all data after it
                dfcontrib = dfcontrib.iloc[idx[0]+1:, :]
        except AttributeError:
            print("WARNING: Could not parse contribution data structure")
        
        # Clean up the data
        dfcontrib.dropna(axis=1, how="all", inplace=True)
        dfcontrib.dropna(how="all", inplace=True)
        
        # Drop the first column (usually contains row labels)
        if dfcontrib.shape[1] > 0:
            dfcontrib.drop(columns=dfcontrib.columns[0], inplace=True)
                
        # Reset index and set up column names based on first row
        dfcontrib = dfcontrib.dropna(how="all").reset_index(drop=True)
        
        # Apply the exact column handling logic from draft_Read_file_Multisites.py
        if dfcontrib.shape[0] > 0:
            dfcontrib.loc[0, 1] = "Station"
            dfcontrib.loc[0, 2] = "Date"
            
            # Use first row as column names
            dfcontrib.columns = dfcontrib.loc[0, :]
            
            # Drop the header row 
            dfcontrib = dfcontrib.iloc[1:]
            
            # Convert Date column to datetime BEFORE setting it as index
            try:
                print("Converting Date column to datetime format...")
                dfcontrib["Date"] = pd.to_datetime(dfcontrib["Date"], errors="coerce")
                # Check if we have valid dates
                invalid_dates = dfcontrib["Date"].isna().sum()
                if invalid_dates > 0:
                    print(f"Warning: {invalid_dates} invalid date values found and replaced with NaT")
                
                # Set Date as index and drop rows with NaT dates
                dfcontrib = dfcontrib.set_index("Date").dropna(subset=["Date"])
                print(f"Date index type: {type(dfcontrib.index)}")
                
                # Verify we have a DatetimeIndex
                if not isinstance(dfcontrib.index, pd.DatetimeIndex):
                    print("Warning: Index is not a DatetimeIndex after processing")
            except Exception as e:
                print(f"Error converting dates: {str(e)}")
                # If date conversion fails, ensure we have a date column for later processing
                if "Date" not in dfcontrib.columns and isinstance(dfcontrib.index, pd.Index):
                    dfcontrib = dfcontrib.reset_index()
                    dfcontrib.columns = ["Date"] + list(dfcontrib.columns[1:])
        
        # Replace -999 with NaN (common missing value code in PMF)
        dfcontrib.replace({-999: np.nan}, inplace=True)
        
        # Convert data to appropriate types
        dfcontrib = dfcontrib.infer_objects()
        
        return dfcontrib

    def read_base_contributions(self) -> None:
        """Read base contributions with specific multi-site handling."""
        print("\n--- Reading Base Contributions (Multi-site) ---")
        if self.pmf.profiles is None:
            try:
                print("DEBUG: Profiles not loaded, attempting to read base profiles first.")
                self.read_base_profiles()
            except Exception as e:
                print(f"Warning: Could not read base profiles before contributions: {e}")
                return

        file_path = f"{self.basename}_base.xlsx"
        try:
            dfcontrib_raw = pd.read_excel(
                file_path,
                sheet_name=['Contributions'],
                header=None,
                engine=XLSX_ENGINE
            )["Contributions"]
            print(f"DEBUG: Successfully read raw data from {file_path}")
        except FileNotFoundError:
            print(f"Error: Base contributions file not found: {file_path}")
            self.pmf.dfcontrib_b = pd.DataFrame()
            return
        except Exception as e:
            print(f"Error reading base contributions file {file_path}: {e}")
            self.pmf.dfcontrib_b = pd.DataFrame()
            return

        # Print raw data summary
        print(f"DEBUG: Raw data shape: {dfcontrib_raw.shape}")
        
        try:
            # Process using our specialized multi-site method
            self.pmf.dfcontrib_b = self._process_contributions(dfcontrib_raw)
            if self.pmf.dfcontrib_b is not None and not self.pmf.dfcontrib_b.empty:
                print(f"Success: Multi-site base contributions loaded with shape {self.pmf.dfcontrib_b.shape}")
            else:
                print("Warning: Processed base contributions resulted in empty DataFrame")
                self.pmf.dfcontrib_b = pd.DataFrame()
        except Exception as e:
            print(f"Error processing base contributions: {str(e)}")
            self.pmf.dfcontrib_b = pd.DataFrame()

    def read_constrained_contributions(self) -> None:
        """Read constrained contributions with specific multi-site handling."""
        print("\n--- Reading Constrained Contributions (Multi-site) ---")
        if self.pmf.profiles is None:
            try:
                print("DEBUG: Profiles not loaded, attempting to read profiles first.")
                # Try constrained first
                self.read_constrained_profiles()
                # Fall back to base if needed
                if self.pmf.profiles is None:
                    self.read_base_profiles()
            except Exception as e:
                print(f"Warning: Could not read profiles before constrained contributions: {e}")
                return

        file_path = f"{self.basename}_Constrained.xlsx"
        try:
            dfcontrib_raw = pd.read_excel(
                file_path,
                sheet_name=['Contributions'],
                header=None,
                engine=XLSX_ENGINE
            )["Contributions"]
            print(f"DEBUG: Successfully read raw data from {file_path}")
        except FileNotFoundError:
            print(f"Error: Constrained contributions file not found: {file_path}")
            self.pmf.dfcontrib_c = pd.DataFrame()
            return
        except Exception as e:
            print(f"Error reading constrained contributions file {file_path}: {e}")
            self.pmf.dfcontrib_c = pd.DataFrame()
            return

        # Print raw data summary
        print(f"DEBUG: Raw data shape: {dfcontrib_raw.shape}")
        
        try:
            # Process using our specialized multi-site method
            self.pmf.dfcontrib_c = self._process_contributions(dfcontrib_raw)
            if self.pmf.dfcontrib_c is not None and not self.pmf.dfcontrib_c.empty:
                print(f"Success: Multi-site constrained contributions loaded with shape {self.pmf.dfcontrib_c.shape}")
            else:
                print("Warning: Processed constrained contributions resulted in empty DataFrame")
                self.pmf.dfcontrib_c = pd.DataFrame()
        except Exception as e:
            print(f"Error processing constrained contributions: {str(e)}")
            self.pmf.dfcontrib_c = pd.DataFrame()
    
    def _get_uncertainty_sheet_name(self, file_path: str, is_constrained: bool = False) -> str:
        """
        Override to handle different sheet names in multi-site files.
        """
        # First attempt standard names
        if is_constrained:
            return "Constrained Error Est. Summary"
        else:
            return "Error Estimation Summary"
    
    def _process_disp_swap_data(self, rawdf: pd.DataFrame, pmf: Any) -> pd.DataFrame:
        """
        Override to handle multi-site-specific DISP swap data format.
        """
        # Multi-site format needs case-insensitive search
        idx = rawdf.iloc[:, 1].str.contains("Swaps", case=False, na=False)
        if idx.any():
            print("Found DISP swap information in multi-site format")
            df_swap = rawdf.loc[idx, :].dropna(axis=1).iloc[:, 1:].reset_index(drop=True)
            
            # Ensure we have the right number of columns for the profiles
            if df_swap.shape[1] < len(pmf.profiles):
                print(f"Warning: Swap data has {df_swap.shape[1]} columns but {len(pmf.profiles)} profiles")
                # Try to pad with NaNs if needed
                for i in range(df_swap.shape[1], len(pmf.profiles)):
                    df_swap[i] = np.nan
            elif df_swap.shape[1] > len(pmf.profiles):
                print(f"Warning: Swap data has more columns ({df_swap.shape[1]}) than profiles ({len(pmf.profiles)})")
                # Use only the first N columns
                df_swap = df_swap.iloc[:, :len(pmf.profiles)]
            
            # Set column names to profile names
            df_swap.columns = pmf.profiles
            df_swap.index = ["swap count"]
            return df_swap
        else:
            return None
        
    