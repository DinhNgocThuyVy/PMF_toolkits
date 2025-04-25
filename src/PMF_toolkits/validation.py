"""
PMF output validation module for comparing model results with reference profiles.

This module provides tools to validate PMF output by comparing factor profiles 
with reference profiles from a library database, calculating ratios between 
species, and assessing similarities using PD (Pearson Distance) and SID 
(Standardized Identity Distance) metrics.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from pathlib import Path
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set

from .core import PMF
from .analysis import compute_similarity_metrics, compute_SID, compute_PD

# Configure logger
logger = logging.getLogger('PMF_toolkits.validation')

class OutputValidator:
    """
    Validate and compare PMF output with reference data and standards.
    
    This class provides methods to evaluate PMF factors against reference profiles,
    check elemental ratios against expected ranges, and visualize similarity metrics.
    
    Parameters
    ----------
    site : str
        Site name for the PMF analysis
    bdir : str
        Base directory containing PMF output files
    reference_dir : str, optional
        Directory containing reference data files (default: "reference_data/")
    """
    
    # Define species mapping for standardization
    SPECIES_MAPPING = {
        # Carbon fractions
        "OC": "Organic carbon",
        "OC*": "Organic carbon",
        "EC": "Elemental Carbon",
        "TC": "Total carbon",
        "BC": "Black Carbon", 
        "BCff": "Black Carbon", 
        "BCwb": "Brown Carbon",
        
        # Ions
        "Cl-": "Chloride ion",
        "NO3-": "Nitrate",
        "SO42-": "Sulfate",
        "SO4--": "Sulfate", 
        "SO4²-": "Sulfate",
        "Na+": "Sodium ion",
        "NH4+": "Ammonium",
        "K+": "Potassium ion",
        "Mg2+": "Magnesium",
        "Ca2+": "Calcium",
        
        # Elements
        "As": "Arsenic",
        "Ba": "Barium",
        "Cd": "Cadmium",
        "Cr": "Chromium",
        "Cu": "Copper",
        "Mn": "Manganese",
        "Ni": "Nickel",
        "Pb": "Lead",
        "Sb": "Antimony",
        "V": "Vanadium",
        "Zn": "Zinc",
        "Fe": "Iron",
        "Rb": "Rubidium",
        "Se": "Selenium",
        "Sn": "Tin",
        "Ti": "Titanium",
        "Al": "Aluminum",
        "Cs": "Cesium",
        "Sr": "Strontium",
        
        # Organic tracers
        "Levoglucosan": "Levoglucosan",
        "Mannosan": "Mannosan",
        "Galactosan": "Galactosan",
        "Arabitol": "Arabitol",
        "Mannitol": "Mannitol",
        "MSA": "Methanesulfonate",
        "Oxalate": "Oxalate",
        "Phthalic": "Phthalic acid",
        "Maleic": "Maleic acid",
    }
    
    # Define key diagnostic ratios for different source types
    SOURCE_DIAGNOSTIC_RATIOS = {
        "Biomass burning": [
            ("Levoglucosan", "Mannosan"),
            ("OC", "EC"),
            ("K+", "EC"),
            ("K+", "Levoglucosan")
        ],
        "Traffic": [
            ("Cu", "Sb"),
            ("OC", "EC"),
            ("Fe", "Cu")
        ],
        "Marine": [
            ("Cl-", "Na+"),
            ("Mg2+", "Na+"),
            ("SO42-", "Na+")
        ],
        "Dust": [
            ("Ca2+", "Al"),
            ("Fe", "Al"),
            ("Ti", "Al")
        ],
        "Oil combustion": [
            ("V", "Ni"),
            ("Ni", "EC")
        ],
        "Secondary sulfate": [
            ("NH4+", "SO42-")
        ],
        "Secondary nitrate": [
            ("NH4+", "NO3-")
        ]
    }
    
    def __init__(self, site: str, bdir: str, reference_dir: Optional[str] = None):
        """
        Initialize OutputValidator with site information and directories.
        
        Parameters
        ----------
        site : str
            Site name for the PMF run
        bdir : str
            Base directory containing PMF output files
        reference_dir : str, optional
            Directory containing reference data files
        """
        self.site = site
        self.bdir = bdir
        
        # Set reference directory - use default if not provided
        if reference_dir is None:
            self.reference_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reference_data")
        else:
            self.reference_dir = reference_dir
            
        # Cache for reference data
        self._ratio_cache = None
        self._source_profiles_cache = None
        self._species_eu_cache = None
        
        # Initialize the PMF object but don't read files yet
        self.pmf = None
        self.load_pmf()

    def load_pmf(self):
        """
        Load the PMF data from files.
        """
        if self.pmf is None:
            self.pmf = self.read_pmf()
    
    def _get_reference_file_path(self, filename: str) -> str:
        """
        Get full path to a reference file, checking various locations.
        
        Parameters
        ----------
        filename : str
            Name of the reference file

        Returns
        -------
        str
            Full path to the reference file
        
        Raises
        ------
        FileNotFoundError
            If the reference file cannot be found
        """
        # Check in reference_dir
        if os.path.exists(os.path.join(self.reference_dir, filename)):
            return os.path.join(self.reference_dir, filename)
        
        # Check alternative locations
        alt_locations = [
            filename,
            f"../Data/{filename}",
            f"Data/{filename}",
            f"../Result/Table/{filename}",
            os.path.join(os.getcwd(), filename),
            os.path.join(os.getcwd(), "reference_data", filename),
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                return location
                
        raise FileNotFoundError(f"Reference file '{filename}' not found in {self.reference_dir} or alternative locations")

    def load_ratio_data(self) -> pd.DataFrame:
        """
        Load reference ratio data for element pairs.

        Returns
        -------
        pd.DataFrame
            DataFrame containing reference ratios for element pairs
        """
        if self._ratio_cache is not None:
            return self._ratio_cache
            
        try:
            filename = "Ratio.xlsx"
            filepath = self._get_reference_file_path(filename)
            ratio_data = pd.read_excel(filepath)
            self._ratio_cache = ratio_data
            return ratio_data
        except FileNotFoundError as e:
            logger.warning(f"Ratio file not found: {e}")
            return pd.DataFrame(columns=["Specie 1", "Specie2", "Min", "Max", "Source"])
        except Exception as e:
            logger.error(f"Error loading ratio data: {str(e)}")
            return pd.DataFrame(columns=["Specie 1", "Specie2", "Min", "Max", "Source"])
    
    def load_species_eu_data(self) -> pd.DataFrame:
        """
        Load European species data for reference.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing species data
        """
        if self._species_eu_cache is not None:
            return self._species_eu_cache
            
        try:
            filename = "SpecieEU.xlsx"
            filepath = self._get_reference_file_path(filename)
            species_data = pd.read_excel(filepath)
            self._species_eu_cache = species_data
            return species_data
        except FileNotFoundError as e:
            logger.warning(f"Species EU file not found: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading species EU data: {str(e)}")
            return pd.DataFrame()
    
    def load_source_profiles(self) -> pd.DataFrame:
        """
        Load reference source profiles.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing reference source profiles
        """
        if self._source_profiles_cache is not None:
            return self._source_profiles_cache
            
        try:
            filename = "Source_reference_profiles.csv"
            filepath = self._get_reference_file_path(filename)
            df = pd.read_csv(
                filepath,
                encoding='unicode_escape', 
                on_bad_lines='skip',
                na_values=["not detected", "Volos"],
                sep=";"
            )
            df["Relative Mass"] = df["Relative Mass"].apply(lambda x: np.nan if x < 0 else x)
            df = df.dropna(how="all", axis=0)
            self._source_profiles_cache = df
            return df
        except FileNotFoundError as e:
            logger.warning(f"Source profiles file not found: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading source profiles: {str(e)}")
            return pd.DataFrame()
    
    def load_color_table(self) -> pd.DataFrame:
        """
        Load color table for plot styling.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing color assignments for sources/species
        """
        try:
            filename = "color_table.xlsx"
            filepath = self._get_reference_file_path(filename)
            color_table = pd.read_excel(filepath)
            return color_table
        except FileNotFoundError as e:
            logger.warning(f"Color table not found: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading color table: {str(e)}")
            return pd.DataFrame()

    def read_pmf(self, multisite: bool = False) -> PMF:
        """
        Read PMF data from output files.
        
        Parameters
        ----------
        multisite : bool, default=False
            Whether to read multi-site data
            
        Returns
        -------
        PMF
            Loaded PMF object
        """
        pmf = PMF(site=self.site, reader="xlsx", BDIR=self.bdir, multisites=multisite)
        pmf.read.read_all()
        return pmf
    
    def read_bootstrap_profiles(self, multisite: bool = False) -> pd.DataFrame:
        """
        Read bootstrap profiles from PMF output.
        
        Parameters
        ----------
        multisite : bool, default=False
            Whether to read multi-site data
            
        Returns
        -------
        pd.DataFrame
            Bootstrap profiles
        """
        self.load_pmf()
        if hasattr(self.pmf, 'dfBS_profile_c') and self.pmf.dfBS_profile_c is not None:
            return self.pmf.dfBS_profile_c
        elif hasattr(self.pmf, 'dfBS_profile_b') and self.pmf.dfBS_profile_b is not None:
            return self.pmf.dfBS_profile_b
        else:
            logger.warning("No bootstrap profiles available")
            return pd.DataFrame()
    
    def read_regression_diagnostics(self, constrained: bool = True) -> pd.DataFrame:
        """
        Read regression diagnostics from PMF output files.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        pd.DataFrame
            Regression diagnostics data
        """
        try:
            self.load_pmf()
            if constrained:
                directory = f"{self.bdir}/{self.site}_Constrained.xlsx"
            else:
                directory = f"{self.bdir}/{self.site}_base.xlsx"
                
            dfbase = pd.read_excel(
                directory,
                sheet_name=['Profiles'],
                header=None,
            )["Profiles"]
            
            # Find regression diagnostics section
            idx = dfbase.iloc[:, 0].str.contains("Regression diagnostics", na=False)
            if not idx.any():
                return pd.DataFrame()
                
            idx_1 = idx[idx].index.tolist()[0]
            dfbase = dfbase.loc[idx_1:, :]
            
            # Clean up data
            dfbase.dropna(axis=1, how="all", inplace=True)
            dfbase.dropna(axis=0, how="all", inplace=True)
            dfbase.reset_index(drop=True, inplace=True)
            
            # Find header row
            header_idx = dfbase.iloc[:, 0].str.contains("Species", na=False)
            if not header_idx.any():
                return pd.DataFrame()
                
            header_row = header_idx[header_idx].index[0]
            
            # Set column names from header row
            dfbase.columns = dfbase.iloc[header_row]
            
            # Extract data rows
            dfbase = dfbase.iloc[header_row+1:].copy()
            dfbase.set_index("Species", inplace=True)
            
            # Rename columns for clarity
            if "R-Square" in dfbase.columns:
                dfbase = dfbase.rename(columns={"R-Square": "R2"})
                
            return dfbase
        except Exception as e:
            logger.error(f"Error reading regression diagnostics: {str(e)}")
            return pd.DataFrame()
    
    def normalize_species_name(self, species_name: str) -> str:
        """
        Normalize species names for consistent comparison.
        
        Parameters
        ----------
        species_name : str
            Species name to normalize
            
        Returns
        -------
        str
            Normalized species name
        """
        # Remove common suffixes and prefixes
        clean_name = re.sub(r'[-_\s+]', '', species_name)
        
        # Try direct match
        if species_name in self.SPECIES_MAPPING:
            return self.SPECIES_MAPPING[species_name]
        
        # Try case-insensitive match
        for key, value in self.SPECIES_MAPPING.items():
            if key.lower() == species_name.lower():
                return value
                
        # Try to match clean_name
        for key, value in self.SPECIES_MAPPING.items():
            if re.sub(r'[-_\s+]', '', key).lower() == clean_name.lower():
                return value
        
        # Return original if no match found
        return species_name
    
    def find_species_in_profile(self, species_name: str, constrained: bool = True) -> str:
        """
        Find best match for a species name in available profiles.
        
        Parameters
        ----------
        species_name : str
            Species name to find
        constrained : bool, default=True
            Whether to use constrained profiles
            
        Returns
        -------
        str
            Actual species name in profiles, or None if not found
        """
        self.load_pmf()
        
        # Get appropriate profile data
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return None
            
        # First try exact match
        if species_name in profile.index:
            return species_name
        
        # Try common variants
        # For example, "OC" might be stored as "OC*"
        variants = {
            "OC": ["OC*", "Organic Carbon"],
            "SO4": ["SO42-", "SO4--", "SO4--", "Sulfate"],
            "NO3": ["NO3-", "NO3_", "Nitrate"],
            "Cl": ["Cl-", "Cl⁻", "Chloride"],
            "NH4": ["NH4+", "NH4⁺", "Ammonium"],
            "Na": ["Na+", "Na⁺", "Sodium"],
            "K": ["K+", "K⁺", "Potassium"]
        }
        
        # Check if our target is in the variants dict
        for base, options in variants.items():
            if species_name == base or species_name in options:
                # Try each variant
                for variant in options + [base]:
                    if variant in profile.index:
                        return variant
        
        # Try case-insensitive match
        for idx in profile.index:
            if idx.lower() == species_name.lower():
                return idx
        
        # Nothing found
        return None
    
    def calculate_ratio(self, source: str, specie1: str, specie2: str, constrained: bool = True) -> float:
        """
        Calculate ratio between two species for a given source.
        
        Parameters
        ----------
        source : str
            Source/factor name
        specie1 : str
            First species
        specie2 : str
            Second species
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        float
            Calculated ratio value
        """
        self.load_pmf()
        
        # Get appropriate profile data
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return float('nan')
        
        # Check if source exists
        if source not in profile.columns:
            logger.warning(f"Source '{source}' not found in profiles")
            return float('nan')
        
        # Try to find the best match for species names
        specie1_actual = self.find_species_in_profile(specie1, constrained)
        specie2_actual = self.find_species_in_profile(specie2, constrained)
        
        if specie1_actual is None:
            logger.warning(f"Species '{specie1}' not found in profiles")
            return float('nan')
        
        if specie2_actual is None:
            logger.warning(f"Species '{specie2}' not found in profiles")
            return float('nan')
        
        # Check for zero denominator
        denominator = profile.loc[specie2_actual, source]
        if np.isclose(denominator, 0) or np.isnan(denominator):
            logger.warning(f"Denominator species '{specie2_actual}' has zero or NaN value, cannot calculate ratio")
            return float('nan')
        
        # Calculate ratio
        ratio = profile.loc[specie1_actual, source] / denominator
        return ratio
    
    def compare_ratio(self, source: str, specie1: str, specie2: str, constrained: bool = True) -> Dict:
        """
        Compare species ratio for a factor against reference values.
        
        Parameters
        ----------
        source : str
            Source/factor name
        specie1 : str
            First species
        specie2 : str
            Second species
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        Dict
            Dictionary containing:
            - success: bool - Whether any matches were found
            - matches: list - List of matching reference sources
        """
        # Calculate the ratio first
        ratio_value = self.calculate_ratio(source, specie1, specie2, constrained)
        
        if np.isnan(ratio_value):
            return {"success": False, "matches": []}
        
        # Try to find best matches for the species names
        specie1_actual = self.find_species_in_profile(specie1, constrained) or specie1
        specie2_actual = self.find_species_in_profile(specie2, constrained) or specie2
        
        # Get ratio reference file
        try:
            ratio_file = self._get_reference_file_path("Ratio.xlsx")
        except FileNotFoundError:
            logger.warning("Ratio reference file not found")
            ratio_file = None
        
        # Now use the ratio_comparison function but convert the result
        success, matches = ratio_comparison(specie1_actual, specie2_actual, ratio_value, ratio_file)
        
        # Convert to the expected dictionary format
        return {
            "success": success,
            "matches": matches
        }
    
    def auto_detect_source_type(self, source: str, constrained: bool = True) -> Dict[str, Any]:
        """
        Automatically detect the most likely source type based on key diagnostic ratios.
        
        Parameters
        ----------
        source : str
            Source/factor name
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        Dict
            Dictionary containing source type analysis
        """
        results = {
            "source": source,
            "likely_type": None,
            "confidence": 0.0,  # 0 to 1 scale
            "ratios_analyzed": [],
            "matching_profiles": [],
            "supporting_evidence": []
        }
        
        # Calculate all applicable diagnostic ratios
        ratio_results = {}
        available_species = set()
        
        # Get available species
        self.load_pmf()
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            return results
            
        if source not in profile.columns:
            return results
            
        for species in profile.index:
            available_species.add(species)
            # Also add any potential alternate names
            for base, variants in {
                "OC": ["OC*", "Organic Carbon"],
                "SO4": ["SO42-", "SO4--", "Sulfate"],
                "NO3": ["NO3-", "Nitrate"]
            }.items():
                if species in variants:
                    available_species.add(base)
        
        # Try each diagnostic ratio if both species are available
        for source_type, ratios in self.SOURCE_DIAGNOSTIC_RATIOS.items():
            for specie1, specie2 in ratios:
                # Check if both species are found or have equivalents
                specie1_found = self.find_species_in_profile(specie1, constrained)
                specie2_found = self.find_species_in_profile(specie2, constrained)
                
                if specie1_found and specie2_found:
                    try:
                        ratio_value = self.calculate_ratio(source, specie1_found, specie2_found, constrained)
                        if not np.isnan(ratio_value):
                            ratio_comparison_result = self.compare_ratio(source, specie1_found, specie2_found, constrained)
                            ratio_results[(specie1, specie2)] = {
                                "value": ratio_value,
                                "matches": ratio_comparison_result.get("matches", []),
                                "success": ratio_comparison_result.get("success", False)
                            }
                            results["ratios_analyzed"].append({
                                "ratio": f"{specie1}/{specie2}",
                                "actual_species": f"{specie1_found}/{specie2_found}",
                                "value": ratio_value,
                                "matches": [m["Source"] for m in ratio_comparison_result.get("matches", [])]
                            })
                    except Exception as e:
                        logger.debug(f"Error calculating {specie1}/{specie2} ratio: {str(e)}")
        
        # Calculate source type scores based on matching ratios
        source_scores = {}
        for (specie1, specie2), result in ratio_results.items():
            if result["success"]:
                for match in result["matches"]:
                    source_type = match["Source"]
                    if source_type not in source_scores:
                        source_scores[source_type] = {"matches": 0, "total_analyzed": 0}
                    source_scores[source_type]["matches"] += 1
                    
        # Count how many ratios were analyzed for each source type
        for source_type, ratios in self.SOURCE_DIAGNOSTIC_RATIOS.items():
            for specie1, specie2 in ratios:
                if (specie1, specie2) in ratio_results:
                    if source_type not in source_scores:
                        source_scores[source_type] = {"matches": 0, "total_analyzed": 0}
                    source_scores[source_type]["total_analyzed"] += 1
        
        # Calculate confidence scores and find best match
        best_score = 0
        for source_type, score_data in source_scores.items():
            if score_data["total_analyzed"] > 0:
                confidence = score_data["matches"] / score_data["total_analyzed"]
                score_data["confidence"] = confidence
                if confidence > best_score:
                    best_score = confidence
                    results["likely_type"] = source_type
                    results["confidence"] = confidence
        
        # Also get similarity metrics for reference profiles
        similarity_metrics = self.calculate_similarity_metrics(source, constrained=constrained)
        if not similarity_metrics.empty:
            best_matches = similarity_metrics.sort_values("PD").head(3)
            results["matching_profiles"] = best_matches.to_dict("records")
        
        # Collect supporting evidence
        if results["likely_type"]:
            results["supporting_evidence"].append(f"Matched {source_scores[results['likely_type']]['matches']} out of {source_scores[results['likely_type']]['total_analyzed']} diagnostic ratios for {results['likely_type']}")
            
            # Add specific ratio evidence
            for ratio_info in results["ratios_analyzed"]:
                if any(results["likely_type"] in match for match in ratio_info["matches"]):
                    results["supporting_evidence"].append(f"{ratio_info['ratio']} ratio of {ratio_info['value']:.2f} matches {results['likely_type']} profile")
        
        return results

    def calculate_similarity_metrics(self, source: str, constrained: bool = True) -> pd.DataFrame:
        """
        Calculate similarity metrics between PMF factor and reference profiles.
        
        Parameters
        ----------
        source : str
            Source/factor name
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing similarity metrics for all reference profiles
        """
        self.load_pmf()
        
        # Get factor profile
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return pd.DataFrame()
            
        if source not in profile.columns:
            logger.warning(f"Source '{source}' not found in profiles")
            return pd.DataFrame()
            
        # Load reference profiles
        ref_data = self.load_source_profiles()
        if ref_data.empty:
            logger.warning("No reference profiles available")
            return pd.DataFrame()
            
        # Prepare results
        results = []
        
        # Get normalized PMF profile
        if hasattr(self.pmf, 'totalVar') and self.pmf.totalVar is not None and self.pmf.totalVar in profile.index:
            pmf_profile = profile[source] / profile.loc[self.pmf.totalVar, source]
        else:
            pmf_profile = profile[source] / profile[source].sum()
            
        # Iterate through unique reference sources
        for ref_id in ref_data["Id"].unique():
            ref_subset = ref_data[ref_data["Id"] == ref_id]
            
            if ref_subset.empty:
                continue
                
            # Extract reference name and other metadata
            ref_name = ref_subset["Name"].iloc[0] if "Name" in ref_subset.columns else f"Reference {ref_id}"
            ref_place = ref_subset["Place"].iloc[0] if "Place" in ref_subset.columns else "Unknown"
            ref_size = ref_subset["Particle Size"].iloc[0] if "Particle Size" in ref_subset.columns else "Unknown"
            
            # Prepare reference profile
            ref_profile = pd.Series(index=ref_subset["Specie"], data=ref_subset["Relative Mass"].values)
            ref_profile.dropna(inplace=True)
            
            # Map PMF species to reference species nomenclature for comparison
            pmf_profile_mapped = pd.Series()
            for sp in pmf_profile.index:
                mapped_name = self.normalize_species_name(sp)
                if mapped_name in ref_profile.index:
                    pmf_profile_mapped[mapped_name] = pmf_profile[sp]
            
            # Calculate metrics if we have enough common species
            common_species = set(pmf_profile_mapped.index) & set(ref_profile.index)
            if len(common_species) >= 4:  # At least 4 common species
                # Calculate SID
                sid = compute_SID(
                    pd.DataFrame({source: pmf_profile_mapped}),
                    pd.DataFrame({"ref": ref_profile}),
                    source, "ref", isRelativeMass=True
                )
                
                # Calculate PD
                pd_value = compute_PD(
                    pd.DataFrame({source: pmf_profile_mapped}),
                    pd.DataFrame({"ref": ref_profile}),
                    source, "ref", isRelativeMass=True
                )
                
                results.append({
                    "Reference ID": ref_id,
                    "Reference Name": ref_name,
                    "Place": ref_place,
                    "Particle Size": ref_size,
                    "SID": sid,
                    "PD": pd_value,
                    "Shared Species": len(common_species),
                    "Common Species": ", ".join(sorted(common_species))
                })
        
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=["Reference ID", "Reference Name", "Place", 
                                      "Particle Size", "SID", "PD", "Shared Species", "Common Species"])

    def find_similar_sources(self, source: str, pd_threshold: float = 0.4, 
                           sid_threshold: float = 0.8, min_species: int = 5) -> pd.DataFrame:
        """
        Find similar sources to the given factor from reference profiles.
        
        Parameters
        ----------
        source : str
            Source/factor name
        pd_threshold : float, default=0.4
            Maximum Pearson Distance for considering sources similar
        sid_threshold : float, default=0.8
            Maximum Standardized Identity Distance for considering sources similar
        min_species : int, default=5
            Minimum number of shared species
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing similar sources
        """
        # Calculate similarity metrics for all references
        metrics = self.calculate_similarity_metrics(source)
        
        if metrics.empty:
            return pd.DataFrame()
            
        # Filter based on thresholds
        similar = metrics[
            (metrics["PD"] <= pd_threshold) & 
            (metrics["SID"] <= sid_threshold) & 
            (metrics["Shared Species"] >= min_species)
        ].copy()
        
        # Sort by similarity (lower PD is better)
        return similar.sort_values("PD")

    def get_key_species_for_source(self, source_type: str) -> List[str]:
        """
        Get key species/tracers for a source type.
        
        Parameters
        ----------
        source_type : str
            Source type (e.g., "Biomass burning", "Traffic")
            
        Returns
        -------
        List[str]
            List of key species for this source type
        """
        # Define key species for different source types
        key_species_map = {
            "Biomass burning": ["Levoglucosan", "Mannosan", "Galactosan", "K+", "OC", "EC"],
            "Traffic": ["EC", "Cu", "Sb", "Zn", "Fe", "OC"],
            "Marine": ["Na+", "Cl-", "Mg2+"],
            "Dust": ["Al", "Si", "Ca2+", "Fe", "Ti"],
            "Oil combustion": ["V", "Ni"],
            "Secondary sulfate": ["SO42-", "NH4+"],
            "Secondary nitrate": ["NO3-", "NH4+"],
            "Secondary organic": ["Oxalate", "MSA"]
        }
        
        # Return list for the requested source type, or empty list if not found
        return key_species_map.get(source_type, [])
    
    def analyze_missing_species(self, constrained: bool = True) -> Dict[str, List[str]]:
        """
        Analyze which key source tracers are missing from the dataset.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained run profiles
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping source types to lists of missing key tracers
        """
        self.load_pmf()
        
        # Get profile data
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return {}
            
        # Get available species (including possible variants)
        available_species = set()
        for species in profile.index:
            available_species.add(species)
            # Also check for alternate forms
            for base, variants in {
                "OC": ["OC*", "Organic Carbon"],
                "EC": ["Elemental Carbon", "BC"],
                "SO4": ["SO42-", "SO4²-", "Sulfate"],
                "NO3": ["NO3-", "Nitrate"],
                "NH4": ["NH4+", "Ammonium"],
                "Cl": ["Cl-", "Chloride"],
                "Na": ["Na+", "Sodium"],
                "K": ["K+", "Potassium"],
                "Ca": ["Ca2+", "Calcium"],
                "Mg": ["Mg2+", "Magnesium"]
            }.items():
                if species in variants:
                    available_species.add(base)
                elif species == base:
                    for v in variants:
                        available_species.add(v)
        
        # Check for missing species for each source type
        missing = {}
        for source_type, key_species in {
            "Biomass burning": self.get_key_species_for_source("Biomass burning"),
            "Traffic": self.get_key_species_for_source("Traffic"),
            "Marine": self.get_key_species_for_source("Marine"),
            "Dust": self.get_key_species_for_source("Dust"),
            "Oil combustion": self.get_key_species_for_source("Oil combustion"),
            "Secondary sulfate": self.get_key_species_for_source("Secondary sulfate"),
            "Secondary nitrate": self.get_key_species_for_source("Secondary nitrate"),
            "Secondary organic": self.get_key_species_for_source("Secondary organic")
        }.items():
            # For each species, check if it or any variant is available
            missing_species = []
            for species in key_species:
                found = False
                for available in available_species:
                    if (species.lower() == available.lower() or 
                        self.normalize_species_name(species) == self.normalize_species_name(available)):
                        found = True
                        break
                if not found:
                    missing_species.append(species)
                    
            if missing_species:
                missing[source_type] = missing_species
                
        return missing

    def plot_similarity_diagram(self, source: str, max_sources: int = 20, 
                              figsize: tuple = (10, 8), 
                              dpi: int = 200, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create similarity diagram comparing factor to reference profiles.
        
        Parameters
        ----------
        source : str
            Source/factor name
        max_sources : int, default=20
            Maximum number of reference sources to include
        figsize : tuple, default=(10, 8)
            Figure size
        dpi : int, default=200
            DPI for figure
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        # Get similarity metrics
        metrics = self.calculate_similarity_metrics(source)
        
        if metrics.empty:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No reference data available", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            return plt.gcf()
        
        # Filter and sort by PD
        metrics_sorted = metrics.sort_values("PD")
        
        # Limit to max_sources
        if len(metrics_sorted) > max_sources:
            metrics_sorted = metrics_sorted.head(max_sources)
            
        # Create the plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        
        # Plot each reference as a point
        for i, (_, row) in enumerate(metrics_sorted.iterrows()):
            marker_size = min(100, max(20, row["Shared Species"] * 5))  # Scale marker by shared species
            ax.scatter(
                row["SID"], row["PD"], 
                s=marker_size,
                alpha=0.7,
                label=f"{row['Reference Name']} ({row['Shared Species']})"
            )
            
        # Add rectangle for "good match" region
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (0, 0), 
            width=1, 
            height=0.4, 
            facecolor="green", 
            alpha=0.2, 
            zorder=-1
        )
        ax.add_patch(rect)
        
        # Set plot details
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("SID (Standardized Identity Distance)", fontsize=14)
        ax.set_ylabel("PD (Pearson Distance)", fontsize=14)
        ax.set_title(f'Source Profile Similarity: {source}', fontsize=16)
        
        # Add legend
        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.100, right=0.70)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=10)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor='white')
            
        return fig
    
    def analyze_all_key_ratios(self, source: str, constrained: bool = True) -> pd.DataFrame:
        """
        Calculate and analyze all key diagnostic ratios for a source.
        
        Parameters
        ----------
        source : str
            Source/factor name
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        pd.DataFrame
            DataFrame with ratio analysis results
        """
        self.load_pmf()
        
        # Get profile data
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profile = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profile = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return pd.DataFrame()
            
        if source not in profile.columns:
            logger.warning(f"Source '{source}' not found in profiles")
            return pd.DataFrame()
            
        # Collect all possible ratio pairs
        ratio_pairs = []
        for source_type, ratios in self.SOURCE_DIAGNOSTIC_RATIOS.items():
            for specie1, specie2 in ratios:
                ratio_pairs.append((specie1, specie2, source_type))
                
        # Calculate ratios and collect results
        results = []
        
        for specie1, specie2, source_type in ratio_pairs:
            # Find species in the data
            specie1_found = self.find_species_in_profile(specie1, constrained)
            specie2_found = self.find_species_in_profile(specie2, constrained)
            
            if specie1_found and specie2_found:
                try:
                    ratio_value = self.calculate_ratio(source, specie1_found, specie2_found, constrained)
                    if not np.isnan(ratio_value):
                        ratio_result = self.compare_ratio(source, specie1_found, specie2_found, constrained)
                        
                        # Get matching sources from ratio comparison
                        matching_sources = [m["Source"] for m in ratio_result.get("matches", [])]
                        min_max_range = [(m["Min"], m["Max"]) for m in ratio_result.get("matches", [])]
                        
                        results.append({
                            "Ratio": f"{specie1}/{specie2}",
                            "Actual Species": f"{specie1_found}/{specie2_found}",
                            "Value": ratio_value,
                            "Diagnostic For": source_type,
                            "Matches": ", ".join(matching_sources) if matching_sources else "None",
                            "Range": ", ".join([f"{min:.2f}-{max:.2f}" for min, max in min_max_range]) if min_max_range else "N/A",
                            "Success": ratio_result.get("success", False)
                        })
                except Exception as e:
                    # Skip problematic ratios
                    logger.debug(f"Error calculating {specie1}/{specie2} ratio: {str(e)}")
        
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=["Ratio", "Actual Species", "Value", "Diagnostic For", "Matches", "Range", "Success"])
    
    def validate_all_sources(self, constrained: bool = True) -> pd.DataFrame:
        """
        Validate all sources in the PMF solution.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained run results
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing validation results for all sources
        """
        self.load_pmf()
        
        # Get profiles and list of sources
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profiles = self.pmf.dfprofiles_c
            sources = profiles.columns.tolist()
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profiles = self.pmf.dfprofiles_b
            sources = profiles.columns.tolist()
        else:
            logger.warning("No profile data available")
            return pd.DataFrame()
        
        # Prepare results
        results = []
        
        # Process each source
        for source in sources:
            logger.info(f"Validating source: {source}")
            
            # Auto-detect source type
            source_detection = self.auto_detect_source_type(source, constrained)
            likely_type = source_detection.get("likely_type")
            confidence = source_detection.get("confidence", 0)
            
            # Get best matching reference
            metrics = self.calculate_similarity_metrics(source, constrained=constrained)
            
            if not metrics.empty:
                best_match = metrics.sort_values("PD").iloc[0]
                
                # Calculate key ratios if available
                ratio_pairs = [
                    ("OC", "EC"),
                    ("K+", "SO42-"),
                    ("NO3-", "SO42-"),
                    ("Levoglucosan", "Mannosan"),
                    ("Cu", "Sb"),
                    ("V", "Ni")
                ]
                
                ratios = {}
                for specie1, specie2 in ratio_pairs:
                    try:
                        specie1_found = self.find_species_in_profile(specie1, constrained)
                        specie2_found = self.find_species_in_profile(specie2, constrained)
                        
                        if specie1_found and specie2_found:
                            ratio = self.calculate_ratio(source, specie1_found, specie2_found, constrained)
                            if not np.isnan(ratio):
                                ratios[f"{specie1}/{specie2}"] = ratio
                    except Exception:
                        pass
                
                result = {
                    "Source": source,
                    "Best Match": best_match["Reference Name"],
                    "PD": best_match["PD"],
                    "SID": best_match["SID"],
                    "Shared Species": best_match["Shared Species"],
                    "Likely Type": likely_type or "Unknown",
                    "Confidence": f"{confidence:.0%}" if confidence > 0 else "Low",
                    "Match Quality": "Good" if (best_match["PD"] < 0.4 and best_match["SID"] < 0.8) else "Poor"
                }
                
                # Add ratios
                result.update(ratios)
                results.append(result)
            else:
                results.append({
                    "Source": source,
                    "Best Match": "No matches found",
                    "PD": np.nan,
                    "SID": np.nan,
                    "Shared Species": 0,
                    "Likely Type": likely_type or "Unknown",
                    "Confidence": f"{confidence:.0%}" if confidence > 0 else "Low",
                    "Match Quality": "Unknown"
                })
        
        return pd.DataFrame(results)
    
    def plot_key_species_by_factor(self, constrained: bool = True, 
                                 normalize: bool = True,
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot heatmap showing key species contribution by factor.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained run profiles
        normalize : bool, default=True
            Whether to normalize contributions by row (species)
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        plt.Figure
            Matplotlib Figure object
        """
        self.load_pmf()
        
        # Get profile data
        if constrained and hasattr(self.pmf, 'dfprofiles_c') and self.pmf.dfprofiles_c is not None:
            profiles = self.pmf.dfprofiles_c
        elif hasattr(self.pmf, 'dfprofiles_b') and self.pmf.dfprofiles_b is not None:
            profiles = self.pmf.dfprofiles_b
        else:
            logger.warning("No profile data available")
            return None
            
        # Collect key species across all source types
        key_species_set = set()
        for source_type in self.SOURCE_DIAGNOSTIC_RATIOS.keys():
            key_species_set.update(self.get_key_species_for_source(source_type))
            
        # Find which key species are present in the data
        present_species = []
        for species in key_species_set:
            species_found = self.find_species_in_profile(species, constrained)
            if species_found:
                present_species.append(species_found)
                
        # If we have too few species, add some from the data
        if len(present_species) < 5 and profiles.shape[0] > 5:
            # Add some of the top species by concentration
            missing_count = 5 - len(present_species)
            top_species = profiles.sum(axis=1).sort_values(ascending=False).index
            for sp in top_species:
                if sp not in present_species and sp != self.pmf.totalVar:
                    present_species.append(sp)
                    missing_count -= 1
                    if missing_count <= 0:
                        break
        
        # Filter profile data to include only key species
        key_profiles = profiles.loc[present_species]
        
        # Normalize if requested
        if normalize:
            # Normalize by row (species)
            row_sums = key_profiles.sum(axis=1)
            key_profiles = key_profiles.div(row_sums, axis=0) * 100
            
        # Set up the figure
        if figsize is None:
            figsize = (12, len(present_species) * 0.5 + 2)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the heatmap
        im = sns.heatmap(key_profiles, cmap="YlOrRd", ax=ax, 
                        annot=True, fmt=".1f", linewidths=0.5, 
                        cbar_kws={'label': '% contribution' if normalize else 'Concentration'})
        
        # Set title and labels
        ax.set_title("Key Species Distribution by Factor", fontsize=15)
        ax.set_xlabel("Factors", fontsize=12)
        ax.set_ylabel("Key Species", fontsize=12)
        
        # Rotate x-tick labels
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        return fig

def ratio_comparison(specie1: str, specie2: str, ratio_value: float, 
                   ratio_file: Optional[str] = None) -> Tuple[bool, List]:
    """
    Compare a species ratio against reference values.
    
    Parameters
    ----------
    specie1 : str
        First species
    specie2 : str
        Second species
    ratio_value : float
        Calculated ratio value to check
    ratio_file : str, optional
        Path to ratio reference file
        
    Returns
    -------
    Tuple[bool, List]
        - Boolean indicating whether ratio is within any reference range
        - List of matching reference data
    """
    try:
        # Load ratio reference data
        if ratio_file and os.path.exists(ratio_file):
            ratio_tab = pd.read_excel(ratio_file)
        else:
            # Try some common locations
            for path in ["Ratio.xlsx", "Data/Ratio.xlsx", "../Data/Ratio.xlsx"]:
                if os.path.exists(path):
                    ratio_tab = pd.read_excel(path)
                    break
            else:
                logger.warning("Ratio reference file not found")
                return False, []
                
        if ratio_tab.empty:
            logger.warning("Ratio table is empty")
            return False, []
        
        # Create flexible match function using regex
        def fuzzy_match(s1, s2):
            # Clean strings for comparison (remove non-alphanumeric and make lowercase)
            s1_clean = re.sub(r'[^a-zA-Z0-9]', '', s1.lower())
            s2_clean = re.sub(r'[^a-zA-Z0-9]', '', s2.lower())
            
            # Try exact match first
            if s1_clean == s2_clean:
                return True
            
            # Try prefix match (for shortened forms, e.g. "SO4" matching "SO42-")
            if s1_clean.startswith(s2_clean) or s2_clean.startswith(s1_clean):
                return True
                
            # Handle special cases
            equivalents = {
                "oc": ["oc*", "organiccarbon"],
                "so4": ["so42-", "sulfate", "sulphate"],
                "no3": ["no3-", "nitrate"],
                "nh4": ["nh4+", "ammonium"],
                "cl": ["cl-", "chloride"],
                "na": ["na+", "sodium"],
                "k": ["k+", "potassium"],
                "ca": ["ca2+", "calcium"],
                "mg": ["mg2+", "magnesium"]
            }
            
            for base, variants in equivalents.items():
                if s1_clean == base or s1_clean in variants:
                    if s2_clean == base or s2_clean in variants:
                        return True
                        
            return False
        
        matches = []
        
        # Check direct ratio (specie1/specie2)
        matching_rows = []
        for idx, row in ratio_tab.iterrows():
            if (fuzzy_match(str(row["Specie 1"]), specie1) and 
                fuzzy_match(str(row["Specie2"]), specie2) and
                row["Min"] <= ratio_value <= row["Max"]):
                matching_rows.append(row)
        
        if matching_rows:
            for row in matching_rows:
                matches.append({
                    "Source": row["Source"],
                    "Min": row["Min"],
                    "Max": row["Max"],
                    "Ratio": f"{specie1}/{specie2}",
                    "Value": ratio_value
                })
            return True, matches
        
        # Check inverse ratio (specie2/specie1)
        matching_rows = []
        inverse_ratio = 1.0 / ratio_value if ratio_value != 0 else float('inf')
        
        for idx, row in ratio_tab.iterrows():
            if (fuzzy_match(str(row["Specie 1"]), specie2) and 
                fuzzy_match(str(row["Specie2"]), specie1) and
                row["Min"] <= inverse_ratio <= row["Max"]):
                matching_rows.append(row)
        
        if matching_rows:
            for row in matching_rows:
                matches.append({
                    "Source": row["Source"],
                    "Min": 1.0 / row["Max"] if row["Max"] != 0 else 0,
                    "Max": 1.0 / row["Min"] if row["Min"] != 0 else float('inf'),
                    "Ratio": f"{specie1}/{specie2}",
                    "Value": ratio_value
                })
            return True, matches
        
        # No matches found
        return False, []
        
    except Exception as e:
        logger.error(f"Error in ratio comparison: {str(e)}")
        return False, []

def get_source_categories_from_ratios(ratio_results: Dict) -> Dict[str, float]:
    """
    Identify likely source categories based on multiple ratio results.
    
    Parameters
    ----------
    ratio_results : dict
        Dictionary of ratio results from compare_ratio
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping source categories to confidence scores (0-1)
    """
    # Example implementation - enhance this with your specific logic
    source_scores = {}
    total_ratios = len(ratio_results)
    
    if total_ratios == 0:
        return {}
        
    for ratio_key, result in ratio_results.items():
        if result.get("success", False):
            for match in result.get("matches", []):
                source = match.get("Source", "Unknown")
                if source not in source_scores:
                    source_scores[source] = 0
                source_scores[source] += 1
    
    # Convert counts to confidence scores
    return {src: count / total_ratios for src, count in source_scores.items()}