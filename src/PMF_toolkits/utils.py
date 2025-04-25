import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PMF_toolkits')

def add_season(df: pd.DataFrame, month=True):
    """Add season column to a dataframe based on DatetimeIndex or Date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DatetimeIndex or Date column
    month : bool, default=True
        Whether to add month column
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with season column added
        
    Raises
    ------
    ValueError
        If no DatetimeIndex or Date column is found
    """
    df2 = df.copy()
    
    # Try to get datetime information from the index
    if isinstance(df2.index, pd.DatetimeIndex):
        if month:
            df2["month"] = df2.index.month
        
        month_to_season = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 
            9: "Fall", 10: "Fall", 11: "Fall"
        }
        
        df2["season"] = df2.index.month.map(month_to_season)
        
    # If not DatetimeIndex, try to find a Date column
    elif "Date" in df2.columns:
        # Try to convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df2["Date"]):
            try:
                df2["Date"] = pd.to_datetime(df2["Date"])
            except:
                raise ValueError("Date column could not be converted to datetime")
        
        if month:
            df2["month"] = df2["Date"].dt.month
            
        month_to_season = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 
            9: "Fall", 10: "Fall", 11: "Fall"
        }
        
        df2["season"] = df2["Date"].dt.month.map(month_to_season)
        
    # Special case for multi-site data: If Station column exists but no Date
    elif "Station" in df2.columns and not isinstance(df2.index, pd.DatetimeIndex):
        # Check if index might be dates in string format
        try:
            df2.index = pd.to_datetime(df2.index)
            df2 = add_season(df2, month=month)  # Recursive call now that index is fixed
            return df2
        except:
            raise ValueError("DataFrame has Station column but no proper DatetimeIndex or Date column")
    else:
        raise ValueError("DataFrame must have a DatetimeIndex or Date column")
        
    return df2

def get_sourcesCategories(profiles: List[str]) -> List[str]:
    """
    Map profile names to standardized source categories.
    
    Parameters
    ----------
    profiles : list of str
        Profile names to standardize
        
    Returns
    -------
    list of str
        Standardized category names
    """
    # Define mapping of source profile terms to standard categories
    category_mapping = {
        # Traffic/vehicle related
        'traffic': 'Traffic',
        'vehicle': 'Traffic', 
        'exhaust': 'Traffic',
        'road': 'Traffic',
        'diesel': 'Traffic',
        'gasoline': 'Traffic',
        
        # Biomass burning
        'biomass': 'Biomass',
        'burning': 'Biomass',
        'wood': 'Biomass',
        'combustion': 'Biomass',
        'fire': 'Biomass',
        'smoke': 'Biomass',
        
        # Industry
        'industry': 'Industry',
        'industrial': 'Industry',
        'factory': 'Industry',
        'smelter': 'Industry',
        'steel': 'Industry',
        'power': 'Industry',
        'coal': 'Industry',
        
        # Dust/soil
        'dust': 'Dust',
        'soil': 'Dust',
        'crustal': 'Dust',
        'mineral': 'Dust',
        'road dust': 'Dust',
        'resuspension': 'Dust',
        
        # Secondary sources
        'secondary': 'Secondary',
        'sulfate': 'Secondary',
        'nitrate': 'Secondary',
        'ammonium': 'Secondary',
        'ams': 'Secondary',
        'soa': 'Secondary',
        
        # Marine/sea
        'sea': 'Marine',
        'marine': 'Marine',
        'salt': 'Marine',
        'ocean': 'Marine',
        
        # Oil combustion
        'oil': 'Oil',
        'heating': 'Oil',
        'fuel': 'Oil',
        'combustion': 'Oil',
        
        # Cooking
        'cook': 'Cooking', 
        'food': 'Cooking',
        'restaurant': 'Cooking',
        
        # Aged/Mixed
        'aged': 'Mixed',
        'mixed': 'Mixed',
        'regional': 'Mixed',
        'unidentified': 'Mixed',
        'unknown': 'Mixed',
    }
    
    result = []
    for profile in profiles:
        # Convert to lowercase for case-insensitive matching
        profile_lower = profile.lower()
        
        # Try to find a match
        matched = False
        for key, category in category_mapping.items():
            if key in profile_lower:
                result.append(category)
                matched = True
                break
                
        # If no match found, keep original name
        if not matched:
            result.append(profile)
            
    return result

def get_sourceColor(source: Optional[str] = None) -> Union[str, Dict[str, str]]:
    """
    Get color for a PMF source/factor or return a complete mapping.
    
    Parameters
    ----------
    source : str, optional
        Source name to get color for
        
    Returns
    -------
    str or dict
        If source is provided, returns color code for that source.
        Otherwise, returns complete dictionary mapping sources to colors.
        
    Examples
    --------
    >>> get_sourceColor("Traffic")
    '#000000'
    >>> colors = get_sourceColor()
    >>> colors["Marine"]
    '#33b0f6' # Note: Color might differ based on updated dict
    """
    # Updated color dictionary based on draft_utils.py
    color_dict = {
        # Traffic/vehicle related
        "Traffic": "#000000",
        "Traffic 1": "#000000",
        "Traffic 2": "#102262",
        "Road traffic": "#000000",
        "Primary traffic": "#000000",
        "Traffic_ind": "#000000",
        "Traffic_exhaust": "#000000",
        "Traffic_dir": "#444444",
        "Traffic_non-exhaust": "#444444",
        "Resuspended_dust": "#444444", # Often linked to traffic non-exhaust
        "Oil/Vehicular": "#000000",
        "Road traffic/oil combustion": "#000000",
        
        # Biomass burning - Added "Biomass" as direct entry
        "Biomass": "#92d050",  # Added direct match for Biomass
        "Biomass_burning": "#92d050",
        "Biomass burning": "#92d050",
        "Biomass_burning1": "#92d050",
        "Biomass_burning2": "#92d050",
        
        # Secondary sources - Added "Secondary" as direct entry
        "Secondary": "#0000cc",  # Added direct match for Secondary
        "Sulphate-rich": "#ff2a2a",
        "Sulphate_rich": "#ff2a2a",
        "Sulfate-rich": "#ff2a2a",
        "Sulfate_rich": "#ff2a2a",
        "Sulfate rich": "#ff2a2a",
        "Nitrate-rich": "#217ecb",
        "Nitrate_rich": "#217ecb",
        "Nitrate rich": "#217ecb",
        "Secondary_inorganics": "#0000cc", # Generic secondary inorganic
        "Secondary inorganics": "#0000cc",
        "MSA_rich": "#ff7f2a", # Often related to marine/biogenic SOA
        "MSA-rich": "#ff7f2a",
        "MSA rich" : "#ff7f2a",
        "Secondary_oxidation": "#ff87dc", # Generic SOA
        "Secondary_biogenic_oxidation": "#ff87dc",
        "Secondary oxidation": "#ff87dc",
        "Secondary biogenic oxidation": "#ff87dc",
        "Marine SOA": "#ff7f2a", # Grouping with MSA-rich
        "Biogenic SOA": "#8c564b", # Different color for distinction if needed
        "Anthropogenic SOA": "#8c564b", # Same as Biogenic SOA for now
        
        # Industrial - Added "Industry" as direct entry
        "Industry": "#7030a0",  # Added direct match for Industry
        "Industrial": "#7030a0",
        "Industries": "#7030a0",
        "Indus/veh": "#5c304b", # Mix category
        "Industry/traffic": "#5c304b",
        "Arcellor": "#7030a0", # Specific industry
        "Siderurgie": "#7030a0", # Specific industry (Steelmaking)
        
        # Marine/HFO
        "Marine/HFO": "#a37f15", # Mix category
        "Aged seasalt/HFO": "#8c564b", # Mix category
        "Marine_biogenic": "#fc564b", # Potentially primary marine bio
        "HFO": "#70564b", # Heavy Fuel Oil
        "HFO (stainless)": "#70564b",
        "Oil": "#70564b", # General oil
        "Vanadium rich": "#70564b", # Often HFO tracer
        "Cadmium rich": "#70564b", # Industrial/combustion tracer
        "Marine": "#33b0f6", # General marine aerosol
        "Marin": "#33b0f6",
        "Salt": "#00b0f0", # General salt
        "Seasalt": "#00b0f0",
        "Sea salt" : "#00b0f0",
        "Sea-road salt": "#209ecc", # Mix category
        "Sea/road salt": "#209ecc",
        "Fresh sea salt": "#00b0f0",
        "Fresh seasalt": "#00b0f0",
        "Aged_salt": "#97bdff", # Aged sea salt
        "Aged seasalt": "#97bdff",
        "Aged sea salt": "#97bdff",
        
        # Biogenic
        "Fungal spores": "#ffc000", # Primary biogenic
        "Primary_biogenic": "#ffc000",
        "Primary biogenic": "#ffc000",
        "Biogenique": "#ffc000",
        "Biogenic": "#ffc000",
        
        # Dust/soil
        "Dust": "#dac6a2", # General dust
        "Mineral dust": "#dac6a2",
        "Mineral": "#dac6a2", # Added direct match for Mineral
        "Crustal_dust": "#dac6a2",
        
        # Plant debris
        "Plant debris": "#2aff80", # Primary biogenic
        "Plant_debris": "#2aff80",
        "Débris végétaux": "#2aff80",
        
        # Other
        "Choride": "#80e5ff", # Could be marine or industrial
        "PM other": "#cccccc", # Unspecified
        "Traffic/dust (Mix)": "#333333", # Mix category
        "SOA/sulfate (Mix)": "#6c362b", # Mix category
        "Sulfate rich/HFO": "#8c56b4", # Mix category
        "nan": "#ffffff", # For missing/NaN values
        "Undetermined": "#666666", # Unidentified factors
        "SOA1":"#ff87dc", # If multiple SOA factors
        "SOA2" : "green", # Placeholder, choose a distinct green/other color
        "Secondary oxidation 2": "green" # Placeholder
    }
    
    # Just return the color if source is provided
    if source is not None:
        # Try direct match first (case-sensitive)
        if source in color_dict:
            return color_dict[source]
        
        # Try case-insensitive match
        source_lower = source.lower()
        for key, color in color_dict.items():
            if key.lower() == source_lower:
                return color

        # Improved partial matching - check if the source name contains any of the 
        # dictionary keys or vice versa
        for key, color in color_dict.items():
            if key.lower() in source_lower or source_lower in key.lower():
                logger.info(f"Using color for '{key}' as a partial match for '{source}'")
                return color
                
        # Fallback: Try category matching from get_sourcesCategories
        try:
            # Assuming get_sourcesCategories is defined in the same module
            categories = get_sourcesCategories([source]) 
            category = categories[0]
            if category in color_dict:
                return color_dict[category]
            # Try case-insensitive category match
            category_lower = category.lower()
            for key, color in color_dict.items():
                 if key.lower() == category_lower:
                    return color
        except NameError: # If get_sourcesCategories is not available
             pass # Continue to default

        # Default color if no match found
        logger.warning(f"No specific color found for source '{source}'. Using default gray.")
        return '#808080'  # Default gray
    
    # Return the entire dictionary if no specific source is requested
    return color_dict

def format_xaxis_timeseries(ax: plt.Axes) -> None:
    """
    Improve formatting of time series x-axis to prevent label overlap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(pd.date_range('2020-01-01', '2020-12-31'), range(366))
    >>> format_xaxis_timeseries(ax)
    """
    # Get the date range
    try:
        # Try to extract date range from axis
        dates = mdates.num2date(ax.get_xticks())
        date_range = (dates[0], dates[-1])
        time_span = (date_range[1] - date_range[0]).days
        
        # Select formatter based on time span
        if time_span > 365*2:
            # For multi-year data
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif time_span > 180:
            # For ~year-long data
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator([1, 4, 7, 10]))  # Jan, Apr, Jul, Oct
        elif time_span > 30:
            # For month-scale data
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))  # 1st and 15th
        else:
            # For short time series
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        
        # Set rotation to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout to prevent cutoff
        plt.tight_layout()
        
    except Exception as e:
        logger.warning(f"Unable to format time axis: {str(e)}")

def pretty_specie(text: Union[str, List[str], pd.Index]) -> Union[str, List[str]]:
    """
    Format chemical species names for nice display in plots.
    
    Parameters
    ----------
    text : str or list of str or pd.Index
        Chemical species name(s) to format
        
    Returns
    -------
    str or list of str
        Formatted species name(s)
    
    Examples
    --------
    >>> pretty_specie("SO42-")
    'SO₄²⁻'
    >>> pretty_specie(["Cl-", "Na+"])
    ['Cl⁻', 'Na⁺']
    """
    # Chemical species formatting dictionary
    map_species = {
       "Cl-": "Cl$^-$",
        "Na+": "Na$^+$",
        "K+": "K$^+$",
        "NO3-": "NO$_3^-$",
        "NH4+": "NH$_4^+$",
        "SO42-": "SO$_4^{2-}$",
        "Mg2+": "Mg$^{2+}$",
        "Ca2+": "Ca$^{2+}$",
        "nss-SO42-": "nss-SO$_4^{2-}$",
        "OP_DTT_m3": "OP$^{DTT}_v$",
        "OP_AA_m3": "OP$^{AA}_v$",
        "OP_DTT_µg": "OP$^{DTT}_m$",
        "OP_AA_µg": "OP$^{AA}_m$",
        "PM_µg/m3": "PM mass",
        "PM10": "PM$_{10}$",
        "PM2.5": "PM$_{2.5}$"
    }
    
    # Convert pandas Index to list
    if isinstance(text, pd.Index):
        text = text.tolist()
    
    # Process single string
    if isinstance(text, str):
        return map_species.get(text, text)
    
    # Process list of strings
    elif isinstance(text, (list, tuple)):
        return [map_species.get(str(item), str(item)) for item in text]
    
    # Handle unexpected input type
    else:
        raise TypeError("`text` must be a string, list of strings, or pandas Index")

def to_relative_mass(df: Union[pd.DataFrame, pd.Series], totalVar: str = "PM10") -> Union[pd.DataFrame, pd.Series]:
    """
    Normalize profiles to relative mass with respect to the total variable.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Profile data with species as index
    totalVar : str, default "PM10"
        Name of the total variable to normalize by
        
    Returns
    -------

    pd.DataFrame or pd.Series
        Normalized profile data
        
    Examples
    --------
    >>> profile_df = pd.DataFrame({
    ...     'Factor1': [10, 2, 1],
    ...     'Factor2': [8, 3, 1.5]
    ... }, index=['PM10', 'OC', 'EC'])
    >>> to_relative_mass(profile_df)
       Factor1  Factor2
    OC     0.2    0.375
    EC     0.1    0.187
    """
    if isinstance(df, pd.DataFrame):
        if totalVar not in df.index or np.isclose(df.loc[totalVar], 0).all():
            return pd.DataFrame(dtype=float)
        relative_df = df.div(df.loc[totalVar], axis=1)
        return relative_df.drop(totalVar, errors='ignore')
    elif isinstance(df, pd.Series):
        if totalVar not in df.index or np.isclose(df.loc[totalVar], 0):
            return pd.Series(dtype=float)
        relative_series = df / df.loc[totalVar]
        return relative_series.drop(totalVar, errors='ignore')
    else:
        raise TypeError("`df` must be a pandas DataFrame or Series")
