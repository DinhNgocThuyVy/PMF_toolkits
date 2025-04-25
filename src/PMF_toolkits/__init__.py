"""
PMF_toolkits: Positive Matrix Factorization tools for environmental data analysis.

This package provides tools for loading, analyzing, and visualizing PMF 
results from EPA PMF5.0, including preprocessing functions for data preparation.

Theory
------
PMF decomposes a data matrix X into two matrices:
- G (source contributions)
- F (source profiles/factor loadings)

The basic model is:
    X = GF + E

where:
    X: n × m matrix of measurements (n samples, m species)
    G: n × p matrix of source contributions
    F: p × m matrix of source profiles
    E: n × m matrix of residuals
    p: number of factors/sources

The PMF algorithm minimizes Q:
    Q = Σᵢⱼ(eᵢⱼ/sᵢⱼ)²

where:
    eᵢⱼ: residual for species j in sample i
    sᵢⱼ: uncertainty for species j in sample i

Key Features
-----------
- Data preprocessing:
    - Below detection limit handling
    - Missing value imputation
    - Uncertainty calculation methods
    
- Factor analysis:
    - Profile analysis
    - Source contribution assessment
    - Factor interpretation aids
    
- Uncertainty analysis:
    - Bootstrap (BS)
    - Displacement (DISP)
    - BS-DISP combined
    
- Visualization:
    - Profile plots
    - Time series
    - Seasonal patterns
    - Diagnostic plots

Example
-------
>>> from PMF_toolkits import PMF
>>> # Initialize PMF with xlsx output
>>> pmf = PMF(site="urban_site", reader="xlsx", BDIR="pmf_outputs/")
>>> # Read all available data
>>> pmf.read.read_all()
>>> # Plot factor profiles
>>> pmf.visualization.plot_factor_profiles()

References
----------
1. Paatero, P., Tapper, U., 1994. Positive matrix factorization: A non-negative factor
   model with optimal utilization of error estimates of data values. Environmetrics 5, 111–126.
2. Norris, G., Duvall, R., Brown, S., Bai, S., 2014. EPA Positive Matrix Factorization
   (PMF) 5.0 Fundamentals and User Guide. EPA/600/R-14/108.

See Also
--------
EPA PMF 5.0 : https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses

"""

from .core import PMF
from .visualization import PMFVisualization
# Import the consolidated function from analysis
from .analysis import PMFAnalysis, compute_similarity_metrics
from .preprocessing import PMFPreprocessor, load_concentration_data, summarize_dataset
from .readers import XlsxReader, MultisitesReader
from .validation import OutputValidator, ratio_comparison
from .utils import (
    add_season,
    get_sourceColor,
    format_xaxis_timeseries,
    pretty_specie,
    to_relative_mass,
    get_sourcesCategories
)

__version__ = "0.2.1"  # Standardized version number across package

__all__ = [
    # Core
    'PMF',
    # Modules
    'PMFAnalysis',
    'PMFVisualization',
    'PMFPreprocessor',
    'OutputValidator',
    # Readers
    'XlsxReader',
    'MultisitesReader',
    # Analysis Functions
    'compute_similarity_metrics',
    # Preprocessing Functions
    'load_concentration_data',
    'summarize_dataset',
    # Validation Functions
    'ratio_comparison',
    # Utils
    'add_season',
    'get_sourceColor',
    'format_xaxis_timeseries',
    'pretty_specie',
    'to_relative_mass',
    'get_sourcesCategories',
]

# Configure logging
import logging
logging.getLogger('PMF_toolkits').addHandler(logging.NullHandler())
