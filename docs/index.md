# PMF_toolkits Documentation

## Overview

PMF_toolkits is a Python package designed for working with Positive Matrix Factorization (PMF) analysis results from the EPA PMF 5.0 software. It provides tools for loading, analyzing, and visualizing PMF outputs, enabling researchers to effectively interpret source apportionment results in environmental studies.

## What is PMF?

Positive Matrix Factorization (PMF) is a multivariate factor analysis technique that decomposes a matrix of environmental sample data into two matrices: factor contributions and factor profiles. It is widely used for source apportionment studies in air quality research to identify and quantify the sources of atmospheric pollutants.

### Mathematical Foundation

PMF decomposes a data matrix X into two matrices:
- G (source contributions)
- F (source profiles/factor loadings)

The basic model is:

$$X = GF + E$$

where:
- X: n × m matrix of measurements (n samples, m species)
- G: n × p matrix of source contributions
- F: p × m matrix of source profiles
- E: n × m matrix of residuals
- p: number of factors/sources

PMF uses a weighted least-squares approach, minimizing the objective function Q:

$$Q = \sum_{i=1}^{n}\sum_{j=1}^{m}\left(\frac{e_{ij}}{s_{ij}}\right)^2$$

where:
- e<sub>ij</sub> is the residual for species j in sample i
- s<sub>ij</sub> is the uncertainty for species j in sample i

### Advantages of PMF

- Non-negative constraint on factor profiles and contributions
- Weighted least-squares approach that can handle below-detection-limit values
- Robust handling of outliers and missing data
- Estimation of uncertainties through bootstrap, displacement, and BS-DISP methods

## Package Features

PMF_toolkits offers a comprehensive suite of tools for working with PMF results:

### Data Loading and Preprocessing

- Read EPA PMF 5.0 output files (Base, Constrained, Bootstrap, DISP, BS-DISP)
- Support for both single-site and multi-site data
- Handle below-detection-limit values and missing data
- Calculate uncertainties using various methods (percentage, detection limit-based, Polissar)

### Analysis

- Calculate explained variation by each factor
- Compute similarity metrics between profiles (SID, PD, COD)
- Analyze temporal patterns and seasonal contributions
- Detect potentially mixed factors
- Compare results from different runs

### Visualization

- Factor profile plots with uncertainties
- Time series plots of source contributions
- Seasonal contribution patterns
- Stacked profiles and contributions
- Comprehensive diagnostic plots

### Uncertainty Analysis

- Bootstrap analysis
- Displacement (DISP) analysis
- BS-DISP combined analysis
- Visualization of uncertainties

## Applications

PMF_toolkits is especially useful for:

- Air quality source apportionment studies
- Receptor modeling of particulate matter
- Environmental forensics
- Trend analysis of pollution sources
- Comparison of PMF results across different sites or studies

## Getting Started

To begin using PMF_toolkits, see the [Quickstart Guide](quickstart.md) for installation instructions and basic usage examples.

## References

1. Paatero, P., Tapper, U., 1994. Positive matrix factorization: A non-negative factor model with optimal utilization of error estimates of data values. Environmetrics 5, 111–126.

2. Norris, G., Duvall, R., Brown, S., Bai, S., 2014. EPA Positive Matrix Factorization (PMF) 5.0 Fundamentals and User Guide. EPA/600/R-14/108.

3. Paatero, P., Eberly, S., Brown, S.G., Norris, G.A., 2014. Methods for estimating uncertainty in factor analytic solutions. Atmospheric Measurement Techniques 7, 781–797.

4. Hopke, P.K., 2016. Review of receptor modeling methods for source apportionment. Journal of the Air & Waste Management Association 66, 237–259.
