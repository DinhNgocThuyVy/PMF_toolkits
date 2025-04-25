# PMF_toolkits

Python tools for handling, analyzing and visualizing EPA PMF5.0 outputs from receptor modeling studies.

## Installation

Install from PyPI:

```bash
pip install PMF_toolkits
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/DinhNgocThuyVy/PMF_toolkits.git
```

## Quick Start

```python
from PMF_toolkits import PMF

# Initialize PMF with Excel outputs
pmf = PMF(site="urban_site", reader="xlsx", BDIR="pmf_outputs/")

# Read all data
pmf.read.read_all()

# Plot factor profiles
pmf.visualization.plot_factor_profiles()
```

## Citation

If you use PMF_toolkits in your research, please cite:

```
Dinh, N.T.V. (2023). PMF_toolkits: Python tools for analysis of Positive Matrix Factorization results.
GitHub repository: https://github.com/DinhNgocThuyVy/PMF_toolkits
```