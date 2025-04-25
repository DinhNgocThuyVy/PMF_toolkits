# Readers Module Documentation

## Overview

The Readers module provides classes for loading PMF output data from different sources (Excel files, SQL databases) and handling different formats (single-site, multi-site). It includes robust error handling and data validation.

## Class Hierarchy

```
BaseReader (ABC)
├── XlsxReader
│   └── MultisitesReader
└── SqlReader
```

## BaseReader Class

Abstract base class defining the interface for all readers.

### Methods

#### read_base_profiles

```python
@abstractmethod
def read_base_profiles(self) -> None:
    """
    Read base run factor profiles.

    Sets the following attributes:
    - pmf.dfprofiles_b: Factor profiles
    - pmf.profiles: Factor names
    - pmf.species: Species names
    """
```

#### read_constrained_profiles

```python
@abstractmethod
def read_constrained_profiles(self) -> None:
    """
    Read constrained run factor profiles.

    Sets:
    - pmf.dfprofiles_c: Constrained factor profiles
    """
```

#### read_base_bootstrap

```python
@abstractmethod
def read_base_bootstrap(self) -> None:
    """
    Read base run bootstrap results.

    Sets:
    - pmf.dfBS_profile_b: Bootstrap profiles
    - pmf.dfbootstrap_mapping_b: Bootstrap mapping
    """
```

## XlsxReader Class

Reads PMF outputs from Excel files in EPA PMF 5.0 format.

### Initialization
```python
def __init__(self,
             BDIR: str,
             site: str,
             pmf: Any,
             multisites: bool = False):
    """
    Initialize Excel reader.

    Parameters
    ----------
    BDIR : str
        Base directory containing PMF output files
    site : str
        Site name (used in filenames)
    pmf : PMF
        Parent PMF object
    multisites : bool, default=False
        Whether files contain multiple sites
    """
```

### Key Methods

#### read_all
```python
def read_all(self) -> None:
    """
    Read all available PMF output files.
    
    Reads in order:
    1. Base profiles
    2. Base contributions
    3. Base bootstrap
    4. Base uncertainties
    5. Constrained profiles
    6. Constrained contributions
    7. Constrained bootstrap
    8. Constrained uncertainties
    """
```

#### _split_df_by_nan

```python
def _split_df_by_nan(self,
                     df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split dataframe at NaN rows.

    Used for parsing bootstrap results where factors
    are separated by empty rows.
    """
```

## SqlReader Class

Reads PMF outputs from SQL databases.

### Initialization

```python
def __init__(self,
             site: str,
             pmf: Any,
             SQL_connection: Any,
             SQL_table_names: Optional[Dict[str, str]] = None,
             SQL_program: Optional[str] = None):
    """
    Initialize SQL reader.

    Parameters
    ----------
    site : str
        Site name
    pmf : PMF
        Parent PMF object
    SQL_connection : sqlalchemy.engine.Engine
        Database connection
    SQL_table_names : dict, optional
        Custom table names mapping
    SQL_program : str, optional
        Program identifier in database
    """
```

### Key Methods

#### _read_table

```python
def _read_table(self,
                table: str,
                read_sql_kws: Dict = {}) -> pd.DataFrame:
    """
    Read a table from the database.

    Parameters
    ----------
    table : str
        Table name key from SQL_table_names
    read_sql_kws : dict
        Additional keywords for pd.read_sql
    """
```

## MultisitesReader Class

Extends XlsxReader for multi-site PMF outputs.

### Features

- Auto-detection of multi-site formats
- Parallel loading of multiple sites
- Cross-site validation
- Combined uncertainty estimation

### Methods

#### _detect_multisite_format

```python
def _detect_multisite_format(self,
                           df: pd.DataFrame) -> bool:
    """
    Detect if file contains multi-site data.
    """
```

## Examples

### Using XlsxReader

```python
from PMF_toolkits import PMF

# Initialize PMF with Excel reader
pmf = PMF(site="urban_site",
          reader="xlsx",
          BDIR="pmf_outputs/")

# Read all available data
pmf.read.read_all()
```

### Using SqlReader
```python
from sqlalchemy import create_engine
from PMF_toolkits import PMF

# Create database connection
engine = create_engine("postgresql://user:pass@localhost/pmf_db")

# Initialize PMF with SQL reader
pmf = PMF(site="urban_site",
          reader="sql",
          SQL_connection=engine)

# Read all available data
pmf.read.read_all()
```

### Multi-site Analysis

```python
# Initialize with multi-site reader
pmf = PMF(site="all_sites",
          reader="xlsx",
          BDIR="pmf_outputs/",
          multisites=True)

# Read and combine data
pmf.read.read_all()
```

## File Formats

### Excel Output Files

```
site_Base.xlsx              # Base run results
site_Constrained.xlsx       # Constrained run results
site_boot.xlsx             # Bootstrap results
site_Gcon_profile_boot.xlsx # Constrained bootstrap
```

### SQL Tables
```sql
PMF_dfprofiles_b      -- Base profiles
PMF_dfprofiles_c      -- Constrained profiles
PMF_dfcontrib_b       -- Base contributions
PMF_dfcontrib_c       -- Constrained contributions
PMF_dfBS_profile_b    -- Base bootstrap
PMF_dfBS_profile_c    -- Constrained bootstrap
```

## Best Practices

1. **Error Handling**
   - Check file existence
   - Validate data format
   - Handle missing values
   - Report loading errors

2. **Data Validation**
   - Verify column names
   - Check data types
   - Validate value ranges
   - Ensure consistency

3. **Performance**
   - Use appropriate reader
   - Load data efficiently
   - Handle large files
   - Cache when needed

::: PMF_toolkits.readers.BaseReader
    handler: python
    selection:
      members:
        - read_base_profiles
        - read_constrained_profiles
        - read_base_bootstrap
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2

::: PMF_toolkits.readers.XlsxReader
    handler: python
    selection:
      members:
        - __init__
        - read_all
        - _split_df_by_nan
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2

::: PMF_toolkits.readers.MultisitesReader
    handler: python
    selection:
      members:
        - __init__
        - _detect_multisite_format
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 2
