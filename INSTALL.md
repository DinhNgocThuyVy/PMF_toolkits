
# Installing PMF_toolkits for Development

## Installation Instructions

### Option 1: Standard Installation  (Recommended)

Install the package directly:

```bash
pip install git+https://github.com/DinhNgocThuyVy/PMF_toolkits
```

### Option 2: Development Mode

This method allows you to modify the package code and have changes reflect immediately without reinstalling:

1. Open a command prompt/terminal and navigate to the package directory:

   ```bash
   cd "PMF_toolkits"
   ```

2. Install the require packages in development mode:

   ```bash
   python run_tests.py
   ```

3. To update the PMF package after some corrections

- Use vscode to detect the temp_env automatically

   ```bash
   pip install .
   ```

## Verifying Installation

To verify that PMF_toolkits is installed correctly, run the following in a Jupyter notebook:

```python
import PMF_toolkits
print(PMF_toolkits.__version__)  # Should display the version number

# Create a basic PMF instance
from PMF_toolkits import PMF
pmf = PMF(site="test", reader=None, savedir="./")
print("Installation successful!")
```