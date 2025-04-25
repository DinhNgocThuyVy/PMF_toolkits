from setuptools import setup, find_packages

setup(
    name="PMF_toolkits",
    version="0.2.1",  # Standardized version across package
    description="EPA PMF5 output analysis tools in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dinh Ngoc Thuy Vy",
    author_email="dinhvy2101@gmail.com",
    url="https://github.com/DinhNgocThuyVy/PMF_toolkits",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "PMF_toolkits.data": ["*.csv", "*.xlsx"],
    },
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "openpyxl>=3.0.0"  # For Excel file support
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
