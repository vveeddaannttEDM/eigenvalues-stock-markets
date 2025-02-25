# Eigenvalue Analysis of Stock Market Correlation Matrix

## Overview
This repository contains a Python script that performs eigenvalue analysis on a stock market correlation matrix using **Random Matrix Theory (RMT)**. The script generates synthetic stock return data, computes the correlation matrix, extracts eigenvalues, and analyzes their statistical properties.

## Features
- **Generates synthetic stock return data** (100 days, 10 stocks)
- **Computes the correlation matrix** of stock returns
- **Extracts eigenvalues and eigenvectors** using **eigh** from SciPy
- **Plots the eigenvalue distribution**
- **Calculates cumulative explained variance** and saves results
- **Computes principal components** and visualizes the first two

## Files
- `scripts/eigenvalue_analysis.py`: Main script for eigenvalue analysis
- `data/stock_data.csv`: Example dataset (generated if not provided)
- `eigenvalues.csv`: Saved eigenvalues of the correlation matrix
- `cumulative_variance.csv`: Cumulative explained variance values
- `principal_components.csv`: Computed principal components

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install numpy pandas matplotlib scipy
```

## How to Run
Clone this repository and run the script:
```sh
git clone <repo_url>
cd eigenvalues-stock-markets
python scripts/eigenvalue_analysis.py
```

## Outputs
After running the script, you will get:
- **Eigenvalue distribution plot**
- **Cumulative explained variance plot**
- **Principal component scatter plot**
- **CSV files with numerical results**

## License
This project is licensed under the MIT License.
