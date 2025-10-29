# Machine Learning and Security: Protecting Systems with Data and Algorithms
by Clarence Chio and‎ David Freeman (Authors), published by O'Reilly Media

[Visit Website](https://mlsec.net/) | [Purchase on Amazon](https://amzn.to/2FmVDYQ) | [Read on O'Reilly Safari](http://shop.oreilly.com/product/0636920065555.do)

This repository contains accompanying resources, exercises, datasets, and sample code for the Machine Learning and Security book released in Feb 2018.

<img src="mlsec-cover-oreilly.jpg" width="50%" height="50%">

## Dependencies

### System packages
* Python 3.14+ (Updated for modern compatibility)
* Spark 2.2.0 (Pre-built for Apache Hadoop 2.7 and later)

### Python packages (Updated to Latest Versions)
* jupyter 1.1.1
* pandas 2.3.3
* matplotlib 3.10.7
* seaborn 0.13.2
* numpy 2.3.4
* scikit-learn 1.7.2
* nltk 3.9.2
* datasketch 1.6.5
* tensorflow (latest)
* keras 3.12.0
* ~~pyflux 0.4.15~~ (Replaced with custom ARIMA implementation using scikit-learn)
* imbalanced-learn 0.14.0
* ~~spark-sklearn 0.2.3~~ (Available but may have compatibility issues)
* lime (latest)
* scipy 1.16.3 (Added for enhanced mathematical operations)

### Compatibility Notes
⚠️ **Important Updates for Modern Python (3.14+):**

1. **PyFlux Replacement**: The original `pyflux` package is incompatible with Python 3.14+. The ARIMA forecasting examples (Chapter 3) have been updated to use a custom implementation with scikit-learn's LinearRegression and time series feature engineering.

2. **Pandas API Changes**: Removed deprecated `infer_datetime_format` parameter from CSV reading operations.

3. **Package Versions**: All packages have been updated to their latest stable versions for better performance and security.

4. **Installation**: Use `pip install -r requirements.txt` to install compatible versions, or install individual packages using the latest versions listed above.

## Updated Examples and Compatibility Fixes

### Chapter 3: ARIMA Forecasting (`arima-forecasting.ipynb`)
✅ **Fully Updated for Python 3.14+**
- **PyFlux Replacement**: Replaced deprecated PyFlux library with custom ARIMA implementation
- **New Features**: 
  - Time series feature engineering using lagged values and moving averages
  - LinearRegression-based forecasting model
  - Custom visualization functions for forecast plotting
  - Performance metrics (MSE) calculation
- **Maintained Functionality**: All original educational objectives preserved with modern, compatible code

### General Notebook Updates
- **Pandas Compatibility**: Fixed deprecated `infer_datetime_format` warnings
- **Import Statements**: Updated to use modern package versions
- **Error Handling**: Enhanced robustness for newer Python versions

## Quick Start
```bash
# Clone the repository
git clone <repository-url>

# Navigate to project directory
cd ml-sec

# Install dependencies (modern versions)
pip install jupyter pandas matplotlib seaborn numpy scikit-learn nltk datasketch tensorflow keras imbalanced-learn lime scipy

# Start Jupyter
jupyter notebook
```
