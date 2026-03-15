# Shift - Data Drift Detection CLI

Shift is a command-line tool built to detect data drift in datasets. I designed this to quickly check if the distribution of incoming data has shifted compared to a training baseline, which is a common issue that degrades machine learning model performance over time.

Instead of writing custom Python scripts every time I need to check a new dataset, this tool provides a fast, reusable CLI interface. It is built on top of **Polars** for fast data processing and **Typer** for the command-line interface.

## Core Features

- **Numerical Data:** Uses the Kolmogorov-Smirnov (KS) test to detect distribution changes in continuous variables.
- **Categorical Data:** Calculates the Population Stability Index (PSI) to measure shifts in discrete categories.
- **Multiple Formats:** Supports both CSV and Parquet files natively.
- **Flexible Data Splitting:** Compare two separate datasets (e.g., train vs. production) or split a single dataset using a datetime column and specific time windows.
- **Pipeline Ready:** Can export the drift analysis results directly to a JSON file.

## Installation

To install the project locally, clone the repository, navigate to the root directory where `pyproject.toml` is located, and install it in editable mode using pip:

    git clone https://github.com/yourusername/shift.git
    cd shift
    pip install -e .

## Usage

Once installed, the `drift` command will be available globally in your terminal. 

### 1. Comparing Two Separate Files
If you have your reference data (baseline) and your current data (target) in separate files, you can compare them directly. This is useful for comparing a new batch of data against your original training set.

    drift detect new_data.csv --baseline training_data.csv

![Using two separate files](https://github.com/user-attachments/assets/d7451361-a142-4d67-abb2-ab60d2bef2b1)

### 2. Time-Window Splitting on a Single File
If all your data is appended to a single file, you can define a time window to slice the target data. The tool will parse the datetime column and compare the specified window against the rest of the dataset.

    drift detect all_data.csv --start "2025-01-01" --end "2025-03-01" --col "timestamp"

![Using one file and separating using time](https://github.com/user-attachments/assets/3292e0b2-8b83-4ad9-977a-bbcc4a529bdf)

## Configuration and Tuning

You can override the default statistical thresholds directly from the command line depending on how strict you want the drift detection to be:

- `--psi-threshold`: Change the threshold for categorical drift (Default is 0.2).
- `--ks-alpha`: Change the significance level (p-value threshold) for the KS test (Default is 0.05).
- `--json`: Provide a file path to export the results as a JSON file instead of just printing to the console.

Example with custom parameters and JSON export:

    drift detect production.parquet --baseline train.parquet --ks-alpha 0.01 --json report.json

To see all available arguments and options, run:

    drift detect --help# Shift - Data Drift Detector

This is a command line tool I made to detect data drift in datasets. It checks for changes in both numerical and categorical features between your reference data and current data.

## Installation

To install the project locally, you can use pip. Navigate to the root directory where `pyproject.toml` is located and run:

```bash
pip install -e .
```

## How to use

Once installed, you can run the tool from your terminal. Here are a couple of ways you can use it:

### 1. Using two separate files
If you have your reference data and your current data in separate CSV files, you can compare them directly.

![Using two separate files](https://github.com/user-attachments/assets/d7451361-a142-4d67-abb2-ab60d2bef2b1)

### 2. Using one file split by time
If all your data is in a single file, you can use a time column to split the dataset into reference and current data before running the drift detection.

![Using one file and separating using time](https://github.com/user-attachments/assets/3292e0b2-8b83-4ad9-977a-bbcc4a529bdf)

## Features
- Detects drift in numeric columns.
- Detects drift in categorical columns.
- Configurable settings.
