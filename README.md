# Shift - Data Drift Detection CLI

Shift is a fast, terminal-based tool designed to detect data drift in machine learning datasets. It evaluates changes in feature distributions between a reference (baseline) dataset and a target dataset, providing an automated way to monitor data integrity over time.

The tool is built using Polars for high-performance data manipulation and Typer for the command-line interface.

## Core Features

* **Numerical Drift Detection:** Implements the Kolmogorov-Smirnov (KS) test to identify statistically significant shifts in continuous variables.
* **Categorical Drift Detection:** Calculates the Population Stability Index (PSI) to measure distribution changes in discrete categories.
* **Flexible Data Slicing:** Compare two distinct files (e.g., train vs. production) or slice a single dataset into reference and target windows using datetime columns.
* **Configurable Thresholds:** Adjust statistical significance levels (alpha) and PSI thresholds directly from the command line.
* **Pipeline Integration:** Export detection results to JSON format for downstream processing, logging, or CI/CD pipelines.

## Installation

To install the project locally, clone the repository and install it in editable mode using pip. Navigate to the root directory where `pyproject.toml` is located and run:

    git clone https://github.com/BadrELAFI/shift.git
    cd shift
    pip install -e .

## Usage

Once installed, the `drift` command will be available globally in your terminal. 

### Help Command

You can use the `--help` flag to see all available arguments, options, and default values.

![Help Command](https://github.com/user-attachments/assets/f37c7bbd-770a-46f6-b132-4dbf62d0dfba)

    Usage: drift [OPTIONS] TARGET                                                        
                                                                                         
    ╭─ Arguments ────────────────────────────────────────────────────────────────────────╮
    │ * target      TEXT  Target dataset path (CSV/Parquet) [required]                │
    ╰────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ──────────────────────────────────────────────────────────────────────────╮
    │ --baseline            -b         TEXT   Baseline dataset path                      │
    │ --start               -s         TEXT   Target window start                        │
    │ --end                 -e         TEXT   Target window end                          │
    │ --date-format         -df        TEXT   Datetime format (optional, Polars will     │
    │                                         infer if omitted)                          │
    │ --col                            TEXT   Datetime column name [default: timestamp]  │
    │ --psi-threshold       -psit      FLOAT  Threshold for psi test (default: 0.2)      │
    │                                         [default: 0.2]                             │
    │ --ks-alpha            -ksa       FLOAT  significance value for ks test (default:   │
    │                                         0.05)                                      │
    │                                         [default: 0.05]                            │
    │ --json                -j         TEXT   path for json output file                  │
    │ --install-completion                    Install completion for the current shell.  │
    │ --show-completion                       Show completion for the current shell, to  │
    │                                         copy it or customize the installation.     │
    │ --help                                  Show this message and exit.                │
    ╰────────────────────────────────────────────────────────────────────────────────────╯

### 1. Comparing Two Separate Files

If you have your reference data and your current data in separate files, you can compare them directly by passing the target file and the `--baseline` flag.

![Using two separate files](https://github.com/user-attachments/assets/d7451361-a142-4d67-abb2-ab60d2bef2b1)

### 2. Time-Window Splitting on a Single File

If all your data is contained in a single file, you can use a datetime column to isolate a specific target window. The tool will parse the timestamps and compare your defined target window against the rest of the dataset.

![Using one file and separating using time](https://github.com/user-attachments/assets/3292e0b2-8b83-4ad9-977a-bbcc4a529bdf)