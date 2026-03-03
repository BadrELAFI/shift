import polars as pl
from utils.descriptive_stats import DescriptiveStats
from utils.time_parser import TimeParser
from detector.numeric_drift_detector import NumericDriftDetector
from pathlib import Path
from utils.helper import get_numerical_drift_elligible_numeric_column


def load_df(path: str) -> pl.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix == ".csv":
        return pl.read_csv(path)
    elif suffix == ".ipc" or suffix == ".feather":
        return pl.read_ipc(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def run(parameters: dict):
    if not parameters:
        raise ValueError(
            "It seems to be a problem with the parameters of the tool. Please try again"
        )

    if parameters["baseline"] is None:
        run_joined_ds(parameters)
    else:
        run_separate_ds(parameters)


def run_joined_ds(parameters: dict):
    dataset = load_df(parameters["target"])
    timeparser = TimeParser(user_input_format=parameters["date_format"])
    dataset = timeparser.parse_time_new(dataset, parameters["date_column"])
    date_start = timeparser.parse_time_start_end(parameters["start"])
    date_end = timeparser.parse_time_start_end(parameters["end"])

    parsedcol = f"parsed_{parameters['date_column']}"

    target_ds = dataset.filter(pl.col(parsedcol).is_between(date_start, date_end))
    baseline_ds = dataset.filter(~pl.col(parsedcol).is_between(date_start, date_end))

    ignored_columns = [parsedcol, parameters["date_column"]]
    elligible_features = get_numerical_drift_elligible_numeric_column(
        baseline_ds, ignored_columns
    )
    print(
        f"found {len(elligible_features)} eligible numeric features for drift detection."
    )

    numerical_detector = NumericDriftDetector(
        alpha=parameters["ks_alpha"], psi_threshold=parameters["psi_threshold"]
    )

    for col in elligible_features:
        column_report = numerical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )
        print(numerical_detector.format_cli_summary(column_report=column_report))


def run_separate_ds(parameters: dict):
    target_ds = load_df(parameters["target"])
    baseline_ds = load_df(parameters["baseline"])

    # no need to parse time since they are already seperated ?

    ignored_columns = [parameters["date_column"]]
    elligible_features = get_numerical_drift_elligible_numeric_column(
        baseline_ds, ignored_columns
    )
    print(
        f"found {len(elligible_features)} eligible numeric features for drift detection."
    )

    numerical_detector = NumericDriftDetector(
        alpha=parameters["ks_alpha"], psi_threshold=parameters["psi_threshold"]
    )
    for col in elligible_features:
        print(DescriptiveStats.get_stats(target_ds[col]))
        print(DescriptiveStats.get_stats(baseline_ds[col]))

    for col in elligible_features:
        column_report = numerical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )
        print(numerical_detector.format_cli_summary(column_report=column_report))
