import polars as pl
from utils.descriptive_stats import DescriptiveStats
from utils.time_parser import TimeParser
from detector.numeric_drift_detector import NumericDriftDetector
from detector.categorical_drift_detector import CategoricalDriftDetector
from pathlib import Path
from utils.helper import (
    NpEncoder,
    get_numerical_drift_elligible_column,
    get_categorical_drift_elligble_column,
)
import json


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


def print_summary(summary: dict) -> str:
    result = (
        f"=========================================\n"
        f"          DRIFT SUMMARY REPORT           \n"
        f"=========================================\n"
        f"Total Features Analyzed: {summary['total_analyzed']}\n"
        f"Features with Drift: {summary['drift_detected']}\n"
        f"Features without Drift: {summary['no_drift']}\n"
        f"Features with Errors: {summary['errors']}\n"
        f"=========================================\n"
    )

    return result


def export_json(json_export: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(json_export, f, cls=NpEncoder)

    print(f"json file generated at {filename}")


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
    generate_json = bool(parameters["json_output"])
    parsedcol = f"parsed_{parameters['date_column']}"

    target_ds = dataset.filter(pl.col(parsedcol).is_between(date_start, date_end))
    baseline_ds = dataset.filter(~pl.col(parsedcol).is_between(date_start, date_end))

    summary = {"total_analyzed": 0, "drift_detected": 0, "no_drift": 0, "errors": 0}
    json_export = {"summary": summary, "feature_metrics": []}

    ignored_columns = [parsedcol, parameters["date_column"]]
    elligible_numeric_features = get_numerical_drift_elligible_column(
        baseline_ds, ignored_columns
    )
    elligible_categoric_features = get_categorical_drift_elligble_column(
        baseline_ds, excluded_cols=[]
    )

    print(
        f"found {len(elligible_numeric_features)} eligible numeric features for drift detection."
        f"found {len(elligible_categoric_features)} eligible categorical features for drift detection."
    )

    numerical_detector = NumericDriftDetector(
        alpha=parameters["ks_alpha"], psi_threshold=parameters["psi_threshold"]
    )
    categorical_detector = CategoricalDriftDetector(
        psi_threshold=parameters["psi_threshold"]
    )

    for col in elligible_numeric_features:
        summary["total_analyzed"] += 1

        column_report_numeric = numerical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )

        json_export["feature_metrics"].append(column_report_numeric)

        if column_report_numeric["status"] == "error":
            summary["errors"] += 1
        elif column_report_numeric["overall_drift_detected"]:
            summary["drift_detected"] += 1
        else:
            summary["no_drift"] += 1

        print(
            numerical_detector.format_cli_summary(column_report=column_report_numeric)
        )

    for col in elligible_categoric_features:
        summary["total_analyzed"] += 1

        column_report_categoric = categorical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )

        json_export["feature_metrics"].append(column_report_categoric)

        if column_report_categoric["status"] == "error":
            summary["errors"] += 1
        elif column_report_categoric["overall_drift_detected"]:
            summary["drift_detected"] += 1
        else:
            summary["no_drift"] += 1

        print(
            categorical_detector.format_cli_summary(
                column_report=column_report_categoric
            )
        )

    if generate_json:
        export_json(json_export, parameters["json_output"])

    print(print_summary(summary))


def run_separate_ds(parameters: dict):
    target_ds = load_df(parameters["target"])
    baseline_ds = load_df(parameters["baseline"])
    generate_json = bool(parameters["json_output"])

    # no need to parse time since they are already seperated ?

    summary = {"total_analyzed": 0, "drift_detected": 0, "no_drift": 0, "errors": 0}
    json_export = {"summary": summary, "feature_metrics": []}

    ignored_columns = [parameters["date_column"]]
    elligible_numeric_features = get_numerical_drift_elligible_column(
        baseline_ds, ignored_columns
    )
    elligible_categoric_features = get_categorical_drift_elligble_column(
        baseline_ds, excluded_cols=[]
    )
    print(
        f"found {len(elligible_numeric_features)} eligible numeric features for drift detection."
        f"found {len(elligible_categoric_features)} eligible categorical features for drift detection."
    )

    numerical_detector = NumericDriftDetector(
        alpha=parameters["ks_alpha"], psi_threshold=parameters["psi_threshold"]
    )
    categorical_detector = CategoricalDriftDetector(
        psi_threshold=parameters["psi_threshold"]
    )

    for col in elligible_numeric_features:
        summary["total_analyzed"] += 1
        column_report_numeric = numerical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )

        json_export["feature_metrics"].append(column_report_numeric)

        if column_report_numeric["status"] == "error":
            summary["errors"] += 1
        elif column_report_numeric["overall_drift_detected"]:
            summary["drift_detected"] += 1
        else:
            summary["no_drift"] += 1

        print(
            numerical_detector.format_cli_summary(column_report=column_report_numeric)
        )

    for col in elligible_categoric_features:
        column_report_categoric = categorical_detector.evaluate_column(
            baseline_ds, target_ds, col, col
        )

        json_export["feature_metrics"].append(column_report_categoric)

        if column_report_categoric["status"] == "error":
            summary["errors"] += 1
        elif column_report_categoric["overall_drift_detected"]:
            summary["drift_detected"] += 1
        else:
            summary["no_drift"] += 1

        print(
            categorical_detector.format_cli_summary(
                column_report=column_report_categoric
            )
        )

    if generate_json:
        export_json(json_export, parameters["json_output"])

    print(print_summary(summary))
