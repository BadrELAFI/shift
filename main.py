import polars as pl
from utils.descriptive_stats import DescriptiveStats
from utils.time_parser import TimeParser
from detector.numeric_drift_detector import NumericDriftDetector
from cli.config_loader import ConfigLoader
from pathlib import Path
from cli.interface import detect


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

    return {}


if __name__ == "__main__":
    yamlconf = ConfigLoader()
    conf = yamlconf.load_config()

    # 2. Load DataFrames
    df_base = load_df(conf["datasets"]["baseline_path"])
    df_target = load_df(conf["datasets"]["target_path"])

    detector = NumericDriftDetector()

    print("\n--- Running Numeric Drift Tests ---")
    for test_config in conf["tests"]["numeric"]:
        col = test_config["column"]
        psi_limit = test_config.get("psi_threshold", 0.2)

        # Calculate PSI
        psi_results = detector.calculate_psi(
            df_base, df_target, column_baseline=col, column_target=col, nbins=10
        )

        ks_test = detector.kstest(
            df_base, df_target, column_baseline=col, column_target=col
        )

        # Output Results
        status = "DRIFT" if psi_results["drift_detected"] else "STABLE"
        print(f"[{status}] Column: {col:15} | PSI: {psi_results['psi_value']:.4f}")

        print(ks_test)
