import polars as pl
from scipy.stats import chi2_contingency, contingency
import numpy as np


class CategoricalDriftDetector:
    def __init__(self, psi_threshold: float = 0.2) -> None:
        self.psi_threshold = psi_threshold

    def _chi_square(
        self,
        df_baseline: pl.DataFrame,
        df_target: pl.DataFrame,
        column_baseline: str,
        column_target: str,
    ):
        baseline_counts = (
            df_baseline.select(column_baseline).to_series().value_counts().sort("count")
        )

        target_counts = (
            df_target.select(column_target).to_series().value_counts().sort("count")
        )

        baseline_counts = baseline_counts.rename(
            {column_baseline: "category", "count": "baseline_freq"}
        )
        target_counts = target_counts.rename(
            {column_target: "category", "count": "target_freq"}
        )

        combined = baseline_counts.join(
            target_counts, on="category", how="full"
        ).fill_null(0)

        contingency_table = combined.select(["baseline_freq", "target_freq"]).to_numpy()

        chi_2, p_value, dof, expected = chi2_contingency(contingency_table)

        baseline_total = combined["baseline_freq"].sum()
        target_total = combined["target_freq"].sum()

        distribution_comparaison = (
            combined.with_columns(
                [
                    (pl.col("baseline_freq").cast(pl.Float64) / baseline_total)
                    .round(4)
                    .alias("'baseline_proportion"),
                    (pl.col("target_freq").cast(pl.Float64) / target_total)
                    .round(4)
                    .alias("target_proportion"),
                ]
            )
            .with_columns(
                [
                    (pl.col("target_proportion") - pl.col("baseline_proportion"))
                    .round(4)
                    .alias("delta_proportion")
                ]
            )
            .sort("delta_proportion", descending=True)
        )

        return {
            "chi2_statistic": chi_2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "drift_detected": p_value < 0.05,
            "distribution_comparison": distribution_comparaison,
        }

    def evaluate_column(
        self,
        df_baseline: pl.DataFrame,
        df_target: pl.DataFrame,
        column_baseline: str,
        column_target: str,
    ) -> dict:
        """Runs all drift tests on a single column and returns a structured report."""
        try:
            chi2_results = self._chi_square(
                df_baseline, df_target, column_baseline, column_target
            )

            """    
            overall_drift = (
                chi2_results["drift_detected"] or psi_results["drift_detected"]
            )
            """

            return {
                "feature_name": column_target,
                "status": "success",
                # "overall_drift_detected": overall_drift,
                "chi2_test": chi2_results,
            }
        except Exception as e:
            return {
                "feature_name": column_target,
                "status": "error",
                "error_message": str(e),
                "overall_drift_detected": False,
            }

    def format_cli_summary(self, column_report: dict) -> str:
        feature = column_report["feature_name"]

        if column_report["status"] == "error":
            return (
                f"[ERROR] Feature: {feature} | Reason: {column_report['error_message']}\n"
                + "-" * 40
            )

        drift_icon = (
            "DRIFT DETECTED" if column_report["overall_drift_detected"] else "No Drift"
        )
        chi2 = column_report["chi2_test"]
        psi = column_report["psi_test"]

        report_str = (
            f"Feature: **{feature}** | {drift_icon}\n"
            f"  ├─ Chi Squared Test \n"
            f"  │  ├─ chi2-Stat : {chi2['chi2_statistic']:.4f}\n"
            f"  │  ├─ P-Value : {chi2['p_value']:.4e}\n"
            f"  │  └─ Drift   : {'Yes' if chi2['overall_drift_detected'] else 'No'}\n"
            f"  └─ Population Stability Index (threshold={self.psi_threshold})\n"
            f"     ├─ PSI     : {psi['psi_value']:.4f}\n"
            f"     ├─ Shift   : {psi['interpretation']}\n"
            f"     └─ Drift   : {'Yes' if psi['drift_detected'] else 'No'}\n" + "-" * 40
        )
        return report_str
