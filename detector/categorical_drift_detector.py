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
