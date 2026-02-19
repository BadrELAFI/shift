from scipy.stats import ks_2samp
import numpy as np
import polars as pl


class NumericDriftDetector:
    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.2) -> None:
        self.alpha = alpha
        self.psi_threshold = psi_threshold

    def kstest(
        self,
        df_baseline,
        df_target,
        column_baseline: str,
        column_target: str,
    ) -> dict:
        baseline = df_baseline[column_baseline].drop_nulls().to_numpy()
        target = df_target[column_target].drop_nulls().to_numpy()

        ks_stat, p_value = ks_2samp(baseline, target)

        return {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "alpha": self.alpha,
            "drift_detected": p_value < self.alpha,
        }

    def make_psi_quantile_bins(
        self,
        df_baseline: pl.DataFrame,
        column_baseline: str,
        nbins: int = 10,
    ) -> np.ndarray:
        quantiles = np.linspace(0, 1, nbins + 1)

        quantiles_expr = [
            pl.col(column_baseline).drop_nulls().quantile(q).alias(f"q_{q}")
            for q in quantiles
        ]

        edges = df_baseline.select(quantiles_expr).to_numpy().flatten().astype(float)

        edges = np.unique(edges)

        if len(edges) < 3:
            raise ValueError("not enough unique values to create buckets for psi test")

        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        return edges

    def binned_proportions(self, df, column: str, breaks: list) -> list:
        binned = (
            df.select(pl.col(column).drop_nulls())
            .with_columns(pl.col(column).cut(breaks=breaks).alias("bin"))
            .group_by("bin", maintain_order=True)
            .len()
            .sort("bin")
        )

        total = binned["len"].sum()
        if total == 0:
            return [0.0] * (len(breaks) + 1)

        return (binned["len"] / total).to_list()

    def calculate_psi(
        self, df_baseline, df_target, column_baseline, column_target, nbins=10
    ) -> dict:
        edges = self.make_psi_quantile_bins(df_baseline, column_baseline, nbins)
        breaks = edges[1:-1].tolist()

        # proportions
        prop_expected = self.binned_proportions(df_baseline, column_baseline, breaks)
        prop_actual = self.binned_proportions(df_target, column_target, breaks)

        epsilon = 1e-6  # cas ou on divise par 0 car on calucle ln(actual/expected)
        expected = np.array(prop_expected) + epsilon
        actual = np.array(prop_actual) + epsilon

        psi = np.sum((actual - expected) * np.log(actual / expected))

        return {
            "psi_value": psi,
            "drift_detected": psi > self.psi_threshold,
            "interpretation": self._interpret_psi(psi),
        }

    def _interpret_psi(self, psi: float) -> str:
        if psi < 0.1:
            return "Low shift"
        if psi < 0.2:
            return "Moderate shift"
        return "Significant shift"
