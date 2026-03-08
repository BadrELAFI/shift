from scipy.stats import ks_2samp
import numpy as np
import polars as pl


class NumericDriftDetector:
    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.2) -> None:
        self.alpha = alpha
        self.psi_threshold = psi_threshold

    def _kstest(
        self,
        df_baseline: pl.DataFrame,
        df_target: pl.DataFrame,
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

    def _make_psi_quantile_bins(
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

    def _binned_proportions(self, df, column: str, breaks: list) -> list:
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

    def _calculate_psi(
        self, df_baseline, df_target, column_baseline, column_target, nbins=10
    ) -> dict:
        edges = self._make_psi_quantile_bins(df_baseline, column_baseline, nbins)
        breaks = edges[1:-1].tolist()

        # proportions
        prop_expected = self._binned_proportions(df_baseline, column_baseline, breaks)
        prop_actual = self._binned_proportions(df_target, column_target, breaks)

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

    def evaluate_column(
        self,
        df_baseline: pl.DataFrame,
        df_target: pl.DataFrame,
        column_baseline: str,
        column_target: str,
    ) -> dict:
        """Runs all drift tests on a single column and returns a structured report."""
        try:
            ks_results = self._kstest(
                df_baseline, df_target, column_baseline, column_target
            )
            psi_results = self._calculate_psi(
                df_baseline, df_target, column_baseline, column_target
            )

            overall_drift = (
                ks_results["drift_detected"] or psi_results["drift_detected"]
            )

            return {
                "feature_name": column_target,
                "status": "success",
                "overall_drift_detected": overall_drift,
                "ks_test": ks_results,
                "psi_test": psi_results,
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
        ks = column_report["ks_test"]
        psi = column_report["psi_test"]

        report_str = (
            f"Feature: **{feature}** | {drift_icon}\n"
            f"  ├─ Kolmogorov-Smirnov Test (alpha={ks['alpha']})\n"
            f"  │  ├─ KS-Stat : {ks['ks_stat']:.4f}\n"
            f"  │  ├─ P-Value : {ks['p_value']:.4e}\n"
            f"  │  └─ Drift   : {'Yes' if ks['drift_detected'] else 'No'}\n"
            f"  └─ Population Stability Index (threshold={self.psi_threshold})\n"
            f"     ├─ PSI     : {psi['psi_value']:.4f}\n"
            f"     ├─ Shift   : {psi['interpretation']}\n"
            f"     └─ Drift   : {'Yes' if psi['drift_detected'] else 'No'}\n" + "-" * 40
        )
        return report_str
