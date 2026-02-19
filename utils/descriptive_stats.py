import polars as pl


class DescriptiveStats:
    @staticmethod
    def get_stats(serie: pl.Series) -> dict:
        df = serie.describe()
        stats = dict(zip(df["statistic"], df["value"]))
        return {
            "mean": stats["mean"],
            "median": stats["50%"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "Q1": stats["25%"],
            "Q3": stats["75%"],
        }
