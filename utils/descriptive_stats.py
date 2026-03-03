import polars as pl


class DescriptiveStats:
    @staticmethod
    def get_stats(serie: pl.Series) -> dict:
        df = serie.describe()
        stats = dict(zip(df["statistic"], df["value"]))
        return {
            "mean": stats.get("mean"),
            "median": stats.get("50%"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "Q1": stats.get("25%"),
            "Q3": stats.get("75%"),
        }
