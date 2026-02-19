import polars as pl


class TimeParser:
    def __init__(self, user_input_format=None, max_fail_ratio=10.0):
        self.user_input_format = user_input_format
        self.max_fail_ratio = max_fail_ratio

    def parse_time_new(self, df: pl.DataFrame, column_name: str) -> pl.DataFrame:
        parsedcol = f"parsed_{column_name}"

        if self.user_input_format:
            expr = (
                pl.col(column_name)
                .cast(pl.Utf8)
                .str.to_datetime(format=self.user_input_format, strict=False)
                .alias(parsedcol)
            )
        else:
            expr = (
                pl.col(column_name)
                .cast(pl.Utf8)
                .str.to_datetime(strict=False)
                .alias(parsedcol)
            )

        df = df.with_columns(expr)

        failure_stats = df.select(
            total=pl.col(column_name).is_not_null(),
            failed=pl.col(parsedcol).is_null() & pl.col(column_name).is_not_null(),
        ).sum()

        failed_count = failure_stats.get_column("failed")[0]
        total_count = failure_stats.get_column("total")[0]
        failure_rate = (failed_count / total_count) * 100

        if total_count == 0:
            raise ValueError(f"Column '{column_name}' contains no valid values")

        if failure_rate > self.max_fail_ratio:
            raise ValueError(
                f"Failed to parse {failure_rate: .2f}% of values in column {column_name} "
                f"please specify --time-format explicitly"
            )

        return df
