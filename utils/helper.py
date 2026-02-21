import polars as pl
import polars.selectors as cs


def get_numerical_drift_elligible_numeric_column(
    df: pl.DataFrame, excluded_cols: list[str]
) -> list[str]:
    df_filtered = df.drop(excluded_cols, strict=False)

    numeric_df = df_filtered.select(cs.numeric())

    elligible_cols = []

    total_rows = numeric_df.height

    for col in df_filtered.get_columns():
        if col.null_count() == total_rows:
            continue

        num_unique_values = col.n_unique()
        if num_unique_values <= 1:
            continue

        valid_rows_count = total_rows - col.null_count()

        if num_unique_values == valid_rows_count:
            continue

        elligible_cols.append(col.name)

    return elligible_cols
