import polars as pl
import polars.selectors as cs


def get_numerical_drift_elligible_column(
    df: pl.DataFrame, excluded_cols: list[str]
) -> list[str]:
    df_filtered = df.drop(excluded_cols, strict=False)

    numeric_df = df_filtered.select(cs.numeric())

    elligible_cols = []

    total_rows = numeric_df.height

    for col in numeric_df.get_columns():
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


def get_categorical_drift_elligble_column(
    df: pl.DataFrame,
    excluded_cols: list[str],
    max_cardinality: int = 50,
    null_threshold: float = 0.9,
) -> list[str]:
    df_filtered = df.drop(excluded_cols, strict=False)

    categoric_df = df_filtered.select(cs.categorical() | cs.string())
    eligible_cols = []

    total_rows = categoric_df.height

    for col in categoric_df.get_columns():
        null_count = col.null_count()
        num_unique = col.n_unique()

        if null_count == total_rows or num_unique <= 1:
            continue

        if (null_count / total_rows) > null_threshold:
            continue

        if num_unique > max_cardinality or num_unique == (total_rows - null_count):
            continue

        eligible_cols.append(col.name)

    return eligible_cols
