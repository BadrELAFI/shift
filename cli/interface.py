import typer
from typing import Optional
from config_loader import ConfigLoader

app = typer.Typer(help="Shift: The data drift detection tool")


@app.command()
def detect(
    target: str = typer.Argument(..., help="Target dataset path (CSV/Parquet)"),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", "-b", help="Baseline dataset path"
    ),
    start: Optional[str] = typer.Option(
        None, "--start", "-s", help="Target window start"
    ),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="Target window end"),
    date_format: str = typer.Option(
        None,
        "--date-format",
        "-df",
        help="Datetime format (optional, Polars will infer if omitted)",
    ),
    column: str = typer.Option("timestamp", "--col", help="Datetime column name"),
    psi_threshold: Optional[float] = typer.Option(
        0.2, "--psi-threshold", "-psit", help="Threshold for psi test (default: 0.2)"
    ),
    ks_alpha: Optional[float] = typer.Option(
        0.05,
        "--ks-alpha",
        "-ksa",
        help="significance value for ks test (default: 0.05)",
    ),
) -> dict:
    config = ConfigLoader.load_config()
    typer.echo(f"Initializing detection on: {target}")

    if baseline and (start or end):
        raise typer.BadParameter("Use either --baseline or --start/--end, not both.")

    if (start and not end) or (end and not start):
        raise typer.BadParameter("--start and --end must be provided together.")

    if not baseline and not (start and end):
        raise typer.BadParameter(
            "You must provide either --baseline or both --start and --end."
        )
    psi_threshold = (
        psi_threshold
        if 0 < psi_threshold < 1
        else config.get("tests", {}).get("defaults", {}).get("psi_threshold", 0.2)
    )

    ks_alpha = (
        ks_alpha
        if 0 < ks_alpha < 0.2
        else config.get("tests", {}).get("defaults", {}).get("ks_alpha", 0.05)
    )

    return {
        "target": target,
        "baseline": baseline,
        "start": start,
        "end": end,
        "date_column": column,
        "psi_threshold": psi_threshold,
        "ks_alpha": ks_alpha,
        "date_format": date_format,
    }
