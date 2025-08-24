"""Simple CLI for wignado utilities using Typer.

Commands:
- normalize: normalize a comma-separated list of numeric values
- stats: compute chunk statistics for values
- convert: run (or dry-run) the BigWig -> TileDB conversion
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import typer
import numpy as np

from .engine import (
    compute_chunk_statistics,
    BigWigToTileDBConverter,
    ConverterConfig,
)

app = typer.Typer(help="wignado: small CLI for quick ops")


def _parse_values(values: str) -> np.ndarray:
    parts = [v.strip() for v in values.split(",") if v.strip()]
    parsed: List[float] = []
    for p in parts:
        if p.lower() in {"nan", "nanf"}:
            parsed.append(float("nan"))
        elif p.lower() in {"inf", "+inf", "inff"}:
            parsed.append(float("inf"))
        elif p.lower() in {"-inf", "-inff"}:
            parsed.append(float("-inf"))
        else:
            parsed.append(float(p))
    return np.array(parsed)


@app.command()
def query(values: str = typer.Option(..., help="Comma-separated numeric values")) -> None:
    """Run a single query-like operation over numeric values and print stats.

    This command computes mean/max/min and valid count ignoring NaN/Inf.
    It replaces the previous normalize/stats helpers and provides a single
    entry point suitable for quick CLI checks.
    """
    arr = _parse_values(values)
    mean_val, max_val, min_val, cnt = compute_chunk_statistics(arr.astype(np.float64))
    typer.echo(f"mean={mean_val}, max={max_val}, min={min_val}, valid_count={cnt}")


@app.command()
def convert(
    bigwig: Path = typer.Argument(..., help="Path to input bigWig file"),
    tiledb_path: Path = typer.Argument(..., help="Path to output TileDB directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print config and exit without running"),
) -> None:
    """Run a full BigWig -> TileDB conversion (or dry-run to show config).

    This command calls the `BigWigToTileDBConverter` from the package.
    """
    config = ConverterConfig()
    typer.echo(f"Using config: {config}")
    if dry_run:
        typer.echo("Dry run: not executing conversion")
        raise typer.Exit()

    conv = BigWigToTileDBConverter(bigwig, tiledb_path, config=config)
    typer.echo("Starting conversion...")
    metrics = conv.run_conversion()
    typer.echo(f"Conversion complete: time={metrics.total_time_seconds:.2f}s, values={metrics.total_values}")


if __name__ == "__main__":
    app()


def main() -> None:
    """Entrypoint wrapper used by console scripts.

    If the script is installed as `convert` or `query`, this wrapper inserts
    the appropriate subcommand so users can invoke the single-action binary
    without specifying the subcommand name explicitly.
    """
    prog = Path(sys.argv[0]).stem
    # if invoked as `convert` or `query`, default to the corresponding subcommand
    if prog in {"convert", "query"}:
        if len(sys.argv) == 1 or sys.argv[1] not in app.commands:
            sys.argv.insert(1, prog)
    app()
