#!/usr/bin/env python3
"""Optuna tuner for BigWig -> TileDB conversion

Usage: python scripts/optuna_tune_conversion.py --bigwig /path/to/file.bigWig --trials 20

This script runs an Optuna study that varies key fields of `ConverterConfig`
and measures the conversion time from `BigwigToTileDBConverter.convert()`.

Notes:
- Requires the normal runtime deps (pybigtools, tiledb, optuna). If missing,
  the script will print a helpful message.
- By default each trial writes its TileDB output to a temporary directory and
  is removed after the trial. Use --keep to retain outputs for inspection.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
import logging

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

try:
    from wignado.engine import BigwigToTileDBConverter, ConverterConfig
except Exception as e:  # pragma: no cover - will error at runtime if core deps missing
    BigwigToTileDBConverter = None
    ConverterConfig = None
    _import_error = e


def objective_factory(bigwig_path: Path, keep_outputs: bool = False):
    """Return an objective function bound to the provided bigwig path."""

    def objective(trial):
        # Suggest parameters to tune
        chunk_size = trial.suggest_int("chunk_size", 10_000, 5_000_000, log=True)
        tile_size = trial.suggest_int("tile_size", 1_000, 200_000, log=True)
        compression_level = trial.suggest_int("compression_level", 1, 9)
        buffer_size = trial.suggest_int("buffer_size", 50_000, 2_000_000, log=True)
        max_workers = trial.suggest_int("max_workers", 1, 16)
        memory_limit_gb = trial.suggest_float("memory_limit_gb", 1.0, 64.0, log=True)
        cache_size_gb = trial.suggest_float("cache_size_gb", 0.1, 32.0, log=True)
        write_batch_size = trial.suggest_int("write_batch_size", 1, 256, log=True)
        value_dtype = trial.suggest_categorical("value_dtype", ["float32", "float16"])

        # Build config
        cfg = ConverterConfig(
            overwrite=True,
            chunk_size=chunk_size,
            tile_size=tile_size,
            compression_level=compression_level,
            buffer_size=buffer_size,
            max_workers=max_workers,
            memory_limit_gb=memory_limit_gb,
            cache_size_gb=cache_size_gb,
            write_batch_size=write_batch_size,
            value_dtype=("float16" if value_dtype == "float16" else "float32"),
        )

        # Prepare unique output directory for this trial
        base_tmp = Path(tempfile.mkdtemp(prefix="wignado_optuna_"))
        out_dir = base_tmp / f"trial_{trial.number}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            conv = BigwigToTileDBConverter(str(bigwig_path), str(out_dir), config=cfg)
        except Exception as e:
            # If initialization fails (missing deps or bad file) report a large penalty
            trial.set_user_attr("init_error", str(e))
            # cleanup temp dir
            if not keep_outputs:
                shutil.rmtree(base_tmp, ignore_errors=True)
            return 1e9

        start = time.perf_counter()
        try:
            metrics = conv.convert()
            elapsed = metrics.total_time_seconds if metrics and metrics.total_time_seconds > 0 else (time.perf_counter() - start)
        except Exception as e:
            # Record error and return large objective value
            trial.set_user_attr("runtime_error", str(e))
            elapsed = 1e9

        # Optionally cleanup the tiledb output
        if not keep_outputs:
            try:
                shutil.rmtree(base_tmp)
            except Exception:
                pass

        # Optuna minimizes the objective (we want minimum time)
        return float(elapsed)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BigWig -> TileDB conversion with Optuna")
    parser.add_argument("--bigwig", required=True, type=Path, help="Path to input .bigWig file")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for the study")
    parser.add_argument("--study-name", type=str, default="wignado_conversion", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optional optuna sqlite storage URI, e.g. sqlite:///optuna.db")
    parser.add_argument("--keep", action="store_true", help="Keep per-trial TileDB outputs for inspection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if optuna is None:
        raise RuntimeError("optuna is not installed. Install it with `pip install optuna`")

    if BigwigToTileDBConverter is None or ConverterConfig is None:
        raise RuntimeError(f"Failed to import converter: {_import_error}")

    if not args.bigwig.exists():
        raise FileNotFoundError(f"BigWig file not found: {args.bigwig}")

    study = optuna.create_study(study_name=args.study_name, direction="minimize", storage=args.storage, load_if_exists=True)

    objective = objective_factory(args.bigwig, keep_outputs=args.keep)

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best trial:")
    print(f"  Value (seconds): {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
