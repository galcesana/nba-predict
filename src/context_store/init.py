"""CLI entry point for initializing the forecast context store."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.context_store.schema import initialize_context_store
from src.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Initialize the forecast context store.")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--parquet-root", type=Path, default=None)
    args = parser.parse_args(argv)

    db_path, parquet_root = initialize_context_store(
        db_path=args.db_path,
        parquet_root=args.parquet_root,
    )
    print(f"Initialized context store DB: {db_path}")
    print(f"Initialized context store Parquet root: {parquet_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
