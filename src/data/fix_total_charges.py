"""
fix_total_charges.py

Replaces blank-space values in the 'Total Charges' column of the raw
Telco churn CSV with 0.0 and writes the cleaned file back to disk.

Usage:
    python -m src.data.fix_total_charges
    # or with custom paths:
    python -m src.data.fix_total_charges --input data/raw/Telco_customer_churn.csv \
                                          --output data/raw/Telco_customer_churn.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco_customer_churn.csv"
COLUMN = "Total Charges"


def fix_total_charges(input_path: Path, output_path: Path) -> None:
    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)

    if COLUMN not in df.columns:
        raise KeyError(f"Column '{COLUMN}' not found. Available columns: {df.columns.tolist()}")

    # Count blank-space entries before fix
    blank_mask = df[COLUMN].astype(str).str.strip() == ""
    blank_count = blank_mask.sum()
    logger.info("Found %d blank-space value(s) in '%s'", blank_count, COLUMN)

    # Replace blanks with NaN then coerce to float, filling NaN with 0.0
    df[COLUMN] = pd.to_numeric(df[COLUMN], errors="coerce").fillna(0.0)

    logger.info("Replaced %d blank(s) with 0.0", blank_count)
    logger.info("Writing cleaned file to %s", output_path)
    df.to_csv(output_path, index=False)
    logger.info("Done. Shape: %s", df.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix blank Total Charges values in Telco CSV.")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT, help="Path to input CSV")
    parser.add_argument("--output", type=Path, default=None,          help="Path to output CSV (defaults to input)")
    args = parser.parse_args()

    output = args.output or args.input
    fix_total_charges(args.input, output)


if __name__ == "__main__":
    main()
