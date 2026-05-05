"""
Shared utilities for the replication package.

The panels in ``data/`` are already winsorised at the 1st and 99th
percentiles of the full panel (see the data-construction notes in
``codebook.md``). The analysis scripts therefore only need to standardise
within each estimation sample.

- ``load_panel`` reads a panel CSV with consistent dtypes for the firm-year
  identifiers, so the pandas merge / fixed-effects machinery sees the right
  columns no matter where the data is read from.

- ``standardise_in_sample`` z-scores the listed columns using the within-sample
  mean and standard deviation, so coefficients on standardised regressors can
  be read as standard-deviation responses.

- ``demean`` subtracts the within-group mean. Two sequential calls on the
  firm and year identifiers absorb two-way fixed effects manually, used
  by ``03_iv.py`` for the 2SLS specification.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Paths relative to the repository root
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
OUTPUT_DIR = REPO / "output"  # gitignored; created on first use


def load_panel(name: str) -> pd.DataFrame:
    """Load one of the panel CSVs in ``data/``. Casts ``stkcd`` to a
    six-character string and ``year`` to int, which matches the rest of the
    pipeline."""
    df = pd.read_csv(DATA_DIR / name, dtype={"stkcd": str})
    df["stkcd"] = df["stkcd"].str.zfill(6)
    df["year"] = df["year"].astype(int)
    return df


def standardise_in_sample(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Z-score the listed columns using their within-sample mean and SD.
    Returns a copy of ``df``; the original is untouched."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        sd = out[c].std()
        if sd and sd > 0:
            out[c] = (out[c] - out[c].mean()) / sd
    return out


def demean(df: pd.DataFrame, cols: list, by: str) -> pd.DataFrame:
    """Subtract the within-group mean from each column. Used for two-way
    fixed effects via successive demeaning."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def ensure_output_dir() -> Path:
    """Create the gitignored ``output/`` directory if it does not exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR
