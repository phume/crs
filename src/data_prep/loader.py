"""
Data Loading
Utilities for loading raw transaction data, client profiles, and STR labels.
"""

import pandas as pd
from pathlib import Path

from src.utils.config import RAW_DIR


def load_transactions(
    path: Path = None,
    date_col: str = "txn_date",
) -> pd.DataFrame:
    """
    Load transaction-level data.

    Parameters
    ----------
    path : Path to CSV/parquet file. Defaults to RAW_DIR / "transactions.csv".
    date_col : column to parse as datetime.

    Returns DataFrame with parsed dates.
    """
    if path is None:
        path = RAW_DIR / "transactions.csv"
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[date_col])

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_client_profiles(path: Path = None) -> pd.DataFrame:
    """
    Load KYC/CDD client profile data.

    Expected columns include: client_id, industry, occupation,
    expected_turnover, geography, size_bucket, onboarding_date, etc.
    """
    if path is None:
        path = RAW_DIR / "client_profiles.csv"
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_str_labels(path: Path = None) -> pd.DataFrame:
    """
    Load STR (Suspicious Transaction Report) labels.

    Expected columns: client_id, str_flag (0/1), str_date (optional).
    """
    if path is None:
        path = RAW_DIR / "str_labels.csv"
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
