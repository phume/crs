"""
Data Preprocessing
Windowing, train/test splitting, and feature matrix assembly.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils.config import cfg


def window_transactions(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    date_col: str = "txn_date",
) -> pd.DataFrame:
    """
    Filter transactions to the observation window ending at reference_date.

    Parameters
    ----------
    transactions : full transaction history
    reference_date : end of observation window
    date_col : date column name

    Returns filtered DataFrame.
    """
    window_months = cfg.window.window_months
    start_date = reference_date - pd.DateOffset(months=window_months)
    mask = (transactions[date_col] >= start_date) & (
        transactions[date_col] <= reference_date
    )
    return transactions.loc[mask].copy()


def create_labels(
    client_ids: pd.Index,
    str_labels: pd.DataFrame,
    reference_date: pd.Timestamp,
    client_id_col: str = "client_id",
    str_flag_col: str = "str_flag",
    str_date_col: str = "str_date",
) -> pd.Series:
    """
    Create binary labels for supervised training.

    A client is labelled positive if an STR was filed within the label horizon
    after the reference date, or if str_date is unavailable and str_flag == 1.
    """
    horizon = cfg.window.label_horizon_months
    labels = pd.Series(0, index=client_ids, name="label")

    if str_date_col in str_labels.columns:
        horizon_end = reference_date + pd.DateOffset(months=horizon)
        str_labels = str_labels.copy()
        str_labels[str_date_col] = pd.to_datetime(str_labels[str_date_col])
        flagged = str_labels[
            (str_labels[str_flag_col] == 1)
            & (str_labels[str_date_col] > reference_date)
            & (str_labels[str_date_col] <= horizon_end)
        ][client_id_col]
    else:
        flagged = str_labels[str_labels[str_flag_col] == 1][client_id_col]

    labels.loc[labels.index.isin(flagged)] = 1
    return labels


def temporal_train_test_split(
    transactions: pd.DataFrame,
    str_labels: pd.DataFrame,
    split_date: pd.Timestamp,
    client_id_col: str = "client_id",
    date_col: str = "txn_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Temporal split: train on transactions before split_date, test after.

    Returns (train_txns, test_txns, train_ref_date, test_ref_date).
    """
    train_ref = split_date
    test_ref = transactions[date_col].max()

    train_txns = window_transactions(transactions, train_ref, date_col)
    test_txns = window_transactions(transactions, test_ref, date_col)

    return train_txns, test_txns, train_ref, test_ref


def assemble_feature_matrix(
    pillar_features: dict,
    client_id_col: str = "client_id",
) -> pd.DataFrame:
    """
    Combine feature DataFrames from multiple pillars into a single matrix.

    Parameters
    ----------
    pillar_features : dict mapping pillar name to DataFrame (indexed by client_id)

    Returns combined feature matrix with all pillar columns.
    """
    dfs = list(pillar_features.values())
    if not dfs:
        return pd.DataFrame()

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.fillna(0.0)
    return combined
