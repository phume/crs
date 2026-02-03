"""
TSFresh Feature Extraction Engine
Cross-pillar automated time-series feature extraction.
"""

import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from src.utils.config import cfg


FC_PARAMS = {
    "ComprehensiveFCParameters": ComprehensiveFCParameters(),
    "MinimalFCParameters": MinimalFCParameters(),
}


def build_time_series(
    transactions: pd.DataFrame,
    client_id_col: str = "client_id",
    date_col: str = "txn_date",
    amount_col: str = "amount",
    freq: str = "W",
) -> pd.DataFrame:
    """
    Build weekly (or other frequency) aggregated time series per client
    suitable for TSFresh ingestion.

    Returns DataFrame with columns: [id, time, txn_count, txn_amount_sum,
    txn_amount_mean, txn_amount_max].
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    agg = (
        df.groupby([client_id_col, pd.Grouper(key=date_col, freq=freq)])
        .agg(
            txn_count=(amount_col, "count"),
            txn_amount_sum=(amount_col, "sum"),
            txn_amount_mean=(amount_col, "mean"),
            txn_amount_max=(amount_col, "max"),
        )
        .reset_index()
    )

    # TSFresh needs sequential integer time index per client
    agg["time_idx"] = agg.groupby(client_id_col).cumcount()

    return agg.rename(columns={client_id_col: "id"})


def extract_tsfresh_features(
    ts_df: pd.DataFrame,
    column_id: str = "id",
    column_sort: str = "time_idx",
) -> pd.DataFrame:
    """
    Extract comprehensive TSFresh features from time-series DataFrame.

    Parameters
    ----------
    ts_df : DataFrame with columns [id, time_idx, txn_count, txn_amount_sum, ...]

    Returns
    -------
    DataFrame indexed by client id with extracted features.
    """
    value_cols = [
        c for c in ts_df.columns if c not in [column_id, column_sort, "txn_date"]
    ]

    fc_params = FC_PARAMS.get(cfg.tsfresh.fc_parameters, ComprehensiveFCParameters())

    features = extract_features(
        ts_df[[column_id, column_sort] + value_cols],
        column_id=column_id,
        column_sort=column_sort,
        default_fc_parameters=fc_params,
        n_jobs=cfg.tsfresh.n_jobs,
        chunksize=cfg.tsfresh.chunksize,
        disable_progressbar=False,
    )

    impute(features)
    return features


def select_relevant_features(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Apply Benjamini-Yekutieli relevance filtering.

    Parameters
    ----------
    features : TSFresh-extracted feature matrix
    target   : binary STR labels aligned with features index

    Returns
    -------
    Filtered DataFrame with only statistically relevant features.
    """
    return select_features(
        features,
        target,
        fdr_level=cfg.tsfresh.fdr_level,
        n_jobs=cfg.tsfresh.n_jobs,
    )
