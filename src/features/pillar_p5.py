"""
Pillar 5: Profile Consistency Risk
Features measuring deviation between observed behavior and declared client profile.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from src.utils.config import cfg


def compute_p5_features(
    transactions: pd.DataFrame,
    client_profiles: pd.DataFrame,
    client_id_col: str = "client_id",
    amount_col: str = "amount",
    date_col: str = "txn_date",
    expected_turnover_col: str = "expected_turnover",
    industry_col: str = "industry",
    occupation_col: str = "occupation",
) -> pd.DataFrame:
    """
    Compute Profile Consistency features per client.

    Parameters
    ----------
    transactions : transaction-level data
    client_profiles : KYC/CDD profile data with expected turnover, industry, etc.

    Returns DataFrame indexed by client_id with P5 feature columns.
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    profiles = client_profiles.set_index(client_id_col) if client_id_col in client_profiles.columns else client_profiles

    features = {}

    # Client-level aggregates
    client_agg = df.groupby(client_id_col).agg(
        total_volume=(amount_col, "sum"),
        txn_count=(amount_col, "count"),
        mean_amount=(amount_col, "mean"),
        std_amount=(amount_col, "std"),
    )

    for cid in client_agg.index:
        row = client_agg.loc[cid]
        feat = {}

        # Turnover ratio
        if expected_turnover_col in profiles.columns and cid in profiles.index:
            expected = profiles.loc[cid, expected_turnover_col]
            if pd.notna(expected) and expected > 0:
                feat["p5_turnover_ratio"] = row["total_volume"] / expected
            else:
                feat["p5_turnover_ratio"] = np.nan
        else:
            feat["p5_turnover_ratio"] = np.nan

        # Dormancy detection
        client_txns = df[df[client_id_col] == cid].sort_values(date_col)
        if len(client_txns) > 1:
            gaps = client_txns[date_col].diff().dt.days.dropna()
            max_gap = gaps.max()
            feat["p5_max_dormancy_days"] = max_gap
            feat["p5_dormancy_flag"] = int(max_gap >= cfg.p5.dormancy_threshold_days)
        else:
            feat["p5_max_dormancy_days"] = 0
            feat["p5_dormancy_flag"] = 0

        # Occupation risk flag (for P6 downstream use too)
        if occupation_col in profiles.columns and cid in profiles.index:
            occ = str(profiles.loc[cid, occupation_col]).lower().strip()
            feat["p5_high_risk_occupation"] = int(
                occ in [o.lower() for o in cfg.p6.high_risk_occupations]
            )
        else:
            feat["p5_high_risk_occupation"] = 0

        feat["p5_total_volume"] = row["total_volume"]
        feat["p5_txn_count"] = row["txn_count"]
        feat["p5_mean_amount"] = row["mean_amount"]
        feat["p5_std_amount"] = row["std_amount"] if pd.notna(row["std_amount"]) else 0.0

        features[cid] = feat

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result


def compute_peer_deviation(
    feature_matrix: pd.DataFrame,
    client_profiles: pd.DataFrame,
    client_id_col: str = "client_id",
    peer_cols: list = None,
) -> pd.Series:
    """
    Compute Mahalanobis distance from peer group centroid.

    Parameters
    ----------
    feature_matrix : numeric features per client (e.g., P1 + P5 features)
    client_profiles : with peer grouping columns (industry, size, geography)
    peer_cols : columns to define peer groups

    Returns Series of peer deviation scores indexed by client_id.
    """
    if peer_cols is None:
        peer_cols = cfg.p5.peer_group_features

    profiles = client_profiles.set_index(client_id_col) if client_id_col in client_profiles.columns else client_profiles
    available_cols = [c for c in peer_cols if c in profiles.columns]

    if not available_cols:
        return pd.Series(0.0, index=feature_matrix.index, name="p5_peer_deviation")

    merged = feature_matrix.join(profiles[available_cols], how="left")
    group_key = available_cols

    scores = pd.Series(np.nan, index=feature_matrix.index, name="p5_peer_deviation")
    numeric_cols = feature_matrix.columns.tolist()

    for _, group_df in merged.groupby(group_key):
        if len(group_df) < 3:
            continue
        data = group_df[numeric_cols].fillna(0)
        centroid = data.mean().values
        cov = data.cov().values
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue
        for idx in group_df.index:
            try:
                scores[idx] = mahalanobis(data.loc[idx].values, centroid, cov_inv)
            except Exception:
                scores[idx] = np.nan

    return scores.fillna(0.0)
