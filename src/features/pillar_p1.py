"""
Pillar 1: Volume & Velocity Risk
Features capturing transaction volume, frequency, and rate-of-change anomalies.
"""

import numpy as np
import pandas as pd

from src.utils.config import cfg


def compute_p1_features(
    transactions: pd.DataFrame,
    client_id_col: str = "client_id",
    date_col: str = "txn_date",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Compute Volume & Velocity features per client.

    Returns DataFrame indexed by client_id with P1 feature columns.
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    features = {}

    for cid, grp in df.groupby(client_id_col):
        amounts = grp[amount_col]
        dates = grp[date_col]
        span_days = max((dates.max() - dates.min()).days, 1)

        # Volume
        features[cid] = {
            "p1_txn_count": len(grp),
            "p1_txn_total": amounts.sum(),
            "p1_txn_mean": amounts.mean(),
            "p1_txn_median": amounts.median(),
            "p1_txn_std": amounts.std() if len(grp) > 1 else 0.0,
            "p1_txn_max": amounts.max(),
            "p1_txn_min": amounts.min(),
            "p1_txn_cv": (amounts.std() / amounts.mean()) if amounts.mean() != 0 else 0.0,
            "p1_txn_skew": amounts.skew() if len(grp) > 2 else 0.0,
            "p1_txn_kurtosis": amounts.kurtosis() if len(grp) > 3 else 0.0,

            # Velocity
            "p1_daily_rate": len(grp) / span_days,
            "p1_daily_amount_rate": amounts.sum() / span_days,
        }

        # Inter-transaction time
        if len(grp) > 1:
            intervals = dates.sort_values().diff().dt.total_seconds().dropna() / 3600
            features[cid].update({
                "p1_itt_mean_hours": intervals.mean(),
                "p1_itt_std_hours": intervals.std(),
                "p1_itt_min_hours": intervals.min(),
                "p1_itt_max_hours": intervals.max(),
            })
        else:
            features[cid].update({
                "p1_itt_mean_hours": np.nan,
                "p1_itt_std_hours": np.nan,
                "p1_itt_min_hours": np.nan,
                "p1_itt_max_hours": np.nan,
            })

        # Weekly aggregation for velocity change
        weekly = grp.set_index(date_col).resample("W")[amount_col]
        weekly_counts = weekly.count()
        weekly_sums = weekly.sum()
        if len(weekly_counts) > 1:
            features[cid]["p1_weekly_count_trend"] = np.polyfit(
                range(len(weekly_counts)), weekly_counts.values, 1
            )[0]
            features[cid]["p1_weekly_amount_trend"] = np.polyfit(
                range(len(weekly_sums)), weekly_sums.values, 1
            )[0]
        else:
            features[cid]["p1_weekly_count_trend"] = 0.0
            features[cid]["p1_weekly_amount_trend"] = 0.0

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result
