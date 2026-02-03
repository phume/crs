"""
Pillar 4: Typology Alignment Risk
Features quantifying similarity to known ML typologies (structuring, layering,
funnel accounts, rapid fund-through).
"""

import numpy as np
import pandas as pd

from src.utils.config import cfg


def compute_p4_features(
    transactions: pd.DataFrame,
    client_id_col: str = "client_id",
    date_col: str = "txn_date",
    amount_col: str = "amount",
    direction_col: str = "direction",
    counterparty_col: str = "counterparty_id",
) -> pd.DataFrame:
    """
    Compute Typology Alignment features per client.

    Returns DataFrame indexed by client_id with P4 feature columns.
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    threshold = cfg.p4.reporting_threshold
    proximity = cfg.p4.threshold_proximity_pct
    round_tol = cfg.p4.round_number_tolerance

    features = {}

    for cid, grp in df.groupby(client_id_col):
        amounts = grp[amount_col]
        n_txn = len(grp)

        # --- Structuring indicators ---
        near_threshold = amounts.between(
            threshold * (1 - proximity), threshold, inclusive="left"
        )
        just_below_count = near_threshold.sum()
        just_below_ratio = just_below_count / n_txn if n_txn > 0 else 0.0

        # --- Round-number ratio ---
        is_round = (amounts % round_tol == 0)
        round_ratio = is_round.mean()

        # --- Flow-through ratio ---
        if direction_col in grp.columns:
            inbound = grp.loc[grp[direction_col] == "inbound", amount_col].sum()
            outbound = grp.loc[grp[direction_col] == "outbound", amount_col].sum()
        else:
            inbound = amounts.sum() / 2
            outbound = amounts.sum() / 2
        flow_through_ratio = outbound / inbound if inbound > 0 else 0.0

        # --- Rapid fund-through ---
        rapid_hours = cfg.p4.rapid_throughput_hours
        sorted_txns = grp.sort_values(date_col)
        rapid_count = 0
        if direction_col in grp.columns and len(sorted_txns) > 1:
            inbound_txns = sorted_txns[sorted_txns[direction_col] == "inbound"]
            outbound_txns = sorted_txns[sorted_txns[direction_col] == "outbound"]
            for _, in_txn in inbound_txns.iterrows():
                time_diff = (outbound_txns[date_col] - in_txn[date_col]).dt.total_seconds() / 3600
                rapid_count += ((time_diff > 0) & (time_diff <= rapid_hours)).sum()
        rapid_ratio = rapid_count / n_txn if n_txn > 0 else 0.0

        # --- Funnel account (many-to-one / one-to-many) ---
        if counterparty_col in grp.columns and direction_col in grp.columns:
            senders = grp.loc[grp[direction_col] == "inbound", counterparty_col].nunique()
            receivers = grp.loc[grp[direction_col] == "outbound", counterparty_col].nunique()
            funnel_ratio = (
                senders / max(receivers, 1) if senders > receivers
                else receivers / max(senders, 1)
            )
        else:
            senders, receivers = 0, 0
            funnel_ratio = 1.0

        features[cid] = {
            "p4_just_below_threshold_count": just_below_count,
            "p4_just_below_threshold_ratio": just_below_ratio,
            "p4_round_number_ratio": round_ratio,
            "p4_flow_through_ratio": flow_through_ratio,
            "p4_rapid_throughput_count": rapid_count,
            "p4_rapid_throughput_ratio": rapid_ratio,
            "p4_funnel_ratio": funnel_ratio,
            "p4_n_senders": senders,
            "p4_n_receivers": receivers,
            "p4_inbound_total": inbound,
            "p4_outbound_total": outbound,
        }

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result
