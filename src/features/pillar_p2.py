"""
Pillar 2: Network & Counterparty Risk
Features capturing transaction network structure and counterparty risk profiles.
"""

import numpy as np
import pandas as pd
import networkx as nx

from src.utils.config import cfg


def build_transaction_graph(
    transactions: pd.DataFrame,
    source_col: str = "client_id",
    target_col: str = "counterparty_id",
    amount_col: str = "amount",
) -> nx.DiGraph:
    """Build directed weighted transaction graph."""
    G = nx.DiGraph()
    for _, row in transactions.iterrows():
        src, tgt = row[source_col], row[target_col]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += row[amount_col]
            G[src][tgt]["count"] += 1
        else:
            G.add_edge(src, tgt, weight=row[amount_col], count=1)
    return G


def compute_p2_features(
    transactions: pd.DataFrame,
    client_id_col: str = "client_id",
    counterparty_col: str = "counterparty_id",
    amount_col: str = "amount",
    direction_col: str = "direction",
    counterparty_country_col: str = "counterparty_country",
) -> pd.DataFrame:
    """
    Compute Network & Counterparty features per client.

    Returns DataFrame indexed by client_id with P2 feature columns.
    """
    df = transactions.copy()
    features = {}

    # Build graph for centrality
    G = build_transaction_graph(df, client_id_col, counterparty_col, amount_col)

    # Centrality measures (computed once for all nodes)
    degree_cent = nx.degree_centrality(G)
    try:
        between_cent = nx.betweenness_centrality(G, k=min(100, len(G)))
    except Exception:
        between_cent = {n: 0.0 for n in G.nodes()}
    try:
        close_cent = nx.closeness_centrality(G)
    except Exception:
        close_cent = {n: 0.0 for n in G.nodes()}
    try:
        pr = nx.pagerank(G, max_iter=100)
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    high_risk = set(cfg.p2.high_risk_jurisdictions)

    for cid, grp in df.groupby(client_id_col):
        counterparties = grp[counterparty_col].unique()
        n_cp = len(counterparties)

        # Direction-based counts
        if direction_col in grp.columns:
            n_inbound = grp[grp[direction_col] == "inbound"][counterparty_col].nunique()
            n_outbound = grp[grp[direction_col] == "outbound"][counterparty_col].nunique()
        else:
            n_inbound = n_cp
            n_outbound = n_cp

        # Counterparty concentration (HHI)
        cp_amounts = grp.groupby(counterparty_col)[amount_col].sum()
        total = cp_amounts.sum()
        hhi = ((cp_amounts / total) ** 2).sum() if total > 0 else 1.0

        # Geographic risk
        if counterparty_country_col in grp.columns:
            countries = grp[counterparty_country_col]
            geo_risk = countries.isin(high_risk).mean() if len(countries) > 0 else 0.0
        else:
            geo_risk = 0.0

        features[cid] = {
            "p2_n_counterparties": n_cp,
            "p2_n_inbound_cp": n_inbound,
            "p2_n_outbound_cp": n_outbound,
            "p2_cp_hhi": hhi,
            "p2_geo_risk_fraction": geo_risk,
            "p2_degree_centrality": degree_cent.get(cid, 0.0),
            "p2_betweenness_centrality": between_cent.get(cid, 0.0),
            "p2_closeness_centrality": close_cent.get(cid, 0.0),
            "p2_pagerank": pr.get(cid, 0.0),
        }

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result
