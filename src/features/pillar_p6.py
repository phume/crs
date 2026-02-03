"""
Pillar 6: Strategic Scheme Alignment Risk (Tier 2)
Meta-pillar consuming Tier 1 sub-scores and scheme-specific features to
detect multi-dimensional laundering schemes defined by FIU operational alerts.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from src.utils.config import cfg
from src.features.scheme_templates import get_scheme_signatures


def compute_tier1_interaction_features(
    tier1_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute interaction features from Tier 1 sub-scores.

    Parameters
    ----------
    tier1_scores : DataFrame with columns [S_P1, S_P2, S_P3, S_P4, S_P5]

    Returns DataFrame with pairwise interactions and summary stats.
    """
    pillar_cols = ["S_P1", "S_P2", "S_P3", "S_P4", "S_P5"]
    df = tier1_scores[pillar_cols].copy()

    interactions = pd.DataFrame(index=df.index)

    # Pairwise products
    for i, c1 in enumerate(pillar_cols):
        for c2 in pillar_cols[i + 1:]:
            col_name = f"p6_inter_{c1}x{c2}"
            interactions[col_name] = df[c1] * df[c2]

    # Higher-order: P2 x P4 x P5 (network + typology + profile)
    interactions["p6_inter_P2xP4xP5"] = df["S_P2"] * df["S_P4"] * df["S_P5"]

    # Co-elevation count
    threshold = cfg.p6.co_elevation_threshold
    interactions["p6_co_elevation_count"] = (df >= threshold).sum(axis=1)

    # Summary stats across pillars
    interactions["p6_pillar_max"] = df.max(axis=1)
    interactions["p6_pillar_mean"] = df.mean(axis=1)
    interactions["p6_pillar_std"] = df.std(axis=1)

    return interactions


def compute_scheme_similarity(
    tier1_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cosine similarity between each client's pillar vector
    and pre-defined scheme signatures.

    Returns DataFrame with similarity score per scheme and best match.
    """
    pillar_cols = ["S_P1", "S_P2", "S_P3", "S_P4", "S_P5"]
    signatures = get_scheme_signatures()

    results = pd.DataFrame(index=tier1_scores.index)

    for scheme_name, sig_vector in signatures.items():
        sig_arr = np.array([sig_vector[c] for c in pillar_cols])

        sims = []
        for idx in tier1_scores.index:
            client_vec = tier1_scores.loc[idx, pillar_cols].values.astype(float)
            if np.linalg.norm(client_vec) == 0 or np.linalg.norm(sig_arr) == 0:
                sims.append(0.0)
            else:
                sims.append(1.0 - cosine(client_vec, sig_arr))
        results[f"p6_sim_{scheme_name}"] = sims

    results["p6_max_scheme_sim"] = results.max(axis=1)
    results["p6_best_scheme"] = results.drop(
        columns=["p6_max_scheme_sim"]
    ).idxmax(axis=1).str.replace("p6_sim_", "")

    return results


def compute_scheme_specific_features(
    transactions: pd.DataFrame,
    client_profiles: pd.DataFrame,
    client_id_col: str = "client_id",
    amount_col: str = "amount",
    date_col: str = "txn_date",
    direction_col: str = "direction",
    counterparty_country_col: str = "counterparty_country",
    counterparty_sector_col: str = "counterparty_sector",
    occupation_col: str = "occupation",
) -> pd.DataFrame:
    """
    Compute scheme-specific features grounded in FINTRAC operational alerts.
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    profiles = (
        client_profiles.set_index(client_id_col)
        if client_id_col in client_profiles.columns
        else client_profiles
    )

    flagged_corridors = set(cfg.p6.flagged_corridors)
    high_risk_sectors = set(cfg.p6.high_risk_sectors)
    high_risk_occs = {o.lower() for o in cfg.p6.high_risk_occupations}

    features = {}

    for cid, grp in df.groupby(client_id_col):
        feat = {}
        amounts = grp[amount_col]
        n_txn = len(grp)

        # --- Flow symmetry ---
        if direction_col in grp.columns:
            inbound = grp.loc[grp[direction_col] == "inbound", amount_col].sum()
            outbound = grp.loc[grp[direction_col] == "outbound", amount_col].sum()
            max_flow = max(inbound, outbound)
            feat["p6_flow_symmetry"] = (
                1.0 - abs(inbound - outbound) / max_flow if max_flow > 0 else 0.0
            )
        else:
            feat["p6_flow_symmetry"] = 0.0
            inbound, outbound = 0.0, 0.0

        # --- Geographic corridor concentration ---
        if counterparty_country_col in grp.columns:
            countries = grp[counterparty_country_col].dropna()
            if len(countries) > 0:
                # Corridor flag fraction
                # Simplified: check if counterparty country is in flagged list
                flagged_countries = {c[0] for c in flagged_corridors} | {c[1] for c in flagged_corridors}
                feat["p6_flagged_corridor_ratio"] = countries.isin(flagged_countries).mean()
                # HHI of country distribution
                country_counts = countries.value_counts(normalize=True)
                feat["p6_corridor_hhi"] = (country_counts ** 2).sum()
            else:
                feat["p6_flagged_corridor_ratio"] = 0.0
                feat["p6_corridor_hhi"] = 0.0
        else:
            feat["p6_flagged_corridor_ratio"] = 0.0
            feat["p6_corridor_hhi"] = 0.0

        # --- Sector destination patterns ---
        if counterparty_sector_col in grp.columns:
            sectors = grp[counterparty_sector_col].dropna()
            if len(sectors) > 0:
                feat["p6_high_risk_sector_ratio"] = sectors.str.lower().isin(
                    high_risk_sectors
                ).mean()
            else:
                feat["p6_high_risk_sector_ratio"] = 0.0
        else:
            feat["p6_high_risk_sector_ratio"] = 0.0

        # --- Pass-through velocity ---
        if direction_col in grp.columns:
            sorted_txns = grp.sort_values(date_col)
            inbound_times = sorted_txns.loc[sorted_txns[direction_col] == "inbound", date_col]
            outbound_times = sorted_txns.loc[sorted_txns[direction_col] == "outbound", date_col]
            if len(inbound_times) > 0 and len(outbound_times) > 0:
                # For each inbound, find nearest subsequent outbound
                velocities = []
                for in_t in inbound_times:
                    subsequent = outbound_times[outbound_times > in_t]
                    if len(subsequent) > 0:
                        gap_hours = (subsequent.iloc[0] - in_t).total_seconds() / 3600
                        velocities.append(gap_hours)
                feat["p6_passthrough_velocity_median"] = (
                    np.median(velocities) if velocities else np.nan
                )
                feat["p6_passthrough_velocity_min"] = (
                    np.min(velocities) if velocities else np.nan
                )
            else:
                feat["p6_passthrough_velocity_median"] = np.nan
                feat["p6_passthrough_velocity_min"] = np.nan
        else:
            feat["p6_passthrough_velocity_median"] = np.nan
            feat["p6_passthrough_velocity_min"] = np.nan

        # --- Round-figure 50K increments (TBML indicator) ---
        round_50k = (amounts % 50_000 == 0) & (amounts > 0)
        feat["p6_round_50k_ratio"] = round_50k.mean()

        # --- Occupation-activity severity ---
        if occupation_col in profiles.columns and cid in profiles.index:
            occ = str(profiles.loc[cid, occupation_col]).lower().strip()
            is_high_risk_occ = occ in high_risk_occs
            volume = amounts.sum()
            feat["p6_occ_activity_severity"] = (
                np.log1p(volume) if is_high_risk_occ else 0.0
            )
            feat["p6_high_risk_occ_flag"] = int(is_high_risk_occ)
        else:
            feat["p6_occ_activity_severity"] = 0.0
            feat["p6_high_risk_occ_flag"] = 0

        features[cid] = feat

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result


def compute_p6_features(
    tier1_scores: pd.DataFrame,
    transactions: pd.DataFrame,
    client_profiles: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Full P6 feature computation: interactions + scheme similarity + scheme-specific.

    Parameters
    ----------
    tier1_scores : DataFrame with [S_P1, ..., S_P5] per client
    transactions : raw transaction data
    client_profiles : KYC/CDD profile data

    Returns combined P6 feature DataFrame.
    """
    interactions = compute_tier1_interaction_features(tier1_scores)
    similarity = compute_scheme_similarity(tier1_scores)
    scheme_feats = compute_scheme_specific_features(
        transactions, client_profiles, **kwargs
    )

    # Align indexes
    common_idx = interactions.index.intersection(similarity.index).intersection(
        scheme_feats.index
    )
    return pd.concat(
        [
            interactions.loc[common_idx],
            similarity.loc[common_idx],
            scheme_feats.loc[common_idx],
        ],
        axis=1,
    )
