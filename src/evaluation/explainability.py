"""
Explainability Module
Four-level explainability: Pillar, Feature (SHAP), Scheme (P6 match), Temporal.
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from typing import Dict, List, Optional

from src.utils.config import cfg
from src.features.scheme_templates import get_scheme_descriptions


# ---------------------------------------------------------------------------
# Level 1: Pillar-Level Decomposition
# ---------------------------------------------------------------------------

def pillar_decomposition(
    pillar_scores: pd.DataFrame,
    client_id: str = None,
) -> pd.DataFrame:
    """
    Decompose final B-CRS score into pillar-level contributions.

    If client_id is provided, returns a single-row explanation.
    Otherwise returns decomposition for all clients.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]

    if client_id is not None:
        if client_id not in pillar_scores.index:
            raise ValueError(f"Client {client_id} not found.")
        row = pillar_scores.loc[[client_id], score_cols]
    else:
        row = pillar_scores[score_cols]

    # Rank pillars by contribution for each client
    ranked = row.apply(
        lambda r: r.sort_values(ascending=False).index.tolist(), axis=1
    )
    row = row.copy()
    row["pillar_ranking"] = ranked
    return row


def generate_pillar_radar_data(
    pillar_scores: pd.DataFrame,
    client_id: str,
) -> Dict[str, float]:
    """
    Extract pillar scores for a single client in radar chart format.

    Returns {pillar_name: score} dict suitable for matplotlib radar plots.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    if client_id not in pillar_scores.index:
        raise ValueError(f"Client {client_id} not found.")

    return {
        col.replace("S_", ""): pillar_scores.loc[client_id, col]
        for col in score_cols
    }


# ---------------------------------------------------------------------------
# Level 2: Feature-Level SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: xgb.Booster,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SHAP values for XGBoost model predictions.

    Returns DataFrame of SHAP values with same columns as X.
    """
    explainer = shap.TreeExplainer(model)
    dmatrix = xgb.DMatrix(X)
    shap_values = explainer.shap_values(dmatrix)
    return pd.DataFrame(shap_values, index=X.index, columns=X.columns)


def top_features_for_client(
    shap_df: pd.DataFrame,
    client_id: str,
    top_n: int = 10,
) -> pd.Series:
    """
    Return top-N contributing features for a specific client.
    """
    if client_id not in shap_df.index:
        raise ValueError(f"Client {client_id} not found in SHAP values.")
    client_shap = shap_df.loc[client_id].abs().sort_values(ascending=False)
    return client_shap.head(top_n)


def global_feature_importance(
    shap_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.Series:
    """
    Global feature importance based on mean absolute SHAP values.
    """
    return shap_df.abs().mean().sort_values(ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Level 3: Scheme-Level Explanation (P6)
# ---------------------------------------------------------------------------

def scheme_explanation(
    p6_similarity: pd.DataFrame,
    client_id: str,
) -> Dict[str, object]:
    """
    Explain scheme alignment for a specific client.

    Parameters
    ----------
    p6_similarity : DataFrame from compute_scheme_similarity()
    client_id : target client

    Returns dict with best matching scheme, similarity score, and description.
    """
    if client_id not in p6_similarity.index:
        raise ValueError(f"Client {client_id} not found.")

    row = p6_similarity.loc[client_id]
    best_scheme = row.get("p6_best_scheme", "unknown")
    max_sim = row.get("p6_max_scheme_sim", 0.0)

    descriptions = get_scheme_descriptions()

    sim_cols = [c for c in p6_similarity.columns if c.startswith("p6_sim_")]
    all_sims = {
        c.replace("p6_sim_", ""): row[c] for c in sim_cols
    }

    return {
        "client_id": client_id,
        "best_scheme": best_scheme,
        "best_similarity": max_sim,
        "scheme_description": descriptions.get(best_scheme, "Unknown scheme"),
        "all_similarities": all_sims,
    }


# ---------------------------------------------------------------------------
# Level 4: Temporal Explanation
# ---------------------------------------------------------------------------

def temporal_risk_trajectory(
    pillar_scores_over_time: Dict[str, pd.DataFrame],
    client_id: str,
) -> pd.DataFrame:
    """
    Build a time series of pillar scores for a client across windows.

    Parameters
    ----------
    pillar_scores_over_time : {window_label: pillar_scores_df}
    client_id : target client

    Returns DataFrame with window labels as rows and pillar scores as columns.
    """
    rows = []
    for window_label, scores in pillar_scores_over_time.items():
        if client_id in scores.index:
            score_cols = [c for c in scores.columns if c.startswith("S_P")]
            row = scores.loc[client_id, score_cols].to_dict()
            row["window"] = window_label
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("window")


def detect_risk_escalation(
    trajectory: pd.DataFrame,
    threshold: float = 0.15,
) -> Dict[str, object]:
    """
    Detect significant risk escalation across observation windows.

    Returns dict with escalating pillars and magnitude.
    """
    if len(trajectory) < 2:
        return {"escalating_pillars": [], "max_delta": 0.0}

    first = trajectory.iloc[0]
    last = trajectory.iloc[-1]
    deltas = last - first

    escalating = deltas[deltas > threshold].sort_values(ascending=False)
    return {
        "escalating_pillars": escalating.index.tolist(),
        "deltas": escalating.to_dict(),
        "max_delta": escalating.max() if len(escalating) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Narrative Generation
# ---------------------------------------------------------------------------

def generate_client_narrative(
    client_id: str,
    pillar_scores: pd.DataFrame,
    shap_df: pd.DataFrame = None,
    p6_similarity: pd.DataFrame = None,
    top_n_features: int = 5,
) -> str:
    """
    Generate a human-readable risk narrative for a client.

    Combines all four explainability levels into a text summary.
    """
    lines = [f"=== Risk Assessment: Client {client_id} ===\n"]

    # Level 1: Pillar decomposition
    radar = generate_pillar_radar_data(pillar_scores, client_id)
    pillar_names = {
        "P1": "Volume & Velocity",
        "P2": "Network & Counterparty",
        "P3": "Temporal Behavior",
        "P4": "Typology Alignment",
        "P5": "Profile Consistency",
        "P6": "Strategic Scheme Alignment",
    }
    lines.append("PILLAR SCORES:")
    for pillar, score in sorted(radar.items(), key=lambda x: -x[1]):
        name = pillar_names.get(pillar, pillar)
        bar = "█" * int(score * 20)
        lines.append(f"  {pillar} ({name}): {score:.3f}  {bar}")
    lines.append("")

    # Level 2: Top SHAP features
    if shap_df is not None and client_id in shap_df.index:
        top_feats = top_features_for_client(shap_df, client_id, top_n_features)
        lines.append("TOP RISK DRIVERS:")
        for feat, val in top_feats.items():
            lines.append(f"  {feat}: SHAP={val:.4f}")
        lines.append("")

    # Level 3: Scheme alignment
    if p6_similarity is not None and client_id in p6_similarity.index:
        scheme_info = scheme_explanation(p6_similarity, client_id)
        lines.append("SCHEME ALIGNMENT:")
        lines.append(f"  Best match: {scheme_info['best_scheme']} "
                     f"(similarity={scheme_info['best_similarity']:.3f})")
        lines.append(f"  Description: {scheme_info['scheme_description']}")
        lines.append("")

    return "\n".join(lines)
