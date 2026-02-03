"""
Tier 1 Unsupervised Models
Per-pillar Isolation Forest anomaly scoring for unsupervised risk component.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

from src.utils.config import cfg


def train_pillar_iforest(
    X: pd.DataFrame,
    pillar_name: str,
    contamination: float = None,
) -> Tuple[IsolationForest, StandardScaler, np.ndarray]:
    """
    Train Isolation Forest for a single pillar.

    Parameters
    ----------
    X : feature matrix for this pillar
    pillar_name : identifier
    contamination : expected anomaly fraction (auto if None)

    Returns
    -------
    (model, scaler, anomaly_scores) : trained model, scaler, and anomaly scores [0, 1].
    """
    if contamination is None:
        contamination = cfg.model.contamination or "auto"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    model = IsolationForest(
        contamination=contamination,
        random_state=cfg.model.random_state,
        n_estimators=200,
        max_samples="auto",
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Convert decision function to [0, 1] anomaly score
    # decision_function: lower = more anomalous
    raw_scores = model.decision_function(X_scaled)
    anomaly_scores = _normalize_anomaly_scores(raw_scores)

    return model, scaler, anomaly_scores


def _normalize_anomaly_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert Isolation Forest decision_function output to [0, 1] risk scores.
    More anomalous (lower raw) -> higher risk score.
    """
    # Negate so higher = more anomalous
    negated = -raw_scores
    min_val = negated.min()
    max_val = negated.max()
    if max_val - min_val == 0:
        return np.zeros_like(negated)
    return (negated - min_val) / (max_val - min_val)


def predict_pillar_iforest(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate anomaly scores for new data using a trained Isolation Forest.
    """
    X_scaled = scaler.transform(X.fillna(0))
    raw_scores = model.decision_function(X_scaled)
    return _normalize_anomaly_scores(raw_scores)


def train_all_tier1_unsupervised(
    pillar_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, Tuple[IsolationForest, StandardScaler]], pd.DataFrame]:
    """
    Train Isolation Forest for all Tier 1 pillars.

    Parameters
    ----------
    pillar_features : {pillar_name: feature_df}

    Returns
    -------
    (pillar_models, anomaly_scores_df) : models and anomaly score DataFrame.
    """
    pillar_models = {}
    scores_df = pd.DataFrame()

    for pillar_name, X in pillar_features.items():
        model, scaler, scores = train_pillar_iforest(X, pillar_name)
        pillar_models[pillar_name] = (model, scaler)

        col_name = f"U_{pillar_name}"
        score_series = pd.Series(scores, index=X.index, name=col_name)
        scores_df = scores_df.join(score_series, how="outer") if len(scores_df) > 0 else pd.DataFrame(score_series)

    return pillar_models, scores_df.fillna(0.0)


def blend_pillar_scores(
    supervised_scores: pd.DataFrame,
    unsupervised_scores: pd.DataFrame,
    alpha: float = None,
) -> pd.DataFrame:
    """
    Blend supervised and unsupervised scores per pillar.

    S_blended = alpha * S_supervised + (1 - alpha) * S_unsupervised

    Parameters
    ----------
    supervised_scores : DataFrame with columns [S_P1, ..., S_P5]
    unsupervised_scores : DataFrame with columns [U_P1, ..., U_P5]
    alpha : blending weight for supervised component

    Returns DataFrame with columns [S_P1, ..., S_P5] (blended).
    """
    if alpha is None:
        alpha = cfg.model.default_alpha

    blended = pd.DataFrame(index=supervised_scores.index)
    pillars = ["P1", "P2", "P3", "P4", "P5"]

    for p in pillars:
        sup_col = f"S_{p}"
        unsup_col = f"U_{p}"
        if sup_col in supervised_scores.columns and unsup_col in unsupervised_scores.columns:
            blended[sup_col] = (
                alpha * supervised_scores[sup_col]
                + (1 - alpha) * unsupervised_scores[unsup_col]
            )
        elif sup_col in supervised_scores.columns:
            blended[sup_col] = supervised_scores[sup_col]
        elif unsup_col in unsupervised_scores.columns:
            blended[sup_col] = unsupervised_scores[unsup_col]

    return blended
