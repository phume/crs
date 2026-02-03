"""
Evaluation Metrics
AUROC, AUPRC, Brier score, Precision@K, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, Tuple

from src.utils.config import cfg


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k_pcts: list = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a model.

    Parameters
    ----------
    y_true : binary labels
    y_score : predicted risk scores [0, 1]
    top_k_pcts : percentages for Precision@K

    Returns dict of metric_name -> value.
    """
    if top_k_pcts is None:
        top_k_pcts = cfg.evaluation.top_k_pcts

    metrics = {}

    # AUROC
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auroc"] = np.nan

    # AUPRC
    try:
        metrics["auprc"] = average_precision_score(y_true, y_score)
    except ValueError:
        metrics["auprc"] = np.nan

    # Brier score (lower is better)
    metrics["brier_score"] = brier_score_loss(y_true, y_score)

    # Precision@K
    for k_pct in top_k_pcts:
        metrics[f"precision_at_{k_pct}pct"] = precision_at_k(y_true, y_score, k_pct)

    # Recall@K
    for k_pct in top_k_pcts:
        metrics[f"recall_at_{k_pct}pct"] = recall_at_k(y_true, y_score, k_pct)

    return metrics


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_pct: float,
) -> float:
    """
    Precision at top-K percent of ranked clients.

    Parameters
    ----------
    y_true : binary labels
    y_score : risk scores
    k_pct : percentage of clients to consider (e.g., 5 for top 5%)
    """
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return np.mean(np.array(y_true)[top_k_idx])


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_pct: float,
) -> float:
    """
    Recall at top-K percent: fraction of positives captured in top-K.
    """
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0
    return np.sum(np.array(y_true)[top_k_idx]) / n_pos


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = None,
    ci: float = None,
    random_state: int = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if n_bootstrap is None:
        n_bootstrap = cfg.evaluation.bootstrap_n
    if ci is None:
        ci = cfg.evaluation.bootstrap_ci
    if random_state is None:
        random_state = cfg.model.random_state

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point_est = metric_fn(y_true, y_score)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = np.array(y_true)[idx]
        y_s = np.array(y_score)[idx]
        # Skip if single class in bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue
        try:
            boot_scores.append(metric_fn(y_t, y_s))
        except ValueError:
            continue

    if not boot_scores:
        return point_est, np.nan, np.nan

    alpha = 1 - ci
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return point_est, lower, upper


def compare_models(
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    top_k_pcts: list = None,
) -> pd.DataFrame:
    """
    Compare multiple models across all metrics.

    Parameters
    ----------
    y_true : binary labels
    model_scores : {model_name: predicted_scores}

    Returns DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, scores in model_scores.items():
        metrics = compute_all_metrics(y_true, scores, top_k_pcts)
        metrics["model"] = model_name
        rows.append(metrics)

    result = pd.DataFrame(rows).set_index("model")
    return result


def compute_pillar_correlations(
    pillar_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations between pillar sub-scores.

    Used to verify pillar independence / complementarity.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    return pillar_scores[score_cols].corr()
