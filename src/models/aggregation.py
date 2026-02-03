"""
Score Aggregation Strategies
Three methods to combine pillar sub-scores into a final B-CRS score.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gaussian_kde
from typing import Tuple, Optional

from src.utils.config import cfg


# ---------------------------------------------------------------------------
# Strategy 1: Weighted Linear Combination
# ---------------------------------------------------------------------------

def weighted_linear_aggregation(
    pillar_scores: pd.DataFrame,
    weights: dict = None,
) -> pd.Series:
    """
    Weighted average of pillar sub-scores.

    Parameters
    ----------
    pillar_scores : DataFrame with columns [S_P1, ..., S_P5, S_P6]
    weights : {pillar_col: weight}. If None, equal weights are used.

    Returns Series of final B-CRS scores.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]

    if weights is None:
        weights = {c: 1.0 / len(score_cols) for c in score_cols}

    # Normalize weights to sum to 1
    total = sum(weights.get(c, 0) for c in score_cols)
    if total == 0:
        total = 1.0
    norm_weights = {c: weights.get(c, 0) / total for c in score_cols}

    result = sum(pillar_scores[c] * w for c, w in norm_weights.items())
    result.name = "bcrs_weighted_linear"
    return result


def optimize_linear_weights(
    pillar_scores: pd.DataFrame,
    y: pd.Series,
    n_restarts: int = 50,
) -> dict:
    """
    Find optimal linear weights via random search maximizing AUPRC.
    """
    from sklearn.metrics import average_precision_score

    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    common_idx = pillar_scores.index.intersection(y.index)
    X = pillar_scores.loc[common_idx, score_cols]
    y_aligned = y.loc[common_idx]

    best_score = -1
    best_weights = None
    rng = np.random.RandomState(cfg.model.random_state)

    for _ in range(n_restarts):
        raw_w = rng.dirichlet(np.ones(len(score_cols)))
        w = dict(zip(score_cols, raw_w))
        combined = sum(X[c] * w[c] for c in score_cols)
        ap = average_precision_score(y_aligned, combined)
        if ap > best_score:
            best_score = ap
            best_weights = w

    return best_weights


# ---------------------------------------------------------------------------
# Strategy 2: Gradient Boosting Aggregation
# ---------------------------------------------------------------------------

def gradient_boosting_aggregation(
    pillar_scores: pd.DataFrame,
    y: pd.Series,
    n_folds: int = None,
) -> Tuple[list, pd.Series]:
    """
    Train a lightweight XGBoost on pillar sub-scores as meta-features.

    Parameters
    ----------
    pillar_scores : DataFrame with S_P1..S_P6 columns
    y : binary labels

    Returns (models, oof_predictions).
    """
    if n_folds is None:
        n_folds = cfg.model.n_folds

    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    common_idx = pillar_scores.index.intersection(y.index)
    X = pillar_scores.loc[common_idx, score_cols]
    y_aligned = y.loc[common_idx]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auprc",
        "tree_method": "hist",
        "max_depth": 3,  # shallow — only combining 6 features
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "min_child_weight": 10,
        "random_state": cfg.model.random_state,
        "verbosity": 0,
    }

    n_pos = y_aligned.sum()
    n_neg = len(y_aligned) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cfg.model.random_state
    )
    oof = np.zeros(len(y_aligned))
    models = []

    for train_idx, val_idx in skf.split(X, y_aligned):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y_aligned.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y_aligned.iloc[val_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        oof[val_idx] = model.predict(dval)
        models.append(model)

    oof_series = pd.Series(oof, index=common_idx, name="bcrs_gbm")
    return models, oof_series


def predict_gbm_aggregation(
    models: list,
    pillar_scores: pd.DataFrame,
) -> np.ndarray:
    """Predict final score using GBM aggregation models."""
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    dtest = xgb.DMatrix(pillar_scores[score_cols])
    preds = np.column_stack([m.predict(dtest) for m in models])
    return preds.mean(axis=1)


# ---------------------------------------------------------------------------
# Strategy 3: Bayesian Posterior Aggregation
# ---------------------------------------------------------------------------

def bayesian_aggregation(
    pillar_scores: pd.DataFrame,
    y: pd.Series,
    bandwidth: str = None,
) -> pd.Series:
    """
    Bayesian posterior probability aggregation using kernel density estimation.

    P(STR | S_P1, ..., S_P6) ∝ P(S_P1, ..., S_P6 | STR) * P(STR)

    Estimates class-conditional densities via Gaussian KDE on pillar scores.

    Parameters
    ----------
    pillar_scores : DataFrame with S_P1..S_P6 columns
    y : binary labels
    bandwidth : KDE bandwidth method

    Returns Series of posterior probabilities.
    """
    if bandwidth is None:
        bandwidth = cfg.aggregation.bayesian_kde_bandwidth

    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    common_idx = pillar_scores.index.intersection(y.index)
    X = pillar_scores.loc[common_idx, score_cols].values
    y_aligned = y.loc[common_idx].values

    # Prior
    prior_pos = y_aligned.mean()
    prior_neg = 1 - prior_pos

    # Class-conditional KDEs
    X_pos = X[y_aligned == 1].T
    X_neg = X[y_aligned == 0].T

    if X_pos.shape[1] < 2 or X_neg.shape[1] < 2:
        # Not enough data for KDE
        return pd.Series(0.5, index=common_idx, name="bcrs_bayesian")

    try:
        kde_pos = gaussian_kde(X_pos, bw_method=bandwidth)
        kde_neg = gaussian_kde(X_neg, bw_method=bandwidth)
    except np.linalg.LinAlgError:
        return pd.Series(0.5, index=common_idx, name="bcrs_bayesian")

    # Evaluate on all clients
    all_X = pillar_scores[score_cols].values.T
    log_lik_pos = kde_pos.logpdf(all_X)
    log_lik_neg = kde_neg.logpdf(all_X)

    # Log posterior (unnormalized)
    log_post_pos = log_lik_pos + np.log(prior_pos + 1e-10)
    log_post_neg = log_lik_neg + np.log(prior_neg + 1e-10)

    # Softmax normalization
    max_log = np.maximum(log_post_pos, log_post_neg)
    posterior = np.exp(log_post_pos - max_log) / (
        np.exp(log_post_pos - max_log) + np.exp(log_post_neg - max_log)
    )

    result = pd.Series(posterior, index=pillar_scores.index, name="bcrs_bayesian")
    return result


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def aggregate_scores(
    pillar_scores: pd.DataFrame,
    y: pd.Series = None,
    strategy: str = "weighted_linear",
    **kwargs,
) -> pd.Series:
    """
    Aggregate pillar sub-scores using the specified strategy.

    Parameters
    ----------
    pillar_scores : DataFrame with S_P1..S_P6 columns
    y : binary labels (required for gbm and bayesian)
    strategy : one of "weighted_linear", "gradient_boosting", "bayesian"

    Returns Series of final B-CRS scores.
    """
    if strategy == "weighted_linear":
        return weighted_linear_aggregation(pillar_scores, **kwargs)
    elif strategy == "gradient_boosting":
        if y is None:
            raise ValueError("Labels required for gradient_boosting aggregation.")
        _, oof = gradient_boosting_aggregation(pillar_scores, y, **kwargs)
        return oof
    elif strategy == "bayesian":
        if y is None:
            raise ValueError("Labels required for bayesian aggregation.")
        return bayesian_aggregation(pillar_scores, y, **kwargs)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
