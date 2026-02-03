"""
Tier 2 Model: P6 Strategic Scheme Alignment
Meta-model consuming Tier 1 blended scores + P6 features.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from typing import Tuple

from src.utils.config import cfg
from src.features.pillar_p6 import compute_p6_features


def build_p6_feature_matrix(
    tier1_blended: pd.DataFrame,
    transactions: pd.DataFrame,
    client_profiles: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Construct the full P6 feature matrix from Tier 1 scores and raw data.

    Parameters
    ----------
    tier1_blended : DataFrame with [S_P1, ..., S_P5] blended pillar scores
    transactions : raw transaction data
    client_profiles : KYC/CDD data

    Returns P6 feature DataFrame.
    """
    return compute_p6_features(tier1_blended, transactions, client_profiles, **kwargs)


def train_p6_model(
    X_p6: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
    n_folds: int = None,
) -> Tuple[list, np.ndarray]:
    """
    Train XGBoost for P6 using out-of-fold Tier 1 scores to prevent leakage.

    Parameters
    ----------
    X_p6 : P6 feature matrix (built from OOF Tier 1 predictions)
    y : binary STR labels

    Returns (models, oof_scores).
    """
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auprc",
            "tree_method": "hist",
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 10,
            "random_state": cfg.model.random_state,
            "verbosity": 0,
        }
    if n_folds is None:
        n_folds = cfg.model.n_folds

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos

    common_idx = X_p6.index.intersection(y.index)
    X = X_p6.loc[common_idx]
    y_aligned = y.loc[common_idx]

    # Drop non-numeric columns (e.g., p6_best_scheme)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cfg.model.random_state
    )
    oof_scores = np.zeros(len(y_aligned))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_aligned)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_aligned.iloc[train_idx], y_aligned.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose_eval=False,
        )

        oof_scores[val_idx] = model.predict(dval)
        models.append(model)

    oof_series = pd.Series(oof_scores, index=common_idx, name="S_P6")
    return models, oof_series


def predict_p6(
    models: list,
    X_p6: pd.DataFrame,
) -> np.ndarray:
    """
    Generate P6 predictions by averaging across fold models.
    """
    numeric_cols = X_p6.select_dtypes(include=[np.number]).columns
    dtest = xgb.DMatrix(X_p6[numeric_cols])
    preds = np.column_stack([m.predict(dtest) for m in models])
    return preds.mean(axis=1)
