"""
Baseline Models
Reference models for comparative evaluation against B-CRS.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from typing import Tuple

from src.utils.config import cfg


def flat_xgboost_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = None,
) -> Tuple[list, np.ndarray]:
    """
    Flat XGBoost: all features concatenated, no pillar structure.

    This is the primary baseline — same features as B-CRS but without
    hierarchical decomposition or pillar-level scoring.

    Returns (models, oof_predictions).
    """
    if n_folds is None:
        n_folds = cfg.model.n_folds

    # Drop non-numeric
    X_num = X.select_dtypes(include=[np.number])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auprc",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": cfg.model.random_state,
        "verbosity": 0,
    }

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cfg.model.random_state
    )
    oof = np.zeros(len(y))
    models = []

    for train_idx, val_idx in skf.split(X_num, y):
        dtrain = xgb.DMatrix(X_num.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X_num.iloc[val_idx], label=y.iloc[val_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose_eval=False,
        )
        oof[val_idx] = model.predict(dval)
        models.append(model)

    return models, oof


def static_kyc_baseline(
    client_profiles: pd.DataFrame,
    y: pd.Series,
    client_id_col: str = "client_id",
    risk_features: list = None,
    n_folds: int = None,
) -> Tuple[list, np.ndarray]:
    """
    Static KYC baseline: model trained on profile features only (no transactions).

    This replicates traditional CRS approaches that rely solely on KYC data.
    """
    if n_folds is None:
        n_folds = cfg.model.n_folds

    profiles = (
        client_profiles.set_index(client_id_col)
        if client_id_col in client_profiles.columns
        else client_profiles
    )

    if risk_features is None:
        risk_features = [
            "risk_rating",
            "pep_flag",
            "sanctions_flag",
            "country_risk_score",
            "industry_risk_score",
            "account_age_months",
        ]

    available = [c for c in risk_features if c in profiles.columns]
    if not available:
        raise ValueError("No KYC risk features found in client_profiles.")

    common_idx = profiles.index.intersection(y.index)
    X = profiles.loc[common_idx, available]
    y_aligned = y.loc[common_idx]

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auprc",
        "tree_method": "hist",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
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

    return models, oof


def volume_only_baseline(
    transactions: pd.DataFrame,
    y: pd.Series,
    client_id_col: str = "client_id",
    amount_col: str = "amount",
) -> pd.Series:
    """
    Naive volume-based baseline: rank clients by total transaction volume.

    Returns normalized volume scores as risk predictions.
    """
    volume = transactions.groupby(client_id_col)[amount_col].sum()
    common_idx = volume.index.intersection(y.index)
    scores = volume.loc[common_idx]

    # Normalize to [0, 1]
    min_v, max_v = scores.min(), scores.max()
    if max_v - min_v > 0:
        scores = (scores - min_v) / (max_v - min_v)
    else:
        scores = pd.Series(0.5, index=common_idx)

    scores.name = "baseline_volume"
    return scores
