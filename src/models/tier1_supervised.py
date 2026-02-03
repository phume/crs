"""
Tier 1 Supervised Models
Per-pillar XGBoost classifiers trained on STR labels with out-of-fold predictions.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Tuple, Optional
import optuna

from src.utils.config import cfg


def _default_xgb_params() -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auprc",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 1.0,
        "random_state": cfg.model.random_state,
        "verbosity": 0,
    }


def train_pillar_model(
    X: pd.DataFrame,
    y: pd.Series,
    pillar_name: str,
    params: dict = None,
    n_folds: int = None,
) -> Tuple[list, np.ndarray]:
    """
    Train XGBoost for a single pillar using stratified K-fold.

    Parameters
    ----------
    X : feature matrix for this pillar
    y : binary STR labels
    pillar_name : identifier (e.g., "P1")
    params : XGBoost parameters (defaults used if None)
    n_folds : number of CV folds

    Returns
    -------
    (models, oof_scores) : list of trained models and out-of-fold predictions.
    """
    if params is None:
        params = _default_xgb_params()
    if n_folds is None:
        n_folds = cfg.model.n_folds

    # Adjust scale_pos_weight for class imbalance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cfg.model.random_state
    )
    oof_scores = np.zeros(len(y))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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

    return models, oof_scores


def predict_pillar(
    models: list,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions by averaging across fold models.
    """
    dtest = xgb.DMatrix(X)
    preds = np.column_stack([m.predict(dtest) for m in models])
    return preds.mean(axis=1)


def train_all_tier1(
    pillar_features: Dict[str, pd.DataFrame],
    y: pd.Series,
    params: dict = None,
) -> Tuple[Dict[str, list], pd.DataFrame]:
    """
    Train supervised models for all Tier 1 pillars (P1-P5).

    Parameters
    ----------
    pillar_features : {pillar_name: feature_df}
    y : binary labels aligned with feature indices

    Returns
    -------
    (pillar_models, oof_scores_df) : models per pillar and OOF score DataFrame.
    """
    pillar_models = {}
    oof_df = pd.DataFrame(index=y.index)

    for pillar_name, X in pillar_features.items():
        common_idx = X.index.intersection(y.index)
        models, oof = train_pillar_model(
            X.loc[common_idx], y.loc[common_idx], pillar_name, params
        )
        pillar_models[pillar_name] = models
        oof_series = pd.Series(oof, index=common_idx, name=f"S_{pillar_name}")
        oof_df = oof_df.join(oof_series, how="left")

    return pillar_models, oof_df.fillna(0.0)


def optuna_tune_pillar(
    X: pd.DataFrame,
    y: pd.Series,
    pillar_name: str,
    n_trials: int = None,
    timeout: int = None,
) -> dict:
    """
    Hyperparameter tuning for a pillar model using Optuna.

    Returns best parameters dict.
    """
    if n_trials is None:
        n_trials = cfg.model.optuna_n_trials
    if timeout is None:
        timeout = cfg.model.optuna_timeout

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auprc",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": cfg.model.random_state,
            "verbosity": 0,
        }

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

        skf = StratifiedKFold(
            n_splits=cfg.model.n_folds,
            shuffle=True,
            random_state=cfg.model.random_state,
        )

        scores = []
        for train_idx, val_idx in skf.split(X, y):
            dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=cfg.model.early_stopping_rounds,
                verbose_eval=False,
            )
            preds = model.predict(dval)
            from sklearn.metrics import average_precision_score

            scores.append(average_precision_score(y.iloc[val_idx], preds))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params
