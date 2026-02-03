"""
Ablation Studies
Pillar ablation analysis and P6 contribution measurement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from itertools import combinations

from src.evaluation.metrics import compute_all_metrics
from src.models.aggregation import aggregate_scores
from src.utils.config import cfg


def leave_one_pillar_out(
    pillar_scores: pd.DataFrame,
    y: pd.Series,
    strategy: str = "weighted_linear",
) -> pd.DataFrame:
    """
    Leave-one-pillar-out ablation study.

    For each pillar, remove it and re-aggregate the remaining scores.
    Measure the impact on AUROC, AUPRC, and P@K.

    Returns DataFrame with ablation results.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    common_idx = pillar_scores.index.intersection(y.index)
    y_aligned = y.loc[common_idx]

    # Full model baseline
    full_scores = aggregate_scores(
        pillar_scores.loc[common_idx], y_aligned, strategy=strategy
    )
    full_metrics = compute_all_metrics(y_aligned.values, full_scores.values)

    results = [{"ablation": "full_model", **full_metrics}]

    for drop_col in score_cols:
        remaining = [c for c in score_cols if c != drop_col]
        reduced_df = pillar_scores.loc[common_idx, remaining]
        reduced_scores = aggregate_scores(reduced_df, y_aligned, strategy=strategy)
        reduced_metrics = compute_all_metrics(y_aligned.values, reduced_scores.values)

        # Delta from full model
        row = {"ablation": f"drop_{drop_col}"}
        for metric_name, value in reduced_metrics.items():
            row[metric_name] = value
            row[f"delta_{metric_name}"] = full_metrics.get(metric_name, 0) - value
        results.append(row)

    return pd.DataFrame(results).set_index("ablation")


def p6_contribution_analysis(
    pillar_scores_with_p6: pd.DataFrame,
    pillar_scores_without_p6: pd.DataFrame,
    y: pd.Series,
    strategy: str = "weighted_linear",
) -> Dict[str, float]:
    """
    Measure the marginal contribution of P6 (Tier 2 meta-pillar).

    Compares B-CRS(P1-P5) vs B-CRS(P1-P6) across all metrics.

    Returns dict of metric deltas.
    """
    common_idx = (
        pillar_scores_with_p6.index
        .intersection(pillar_scores_without_p6.index)
        .intersection(y.index)
    )
    y_aligned = y.loc[common_idx]

    scores_with = aggregate_scores(
        pillar_scores_with_p6.loc[common_idx], y_aligned, strategy=strategy
    )
    scores_without = aggregate_scores(
        pillar_scores_without_p6.loc[common_idx], y_aligned, strategy=strategy
    )

    metrics_with = compute_all_metrics(y_aligned.values, scores_with.values)
    metrics_without = compute_all_metrics(y_aligned.values, scores_without.values)

    deltas = {}
    for metric_name in metrics_with:
        deltas[f"{metric_name}_with_p6"] = metrics_with[metric_name]
        deltas[f"{metric_name}_without_p6"] = metrics_without[metric_name]
        deltas[f"delta_{metric_name}"] = (
            metrics_with[metric_name] - metrics_without[metric_name]
        )

    return deltas


def pillar_combination_sweep(
    pillar_scores: pd.DataFrame,
    y: pd.Series,
    strategy: str = "weighted_linear",
    min_pillars: int = 1,
) -> pd.DataFrame:
    """
    Exhaustive evaluation of all pillar combinations.

    Useful for demonstrating that the full set provides the best performance
    and for understanding pillar synergies.

    Returns DataFrame sorted by AUPRC descending.
    """
    score_cols = [c for c in pillar_scores.columns if c.startswith("S_P")]
    common_idx = pillar_scores.index.intersection(y.index)
    y_aligned = y.loc[common_idx]

    results = []
    for r in range(min_pillars, len(score_cols) + 1):
        for combo in combinations(score_cols, r):
            subset = pillar_scores.loc[common_idx, list(combo)]
            scores = aggregate_scores(subset, y_aligned, strategy=strategy)
            metrics = compute_all_metrics(y_aligned.values, scores.values)
            results.append({
                "pillars": "+".join(combo),
                "n_pillars": len(combo),
                **metrics,
            })

    result_df = pd.DataFrame(results)
    return result_df.sort_values("auprc", ascending=False).reset_index(drop=True)
