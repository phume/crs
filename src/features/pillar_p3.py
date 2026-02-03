"""
Pillar 3: Temporal Behavior Risk
Features capturing timing patterns, periodicity, entropy, and regime changes.
"""

import numpy as np
import pandas as pd

from src.utils.config import cfg


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a distribution."""
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0


def _approx_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate entropy (simplified)."""
    if len(series) < m + 1:
        return 0.0
    r = r_factor * np.std(series)
    if r == 0:
        return 0.0

    def phi(m_):
        n = len(series)
        templates = np.array([series[i: i + m_] for i in range(n - m_ + 1)])
        counts = np.zeros(len(templates))
        for i, t in enumerate(templates):
            counts[i] = np.sum(np.max(np.abs(templates - t), axis=1) <= r)
        return np.sum(np.log(counts / (n - m_ + 1))) / (n - m_ + 1)

    return abs(phi(m) - phi(m + 1))


def compute_p3_features(
    transactions: pd.DataFrame,
    client_id_col: str = "client_id",
    date_col: str = "txn_date",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Compute Temporal Behavior features per client.

    Returns DataFrame indexed by client_id with P3 feature columns.
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    features = {}

    for cid, grp in df.groupby(client_id_col):
        dates = grp[date_col].sort_values()

        # Time-of-day distribution entropy
        hours = dates.dt.hour
        n_bins = cfg.p3.time_bins
        hour_bins = np.histogram(hours, bins=n_bins, range=(0, 24))[0]
        tod_entropy = _entropy(hour_bins)

        # Day-of-week distribution entropy
        dow = dates.dt.dayofweek
        dow_counts = np.bincount(dow, minlength=7)
        dow_entropy = _entropy(dow_counts)

        # Weekend ratio
        weekend_ratio = (dow >= 5).mean()

        feat = {
            "p3_tod_entropy": tod_entropy,
            "p3_dow_entropy": dow_entropy,
            "p3_weekend_ratio": weekend_ratio,
        }

        # Autocorrelation on daily amount series
        daily = grp.set_index(date_col).resample("D")[amount_col].sum().fillna(0)
        if len(daily) > max(cfg.p3.autocorrelation_lags) + 1:
            for lag in cfg.p3.autocorrelation_lags:
                ac = daily.autocorr(lag=lag)
                feat[f"p3_autocorr_lag{lag}"] = ac if not np.isnan(ac) else 0.0
        else:
            for lag in cfg.p3.autocorrelation_lags:
                feat[f"p3_autocorr_lag{lag}"] = 0.0

        # Approximate entropy on daily amounts
        daily_vals = daily.values
        feat["p3_approx_entropy"] = _approx_entropy(daily_vals)

        # Trend strength via linear fit residual ratio
        if len(daily_vals) > 2:
            x = np.arange(len(daily_vals))
            coeffs = np.polyfit(x, daily_vals, 1)
            trend_line = np.polyval(coeffs, x)
            residuals = daily_vals - trend_line
            total_var = np.var(daily_vals)
            feat["p3_trend_strength"] = (
                1 - np.var(residuals) / total_var if total_var > 0 else 0.0
            )
        else:
            feat["p3_trend_strength"] = 0.0

        # Change-point count (simple threshold-based)
        if len(daily_vals) > 10:
            rolling_mean = pd.Series(daily_vals).rolling(7, min_periods=1).mean()
            diffs = rolling_mean.diff().abs()
            threshold = diffs.mean() + 2 * diffs.std() if diffs.std() > 0 else np.inf
            feat["p3_changepoint_count"] = int((diffs > threshold).sum())
        else:
            feat["p3_changepoint_count"] = 0

        features[cid] = feat

    result = pd.DataFrame.from_dict(features, orient="index")
    result.index.name = client_id_col
    return result
