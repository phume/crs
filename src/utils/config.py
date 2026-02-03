"""
B-CRS Configuration
Central configuration for the Behavioral Client Risk Score framework.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Observation Window
# ---------------------------------------------------------------------------
@dataclass
class WindowConfig:
    """Temporal observation window settings."""
    window_months: int = 6
    recompute_interval_months: int = 1  # rolling recompute frequency
    label_horizon_months: int = 6       # STR look-ahead for labelling


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
@dataclass
class TSFreshConfig:
    """TSFresh feature extraction settings."""
    fc_parameters: str = "ComprehensiveFCParameters"  # or MinimalFCParameters
    n_jobs: int = 4
    chunksize: int = 20
    fdr_level: float = 0.05  # Benjamini-Yekutieli significance level


# ---------------------------------------------------------------------------
# Pillar Feature Configs
# ---------------------------------------------------------------------------
@dataclass
class P1Config:
    """Volume & Velocity Risk."""
    periods: list = field(default_factory=lambda: ["D", "W", "M"])
    baseline_window_months: int = 12  # historical baseline for z-scores
    zscore_threshold: float = 2.0


@dataclass
class P2Config:
    """Network & Counterparty Risk."""
    centrality_measures: list = field(
        default_factory=lambda: ["degree", "betweenness", "closeness", "pagerank"]
    )
    community_algorithm: str = "louvain"
    high_risk_jurisdictions: list = field(default_factory=list)  # populated from data


@dataclass
class P3Config:
    """Temporal Behavior Risk."""
    autocorrelation_lags: list = field(default_factory=lambda: [1, 7, 14, 30])
    time_bins: int = 6  # bins for time-of-day distribution
    changepoint_method: str = "Pelt"  # ruptures method
    changepoint_penalty: float = 3.0


@dataclass
class P4Config:
    """Typology Alignment Risk."""
    reporting_threshold: float = 10_000.0  # local currency threshold
    threshold_proximity_pct: float = 0.10  # 10% below threshold
    rapid_throughput_hours: int = 72       # fund-through window
    round_number_tolerance: float = 100.0


@dataclass
class P5Config:
    """Profile Consistency Risk."""
    peer_group_features: list = field(
        default_factory=lambda: ["industry", "size_bucket", "geography"]
    )
    drift_window_months: int = 3  # compare current vs prior window
    dormancy_threshold_days: int = 90


@dataclass
class P6Config:
    """Strategic Scheme Alignment Risk (Tier 2)."""
    co_elevation_threshold: float = 0.7   # pillar score threshold for co-elevation count
    scheme_templates: dict = field(default_factory=lambda: {
        "underground_banking": {
            "description": "FINTRAC Project ATHENA - Underground Banking / IVTS",
            "signature": {"P1": 0.6, "P2": 0.9, "P3": 0.4, "P4": 0.8, "P5": 0.9},
            "weight": {"P1": 0.10, "P2": 0.30, "P3": 0.05, "P4": 0.25, "P5": 0.30},
        },
        "trade_based_ml": {
            "description": "FINTRAC - Trade-Based Money Laundering",
            "signature": {"P1": 0.5, "P2": 0.8, "P3": 0.3, "P4": 0.9, "P5": 0.7},
            "weight": {"P1": 0.10, "P2": 0.25, "P3": 0.05, "P4": 0.35, "P5": 0.25},
        },
        "professional_ml": {
            "description": "FINTRAC - Professional Money Laundering via MSBs",
            "signature": {"P1": 0.8, "P2": 0.9, "P3": 0.7, "P4": 0.6, "P5": 0.8},
            "weight": {"P1": 0.20, "P2": 0.30, "P3": 0.15, "P4": 0.10, "P5": 0.25},
        },
    })
    flagged_corridors: list = field(
        default_factory=lambda: [
            ("CN", "CA"), ("HK", "CA"), ("CN", "US"), ("HK", "US"),
        ]
    )
    high_risk_sectors: list = field(
        default_factory=lambda: [
            "real_estate", "automotive", "legal", "securities", "msb",
        ]
    )
    high_risk_occupations: list = field(
        default_factory=lambda: ["student", "homemaker", "unemployed", "retired"]
    )


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Model training settings."""
    # Supervised
    supervised_algorithm: str = "xgboost"
    n_folds: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42

    # Unsupervised
    unsupervised_algorithm: str = "isolation_forest"
    contamination: Optional[float] = None  # auto-estimated from STR prevalence

    # Blending
    default_alpha: float = 0.7  # supervised weight in pillar blending

    # Hyperparameter search
    optuna_n_trials: int = 100
    optuna_timeout: int = 3600  # seconds


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
@dataclass
class AggregationConfig:
    """Aggregation strategy settings."""
    strategies: list = field(
        default_factory=lambda: ["weighted_linear", "gradient_boosting", "bayesian"]
    )
    bayesian_kde_bandwidth: str = "silverman"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Evaluation settings."""
    top_k_pcts: list = field(default_factory=lambda: [1, 5, 10, 20])
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    correlation_threshold_high: float = 0.8  # pillar independence check
    correlation_threshold_low: float = 0.1


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------
@dataclass
class BCRSConfig:
    """Master configuration aggregating all sub-configs."""
    window: WindowConfig = field(default_factory=WindowConfig)
    tsfresh: TSFreshConfig = field(default_factory=TSFreshConfig)
    p1: P1Config = field(default_factory=P1Config)
    p2: P2Config = field(default_factory=P2Config)
    p3: P3Config = field(default_factory=P3Config)
    p4: P4Config = field(default_factory=P4Config)
    p5: P5Config = field(default_factory=P5Config)
    p6: P6Config = field(default_factory=P6Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


# Singleton default config
cfg = BCRSConfig()
