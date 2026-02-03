# B-CRS: Behavioral Client Risk Score for AML

## Project Overview

Research project building a novel **two-tier, six-pillar Behavioral Client Risk Score (B-CRS)** framework for Anti-Money Laundering (AML). The framework moves beyond traditional static KYC-based risk scoring by leveraging client behavioral transaction patterns to dynamically assess money laundering risk. Target publication: Journal of Money Laundering Control (JMLC) or IEEE-tier venue.

**Key innovation:** A hierarchical architecture where Tier 1 (P1–P5) captures independent behavioral dimensions, and Tier 2 (P6) detects multi-dimensional laundering schemes by consuming Tier 1 sub-scores alongside scheme-specific features grounded in FINTRAC/FinCEN operational alerts.

## Architecture

```
                    Tier 1 (Independent Behavioral Pillars)
    ┌──────┬──────┬──────┬──────┬──────┐
    │  P1  │  P2  │  P3  │  P4  │  P5  │
    │Vol & │Net & │Temp  │Typo  │Prof  │
    │Veloc │Cpty  │Behav │Align │Consi │
    └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
       │      │      │      │      │
       └──────┴──────┼──────┴──────┘
                     ▼
              Tier 2 (Meta-Pillar)
            ┌────────────────┐
            │  P6: Strategic │
            │  Scheme Align  │
            └───────┬────────┘
                    ▼
            ┌──────────────┐
            │ Aggregation  │  ← 3 strategies: weighted linear, GB, Bayesian
            │ (S_P1..S_P6) │
            └──────┬───────┘
                   ▼
              Final B-CRS
```

### Pillars

| Pillar | Name | Tier | Description |
|--------|------|------|-------------|
| P1 | Volume & Velocity Risk | 1 | Transaction counts, amounts, z-scores vs historical baseline |
| P2 | Network & Counterparty Risk | 1 | Graph centrality, counterparty concentration (HHI), geographic risk |
| P3 | Temporal Behavior Risk | 1 | Entropy, autocorrelation, change-point detection, periodicity |
| P4 | Typology Alignment Risk | 1 | Structuring indicators, round numbers, rapid fund-through, funnel accounts |
| P5 | Profile Consistency Risk | 1 | Turnover ratio vs declared, peer deviation (Mahalanobis), dormancy |
| P6 | Strategic Scheme Alignment | 2 | Consumes S_P1–S_P5 + scheme-specific features; cosine similarity to FINTRAC scheme templates |

### Scoring Pipeline

1. **Feature Engineering**: Each pillar computes features from raw transactions + client profiles
2. **Tier 1 Scoring**: Per-pillar XGBoost (supervised) + Isolation Forest (unsupervised), blended via `S_Pi = alpha * supervised + (1-alpha) * unsupervised`
3. **Tier 2 P6**: Uses out-of-fold Tier 1 predictions (prevents data leakage) + scheme-specific features
4. **Aggregation**: Combines S_P1–S_P6 into final score via one of three strategies
5. **Explainability**: Four levels — pillar decomposition, SHAP features, scheme alignment, temporal trajectory

## Directory Structure

```
client_Risk_score/
├── paper/
│   ├── bcrs_paper.tex              # IEEEtran conference format manuscript
│   ├── bcrs_paper_v0.4.pdf         # Current compiled PDF
│   ├── archived/                   # Previous PDF versions (v0.1–v0.3)
│   └── paper_log/changelog.md      # Version history
├── notes/
│   └── implementation_plan.md      # 6-phase implementation roadmap
├── data/
│   ├── raw/                        # transactions.csv, client_profiles.csv, str_labels.csv
│   ├── processed/                  # Cleaned datasets
│   └── features/                   # Per-pillar feature stores
├── src/
│   ├── data_prep/
│   │   ├── loader.py               # Load CSV/Parquet data files
│   │   └── preprocessing.py        # Windowing, temporal split, label creation
│   ├── features/
│   │   ├── pillar_p1.py            # 18 Volume & Velocity features
│   │   ├── pillar_p2.py            # 9 Network features (requires networkx)
│   │   ├── pillar_p3.py            # ~12 Temporal features (entropy, autocorrelation)
│   │   ├── pillar_p4.py            # 11 Typology features (structuring, funnel)
│   │   ├── pillar_p5.py            # 8 Profile features + peer deviation (Mahalanobis)
│   │   ├── pillar_p6.py            # Tier 2: interactions + scheme similarity + scheme-specific
│   │   ├── scheme_templates.py     # FINTRAC scheme signatures (underground banking, TBML, professional ML)
│   │   └── tsfresh_engine.py       # TSFresh extraction with relevance filtering
│   ├── models/
│   │   ├── tier1_supervised.py     # XGBoost per-pillar with Optuna tuning, OOF predictions
│   │   ├── tier1_unsupervised.py   # Isolation Forest per-pillar, score normalization, blending
│   │   ├── tier2_p6.py             # P6 meta-model on OOF Tier 1 scores
│   │   ├── aggregation.py          # Weighted linear, gradient boosting, Bayesian KDE
│   │   └── baselines.py            # Flat XGBoost, static KYC, volume-only
│   ├── evaluation/
│   │   ├── metrics.py              # AUROC, AUPRC, Brier, P@k, R@k, bootstrap CI
│   │   ├── ablation.py             # Leave-one-out, P6 contribution, combination sweep
│   │   └── explainability.py       # SHAP, radar charts, scheme narratives, temporal trajectories
│   └── utils/
│       └── config.py               # BCRSConfig dataclass — all parameters centralized
├── notebooks/                      # Planned: 01_eda through 06_results_analysis (not yet created)
├── requirements.txt
└── claude.md
```

## Configuration

All parameters are centralized in `src/utils/config.py` via nested dataclasses under `BCRSConfig`. Access the singleton as:

```python
from src.utils.config import cfg

cfg.model.n_folds          # 5
cfg.p4.reporting_threshold # 10_000.0
cfg.p6.scheme_templates    # dict with 3 FINTRAC scheme signatures
```

Key config sections: `WindowConfig`, `TSFreshConfig`, `P1Config`–`P6Config`, `ModelConfig`, `AggregationConfig`, `EvalConfig`.

## Important Design Decisions

- **Temporal split only** — no random train/test splits. Use `preprocessing.temporal_train_test_split()`.
- **Out-of-fold predictions** — Tier 1 OOF scores feed P6 to prevent data leakage.
- **TSFresh is an implementation detail**, not a headline contribution. Do not emphasize it in abstracts or contributions.
- **Blending alpha** default is 0.7 (supervised-leaning), tunable per pillar on validation set.
- **P6 scheme templates** are derived from FINTRAC operational alerts (Project ATHENA for underground banking, trade-based ML, professional ML through MSBs). Signatures define expected pillar elevation profiles.
- **Cosine similarity** is used to match client pillar vectors to scheme templates.
- **Four-level explainability**: pillar scores, SHAP feature importance, scheme match narrative, temporal risk trajectory.

## Paper Management

- Paper is in **IEEEtran conference format** (`\documentclass[conference]{IEEEtran}`)
- Every update to `bcrs_paper.tex` must also recompile the PDF and update `paper_log/changelog.md`
- PDF naming convention: `bcrs_paper_v{X.Y}.pdf`
- Old PDFs go to `paper/archived/`
- Compile with: `pdflatex bcrs_paper.tex` (run twice for references)

## Expected Data Schema

**transactions.csv**: `client_id, txn_date, amount, direction (inbound/outbound), counterparty_id, counterparty_country, counterparty_sector, channel, currency`

**client_profiles.csv**: `client_id, industry, occupation, expected_turnover, geography, size_bucket, onboarding_date, risk_rating, pep_flag, sanctions_flag, country_risk_score, industry_risk_score, account_age_months`

**str_labels.csv**: `client_id, str_flag (0/1), str_date`

## What Remains (Not Yet Implemented)

- Jupyter notebooks (01_eda through 06_results_analysis)
- Actual data loading and end-to-end pipeline run
- Populating paper results tables with real experimental outputs
- Paper finalization and submission formatting
