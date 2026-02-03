# B-CRS Implementation Plan
## Behavioral Client Risk Score Framework (Two-Tier Architecture)

---

## Phase 1: Data Preparation & Exploration

### 1.1 Data Inventory
- Catalog available data sources:
  - Transaction data (amounts, dates, counterparties, channels, currencies)
  - KYC/CDD data (customer type, industry, geography, PEP status, products, occupation)
  - STR data (filing dates, client IDs, STR narratives if available)
  - Existing static CRS scores
  - Sector/destination data (real estate, automotive, legal, securities counterparties)
  - Geographic corridor data (origin-destination country pairs)
- Document data schema, coverage period, and volume
- Identify data quality issues (missing fields, inconsistencies)

### 1.2 Data Extraction & Cleaning
- Extract transaction data for the study period (recommend 3+ years for train/test split)
- Link transactions to client IDs and STR labels
- Handle missing values, duplicates, and data quality issues
- Anonymize/pseudonymize as required by internal data governance
- Enrich counterparty data with sector classification where possible

### 1.3 Exploratory Data Analysis
- Client population statistics (count, segment distribution)
- STR prevalence rate (class imbalance assessment)
- Transaction volume distributions by client segment
- Temporal patterns in STR filings
- Preliminary correlation analysis between existing CRS and STR outcomes
- Geographic corridor analysis (top origin-destination pairs)
- Sector-destination flow analysis (which sectors receive outflows)

### 1.4 Train/Test Split Strategy
- **Temporal split** (critical — no random splitting):
  - Training period: e.g., Jan 2020 – Dec 2022
  - Validation period: e.g., Jan 2023 – Jun 2023
  - Test period: e.g., Jul 2023 – Dec 2023
- Observation window T = 6 months (configurable)
- Label assignment: client labeled positive if STR filed during or within N months after observation window

---

## Phase 2: Feature Engineering

### 2.1 TSFresh Feature Extraction (Cross-Pillar)
- For each client and observation window, construct time series:
  - Daily/weekly transaction count series
  - Daily/weekly transaction amount series (total, mean, max)
  - Inter-transaction interval series
  - Inbound vs. outbound amount series
- Run TSFresh `extract_features()` with `ComprehensiveFCParameters`
- Apply `select_features()` with Benjamini-Yekutieli relevance filtering
- Store extracted features in a feature store/database for reuse

### 2.2 Pillar 1: Volume & Velocity Features
- Transaction count per period (daily, weekly, monthly)
- Total/mean/max/min transaction amounts per period
- Standard deviation and coefficient of variation of amounts
- Inter-transaction time statistics (mean, std, min, max)
- Z-score of current metrics vs. client's own historical baseline
- Month-over-month growth rates
- TSFresh: statistical features from amount and count time series

### 2.3 Pillar 2: Network & Counterparty Features
- Build client-level transaction graph (bipartite or projected)
- Unique counterparty count (inbound, outbound, total)
- Counterparty concentration: Herfindahl-Hirschman Index (HHI)
- New counterparty rate: % of counterparties first seen in current window
- Geographic risk exposure: % transactions with high-risk jurisdictions
- Network centrality: degree, betweenness, closeness, PageRank
- Community detection (Louvain) and cross-community transaction ratio
- Tools: NetworkX or igraph for graph construction and metrics

### 2.4 Pillar 3: Temporal Behavior Features (TSFresh-Heavy)
- Autocorrelation at lags 1, 7, 14, 30
- Approximate entropy & sample entropy
- Time-of-day distribution (bin into 4-6 periods, compute entropy)
- Day-of-week distribution entropy
- Weekend/holiday transaction ratio
- Change-point detection (CUSUM, ruptures library)
- Trend strength (STL decomposition residuals)
- Lempel-Ziv complexity of timing sequences

### 2.5 Pillar 4: Typology Alignment Features
- Structuring indicators:
  - Count of transactions within 10% of reporting threshold
  - Just-below-threshold frequency ratio
- Round-number transaction ratio (amounts ending in 000, 00)
- Rapid fund-through: deposits followed by withdrawals within 1-3 days
- Flow-through ratio: total outbound / total inbound
- Funnel account: count of unique senders vs. receivers ratio
- Cash transaction ratio (if channel data available)
- Cross-border transaction ratio vs. declared business scope

### 2.6 Pillar 5: Profile Consistency Features
- Transaction volume vs. declared expected turnover ratio
- Peer group definition: cluster clients by (industry, size, geography)
- Mahalanobis distance from peer group centroid
- Behavioral drift: cosine similarity between current and historical feature vectors
- Dormancy detection: periods of zero activity followed by sudden resumption
- Product usage anomaly: products used vs. declared business purpose

### 2.7 Pillar 6: Strategic Scheme Alignment Features (Tier 2)

P6 is a **meta-pillar** that consumes Tier 1 sub-scores (S_P1–S_P5) alongside scheme-specific features. Feature engineering for P6 occurs AFTER Tier 1 pillar models produce sub-scores.

#### 2.7.1 Tier 1 Interaction Features
- Pairwise interaction terms: S_P1×S_P2, S_P1×S_P5, S_P2×S_P4, S_P2×S_P5, S_P4×S_P5, etc.
- Higher-order interactions: S_P2×S_P4×S_P5 (network + typology + profile)
- Pillar co-elevation count: number of pillars above threshold (e.g., >0.7)
- Maximum Tier 1 sub-score and standard deviation across pillars

#### 2.7.2 Scheme Template Similarity
- Define scheme signature vectors from FINTRAC/FinCEN operational alerts:
  - **Underground Banking (Project ATHENA):** High P2 (geographic corridor) + High P5 (occupation mismatch) + High P4 (pass-through) + Moderate P1 (volume)
  - **Trade-Based ML:** High P4 (round figures, flow-through) + High P2 (import/export counterparties) + Moderate P5 (sector mismatch)
  - **Professional ML:** High P2 (hub-like network) + High P1 (high volume) + High P3 (MSB-like temporal patterns) + High P5 (personal/business mirroring)
- Cosine similarity between client's [S_P1,...,S_P5] vector and each scheme template
- Maximum scheme similarity score and best-matching scheme ID

#### 2.7.3 Underground Banking Indicators (FINTRAC Project ATHENA)
- Flow symmetry ratio: |total_inbound - total_outbound| / max(total_inbound, total_outbound)
- Geographic corridor concentration: HHI of origin-destination country pairs
- Specific corridor flags: % of flows on FINTRAC-flagged corridors (e.g., China/HK)
- Pass-through velocity: median time between receipt and disbursement
- Sector-destination pattern: % outflows to real estate, automotive, legal, securities
- Stated purpose mismatch: funds marked "tuition"/"living" used for real estate/investments

#### 2.7.4 Trade-Based ML Indicators (FINTRAC Operational Alert)
- Round-figure payment ratio in US$50K increments
- Foreign currency account flow-through rate
- Counterparty profile: fraction of counterparties with limited/no online presence
- Import/export entity counterparty concentration
- Multiple currency exchange frequency (CAD/USD/EUR cycles)
- Invoice-like payment patterns (regular amounts to same counterparties)

#### 2.7.5 Professional ML / MSB-Like Indicators
- Personal account activity mirroring business patterns
- Multiple listed occupations, addresses, or phone numbers
- Hub score: high betweenness centrality + diverse counterparty sectors
- Occupation flags: student/homemaker/unemployed + high-value transactions
- Multiple bank relationship indicator
- Structured deposits at multiple branches/locations same day

#### 2.7.6 Occupation-Activity Severity Score
- Map declared occupation to expected transaction profile
- Compute severity = f(declared_income_proxy, observed_volume, counterparty_risk)
- Flag: student/homemaker/unemployed with cross-border flows > threshold

---

## Phase 3: Pillar Model Development

### 3.1 Tier 1: Supervised Models (P1–P5)
- Algorithm: XGBoost (primary), with LightGBM as alternative
- Target: STR label (binary)
- Features: pillar-specific feature set
- Hyperparameter tuning: Bayesian optimization (Optuna) with 5-fold CV
- Calibration: Platt scaling or isotonic regression to produce calibrated probabilities
- Feature importance: SHAP values for explainability

### 3.2 Tier 1: Unsupervised Models (P1–P5)
- Algorithm: Isolation Forest (primary), with Local Outlier Factor as alternative
- Features: same pillar-specific feature set
- Contamination parameter: estimate from STR prevalence rate
- Output: anomaly score normalized to [0, 1]
- Validation: check correlation with STR labels (should be positive but not perfect)

### 3.3 Tier 1: Pillar Score Blending (P1–P5)
- For each pillar i:
  `S_Pi = alpha_i * supervised_prob + (1 - alpha_i) * anomaly_score`
- Tune alpha_i on validation set
- Starting point: alpha = 0.7 (lean on supervised given STR labels available)

### 3.4 Tier 1: Pillar Independence Validation
- Compute correlation matrix between P1–P5 sub-scores
- Target: moderate correlations (0.2–0.5) indicating related but distinct dimensions
- If pillars are too correlated (>0.8): revisit feature assignments
- If uncorrelated (<0.1): verify each pillar individually has predictive power

### 3.5 Tier 2: P6 Model Development
**IMPORTANT:** P6 models are trained AFTER Tier 1 models produce sub-scores on the training set. Use out-of-fold predictions from Tier 1 to avoid data leakage.

- Generate Tier 1 sub-scores using out-of-fold predictions (5-fold CV on training data)
- Compute P6 interaction features and scheme-specific features (Section 2.7)
- Train P6 supervised model (XGBoost) on combined feature set: [interaction features, scheme features, Tier 1 sub-scores]
- Train P6 unsupervised model (Isolation Forest) on same feature set
- Blend: `S_P6 = alpha_6 * supervised_prob + (1 - alpha_6) * anomaly_score`
- Validate: P6 should capture additional signal beyond Tier 1 pillars — test incremental AUROC

### 3.6 P6 Ablation Study
- Compare B-CRS with P1–P5 only vs. P1–P6
- Test P6 with only interaction features (no scheme-specific features) vs. full P6
- Test P6 with only scheme-specific features (no Tier 1 sub-scores) vs. full P6
- This demonstrates the value of the two-tier architecture

---

## Phase 4: Aggregation & Final Score

### 4.1 Implement Three Aggregation Strategies
1. **Weighted Linear**: Learn weights via logistic regression on sub-scores (KYC + P1–P6)
2. **Gradient Boosting**: XGBoost meta-model on 7 sub-scores (KYC + P1–P6)
3. **Bayesian**: Estimate likelihood ratios P(S_i | STR) / P(S_i | non-STR) using KDE

### 4.2 Compare Aggregation Performance
- Evaluate all three on test set, for both P1–P5 only and P1–P6 configurations
- Metrics: AUROC, AUPRC, Brier score, Precision@5%, Precision@10%
- Select best-performing for final B-CRS

### 4.3 Baseline Comparisons
- Baseline 1: Existing static KYC CRS
- Baseline 2: Flat XGBoost on ALL features (no pillar decomposition)
- Baseline 3: Reite et al. replication (if feasible with available data)
- Baseline 4: B-CRS P1–P5 only (to isolate P6 contribution)
- Statistical significance: bootstrap confidence intervals or DeLong test for AUROC

---

## Phase 5: Explainability & Visualization

### 5.1 Pillar-Level Dashboard
- Radar/spider chart showing P1–P6 sub-scores per client
- Color-coded risk bands (Low/Medium/High/Very High)
- Historical trend of each pillar score over time
- P6 displayed distinctly as "Scheme Alignment" with matched scheme label

### 5.2 Feature-Level Explanations
- SHAP waterfall plots per pillar showing top contributing features
- Natural language risk narrative generator:
  "Client X scored High (0.82) primarily due to elevated Network Risk (P2=0.91):
   12 new counterparties in high-risk jurisdictions detected in the last 3 months,
   counterparty concentration dropped significantly."

### 5.3 Scheme-Level Explanations (P6-specific)
- When P6 is elevated, display:
  - Best-matching scheme template (e.g., "Underground Banking - Project ATHENA")
  - Scheme similarity score and which Tier 1 pillars contribute
  - Specific FIU operational alert reference
  - Key scheme-specific features driving the score
- Example narrative:
  "P6 Strategic Scheme Alignment = 0.87 (Very High). Behavioral fingerprint
   matches FINTRAC Project ATHENA underground banking indicators:
   - Geographic corridor concentration: 78% of flows on China/HK corridor
   - Flow symmetry ratio: 0.94 (near-equal in/out)
   - Pass-through velocity: median 36 hours
   - Occupation-activity mismatch: student with $2.1M annual volume
   - Sector destination: 62% of outflows to real estate"

### 5.4 Temporal Visualization
- B-CRS trajectory over time with change-point annotations
- Overlay significant events (STR filings, KYC updates, account changes)

---

## Phase 6: Evaluation & Paper Writing

### 6.1 Quantitative Results
- Populate results table (AUROC, AUPRC, Brier, P@k for all models)
- Statistical significance tests
- Ablation study: remove one pillar at a time to assess contribution
- **Two-tier ablation: P1–P5 vs. P1–P6 comparison** (key result)
- P6 component ablation: interaction features only vs. scheme features only vs. full P6
- Sensitivity analysis on observation window T and alpha blending parameters

### 6.2 Qualitative Assessment
- Present B-CRS outputs to AML domain experts for coherence assessment
- Document case studies: select 3-5 interesting clients and walk through pillar decomposition
- **P6 case studies: clients matching FINTRAC scheme patterns**
- Compare B-CRS explanations to actual STR narratives (if available)

### 6.3 Paper Finalization
- Populate experimental results in LaTeX template
- Write discussion section based on findings
- Review against JMLC/IEEEtran submission guidelines
- Internal review and sign-off

---

## Technical Stack

| Component | Tool/Library |
|-----------|-------------|
| Language | Python 3.10+ |
| Data Processing | pandas, polars |
| Feature Extraction | tsfresh |
| Graph Features | networkx, igraph |
| ML Models | xgboost, lightgbm, scikit-learn |
| Hyperparameter Tuning | optuna |
| Explainability | shap |
| Change Detection | ruptures |
| Similarity Metrics | scipy (cosine_similarity), sklearn |
| Visualization | matplotlib, seaborn, plotly |
| Statistical Testing | scipy, statsmodels |
| Paper | LaTeX (IEEEtran) |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low STR count (class imbalance) | Poor supervised model performance | SMOTE/ADASYN oversampling, focal loss, lean on unsupervised component |
| Data quality issues | Unreliable features | Thorough EDA in Phase 1, robust feature engineering |
| Pillar redundancy | Framework adds no value over flat model | Feature assignment review, pillar independence validation |
| TSFresh feature explosion | Overfitting, high dimensionality | Relevance filtering, feature selection within each pillar |
| Computational cost of graph features | Slow pipeline | Approximate centrality algorithms, sampling for large graphs |
| Single institution bias | Limited generalizability | Acknowledge as limitation, discuss federated future work |
| P6 data leakage from Tier 1 | Overfitting P6 | Use out-of-fold Tier 1 predictions to train P6 |
| P6 scheme templates too narrow | Miss novel schemes | Unsupervised component in P6, template updating process |
| Sector/occupation data missing | Weak P6 features | Graceful degradation — P6 falls back to interaction features only |

---

## Folder Structure

```
client_Risk_score/
├── paper/
│   ├── bcrs_paper.tex          # LaTeX manuscript (IEEEtran)
│   └── archived/               # Previous PDF versions
├── notes/
│   └── implementation_plan.md  # This file
├── data/
│   ├── raw/                    # Raw extracted data
│   ├── processed/              # Cleaned, merged datasets
│   └── features/               # Feature store (per pillar)
├── src/
│   ├── data_prep/              # Data extraction and cleaning
│   ├── features/
│   │   ├── tsfresh_engine.py   # TSFresh extraction pipeline
│   │   ├── pillar_p1.py        # Volume & Velocity features
│   │   ├── pillar_p2.py        # Network & Counterparty features
│   │   ├── pillar_p3.py        # Temporal Behavior features
│   │   ├── pillar_p4.py        # Typology Alignment features
│   │   ├── pillar_p5.py        # Profile Consistency features
│   │   ├── pillar_p6.py        # Strategic Scheme Alignment features
│   │   └── scheme_templates.py # FINTRAC/FinCEN scheme signature definitions
│   ├── models/
│   │   ├── tier1_supervised.py   # XGBoost per-pillar models (P1–P5)
│   │   ├── tier1_unsupervised.py # Isolation Forest per-pillar models (P1–P5)
│   │   ├── tier2_p6.py           # P6 meta-pillar model
│   │   ├── aggregation.py        # Aggregation strategies
│   │   └── baselines.py          # Baseline models
│   ├── evaluation/
│   │   ├── metrics.py          # AUROC, AUPRC, P@k, etc.
│   │   ├── ablation.py         # Pillar ablation and P6 contribution analysis
│   │   └── explainability.py   # SHAP, radar charts, scheme narratives
│   └── utils/
│       └── config.py           # Parameters, paths, constants
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_tier1_modeling.ipynb
│   ├── 04_tier2_p6_modeling.ipynb
│   ├── 05_aggregation.ipynb
│   └── 06_results_analysis.ipynb
└── requirements.txt
```
