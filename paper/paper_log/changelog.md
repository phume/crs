# B-CRS Paper Changelog

## v0.4 — 2026-02-03
**Added P6 (Strategic Scheme Alignment) and Two-Tier Architecture**
- Introduced Tier 2 meta-pillar P6 that consumes Tier 1 sub-scores (P1–P5) alongside scheme-specific features to detect multi-dimensional laundering schemes
- P6 grounded in FINTRAC operational alerts: Project ATHENA (underground banking), professional ML through trade/MSBs
- Updated abstract to describe two-tier architecture with six pillars
- Added "Two-Tier Scheme Detection" as new contribution in Introduction
- Added "FIU Operational Alerts and Strategic Scheme Detection" subsection to Related Work
- Redesigned framework figure to show Tier 1 / Tier 2 separation
- Updated equations throughout: aggregation now uses P1–P6 (7-dimensional sub-score vector)
- Added full P6 methodology section with Tier 1 interaction features, scheme template similarity, underground banking indicators, trade-based ML indicators, professional ML indicators, occupation-activity severity score
- Added "Scheme Level" to explainability layer (now four levels)
- Expanded results table with P1–P5 vs. P1–P6 comparison rows
- Updated conclusion to reflect two-tier architecture
- Added FINTRAC references (fintrac_ub2023, fintrac_pml2020)

## v0.3 — 2026-02-03
**Converted to IEEEtran conference format**
- Switched document class from `article` to `IEEEtran` conference
- Replaced natbib (`\citep`/`\citet`) with `\cite`
- Converted bibliography to IEEE numbered format
- Replaced TikZ figure with fbox text diagram
- Updated author block to `\IEEEauthorblockN`/`\IEEEauthorblockA`
- Changed keywords to `\begin{IEEEkeywords}`
- Renamed "Literature Review" to "Background and Related Work"

## v0.2 — 2026-02-03
**Removed TSFresh from abstract and contributions**
- Removed TSFresh mention from abstract (implementation detail, not a contribution)
- Removed TSFresh as standalone contribution bullet in Introduction
- TSFresh remains in methodology sections where it belongs

## v0.1 — 2026-02-03
**Initial draft**
- Full paper structure with P1–P5 pillars
- Standard article class with natbib
- TikZ framework diagram
- 20 references
