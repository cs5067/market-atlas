# Methodology and limitations

## What the algorithms do

The tick-probe builder starts after each five-minute signal candle closes. It locates the first eligible bid/ask quote and a bounded future horizon using binary search. Four segment trees summarize bid and ask minima/maxima. Tree traversal skips ranges that cannot cross a threshold, retaining the first chronological crossing for each target or stop. Outcome tables resolve each side separately and mark equal-time competing outcomes as ties.

Rule mining fits categorical conditions and numeric thresholds inside the training window. Compatible conditions are combined into evidence cards, filtered by support and lift. The sequence module turns card matches into lagged rolling evidence and explicit phase-transition rules. The next chronological block is then evaluated with those learned cards.

The component scorer learns a base outcome distribution and a frequency-weighted deviation for each feature bucket. Its pair extension subtracts the two individual feature effects from a joint effect to obtain a residual contribution. Small category products use dense counts; larger products use packed pair keys and only store observed combinations. Unknown categories have no learned contribution. Scores are clipped and normalized for reporting; this normalization does not make them calibrated probabilities.

## What is reproducible here

`examples/synthetic_demo.py` and the unit tests exercise these original functions with invented in-memory observations. The demo's relationships are deliberately constructed so that pattern extraction is visible. They provide no estimate of market accuracy or returns.

The retained research scripts include their historical file-based runners. They require externally supplied feature tables and ticks. The public snapshot does not include the full feature-construction pipeline, original datasets, broker execution, private journals or recorded experimental outputs. Running a research script's `--help` is not universally supported: several historical runners have fixed local input paths. Use the demo for a first run.

## Evaluation boundaries

- The outer strategy experiments use later chronological blocks for evaluation, but training labels can have outcome windows extending across a fold boundary. A future rigorous evaluation needs outcome-time purging or an embargo.
- Historical upstream feature engineering included full-sample volatility ranks, indicator backfills, higher-timeframe alignment issues and session extrema broadcast to earlier rows. Filtering obvious target column names is insufficient to establish causality for arbitrary input tables.
- The optional LightGBM variant removes named future-derived features and computes running session extrema. Its inner `KFold(shuffle=False)` still trains on later blocks when validating an earlier block. This is not forward-chaining temporal validation, despite chronological outer folds.
- Rule/template selection and large feature-pair searches can overfit through repeated experimentation. Neither a high support count nor a confident score establishes a stable relationship.
- Tick thresholds establish event order within the recorded tape. They do not simulate market impact, queue position, broker latency or guaranteed execution at an exact threshold. There is no claim of high-frequency execution performance.
- Some historical strategy harnesses treat unresolved horizon outcomes as zero return. That convention is not a model of closing an actual position.

The useful result of this work is a set of data-processing, search, pattern-analysis and evaluation tools. The experiments did not establish a reliable profitable strategy. The next research priorities would be causal feature reconstruction, forward-only tuning, purged evaluation, and realistic execution assumptions before interpreting predictive results.
