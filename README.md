# Market Atlas

Python tools for studying market patterns and testing trading ideas against bid/ask tick data. The project combines data preparation, custom threshold-search algorithms, interpretable rule mining, sequential patterns and chronological model evaluation.

This repository contains eight research modules from a larger personal trading project, plus a reproducible synthetic-data demonstration. The original project also included a MetaTrader execution bridge; this public edition focuses on the offline research code. It contains no broker connection, account records or market datasets.

## Run the demo

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The demo calls the actual research functions with seeded, invented observations. It finds tick-price crossings, mines rule combinations from two training blocks, applies sequence rules to a later block, and scores feature interactions on held-out rows. No credentials, downloads or input files are required. The output describes algorithm behavior, not trading performance.

## Engineering details

| Area | Implementation |
| --- | --- |
| Market-data preparation | Chunked pandas ingestion of MetaTrader tick exports; timestamp and quote checks; Parquet partitions; bid/ask/mid OHLC reconstruction. |
| Tick outcome labeling | NumPy `searchsorted` locates entry and horizon boundaries. Custom min/max segment trees find the first stop or target crossing without scanning every tick for each query. |
| Pattern recognition | Numeric thresholds and categorical conditions form evidence cards. Boolean-mask combinations search for compatible two- and three-feature rules, ranked by support and outcome lift. |
| Sequence analysis | Shifted rolling windows identify transitions in recent evidence, such as a directional start following a neutral phase. These are explicit rules rather than a neural sequence model. |
| Interaction scoring | Train-fitted quantile bins and category codes feed an additive scoring model. Feature pairs use dense `bincount` tables or packed 64-bit keys with sparse lookup, depending on cardinality. |
| Strategy evaluation | Rolling and expanding training windows produce later-period predictions, evidence ledgers and calibration reports. An optional LightGBM experiment provides a tree-based comparison. |

## Source guide

- [Tick ingestion and candles](research/tick_vr_ingest.py)
- [Segment tree and tick probes](research/build_tick_dual_probe_table.py)
- [Outcome labels and ties](research/build_tick_winner_tables.py)
- [Rule-card mining and rolling evaluation](research/atlas_bot_walkforward.py)
- [Lagged sequence patterns](research/atlas_sequence_story_walkforward.py)
- [Component scoring and expanding evaluation](research/atlas_probability_equation_v3_walkforward.py)
- [Dense and sparse feature-pair scoring](research/atlas_probability_equation_v3_full_pairs_walkforward.py)
- [Optional LightGBM experiment](research/atlas_lightgbm_walkforward_clean.py)

The original command-line research runs expect local tick files or prepared feature tables that are not distributed here. They are retained for inspection and adaptation; the synthetic demo is the supported self-contained entry point. Parquet IO needs `requirements-data.txt`; the optional model also needs `requirements-ml.txt`.

## Validation and research limits

The tests compare segment-tree queries and both pair-scoring implementations with independent, simple reference calculations. They also check that transforms are fitted on training rows, held-out labels do not change predictions, unseen categories contribute no single-feature vote, and sequence windows exclude the current row.

The experiments did not establish a reliable predictive edge. They were still useful: a few ideas could be tested reliably, others became better understood, and the data-processing, search and evaluation tools are reusable. Chronological outer folds alone do not eliminate leakage: the historical upstream feature pipeline contained future-derived features, outcome horizons are not purged at fold boundaries, and the optional LightGBM script uses ordinary K-fold tuning inside its outer training window. Its filename includes `clean` because it removed specific known features, not because the whole pipeline is proven leakage-free. Normalized component scores are not calibrated probabilities.

See [methodology and limitations](docs/methodology.md) for the evaluation boundaries and [source provenance](docs/provenance.md) for the publication changes. This is a research and software-engineering project, not a demonstrated profitable trading system.
