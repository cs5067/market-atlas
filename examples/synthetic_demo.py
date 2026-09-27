#!/usr/bin/env python3
"""Exercise the research algorithms with deterministic, invented observations.

Run from any directory: python examples/synthetic_demo.py
No files, broker connections, credentials, or real market data are required.
The generated relationships are deliberately artificial; output is a software
demonstration, not evidence that any strategy predicts financial markets.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# The preserved research scripts use sibling imports rather than a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import atlas_bot_walkforward as atlas
import atlas_probability_equation_v3_full_pairs_walkforward as pairs
import atlas_sequence_story_walkforward as sequence
from build_tick_dual_probe_table import SegmentTree


def synthetic_observations() -> pd.DataFrame:
    """Return three chronological synthetic blocks with known artificial labels."""
    rng = np.random.default_rng(1729)
    rows_per_month = 96
    times = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(start, periods=rows_per_month, freq="5min").to_numpy()
                for start in ("2025-01-01", "2025-02-01", "2025-03-01")
            ]
        )
    )
    row = np.arange(len(times))
    regime = (row // 4) % 3
    # This relationship is invented to make pattern extraction observable.
    winner = np.array(["BUY", "SELL", "NONE"])[regime].copy()
    noise = rng.random(len(row)) < 0.15
    winner[noise] = rng.choice(["BUY", "SELL", "NONE"], size=int(noise.sum()))
    frame = pd.DataFrame(
        {
            "atlas_datetime": times,
            "synthetic_regime": np.array(["rising", "falling", "flat"])[regime],
            "synthetic_strength": regime + rng.normal(0, 0.25, len(row)),
            "synthetic_activity": rng.uniform(0, 1, len(row)),
            "tick_race_winner": winner,
        }
    )
    # Invented target clocks are used only to construct training labels.
    for side in ("buy", "sell"):
        is_winner = winner == side.upper()
        has_direction = winner != "NONE"
        frame[f"{side}_time_to_plus_2r_sec"] = np.where(is_winner, 10.0, np.nan)
        frame[f"{side}_time_to_minus_1r_sec"] = np.where(
            has_direction & ~is_winner, 10.0, np.nan
        )
    return frame


def run_demo() -> dict:
    ticks = np.array([100.0, 100.2, 99.8, 100.4, 100.1, 99.6, 100.5])
    upper = SegmentTree(ticks, "max").first_ge(1, 6, 100.4)
    lower = SegmentTree(ticks, "min").first_le(1, 6, 99.7)

    frame = synthetic_observations()
    allowed = ["synthetic_regime", "synthetic_strength", "synthetic_activity"]
    month = frame["atlas_datetime"].dt.strftime("%Y-%m")
    train = frame.loc[month < "2025-03"].reset_index(drop=True)
    test = frame.loc[month == "2025-03"].reset_index(drop=True)
    targets = atlas._build_target_frame(train)
    cards = atlas._mine_cards(
        train[allowed], targets, min_n=8, top_atoms=6, max_combo_size=2
    )
    state = sequence._state_from_cards(cards, test[allowed], top_k=5)
    stories = sequence._story_masks(state, allow_start_only=False)

    # Use the original chronological scorer; suppress its progress chatter.
    with redirect_stdout(io.StringIO()):
        predictions, _ = pairs._score_fold(
            frame,
            frame["tick_race_winner"],
            month,
            allowed,
            fold_i=2,
            months=["2025-01", "2025-02", "2025-03"],
        )
    scores = predictions[["p_buy", "p_sell", "p_none"]].to_numpy(float)
    return {
        "data": "SYNTHETIC ONLY; not trading-performance evidence",
        "seed": 1729,
        "threshold_search": {
            "first_index_at_or_above_100_4": upper,
            "first_index_at_or_below_99_7": lower,
        },
        "chronological_split": {"training_rows": len(train), "test_rows": len(test)},
        "pattern_mining": {
            "training_cards": len(cards),
            "held_out_rows_matching_any_story": int(
                np.logical_or.reduce([mask for _, _, mask in stories]).sum()
            ),
        },
        "v3_scoring": {
            "input_features": len(allowed),
            "feature_pairs": len(allowed) * (len(allowed) - 1) // 2,
            "held_out_rows_scored": len(predictions),
            "finite_normalized_scores": bool(
                np.isfinite(scores).all()
                and (scores >= 0).all()
                and np.allclose(scores.sum(axis=1), 100.0)
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
