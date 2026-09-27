"""Bounded offline correctness checks against independent reference calculations.

Run: python -m unittest discover -s tests -v
All observations in this file are synthetic. No profitability assertion is made.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import atlas_bot_walkforward as atlas
import atlas_probability_equation_v3_walkforward as v3
import atlas_probability_equation_v3_full_pairs_walkforward as pairs
import atlas_sequence_story_walkforward as sequence
from build_tick_dual_probe_table import SegmentTree


class SegmentTreeTests(unittest.TestCase):
    def test_crossings_and_ranges_match_linear_scan(self):
        rng = np.random.default_rng(73)
        # Empty, singleton, power-of-two and uneven leaf counts exercise padding.
        for n in (0, 1, 2, 3, 8, 17, 32):
            values = rng.integers(-4, 5, n).astype(float)
            for mode in ("min", "max"):
                tree = SegmentTree(values, mode)
                for left in range(-1, n + 2):
                    for right in range(-1, n + 2):
                        indices = list(range(max(0, left), min(n - 1, right) + 1))
                        neutral = np.inf if mode == "min" else -np.inf
                        expected_value = (
                            (min if mode == "min" else max)(values[i] for i in indices)
                            if indices else neutral
                        )
                        self.assertEqual(tree.range_value(left, right), expected_value)
                        for threshold in (-5.0, -2.0, 0.0, 2.0, 5.0):
                            expected = next(
                                (
                                    i for i in indices
                                    if (values[i] >= threshold if mode == "max"
                                        else values[i] <= threshold)
                                ),
                                -1,
                            )
                            found = (tree.first_ge if mode == "max" else tree.first_le)(
                                left, right, threshold
                            )
                            self.assertEqual(found, expected, (mode, values, left, right, threshold))

    def test_mode_misuse_is_rejected(self):
        with self.assertRaises(ValueError):
            SegmentTree(np.array([1.0]), "sum")
        with self.assertRaises(ValueError):
            SegmentTree(np.array([1.0]), "min").first_ge(0, 0, 1)
        with self.assertRaises(ValueError):
            SegmentTree(np.array([1.0]), "max").first_le(0, 0, 1)


class TrainingBoundaryTests(unittest.TestCase):
    @staticmethod
    def frame():
        n_train, n_test = 120, 24
        frame = pd.DataFrame({
            "atlas_datetime": list(pd.date_range("2025-01-01", periods=n_train, freq="5min"))
            + list(pd.date_range("2025-02-01", periods=n_test, freq="5min")),
            "value": np.r_[np.arange(n_train), np.arange(n_test) + 1_000_000],
            "category": ["known-a", "known-b"] * (n_train // 2) + ["unseen"] * n_test,
        })
        labels = pd.Series(np.resize(["BUY", "SELL", "NONE"], len(frame)))
        months = frame["atlas_datetime"].dt.strftime("%Y-%m")
        return frame, labels, months

    def test_fold_fits_transforms_using_only_training_rows(self):
        frame, labels, months = self.frame()
        observed = []
        original = v3._fit_transform_spec

        def capture(series):
            spec = original(series)
            observed.append((series.copy(), spec))
            return spec

        with patch.object(v3, "_fit_transform_spec", side_effect=capture):
            with redirect_stdout(io.StringIO()):
                predictions, _ = pairs._score_fold(
                    frame, labels, months, ["value", "category"], 1, ["2025-01", "2025-02"]
                )
        self.assertEqual(len(observed), 2)
        for (seen, _), col in zip(observed, ["value", "category"]):
            pd.testing.assert_series_equal(seen, frame.loc[:119, col])
        numeric_spec = observed[0][1]
        expected_edges = np.quantile(np.arange(120), np.linspace(0, 1, 11))
        expected_edges[[0, -1]] = [-np.inf, np.inf]
        np.testing.assert_allclose(numeric_spec.edges, expected_edges)
        self.assertEqual(len(predictions), 24)
        scores = predictions[["p_buy", "p_sell", "p_none"]].to_numpy(float)
        np.testing.assert_allclose(scores.sum(axis=1), 100)

    def test_held_out_labels_do_not_change_predictions(self):
        frame, labels, months = self.frame()
        changed = labels.copy()
        changed.iloc[120:] = "NONE"
        with redirect_stdout(io.StringIO()):
            first, _ = pairs._score_fold(
                frame, labels, months, ["value", "category"], 1, ["2025-01", "2025-02"]
            )
            second, _ = pairs._score_fold(
                frame, changed, months, ["value", "category"], 1, ["2025-01", "2025-02"]
            )
        pd.testing.assert_frame_equal(
            first[["p_buy", "p_sell", "p_none"]], second[["p_buy", "p_sell", "p_none"]]
        )

    def test_unseen_category_contributes_no_single_vote(self):
        tr, te, k = pairs._encode_train_test(
            pd.Series(["red", "blue", "red", "blue"]),
            pd.Series(["green", "red", "blue"]),
        )
        self.assertEqual(k, 2)
        self.assertEqual(te[0], -1)
        base = np.array([0.5, 0.5, 0.0])
        votes, _ = pairs._single_votes_and_effects(tr, np.array([0, 1, 0, 1]), k, base)
        scores = np.tile(base, (3, 1))
        pairs._apply_single_votes(scores, te, votes)
        np.testing.assert_allclose(scores[0], base)
        self.assertGreater(scores[1, 0], base[0])
        self.assertGreater(scores[2, 1], base[1])

    def test_obvious_target_and_future_columns_are_blocked(self):
        for name in ("tick_race_winner", "next_m5_close", "buy_time_to_plus_2r_sec"):
            self.assertIsNotNone(v3._leakage_reason(name))
        self.assertIsNone(v3._leakage_reason("observed_range"))


class PairContributionTests(unittest.TestCase):
    def test_dense_and_sparse_match_direct_conditional_count_oracle(self):
        rng = np.random.default_rng(918)
        for k_a, k_b in ((2, 3), (7, 5), (19, 23)):
            train_a = rng.integers(0, k_a, size=100, dtype=np.int32)
            train_b = rng.integers(0, k_b, size=100, dtype=np.int32)
            y = rng.integers(0, 3, size=100, dtype=np.int8)
            # Include every pair, pairs absent from training, and unknown categories.
            queries = [(a, b) for a in range(k_a) for b in range(k_b)] + [(-1, 0), (0, -1)]
            test_a, test_b = np.array(queries, dtype=np.int32).T
            base = np.array([sum(y == o) / len(y) for o in range(3)])

            def effects(values, count):
                out = np.zeros((count, 3))
                for code in range(count):
                    members = [i for i, value in enumerate(values) if value == code]
                    distribution = np.array(
                        [sum(y[i] == outcome for i in members) / len(members)
                         if members else 0 for outcome in range(3)]
                    )
                    out[code] = distribution - base
                return out

            effect_a = effects(train_a, k_a)
            effect_b = effects(train_b, k_b)
            oracle = np.zeros((len(queries), 3))
            for row, (a, b) in enumerate(queries):
                members = [i for i in range(len(y)) if train_a[i] == a and train_b[i] == b]
                if members:
                    conditional = np.array(
                        [sum(y[i] == o for i in members) / len(members) for o in range(3)]
                    )
                    oracle[row] = (
                        conditional - base - effect_a[a] - effect_b[b]
                    ) * len(members) / len(y)
            dense = pairs._pair_contrib_low_card(
                train_a, train_b, test_a, test_b, y, k_b, effect_a, effect_b, base
            )
            sparse = pairs._pair_contrib_unique(
                train_a, train_b, test_a, test_b, y, effect_a, effect_b, base
            )
            np.testing.assert_allclose(dense, oracle, atol=1e-14)
            np.testing.assert_allclose(sparse, oracle, atol=1e-14)


class SequenceTests(unittest.TestCase):
    def test_lagged_windows_exclude_current_and_future_rows(self):
        mask = np.array([False, True, False, False, True, True, False])
        for module in (atlas, sequence):
            for window in (1, 2, 5, 20):
                expected = np.array([mask[max(0, i - window):i].sum() for i in range(len(mask))])
                np.testing.assert_array_equal(module._prev_count(mask, window), expected)
                np.testing.assert_array_equal(module._prev_any(mask, window), expected > 0)
                for i in range(len(mask)):
                    changed = mask.copy()
                    changed[i:] = ~changed[i:]
                    np.testing.assert_array_equal(
                        module._prev_count(changed, window)[:i + 1], expected[:i + 1]
                    )


if __name__ == "__main__":
    unittest.main()
