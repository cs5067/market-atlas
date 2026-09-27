#!/usr/bin/env python3
"""Atlas probability equation V3 + all feature-pair combo layer.

Offline only.  This does not touch live trading.

This variant evaluates all feature-pair interactions:

- Same expanding monthly walk-forward as V3.
- Same single component vote:
      single_vote = (P(outcome | A) - P_base(outcome)) * fire_rate(A)
- Add every feature pair; 208 inputs yield C(208, 2)=21,528 pairs.
- Pair residual vote:
      pair_extra = (P(outcome | A+B) - P_base(outcome))
                   - ((P(outcome | A)-P_base(outcome))
                      + (P(outcome | B)-P_base(outcome)))
      pair_vote = pair_extra * fire_rate(A+B)
- No min-N threshold. No pre-selection. No smoothing constant.
- One prediction CSV is written per fold so the run can resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import atlas_probability_equation_v3_walkforward as v3


ROOT = Path(__file__).resolve().parent
SOURCE = v3.SOURCE
OUT_DIR = ROOT / "data/market_atlas/atlas_probability_equation_v3_full_pairs"
FOLD_DIR = OUT_DIR / "fold_predictions"

REPORT_MD = OUT_DIR / "ATLAS_PROBABILITY_EQUATION_V3_FULL_PAIRS_REPORT.md"
PREDICTIONS_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_FULL_PAIRS_PREDICTIONS.csv"
FOLDS_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_FULL_PAIRS_FOLDS.csv"
CALIBRATION_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_FULL_PAIRS_CALIBRATION_BY_SCORE.csv"
INPUT_LEDGER_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_FULL_PAIRS_INPUT_LEDGER.csv"
SUMMARY_JSON = OUT_DIR / "ATLAS_PROBABILITY_V3_FULL_PAIRS_SUMMARY.json"

OUTCOMES = v3.OUTCOMES
EPS = v3.EPS
PACK_SHIFT = np.uint64(32)
PACK_MASK = np.uint64(0xFFFFFFFF)
LOW_CARD_PRODUCT_LIMIT = 350_000


def _encode_train_test(
    train_values: pd.Series,
    test_values: pd.Series,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_str = train_values.astype(str).to_numpy()
    cats, train_codes = np.unique(train_str, return_inverse=True)
    test_str = test_values.astype(str).to_numpy()
    pos = np.searchsorted(cats, test_str)
    valid = (pos < len(cats)) & (cats[np.minimum(pos, len(cats) - 1)] == test_str)
    test_codes = np.full(len(test_str), -1, dtype=np.int32)
    test_codes[valid] = pos[valid].astype(np.int32)
    return train_codes.astype(np.int32), test_codes, int(len(cats))


def _counts_by_code(codes: np.ndarray, y_code: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.bincount(codes, minlength=k).astype(float)
    counts = np.vstack(
        [
            np.bincount(codes[y_code == outcome_i], minlength=k).astype(float)
            for outcome_i in range(3)
        ]
    ).T
    return n, counts


def _single_votes_and_effects(
    codes: np.ndarray,
    y_code: np.ndarray,
    k: int,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, counts = _counts_by_code(codes, y_code, k)
    raw = np.divide(counts, n[:, None], out=np.zeros_like(counts), where=n[:, None] > 0)
    effect = raw - base[None, :]
    fire = n / max(len(codes), 1)
    vote = effect * fire[:, None]
    return vote, effect


def _apply_single_votes(
    raw_scores: np.ndarray,
    test_codes: np.ndarray,
    vote_by_code: np.ndarray,
) -> None:
    valid = test_codes >= 0
    if valid.any():
        raw_scores[valid, :] += vote_by_code[test_codes[valid], :]


def _pack_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint64) << PACK_SHIFT) | b.astype(np.uint64)


def _pair_contrib_low_card(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    y_code: np.ndarray,
    k_b: int,
    effect_a: np.ndarray,
    effect_b: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    n_train = len(train_a)
    n_test = len(test_a)
    product = int((int(train_a.max()) + 1) * k_b)
    pair_train = train_a.astype(np.int64) * int(k_b) + train_b.astype(np.int64)
    n = np.bincount(pair_train, minlength=product).astype(float)
    nz = np.flatnonzero(n > 0)
    if len(nz) == 0:
        return np.zeros((n_test, 3), dtype=float)
    counts = np.vstack(
        [
            np.bincount(pair_train[y_code == outcome_i], minlength=product).astype(float)[nz]
            for outcome_i in range(3)
        ]
    ).T
    n_nz = n[nz]
    raw = counts / n_nz[:, None]
    a_codes = (nz // int(k_b)).astype(np.int64)
    b_codes = (nz % int(k_b)).astype(np.int64)
    combo_effect = raw - base[None, :]
    expected_parts = effect_a[a_codes, :] + effect_b[b_codes, :]
    extra = combo_effect - expected_parts
    fire = n_nz / max(n_train, 1)
    vote_nz = extra * fire[:, None]

    vote_full = np.zeros((product, 3), dtype=float)
    vote_full[nz, :] = vote_nz
    out = np.zeros((n_test, 3), dtype=float)
    valid = (test_a >= 0) & (test_b >= 0)
    if valid.any():
        pair_test = test_a[valid].astype(np.int64) * int(k_b) + test_b[valid].astype(np.int64)
        in_range = pair_test < product
        if in_range.any():
            idx = np.flatnonzero(valid)[in_range]
            out[idx, :] = vote_full[pair_test[in_range], :]
    return out


def _pair_contrib_unique(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    y_code: np.ndarray,
    effect_a: np.ndarray,
    effect_b: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    n_train = len(train_a)
    n_test = len(test_a)
    keys = _pack_pair(train_a, train_b)
    unique, inv, n = np.unique(keys, return_inverse=True, return_counts=True)
    counts = np.vstack(
        [
            np.bincount(inv[y_code == outcome_i], minlength=len(unique)).astype(float)
            for outcome_i in range(3)
        ]
    ).T
    raw = counts / n[:, None]
    a_codes = (unique >> PACK_SHIFT).astype(np.int64)
    b_codes = (unique & PACK_MASK).astype(np.int64)
    combo_effect = raw - base[None, :]
    expected_parts = effect_a[a_codes, :] + effect_b[b_codes, :]
    extra = combo_effect - expected_parts
    fire = n.astype(float) / max(n_train, 1)
    votes = extra * fire[:, None]

    out = np.zeros((n_test, 3), dtype=float)
    valid = (test_a >= 0) & (test_b >= 0)
    if valid.any():
        test_keys = _pack_pair(test_a[valid], test_b[valid])
        pos = np.searchsorted(unique, test_keys)
        seen = (pos < len(unique)) & (unique[np.minimum(pos, len(unique) - 1)] == test_keys)
        if seen.any():
            idx = np.flatnonzero(valid)[seen]
            out[idx, :] = votes[pos[seen], :]
    return out


def _score_fold(df: pd.DataFrame, y: pd.Series, month: pd.Series, allowed_cols: list[str], fold_i: int, months: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_name = months[fold_i]
    train_mask = month.isin(months[:fold_i])
    test_mask = month.eq(fold_name)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    train_y = y.iloc[train_idx].reset_index(drop=True)
    test_y = y.iloc[test_idx].reset_index(drop=True)
    y_code = train_y.map({"BUY": 0, "SELL": 1, "NONE": 2}).to_numpy(np.int8)
    base_p = v3._base_probs(train_y)
    base = np.array([base_p[o] for o in OUTCOMES], dtype=float)
    raw_scores = np.tile(base, (len(test_idx), 1))

    train_codes: list[np.ndarray] = []
    test_codes: list[np.ndarray] = []
    k_list: list[int] = []
    effects: list[np.ndarray] = []
    single_votes: list[np.ndarray] = []

    print(f"{fold_name}: transform+singles start rows train={len(train_idx)} test={len(test_idx)}", flush=True)
    for col_i, col in enumerate(allowed_cols):
        spec = v3._fit_transform_spec(df.loc[train_idx, col])
        train_comp = v3._apply_transform(df.loc[train_idx, col], spec).reset_index(drop=True)
        test_comp = v3._apply_transform(df.loc[test_idx, col], spec).reset_index(drop=True)
        tr_code, te_code, k = _encode_train_test(train_comp, test_comp)
        vote, effect = _single_votes_and_effects(tr_code, y_code, k, base)
        _apply_single_votes(raw_scores, te_code, vote)
        train_codes.append(tr_code)
        test_codes.append(te_code)
        k_list.append(k)
        effects.append(effect)
        single_votes.append(vote)
        if (col_i + 1) % 50 == 0:
            print(f"{fold_name}: singles {col_i + 1}/{len(allowed_cols)}", flush=True)

    pair_total = len(allowed_cols) * (len(allowed_cols) - 1) // 2
    pair_seen = 0
    print(f"{fold_name}: pair layer start pairs={pair_total}", flush=True)
    for i in range(len(allowed_cols)):
        train_a = train_codes[i]
        test_a = test_codes[i]
        effect_a = effects[i]
        k_a = k_list[i]
        for j in range(i + 1, len(allowed_cols)):
            k_b = k_list[j]
            product = int(k_a) * int(k_b)
            if product <= LOW_CARD_PRODUCT_LIMIT:
                raw_scores += _pair_contrib_low_card(
                    train_a,
                    train_codes[j],
                    test_a,
                    test_codes[j],
                    y_code,
                    k_b,
                    effect_a,
                    effects[j],
                    base,
                )
            else:
                raw_scores += _pair_contrib_unique(
                    train_a,
                    train_codes[j],
                    test_a,
                    test_codes[j],
                    y_code,
                    effect_a,
                    effects[j],
                    base,
                )
            pair_seen += 1
            if pair_seen % 1000 == 0:
                print(f"{fold_name}: pairs {pair_seen}/{pair_total}", flush=True)

    probs, clipped_rows = v3._normalize_scores(raw_scores)
    out = pd.DataFrame(
        {
            "fold": fold_name,
            "row_index": test_idx,
            "atlas_datetime": df.loc[test_idx, "atlas_datetime"].to_numpy(),
            "winner": test_y.to_numpy(str),
            "p_buy": probs[:, 0] * 100.0,
            "p_sell": probs[:, 1] * 100.0,
            "p_none": probs[:, 2] * 100.0,
        }
    )
    prob_mat = out[["p_buy", "p_sell", "p_none"]].to_numpy(float)
    pred_idx = prob_mat.argmax(axis=1)
    out["predicted_outcome"] = np.array(OUTCOMES, dtype=object)[pred_idx]
    trade_side_idx = out[["p_buy", "p_sell"]].to_numpy(float).argmax(axis=1)
    out["chosen_trade_side"] = np.where(trade_side_idx == 0, "BUY", "SELL")
    out["trade_score_pct"] = np.where(trade_side_idx == 0, out["p_buy"], out["p_sell"])
    out["would_trade"] = out["predicted_outcome"].isin(["BUY", "SELL"])
    out["trade_win"] = out["winner"].eq(out["chosen_trade_side"])
    out["trade_r"] = np.where(out["would_trade"], np.where(out["trade_win"], 2.0, -1.0), 0.0)
    out["probability_clipped"] = clipped_rows > 0
    metrics = v3._metrics_for_predictions(out)
    metrics.update(
        {
            "fold": fold_name,
            "train_months": fold_i,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "pairs": int(pair_total),
            "clipped_rows": int(clipped_rows),
            "train_buy_base_pct": round(base_p["BUY"] * 100.0, 6),
            "train_sell_base_pct": round(base_p["SELL"] * 100.0, 6),
            "train_none_base_pct": round(base_p["NONE"] * 100.0, 6),
        }
    )
    return out, metrics


def _fold_file(fold: str) -> Path:
    return FOLD_DIR / f"pred_{fold}.csv"


def _metrics_file(fold: str) -> Path:
    return FOLD_DIR / f"metrics_{fold}.json"


def _write_report(predictions: pd.DataFrame, folds: pd.DataFrame, allowed_cols: list[str], total_rows: int, blocked_cols: int) -> None:
    calibration = v3._calibration(predictions)
    metrics = v3._metrics_for_predictions(predictions)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    summary = {
        "source": str(SOURCE),
        "rows_total": int(total_rows),
        "input_columns": int(len(allowed_cols)),
        "blocked_columns": int(blocked_cols),
        "pairs_per_fold": int(len(allowed_cols) * (len(allowed_cols) - 1) // 2),
        "folds": int(len(folds)),
        "overall": metrics,
        "formula": {
            "single_vote": "(P(outcome|A)-P_base(outcome))*fire_rate(A)",
            "pair_extra": "(P(outcome|A+B)-P_base(outcome))-((P(outcome|A)-P_base(outcome))+(P(outcome|B)-P_base(outcome)))",
            "pair_vote": "pair_extra*fire_rate(A+B)",
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    calib_view = calibration[calibration["trade_rows"].gt(0)].sort_values("score")
    lines = [
        "# Atlas Probability Equation V3 Full Pair Walk-Forward Report",
        "",
        "Offline only. Live bot was not changed.",
        "",
        "## Formula",
        "",
        "```text",
        "single_vote = (P(outcome | A) - P_base(outcome)) * fire_rate(A)",
        "",
        "pair_extra = (P(outcome | A+B) - P_base(outcome))",
        "             - ((P(outcome | A) - P_base(outcome))",
        "                + (P(outcome | B) - P_base(outcome)))",
        "",
        "pair_vote = pair_extra * fire_rate(A+B)",
        "",
        "raw_probability = P_base + all single votes + all pair votes",
        "final_probability = clipped/normalized raw_probability",
        "```",
        "",
        "No smoothing constants. No min-N threshold. No pair pre-selection.",
        "",
        "## Data",
        "",
        f"- Rows total: `{total_rows:,}`",
        f"- Input columns: `{len(allowed_cols):,}`",
        f"- Blocked future/leakage columns: `{blocked_cols:,}`",
        f"- Pairs per fold: `{len(allowed_cols) * (len(allowed_cols) - 1) // 2:,}`",
        f"- Completed folds: `{len(folds):,}`",
        "",
        "## Overall Out-Of-Sample Result",
        "",
        f"- Scored rows: `{metrics['rows']:,}`",
        f"- BUY/SELL predictions: `{metrics['trade_rows']:,}`",
        f"- NONE predictions: `{metrics['predicted_none_rows']:,}`",
        f"- Multiclass accuracy: `{metrics['multi_class_accuracy_pct']:.2f}%`",
        f"- Trade-side win rate: `{metrics['trade_win_pct']:.2f}%`",
        f"- Trade total R taking every BUY/SELL prediction: `{metrics['trade_total_r']:.2f}R`",
        f"- Trade mean R: `{metrics['trade_mean_r']:.4f}R`",
        f"- Trade max DD: `{metrics['trade_max_dd_r']:.2f}R`",
        "",
        "## Calibration By Score",
        "",
        v3._md_table(
            calib_view,
            ["score", "rows", "trade_rows", "actual_chosen_side_win_pct", "actual_none_pct", "trade_total_r", "trade_mean_r"],
            100,
        ),
        "",
        "## Fold Results",
        "",
        v3._md_table(
            folds.sort_values("fold"),
            ["fold", "train_rows", "test_rows", "trade_rows", "trade_win_pct", "trade_total_r", "trade_mean_r", "trade_max_dd_r"],
            100,
        ),
        "",
        "## Files",
        "",
        f"- Predictions: `{PREDICTIONS_CSV}`",
        f"- Folds: `{FOLDS_CSV}`",
        f"- Calibration: `{CALIBRATION_CSV}`",
        f"- Summary JSON: `{SUMMARY_JSON}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SOURCE).reset_index(drop=True)
    y = v3._winner_series(df)
    ts = pd.to_datetime(df["atlas_datetime"], errors="coerce", utc=True)
    month = v3._month_key(ts)
    allowed_cols = [col for col in df.columns if v3._leakage_reason(col) is None]
    input_ledger = pd.DataFrame(
        [
            {
                "column": col,
                "role": "input" if col in allowed_cols else "blocked_answer_key_or_identifier",
                "reason": "decision_time_or_current_bar_value" if col in allowed_cols else v3._leakage_reason(col),
            }
            for col in df.columns
        ]
    )
    input_ledger.to_csv(INPUT_LEDGER_CSV, index=False)

    months = sorted(month.dropna().unique().tolist())
    for fold_i in range(1, len(months)):
        fold_name = months[fold_i]
        pred_path = _fold_file(fold_name)
        metric_path = _metrics_file(fold_name)
        if pred_path.exists() and metric_path.exists():
            print(f"{fold_name}: checkpoint exists, skipping", flush=True)
            continue
        pred, metrics = _score_fold(df, y, month, allowed_cols, fold_i, months)
        pred.to_csv(pred_path, index=False)
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"{fold_name}: DONE trades={metrics['trade_rows']} r={metrics['trade_total_r']} "
            f"wr={metrics['trade_win_pct']}%",
            flush=True,
        )

    pred_files = sorted(FOLD_DIR.glob("pred_*.csv"))
    metric_files = sorted(FOLD_DIR.glob("metrics_*.json"))
    predictions = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    fold_rows = []
    for p in metric_files:
        fold_rows.append(json.loads(p.read_text(encoding="utf-8")))
    folds = pd.DataFrame(fold_rows)
    _write_report(predictions, folds, allowed_cols, len(df), len(df.columns) - len(allowed_cols))
    print(f"Wrote {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
