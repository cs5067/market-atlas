#!/usr/bin/env python3
"""Atlas probability equation V3 walk-forward.

Offline only.  This script does not touch the live bot.

V3 principle:

- Every decision-time input column is used.
- Explicit future/outcome columns are blocked by name; upstream feature causality is not guaranteed.
- For each training window, every component's vote is:

      vote(outcome) = (P(outcome | component) - P_base(outcome)) * fire_rate(component)

  No smoothing constant.  No min-N threshold.  Rarity is part of the vote through
  fire_rate, while the raw when-seen probability remains available in the ledger.

- The equation is learned on past rows only and scored on future rows only.
- The final calibration curve is built only from out-of-sample scores.

This first walk-forward applies the V3 component equation to all columns,
including existing composite columns already in the Atlas table.  It does not
yet generate all possible pair/triple interaction columns at scale; that would
be a separate interaction-index engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/market_atlas/tick_winner_tables/TICK_RACE_LABELED_ALL_M5_EXTERNAL_THEORY_COLUMNS.parquet"
OUT_DIR = ROOT / "data/market_atlas/atlas_probability_equation_v3"

REPORT_MD = OUT_DIR / "ATLAS_PROBABILITY_EQUATION_V3_WALKFORWARD_REPORT.md"
PREDICTIONS_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_WALKFORWARD_PREDICTIONS.csv"
FOLDS_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_WALKFORWARD_FOLDS.csv"
CALIBRATION_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_CALIBRATION_BY_SCORE.csv"
COMPONENT_SAMPLE_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_COMPONENT_SAMPLE.csv"
INPUT_LEDGER_CSV = OUT_DIR / "ATLAS_PROBABILITY_V3_INPUT_LEDGER.csv"
SUMMARY_JSON = OUT_DIR / "ATLAS_PROBABILITY_V3_SUMMARY.json"

OUTCOMES = ("BUY", "SELL", "NONE")
EPS = 1e-9

EXCLUDE_EXACT = {
    "tick_race_winner",
    "tick_race_reason",
    "tick_race_tie",
    "source_tick_month",
    "signal_bar_time",
    "entry_target_time",
    "entry_tick_time",
    "atlas_datetime",
    "prev_m5_datetime",
    "signal_m5_datetime",
}
EXCLUDE_PREFIXES = ("next_m5_",)
EXCLUDE_SUBSTRINGS = (
    "clean_plus_2r",
    "plus_2r",
    "minus_1r",
    "elapsed_to",
    "time_to_",
    "mfe",
    "mae",
    "final_r",
    "static_3r",
    "ambiguous",
    "sl_then",
    "09_then",
    "better_than",
    "neither_valid",
    "both_valid",
    "race_",
    "best_side",
    "edge_gap",
    "horizon_tick",
    "buy_minus_sell",
    "atlas_horizon",
)


@dataclass
class TransformSpec:
    kind: str
    edges: list[float] | None = None
    as_datetime: bool = False


def _leakage_reason(col: str) -> str | None:
    if col in EXCLUDE_EXACT:
        return "answer_key_or_identifier"
    if col.startswith(EXCLUDE_PREFIXES):
        return "future_next_bar_column"
    low = col.lower()
    for part in EXCLUDE_SUBSTRINGS:
        if part in low:
            return f"future_outcome_contains_{part}"
    return None


def _winner_series(df: pd.DataFrame) -> pd.Series:
    out = df["tick_race_winner"].astype(str).str.upper()
    return out.where(out.isin(OUTCOMES), "NONE")


def _clean_value(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a", "<na>"}:
        return "MISSING"
    return text.replace("|", "/")


def _datetime_bucket(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    comp = pd.Series("MISSING", index=s.index, dtype="object")
    valid = parsed.notna()
    minute_of_day = parsed.dt.hour * 60 + parsed.dt.minute
    m5_slot = (minute_of_day // 5).astype("Int64")
    sec_in_5m = ((parsed.dt.minute % 5) * 60 + parsed.dt.second).astype("Int64")
    sec30 = (sec_in_5m // 30).astype("Int64")
    comp.loc[valid] = (
        "DOW=" + parsed.dt.dayofweek.astype("Int64").astype(str)
        + "|MONTH=" + parsed.dt.month.astype("Int64").astype(str)
        + "|M5_SLOT=" + m5_slot.astype(str)
        + "|SEC30=" + sec30.astype(str)
    ).loc[valid]
    return comp


def _fit_transform_spec(s: pd.Series) -> TransformSpec:
    non_null = s.dropna()
    unique_n = int(non_null.nunique(dropna=True))
    if not pd.api.types.is_numeric_dtype(s) and unique_n > 48:
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        if len(parsed) and float(parsed.notna().mean()) >= 0.80:
            return TransformSpec("datetime_bucket", as_datetime=True)
    if pd.api.types.is_bool_dtype(s) or unique_n <= 2:
        return TransformSpec("category")
    if pd.api.types.is_numeric_dtype(s) and unique_n > 20:
        numeric = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = numeric.dropna()
        if valid.nunique() < 4:
            return TransformSpec("category")
        try:
            _, edges = pd.qcut(valid, q=min(10, valid.nunique()), duplicates="drop", retbins=True)
        except ValueError:
            _, edges = pd.cut(valid, bins=min(10, valid.nunique()), duplicates="drop", retbins=True)
        edges = np.unique(edges.astype(float))
        if len(edges) < 2:
            return TransformSpec("category")
        edges[0] = -np.inf
        edges[-1] = np.inf
        return TransformSpec("numeric_quantile_bucket", edges=edges.tolist())
    return TransformSpec("category")


def _apply_transform(s: pd.Series, spec: TransformSpec) -> pd.Series:
    if spec.kind == "datetime_bucket":
        return _datetime_bucket(s)
    if spec.kind == "numeric_quantile_bucket" and spec.edges:
        numeric = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        comp = pd.Series("MISSING", index=s.index, dtype="object")
        valid = numeric.notna()
        if valid.any():
            comp.loc[valid] = pd.cut(numeric.loc[valid], bins=spec.edges, include_lowest=True, duplicates="drop").astype(str)
        return comp
    return s.map(_clean_value)


def _base_probs(y: pd.Series) -> dict[str, float]:
    n = len(y)
    return {outcome: float(y.eq(outcome).sum()) / max(n, 1) for outcome in OUTCOMES}


def _learn_component_votes(
    train_components: pd.Series,
    train_y: pd.Series,
    base_p: dict[str, float],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    total = len(train_components)
    tmp = pd.DataFrame({"component": train_components.astype(str), "winner": train_y})
    grouped = tmp.groupby(["component", "winner"], observed=True).size().unstack(fill_value=0)
    votes: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for value, counts in grouped.iterrows():
        n = int(counts.sum())
        fire_rate = n / max(total, 1)
        raw = np.array([float(counts.get(outcome, 0)) / max(n, 1) for outcome in OUTCOMES], dtype=float)
        base = np.array([base_p[outcome] for outcome in OUTCOMES], dtype=float)
        effect = raw - base
        vote = effect * fire_rate
        votes[str(value)] = vote
        buy_exp = 3.0 * raw[0] - 1.0
        sell_exp = 3.0 * raw[1] - 1.0
        best_side = "BUY" if buy_exp >= sell_exp else "SELL"
        rows.append(
            {
                "component_value": str(value),
                "n": n,
                "fire_rate_pct": fire_rate * 100.0,
                "raw_buy_pct_when_seen": raw[0] * 100.0,
                "raw_sell_pct_when_seen": raw[1] * 100.0,
                "raw_none_pct_when_seen": raw[2] * 100.0,
                "buy_effect_pct": effect[0] * 100.0,
                "sell_effect_pct": effect[1] * 100.0,
                "none_effect_pct": effect[2] * 100.0,
                "buy_vote_pct_points": vote[0] * 100.0,
                "sell_vote_pct_points": vote[1] * 100.0,
                "none_vote_pct_points": vote[2] * 100.0,
                "best_trade_side_when_seen": best_side,
                "best_trade_expected_r_when_seen": max(buy_exp, sell_exp),
                "coverage_value_r": max(buy_exp, sell_exp) * fire_rate,
            }
        )
    return votes, pd.DataFrame(rows)


def _normalize_scores(raw_scores: np.ndarray) -> tuple[np.ndarray, int]:
    # The math guarantees raw_scores sum to 1 across outcomes per row (each
    # vote contributes 0 to the cross-outcome sum because effects sum to 0).
    # So when all values are non-negative, the row IS already a valid
    # probability distribution and we leave it alone.
    #
    # Only when an outcome's accumulated votes drive it below zero do we need
    # to handle it. In that case we shift UP by exactly the most-negative
    # amount in that row (a row-specific shift, no global constant), so the
    # smallest value lands at 0. Then renormalize.
    #
    # Critically: rows where no outcome went negative are untouched. The old
    # code clipped negatives to 1e-9 and renormalized, which made the largest
    # outcome dominate at ~100% any time even a slight negative occurred.
    clipped_count = int((raw_scores < 0).any(axis=1).sum())
    min_per_row = raw_scores.min(axis=1, keepdims=True)
    shift = np.minimum(min_per_row, 0.0)  # 0 if min>=0, else the negative
    shifted = raw_scores - shift          # subtracts non-positive => raises floor to 0
    denom = shifted.sum(axis=1, keepdims=True)
    uniform = np.full_like(shifted, 1.0 / shifted.shape[1])
    probs = np.where(denom > 0, np.divide(shifted, denom, out=uniform.copy(), where=denom > 0), uniform)
    return probs, clipped_count


def _month_key(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").astype(str)


def _metrics_for_predictions(pred: pd.DataFrame) -> dict[str, Any]:
    total_r = float(pred["trade_r"].sum()) if len(pred) else 0.0
    eq = pred["trade_r"].cumsum() if len(pred) else pd.Series(dtype=float)
    max_dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    return {
        "rows": int(len(pred)),
        "trade_rows": int(pred["would_trade"].sum()) if len(pred) else 0,
        "predicted_none_rows": int(pred["predicted_outcome"].eq("NONE").sum()) if len(pred) else 0,
        "multi_class_accuracy_pct": round(float(pred["predicted_outcome"].eq(pred["winner"]).mean() * 100.0), 6) if len(pred) else 0.0,
        "trade_win_pct": round(float(pred.loc[pred["would_trade"], "trade_win"].mean() * 100.0), 6) if int(pred["would_trade"].sum()) else 0.0,
        "trade_total_r": round(total_r, 6),
        "trade_mean_r": round(float(pred.loc[pred["would_trade"], "trade_r"].mean()), 6) if int(pred["would_trade"].sum()) else 0.0,
        "trade_max_dd_r": round(max_dd, 6),
    }


def _calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pred.empty:
        return pd.DataFrame(rows)
    pred = pred.copy()
    pred["trade_score_int"] = pred["trade_score_pct"].round().clip(0, 100).astype(int)
    for score, sub in pred.groupby("trade_score_int"):
        trade = sub[sub["would_trade"]]
        rows.append(
            {
                "score": int(score),
                "rows": int(len(sub)),
                "trade_rows": int(len(trade)),
                "actual_chosen_side_win_pct": round(float(trade["trade_win"].mean() * 100.0), 6) if len(trade) else 0.0,
                "actual_none_pct": round(float(sub["winner"].eq("NONE").mean() * 100.0), 6),
                "avg_pred_buy_pct": round(float(sub["p_buy"].mean()), 6),
                "avg_pred_sell_pct": round(float(sub["p_sell"].mean()), 6),
                "avg_pred_none_pct": round(float(sub["p_none"].mean()), 6),
                "trade_total_r": round(float(trade["trade_r"].sum()), 6) if len(trade) else 0.0,
                "trade_mean_r": round(float(trade["trade_r"].mean()), 6) if len(trade) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("score")


def _md_table(df: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.loc[:, cols].head(limit)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in sub.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SOURCE).reset_index(drop=True)
    y = _winner_series(df)
    ts = pd.to_datetime(df["atlas_datetime"], errors="coerce", utc=True)
    month = _month_key(ts)

    allowed_cols = [col for col in df.columns if _leakage_reason(col) is None]
    input_ledger = pd.DataFrame(
        [
            {
                "column": col,
                "role": "input" if col in allowed_cols else "blocked_answer_key_or_identifier",
                "reason": "decision_time_or_current_bar_value" if col in allowed_cols else _leakage_reason(col),
            }
            for col in df.columns
        ]
    )
    input_ledger.to_csv(INPUT_LEDGER_CSV, index=False)

    months = sorted(month.dropna().unique().tolist())
    all_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    component_samples: list[pd.DataFrame] = []

    # Expanding walk-forward: train on all prior months, test the next month.
    for fold_i in range(1, len(months)):
        train_mask = month.isin(months[:fold_i])
        test_mask = month.eq(months[fold_i])
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_y = y.iloc[train_idx].reset_index(drop=True)
        test_y = y.iloc[test_idx].reset_index(drop=True)
        base_p = _base_probs(train_y)
        base_arr = np.array([base_p[outcome] for outcome in OUTCOMES], dtype=float)
        raw_scores = np.tile(base_arr, (len(test_idx), 1))

        fold_component_rows: list[pd.DataFrame] = []
        for col in allowed_cols:
            spec = _fit_transform_spec(df.loc[train_idx, col])
            train_comp = _apply_transform(df.loc[train_idx, col], spec).reset_index(drop=True)
            test_comp = _apply_transform(df.loc[test_idx, col], spec).reset_index(drop=True)
            votes, comp_stats = _learn_component_votes(train_comp, train_y, base_p)
            if len(component_samples) < 3 and not comp_stats.empty:
                sample = comp_stats.assign(fold=months[fold_i], feature=col).sort_values(
                    ["coverage_value_r", "best_trade_expected_r_when_seen"], ascending=[False, False]
                ).head(20)
                fold_component_rows.append(sample)
            mapped = np.zeros((len(test_idx), 3), dtype=float)
            for value, vote in votes.items():
                m = test_comp.eq(value).to_numpy()
                if m.any():
                    mapped[m, :] = vote
            raw_scores += mapped
        probs, clipped_rows = _normalize_scores(raw_scores)

        out = pd.DataFrame(
            {
                "fold": months[fold_i],
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
        all_predictions.append(out)

        m = _metrics_for_predictions(out)
        m.update(
            {
                "fold": months[fold_i],
                "train_months": fold_i,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "clipped_rows": clipped_rows,
                "train_buy_base_pct": round(base_p["BUY"] * 100.0, 6),
                "train_sell_base_pct": round(base_p["SELL"] * 100.0, 6),
                "train_none_base_pct": round(base_p["NONE"] * 100.0, 6),
            }
        )
        fold_rows.append(m)
        if fold_component_rows:
            component_samples.append(pd.concat(fold_component_rows, ignore_index=True))
        print(
            f"fold {months[fold_i]} rows={len(test_idx)} trades={m['trade_rows']} "
            f"r={m['trade_total_r']} wr={m['trade_win_pct']}%",
            flush=True,
        )

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    calibration = _calibration(predictions)
    component_sample = pd.concat(component_samples, ignore_index=True) if component_samples else pd.DataFrame()

    predictions.to_csv(PREDICTIONS_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    component_sample.to_csv(COMPONENT_SAMPLE_CSV, index=False)

    metrics = _metrics_for_predictions(predictions)
    summary = {
        "source": str(SOURCE),
        "rows_total": int(len(df)),
        "input_columns": int(len(allowed_cols)),
        "blocked_columns": int(len(df.columns) - len(allowed_cols)),
        "folds": int(len(folds)),
        "overall": metrics,
        "formula": {
            "component_effect": "P_train(outcome|component)-P_train(outcome)",
            "component_vote": "component_effect * component_fire_rate",
            "row_raw_probability": "P_train(outcome)+sum(active_component_votes)",
            "row_probability": "clip to probability bounds and normalize",
            "calibration": "built only from out-of-sample predictions",
        },
        "outputs": {
            "predictions": str(PREDICTIONS_CSV),
            "folds": str(FOLDS_CSV),
            "calibration": str(CALIBRATION_CSV),
            "component_sample": str(COMPONENT_SAMPLE_CSV),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    calib_nonzero = calibration[calibration["trade_rows"].gt(0)].copy()
    calib_view = calib_nonzero.sort_values("score")
    fold_view = folds.sort_values("fold")

    lines = [
        "# Atlas Probability Equation V3 Walk-Forward Report",
        "",
        "Offline only. Live bot was not changed.",
        "",
        "## Formula",
        "",
        "```text",
        "effect(component, outcome)",
        "  = P_train(outcome | component) - P_train(outcome)",
        "",
        "fire_rate(component)",
        "  = rows_with_component / train_rows",
        "",
        "vote(component, outcome)",
        "  = effect(component, outcome) * fire_rate(component)",
        "",
        "raw_probability(row, outcome)",
        "  = P_train(outcome) + sum(active component votes)",
        "",
        "final_probability",
        "  = raw_probability clipped to probability bounds and normalized",
        "```",
        "",
        "No smoothing constants. No min-N threshold. No same-data judging.",
        "",
        "## Data",
        "",
        f"- Rows total: `{len(df):,}`",
        f"- Input columns: `{len(allowed_cols):,}`",
        f"- Blocked future/leakage columns: `{len(df.columns) - len(allowed_cols):,}`",
        f"- Walk-forward folds: `{len(folds):,}`",
        "- Fold design: train on all prior calendar months, test the next unseen month.",
        "",
        "## Overall Out-Of-Sample Result",
        "",
        f"- Scored rows: `{metrics['rows']:,}`",
        f"- Rows where equation chose BUY/SELL over NONE: `{metrics['trade_rows']:,}`",
        f"- Rows where equation chose NONE: `{metrics['predicted_none_rows']:,}`",
        f"- Multiclass accuracy: `{metrics['multi_class_accuracy_pct']:.2f}%`",
        f"- Trade-side win rate: `{metrics['trade_win_pct']:.2f}%`",
        f"- Trade total R if taking every BUY/SELL prediction: `{metrics['trade_total_r']:.2f}R`",
        f"- Trade mean R: `{metrics['trade_mean_r']:.4f}R`",
        f"- Trade max drawdown: `{metrics['trade_max_dd_r']:.2f}R`",
        "",
        "## Calibration By Score",
        "",
        "This is the curve we need before assigning grades. `score` is the predicted BUY/SELL side percentage rounded to 0-100.",
        "",
        _md_table(
            calib_view,
            ["score", "rows", "trade_rows", "actual_chosen_side_win_pct", "actual_none_pct", "trade_total_r", "trade_mean_r"],
            80,
        ),
        "",
        "## Fold Results",
        "",
        _md_table(
            fold_view,
            ["fold", "train_rows", "test_rows", "trade_rows", "trade_win_pct", "trade_total_r", "trade_mean_r", "trade_max_dd_r"],
            80,
        ),
        "",
        "## Files",
        "",
        f"- Predictions: `{PREDICTIONS_CSV}`",
        f"- Folds: `{FOLDS_CSV}`",
        f"- Calibration curve: `{CALIBRATION_CSV}`",
        f"- Component sample: `{COMPONENT_SAMPLE_CSV}`",
        f"- Input ledger: `{INPUT_LEDGER_CSV}`",
        f"- Summary JSON: `{SUMMARY_JSON}`",
        "",
        "## Important Limitation",
        "",
        "This V3 walk-forward uses all visible columns and existing composite columns, but it does not yet generate every possible pair/triple interaction at scale. The next interaction engine should add combo residual votes with the same residual-vote formula:",
        "",
        "```text",
        "combo_extra = observed_combo_effect - expected_from_parts",
        "combo_vote  = combo_extra * combo_fire_rate",
        "```",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
