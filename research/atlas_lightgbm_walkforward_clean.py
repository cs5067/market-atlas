#!/usr/bin/env python3
"""Offline multiclass LightGBM experiment with expanding monthly outer folds.

Predicts tick_race_winner (BUY/SELL/NONE) from prepared feature tables and
writes predictions, feature importance and calibration reports comparable to
V3. Known session-extrema and identifier features are excluded; running
session extrema replace some future-derived historical inputs.

The historical filename 'clean' refers to those specific exclusions. Inner
hyperparameter tuning uses ordinary KFold, not forward-chaining validation.
Outer chronology and name-based exclusions do not prove the pipeline free
of leakage. See docs/methodology.md for the retained research limitations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Suppress noisy LightGBM warnings about categorical column types.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import lightgbm as lgb
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/market_atlas/tick_winner_tables/TICK_RACE_LABELED_ALL_M5_EXTERNAL_THEORY_COLUMNS.parquet"
OUT_DIR = ROOT / "data/market_atlas/atlas_lightgbm_walkforward_clean"
FOLD_DIR = OUT_DIR / "fold_predictions"

# ALL-HOURS CLEAN VARIANT: same leakage fixes as Asia-clean run, but no Asia hour filter.
# We test whether the model can find signal across the full 24-hour day once the
# asia_high/low leakage family is removed.
ASIA_HOURS = None  # disabled

REPORT_MD = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_WALKFORWARD_REPORT.md"
PREDICTIONS_CSV = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_PREDICTIONS.csv"
FOLDS_CSV = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_FOLDS.csv"
CALIBRATION_CSV = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_CALIBRATION_BY_SCORE.csv"
INPUT_LEDGER_CSV = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_INPUT_LEDGER.csv"
HYPERPARAM_JSON = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_HYPERPARAMS.json"
FEATURE_IMPORTANCE_CSV = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_FEATURE_IMPORTANCE.csv"
SUMMARY_JSON = OUT_DIR / "ATLAS_LIGHTGBM_CLEAN_SUMMARY.json"

OUTCOMES = ("BUY", "SELL", "NONE")
OUTCOME_TO_INT = {name: i for i, name in enumerate(OUTCOMES)}

# === leakage exclusion (identical rules to V3) ===

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
    # Absolute date/timestamp anchors slipped through the substring filter and
    # were the #1 gain in the first run, which made the model effectively encode
    # "what time period this bar is from" rather than what the bar actually shows.
    # Hour-of-day / day-of-week / session features are still allowed; raw
    # absolute dates and intra-bar tick timestamps are blocked.
    "atlas_day",
    "pressure_first_tick",
    "pressure_last_tick",
    # Monotonically increasing row index 0..N. The model can split on it to
    # detect "which chronological era this bar belongs to" and overfit to time
    # trends without learning anything about the bar's actual content.
    "atlas_bar_index",
    # === ASIA HIGH/LOW LEAKAGE (discovered by adversarial audit 2026-06-09) ===
    # These columns hold the FINAL Asia session high/low broadcast to every M5
    # bar inside the 00-05 UTC window. P(asia_high >= close) = 1.000 across all
    # 56,952 Asia rows and nunique == 1 per session. The model uses these to
    # "predict" price will reach the extremes — i.e. it is told the answer.
    # We block these and provide clean running-only replacements computed at
    # load time (see _add_running_extremes() below).
    "atlas_asia_high",
    "atlas_asia_low",
    "ext_dist_to_asia_high_atr",
    "ext_dist_to_asia_low_atr",
    "ext_abs_dist_to_asia_high_atr",
    "ext_abs_dist_to_asia_low_atr",
    "ext_near_asia_high_025atr",
    "ext_near_asia_low_025atr",
    "ext_break_asia_high_hold",
    "ext_break_asia_low_hold",
    "ext_sweep_asia_high_reject",
    "ext_sweep_asia_low_reclaim",
    "atlas_asia_location",
    # === PREV-DAY EXTREME FLAGS (also constant per day, broadcast-leaky) ===
    # Boolean flags computed using full-day lookback against prev day H/L.
    # The raw prev_high / prev_low columns themselves are fine (genuinely known
    # at start of today), but these derived flags are computed once per day and
    # broadcast back, which leaks intra-day future info.
    "ext_near_prev_high_025atr",
    "ext_near_prev_low_025atr",
    "ext_break_prev_high_hold",
    "ext_break_prev_low_hold",
    "ext_sweep_prev_high_reject",
    "ext_sweep_prev_low_reclaim",
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


def _month_key(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").astype(str)


def _add_running_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Add CLEAN replacements for the leaked atlas_asia_high/low feature family.

    Computes session-to-date running max/min of OHLC high/low *up to the bar
    before the current one* — so the value on the 00:00 bar uses no Asia data
    at all (defaults to the current bar's open), and the value on the 03:00
    bar uses only 00:00 through 02:55 closes.  No future leakage.

    Adds four numeric features:
      ext_dist_to_asia_runhigh_atr   = (Close - running_high) / atr
      ext_dist_to_asia_runlow_atr    = (Close - running_low)  / atr
      ext_abs_dist_to_asia_runhigh_atr
      ext_abs_dist_to_asia_runlow_atr

    The df must already have atlas_datetime parsed, atlas_High, atlas_Low,
    atlas_Close, atlas_atr present (verified in main()).
    """
    out = df.copy()
    ts = pd.to_datetime(out["atlas_datetime"], errors="coerce", utc=True)
    out["__date"] = ts.dt.date

    # cummax / cummin of high and low PER calendar day, shifted by 1 bar so the
    # value reflects what was known at the start of the current bar (not its
    # own high/low).
    grp_high = out.groupby("__date", sort=False)["atlas_High"]
    grp_low = out.groupby("__date", sort=False)["atlas_Low"]
    run_high = grp_high.cummax().shift(1)
    run_low = grp_low.cummin().shift(1)

    # On the first bar of each day, running extreme is undefined — fall back to
    # this bar's open so distance is zero rather than NaN.
    first_of_day = (out["__date"] != out["__date"].shift(1))
    run_high = run_high.where(~first_of_day, out["atlas_Open"])
    run_low = run_low.where(~first_of_day, out["atlas_Open"])

    atr = out["atlas_atr"].replace(0, np.nan)
    out["ext_dist_to_asia_runhigh_atr"] = (out["atlas_Close"] - run_high) / atr
    out["ext_dist_to_asia_runlow_atr"] = (out["atlas_Close"] - run_low) / atr
    out["ext_abs_dist_to_asia_runhigh_atr"] = out["ext_dist_to_asia_runhigh_atr"].abs()
    out["ext_abs_dist_to_asia_runlow_atr"] = out["ext_dist_to_asia_runlow_atr"].abs()

    out = out.drop(columns="__date")
    return out


# === feature preparation ===

def _prepare_features(df: pd.DataFrame, allowed_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return X with proper dtypes for LightGBM. Categorical columns become 'category' dtype.

    Builds a dict of series first, then concatenates once at the end to avoid the
    DataFrame-fragmentation perf hit you get from repeated insert.
    """
    cols: dict[str, pd.Series] = {}
    categorical_cols: list[str] = []

    for col in allowed_cols:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            cols[col] = s.astype("Int64").astype("category")
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            cols[col] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            text = s.astype(str)
            cols[col] = text.astype("category")
            categorical_cols.append(col)

    X = pd.concat(cols, axis=1)
    X.columns = list(cols.keys())
    return X, categorical_cols


# === hyperparameter tuning on first training fold ===

def _tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: list[str],
    rng_seed: int = 17,
) -> dict[str, Any]:
    """Inner 3-fold CV on the first walk-forward training set to pick hyperparameters.

    Searches a small grid of (num_leaves, learning_rate, min_data_in_leaf) and picks
    the combination that minimizes mean multi_logloss.  The
    grid is enumerated and the data picks.
    """
    grid = []
    for num_leaves in (31, 63, 127):
        for lr in (0.05, 0.1):
            for min_data in (50, 200, 500):
                grid.append({
                    "num_leaves": num_leaves,
                    "learning_rate": lr,
                    "min_data_in_leaf": min_data,
                })

    fixed_params = {
        "objective": "multiclass",
        "num_class": len(OUTCOMES),
        "metric": "multi_logloss",
        "verbose": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": rng_seed,
    }

    best_loss = float("inf")
    best_params: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    kf = KFold(n_splits=3, shuffle=False)

    for cfg in grid:
        fold_losses: list[float] = []
        for tr_idx, va_idx in kf.split(X_train):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_va = y_train[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
            dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain, free_raw_data=False)
            params = {**fixed_params, **cfg}

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=600,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
            )
            fold_losses.append(float(model.best_score["valid_0"]["multi_logloss"]))

        mean_loss = float(np.mean(fold_losses))
        cfg_record = {**cfg, "mean_loss": mean_loss, "fold_losses": fold_losses}
        results.append(cfg_record)
        print(f"   tune cfg={cfg} mean_logloss={mean_loss:.5f}", flush=True)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = {**fixed_params, **cfg}

    return {
        "best": best_params,
        "best_mean_logloss": best_loss,
        "grid_results": results,
    }


# === per-fold training + scoring ===

def _train_and_score_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cat_cols: list[str],
    params: dict[str, Any],
) -> tuple[np.ndarray, lgb.Booster, int]:
    # Inner validation for early stopping: last 10% of train (chronological)
    cut = int(len(X_train) * 0.9)
    X_tr_inner = X_train.iloc[:cut]
    y_tr_inner = y_train[:cut]
    X_va_inner = X_train.iloc[cut:]
    y_va_inner = y_train[cut:]

    dtrain = lgb.Dataset(X_tr_inner, label=y_tr_inner, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va_inner, label=y_va_inner, categorical_feature=cat_cols, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
    )

    probs = model.predict(X_test, num_iteration=model.best_iteration)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(OUTCOMES))
    return probs, model, model.best_iteration


def _predictions_dataframe(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    probs: np.ndarray,
    fold_name: str,
) -> pd.DataFrame:
    pred_idx = probs.argmax(axis=1)
    predicted_outcome = np.array(OUTCOMES, dtype=object)[pred_idx]
    trade_side_idx = probs[:, :2].argmax(axis=1)
    chosen_side = np.where(trade_side_idx == 0, "BUY", "SELL")
    trade_score = np.where(trade_side_idx == 0, probs[:, 0], probs[:, 1]) * 100.0

    winner = np.array(OUTCOMES, dtype=object)[y_test]
    would_trade = np.isin(predicted_outcome, ["BUY", "SELL"])
    trade_win = winner == chosen_side
    trade_r = np.where(would_trade, np.where(trade_win, 2.0, -1.0), 0.0)

    out = pd.DataFrame({
        "fold": fold_name,
        "row_index": test_idx,
        "atlas_datetime": df.loc[test_idx, "atlas_datetime"].to_numpy(),
        "winner": winner,
        "p_buy": probs[:, 0] * 100.0,
        "p_sell": probs[:, 1] * 100.0,
        "p_none": probs[:, 2] * 100.0,
        "predicted_outcome": predicted_outcome,
        "chosen_trade_side": chosen_side,
        "trade_score_pct": trade_score,
        "would_trade": would_trade,
        "trade_win": trade_win,
        "trade_r": trade_r,
    })
    return out


def _metrics_for_predictions(pred: pd.DataFrame) -> dict[str, Any]:
    if len(pred) == 0:
        return {
            "rows": 0,
            "trade_rows": 0,
            "predicted_none_rows": 0,
            "multi_class_accuracy_pct": 0.0,
            "trade_win_pct": 0.0,
            "trade_total_r": 0.0,
            "trade_mean_r": 0.0,
            "trade_max_dd_r": 0.0,
        }
    trade_mask = pred["would_trade"].astype(bool)
    total_r = float(pred["trade_r"].sum())
    eq = pred["trade_r"].cumsum()
    max_dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    return {
        "rows": int(len(pred)),
        "trade_rows": int(trade_mask.sum()),
        "predicted_none_rows": int((~trade_mask).sum()),
        "multi_class_accuracy_pct": round(float(pred["predicted_outcome"].eq(pred["winner"]).mean() * 100.0), 6),
        "trade_win_pct": round(float(pred.loc[trade_mask, "trade_win"].mean() * 100.0), 6) if trade_mask.any() else 0.0,
        "trade_total_r": round(total_r, 6),
        "trade_mean_r": round(float(pred.loc[trade_mask, "trade_r"].mean()), 6) if trade_mask.any() else 0.0,
        "trade_max_dd_r": round(max_dd, 6),
    }


def _calibration_by_score(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    p = pred.copy()
    p["trade_score_int"] = p["trade_score_pct"].round().clip(0, 100).astype(int)
    rows: list[dict[str, Any]] = []
    for score, sub in p.groupby("trade_score_int", observed=True):
        trade_sub = sub[sub["would_trade"]]
        n_trades = len(trade_sub)
        if n_trades > 0:
            wins = trade_sub["trade_win"].sum()
            actual_chosen_side_win_pct = float(wins) / n_trades * 100.0
            trade_total_r = float(trade_sub["trade_r"].sum())
            trade_mean_r = float(trade_sub["trade_r"].mean())
        else:
            actual_chosen_side_win_pct = 0.0
            trade_total_r = 0.0
            trade_mean_r = 0.0
        rows.append({
            "score": int(score),
            "rows": int(len(sub)),
            "trade_rows": int(n_trades),
            "actual_chosen_side_win_pct": round(actual_chosen_side_win_pct, 6),
            "actual_none_pct": round(float(sub["winner"].eq("NONE").mean() * 100.0), 6),
            "trade_total_r": round(trade_total_r, 6),
            "trade_mean_r": round(trade_mean_r, 6),
        })
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int) -> str:
    if df.empty:
        return "_(no rows)_"
    view = df.head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in view.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fold_pred_file(fold: str) -> Path:
    return FOLD_DIR / f"pred_{fold}.csv"


def _fold_metrics_file(fold: str) -> Path:
    return FOLD_DIR / f"metrics_{fold}.json"


def _fold_importance_file(fold: str) -> Path:
    return FOLD_DIR / f"importance_{fold}.csv"


def _write_report(predictions: pd.DataFrame, folds: pd.DataFrame, allowed_cols: list[str], total_rows: int, blocked_cols: int, params: dict[str, Any], tune_loss: float) -> None:
    metrics = _metrics_for_predictions(predictions)
    calibration = _calibration_by_score(predictions)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    summary = {
        "source": str(SOURCE),
        "rows_total": int(total_rows),
        "input_columns": int(len(allowed_cols)),
        "blocked_columns": int(blocked_cols),
        "folds": int(len(folds)),
        "overall": metrics,
        "hyperparameters": params,
        "best_inner_cv_logloss": tune_loss,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    calib_view = calibration[calibration["trade_rows"].gt(0)].sort_values("score") if not calibration.empty else calibration

    lines = [
        "# Atlas LightGBM Walk-Forward Report",
        "",
        "Offline only. Live bot was not changed.",
        "",
        "## Setup",
        "",
        f"- Target: tick_race_winner (BUY / SELL / NONE)",
        f"- Same parquet source as V3.",
        f"- Same leakage exclusions as V3.",
        f"- Same expanding monthly walk-forward as V3.",
        f"- Architecture: LightGBM multiclass classifier.",
        f"- Hyperparameters selected by ordinary K-fold CV within the first outer training fold.",
        f"- Categorical columns: native LightGBM categorical (not one-hot).",
        "",
        "## Data",
        "",
        f"- Rows total: `{total_rows:,}`",
        f"- Input columns: `{len(allowed_cols):,}`",
        f"- Blocked future/leakage columns: `{blocked_cols:,}`",
        f"- Completed folds: `{len(folds):,}`",
        "",
        "## Chosen Hyperparameters",
        "",
        "```json",
        json.dumps({k: v for k, v in params.items() if k in ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq")}, indent=2),
        "```",
        f"Best inner-CV multi-logloss: `{tune_loss:.5f}`.",
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
        _md_table(
            calib_view if not calib_view.empty else pd.DataFrame(),
            ["score", "rows", "trade_rows", "actual_chosen_side_win_pct", "actual_none_pct", "trade_total_r", "trade_mean_r"],
            120,
        ),
        "",
        "## Fold Results",
        "",
        _md_table(
            folds.sort_values("fold"),
            ["fold", "train_rows", "test_rows", "trade_rows", "trade_win_pct", "trade_total_r", "trade_mean_r", "trade_max_dd_r"],
            120,
        ),
        "",
        "## Files",
        "",
        f"- Predictions: `{PREDICTIONS_CSV}`",
        f"- Folds: `{FOLDS_CSV}`",
        f"- Calibration: `{CALIBRATION_CSV}`",
        f"- Feature importance (avg across folds): `{FEATURE_IMPORTANCE_CSV}`",
        f"- Hyperparameter tuning log: `{HYPERPARAM_JSON}`",
        f"- Summary JSON: `{SUMMARY_JSON}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] reading parquet...", flush=True)
    df_full = pd.read_parquet(SOURCE).reset_index(drop=True)
    n_full = len(df_full)
    # ALL-HOURS CLEAN VARIANT: no Asia hour filter applied. Use the entire dataset.
    df = df_full
    print(f"[{time.strftime('%H:%M:%S')}] all-hours mode: using all {len(df):,} rows", flush=True)

    # === CLEAN FEATURES: replace leaky atlas_asia_high/low family ===
    # Add running session-to-date max/min features (computed only from bars
    # strictly before the current one). The leaky originals are blocked via
    # EXCLUDE_EXACT so the model will use these clean replacements instead.
    print(f"[{time.strftime('%H:%M:%S')}] computing clean running Asia extremes...", flush=True)
    df = _add_running_extremes(df)
    # Sanity check: new column should vary within a day (not be constant)
    _date_check = pd.to_datetime(df["atlas_datetime"], utc=True, errors="coerce").dt.date
    _nu_per_day = df.groupby(_date_check)["ext_dist_to_asia_runhigh_atr"].nunique().median()
    print(f"[{time.strftime('%H:%M:%S')}] sanity: ext_dist_to_asia_runhigh_atr median nunique per day = {_nu_per_day} (should be high, e.g. > 30)", flush=True)
    if _nu_per_day < 10:
        raise RuntimeError("Running extreme features appear to be constant per day — pipeline is broken.")

    y = _winner_series(df)
    y_int = y.map(OUTCOME_TO_INT).astype(np.int8).to_numpy()
    ts = pd.to_datetime(df["atlas_datetime"], errors="coerce", utc=True)
    month = _month_key(ts)
    months = sorted(month.dropna().unique().tolist())

    allowed_cols: list[str] = []
    blocked: list[tuple[str, str]] = []
    for col in df.columns:
        reason = _leakage_reason(col)
        if reason is None:
            allowed_cols.append(col)
        else:
            blocked.append((col, reason))

    pd.DataFrame(
        [{"column": col, "role": "input", "reason": "decision_time_or_current_bar_value"} for col in allowed_cols]
        + [{"column": col, "role": "blocked_answer_key_or_identifier", "reason": reason} for col, reason in blocked]
    ).to_csv(INPUT_LEDGER_CSV, index=False)

    print(f"[{time.strftime('%H:%M:%S')}] rows={len(df)} allowed_cols={len(allowed_cols)} blocked={len(blocked)} months={len(months)}", flush=True)

    X_full, cat_cols = _prepare_features(df, allowed_cols)
    print(f"[{time.strftime('%H:%M:%S')}] feature prep done; categorical={len(cat_cols)}/{len(allowed_cols)}", flush=True)

    # === hyperparameter tuning on the first walk-forward training set ===
    # Train = months[:1], i.e. before the first test month.  We use months[:max(2,...)] so the CV has enough data.
    if not HYPERPARAM_JSON.exists():
        tune_train_months = months[: max(6, len(months) // 4)]
        tune_train_mask = month.isin(tune_train_months).to_numpy()
        tune_idx = np.flatnonzero(tune_train_mask)
        print(f"[{time.strftime('%H:%M:%S')}] tuning on {tune_train_months[0]}..{tune_train_months[-1]} ({len(tune_idx)} rows)", flush=True)
        tune_result = _tune_hyperparameters(X_full.iloc[tune_idx], y_int[tune_idx], cat_cols)
        HYPERPARAM_JSON.write_text(json.dumps(tune_result, indent=2), encoding="utf-8")
    else:
        tune_result = json.loads(HYPERPARAM_JSON.read_text(encoding="utf-8"))
        print(f"[{time.strftime('%H:%M:%S')}] loaded cached hyperparameters", flush=True)

    best_params = tune_result["best"]
    best_loss = tune_result["best_mean_logloss"]
    print(f"[{time.strftime('%H:%M:%S')}] best params: {best_params}  cv_logloss={best_loss:.5f}", flush=True)

    importance_accum: dict[str, list[float]] = {}

    # === walk-forward ===
    for fold_i in range(1, len(months)):
        fold_name = months[fold_i]
        pred_path = _fold_pred_file(fold_name)
        metric_path = _fold_metrics_file(fold_name)
        if pred_path.exists() and metric_path.exists():
            print(f"[{time.strftime('%H:%M:%S')}] {fold_name}: cached, skipping", flush=True)
            continue

        train_mask = month.isin(months[:fold_i]).to_numpy()
        test_mask = month.eq(fold_name).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(test_idx) == 0:
            continue

        t0 = time.time()
        probs, model, best_iter = _train_and_score_fold(
            X_full.iloc[train_idx],
            y_int[train_idx],
            X_full.iloc[test_idx],
            y_int[test_idx],
            cat_cols,
            best_params,
        )
        fold_time = time.time() - t0

        pred = _predictions_dataframe(df, test_idx, y_int[test_idx], probs, fold_name)
        metrics = _metrics_for_predictions(pred)
        metrics.update({
            "fold": fold_name,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "best_iteration": int(best_iter),
            "fold_seconds": round(fold_time, 2),
        })
        pred.to_csv(pred_path, index=False)
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # Feature importance
        gain = model.feature_importance(importance_type="gain")
        for name, g in zip(allowed_cols, gain):
            importance_accum.setdefault(name, []).append(float(g))
        pd.DataFrame({"feature": allowed_cols, "gain": gain}).sort_values("gain", ascending=False).to_csv(
            _fold_importance_file(fold_name), index=False
        )

        print(
            f"[{time.strftime('%H:%M:%S')}] {fold_name} DONE  trades={metrics['trade_rows']} "
            f"wr={metrics['trade_win_pct']:.2f}%  R={metrics['trade_total_r']:.2f}  "
            f"acc={metrics['multi_class_accuracy_pct']:.2f}%  best_iter={best_iter}  "
            f"time={fold_time:.1f}s",
            flush=True,
        )

    # === aggregate ===
    pred_files = sorted(FOLD_DIR.glob("pred_*.csv"))
    metric_files = sorted(FOLD_DIR.glob("metrics_*.json"))
    predictions = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    fold_rows = []
    for p in metric_files:
        fold_rows.append(json.loads(p.read_text(encoding="utf-8")))
    folds = pd.DataFrame(fold_rows)

    # Average feature importance
    if importance_accum:
        importance_df = pd.DataFrame([
            {"feature": name, "mean_gain": float(np.mean(gains)), "folds": len(gains)}
            for name, gains in importance_accum.items()
        ]).sort_values("mean_gain", ascending=False)
        importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    _write_report(predictions, folds, allowed_cols, len(df), len(df.columns) - len(allowed_cols), best_params, best_loss)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
