#!/usr/bin/env python3
"""Sequence-story Atlas walk-forward.

Offline sequence-pattern evaluation:

    Watch the predicted market story unfold first.
    Then trade only when a proven story transition appears.

It does not import or change the live bot. Each fold mines clue cards from the
past, builds predicted phase clues from those cards, validates story patterns on
that same past window, then applies only the surviving stories to future rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import atlas_bot_walkforward as v1


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data/market_atlas/atlas_sequence_story"
TRADES_CSV = OUT_DIR / "ATLAS_SEQUENCE_STORY_WALKFORWARD_TRADES.csv"
FOLDS_CSV = OUT_DIR / "ATLAS_SEQUENCE_STORY_WALKFORWARD_FOLDS.csv"
STORIES_CSV = OUT_DIR / "ATLAS_SEQUENCE_STORY_WALKFORWARD_STORIES.csv"
REPORT_MD = OUT_DIR / "ATLAS_SEQUENCE_STORY_WALKFORWARD_REPORT.md"


@dataclass(frozen=True)
class StoryProfile:
    name: str
    top_k_cards: int
    min_train_n: int
    min_train_success: float
    min_train_mean_r: float
    min_side_gap_r: float
    cooldown_bars: int
    max_trades_per_day: int
    allow_start_only: bool


PROFILES = [
    StoryProfile(
        "story_strict",
        top_k_cards=5,
        min_train_n=18,
        min_train_success=0.48,
        min_train_mean_r=0.32,
        min_side_gap_r=0.08,
        cooldown_bars=4,
        max_trades_per_day=5,
        allow_start_only=False,
    ),
    StoryProfile(
        "story_balanced",
        top_k_cards=10,
        min_train_n=30,
        min_train_success=0.43,
        min_train_mean_r=0.20,
        min_side_gap_r=0.06,
        cooldown_bars=3,
        max_trades_per_day=8,
        allow_start_only=False,
    ),
    StoryProfile(
        "story_wide",
        top_k_cards=20,
        min_train_n=45,
        min_train_success=0.39,
        min_train_mean_r=0.12,
        min_side_gap_r=0.05,
        cooldown_bars=3,
        max_trades_per_day=10,
        allow_start_only=False,
    ),
    StoryProfile(
        "story_plus_start_fallback",
        top_k_cards=5,
        min_train_n=80,
        min_train_success=0.40,
        min_train_mean_r=0.15,
        min_side_gap_r=0.06,
        cooldown_bars=3,
        max_trades_per_day=8,
        allow_start_only=True,
    ),
]


def _rank_cards(cards: list[v1.EvidenceCard], target: str) -> list[v1.EvidenceCard]:
    sub = [card for card in cards if card.target == target]
    if target in {"BUY_DIRECT", "SELL_DIRECT"}:
        return sorted(sub, key=lambda c: (c.mean_r, c.win_pct, c.n), reverse=True)
    return sorted(sub, key=lambda c: (c.lift, c.target_pct, c.n), reverse=True)


def _union_mask(cards: list[v1.EvidenceCard], target: str, df: pd.DataFrame, top_k: int) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    for card in _rank_cards(cards, target)[:top_k]:
        mask |= card.mask(df)
    return mask


def _prev_any(mask: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(mask.astype(int))
    return s.shift(1).rolling(window, min_periods=1).max().fillna(0).to_numpy(bool)


def _prev_count(mask: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(mask.astype(int))
    return s.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy(int)


def _state_from_cards(cards: list[v1.EvidenceCard], df: pd.DataFrame, top_k: int) -> dict[str, np.ndarray]:
    buy_start = _union_mask(cards, "BUY_START_REAL2", df, top_k)
    sell_start = _union_mask(cards, "SELL_START_REAL2", df, top_k)
    none_start = _union_mask(cards, "NONE_START_REAL2", df, top_k)
    none_direct = _union_mask(cards, "NONE_DIRECT", df, top_k)
    buy_end = _union_mask(cards, "BUY_END_REAL2", df, top_k)
    sell_end = _union_mask(cards, "SELL_END_REAL2", df, top_k)
    none_end = _union_mask(cards, "NONE_END_REAL2", df, top_k)
    none_pressure = none_start | none_direct

    return {
        "buy_start": buy_start,
        "sell_start": sell_start,
        "none_start": none_start,
        "none_direct": none_direct,
        "buy_end": buy_end,
        "sell_end": sell_end,
        "none_end": none_end,
        "none_pressure": none_pressure,
        "none_pressure_prev5_3": _prev_count(none_pressure, 5) >= 3,
        "none_pressure_prev8_4": _prev_count(none_pressure, 8) >= 4,
        "none_end_prev1": _prev_any(none_end, 1),
        "none_end_prev2": _prev_any(none_end, 2),
        "none_end_prev3": _prev_any(none_end, 3),
        "buy_end_prev3": _prev_any(buy_end, 3),
        "sell_end_prev3": _prev_any(sell_end, 3),
    }


def _story_masks(state: dict[str, np.ndarray], allow_start_only: bool) -> list[tuple[str, str, np.ndarray]]:
    stories: list[tuple[str, str, np.ndarray]] = [
        (
            "NONE_ENDED_THEN_BUY_START_1",
            "BUY",
            state["none_end_prev1"] & state["buy_start"],
        ),
        (
            "NONE_ENDED_THEN_BUY_START_2",
            "BUY",
            state["none_end_prev2"] & state["buy_start"],
        ),
        (
            "NONE_ENDED_THEN_BUY_START_3",
            "BUY",
            state["none_end_prev3"] & state["buy_start"],
        ),
        (
            "NONE_PRESSURE_THEN_BUY_START_5",
            "BUY",
            state["none_pressure_prev5_3"] & state["buy_start"],
        ),
        (
            "NONE_PRESSURE_THEN_BUY_START_8",
            "BUY",
            state["none_pressure_prev8_4"] & state["buy_start"],
        ),
        (
            "SELL_ENDED_THEN_BUY_START_3",
            "BUY",
            state["sell_end_prev3"] & state["buy_start"],
        ),
        (
            "BUY_ENDED_THEN_SELL_START_3",
            "SELL",
            state["buy_end_prev3"] & state["sell_start"],
        ),
        (
            "NONE_ENDED_THEN_SELL_START_1",
            "SELL",
            state["none_end_prev1"] & state["sell_start"],
        ),
        (
            "NONE_ENDED_THEN_SELL_START_2",
            "SELL",
            state["none_end_prev2"] & state["sell_start"],
        ),
        (
            "NONE_ENDED_THEN_SELL_START_3",
            "SELL",
            state["none_end_prev3"] & state["sell_start"],
        ),
    ]
    if allow_start_only:
        stories.extend(
            [
                ("BUY_START_ONLY", "BUY", state["buy_start"]),
                ("SELL_START_ONLY", "SELL", state["sell_start"]),
            ]
        )
    return stories


def _side_arrays(targets: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "BUY":
        return (
            pd.to_numeric(targets["BUY_R"], errors="coerce").fillna(0.0).to_numpy(float),
            targets["BUY_WIN"].astype(bool).to_numpy(bool),
            targets["BUY_LOSS"].astype(bool).to_numpy(bool),
        )
    return (
        pd.to_numeric(targets["SELL_R"], errors="coerce").fillna(0.0).to_numpy(float),
        targets["SELL_WIN"].astype(bool).to_numpy(bool),
        targets["SELL_LOSS"].astype(bool).to_numpy(bool),
    )


def _story_stats(
    fold_name: str,
    profile: StoryProfile,
    train_targets: pd.DataFrame,
    train_state: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    usable: dict[str, dict[str, Any]] = {}
    target_labels = train_targets["winner"].astype(str).str.upper()

    for story_name, side, mask in _story_masks(train_state, profile.allow_start_only):
        r, win, loss = _side_arrays(train_targets, side)
        n = int(mask.sum())
        if n > 0:
            mean_r = float(np.nanmean(r[mask]))
            win_rate = float(win[mask].mean())
            loss_rate = float(loss[mask].mean())
            none_rate = float(target_labels.loc[mask].eq("NONE").mean())
            opposite = "SELL" if side == "BUY" else "BUY"
            opposite_rate = float(target_labels.loc[mask].eq(opposite).mean())
        else:
            mean_r = win_rate = loss_rate = none_rate = opposite_rate = 0.0

        is_usable = (
            n >= profile.min_train_n
            and win_rate >= profile.min_train_success
            and mean_r >= profile.min_train_mean_r
        )
        row = {
            "fold": fold_name,
            "profile": profile.name,
            "story": story_name,
            "side": side,
            "train_n": n,
            "train_mean_r": round(mean_r, 5),
            "train_win_pct": round(win_rate * 100.0, 2),
            "train_loss_pct": round(loss_rate * 100.0, 2),
            "train_none_pct": round(none_rate * 100.0, 2),
            "train_opposite_pct": round(opposite_rate * 100.0, 2),
            "usable": int(is_usable),
        }
        rows.append(row)
        if is_usable:
            usable[story_name] = row
    return pd.DataFrame(rows), usable


def _make_story_trades(
    fold_name: str,
    test: pd.DataFrame,
    test_targets: pd.DataFrame,
    test_state: dict[str, np.ndarray],
    profile: StoryProfile,
    usable_stories: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    story_defs = _story_masks(test_state, profile.allow_start_only)
    story_by_name = {name: (side, mask) for name, side, mask in story_defs}
    ts = pd.to_datetime(test["signal_bar_time"], utc=True, errors="coerce", format="mixed")

    buy_r, buy_win, buy_loss = _side_arrays(test_targets, "BUY")
    sell_r, sell_win, sell_loss = _side_arrays(test_targets, "SELL")
    target_labels = test_targets["winner"].astype(str).str.upper().to_numpy(str)

    rows: list[dict[str, Any]] = []
    day_counts: dict[str, int] = {}
    last_trade_i = -10_000

    for i in range(len(test)):
        day = str(ts.iloc[i].date()) if pd.notna(ts.iloc[i]) else "UNKNOWN"
        if i - last_trade_i < profile.cooldown_bars:
            continue
        if day_counts.get(day, 0) >= profile.max_trades_per_day:
            continue

        candidates: list[dict[str, Any]] = []
        for story_name, story_meta in usable_stories.items():
            side, mask = story_by_name.get(story_name, ("NONE", np.zeros(len(test), dtype=bool)))
            if not mask[i]:
                continue
            candidates.append(
                {
                    "story": story_name,
                    "side": side,
                    "train_mean_r": float(story_meta["train_mean_r"]),
                    "train_win_pct": float(story_meta["train_win_pct"]),
                    "train_n": int(story_meta["train_n"]),
                }
            )
        if not candidates:
            continue

        candidates = sorted(candidates, key=lambda c: (c["train_mean_r"], c["train_win_pct"], c["train_n"]), reverse=True)
        best = candidates[0]
        other_side_best = next((c for c in candidates if c["side"] != best["side"]), None)
        if other_side_best and (best["train_mean_r"] - other_side_best["train_mean_r"] < profile.min_side_gap_r):
            continue

        side = best["side"]
        if side == "BUY":
            r = float(buy_r[i])
            win = bool(buy_win[i])
            loss = bool(buy_loss[i])
        else:
            r = float(sell_r[i])
            win = bool(sell_win[i])
            loss = bool(sell_loss[i])

        day_counts[day] = day_counts.get(day, 0) + 1
        last_trade_i = i
        rows.append(
            {
                "profile": profile.name,
                "fold": fold_name,
                "timestamp": test["signal_bar_time"].iloc[i],
                "side": side,
                "story": best["story"],
                "r": r,
                "win": int(win),
                "loss": int(loss),
                "cheat_winner": target_labels[i],
                "cheat_right": int(target_labels[i] == side),
                "train_story_n": best["train_n"],
                "train_story_mean_r": round(best["train_mean_r"], 5),
                "train_story_win_pct": round(best["train_win_pct"], 2),
                "active_story_count": len(candidates),
                "reason": (
                    f"{best['story']} passed train: n={best['train_n']}, "
                    f"win={best['train_win_pct']:.2f}%, mean_r={best['train_mean_r']:+.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _folds(df: pd.DataFrame, train_months: int, test_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    return v1._folds(df, train_months=train_months, test_months=test_months)


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for profile, g in trades.groupby("profile"):
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce", format="mixed")
        daily = g.assign(day=ts.dt.date).groupby("day")["r"].sum()
        rows.append(
            {
                "profile": profile,
                "trades": int(len(g)),
                "total_r": round(float(g["r"].sum()), 2),
                "mean_r": round(float(g["r"].mean()), 4),
                "cheat_right_pct": round(float(g["cheat_right"].mean() * 100.0), 2),
                "win_rate_pct": round(float(g["win"].mean() * 100.0), 2),
                "loss_rate_pct": round(float(g["loss"].mean() * 100.0), 2),
                "max_dd_r": round(v1._max_drawdown(g["r"]), 2),
                "trade_days": int(daily.shape[0]),
                "red_days": int((daily < 0).sum()),
                "worst_day_r": round(float(daily.min()), 2) if len(daily) else 0.0,
                "median_day_r": round(float(daily.median()), 2) if len(daily) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["total_r", "mean_r"], ascending=[False, False])


def _story_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (profile, story, side), g in trades.groupby(["profile", "story", "side"]):
        rows.append(
            {
                "profile": profile,
                "story": story,
                "side": side,
                "trades": int(len(g)),
                "total_r": round(float(g["r"].sum()), 2),
                "mean_r": round(float(g["r"].mean()), 4),
                "cheat_right_pct": round(float(g["cheat_right"].mean() * 100.0), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["profile", "total_r"], ascending=[True, False])


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].astype(str)
    widths = {col: max(len(str(col)), *(len(v) for v in view[col])) for col in view.columns}
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(v1.TABLE).reset_index(drop=True)
    df = df[df["entry_valid"].astype(bool)].reset_index(drop=True)
    ts = pd.to_datetime(df["signal_bar_time"], utc=True, errors="coerce", format="mixed")
    df = df[ts.notna()].reset_index(drop=True)
    ts = pd.to_datetime(df["signal_bar_time"], utc=True, errors="coerce", format="mixed")

    folds = _folds(df, train_months=12, test_months=2)
    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    story_rows: list[pd.DataFrame] = []

    for fold_idx, (train_start, test_start, test_end) in enumerate(folds, start=1):
        train_mask = (ts >= train_start) & (ts < test_start)
        test_mask = (ts >= test_start) & (ts < test_end)
        train = df.loc[train_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if len(train) < 20_000 or len(test) < 1_000:
            continue

        fold_name = f"f{fold_idx:02d}_{test_start.date()}_{(test_end - pd.Timedelta(days=1)).date()}"
        train_targets = v1._build_target_frame(train)
        test_targets = v1._build_target_frame(test)
        min_n = max(90, int(len(train) * 0.0025))
        cards = v1._mine_cards(train, train_targets, min_n=min_n, top_atoms=35, max_combo_size=3)

        for profile in PROFILES:
            train_state = _state_from_cards(cards, train, profile.top_k_cards)
            test_state = _state_from_cards(cards, test, profile.top_k_cards)
            story_stats, usable = _story_stats(fold_name, profile, train_targets, train_state)
            story_rows.append(story_stats)
            trades = _make_story_trades(fold_name, test, test_targets, test_state, profile, usable)
            if not trades.empty:
                all_trades.append(trades)
            fold_rows.append(
                {
                    "fold": fold_name,
                    "profile": profile.name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "cards": len(cards),
                    "usable_stories": int(sum(story_stats["usable"])),
                    "trades": int(len(trades)),
                    "total_r": round(float(trades["r"].sum()), 2) if not trades.empty else 0.0,
                    "mean_r": round(float(trades["r"].mean()), 4) if not trades.empty else 0.0,
                    "cheat_right_pct": round(float(trades["cheat_right"].mean() * 100.0), 2) if not trades.empty else 0.0,
                    "max_dd_r": round(v1._max_drawdown(trades["r"]), 2) if not trades.empty else 0.0,
                }
            )

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    stories_out = pd.concat(story_rows, ignore_index=True) if story_rows else pd.DataFrame()
    trades_out.to_csv(TRADES_CSV, index=False)
    folds_out.to_csv(FOLDS_CSV, index=False)
    stories_out.to_csv(STORIES_CSV, index=False)

    summary = _summarize(trades_out)
    by_story = _story_summary(trades_out)
    baseline_targets = v1._build_target_frame(df)
    baseline_buy = float(baseline_targets["BUY_WIN"].mean() * 100.0)
    baseline_sell = float(baseline_targets["SELL_WIN"].mean() * 100.0)
    baseline_buy_r = float(baseline_targets["BUY_R"].mean())
    baseline_sell_r = float(baseline_targets["SELL_R"].mean())

    best_profile = summary.iloc[0]["profile"] if not summary.empty else ""
    examples = trades_out[trades_out["profile"].eq(best_profile)].head(15) if best_profile else pd.DataFrame()

    lines = [
        "# Atlas Sequence-Story Walk-Forward Report",
        "",
        "Offline only. Live bot was not touched.",
        "",
        "## What Changed From Atlas v1",
        "",
        "Atlas v1 added sequence as a small score boost. That was not the idea.",
        "",
        "This version first builds a predicted market story from train-only clue cards, then trades only if a whole story transition survived in the training window.",
        "",
        "Example story: `NONE_ENDED_THEN_BUY_START_2` means a predicted NONE-ending clue appeared within the prior 2 bars and the current bar has a predicted BUY-start clue.",
        "",
        "## Baseline",
        "",
        f"- BUY +2R before -1R baseline: `{baseline_buy:.2f}%`, mean `{baseline_buy_r:+.4f}R`.",
        f"- SELL +2R before -1R baseline: `{baseline_sell:.2f}%`, mean `{baseline_sell_r:+.4f}R`.",
        "",
        "## Profile Results",
        "",
        _md_table(summary, max_rows=20),
        "",
        "## Fold Results",
        "",
        _md_table(folds_out.sort_values(["profile", "fold"]), max_rows=80),
        "",
        "## Story Breakdown",
        "",
        _md_table(by_story, max_rows=80),
        "",
        "## Example Trades From Best Profile",
        "",
        _md_table(
            examples[
                [
                    "timestamp",
                    "side",
                    "story",
                    "r",
                    "cheat_winner",
                    "train_story_n",
                    "train_story_win_pct",
                    "reason",
                ]
            ]
            if not examples.empty
            else examples,
            max_rows=15,
        ),
        "",
        "## Plain Read",
        "",
    ]
    if summary.empty:
        lines.append("No story profile found future trades.")
    else:
        best = summary.iloc[0]
        lines.append(
            f"Best profile was `{best['profile']}` with `{best['total_r']:+.2f}R` over `{int(best['trades'])}` trades and `{best['cheat_right_pct']}%` agreement with the offline outcome label."
        )
        lines.append("")
        lines.append(
            "This test answers whether the sequence-story idea survives walk-forward, not whether it should be wired live."
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "trades": int(len(trades_out)),
                "fold_rows": int(len(folds_out)),
                "story_rows": int(len(stories_out)),
                "report": str(REPORT_MD),
                "best_profile": str(best_profile),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
