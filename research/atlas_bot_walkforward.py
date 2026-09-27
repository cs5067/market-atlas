#!/usr/bin/env python3
"""Transparent Market Atlas walk-forward.

This is offline-only. It does not import or modify the live bot.

Atlas is intentionally not a neural net. It builds evidence cards from past
rows only, applies those cards to future rows, and explains every trade in
plain evidence terms: direct side edge, start clue, end/danger clue, NONE
danger, and sequence context.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLE = ROOT / "data/market_atlas/tick_winner_tables/TICK_RACE_LABELED_ALL_M5.parquet"
OUT_DIR = ROOT / "data/market_atlas/atlas_bot"
TRADES_CSV = OUT_DIR / "ATLAS_BOT_WALKFORWARD_TRADES.csv"
FOLDS_CSV = OUT_DIR / "ATLAS_BOT_WALKFORWARD_FOLDS.csv"
CARDS_CSV = OUT_DIR / "ATLAS_BOT_WALKFORWARD_CARDS.csv"
REPORT_MD = OUT_DIR / "ATLAS_BOT_WALKFORWARD_REPORT.md"


EXCLUDE_PREFIXES = ("next_m5_",)
EXCLUDE_EXACT = {
    "tick_race_winner",
    "tick_race_reason",
    "entry_invalid_reason",
    "source_tick_month",
    "signal_bar_time",
    "entry_target_time",
    "entry_tick_time",
    "atlas_datetime",
    "prev_m5_datetime",
    "signal_m5_datetime",
}
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
    "tick_race_tie",
    "buy_minus_sell",
    "synthetic_risk",
    "atlas_horizon",
)


TARGETS = [
    "BUY_DIRECT",
    "SELL_DIRECT",
    "BUY_START_REAL2",
    "SELL_START_REAL2",
    "NONE_START_REAL2",
    "BUY_END_REAL2",
    "SELL_END_REAL2",
    "NONE_END_REAL2",
    "NONE_DIRECT",
]


@dataclass(frozen=True)
class AtomSpec:
    name: str
    feature: str
    kind: str
    value: str | None = None
    op: str | None = None
    threshold: float | None = None

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)
        s = df[self.feature]
        if self.kind == "category":
            vals = s.map(_clean_value)
            return vals.eq(str(self.value)).to_numpy(bool)
        numeric = pd.to_numeric(s, errors="coerce")
        if self.op == "<=":
            return numeric.le(float(self.threshold)).fillna(False).to_numpy(bool)
        if self.op == ">=":
            return numeric.ge(float(self.threshold)).fillna(False).to_numpy(bool)
        return np.zeros(len(df), dtype=bool)


@dataclass(frozen=True)
class EvidenceCard:
    target: str
    parts: tuple[AtomSpec, ...]
    n: int
    mean_r: float
    win_pct: float
    loss_pct: float
    target_pct: float
    baseline_pct: float
    lift: float
    card_score: float

    @property
    def parts_text(self) -> str:
        return " + ".join(part.name for part in self.parts)

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        out = np.ones(len(df), dtype=bool)
        for part in self.parts:
            out &= part.mask(df)
        return out


@dataclass(frozen=True)
class Profile:
    name: str
    min_score: float
    min_direct_r: float
    cooldown_bars: int
    max_trades_per_day: int


PROFILES = [
    Profile("atlas_balanced", min_score=0.30, min_direct_r=0.04, cooldown_bars=3, max_trades_per_day=8),
    Profile("atlas_strict", min_score=0.45, min_direct_r=0.08, cooldown_bars=4, max_trades_per_day=5),
    Profile("atlas_aggressive", min_score=0.18, min_direct_r=0.00, cooldown_bars=2, max_trades_per_day=12),
]


def _clean_value(value: Any) -> str:
    text = str(value if value is not None else "").strip().upper()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return "UNKNOWN"
    return text.replace("|", "/")


def _should_exclude_column(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return True
    if col.startswith(EXCLUDE_PREFIXES):
        return True
    low = col.lower()
    return any(part in low for part in EXCLUDE_SUBSTRINGS)


def _month_floor(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")


def _safe_pct(mask: pd.Series | np.ndarray) -> float:
    if len(mask) == 0:
        return 0.0
    return float(np.asarray(mask, dtype=bool).mean() * 100.0)


def _runs(winners: pd.Series) -> pd.DataFrame:
    labels = winners.fillna("NONE").astype(str).str.upper().tolist()
    n = len(labels)
    run_id: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    run_labels: list[str] = []
    rid = -1
    prev = None
    for i, lab in enumerate(labels):
        if i == 0 or lab != prev:
            if i > 0:
                ends.append(i - 1)
            rid += 1
            starts.append(i)
            run_labels.append(lab)
        run_id.append(rid)
        prev = lab
    ends.append(n - 1)
    lengths = [ends[i] - starts[i] + 1 for i in range(len(run_labels))]
    return pd.DataFrame(
        {
            "run_label": [run_labels[i] for i in run_id],
            "run_len": [lengths[i] for i in run_id],
            "run_pos": [i - starts[run_id[i]] + 1 for i in range(n)],
        }
    )


def _build_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    winner = df["tick_race_winner"].fillna("NONE").astype(str).str.upper()
    out["winner"] = winner
    runs = _runs(winner).set_index(df.index)
    out = pd.concat([out, runs], axis=1)

    for side in ["BUY", "SELL", "NONE"]:
        out[side] = out["winner"].eq(side)
        out[f"{side}_START_REAL2"] = out["run_label"].eq(side) & out["run_pos"].eq(1) & out["run_len"].ge(2)
        out[f"{side}_END_REAL2"] = out["run_label"].eq(side) & out["run_pos"].eq(out["run_len"]) & out["run_len"].ge(2)

    buy_tp = pd.to_numeric(df["buy_time_to_plus_2r_sec"], errors="coerce")
    buy_sl = pd.to_numeric(df["buy_time_to_minus_1r_sec"], errors="coerce")
    sell_tp = pd.to_numeric(df["sell_time_to_plus_2r_sec"], errors="coerce")
    sell_sl = pd.to_numeric(df["sell_time_to_minus_1r_sec"], errors="coerce")

    buy_win = buy_tp.notna() & (buy_sl.isna() | buy_tp.lt(buy_sl))
    buy_loss = buy_sl.notna() & (buy_tp.isna() | buy_sl.lt(buy_tp))
    sell_win = sell_tp.notna() & (sell_sl.isna() | sell_tp.lt(sell_sl))
    sell_loss = sell_sl.notna() & (sell_tp.isna() | sell_sl.lt(sell_tp))

    out["BUY_WIN"] = buy_win
    out["BUY_LOSS"] = buy_loss
    out["SELL_WIN"] = sell_win
    out["SELL_LOSS"] = sell_loss
    out["BUY_R"] = np.where(buy_win, 2.0, np.where(buy_loss, -1.0, 0.0))
    out["SELL_R"] = np.where(sell_win, 2.0, np.where(sell_loss, -1.0, 0.0))
    out["NONE_DIRECT"] = out["NONE"]
    return out


def _build_atoms(train: pd.DataFrame, min_n: int) -> list[AtomSpec]:
    atoms: list[AtomSpec] = []
    for col in train.columns:
        if _should_exclude_column(col):
            continue
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            valid = numeric.dropna()
            if valid.nunique() >= 8:
                qs = valid.quantile([0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
                specs = [
                    ("LOWEST5", "<=", float(qs.loc[0.05])),
                    ("LOWEST10", "<=", float(qs.loc[0.10])),
                    ("LOWEST25", "<=", float(qs.loc[0.25])),
                    ("HIGHEST25", ">=", float(qs.loc[0.75])),
                    ("HIGHEST10", ">=", float(qs.loc[0.90])),
                    ("HIGHEST5", ">=", float(qs.loc[0.95])),
                ]
                for bucket, op, threshold in specs:
                    mask = numeric.le(threshold) if op == "<=" else numeric.ge(threshold)
                    if int(mask.fillna(False).sum()) >= min_n:
                        atoms.append(AtomSpec(f"{col}={bucket}", col, "numeric", op=op, threshold=threshold))
            if s.nunique(dropna=True) <= 48:
                atoms.extend(_categorical_atoms(col, s, min_n))
        else:
            atoms.extend(_categorical_atoms(col, s, min_n))

    seen: set[str] = set()
    unique: list[AtomSpec] = []
    for atom in atoms:
        if atom.name in seen:
            continue
        seen.add(atom.name)
        unique.append(atom)
    return unique


def _categorical_atoms(col: str, s: pd.Series, min_n: int) -> list[AtomSpec]:
    vals = s.map(_clean_value)
    vc = vals.value_counts(dropna=False)
    if len(vc) > 60:
        return []
    atoms: list[AtomSpec] = []
    for value, n in vc.items():
        if value == "UNKNOWN" or int(n) < min_n:
            continue
        atoms.append(AtomSpec(f"{col}={value}", col, "category", value=value))
    return atoms


def _compatible(parts: tuple[AtomSpec, ...]) -> bool:
    features: set[str] = set()
    for part in parts:
        if part.feature in features:
            return False
        features.add(part.feature)
    return True


def _card_from_mask(
    target: str,
    parts: tuple[AtomSpec, ...],
    mask: np.ndarray,
    targets: pd.DataFrame,
    baselines: dict[str, float],
) -> EvidenceCard | None:
    n = int(mask.sum())
    if n <= 0:
        return None
    if target == "BUY_DIRECT":
        r = pd.to_numeric(targets.loc[mask, "BUY_R"], errors="coerce")
        wins = targets.loc[mask, "BUY_WIN"].astype(bool)
        losses = targets.loc[mask, "BUY_LOSS"].astype(bool)
        target_rate = float(wins.mean())
        baseline = baselines[target]
        mean_r = float(r.mean())
        loss_pct = float(losses.mean() * 100.0)
        win_pct = float(target_rate * 100.0)
        lift = target_rate / baseline if baseline else 0.0
        card_score = mean_r + (target_rate - baseline)
    elif target == "SELL_DIRECT":
        r = pd.to_numeric(targets.loc[mask, "SELL_R"], errors="coerce")
        wins = targets.loc[mask, "SELL_WIN"].astype(bool)
        losses = targets.loc[mask, "SELL_LOSS"].astype(bool)
        target_rate = float(wins.mean())
        baseline = baselines[target]
        mean_r = float(r.mean())
        loss_pct = float(losses.mean() * 100.0)
        win_pct = float(target_rate * 100.0)
        lift = target_rate / baseline if baseline else 0.0
        card_score = mean_r + (target_rate - baseline)
    else:
        target_rate = float(targets.loc[mask, target].astype(bool).mean())
        baseline = baselines[target]
        mean_r = 0.0
        win_pct = float(target_rate * 100.0)
        loss_pct = 0.0
        lift = target_rate / baseline if baseline else 0.0
        card_score = (target_rate - baseline) * max(1.0, min(lift, 4.0))

    return EvidenceCard(
        target=target,
        parts=parts,
        n=n,
        mean_r=mean_r,
        win_pct=win_pct,
        loss_pct=loss_pct,
        target_pct=float(target_rate * 100.0),
        baseline_pct=float(baseline * 100.0),
        lift=float(lift),
        card_score=float(card_score),
    )


def _interesting(card: EvidenceCard, min_n: int) -> bool:
    if card.n < min_n:
        return False
    if card.target in {"BUY_DIRECT", "SELL_DIRECT"}:
        return card.mean_r >= 0.05 and card.win_pct >= card.baseline_pct + 3.0
    if card.target == "NONE_DIRECT":
        return card.target_pct >= card.baseline_pct + 5.0 and card.lift >= 1.12
    if "START" in card.target:
        return card.target_pct >= card.baseline_pct + 4.0 and card.lift >= 1.65
    if "END" in card.target:
        return card.target_pct >= card.baseline_pct + 3.0 and card.lift >= 1.45
    return False


def _mine_cards(train: pd.DataFrame, targets: pd.DataFrame, min_n: int, top_atoms: int, max_combo_size: int) -> list[EvidenceCard]:
    atoms = _build_atoms(train, min_n)
    atom_masks = {atom.name: atom.mask(train) for atom in atoms}
    baselines = {
        "BUY_DIRECT": float(targets["BUY_WIN"].mean()),
        "SELL_DIRECT": float(targets["SELL_WIN"].mean()),
        "NONE_DIRECT": float(targets["NONE_DIRECT"].mean()),
        "BUY_START_REAL2": float(targets["BUY_START_REAL2"].mean()),
        "SELL_START_REAL2": float(targets["SELL_START_REAL2"].mean()),
        "NONE_START_REAL2": float(targets["NONE_START_REAL2"].mean()),
        "BUY_END_REAL2": float(targets["BUY_END_REAL2"].mean()),
        "SELL_END_REAL2": float(targets["SELL_END_REAL2"].mean()),
        "NONE_END_REAL2": float(targets["NONE_END_REAL2"].mean()),
    }

    all_cards: list[EvidenceCard] = []
    for target in TARGETS:
        single_scored: list[tuple[float, AtomSpec]] = []
        for atom in atoms:
            mask = atom_masks[atom.name]
            card = _card_from_mask(target, (atom,), mask, targets, baselines)
            if card is None:
                continue
            if _interesting(card, min_n):
                all_cards.append(card)
            # Looser gate for building blocks.
            if target in {"BUY_DIRECT", "SELL_DIRECT"}:
                if card.n >= min_n and (card.mean_r >= -0.02 or card.win_pct >= card.baseline_pct + 1.5):
                    single_scored.append((card.card_score, atom))
            else:
                if card.n >= min_n and (card.lift >= 1.10 or card.target_pct >= card.baseline_pct + 1.5):
                    single_scored.append((card.card_score, atom))

        candidate_atoms = [a for _, a in sorted(single_scored, key=lambda x: x[0], reverse=True)[:top_atoms]]
        seen: set[tuple[str, ...]] = set()
        for size in range(2, max_combo_size + 1):
            for parts in combinations(candidate_atoms, size):
                if not _compatible(parts):
                    continue
                key = tuple(sorted(part.name for part in parts))
                if key in seen:
                    continue
                seen.add(key)
                mask = np.ones(len(train), dtype=bool)
                for part in parts:
                    mask &= atom_masks[part.name]
                if int(mask.sum()) < min_n:
                    continue
                card = _card_from_mask(target, tuple(sorted(parts, key=lambda p: p.name)), mask, targets, baselines)
                if card is not None and _interesting(card, min_n):
                    all_cards.append(card)

    # Keep a compact, explainable card book per fold.
    dedup: dict[tuple[str, str], EvidenceCard] = {}
    for card in all_cards:
        key = (card.target, card.parts_text)
        if key not in dedup or card.card_score > dedup[key].card_score:
            dedup[key] = card
    compact: list[EvidenceCard] = []
    for target in TARGETS:
        sub = [c for c in dedup.values() if c.target == target]
        sub = sorted(sub, key=lambda c: (c.card_score, c.lift, c.n), reverse=True)[:35]
        compact.extend(sub)
    return compact


def _aggregate_cards(cards: list[EvidenceCard], test: pd.DataFrame) -> dict[str, dict[str, Any]]:
    n = len(test)
    board: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        board[target] = {
            "active_count": np.zeros(n, dtype=int),
            "score_sum": np.zeros(n, dtype=float),
            "score_max": np.zeros(n, dtype=float),
            "best_card": np.full(n, "", dtype=object),
            "best_card_n": np.zeros(n, dtype=int),
            "best_card_pct": np.zeros(n, dtype=float),
        }
    for card in cards:
        mask = card.mask(test)
        if not mask.any():
            continue
        target_board = board[card.target]
        strength = card.mean_r if card.target in {"BUY_DIRECT", "SELL_DIRECT"} else (card.target_pct - card.baseline_pct) / 100.0
        active_idx = np.where(mask)[0]
        target_board["active_count"][active_idx] += 1
        target_board["score_sum"][active_idx] += strength
        better = mask & (strength > target_board["score_max"])
        target_board["score_max"][better] = strength
        target_board["best_card"][better] = card.parts_text
        target_board["best_card_n"][better] = card.n
        target_board["best_card_pct"][better] = card.target_pct
    return board


def _prev_any(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr.astype(int))
    return s.shift(1).rolling(window, min_periods=1).max().fillna(0).to_numpy(bool)


def _prev_count(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr.astype(int))
    return s.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy(int)


def _side_result(targets: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "BUY":
        r = pd.to_numeric(targets["BUY_R"], errors="coerce").fillna(0.0).to_numpy(float)
        win = targets["BUY_WIN"].astype(bool).to_numpy(bool)
        loss = targets["BUY_LOSS"].astype(bool).to_numpy(bool)
    else:
        r = pd.to_numeric(targets["SELL_R"], errors="coerce").fillna(0.0).to_numpy(float)
        win = targets["SELL_WIN"].astype(bool).to_numpy(bool)
        loss = targets["SELL_LOSS"].astype(bool).to_numpy(bool)
    return r, win, loss


def _make_decisions(
    fold_name: str,
    test: pd.DataFrame,
    targets: pd.DataFrame,
    cards: list[EvidenceCard],
    profile: Profile,
) -> pd.DataFrame:
    board = _aggregate_cards(cards, test)
    n = len(test)

    buy_direct = board["BUY_DIRECT"]["score_max"]
    sell_direct = board["SELL_DIRECT"]["score_max"]
    buy_start = board["BUY_START_REAL2"]["score_max"]
    sell_start = board["SELL_START_REAL2"]["score_max"]
    none_start = board["NONE_START_REAL2"]["score_max"]
    none_direct = board["NONE_DIRECT"]["score_max"]
    buy_end = board["BUY_END_REAL2"]["score_max"]
    sell_end = board["SELL_END_REAL2"]["score_max"]
    none_end = board["NONE_END_REAL2"]["score_max"]

    none_pressure_now = (none_start > 0.03) | (none_direct > 0.04)
    none_pressure_prev3 = _prev_count(none_pressure_now, 3)
    none_pressure_prev5 = _prev_count(none_pressure_now, 5)
    none_end_prev3 = _prev_any(none_end > 0.03, 3)
    buy_end_prev3 = _prev_any(buy_end > 0.03, 3)
    sell_end_prev3 = _prev_any(sell_end > 0.03, 3)

    buy_sequence = (
        np.where(none_end_prev3 & (buy_start > 0.03), 0.22, 0.0)
        + np.where((none_pressure_prev5 >= 3) & (buy_start > 0.03), 0.12, 0.0)
        + np.where(sell_end_prev3 & (buy_start > 0.03), 0.08, 0.0)
    )
    sell_sequence = (
        np.where(buy_end_prev3 & (sell_start > 0.03), 0.25, 0.0)
        + np.where(none_end_prev3 & (sell_start > 0.03), 0.10, 0.0)
    )

    buy_danger = (
        np.where(none_pressure_now, 0.18, 0.0)
        + np.where(buy_end > 0.03, 0.30, 0.0)
        + np.where(sell_start > 0.04, 0.35, 0.0)
    )
    sell_danger = (
        np.where(none_pressure_now, 0.18, 0.0)
        + np.where(sell_end > 0.03, 0.30, 0.0)
        + np.where(buy_start > 0.04, 0.35, 0.0)
    )

    buy_score = buy_direct + 0.55 * buy_start + buy_sequence - buy_danger
    sell_score = sell_direct + 0.55 * sell_start + sell_sequence - sell_danger

    buy_r, buy_win, buy_loss = _side_result(targets, "BUY")
    sell_r, sell_win, sell_loss = _side_result(targets, "SELL")

    rows: list[dict[str, Any]] = []
    day_counts: dict[str, int] = {}
    last_trade_i = -10_000
    ts = pd.to_datetime(test["signal_bar_time"], utc=True, errors="coerce", format="mixed")

    for i in range(n):
        day = str(ts.iloc[i].date()) if pd.notna(ts.iloc[i]) else "UNKNOWN"
        if i - last_trade_i < profile.cooldown_bars:
            continue
        if day_counts.get(day, 0) >= profile.max_trades_per_day:
            continue

        side = "NONE"
        score = 0.0
        direct_r = 0.0
        reason_bits: list[str] = []
        if buy_score[i] >= profile.min_score and buy_score[i] > sell_score[i] + 0.06 and buy_direct[i] >= profile.min_direct_r:
            side = "BUY"
            score = float(buy_score[i])
            direct_r = float(buy_direct[i])
            reason_bits = [
                f"buy_direct_r={buy_direct[i]:+.3f}",
                f"buy_start_edge={buy_start[i]:+.3f}",
                f"sequence={buy_sequence[i]:+.3f}",
                f"danger={buy_danger[i]:+.3f}",
            ]
        elif sell_score[i] >= profile.min_score and sell_score[i] > buy_score[i] + 0.06 and sell_direct[i] >= profile.min_direct_r:
            side = "SELL"
            score = float(sell_score[i])
            direct_r = float(sell_direct[i])
            reason_bits = [
                f"sell_direct_r={sell_direct[i]:+.3f}",
                f"sell_start_edge={sell_start[i]:+.3f}",
                f"sequence={sell_sequence[i]:+.3f}",
                f"danger={sell_danger[i]:+.3f}",
            ]
        if side == "NONE":
            continue

        r = float(buy_r[i] if side == "BUY" else sell_r[i])
        win = bool(buy_win[i] if side == "BUY" else sell_win[i])
        loss = bool(buy_loss[i] if side == "BUY" else sell_loss[i])
        day_counts[day] = day_counts.get(day, 0) + 1
        last_trade_i = i
        rows.append(
            {
                "profile": profile.name,
                "fold": fold_name,
                "timestamp": test["signal_bar_time"].iloc[i],
                "side": side,
                "r": r,
                "win": int(win),
                "loss": int(loss),
                "score": round(score, 5),
                "direct_r_card": round(direct_r, 5),
                "buy_score": round(float(buy_score[i]), 5),
                "sell_score": round(float(sell_score[i]), 5),
                "none_pressure_prev5": int(none_pressure_prev5[i]),
                "none_end_prev3": int(bool(none_end_prev3[i])),
                "buy_end_prev3": int(bool(buy_end_prev3[i])),
                "sell_end_prev3": int(bool(sell_end_prev3[i])),
                "best_direct_card": board[f"{side}_DIRECT"]["best_card"][i],
                "best_start_card": board[f"{side}_START_REAL2"]["best_card"][i],
                "best_none_card": board["NONE_DIRECT"]["best_card"][i] or board["NONE_START_REAL2"]["best_card"][i],
                "reason": "; ".join(reason_bits),
            }
        )
    return pd.DataFrame(rows)


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.cumsum()
    dd = curve - curve.cummax()
    return float(dd.min())


def _summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for profile, g in trades.groupby("profile"):
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce", format="mixed")
        days = max(1, ts.dt.date.nunique())
        daily = g.assign(day=ts.dt.date).groupby("day")["r"].sum()
        rows.append(
            {
                "profile": profile,
                "trades": int(len(g)),
                "total_r": round(float(g["r"].sum()), 2),
                "mean_r": round(float(g["r"].mean()), 4),
                "win_rate_pct": round(float(g["win"].mean() * 100.0), 2),
                "loss_rate_pct": round(float(g["loss"].mean() * 100.0), 2),
                "max_dd_r": round(_max_drawdown(g["r"]), 2),
                "trades_per_day": round(float(len(g) / days), 2),
                "red_days": int((daily < 0).sum()),
                "worst_day_r": round(float(daily.min()), 2) if len(daily) else 0.0,
                "median_day_r": round(float(daily.median()), 2) if len(daily) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["total_r", "mean_r"], ascending=[False, False])


def _folds(df: pd.DataFrame, train_months: int, test_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    months = sorted(_month_floor(pd.to_datetime(df["signal_bar_time"], utc=True, errors="coerce", format="mixed")).dropna().unique())
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    i = train_months
    while i + test_months <= len(months):
        train_start = months[i - train_months]
        test_start = months[i]
        test_end = months[i + test_months - 1] + pd.DateOffset(months=1)
        folds.append((train_start, test_start, test_end))
        i += test_months
    return folds


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].astype(str)
    widths = {
        col: max(len(str(col)), *(len(str(v)) for v in view[col].tolist()))
        for col in view.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(TABLE).reset_index(drop=True)
    df = df[df["entry_valid"].astype(bool)].reset_index(drop=True)
    ts = pd.to_datetime(df["signal_bar_time"], utc=True, errors="coerce", format="mixed")
    df = df[ts.notna()].reset_index(drop=True)
    ts = pd.to_datetime(df["signal_bar_time"], utc=True, errors="coerce", format="mixed")

    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    card_rows: list[dict[str, Any]] = []
    folds = _folds(df, train_months=12, test_months=2)

    for fold_idx, (train_start, test_start, test_end) in enumerate(folds, start=1):
        train_mask = (ts >= train_start) & (ts < test_start)
        test_mask = (ts >= test_start) & (ts < test_end)
        train = df.loc[train_mask].reset_index(drop=True)
        test = df.loc[test_mask].reset_index(drop=True)
        if len(train) < 20_000 or len(test) < 1_000:
            continue

        train_targets = _build_target_frame(train)
        test_targets = _build_target_frame(test)
        min_n = max(180, int(len(train) * 0.005))
        cards = _mine_cards(train, train_targets, min_n=min_n, top_atoms=30, max_combo_size=3)
        fold_name = f"f{fold_idx:02d}_{test_start.date()}_{(test_end - pd.Timedelta(days=1)).date()}"

        for card in cards:
            card_rows.append(
                {
                    "fold": fold_name,
                    "target": card.target,
                    "n": card.n,
                    "mean_r": round(card.mean_r, 5),
                    "win_pct": round(card.win_pct, 2),
                    "loss_pct": round(card.loss_pct, 2),
                    "target_pct": round(card.target_pct, 2),
                    "baseline_pct": round(card.baseline_pct, 2),
                    "lift": round(card.lift, 3),
                    "score": round(card.card_score, 5),
                    "parts": card.parts_text,
                }
            )

        for profile in PROFILES:
            trades = _make_decisions(fold_name, test, test_targets, cards, profile)
            if not trades.empty:
                all_trades.append(trades)
                fold_rows.append(
                    {
                        "fold": fold_name,
                        "profile": profile.name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "cards": len(cards),
                        "trades": len(trades),
                        "total_r": round(float(trades["r"].sum()), 2),
                        "mean_r": round(float(trades["r"].mean()), 4),
                        "win_rate_pct": round(float(trades["win"].mean() * 100.0), 2),
                        "max_dd_r": round(_max_drawdown(trades["r"]), 2),
                    }
                )
            else:
                fold_rows.append(
                    {
                        "fold": fold_name,
                        "profile": profile.name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "cards": len(cards),
                        "trades": 0,
                        "total_r": 0.0,
                        "mean_r": 0.0,
                        "win_rate_pct": 0.0,
                        "max_dd_r": 0.0,
                    }
                )

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    folds_out = pd.DataFrame(fold_rows)
    cards_out = pd.DataFrame(card_rows)
    trades_out.to_csv(TRADES_CSV, index=False)
    folds_out.to_csv(FOLDS_CSV, index=False)
    cards_out.to_csv(CARDS_CSV, index=False)

    summary = _summarize_trades(trades_out)
    baseline_targets = _build_target_frame(df)
    baseline_buy_r = float(baseline_targets["BUY_R"].mean())
    baseline_sell_r = float(baseline_targets["SELL_R"].mean())
    best_profile = summary.iloc[0]["profile"] if not summary.empty else "NONE"
    sample = trades_out[trades_out["profile"].eq(best_profile)].head(12) if not trades_out.empty else pd.DataFrame()

    lines = [
        "# Atlas Bot Walk-Forward Report",
        "",
        "Offline only. Live bot was not touched.",
        "",
        "## Source Contract",
        "",
        "Atlas was built from `ATLAS_BOT_SOURCE_OF_TRUTH.md`: tick-race truth table, phase/start evidence, NONE danger, sequence context, and wrong-call fast-failure behavior.",
        "",
        "No neural net, no hidden model, no future answer-sheet columns for decisions.",
        "",
        "## Test Shape",
        "",
        "- Train window: rolling prior 12 months.",
        "- Test window: next 2 months.",
        "- At each fold, evidence cards are mined from train only.",
        "- The exact mined cards are applied to future test rows.",
        "- Trade scoring: +2R if selected side reaches +2R before -1R, -1R if it stops first, 0R if unresolved inside the probe horizon.",
        "- Profiles differ only by strictness, cooldown, and daily trade cap.",
        "",
        "## Baseline If You Blindly Trade Every Bar",
        "",
        f"- BUY every bar mean: `{baseline_buy_r:+.4f}R` per bar.",
        f"- SELL every bar mean: `{baseline_sell_r:+.4f}R` per bar.",
        "",
        "## Profile Results",
        "",
        _md_table(summary, max_rows=10),
        "",
        "## Fold Results",
        "",
        _md_table(folds_out.sort_values(["profile", "fold"]), max_rows=60),
        "",
        "## Example Atlas Trades",
        "",
        _md_table(
            sample[
                [
                    "timestamp",
                    "side",
                    "r",
                    "score",
                    "reason",
                    "best_direct_card",
                    "best_start_card",
                    "best_none_card",
                ]
            ]
            if not sample.empty
            else sample,
            max_rows=12,
        ),
        "",
        "## Files",
        "",
        f"- Trades: `{TRADES_CSV}`",
        f"- Folds: `{FOLDS_CSV}`",
        f"- Cards: `{CARDS_CSV}`",
        "",
        "## Plain Read",
        "",
    ]
    if summary.empty:
        lines.append("Atlas did not find any qualifying future trades under these rules.")
    else:
        best = summary.iloc[0]
        lines.extend(
            [
                f"Best profile was `{best['profile']}` with `{best['total_r']:+.2f}R` across `{int(best['trades'])}` trades.",
                "",
                "This is not live approval. It is the first honest exam of whether the Atlas evidence-card idea can survive future rows.",
                "If the best profile is positive across most folds, the idea deserves deeper testing with live-compatible risk models.",
                "If the result is concentrated in one fold or has ugly drawdown, Atlas needs more work before it becomes a trader.",
            ]
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "trades": int(len(trades_out)),
        "fold_rows": int(len(folds_out)),
        "cards": int(len(cards_out)),
        "report": str(REPORT_MD),
        "best_profile": str(best_profile),
    }, indent=2))


if __name__ == "__main__":
    run()
