#!/usr/bin/env python3
"""Build a true tick-order BUY/SELL probe table for every M5 bar.

This is deliberately not a race classifier. It records the raw clocks:

- BUY time to +2R,
- BUY time to -1R,
- SELL time to +2R,
- SELL time to -1R,

using raw MT5 bid/ask ticks after each M5 signal bar. The output also carries
the full market-atlas row plus raw M5 previous/current/next bar context.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_ATLAS = ROOT / "data" / "market_atlas" / "MARKET_ATLAS_FULL.csv.gz"
DEFAULT_M5 = ROOT / "data" / "tick_vr" / "candles" / "full" / "XAUUSDm_M5.csv"
DEFAULT_TICK_DIR = ROOT / "data" / "tick_vr" / "ticks" / "full"
DEFAULT_OUT_DIR = ROOT / "data" / "market_atlas" / "tick_dual_probe"
TICK_COLUMNS = [
    "timestamp_utc",
    "bid",
    "ask",
    "mid",
    "spread",
    "last",
    "volume",
    "volume_real",
    "flags",
    "source",
    "symbol",
]


class SegmentTree:
    def __init__(self, values: np.ndarray, mode: str):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be max or min")
        self.mode = mode
        self.n = int(len(values))
        self.size = 1 << max(1, self.n - 1).bit_length()
        self.neutral = -np.inf if mode == "max" else np.inf
        self.tree = np.full(self.size * 2, self.neutral, dtype=float)
        self.tree[self.size:self.size + self.n] = values.astype(float, copy=False)
        if mode == "max":
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = self.tree[i * 2] if self.tree[i * 2] >= self.tree[i * 2 + 1] else self.tree[i * 2 + 1]
        else:
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = self.tree[i * 2] if self.tree[i * 2] <= self.tree[i * 2 + 1] else self.tree[i * 2 + 1]

    def range_value(self, left: int, right: int) -> float:
        if left < 0:
            left = 0
        if right >= self.n:
            right = self.n - 1
        if left > right or self.n <= 0:
            return float(self.neutral)
        left += self.size
        right += self.size
        res = self.neutral
        if self.mode == "max":
            while left <= right:
                if left & 1:
                    res = max(res, self.tree[left])
                    left += 1
                if not (right & 1):
                    res = max(res, self.tree[right])
                    right -= 1
                left //= 2
                right //= 2
        else:
            while left <= right:
                if left & 1:
                    res = min(res, self.tree[left])
                    left += 1
                if not (right & 1):
                    res = min(res, self.tree[right])
                    right -= 1
                left //= 2
                right //= 2
        return float(res)

    def first_ge(self, left: int, right: int, threshold: float) -> int:
        if self.mode != "max":
            raise ValueError("first_ge requires max tree")
        return self._first_cross(1, 0, self.size - 1, left, right, threshold)

    def first_le(self, left: int, right: int, threshold: float) -> int:
        if self.mode != "min":
            raise ValueError("first_le requires min tree")
        return self._first_cross(1, 0, self.size - 1, left, right, threshold)

    def _first_cross(self, node: int, nl: int, nr: int, ql: int, qr: int, threshold: float) -> int:
        if nr < ql or qr < nl:
            return -1
        value = self.tree[node]
        if self.mode == "max":
            if value < threshold:
                return -1
        else:
            if value > threshold:
                return -1
        if nl == nr:
            return nl if nl < self.n else -1
        mid = (nl + nr) // 2
        left = self._first_cross(node * 2, nl, mid, ql, qr, threshold)
        if left != -1:
            return left
        return self._first_cross(node * 2 + 1, mid + 1, nr, ql, qr, threshold)


def _parse_month(path: Path) -> pd.Timestamp | None:
    m = re.search(r"_(20\d{2})_(\d{2})\.parquet$", path.name)
    if not m:
        return None
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1, tz="UTC")


def _tick_files(tick_dir: Path) -> list[Path]:
    return sorted(Path(tick_dir).glob("XAUUSDm_ticks_20??_??.parquet"))


def _read_ticks(path: Path, *, end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=TICK_COLUMNS)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if end is not None:
        df = df[df["timestamp_utc"] < end]
    df = df.dropna(subset=["timestamp_utc", "bid", "ask"])
    return df


def _prefixed_shift(raw: pd.DataFrame, shift: int, prefix: str) -> pd.DataFrame:
    shifted = raw.shift(shift).copy()
    return shifted.add_prefix(prefix)


def _iso_from_ns(ns: int) -> str:
    if ns <= 0:
        return ""
    return pd.Timestamp(ns, tz="UTC").isoformat()


def _init_float(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def _init_object(n: int) -> np.ndarray:
    return np.full(n, "", dtype=object)


def _month_positions(entry_ns: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    start_ns = int(start.value)
    end_ns = int(end.value)
    return np.flatnonzero((entry_ns >= start_ns) & (entry_ns < end_ns))


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = pd.read_csv(args.atlas)
    atlas["datetime"] = pd.to_datetime(atlas["datetime"], utc=True, errors="coerce")
    atlas = atlas.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if args.start:
        atlas = atlas[atlas["datetime"] >= pd.Timestamp(args.start, tz="UTC")].copy()
    if args.end:
        atlas = atlas[atlas["datetime"] < pd.Timestamp(args.end, tz="UTC")].copy()
    if args.max_rows:
        atlas = atlas.head(int(args.max_rows)).copy()
    atlas = atlas.reset_index(drop=True)

    m5 = pd.read_csv(args.m5)
    m5["datetime"] = pd.to_datetime(m5["datetime"], utc=True, errors="coerce")
    m5 = m5.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    m5_by_time = m5.set_index("datetime", drop=False)
    m5_context = m5_by_time.reindex(atlas["datetime"]).reset_index(drop=True)

    # Signal bar close is the first fair no-lookahead entry moment.
    entry_target = atlas["datetime"] + pd.to_timedelta(5, unit="min")
    entry_target_ns = entry_target.dt.as_unit("ns").astype("int64").to_numpy()
    horizon_ns = int(args.horizon_minutes) * 60 * 1_000_000_000
    max_lag_ns = float(args.max_entry_lag_seconds) * 1_000_000_000
    n = len(atlas)

    context = pd.concat(
        [
            pd.DataFrame({
                "signal_bar_time": atlas["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                "entry_target_time": entry_target.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
            }),
            atlas.add_prefix("atlas_"),
            _prefixed_shift(m5_context, 1, "prev_m5_"),
            m5_context.add_prefix("signal_m5_"),
            _prefixed_shift(m5_context, -1, "next_m5_"),
        ],
        axis=1,
    )

    risk = pd.to_numeric(atlas.get("synthetic_risk_usd"), errors="coerce")
    if risk.isna().all():
        atr = pd.to_numeric(atlas.get("atr"), errors="coerce")
        spread = pd.to_numeric(atlas.get("spread_median"), errors="coerce").fillna(0.0)
        risk = pd.concat([atr, spread * 3.0, pd.Series(args.min_risk_usd, index=atlas.index)], axis=1).max(axis=1)
    risk = risk.fillna(float(args.min_risk_usd)).clip(lower=float(args.min_risk_usd)).to_numpy(float)

    result: dict[str, np.ndarray] = {
        "probe_horizon_minutes": np.full(n, int(args.horizon_minutes), dtype=int),
        "synthetic_risk_usd": risk.copy(),
        "entry_valid": np.full(n, False, dtype=bool),
        "entry_invalid_reason": _init_object(n),
        "entry_tick_time": _init_object(n),
        "entry_tick_lag_sec": _init_float(n),
        "entry_tick_bid": _init_float(n),
        "entry_tick_ask": _init_float(n),
        "entry_tick_mid": _init_float(n),
        "entry_tick_spread": _init_float(n),
        "entry_tick_last": _init_float(n),
        "entry_tick_volume": _init_float(n),
        "entry_tick_volume_real": _init_float(n),
        "entry_tick_flags": _init_float(n),
        "buy_entry_ask_tick": _init_float(n),
        "sell_entry_bid_tick": _init_float(n),
        "buy_plus_2r_price": _init_float(n),
        "buy_minus_1r_price": _init_float(n),
        "sell_plus_2r_price": _init_float(n),
        "sell_minus_1r_price": _init_float(n),
        "buy_time_to_plus_2r_sec": _init_float(n),
        "buy_time_to_minus_1r_sec": _init_float(n),
        "sell_time_to_plus_2r_sec": _init_float(n),
        "sell_time_to_minus_1r_sec": _init_float(n),
        "buy_time_to_plus_2r_min": _init_float(n),
        "buy_time_to_minus_1r_min": _init_float(n),
        "sell_time_to_plus_2r_min": _init_float(n),
        "sell_time_to_minus_1r_min": _init_float(n),
        "buy_plus_2r_tick_time": _init_object(n),
        "buy_minus_1r_tick_time": _init_object(n),
        "sell_plus_2r_tick_time": _init_object(n),
        "sell_minus_1r_tick_time": _init_object(n),
        "buy_plus_2r_tick_bid": _init_float(n),
        "buy_minus_1r_tick_bid": _init_float(n),
        "sell_plus_2r_tick_ask": _init_float(n),
        "sell_minus_1r_tick_ask": _init_float(n),
        "tick_buy_mfe_r": _init_float(n),
        "tick_buy_mae_r": _init_float(n),
        "tick_sell_mfe_r": _init_float(n),
        "tick_sell_mae_r": _init_float(n),
        "horizon_tick_count": _init_float(n),
        "source_tick_month": _init_object(n),
    }

    files = _tick_files(Path(args.tick_dir))
    month_map = {month: path for path in files if (month := _parse_month(path)) is not None}
    processed_months = []

    for month_start in sorted(month_map):
        month_end = month_start + pd.offsets.MonthBegin(1)
        positions = _month_positions(entry_target_ns, month_start, month_end)
        if len(positions) == 0:
            continue

        paths = [month_map[month_start]]
        next_month = month_end
        if next_month in month_map:
            paths.append(month_map[next_month])
        tick_frames = []
        read_end = month_end + pd.Timedelta(minutes=int(args.horizon_minutes) + 10)
        for i, path in enumerate(paths):
            tick_frames.append(_read_ticks(path, end=read_end if i > 0 else None))
        ticks = pd.concat(tick_frames, ignore_index=True)
        ticks = ticks.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "bid", "ask"]).reset_index(drop=True)
        if ticks.empty:
            result["entry_invalid_reason"][positions] = "no_ticks_for_month"
            continue

        ts_ns = ticks["timestamp_utc"].dt.as_unit("ns").astype("int64").to_numpy()
        bid = pd.to_numeric(ticks["bid"], errors="coerce").to_numpy(float)
        ask = pd.to_numeric(ticks["ask"], errors="coerce").to_numpy(float)
        mid = pd.to_numeric(ticks.get("mid"), errors="coerce").to_numpy(float)
        spread = pd.to_numeric(ticks.get("spread"), errors="coerce").to_numpy(float)
        last = pd.to_numeric(ticks.get("last"), errors="coerce").to_numpy(float)
        volume = pd.to_numeric(ticks.get("volume"), errors="coerce").to_numpy(float)
        volume_real = pd.to_numeric(ticks.get("volume_real"), errors="coerce").to_numpy(float)
        flags = pd.to_numeric(ticks.get("flags"), errors="coerce").to_numpy(float)

        entry_idx = np.searchsorted(ts_ns, entry_target_ns[positions], side="left")
        end_idx = np.searchsorted(ts_ns, entry_target_ns[positions] + horizon_ns, side="right") - 1

        in_bounds = entry_idx < len(ts_ns)
        valid_positions = positions[in_bounds]
        valid_entry_idx = entry_idx[in_bounds]
        valid_end_idx = end_idx[in_bounds]
        lag_ns = ts_ns[valid_entry_idx] - entry_target_ns[valid_positions]
        lag_ok = (lag_ns >= 0) & (lag_ns <= max_lag_ns)
        horizon_ok = valid_end_idx >= valid_entry_idx
        ok = lag_ok & horizon_ok

        invalid_positions = valid_positions[~ok]
        for p, lag, h_ok in zip(invalid_positions, lag_ns[~ok], horizon_ok[~ok]):
            if lag < 0:
                reason = "entry_tick_before_target"
            elif lag > max_lag_ns:
                reason = "entry_tick_too_stale_after_bar_close"
            elif not h_ok:
                reason = "no_future_ticks_inside_horizon"
            else:
                reason = "entry_invalid"
            result["entry_invalid_reason"][p] = reason

        out_of_bounds_positions = positions[~in_bounds]
        result["entry_invalid_reason"][out_of_bounds_positions] = "no_entry_tick_after_target"

        positions_ok = valid_positions[ok]
        entry_ok = valid_entry_idx[ok]
        end_ok = valid_end_idx[ok]
        if len(positions_ok) == 0:
            processed_months.append({"month": str(month_start.date()), "entries": int(len(positions)), "valid": 0, "ticks": int(len(ticks))})
            continue

        r = risk[positions_ok]
        buy_entry = ask[entry_ok]
        sell_entry = bid[entry_ok]

        result["entry_valid"][positions_ok] = True
        result["entry_tick_time"][positions_ok] = [_iso_from_ns(int(x)) for x in ts_ns[entry_ok]]
        result["entry_tick_lag_sec"][positions_ok] = (ts_ns[entry_ok] - entry_target_ns[positions_ok]) / 1_000_000_000.0
        result["entry_tick_bid"][positions_ok] = bid[entry_ok]
        result["entry_tick_ask"][positions_ok] = ask[entry_ok]
        result["entry_tick_mid"][positions_ok] = mid[entry_ok]
        result["entry_tick_spread"][positions_ok] = spread[entry_ok]
        result["entry_tick_last"][positions_ok] = last[entry_ok]
        result["entry_tick_volume"][positions_ok] = volume[entry_ok]
        result["entry_tick_volume_real"][positions_ok] = volume_real[entry_ok]
        result["entry_tick_flags"][positions_ok] = flags[entry_ok]
        result["buy_entry_ask_tick"][positions_ok] = buy_entry
        result["sell_entry_bid_tick"][positions_ok] = sell_entry
        result["buy_plus_2r_price"][positions_ok] = buy_entry + 2.0 * r
        result["buy_minus_1r_price"][positions_ok] = buy_entry - r
        result["sell_plus_2r_price"][positions_ok] = sell_entry - 2.0 * r
        result["sell_minus_1r_price"][positions_ok] = sell_entry + r
        result["horizon_tick_count"][positions_ok] = end_ok - entry_ok + 1
        result["source_tick_month"][positions_ok] = month_start.strftime("%Y-%m")

        bid_max = SegmentTree(bid, "max")
        for pos, left, right, threshold, entry_price, rr in zip(positions_ok, entry_ok, end_ok, buy_entry + 2.0 * r, buy_entry, r):
            hit = bid_max.first_ge(int(left), int(right), float(threshold))
            result["tick_buy_mfe_r"][pos] = (bid_max.range_value(int(left), int(right)) - entry_price) / rr
            if hit >= 0:
                sec = (ts_ns[hit] - entry_target_ns[pos]) / 1_000_000_000.0
                result["buy_time_to_plus_2r_sec"][pos] = sec
                result["buy_time_to_plus_2r_min"][pos] = sec / 60.0
                result["buy_plus_2r_tick_time"][pos] = _iso_from_ns(int(ts_ns[hit]))
                result["buy_plus_2r_tick_bid"][pos] = bid[hit]
        del bid_max

        bid_min = SegmentTree(bid, "min")
        for pos, left, right, threshold, entry_price, rr in zip(positions_ok, entry_ok, end_ok, buy_entry - r, buy_entry, r):
            hit = bid_min.first_le(int(left), int(right), float(threshold))
            result["tick_buy_mae_r"][pos] = (bid_min.range_value(int(left), int(right)) - entry_price) / rr
            if hit >= 0:
                sec = (ts_ns[hit] - entry_target_ns[pos]) / 1_000_000_000.0
                result["buy_time_to_minus_1r_sec"][pos] = sec
                result["buy_time_to_minus_1r_min"][pos] = sec / 60.0
                result["buy_minus_1r_tick_time"][pos] = _iso_from_ns(int(ts_ns[hit]))
                result["buy_minus_1r_tick_bid"][pos] = bid[hit]
        del bid_min

        ask_min = SegmentTree(ask, "min")
        for pos, left, right, threshold, entry_price, rr in zip(positions_ok, entry_ok, end_ok, sell_entry - 2.0 * r, sell_entry, r):
            hit = ask_min.first_le(int(left), int(right), float(threshold))
            result["tick_sell_mfe_r"][pos] = (entry_price - ask_min.range_value(int(left), int(right))) / rr
            if hit >= 0:
                sec = (ts_ns[hit] - entry_target_ns[pos]) / 1_000_000_000.0
                result["sell_time_to_plus_2r_sec"][pos] = sec
                result["sell_time_to_plus_2r_min"][pos] = sec / 60.0
                result["sell_plus_2r_tick_time"][pos] = _iso_from_ns(int(ts_ns[hit]))
                result["sell_plus_2r_tick_ask"][pos] = ask[hit]
        del ask_min

        ask_max = SegmentTree(ask, "max")
        for pos, left, right, threshold, entry_price, rr in zip(positions_ok, entry_ok, end_ok, sell_entry + r, sell_entry, r):
            hit = ask_max.first_ge(int(left), int(right), float(threshold))
            result["tick_sell_mae_r"][pos] = (entry_price - ask_max.range_value(int(left), int(right))) / rr
            if hit >= 0:
                sec = (ts_ns[hit] - entry_target_ns[pos]) / 1_000_000_000.0
                result["sell_time_to_minus_1r_sec"][pos] = sec
                result["sell_time_to_minus_1r_min"][pos] = sec / 60.0
                result["sell_minus_1r_tick_time"][pos] = _iso_from_ns(int(ts_ns[hit]))
                result["sell_minus_1r_tick_ask"][pos] = ask[hit]
        del ask_max

        processed_months.append({
            "month": str(month_start.date()),
            "entries": int(len(positions)),
            "valid": int(len(positions_ok)),
            "ticks": int(len(ticks)),
        })
        print(f"{month_start.strftime('%Y-%m')}: entries={len(positions)} valid={len(positions_ok)} ticks={len(ticks)}", flush=True)

    result_df = pd.concat([pd.DataFrame(result), context], axis=1)

    out_full = out_dir / "TICK_DUAL_PROBE_EVERY_M5.csv.gz"
    out_sample = out_dir / "TICK_DUAL_PROBE_SAMPLE_1000.csv"
    out_parquet = out_dir / "TICK_DUAL_PROBE_EVERY_M5.parquet"
    summary_path = out_dir / "TICK_DUAL_PROBE_SUMMARY.json"
    report_path = ROOT / "TICK_DUAL_PROBE_REPORT.md"

    result_df.to_csv(out_full, index=False, compression="gzip")
    result_df.head(1000).to_csv(out_sample, index=False)
    result_df.to_parquet(out_parquet, index=False)

    summary = {
        "rows": int(len(result_df)),
        "columns": int(len(result_df.columns)),
        "valid_entries": int(result_df["entry_valid"].sum()),
        "invalid_entries": int((~result_df["entry_valid"]).sum()),
        "horizon_minutes": int(args.horizon_minutes),
        "max_entry_lag_seconds": float(args.max_entry_lag_seconds),
        "tick_months_processed": processed_months,
        "important_note": (
            "This is not a race-winner table. It records exact tick-order clocks "
            "for BUY +2R, BUY -1R, SELL +2R, and SELL -1R from every M5 signal bar."
        ),
        "outputs": {
            "full_csv_gz": str(out_full.relative_to(ROOT)),
            "full_parquet": str(out_parquet.relative_to(ROOT)),
            "sample_csv": str(out_sample.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Tick Dual Probe Report

## What This Is

This rebuilds the experiment from raw MT5 ticks.

For every M5 signal bar, the table opens a synthetic BUY and SELL at the first
available tick after the M5 bar closes. It then records exact tick-order clocks:

- BUY time to `+2R`,
- BUY time to `-1R`,
- SELL time to `+2R`,
- SELL time to `-1R`.

This report does not crown a race winner. It only creates the raw tape table.

## Counts

- Rows: `{len(result_df):,}`
- Columns: `{len(result_df.columns):,}`
- Valid entries: `{int(result_df['entry_valid'].sum()):,}`
- Invalid entries: `{int((~result_df['entry_valid']).sum()):,}`
- Horizon: `{int(args.horizon_minutes)}` minutes
- Max allowed entry tick lag after M5 close: `{float(args.max_entry_lag_seconds)}` seconds

## Context Included

Each row includes:

- exact entry tick bid/ask/mid/spread/volume/flags/source/symbol,
- full current market-atlas row,
- previous M5 raw MT5 bar,
- signal/current M5 raw MT5 bar,
- next M5 raw MT5 bar,
- exact tick timestamps and prices for each `+2R` / `-1R` event,
- exact tick MFE/MAE within the horizon.

## Files

- `{out_full.relative_to(ROOT)}`
- `{out_parquet.relative_to(ROOT)}`
- `{out_sample.relative_to(ROOT)}`
- `{summary_path.relative_to(ROOT)}`
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", default=str(DEFAULT_ATLAS))
    parser.add_argument("--m5", default=str(DEFAULT_M5))
    parser.add_argument("--tick-dir", default=str(DEFAULT_TICK_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--horizon-minutes", type=int, default=360)
    parser.add_argument("--max-entry-lag-seconds", type=float, default=120.0)
    parser.add_argument("--min-risk-usd", type=float, default=0.60)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
