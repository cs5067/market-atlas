#!/usr/bin/env python3
"""Build BUY and SELL winner tables from the true tick dual-probe table.

This does not split or train anything. It crowns winners from the raw tick
clocks, then writes one full evidence table for BUY winners and one for SELL
winners. Each winner row preserves every column from the dual-probe table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PROBE = ROOT / "data" / "market_atlas" / "tick_dual_probe" / "TICK_DUAL_PROBE_EVERY_M5.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "market_atlas" / "tick_winner_tables"


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _clean_before_stop(tp: pd.Series, sl: pd.Series) -> pd.Series:
    return tp.notna() & (sl.isna() | (tp < sl))


def _insert_front(out: pd.DataFrame, name: str, values) -> None:
    out.insert(0, name, values)


def _elapsed_after_entry(df: pd.DataFrame, event_col: str) -> pd.Series:
    entry = pd.to_datetime(df["entry_tick_time"], utc=True, errors="coerce")
    event = pd.to_datetime(df[event_col], utc=True, errors="coerce")
    return (event - entry).dt.total_seconds()


def _winner_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    side_u = side.upper()
    if side_u == "BUY":
        mask = df["tick_race_winner"].eq("BUY")
        opponent = "SELL"
        entry = "buy_entry_ask_tick"
        target = "buy_plus_2r_price"
        stop = "buy_minus_1r_price"
        tp_sec = "buy_time_to_plus_2r_sec"
        sl_sec = "buy_time_to_minus_1r_sec"
        tp_after_entry_sec = "buy_elapsed_to_plus_2r_after_entry_sec"
        sl_after_entry_sec = "buy_elapsed_to_minus_1r_after_entry_sec"
        tp_min = "buy_time_to_plus_2r_min"
        sl_min = "buy_time_to_minus_1r_min"
        mfe = "tick_buy_mfe_r"
        mae = "tick_buy_mae_r"
        tp_tick_time = "buy_plus_2r_tick_time"
        sl_tick_time = "buy_minus_1r_tick_time"
        opp_tp_sec = "sell_time_to_plus_2r_sec"
        opp_sl_sec = "sell_time_to_minus_1r_sec"
    else:
        mask = df["tick_race_winner"].eq("SELL")
        opponent = "BUY"
        entry = "sell_entry_bid_tick"
        target = "sell_plus_2r_price"
        stop = "sell_minus_1r_price"
        tp_sec = "sell_time_to_plus_2r_sec"
        sl_sec = "sell_time_to_minus_1r_sec"
        tp_after_entry_sec = "sell_elapsed_to_plus_2r_after_entry_sec"
        sl_after_entry_sec = "sell_elapsed_to_minus_1r_after_entry_sec"
        tp_min = "sell_time_to_plus_2r_min"
        sl_min = "sell_time_to_minus_1r_min"
        mfe = "tick_sell_mfe_r"
        mae = "tick_sell_mae_r"
        tp_tick_time = "sell_plus_2r_tick_time"
        sl_tick_time = "sell_minus_1r_tick_time"
        opp_tp_sec = "buy_time_to_plus_2r_sec"
        opp_sl_sec = "buy_time_to_minus_1r_sec"

    out = df.loc[mask].copy()
    # Convenience columns up front; all original evidence columns stay attached.
    _insert_front(out, "opponent_time_to_minus_1r_sec", out[opp_sl_sec])
    _insert_front(out, "opponent_time_to_plus_2r_sec", out[opp_tp_sec])
    _insert_front(out, "winner_minus_1r_tick_time", out[sl_tick_time])
    _insert_front(out, "winner_plus_2r_tick_time", out[tp_tick_time])
    _insert_front(out, "winner_mae_r", out[mae])
    _insert_front(out, "winner_mfe_r", out[mfe])
    _insert_front(out, "winner_elapsed_to_minus_1r_after_entry_min", out[sl_after_entry_sec] / 60.0)
    _insert_front(out, "winner_elapsed_to_plus_2r_after_entry_min", out[tp_after_entry_sec] / 60.0)
    _insert_front(out, "winner_elapsed_to_minus_1r_after_entry_sec", out[sl_after_entry_sec])
    _insert_front(out, "winner_elapsed_to_plus_2r_after_entry_sec", out[tp_after_entry_sec])
    _insert_front(out, "winner_time_to_minus_1r_min", out[sl_min])
    _insert_front(out, "winner_time_to_plus_2r_min", out[tp_min])
    _insert_front(out, "winner_time_to_minus_1r_sec", out[sl_sec])
    _insert_front(out, "winner_time_to_plus_2r_sec", out[tp_sec])
    _insert_front(out, "winner_minus_1r_price", out[stop])
    _insert_front(out, "winner_plus_2r_price", out[target])
    _insert_front(out, "winner_entry_price", out[entry])
    _insert_front(out, "loser_side", opponent)
    _insert_front(out, "winner_side", side_u)
    return out


def build(args: argparse.Namespace) -> dict:
    probe_path = Path(args.probe)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(probe_path)
    entry_valid = df["entry_valid"].astype(bool)
    buy_tp = _series(df, "buy_time_to_plus_2r_sec")
    buy_sl = _series(df, "buy_time_to_minus_1r_sec")
    sell_tp = _series(df, "sell_time_to_plus_2r_sec")
    sell_sl = _series(df, "sell_time_to_minus_1r_sec")

    buy_clean = entry_valid & _clean_before_stop(buy_tp, buy_sl)
    sell_clean = entry_valid & _clean_before_stop(sell_tp, sell_sl)

    both = buy_clean & sell_clean
    buy_only = buy_clean & ~sell_clean
    sell_only = sell_clean & ~buy_clean
    buy_first = both & (buy_tp < sell_tp)
    sell_first = both & (sell_tp < buy_tp)
    tie = both & (buy_tp == sell_tp)

    winner = np.full(len(df), "NONE", dtype=object)
    reason = np.full(len(df), "no_clean_plus_2r_before_own_minus_1r", dtype=object)
    winner[buy_only.to_numpy()] = "BUY"
    reason[buy_only.to_numpy()] = "buy_only_clean_plus_2r_before_minus_1r"
    winner[sell_only.to_numpy()] = "SELL"
    reason[sell_only.to_numpy()] = "sell_only_clean_plus_2r_before_minus_1r"
    winner[buy_first.to_numpy()] = "BUY"
    reason[buy_first.to_numpy()] = "both_clean_buy_plus_2r_first"
    winner[sell_first.to_numpy()] = "SELL"
    reason[sell_first.to_numpy()] = "both_clean_sell_plus_2r_first"
    reason[tie.to_numpy()] = "both_clean_plus_2r_same_tick_tie"

    df = df.copy()
    df["buy_elapsed_to_plus_2r_after_entry_sec"] = _elapsed_after_entry(df, "buy_plus_2r_tick_time")
    df["buy_elapsed_to_minus_1r_after_entry_sec"] = _elapsed_after_entry(df, "buy_minus_1r_tick_time")
    df["sell_elapsed_to_plus_2r_after_entry_sec"] = _elapsed_after_entry(df, "sell_plus_2r_tick_time")
    df["sell_elapsed_to_minus_1r_after_entry_sec"] = _elapsed_after_entry(df, "sell_minus_1r_tick_time")
    df.insert(0, "tick_race_winner", winner)
    df.insert(1, "tick_race_reason", reason)
    df.insert(2, "buy_clean_plus_2r_before_minus_1r", buy_clean.to_numpy())
    df.insert(3, "sell_clean_plus_2r_before_minus_1r", sell_clean.to_numpy())
    df.insert(4, "both_clean_plus_2r", both.to_numpy())
    df.insert(5, "tick_race_tie", tie.to_numpy())
    df.insert(6, "buy_minus_sell_plus_2r_time_gap_sec", sell_tp - buy_tp)

    buy = _winner_table(df, "BUY")
    sell = _winner_table(df, "SELL")
    no_winner = df[df["tick_race_winner"].eq("NONE")]

    buy_csv = out_dir / "BUY_WINNERS_EVERY_MT5_FIELD_TICK_TRUE.csv.gz"
    sell_csv = out_dir / "SELL_WINNERS_EVERY_MT5_FIELD_TICK_TRUE.csv.gz"
    buy_parquet = out_dir / "BUY_WINNERS_EVERY_MT5_FIELD_TICK_TRUE.parquet"
    sell_parquet = out_dir / "SELL_WINNERS_EVERY_MT5_FIELD_TICK_TRUE.parquet"
    buy_sample = out_dir / "BUY_WINNERS_SAMPLE_1000_TICK_TRUE.csv"
    sell_sample = out_dir / "SELL_WINNERS_SAMPLE_1000_TICK_TRUE.csv"
    full_labels = out_dir / "TICK_RACE_LABELED_ALL_M5.parquet"
    summary_path = out_dir / "TICK_WINNER_TABLES_SUMMARY.json"
    report_path = ROOT / "TICK_WINNER_TABLES_REPORT.md"

    buy.to_csv(buy_csv, index=False, compression="gzip")
    sell.to_csv(sell_csv, index=False, compression="gzip")
    buy.to_parquet(buy_parquet, index=False)
    sell.to_parquet(sell_parquet, index=False)
    buy.head(1000).to_csv(buy_sample, index=False)
    sell.head(1000).to_csv(sell_sample, index=False)
    df.to_parquet(full_labels, index=False)

    reason_counts = df["tick_race_reason"].value_counts(dropna=False).to_dict()
    winner_counts = df["tick_race_winner"].value_counts(dropna=False).to_dict()
    summary = {
        "source_probe": str(probe_path.relative_to(ROOT) if probe_path.is_absolute() else probe_path),
        "rows": int(len(df)),
        "columns_labeled_all": int(len(df.columns)),
        "winner_table_columns": int(len(buy.columns)),
        "valid_entries": int(entry_valid.sum()),
        "invalid_entries": int((~entry_valid).sum()),
        "buy_winner_rows": int(len(buy)),
        "sell_winner_rows": int(len(sell)),
        "no_winner_rows": int(len(no_winner)),
        "tie_rows": int(tie.sum()),
        "winner_counts": {str(k): int(v) for k, v in winner_counts.items()},
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "rule": (
            "A side is clean if its own +2R tick happens before its own -1R tick. "
            "If both sides are clean, the earlier +2R tick wins. If neither side "
            "is clean, the row is not included in BUY/SELL winner tables."
        ),
        "outputs": {
            "buy_csv_gz": str(buy_csv.relative_to(ROOT)),
            "sell_csv_gz": str(sell_csv.relative_to(ROOT)),
            "buy_parquet": str(buy_parquet.relative_to(ROOT)),
            "sell_parquet": str(sell_parquet.relative_to(ROOT)),
            "buy_sample": str(buy_sample.relative_to(ROOT)),
            "sell_sample": str(sell_sample.relative_to(ROOT)),
            "labeled_all_m5": str(full_labels.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Tick Winner Tables Report

## What This Is

This crowns winners from the true tick-order dual-probe table.

No split. No training. No office/pre/post. This is just the raw winner evidence:

- every BUY winner row,
- every SELL winner row,
- with every available MT5/tick/atlas/context field attached beside it.

## Crown Rule

A side is clean if its own `+2R` tick happens before its own `-1R` tick.

If both BUY and SELL are clean, the side whose `+2R` tick came first wins.

If neither side is clean, the row is not included in either winner table.

## Counts

| bucket | rows |
|---|---:|
| BUY winners | {len(buy):,} |
| SELL winners | {len(sell):,} |
| No winner | {len(no_winner):,} |
| Ties | {int(tie.sum()):,} |
| All M5 rows | {len(df):,} |

Each winner table has `{len(buy.columns):,}` columns.

## Reason Counts

| reason | rows |
|---|---:|
"""
    for key, value in reason_counts.items():
        report += f"| {key} | {int(value):,} |\n"
    report += f"""
## Files

- `{buy_csv.relative_to(ROOT)}`
- `{sell_csv.relative_to(ROOT)}`
- `{buy_parquet.relative_to(ROOT)}`
- `{sell_parquet.relative_to(ROOT)}`
- `{buy_sample.relative_to(ROOT)}`
- `{sell_sample.relative_to(ROOT)}`
- `{full_labels.relative_to(ROOT)}`
- `{summary_path.relative_to(ROOT)}`
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
