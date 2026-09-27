#!/usr/bin/env python3
"""Build the canonical tick tape used by the VR office replay.

The MT5 exporter writes monthly tab-separated CSV files.  This script turns
those files into a clean local dataset and rebuilds candles from the same bid /
ask tape so later tests do not mix broker ticks with unrelated M5 candles.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "data" / "tick_vr"
TIMEFRAMES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


@dataclass
class FileQA:
    file: str
    size_mb: float
    first_timestamp_utc: str
    last_timestamp_utc: str
    rows: int
    bad_rows: int
    duplicate_timestamps: int
    min_bid: float
    max_bid: float
    min_ask: float
    max_ask: float
    spread_min: float
    spread_median: float
    spread_p95: float
    spread_p99: float
    spread_max: float
    nonpositive_spread_rows: int
    max_gap_seconds: float
    gap_over_5m: int
    gap_over_1h: int
    status: str


def _parse_ts(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce", format="mixed")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _month_from_name(path: Path) -> tuple[int, int] | None:
    m = re.search(r"_(20\d{2})_(\d{2})\.csv$", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _iter_raw_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("*_ticks_20??_??.csv"), key=lambda p: p.name)


def _month_overlaps(path: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> bool:
    parsed = _month_from_name(path)
    if not parsed:
        return True
    year, month = parsed
    month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if start is not None and month_end <= start:
        return False
    if end is not None and month_start >= end:
        return False
    return True


def _clean_chunk(chunk: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    cols = {c: c.strip("<>").lower() for c in chunk.columns}
    chunk = chunk.rename(columns=cols)
    needed = {"date", "time", "bid", "ask"}
    if not needed.issubset(set(chunk.columns)):
        raise ValueError(f"MT5 tick CSV missing required columns: {needed - set(chunk.columns)}")
    ts = pd.to_datetime(
        chunk["date"].astype(str) + " " + chunk["time"].astype(str),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    out = pd.DataFrame({
        "timestamp_utc": ts,
        "bid": pd.to_numeric(chunk["bid"], errors="coerce"),
        "ask": pd.to_numeric(chunk["ask"], errors="coerce"),
    })
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    if "last" in chunk.columns:
        out["last"] = pd.to_numeric(chunk["last"], errors="coerce").fillna(0.0)
    if "volume" in chunk.columns:
        out["volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(0.0)
    if "volume_real" in chunk.columns:
        out["volume_real"] = pd.to_numeric(chunk["volume_real"], errors="coerce").fillna(0.0)
    if "flags" in chunk.columns:
        out["flags"] = pd.to_numeric(chunk["flags"], errors="coerce").fillna(0).astype("int64")
    out["source"] = "MT5"
    out["symbol"] = "XAUUSDm"
    mask = out["timestamp_utc"].notna() & out["bid"].notna() & out["ask"].notna()
    if start is not None:
        mask &= out["timestamp_utc"] >= start
    if end is not None:
        mask &= out["timestamp_utc"] < end
    out = out.loc[mask].copy()
    out = out[(out["bid"] > 0) & (out["ask"] > 0)].copy()
    return out


def _read_clean_file(
    path: Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
        clean = _clean_chunk(chunk, start, end)
        if not clean.empty:
            yield clean


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def _qa_frame(path: Path, frames: list[pd.DataFrame], bad_rows: int) -> FileQA:
    if frames:
        df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    else:
        df = pd.DataFrame(columns=["timestamp_utc", "bid", "ask", "spread"])
    rows = int(len(df))
    if rows:
        diffs = df["timestamp_utc"].diff().dt.total_seconds().dropna()
        dup = int(df["timestamp_utc"].duplicated().sum())
        nonpos = int((pd.to_numeric(df["spread"], errors="coerce") <= 0).sum())
        status = "OK"
        if nonpos:
            status = "WARN_NONPOS_SPREAD"
        if rows < 1000:
            status = "WARN_THIN"
        return FileQA(
            file=path.name,
            size_mb=round(path.stat().st_size / 1024 / 1024, 2),
            first_timestamp_utc=df["timestamp_utc"].iloc[0].isoformat(),
            last_timestamp_utc=df["timestamp_utc"].iloc[-1].isoformat(),
            rows=rows,
            bad_rows=int(bad_rows),
            duplicate_timestamps=dup,
            min_bid=round(float(df["bid"].min()), 5),
            max_bid=round(float(df["bid"].max()), 5),
            min_ask=round(float(df["ask"].min()), 5),
            max_ask=round(float(df["ask"].max()), 5),
            spread_min=round(float(df["spread"].min()), 5),
            spread_median=round(_safe_quantile(df["spread"], 0.50), 5),
            spread_p95=round(_safe_quantile(df["spread"], 0.95), 5),
            spread_p99=round(_safe_quantile(df["spread"], 0.99), 5),
            spread_max=round(float(df["spread"].max()), 5),
            nonpositive_spread_rows=nonpos,
            max_gap_seconds=round(float(diffs.max()) if not diffs.empty else 0.0, 3),
            gap_over_5m=int((diffs > 300).sum()) if not diffs.empty else 0,
            gap_over_1h=int((diffs > 3600).sum()) if not diffs.empty else 0,
            status=status,
        )
    return FileQA(
        file=path.name,
        size_mb=round(path.stat().st_size / 1024 / 1024, 2),
        first_timestamp_utc="",
        last_timestamp_utc="",
        rows=0,
        bad_rows=int(bad_rows),
        duplicate_timestamps=0,
        min_bid=0.0,
        max_bid=0.0,
        min_ask=0.0,
        max_ask=0.0,
        spread_min=0.0,
        spread_median=0.0,
        spread_p95=0.0,
        spread_p99=0.0,
        spread_max=0.0,
        nonpositive_spread_rows=0,
        max_gap_seconds=0.0,
        gap_over_5m=0,
        gap_over_1h=0,
        status="NO_ROWS_IN_RANGE",
    )


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _write_qa_report(out_root: Path, tag: str, rows: list[FileQA], payload: dict) -> None:
    md = out_root / f"VR_TICK_DATA_QA.{tag}.md"
    js = out_root / f"vr_tick_data_qa.{tag}.json"
    row_dicts = [asdict(r) for r in rows]
    payload = {**payload, "files": row_dicts}
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    total_rows = sum(r.rows for r in rows)
    total_size = sum(r.size_mb for r in rows)
    gap_warns = sum(r.gap_over_1h for r in rows)
    lines = [
        "# VR Tick Data QA",
        "",
        f"- Tag: `{tag}`",
        f"- Source dir: `{payload.get('source_dir')}`",
        f"- Files inspected: `{len(rows)}`",
        f"- Rows in range: `{total_rows:,}`",
        f"- Source size inspected: `{total_size:,.1f} MB`",
        f"- >1h gaps counted inside monthly files: `{gap_warns}`",
        "",
        "## File Coverage",
        "",
        _table(row_dicts, [
            "file", "status", "rows", "first_timestamp_utc", "last_timestamp_utc",
            "spread_median", "spread_p95", "spread_p99", "max_gap_seconds",
            "gap_over_5m", "gap_over_1h",
        ]),
        "",
        "## Judge Notes",
        "",
        "- `OK` means the file has usable bid/ask ticks in the requested range.",
        "- Gaps around weekends/rollover are normal; huge gaps inside active sessions need review before full promotion tests.",
        "- Timestamps are treated as UTC-normalized broker tape timestamps. If broker server time differs from UTC, the replay still matches this broker tape consistently.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qa(args: argparse.Namespace) -> dict:
    source_dir = Path(args.source_dir).expanduser()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    start = _parse_ts(args.start)
    end = _parse_ts(args.end)
    files = [p for p in _iter_raw_files(source_dir) if _month_overlaps(p, start, end)]
    rows: list[FileQA] = []
    for path in files:
        frames: list[pd.DataFrame] = []
        raw_rows = 0
        for clean in _read_clean_file(path, start=start, end=end, chunksize=args.chunksize):
            raw_rows += len(clean)
            frames.append(clean[["timestamp_utc", "bid", "ask", "spread"]])
            if args.qa_sample_rows and sum(len(f) for f in frames) >= args.qa_sample_rows:
                break
        rows.append(_qa_frame(path, frames, bad_rows=0))
        print(f"QA {path.name}: {rows[-1].rows:,} rows {rows[-1].status}")
    payload = {
        "source_dir": str(source_dir),
        "start": start.isoformat() if start is not None else "",
        "end": end.isoformat() if end is not None else "",
        "tag": args.tag,
    }
    _write_qa_report(out_root, args.tag, rows, payload)
    return {"files": len(rows), "rows": sum(r.rows for r in rows), "report": str(out_root / f"VR_TICK_DATA_QA.{args.tag}.md")}


def run_ingest(args: argparse.Namespace) -> dict:
    source_dir = Path(args.source_dir).expanduser()
    out_root = Path(args.out_root)
    tick_dir = out_root / "ticks" / args.tag
    tick_dir.mkdir(parents=True, exist_ok=True)
    start = _parse_ts(args.start)
    end = _parse_ts(args.end)
    files = [p for p in _iter_raw_files(source_dir) if _month_overlaps(p, start, end)]
    written = []
    qa_rows: list[FileQA] = []
    for path in files:
        parsed = _month_from_name(path)
        suffix = f"{parsed[0]}_{parsed[1]:02d}" if parsed else path.stem
        out_path = tick_dir / f"XAUUSDm_ticks_{suffix}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path}")
            written.append(str(out_path))
            continue
        frames: list[pd.DataFrame] = []
        for clean in _read_clean_file(path, start=start, end=end, chunksize=args.chunksize):
            frames.append(clean)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(["timestamp_utc", "bid", "ask"]).sort_values("timestamp_utc")
            df.to_parquet(out_path, index=False)
            qa_rows.append(_qa_frame(path, [df[["timestamp_utc", "bid", "ask", "spread"]]], bad_rows=0))
            print(f"wrote {out_path} rows={len(df):,}")
            written.append(str(out_path))
        else:
            qa_rows.append(_qa_frame(path, [], bad_rows=0))
            print(f"no rows in range for {path.name}")
    payload = {
        "source_dir": str(source_dir),
        "out_dir": str(tick_dir),
        "start": start.isoformat() if start is not None else "",
        "end": end.isoformat() if end is not None else "",
        "tag": args.tag,
    }
    _write_qa_report(out_root, args.tag, qa_rows, payload)
    return {"written": written, "report": str(out_root / f"VR_TICK_DATA_QA.{args.tag}.md")}


def _read_parquet_range(tick_dir: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    files = sorted(tick_dir.glob("*.parquet"))
    frames = []
    for path in files:
        df = pd.read_parquet(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        if start is not None:
            df = df[df["timestamp_utc"] >= start]
        if end is not None:
            df = df[df["timestamp_utc"] < end]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "bid", "ask", "mid", "spread"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp_utc", "bid", "ask"])
    out = out.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "bid", "ask"])
    return out


def _build_ohlc(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()
    df = ticks.set_index("timestamp_utc").sort_index()
    mid = df["mid"].resample(rule).ohlc()
    bid = df["bid"].resample(rule).ohlc().add_prefix("bid_")
    ask = df["ask"].resample(rule).ohlc().add_prefix("ask_")
    spread = df["spread"].resample(rule).agg(["mean", "median", "max"]).rename(columns={
        "mean": "spread_mean",
        "median": "spread_median",
        "max": "spread_max",
    })
    counts = df["mid"].resample(rule).size().rename("tick_count")
    out = pd.concat([mid, bid, ask, spread, counts], axis=1).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index().rename(columns={
        "timestamp_utc": "datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })
    return out


def _merge_partial_bars(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        return data
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime")
    agg: dict[str, str] = {}
    for col in data.columns:
        if col == "datetime":
            continue
        if col.endswith("_open") or col in {"Open"}:
            agg[col] = "first"
        elif col.endswith("_high") or col in {"High"}:
            agg[col] = "max"
        elif col.endswith("_low") or col in {"Low"}:
            agg[col] = "min"
        elif col.endswith("_close") or col in {"Close"}:
            agg[col] = "last"
        elif col == "tick_count":
            agg[col] = "sum"
        elif col.startswith("spread_"):
            agg[col] = "mean"
        else:
            agg[col] = "last"
    return data.groupby("datetime", as_index=False).agg(agg).sort_values("datetime").reset_index(drop=True)


def run_candles(args: argparse.Namespace) -> dict:
    out_root = Path(args.out_root)
    tick_dir_arg = getattr(args, "tick_dir", "")
    tick_dir = Path(tick_dir_arg) if tick_dir_arg else out_root / "ticks" / args.tag
    candle_dir = out_root / "candles" / args.tag
    candle_dir.mkdir(parents=True, exist_ok=True)
    start = _parse_ts(args.start)
    end = _parse_ts(args.end)
    files = sorted(tick_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No canonical ticks found in {tick_dir}")
    outputs = {}
    partials: dict[str, list[pd.DataFrame]] = {label: [] for label in TIMEFRAMES}
    total_ticks = 0
    for path in files:
        ticks = pd.read_parquet(path)
        ticks["timestamp_utc"] = pd.to_datetime(ticks["timestamp_utc"], utc=True, errors="coerce")
        if start is not None:
            ticks = ticks[ticks["timestamp_utc"] >= start]
        if end is not None:
            ticks = ticks[ticks["timestamp_utc"] < end]
        ticks = ticks.dropna(subset=["timestamp_utc", "bid", "ask"])
        if ticks.empty:
            continue
        ticks = ticks.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc", "bid", "ask"])
        total_ticks += int(len(ticks))
        for label, rule in TIMEFRAMES.items():
            bars = _build_ohlc(ticks, rule)
            if not bars.empty:
                partials[label].append(bars)
        print(f"candles staged from {path.name} ticks={len(ticks):,}")
    if total_ticks <= 0:
        raise SystemExit(f"No canonical ticks found in {tick_dir} for requested range")
    for label in TIMEFRAMES:
        bars = _merge_partial_bars(partials[label])
        if bars.empty:
            continue
        pq = candle_dir / f"XAUUSDm_{label}.parquet"
        csv = candle_dir / f"XAUUSDm_{label}.csv"
        bars.to_parquet(pq, index=False)
        bars.to_csv(csv, index=False)
        outputs[label] = {"parquet": str(pq), "csv": str(csv), "rows": int(len(bars))}
        print(f"wrote {label} candles rows={len(bars):,}")
    return {"tick_rows": int(total_ticks), "outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical MT5 tick VR tape and candles.")
    parser.add_argument("--source-dir", help="Explicit directory containing MT5 tick exports")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--tag", default="full")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--overwrite", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_qa = sub.add_parser("qa")
    p_qa.add_argument("--qa-sample-rows", type=int, default=0, help="Stop each file after this many cleaned rows; 0 means full file.")
    sub.add_parser("ingest")
    p_candles = sub.add_parser("candles")
    p_candles.add_argument("--tick-dir", default="")
    p_all = sub.add_parser("all")
    p_all.add_argument("--qa-sample-rows", type=int, default=0)
    args = parser.parse_args()
    if args.cmd in {"qa", "ingest", "all"} and not args.source_dir:
        parser.error("--source-dir is required for qa, ingest and all")
    if args.cmd == "qa":
        payload = run_qa(args)
    elif args.cmd == "ingest":
        payload = run_ingest(args)
    elif args.cmd == "candles":
        payload = run_candles(args)
    elif args.cmd == "all":
        payload = {"ingest": run_ingest(args), "candles": run_candles(args)}
    else:
        raise SystemExit(2)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
