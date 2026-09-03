from __future__ import annotations

import io
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CASH_BENCHMARKS, CASH_RATE_SERIES, CFTC_CODES, FRED_SERIES, SYMBOLS, ProjectPaths
from .utils import latest_completed_us_date, retrieval_record, retry_session, write_bytes, write_json


CBOE_LEGACY_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv"
CBOE_DAILY_URL = "https://www.cboe.com/data/mktstat.aspx?dt={date}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"


def parse_cboe_legacy_csv(text: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s*,\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+)")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        date, call, put, total, ratio = match.groups()
        rows.append(
            {
                "date": pd.to_datetime(date, format="%m/%d/%Y"),
                "call_volume": pd.to_numeric(call, errors="coerce"),
                "put_volume": pd.to_numeric(put, errors="coerce"),
                "total_volume": pd.to_numeric(total, errors="coerce"),
                "index_pcr": pd.to_numeric(ratio, errors="coerce"),
            }
        )
    if not rows:
        raise ValueError("No Cboe legacy observations found")
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def parse_cboe_daily_html(text: str) -> float:
    patterns = (
        r'INDEX PUT/CALL RATIO.{0,100}?value\\?["\']\s*:\s*\\?["\']([0-9]+(?:\.[0-9]+)?)',
        r'INDEX PUT/CALL RATIO</td>\s*<td[^>]*>\s*([0-9]+(?:\.[0-9]+)?)\s*</td>',
        r'INDEX PUT/CALL RATIO.{0,120}?([0-9]+(?:\.[0-9]+)?)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))
    raise ValueError("INDEX PUT/CALL RATIO not found")


def parse_cboe_selected_date(text: str) -> pd.Timestamp | None:
    match = re.search(r'selectedDate\\?["\']\s*:\s*\\?["\'](\d{4}-\d{2}-\d{2})', text)
    return pd.Timestamp(match.group(1)) if match else None


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    date_column = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    value_column = series if series in frame.columns else frame.columns[-1]
    result = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(frame[date_column], errors="coerce"),
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    return result.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)


def parse_cftc_records(records: Sequence[dict[str, object]], symbol: str) -> pd.DataFrame:
    columns = [
        "report_date_as_yyyy_mm_dd",
        "cftc_contract_market_code",
        "open_interest_all",
        "lev_money_positions_long",
        "lev_money_positions_short",
        "asset_mgr_positions_long",
        "asset_mgr_positions_short",
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["report_date", "symbol", "source_code", "open_interest", "lev_net_oi", "asset_net_oi"])
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[columns].copy()
    frame["report_date"] = pd.to_datetime(frame["report_date_as_yyyy_mm_dd"], errors="coerce").dt.normalize()
    frame["source_code"] = frame["cftc_contract_market_code"].astype(str).str.strip()
    numeric = columns[2:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["open_interest"] = frame["open_interest_all"].replace(0, np.nan)
    frame["lev_net_oi"] = (
        frame["lev_money_positions_long"] - frame["lev_money_positions_short"]
    ) / frame["open_interest"]
    frame["asset_net_oi"] = (
        frame["asset_mgr_positions_long"] - frame["asset_mgr_positions_short"]
    ) / frame["open_interest"]
    if symbol == "RTY":
        frame["priority"] = frame["source_code"].map({"23977A": 1, "239742": 2}).fillna(0)
        frame = frame.sort_values(["report_date", "priority"]).drop_duplicates("report_date", keep="last")
    else:
        frame = frame.sort_values("report_date").drop_duplicates("report_date", keep="last")
    frame["symbol"] = symbol
    return frame[["report_date", "symbol", "source_code", "open_interest", "lev_net_oi", "asset_net_oi"]].reset_index(drop=True)


def align_released_series(
    released: pd.Series, calendar: Iterable[pd.Timestamp], max_sessions: int = 10
) -> pd.Series:
    calendar_index = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().drop_duplicates().sort_values()
    output = pd.Series(index=calendar_index, dtype=float)
    clean = released.dropna().copy()
    clean.index = pd.DatetimeIndex(pd.to_datetime(clean.index)).normalize()
    for release_date, value in clean.sort_index().items():
        location = calendar_index.searchsorted(release_date, side="left")
        if location < len(calendar_index):
            output.iloc[location] = float(value)
    return output.ffill(limit=max_sessions - 1)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _download_with_cache(url: str, path: Path, timeout: int = 30, params: dict[str, object] | None = None) -> None:
    try:
        response = retry_session().get(url, params=params, timeout=timeout)
        response.raise_for_status()
        write_bytes(path, response.content)
    except Exception as error:
        if not path.exists():
            raise
        warnings.warn(f"Refresh failed for {url}; using cached {path}: {error}", RuntimeWarning, stacklevel=2)


def download_prices(paths: ProjectPaths, start: str, end: pd.Timestamp, refresh: bool) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    import yfinance as yf

    records: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    raw_dir = paths.raw / "yahoo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol, ticker in SYMBOLS.items():
        raw_path = raw_dir / f"{symbol}.csv"
        try:
            downloaded = yf.download(
                ticker,
                start=start,
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if downloaded.empty:
                raise RuntimeError(f"Yahoo returned no data for {ticker}")
            if isinstance(downloaded.columns, pd.MultiIndex):
                downloaded.columns = downloaded.columns.get_level_values(0)
            downloaded.reset_index().to_csv(raw_path, index=False)
        except Exception as error:
            if not raw_path.exists():
                raise
            warnings.warn(f"Yahoo refresh failed for {ticker}; using cache: {error}", RuntimeWarning, stacklevel=2)
        frame = pd.read_csv(raw_path)
        frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
        frame = frame.loc[frame["date"] <= end].copy()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        frame["symbol"] = symbol
        frame["ticker"] = ticker
        frame["log_return"] = np.log(frame["close"]).diff()
        records.append(frame[["date", "symbol", "ticker", "open", "high", "low", "close", "volume", "log_return"]])
        manifest.append(retrieval_record(raw_path, f"https://finance.yahoo.com/quote/{ticker}", len(frame)))
    result = pd.concat(records, ignore_index=True).sort_values(["symbol", "date"])
    _save_frame(result, paths.processed / "prices.parquet")
    return result, manifest


def download_cash_benchmarks(
    paths: ProjectPaths, start: str, end: pd.Timestamp, refresh: bool
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    raw_dir = paths.raw / "yahoo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol, ticker in CASH_BENCHMARKS.items():
        raw_path = raw_dir / f"{symbol}.csv"
        try:
            downloaded = yf.download(
                ticker,
                start=start,
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if downloaded.empty:
                raise RuntimeError(f"Yahoo returned no benchmark data for {ticker}")
            if isinstance(downloaded.columns, pd.MultiIndex):
                downloaded.columns = downloaded.columns.get_level_values(0)
            downloaded.reset_index().to_csv(raw_path, index=False)
        except Exception as error:
            if not raw_path.exists():
                raise
            warnings.warn(f"Yahoo refresh failed for {ticker}; using cache: {error}", RuntimeWarning, stacklevel=2)
        frame = pd.read_csv(raw_path)
        frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.loc[frame["date"] <= end].dropna(subset=["date", "close"]).sort_values("date")
        frame = frame.drop_duplicates("date", keep="last")
        frame["symbol"] = symbol
        frame["ticker"] = ticker
        frame["log_return"] = np.log(frame["close"]).diff()
        frames.append(frame[["date", "symbol", "ticker", "close", "log_return"]])
        manifest.append(retrieval_record(raw_path, f"https://finance.yahoo.com/quote/{ticker}", len(frame)))
    result = pd.concat(frames, ignore_index=True)
    _save_frame(result, paths.processed / "cash_benchmarks.parquet")
    return result, manifest


def _fetch_cboe_day(date: pd.Timestamp, path: Path) -> tuple[pd.Timestamp, float, Path] | None:
    url = CBOE_DAILY_URL.format(date=date.strftime("%Y-%m-%d"))
    response = retry_session().get(url, headers={"RSC": "1"}, timeout=45)
    response.raise_for_status()
    write_bytes(path, response.content)
    selected = parse_cboe_selected_date(response.text)
    if selected is not None and selected != date:
        return None
    try:
        return date, parse_cboe_daily_html(response.text), path
    except ValueError:
        return None


def download_cboe(
    paths: ProjectPaths, end: pd.Timestamp, refresh: bool, max_workers: int = 8
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    raw_dir = paths.raw / "cboe"
    raw_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = raw_dir / "indexpc.csv"
    if refresh or not legacy_path.exists():
        response = retry_session().get(CBOE_LEGACY_URL, timeout=45)
        response.raise_for_status()
        write_bytes(legacy_path, response.content)
    legacy = parse_cboe_legacy_csv(legacy_path.read_text(encoding="utf-8-sig", errors="replace"))
    observations = legacy[["date", "index_pcr"]].copy()
    observations["source"] = "cboe_legacy_csv"
    manifest = [retrieval_record(legacy_path, CBOE_LEGACY_URL, len(legacy))]

    dates = pd.date_range(max(pd.Timestamp("2019-10-07"), legacy["date"].max() + pd.Timedelta(days=1)), end, freq="B")
    cached_rows: list[dict[str, object]] = []
    pending: list[tuple[pd.Timestamp, Path]] = []
    daily_dir = raw_dir / "daily"
    for date in dates:
        path = daily_dir / f"{date:%Y-%m-%d}.html"
        if path.exists() and not refresh:
            text = path.read_text(encoding="utf-8", errors="replace")
            selected = parse_cboe_selected_date(text)
            if selected is not None and selected != date:
                continue
            try:
                cached_rows.append({"date": date, "index_pcr": parse_cboe_daily_html(text), "source": "cboe_daily_page"})
                manifest.append(retrieval_record(path, CBOE_DAILY_URL.format(date=date.strftime("%Y-%m-%d")), 1))
            except ValueError:
                pass
        else:
            pending.append((date, path))
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_cboe_day, date, path): (date, path) for date, path in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    date, _ = futures[future]
                    warnings.warn(f"Cboe day {date:%Y-%m-%d} failed and was skipped: {error}", RuntimeWarning, stacklevel=2)
                    continue
                if result is None:
                    continue
                date, value, path = result
                cached_rows.append({"date": date, "index_pcr": value, "source": "cboe_daily_page"})
                manifest.append(retrieval_record(path, CBOE_DAILY_URL.format(date=date.strftime("%Y-%m-%d")), 1))
    if cached_rows:
        observations = pd.concat([observations, pd.DataFrame(cached_rows)], ignore_index=True)
    observations = observations.sort_values("date").drop_duplicates("date", keep="last")
    observations = observations.loc[(observations["index_pcr"] > 0) & observations["index_pcr"].notna()].reset_index(drop=True)
    _save_frame(observations, paths.processed / "cboe_pcr.parquet")
    return observations, manifest


def download_fred(paths: ProjectPaths, refresh: bool) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    raw_dir = paths.raw / "fred"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for series in FRED_SERIES:
        url = FRED_URL.format(series=series)
        path = raw_dir / f"{series}.csv"
        _download_with_cache(url, path, timeout=30)
        frame = parse_fred_csv(path.read_text(encoding="utf-8-sig", errors="replace"), series)
        if series != "RRPONTSYD":
            frame["value"] = frame["value"] / 1000.0
        frame = frame.rename(columns={"value": series})
        frames.append(frame)
        manifest.append(retrieval_record(path, url, len(frame)))
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="observation_date", how="outer")
    result = result.sort_values("observation_date").reset_index(drop=True)
    _save_frame(result, paths.processed / "fred_liquidity.parquet")
    return result, manifest


def download_cash_rate(paths: ProjectPaths, refresh: bool) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Download the daily three-month Treasury yield used for futures cash collateral."""
    series = CASH_RATE_SERIES
    url = FRED_URL.format(series=series)
    raw_dir = paths.raw / "fred"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{series}.csv"
    if refresh or not path.exists():
        _download_with_cache(url, path, timeout=30)
    frame = parse_fred_csv(path.read_text(encoding="utf-8-sig", errors="replace"), series)
    frame = frame.rename(
        columns={"observation_date": "date", "value": "annual_rate_pct"}
    ).dropna(subset=["annual_rate_pct"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    _save_frame(frame, paths.processed / "cash_rate.parquet")
    return frame, [retrieval_record(path, url, len(frame))]


def download_cftc(paths: ProjectPaths, start: str, refresh: bool) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    raw_dir = paths.raw / "cftc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    select = ",".join(
        [
            "report_date_as_yyyy_mm_dd",
            "cftc_contract_market_code",
            "open_interest_all",
            "lev_money_positions_long",
            "lev_money_positions_short",
            "asset_mgr_positions_long",
            "asset_mgr_positions_short",
        ]
    )
    for symbol, codes in CFTC_CODES.items():
        path = raw_dir / f"{symbol}.json"
        code_expr = ",".join(f'"{code}"' for code in codes)
        params = {
            "$select": select,
            "$where": f'cftc_contract_market_code in ({code_expr}) and report_date_as_yyyy_mm_dd >= "{start}T00:00:00.000"',
            "$order": "report_date_as_yyyy_mm_dd",
            "$limit": 50000,
        }
        _download_with_cache(CFTC_URL, path, timeout=45, params=params)
        records = json.loads(path.read_text(encoding="utf-8"))
        frame = parse_cftc_records(records, symbol)
        frames.append(frame)
        manifest.append(retrieval_record(path, CFTC_URL, len(frame)))
    result = pd.concat(frames, ignore_index=True).sort_values(["symbol", "report_date"])
    _save_frame(result, paths.processed / "cftc_positions.parquet")
    return result, manifest


def data_quality_report(
    prices: pd.DataFrame,
    pcr: pd.DataFrame,
    fred: pd.DataFrame,
    cftc: pd.DataFrame,
    cash_benchmarks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = {
        "prices": (prices, "date"),
        "cboe_pcr": (pcr, "date"),
        "fred": (fred, "observation_date"),
        "cftc": (cftc, "report_date"),
    }
    if cash_benchmarks is not None:
        sources["cash_benchmarks"] = (cash_benchmarks, "date")
    for name, (frame, date_column) in sources.items():
        rows.append(
            {
                "source": name,
                "rows": len(frame),
                "start": frame[date_column].min(),
                "end": frame[date_column].max(),
                "duplicate_rows": int(frame.duplicated().sum()),
                "duplicate_keys": int(frame.duplicated([date_column] + (["symbol"] if "symbol" in frame else [])).sum()),
                "missing_cells": int(frame.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def download_all(
    paths: ProjectPaths,
    start: str = "2006-01-01",
    end: pd.Timestamp | None = None,
    refresh: bool = False,
    cboe_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    paths.ensure()
    cutoff = (end or latest_completed_us_date()).normalize()
    prices, p_manifest = download_prices(paths, start, cutoff, refresh)
    cash_benchmarks, benchmark_manifest = download_cash_benchmarks(paths, start, cutoff, refresh)
    pcr, cboe_manifest = download_cboe(paths, cutoff, refresh, max_workers=cboe_workers)
    fred, fred_manifest = download_fred(paths, refresh)
    cash_rate, cash_rate_manifest = download_cash_rate(paths, refresh)
    cftc, cftc_manifest = download_cftc(paths, start, refresh)
    quality = data_quality_report(prices, pcr, fred, cftc, cash_benchmarks)
    quality.to_csv(paths.outputs / "data_quality.csv", index=False)
    write_json(
        paths.raw / "manifest.json",
        p_manifest
        + benchmark_manifest
        + cboe_manifest
        + fred_manifest
        + cash_rate_manifest
        + cftc_manifest,
    )
    return {
        "prices": prices,
        "cash_benchmarks": cash_benchmarks,
        "pcr": pcr,
        "fred": fred,
        "cash_rate": cash_rate,
        "cftc": cftc,
        "quality": quality,
    }


def load_processed(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    required = {
        "prices": paths.processed / "prices.parquet",
        "pcr": paths.processed / "cboe_pcr.parquet",
        "fred": paths.processed / "fred_liquidity.parquet",
        "cftc": paths.processed / "cftc_positions.parquet",
        "cash_benchmarks": paths.processed / "cash_benchmarks.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing processed data; run download first: " + ", ".join(missing))
    return {name: pd.read_parquet(path) for name, path in required.items()}
