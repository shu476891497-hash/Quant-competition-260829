from __future__ import annotations

import numpy as np
import pandas as pd

from .data import align_released_series


def trailing_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    clip: float = 3.0,
) -> pd.Series:
    minimum = min_periods if min_periods is not None else max(2, window // 2)
    mean = series.rolling(window=window, min_periods=minimum).mean()
    std = series.rolling(window=window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).clip(-clip, clip)


def _pcr_candidates(pcr: pd.DataFrame) -> pd.DataFrame:
    series = pcr.set_index("date")["index_pcr"].sort_index().astype(float)
    log_pcr = np.log(series.where(series > 0))
    output = pd.DataFrame(index=series.index)
    output["pcr_level"] = trailing_zscore(np.log(series.rolling(5, min_periods=3).mean()), 252, 126)
    output["pcr_deviation"] = trailing_zscore(log_pcr - np.log(series.rolling(20, min_periods=10).mean()), 252, 126)
    output["pcr_change"] = trailing_zscore(log_pcr.diff(5), 252, 126)
    return output


def _fred_weekly(fred: pd.DataFrame) -> pd.DataFrame:
    base = fred.copy().sort_values("observation_date").set_index("observation_date")
    weekly = base.loc[base["WRESBAL"].notna(), ["WRESBAL", "WALCL", "WTREGEN"]].copy()
    if "RRPONTSYD" in base:
        rrp = base["RRPONTSYD"].dropna().sort_index()
        if not rrp.empty:
            weekly["RRPONTSYD"] = rrp.reindex(weekly.index, method="ffill")
        else:
            weekly["RRPONTSYD"] = np.nan
    else:
        weekly["RRPONTSYD"] = np.nan
    weekly["net"] = weekly["WALCL"] - weekly["WTREGEN"]
    weekly["net_rrp"] = weekly["net"] - weekly["RRPONTSYD"]
    output = pd.DataFrame(index=weekly.index + pd.Timedelta(days=1))
    output["liq_reserves"] = trailing_zscore(np.log(weekly["WRESBAL"].where(weekly["WRESBAL"] > 0)).diff(4), 156, 78).to_numpy()
    output["liq_net"] = trailing_zscore(weekly["net"].diff(4), 156, 78).to_numpy()
    output["liq_net_rrp"] = trailing_zscore(weekly["net_rrp"].diff(4), 156, 78).to_numpy()
    output.index.name = "release_date"
    return output


def _cot_candidates(cftc_symbol: pd.DataFrame) -> pd.DataFrame:
    base = cftc_symbol.sort_values("report_date").set_index("report_date")
    output = pd.DataFrame(index=base.index + pd.Timedelta(days=3))
    output["cot_lev"] = trailing_zscore(base["lev_net_oi"], 156, 78).to_numpy()
    output["cot_asset"] = trailing_zscore(base["asset_net_oi"], 156, 78).to_numpy()
    output["cot_divergence"] = trailing_zscore(base["asset_net_oi"] - base["lev_net_oi"], 156, 78).to_numpy()
    output.index.name = "release_date"
    return output


def build_candidate_panel(
    prices: pd.DataFrame, pcr: pd.DataFrame, fred: pd.DataFrame, cftc: pd.DataFrame
) -> pd.DataFrame:
    pcr_values = _pcr_candidates(pcr)
    fred_values = _fred_weekly(fred)
    frames: list[pd.DataFrame] = []
    for symbol, price_frame in prices.groupby("symbol", sort=True):
        price_frame = price_frame.sort_values("date").copy()
        calendar = pd.DatetimeIndex(price_frame["date"])
        output = price_frame[["date", "symbol", "close", "log_return"]].copy().set_index("date")
        for column in pcr_values:
            output[column] = pcr_values[column].reindex(calendar)
        for column in fred_values:
            output[column] = align_released_series(fred_values[column], calendar, max_sessions=10)
        symbol_cftc = cftc.loc[cftc["symbol"] == symbol]
        cot_values = _cot_candidates(symbol_cftc)
        for column in cot_values:
            output[column] = align_released_series(cot_values[column], calendar, max_sessions=10)
        frames.append(output.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
