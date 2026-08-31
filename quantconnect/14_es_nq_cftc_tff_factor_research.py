"""
舒歆因子研究：ES/NQ CFTC TFF 资金/持仓（纯因子，不下单）。

使用 CFTC 官方公开 API 的 consolidated TFF Futures-Only 数据：
ES=13874+，NQ=20974+。报告持仓日为周二，但因子最早到周五发布后
才可用；本程序把可用日固定为 report_date + 3 calendar days。
"""

from AlgorithmImports import *
from datetime import datetime, timedelta
from math import log, sqrt
import numpy as np
from cftc_tff_embedded import load_tff_rows


class EsNqCftcTffFactorResearch(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.forward_horizons = (5, 21, 63)
        self.systems = {}
        self.factor_stat_values = {}

        tickers = {
            "ES": (Futures.Indices.SP_500_E_MINI, "13874+"),
            "NQ": (Futures.Indices.NASDAQ_100_E_MINI, "20974+"),
        }
        tff_by_code = self._download_tff()

        for name, (ticker, cftc_code) in tickers.items():
            future = self.add_future(
                ticker,
                Resolution.DAILY,
                extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                contract_depth_offset=0,
            )
            future.set_filter(0, 182)
            self.systems[future.symbol] = {
                "name": name,
                "prices": [],
                "events": [],
                "tff": tff_by_code.get(cftc_code, []),
                "next_tff": 0,
            }

    def _download_tff(self):
        try:
            grouped = load_tff_rows()
            row_count = sum(len(rows) for rows in grouped.values())
            self.set_runtime_statistic("CFTC_DATA", f"embedded_rows={row_count}")
        except Exception as error:
            self.log(f"CFTC_EMBEDDED_ERROR|{type(error).__name__}|{error}")
            self.set_runtime_statistic(
                "CFTC_DATA", f"ERR:{type(error).__name__}:{str(error)[:60]}"
            )
            return {}

        for code, rows in grouped.items():
            rows.sort(key=lambda row: row["report_date"])
            history_asset, history_lev = [], []
            previous = None
            for row in rows:
                row["release_date"] = row["report_date"] + timedelta(days=3)
                history_asset.append(row["asset_mgr"])
                history_lev.append(row["lev_money"])
                row["asset_mgr_change_1w"] = (
                    row["asset_mgr"] - previous["asset_mgr"]
                    if previous is not None else 0.0
                )
                row["lev_money_change_1w"] = (
                    row["lev_money"] - previous["lev_money"]
                    if previous is not None else 0.0
                )
                row["asset_mgr_z156"] = self._causal_zscore(history_asset, 156)
                row["lev_money_z156"] = self._causal_zscore(history_lev, 156)
                previous = row
        return grouped

    @staticmethod
    def _causal_zscore(values, window):
        sample = np.asarray(values[-window:], dtype=float)
        if len(sample) < 26 or np.std(sample, ddof=1) <= 0:
            return 0.0
        return float((sample[-1] - np.mean(sample)) / np.std(sample, ddof=1))

    def on_data(self, data):
        for canonical, system in self.systems.items():
            bar = data.bars.get(canonical)
            if bar is None or bar.close <= 0:
                continue
            system["prices"].append(float(bar.close))

            updated = None
            rows = system["tff"]
            index = system["next_tff"]
            while index < len(rows) and rows[index]["release_date"] <= self.time.date():
                updated = rows[index]
                index += 1
            system["next_tff"] = index

            if updated is not None:
                factors = {
                    "dealer_net_pct": updated["dealer"],
                    "asset_mgr_net_pct": updated["asset_mgr"],
                    "lev_money_net_pct": updated["lev_money"],
                    "nonrept_net_pct": updated["nonrept"],
                    "asset_mgr_change_1w": updated["asset_mgr_change_1w"],
                    "lev_money_change_1w": updated["lev_money_change_1w"],
                    "asset_mgr_z156": updated["asset_mgr_z156"],
                    "lev_money_z156": updated["lev_money_z156"],
                }
                system["events"].append({
                    "release_date": updated["release_date"],
                    "price_index": len(system["prices"]) - 1,
                    "factors": factors,
                })

    def on_end_of_algorithm(self):
        factor_names = (
            "dealer_net_pct",
            "asset_mgr_net_pct",
            "lev_money_net_pct",
            "nonrept_net_pct",
            "asset_mgr_change_1w",
            "lev_money_change_1w",
            "asset_mgr_z156",
            "lev_money_z156",
        )
        for system in self.systems.values():
            for factor in factor_names:
                for horizon in self.forward_horizons:
                    x, y = self._aligned_sample(system, factor, horizon)
                    self._emit(system["name"], factor, horizon, x, y)

    @staticmethod
    def _aligned_sample(system, factor, horizon):
        prices = system["prices"]
        x, y = [], []
        for event in system["events"]:
            i = event["price_index"]
            if i + horizon >= len(prices):
                continue
            value = event["factors"][factor]
            forward_return = log(prices[i + horizon] / prices[i])
            if np.isfinite(value) and np.isfinite(forward_return):
                x.append(float(value))
                y.append(float(forward_return))
        return x, y

    def _emit(self, symbol, factor, horizon, x_values, y_values):
        if len(x_values) < 100:
            return
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if np.std(x) <= 0 or np.std(y) <= 0:
            return
        ic = float(np.corrcoef(x, y)[0, 1])
        rank_ic = float(np.corrcoef(self._rank(x), self._rank(y))[0, 1])
        nw_t = self._newey_west_ic_t(x, y, max(0, horizon // 5 - 1))
        low_cut, high_cut = np.quantile(x, [0.2, 0.8])
        spread_bps = (
            float(np.mean(y[x >= high_cut])) - float(np.mean(y[x <= low_cut]))
        ) * 10_000.0
        short_name = {
            "dealer_net_pct": "DNET",
            "asset_mgr_net_pct": "ANET",
            "lev_money_net_pct": "LNET",
            "nonrept_net_pct": "NNET",
            "asset_mgr_change_1w": "AD1",
            "lev_money_change_1w": "LD1",
            "asset_mgr_z156": "AZ",
            "lev_money_z156": "LZ",
        }[factor]
        statistic_key = f"{symbol}_{short_name}"
        statistic_part = (
            f"H{horizon}:I{ic:.3f}/R{rank_ic:.3f}/T{nw_t:.1f}/Q{spread_bps:.0f}"
        )
        previous_parts = self.factor_stat_values.get(statistic_key, [])
        previous_parts.append(statistic_part)
        self.factor_stat_values[statistic_key] = previous_parts
        self.set_runtime_statistic(statistic_key, ";".join(previous_parts))
        self.log(
            f"F|{symbol}|{short_name}|H{horizon}|N{len(x)}|"
            f"IC{ic:.4f}|RIC{rank_ic:.4f}|T{nw_t:.2f}|Q{spread_bps:.1f}"
        )

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    @staticmethod
    def _newey_west_ic_t(x, y, lag):
        xz = (x - np.mean(x)) / np.std(x, ddof=1)
        yz = (y - np.mean(y)) / np.std(y, ddof=1)
        products = xz * yz
        centered = products - np.mean(products)
        n = len(products)
        long_run_variance = float(np.dot(centered, centered) / n)
        max_lag = min(lag, n - 2)
        for k in range(1, max_lag + 1):
            gamma = float(np.dot(centered[k:], centered[:-k]) / n)
            long_run_variance += 2.0 * (1.0 - k / (max_lag + 1.0)) * gamma
        if long_run_variance <= 0:
            return float("nan")
        return float(np.mean(products) / sqrt(long_run_variance / n))
