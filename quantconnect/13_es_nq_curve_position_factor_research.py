"""
舒歆因子研究：ES/NQ 期限结构 + 资金/持仓（纯因子，不下单）。

每个 t 时点只使用当日已经可见的期货链价格、成交量和 OI；未来
1/5/21 个交易日收益在事后配对。若前月合约在标签区间内变化，该样本
被剔除，避免把未调整的换月跳空当成未来收益。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class EsNqCurvePositionFactorResearch(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.forward_horizons = (1, 5, 21)
        self.systems = {}

        tickers = {
            "ES": Futures.Indices.SP_500_E_MINI,
            "NQ": Futures.Indices.NASDAQ_100_E_MINI,
        }
        for name, ticker in tickers.items():
            future = self.add_future(
                ticker,
                Resolution.DAILY,
                extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.RAW,
                contract_depth_offset=0,
            )
            future.set_filter(7, 365)
            self.systems[future.symbol] = {
                "name": name,
                "rows": [],
                "total_oi_history": [],
            }

    def on_data(self, data):
        for canonical, system in self.systems.items():
            chain = data.future_chains.get(canonical)
            if chain is None:
                continue

            contracts = sorted(
                [
                    contract for contract in chain
                    if contract.expiry > self.time + timedelta(days=7)
                    and contract.last_price > 0
                    and contract.open_interest > 0
                ],
                key=lambda contract: contract.expiry,
            )
            if len(contracts) < 3:
                continue

            front, second, third = contracts[:3]
            gap12 = (second.expiry - front.expiry).days
            gap23 = (third.expiry - second.expiry).days
            gap13 = (third.expiry - front.expiry).days
            if min(gap12, gap23, gap13) <= 0:
                continue

            carry12 = log(front.last_price / second.last_price) * 365.0 / gap12
            carry23 = log(second.last_price / third.last_price) * 365.0 / gap23
            slope13 = log(front.last_price / third.last_price) * 365.0 / gap13
            curvature = carry12 - carry23

            total_oi = float(sum(contract.open_interest for contract in contracts))
            total_volume = float(sum(max(0.0, contract.volume) for contract in contracts))
            if total_oi <= 0:
                continue

            oi_history = system["total_oi_history"]
            oi_history.append(total_oi)
            if len(oi_history) > 400:
                del oi_history[0]

            oi_change_5d = self._log_change(oi_history, 5)
            oi_change_20d = self._log_change(oi_history, 20)
            if oi_change_20d is None:
                continue

            shares = np.asarray(
                [float(contract.open_interest) / total_oi for contract in contracts],
                dtype=float,
            )
            factors = {
                "carry_1_2": carry12,
                "curve_slope_1_3": slope13,
                "curve_curvature": curvature,
                "total_oi_change_5d": oi_change_5d,
                "total_oi_change_20d": oi_change_20d,
                "front_oi_share": float(shares[0]),
                "oi_concentration_hhi": float(np.sum(shares * shares)),
                "volume_to_oi": total_volume / total_oi,
            }
            system["rows"].append({
                "time": self.time,
                "front_symbol": front.symbol,
                "price": float(front.last_price),
                "factors": factors,
            })

    def on_end_of_algorithm(self):
        factor_names = (
            "carry_1_2",
            "curve_slope_1_3",
            "curve_curvature",
            "total_oi_change_5d",
            "total_oi_change_20d",
            "front_oi_share",
            "oi_concentration_hhi",
            "volume_to_oi",
        )
        for system in self.systems.values():
            for factor in factor_names:
                for horizon in self.forward_horizons:
                    x, y = self._aligned_sample(system["rows"], factor, horizon)
                    self._emit(system["name"], factor, horizon, x, y)

    @staticmethod
    def _log_change(history, horizon):
        if len(history) <= horizon or history[-1 - horizon] <= 0:
            return None
        return log(history[-1] / history[-1 - horizon])

    @staticmethod
    def _aligned_sample(rows, factor, horizon):
        x, y = [], []
        for i in range(len(rows) - horizon):
            now = rows[i]
            future = rows[i + horizon]
            if now["front_symbol"] != future["front_symbol"]:
                continue
            value = now["factors"].get(factor)
            if value is None or not np.isfinite(value):
                continue
            forward_return = log(future["price"] / now["price"])
            if np.isfinite(forward_return):
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
        nw_t = self._newey_west_ic_t(x, y, max(0, horizon - 1))
        low_cut, high_cut = np.quantile(x, [0.2, 0.8])
        spread_bps = (
            float(np.mean(y[x >= high_cut])) - float(np.mean(y[x <= low_cut]))
        ) * 10_000.0

        short_name = {
            "carry_1_2": "C12",
            "curve_slope_1_3": "S13",
            "curve_curvature": "CURV",
            "total_oi_change_5d": "DOI5",
            "total_oi_change_20d": "DOI20",
            "front_oi_share": "FOIS",
            "oi_concentration_hhi": "OIHHI",
            "volume_to_oi": "VOI",
        }[factor]
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
