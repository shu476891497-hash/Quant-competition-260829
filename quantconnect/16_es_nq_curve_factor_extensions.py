"""
ES/NQ 期限结构扩展因子研究（纯因子，不下单）。

检验曲率的因果标准化、5/20 日变化、以及用过去 252 个观测对整体
曲线斜率做回归后得到的正交曲率。所有滚动统计只使用当日及以前数据；
未来收益跨换月时丢弃样本。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class EsNqCurveFactorExtensions(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.horizons = (1, 5, 21)
        self.systems = {}
        for name, ticker in {
            "ES": Futures.Indices.SP_500_E_MINI,
            "NQ": Futures.Indices.NASDAQ_100_E_MINI,
        }.items():
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
                "name": name, "rows": [], "curv": [], "slope": []
            }

    def on_data(self, data):
        for canonical, system in self.systems.items():
            chain = data.future_chains.get(canonical)
            if chain is None:
                continue
            contracts = sorted([
                c for c in chain
                if c.expiry > self.time + timedelta(days=7)
                and c.last_price > 0 and c.open_interest > 0
            ], key=lambda c: c.expiry)
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
            curv = carry12 - carry23
            slope = log(front.last_price / third.last_price) * 365.0 / gap13
            ch, sh = system["curv"], system["slope"]
            if len(ch) < 126:
                ch.append(curv)
                sh.append(slope)
                continue
            past_curv = np.asarray(ch[-252:], dtype=float)
            past_slope = np.asarray(sh[-252:], dtype=float)
            curv_std = float(np.std(past_curv, ddof=1))
            curv_z = (
                (curv - float(np.mean(past_curv))) / curv_std
                if curv_std > 0 else 0.0
            )
            slope_variance = float(np.var(past_slope, ddof=1))
            beta = (
                float(np.cov(past_slope, past_curv, ddof=1)[0, 1])
                / slope_variance if slope_variance > 0 else 0.0
            )
            alpha = float(np.mean(past_curv) - beta * np.mean(past_slope))
            orthogonal_curv = curv - alpha - beta * slope
            factors = {
                "CURV": curv,
                "CZ": curv_z,
                "CD5": curv - ch[-5],
                "CD20": curv - ch[-20],
                "ORTH": orthogonal_curv,
                "SD5": slope - sh[-5],
            }
            system["rows"].append({
                "front_symbol": front.symbol,
                "price": float(front.last_price),
                "factors": factors,
            })
            ch.append(curv)
            sh.append(slope)
            if len(ch) > 300:
                del ch[0]
                del sh[0]

    def on_end_of_algorithm(self):
        for system in self.systems.values():
            for factor in ("CURV", "CZ", "CD5", "CD20", "ORTH", "SD5"):
                parts = []
                for horizon in self.horizons:
                    x, y = self._samples(system["rows"], factor, horizon)
                    if len(x) < 100:
                        continue
                    ic, rank_ic, nw_t, spread = self._metrics(x, y, horizon - 1)
                    parts.append(
                        f"H{horizon}:N{len(x)}/I{ic:.3f}/R{rank_ic:.3f}"
                        f"/T{nw_t:.1f}/Q{spread:.0f}"
                    )
                self.set_runtime_statistic(
                    f"{system['name']}_{factor}", ";".join(parts)
                )

    @staticmethod
    def _samples(rows, factor, horizon):
        x, y = [], []
        for index in range(len(rows) - horizon):
            now, future = rows[index], rows[index + horizon]
            if now["front_symbol"] != future["front_symbol"]:
                continue
            value = now["factors"][factor]
            forward_return = log(future["price"] / now["price"])
            if np.isfinite(value) and np.isfinite(forward_return):
                x.append(float(value))
                y.append(float(forward_return))
        return x, y

    def _metrics(self, x_values, y_values, lag):
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        ic = float(np.corrcoef(x, y)[0, 1])
        rank_ic = float(np.corrcoef(self._rank(x), self._rank(y))[0, 1])
        low, high = np.quantile(x, [0.2, 0.8])
        spread = (float(np.mean(y[x >= high])) - float(np.mean(y[x <= low]))) * 1e4
        xz = (x - np.mean(x)) / np.std(x, ddof=1)
        yz = (y - np.mean(y)) / np.std(y, ddof=1)
        products = xz * yz
        centered = products - np.mean(products)
        n = len(products)
        variance = float(np.dot(centered, centered) / n)
        max_lag = min(lag, n - 2)
        for k in range(1, max_lag + 1):
            gamma = float(np.dot(centered[k:], centered[:-k]) / n)
            variance += 2.0 * (1.0 - k / (max_lag + 1.0)) * gamma
        nw_t = float(np.mean(products) / sqrt(variance / n))
        return ic, rank_ic, nw_t, spread

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
