"""
ES/NQ 期限结构候选因子的稳健性检验（纯因子，不下单）。

只复核首轮候选：NQ 曲率/斜率与 ES 成交量-持仓比。每个因子分别在
全样本、三个时间段，以及距到期至少 14/30 天的样本上计算 IC 与
Newey-West t 值。未来收益跨换月时丢弃该样本。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class EsNqFactorRobustness(QCAlgorithm):

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
            self.systems[future.symbol] = {"name": name, "rows": []}

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
            total_oi = float(sum(c.open_interest for c in contracts))
            total_volume = float(sum(max(0.0, c.volume) for c in contracts))
            if total_oi <= 0:
                continue
            system["rows"].append({
                "time": self.time,
                "front_symbol": front.symbol,
                "price": float(front.last_price),
                "dte": (front.expiry - self.time).days,
                "factors": {
                    "CURV": carry12 - carry23,
                    "S13": log(front.last_price / third.last_price) * 365.0 / gap13,
                    "VOI": total_volume / total_oi,
                },
            })

    def on_end_of_algorithm(self):
        for system in self.systems.values():
            factors = ("VOI",) if system["name"] == "ES" else ("CURV", "S13")
            for factor in factors:
                for horizon in self.horizons:
                    samples = self._samples(system["rows"], factor, horizon)
                    parts = []
                    for label, predicate in (
                        ("ALL", lambda row: True),
                        ("P1", lambda row: row["time"].year <= 2015),
                        ("P2", lambda row: 2016 <= row["time"].year <= 2020),
                        ("P3", lambda row: row["time"].year >= 2021),
                        ("D14", lambda row: row["dte"] >= 14),
                        ("D30", lambda row: row["dte"] >= 30),
                    ):
                        x = [s["x"] for s in samples if predicate(s)]
                        y = [s["y"] for s in samples if predicate(s)]
                        if len(x) < 100:
                            parts.append(f"{label}:N{len(x)}")
                            continue
                        ic, nw_t = self._ic_t(x, y, horizon - 1)
                        parts.append(f"{label}:N{len(x)}/I{ic:.3f}/T{nw_t:.1f}")
                    self.set_runtime_statistic(
                        f"{system['name']}_{factor}_H{horizon}", ";".join(parts)
                    )

    @staticmethod
    def _samples(rows, factor, horizon):
        out = []
        for index in range(len(rows) - horizon):
            now, future = rows[index], rows[index + horizon]
            if now["front_symbol"] != future["front_symbol"]:
                continue
            x = now["factors"][factor]
            y = log(future["price"] / now["price"])
            if np.isfinite(x) and np.isfinite(y):
                out.append({"x": float(x), "y": float(y), **now})
        return out

    def _ic_t(self, x_values, y_values, lag):
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        ic = float(np.corrcoef(x, y)[0, 1])
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
        return ic, nw_t
