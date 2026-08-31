"""
最终候选的分期复核（纯因子，不下单）：
ES=过去 252 日回归剔除整体斜率后的曲率；NQ=原始曲率。
2023-2026 单列为 HOLD，提醒它只是后选择伪样本外，并非真正未见数据。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class EsNqFinalCandidateValidation(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.systems = {}
        for name, ticker in {
            "ES": Futures.Indices.SP_500_E_MINI,
            "NQ": Futures.Indices.NASDAQ_100_E_MINI,
        }.items():
            future = self.add_future(
                ticker, Resolution.DAILY, extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.RAW,
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
                c for c in chain if c.expiry > self.time + timedelta(days=7)
                and c.last_price > 0 and c.open_interest > 0
            ], key=lambda c: c.expiry)
            if len(contracts) < 3:
                continue
            f, s, t = contracts[:3]
            g12, g23, g13 = (
                (s.expiry - f.expiry).days,
                (t.expiry - s.expiry).days,
                (t.expiry - f.expiry).days,
            )
            if min(g12, g23, g13) <= 0:
                continue
            c12 = log(f.last_price / s.last_price) * 365.0 / g12
            c23 = log(s.last_price / t.last_price) * 365.0 / g23
            curv = c12 - c23
            slope = log(f.last_price / t.last_price) * 365.0 / g13
            ch, sh = system["curv"], system["slope"]
            if len(ch) >= 126:
                pc = np.asarray(ch[-252:], dtype=float)
                ps = np.asarray(sh[-252:], dtype=float)
                variance = float(np.var(ps, ddof=1))
                beta = float(np.cov(ps, pc, ddof=1)[0, 1]) / variance if variance > 0 else 0
                alpha = float(np.mean(pc) - beta * np.mean(ps))
                system["rows"].append({
                    "time": self.time, "front": f.symbol,
                    "price": float(f.last_price),
                    "dte": (f.expiry - self.time).days,
                    "value": curv - alpha - beta * slope
                    if system["name"] == "ES" else curv,
                })
            ch.append(curv)
            sh.append(slope)
            if len(ch) > 300:
                del ch[0]
                del sh[0]

    def on_end_of_algorithm(self):
        for system in self.systems.values():
            for horizon in (1, 5, 21):
                samples = self._samples(system["rows"], horizon)
                parts = []
                for label, predicate in (
                    ("P1", lambda r: r["time"].year <= 2015),
                    ("P2", lambda r: 2016 <= r["time"].year <= 2020),
                    ("P3", lambda r: 2021 <= r["time"].year <= 2022),
                    ("HOLD", lambda r: r["time"].year >= 2023),
                    ("D30", lambda r: r["dte"] >= 30),
                ):
                    subset = [r for r in samples if predicate(r)]
                    if len(subset) < 100:
                        parts.append(f"{label}:N{len(subset)}")
                        continue
                    ic, ric, nw_t, q = self._metrics(
                        [r["x"] for r in subset], [r["y"] for r in subset], horizon - 1
                    )
                    parts.append(
                        f"{label}:N{len(subset)}/I{ic:.3f}/R{ric:.3f}/T{nw_t:.1f}/Q{q:.0f}"
                    )
                self.set_runtime_statistic(
                    f"{system['name']}_{'ORTH' if system['name']=='ES' else 'CURV'}_H{horizon}",
                    ";".join(parts),
                )

    @staticmethod
    def _samples(rows, horizon):
        out = []
        for i in range(len(rows) - horizon):
            now, future = rows[i], rows[i + horizon]
            if now["front"] != future["front"]:
                continue
            y = log(future["price"] / now["price"])
            if np.isfinite(now["value"]) and np.isfinite(y):
                out.append({"x": float(now["value"]), "y": float(y), **now})
        return out

    def _metrics(self, xv, yv, lag):
        x, y = np.asarray(xv), np.asarray(yv)
        ic = float(np.corrcoef(x, y)[0, 1])
        ric = float(np.corrcoef(self._rank(x), self._rank(y))[0, 1])
        lo, hi = np.quantile(x, [0.2, 0.8])
        q = (float(np.mean(y[x >= hi])) - float(np.mean(y[x <= lo]))) * 1e4
        z = ((x - np.mean(x)) / np.std(x, ddof=1)) * ((y - np.mean(y)) / np.std(y, ddof=1))
        c, n = z - np.mean(z), len(z)
        variance = float(np.dot(c, c) / n)
        max_lag = min(lag, n - 2)
        for k in range(1, max_lag + 1):
            variance += 2 * (1 - k / (max_lag + 1)) * float(np.dot(c[k:], c[:-k]) / n)
        return ic, ric, float(np.mean(z) / sqrt(variance / n)), q

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
