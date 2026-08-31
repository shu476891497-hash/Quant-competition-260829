"""舒歆批次 02A：ES/NQ 期限结构与交易所 OI 纯因子检验。

假设在 docs/SHUXIN_FACTOR_BATCH_02.md 中预先冻结。本算法不下单；因子日
收盘形成，下一根日线才进入未来收益，且决策/进入/退出使用同一实际近月合约。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import erfc, log, sqrt
import numpy as np


class EsNqStructureOiBatch02(QCAlgorithm):

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
                "name": name,
                "rows": [],
                "oimat": [],
                "front_share": [],
                "last_date": None,
            }

    def on_data(self, data):
        for canonical, system in self.systems.items():
            if system["last_date"] == self.time.date():
                continue
            chain = data.future_chains.get(canonical)
            if chain is None:
                continue
            contracts = sorted(
                [
                    contract
                    for contract in chain
                    if contract.expiry > self.time + timedelta(days=7)
                    and contract.last_price > 0
                    and contract.open_interest > 0
                    and contract.volume >= 0
                ],
                key=lambda contract: contract.expiry,
            )
            if len(contracts) < 3:
                continue
            curve = contracts[:4]
            front, second, third = curve[:3]
            gap12 = (second.expiry - front.expiry).days
            gap23 = (third.expiry - second.expiry).days
            if gap12 <= 0 or gap23 <= 0:
                continue

            carry12 = log(front.last_price / second.last_price) * 365.0 / gap12
            carry23 = log(second.last_price / third.last_price) * 365.0 / gap23
            denominator = abs(carry12) + abs(carry23)
            normalized_curvature = (
                (carry12 - carry23) / denominator if denominator > 1e-12 else 0.0
            )

            dte = np.asarray(
                [(contract.expiry - self.time).days for contract in curve], dtype=float
            )
            oi = np.asarray([contract.open_interest for contract in curve], dtype=float)
            volume = np.asarray([contract.volume for contract in curve], dtype=float)
            oi_total = float(np.sum(oi))
            volume_total = float(np.sum(volume))
            if oi_total <= 0 or volume_total <= 0 or oi[0] <= 0 or oi[1] <= 0:
                continue
            oi_maturity = float(np.dot(dte, oi) / oi_total)
            volume_maturity = float(np.dot(dte, volume) / volume_total)
            front_share = float(oi[0] / oi_total)
            oimat_history = system["oimat"]
            front_history = system["front_share"]
            factors = {
                "NCURV": normalized_curvature,
                "OIMAT": oi_maturity,
                "OIMD5": oi_maturity - oimat_history[-5]
                if len(oimat_history) >= 5 else np.nan,
                "FOID5": front_share - front_history[-5]
                if len(front_history) >= 5 else np.nan,
                "OIRAT": log(oi[1] / oi[0]),
                "VMOI": volume_maturity - oi_maturity,
            }
            system["rows"].append(
                {
                    "front_symbol": front.symbol,
                    "price": float(front.last_price),
                    "factors": factors,
                }
            )
            oimat_history.append(oi_maturity)
            front_history.append(front_share)
            system["last_date"] = self.time.date()

    def on_end_of_algorithm(self):
        for system in self.systems.values():
            for factor in ("NCURV", "OIMAT", "OIMD5", "FOID5", "OIRAT", "VMOI"):
                parts = []
                for horizon in self.horizons:
                    x, y = self._samples(system["rows"], factor, horizon)
                    if len(x) < 100 or np.std(x) <= 0 or np.std(y) <= 0:
                        continue
                    ic, rank_ic, nw_t, spread = self._metrics(x, y, horizon - 1)
                    p_value = erfc(abs(nw_t) / sqrt(2.0))
                    parts.append(
                        f"H{horizon}:N{len(x)}/I{ic:.3f}/R{rank_ic:.3f}"
                        f"/T{nw_t:.2f}/P{p_value:.4f}/Q{spread:.0f}"
                    )
                    self.log(
                        f"F|{system['name']}|{factor}|H{horizon}|N{len(x)}|"
                        f"IC{ic:.5f}|RIC{rank_ic:.5f}|T{nw_t:.3f}|"
                        f"P{p_value:.6f}|Q{spread:.2f}"
                    )
                self.set_runtime_statistic(
                    f"{system['name']}_{factor}", ";".join(parts)
                )

    @staticmethod
    def _samples(rows, factor, horizon):
        x, y = [], []
        for index in range(len(rows) - horizon - 1):
            decision = rows[index]
            entry = rows[index + 1]
            exit_row = rows[index + 1 + horizon]
            if not (
                decision["front_symbol"]
                == entry["front_symbol"]
                == exit_row["front_symbol"]
            ):
                continue
            value = decision["factors"][factor]
            forward_return = log(exit_row["price"] / entry["price"])
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
        spread = (
            float(np.mean(y[x >= high])) - float(np.mean(y[x <= low]))
        ) * 1e4
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
        nw_t = (
            float(np.mean(products) / sqrt(variance / n))
            if variance > 0 else float("nan")
        )
        return ic, rank_ic, nw_t, spread

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
