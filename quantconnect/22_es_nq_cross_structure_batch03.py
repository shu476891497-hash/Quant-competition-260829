"""舒歆批次 03A：ES-NQ 跨市场期限结构纯因子检验，0 下单。"""

from AlgorithmImports import *
from datetime import timedelta
from math import erfc, log, sqrt
import numpy as np


class EsNqCrossStructureBatch03(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.horizons = (1, 5, 21)
        self.symbols = {}
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
            self.symbols[name] = future.symbol
        self.rows = []
        self.front_share_history = {"ES": [], "NQ": []}
        self.last_date = None

    def on_data(self, data):
        if self.last_date == self.time.date():
            return
        snapshots = {}
        for name, canonical in self.symbols.items():
            chain = data.future_chains.get(canonical)
            if chain is None:
                return
            contracts = sorted(
                [
                    contract
                    for contract in chain
                    if contract.expiry > self.time + timedelta(days=7)
                    and contract.last_price > 0
                    and contract.open_interest > 0
                ],
                key=lambda contract: contract.expiry,
            )
            if len(contracts) < 3:
                return
            curve = contracts[:4]
            front, second, third = curve[:3]
            gap12 = (second.expiry - front.expiry).days
            gap23 = (third.expiry - second.expiry).days
            if gap12 <= 0 or gap23 <= 0:
                return
            carry = log(front.last_price / second.last_price) * 365.0 / gap12
            carry23 = log(second.last_price / third.last_price) * 365.0 / gap23
            scale = abs(carry) + abs(carry23)
            ncurv = (carry - carry23) / scale if scale > 1e-12 else 0.0
            open_interest = np.asarray(
                [contract.open_interest for contract in curve], dtype=float
            )
            front_share = float(open_interest[0] / np.sum(open_interest))
            history = self.front_share_history[name]
            foi_change = front_share - history[-5] if len(history) >= 5 else np.nan
            snapshots[name] = {
                "symbol": front.symbol,
                "price": float(front.last_price),
                "carry": carry,
                "ncurv": ncurv,
                "foi_change": foi_change,
                "front_share": front_share,
            }
        es, nq = snapshots["ES"], snapshots["NQ"]
        self.rows.append(
            {
                "es_symbol": es["symbol"],
                "nq_symbol": nq["symbol"],
                "es_price": es["price"],
                "nq_price": nq["price"],
                "factors": {
                    "CARRYDIFF": nq["carry"] - es["carry"],
                    "NCURVDIFF": nq["ncurv"] - es["ncurv"],
                    "OIMIGDIFF": nq["foi_change"] - es["foi_change"],
                },
            }
        )
        for name in ("ES", "NQ"):
            self.front_share_history[name].append(snapshots[name]["front_share"])
        self.last_date = self.time.date()

    def on_end_of_algorithm(self):
        for factor in ("CARRYDIFF", "NCURVDIFF", "OIMIGDIFF"):
            parts = []
            for horizon in self.horizons:
                x, y = self._samples(factor, horizon)
                if len(x) < 100 or np.std(x) <= 0 or np.std(y) <= 0:
                    continue
                ic, rank_ic, nw_t, spread = self._metrics(x, y, horizon - 1)
                p_value = erfc(abs(nw_t) / sqrt(2.0))
                parts.append(
                    f"H{horizon}:N{len(x)}/I{ic:.3f}/R{rank_ic:.3f}"
                    f"/T{nw_t:.2f}/P{p_value:.4f}/Q{spread:.0f}"
                )
                self.log(
                    f"F|REL|{factor}|H{horizon}|N{len(x)}|IC{ic:.5f}|"
                    f"RIC{rank_ic:.5f}|T{nw_t:.3f}|P{p_value:.6f}|Q{spread:.2f}"
                )
            self.set_runtime_statistic(factor, ";".join(parts))

    def _samples(self, factor, horizon):
        x, y = [], []
        for index in range(len(self.rows) - horizon - 1):
            decision = self.rows[index]
            entry = self.rows[index + 1]
            exit_row = self.rows[index + 1 + horizon]
            if not (
                decision["es_symbol"] == entry["es_symbol"] == exit_row["es_symbol"]
                and decision["nq_symbol"]
                == entry["nq_symbol"]
                == exit_row["nq_symbol"]
            ):
                continue
            value = decision["factors"][factor]
            relative_return = log(exit_row["nq_price"] / entry["nq_price"]) - log(
                exit_row["es_price"] / entry["es_price"]
            )
            if np.isfinite(value) and np.isfinite(relative_return):
                x.append(float(value))
                y.append(float(relative_return))
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
        for k in range(1, min(lag, n - 2) + 1):
            gamma = float(np.dot(centered[k:], centered[:-k]) / n)
            variance += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
        nw_t = float(np.mean(products) / sqrt(variance / n))
        return ic, rank_ic, nw_t, spread

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
