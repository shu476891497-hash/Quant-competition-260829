"""NQ 标准化曲率 H5 的预声明稳健性复核，不下单、不再调公式。"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class NqNcurvH5Robustness(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        future = self.add_future(
            Futures.Indices.NASDAQ_100_E_MINI,
            Resolution.DAILY,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.RAW,
            contract_depth_offset=0,
        )
        future.set_filter(7, 365)
        self.canonical = future.symbol
        self.rows = []
        self.last_date = None

    def on_data(self, data):
        if self.last_date == self.time.date():
            return
        chain = data.future_chains.get(self.canonical)
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
        front, second, third = contracts[:3]
        gap12 = (second.expiry - front.expiry).days
        gap23 = (third.expiry - second.expiry).days
        if gap12 <= 0 or gap23 <= 0:
            return
        carry12 = log(front.last_price / second.last_price) * 365.0 / gap12
        carry23 = log(second.last_price / third.last_price) * 365.0 / gap23
        scale = abs(carry12) + abs(carry23)
        ncurv = (carry12 - carry23) / scale if scale > 1e-12 else 0.0
        self.rows.append(
            {
                "date": self.time.date(),
                "front_symbol": front.symbol,
                "dte": (front.expiry - self.time).days,
                "price": float(front.last_price),
                "factor": ncurv,
            }
        )
        self.last_date = self.time.date()

    def on_end_of_algorithm(self):
        samples = self._samples()
        filters = {
            "ALL": lambda sample: True,
            "DTE30": lambda sample: sample["dte"] >= 30,
            "DTE45": lambda sample: sample["dte"] >= 45,
            "TRAIN": lambda sample: sample["year"] <= 2017,
            "VALID": lambda sample: 2018 <= sample["year"] <= 2022,
            "HOLD": lambda sample: sample["year"] >= 2023,
        }
        for name, condition in filters.items():
            subset = [sample for sample in samples if condition(sample)]
            self.set_runtime_statistic(name, self._format_metrics(subset))

        yearly = []
        positive_years = 0
        usable_years = 0
        for year in sorted({sample["year"] for sample in samples}):
            subset = [sample for sample in samples if sample["year"] == year]
            if len(subset) < 60:
                continue
            ic = self._correlation(subset)
            positive_years += int(ic > 0)
            usable_years += 1
            yearly.append(f"{year}:{ic:.2f}")
        self.set_runtime_statistic(
            "YEAR_SIGN", f"positive={positive_years}/{usable_years}"
        )
        self.log("YEAR_IC|" + ";".join(yearly))

    def _samples(self):
        samples = []
        horizon = 5
        for index in range(len(self.rows) - horizon - 1):
            decision = self.rows[index]
            entry = self.rows[index + 1]
            exit_row = self.rows[index + 1 + horizon]
            if not (
                decision["front_symbol"]
                == entry["front_symbol"]
                == exit_row["front_symbol"]
            ):
                continue
            samples.append(
                {
                    "year": decision["date"].year,
                    "dte": decision["dte"],
                    "x": decision["factor"],
                    "y": log(exit_row["price"] / entry["price"]),
                }
            )
        return samples

    def _format_metrics(self, samples):
        if len(samples) < 60:
            return f"N{len(samples)}"
        x = np.asarray([sample["x"] for sample in samples], dtype=float)
        y = np.asarray([sample["y"] for sample in samples], dtype=float)
        ic = float(np.corrcoef(x, y)[0, 1])
        rank_ic = float(np.corrcoef(self._rank(x), self._rank(y))[0, 1])
        low, high = np.quantile(x, [0.2, 0.8])
        spread = (
            float(np.mean(y[x >= high])) - float(np.mean(y[x <= low]))
        ) * 1e4
        nw_t = self._newey_west_ic_t(x, y, 4)
        return (
            f"N{len(x)}/I{ic:.3f}/R{rank_ic:.3f}/T{nw_t:.2f}/Q{spread:.0f}"
        )

    @staticmethod
    def _correlation(samples):
        x = np.asarray([sample["x"] for sample in samples], dtype=float)
        y = np.asarray([sample["y"] for sample in samples], dtype=float)
        return float(np.corrcoef(x, y)[0, 1])

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
        variance = float(np.dot(centered, centered) / n)
        for k in range(1, min(lag, n - 2) + 1):
            gamma = float(np.dot(centered[k:], centered[:-k]) / n)
            variance += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
        return float(np.mean(products) / sqrt(variance / n))
