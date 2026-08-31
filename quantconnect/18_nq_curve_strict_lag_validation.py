"""
NQ curve-curvature validation with strict one-bar availability lag (zero orders).

Factor at t uses the observable futures chain at that close. Entry price is t+1,
exit is t+1+horizon. Decision, entry and exit must reference the same contract.
"""

from AlgorithmImports import *
from datetime import timedelta
from math import log, sqrt
import numpy as np


class NqCurveStrictLagValidation(QCAlgorithm):

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
        )
        future.set_filter(7, 365)
        self.canonical = future.symbol
        self.rows = []

    def on_data(self, data):
        chain = data.future_chains.get(self.canonical)
        if chain is None:
            return
        contracts = sorted([
            contract for contract in chain
            if contract.expiry > self.time + timedelta(days=7)
            and contract.last_price > 0 and contract.open_interest > 0
        ], key=lambda contract: contract.expiry)
        if len(contracts) < 3:
            return
        front, middle, back = contracts[:3]
        gap12 = (middle.expiry - front.expiry).days
        gap23 = (back.expiry - middle.expiry).days
        if min(gap12, gap23) <= 0:
            return
        carry12 = log(front.last_price / middle.last_price) * 365.0 / gap12
        carry23 = log(middle.last_price / back.last_price) * 365.0 / gap23
        self.rows.append({
            "time": self.time,
            "front": front.symbol,
            "price": float(front.last_price),
            "dte": (front.expiry - self.time).days,
            "factor": carry12 - carry23,
        })

    def on_end_of_algorithm(self):
        for horizon in (1, 5, 21):
            samples = self._samples(horizon)
            parts = []
            for label, predicate in (
                ("ALL", lambda row: True),
                ("P1", lambda row: row["time"].year <= 2015),
                ("P2", lambda row: 2016 <= row["time"].year <= 2020),
                ("P3", lambda row: 2021 <= row["time"].year <= 2022),
                ("HOLD", lambda row: row["time"].year >= 2023),
                ("D30", lambda row: row["dte"] >= 30),
            ):
                subset = [row for row in samples if predicate(row)]
                if len(subset) < 100:
                    parts.append(f"{label}:N{len(subset)}")
                    continue
                ic, rank_ic, nw_t, spread = self._metrics(
                    [row["x"] for row in subset],
                    [row["y"] for row in subset],
                    horizon - 1,
                )
                parts.append(
                    f"{label}:N{len(subset)}/I{ic:.3f}/R{rank_ic:.3f}"
                    f"/T{nw_t:.1f}/Q{spread:.0f}"
                )
            self.set_runtime_statistic(
                f"NQ_CURV_STRICT_H{horizon}", ";".join(parts)
            )

    def _samples(self, horizon):
        samples = []
        for decision_index in range(len(self.rows) - horizon - 1):
            entry_index = decision_index + 1
            exit_index = entry_index + horizon
            decision = self.rows[decision_index]
            entry = self.rows[entry_index]
            exit_row = self.rows[exit_index]
            if not (
                decision["front"] == entry["front"] == exit_row["front"]
            ):
                continue
            forward_return = log(exit_row["price"] / entry["price"])
            if np.isfinite(decision["factor"]) and np.isfinite(forward_return):
                samples.append({
                    "x": float(decision["factor"]),
                    "y": float(forward_return),
                    **decision,
                })
        return samples

    def _metrics(self, x_values, y_values, lag):
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        ic = float(np.corrcoef(x, y)[0, 1])
        rank_ic = float(np.corrcoef(self._rank(x), self._rank(y))[0, 1])
        low, high = np.quantile(x, [0.2, 0.8])
        spread = (
            float(np.mean(y[x >= high])) - float(np.mean(y[x <= low]))
        ) * 10_000.0
        xz = (x - np.mean(x)) / np.std(x, ddof=1)
        yz = (y - np.mean(y)) / np.std(y, ddof=1)
        products = xz * yz
        centered = products - np.mean(products)
        n = len(products)
        variance = float(np.dot(centered, centered) / n)
        max_lag = min(lag, n - 2)
        for offset in range(1, max_lag + 1):
            covariance = float(
                np.dot(centered[offset:], centered[:-offset]) / n
            )
            variance += 2.0 * (
                1.0 - offset / (max_lag + 1.0)
            ) * covariance
        nw_t = float(np.mean(products) / sqrt(variance / n))
        return ic, rank_ic, nw_t, spread

    @staticmethod
    def _rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
