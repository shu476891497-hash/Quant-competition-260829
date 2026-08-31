"""舒歆批次 02B：ES/NQ CFTC TFF 资金/持仓纯因子检验。

报告周二观察值只在周五发布后进入系统；下一根日线才作为未来收益进入价。
本算法不下单，假设与 FDR 家族已冻结在 SHUXIN_FACTOR_BATCH_02.md。
"""

from AlgorithmImports import *
from datetime import timedelta
from math import erfc, log, sqrt
import numpy as np
from cftc_tff_embedded import load_tff_rows


class EsNqCftcSpreadBatch02(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2026, 6, 2)
        self.set_cash(1_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.horizons = (5, 21)
        self.systems = {}
        tff = load_tff_rows()
        for name, ticker, code in (
            ("ES", Futures.Indices.SP_500_E_MINI, "13874+"),
            ("NQ", Futures.Indices.NASDAQ_100_E_MINI, "20974+"),
        ):
            future = self.add_future(
                ticker,
                Resolution.DAILY,
                extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.RAW,
                contract_depth_offset=0,
            )
            future.set_filter(7, 182)
            reports = sorted(tff[code], key=lambda row: row["report_date"])
            for report in reports:
                report["release_date"] = report["report_date"] + timedelta(days=3)
            self.systems[future.symbol] = {
                "name": name,
                "rows": [],
                "events": [],
                "reports": reports,
                "next_report": 0,
                "previous_report": None,
                "last_date": None,
            }
        self.set_runtime_statistic(
            "CFTC_DATA", f"embedded_rows={sum(len(v) for v in tff.values())}"
        )

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
                ],
                key=lambda contract: contract.expiry,
            )
            if not contracts:
                continue
            front = contracts[0]
            system["rows"].append(
                {"front_symbol": front.symbol, "price": float(front.last_price)}
            )
            system["last_date"] = self.time.date()

            updated = None
            reports = system["reports"]
            index = system["next_report"]
            while index < len(reports) and reports[index]["release_date"] <= self.time.date():
                updated = reports[index]
                index += 1
            system["next_report"] = index
            if updated is None:
                continue
            previous = system["previous_report"]
            factors = {
                "ASPR": updated["asset_mgr_spread"],
                "LSPR": updated["lev_money_spread"],
                "DSPR": updated["dealer_spread"],
                "ALDIV": updated["asset_mgr"] - updated["lev_money"],
                "DAS1": updated["asset_mgr_spread"] - previous["asset_mgr_spread"]
                if previous is not None else np.nan,
                "DLS1": updated["lev_money_spread"] - previous["lev_money_spread"]
                if previous is not None else np.nan,
            }
            system["events"].append(
                {"price_index": len(system["rows"]) - 1, "factors": factors}
            )
            system["previous_report"] = updated

    def on_end_of_algorithm(self):
        for system in self.systems.values():
            for factor in ("ASPR", "LSPR", "DSPR", "ALDIV", "DAS1", "DLS1"):
                parts = []
                for horizon in self.horizons:
                    x, y = self._samples(system, factor, horizon)
                    if len(x) < 100 or np.std(x) <= 0 or np.std(y) <= 0:
                        continue
                    ic, rank_ic, nw_t, spread = self._metrics(
                        x, y, max(0, horizon // 5 - 1)
                    )
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
    def _samples(system, factor, horizon):
        rows = system["rows"]
        x, y = [], []
        for event in system["events"]:
            decision_index = event["price_index"]
            entry_index = decision_index + 1
            exit_index = entry_index + horizon
            if exit_index >= len(rows):
                continue
            decision, entry, exit_row = (
                rows[decision_index],
                rows[entry_index],
                rows[exit_index],
            )
            if not (
                decision["front_symbol"]
                == entry["front_symbol"]
                == exit_row["front_symbol"]
            ):
                continue
            value = event["factors"][factor]
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
