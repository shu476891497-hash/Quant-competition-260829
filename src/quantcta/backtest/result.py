"""Backtest output bundle and reproducible artifact export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    positions: pd.DataFrame
    trades: pd.DataFrame
    gross_pnl_by_instrument: pd.DataFrame
    costs_by_instrument: pd.DataFrame
    net_pnl: pd.Series
    nav: pd.Series
    returns: pd.Series
    notional: pd.DataFrame
    turnover: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    metrics: dict[str, float]
    manifest: dict[str, Any]

    def save(self, directory: str | Path) -> Path:
        """Persist auditable run outputs. The caller controls the destination."""

        output = Path(directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        frames = {
            "positions": self.positions,
            "trades": self.trades,
            "gross_pnl_by_instrument": self.gross_pnl_by_instrument,
            "costs_by_instrument": self.costs_by_instrument,
            "notional": self.notional,
        }
        for name, frame in frames.items():
            frame.to_parquet(output / f"{name}.parquet")
        series = pd.concat(
            {
                "net_pnl": self.net_pnl,
                "nav": self.nav,
                "returns": self.returns,
                "turnover": self.turnover,
                "gross_exposure": self.gross_exposure,
                "net_exposure": self.net_exposure,
            },
            axis=1,
        )
        series.to_parquet(output / "portfolio.parquet")
        (output / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return output
