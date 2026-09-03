from __future__ import annotations

import numpy as np
import pandas as pd

from futures_factors.selection import select_family_candidates


def test_selection_uses_only_is_symbols_and_dates() -> None:
    dates = pd.date_range("2018-01-01", periods=80, freq="B")
    varying_close = np.exp(np.cumsum(0.001 + 0.0008 * np.sin(np.arange(len(dates)) / 4)))
    rows = []
    for symbol in ["ES", "NQ", "YM", "RTY"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "good": float(i),
                    "bad": float(-i),
                    "close": float(varying_close[i]) if symbol != "RTY" else float(1 / varying_close[i]),
                }
            )
    panel = pd.DataFrame(rows)
    evidence, chosen = select_family_candidates(
        panel,
        family="test",
        candidates=["good", "bad"],
        is_start="2018-01-01",
        is_end="2018-12-31",
        symbols=("ES", "NQ", "YM"),
        horizon=5,
    )
    assert chosen["candidate"] in {"good", "bad"}
    detail_symbols = set(evidence.loc[evidence["symbol"] != "SUMMARY", "symbol"].dropna())
    assert detail_symbols <= {"ES", "NQ", "YM"}
    assert "RTY" not in set(evidence["symbol"].dropna())
