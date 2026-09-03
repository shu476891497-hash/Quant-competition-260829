from __future__ import annotations

import pandas as pd
import pytest

from futures_factors.data import (
    align_released_series,
    parse_cboe_daily_html,
    parse_cboe_legacy_csv,
    parse_cftc_records,
    parse_fred_csv,
)


def test_parse_cboe_legacy_csv_skips_disclaimer_and_normalizes_columns() -> None:
    text = """Disclaimer,not,data
DATE,CALL,PUT,TOTAL,P/C RATIO
11/10/2006,170334,200796,371130,1.18
1/3/2007,273555,372879,646434,1.36
"""
    result = parse_cboe_legacy_csv(text)
    assert list(result.columns) == ["date", "call_volume", "put_volume", "total_volume", "index_pcr"]
    assert result.loc[0, "date"] == pd.Timestamp("2006-11-10")
    assert result.loc[1, "date"] == pd.Timestamp("2007-01-03")
    assert result.loc[1, "index_pcr"] == pytest.approx(1.36)


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('{"name":"INDEX PUT/CALL RATIO","value":"1.29"}', 1.29),
        ('INDEX PUT/CALL RATIO</td><td class="x">0.92</td>', 0.92),
    ],
)
def test_parse_cboe_daily_html_supports_json_and_rendered_table(html: str, expected: float) -> None:
    assert parse_cboe_daily_html(html) == pytest.approx(expected)


def test_parse_fred_csv_handles_missing_dot_and_numeric_values() -> None:
    text = "observation_date,WRESBAL\n2024-01-03,3500000\n2024-01-10,.\n"
    result = parse_fred_csv(text, "WRESBAL")
    assert result.loc[0, "value"] == 3_500_000
    assert pd.isna(result.loc[1, "value"])


def test_parse_cftc_records_handles_zero_oi_and_rty_source_precedence() -> None:
    records = [
        {
            "report_date_as_yyyy_mm_dd": "2017-08-15T00:00:00.000",
            "cftc_contract_market_code": "23977A",
            "open_interest_all": "100",
            "lev_money_positions_long": "60",
            "lev_money_positions_short": "40",
            "asset_mgr_positions_long": "55",
            "asset_mgr_positions_short": "45",
        },
        {
            "report_date_as_yyyy_mm_dd": "2017-08-15T00:00:00.000",
            "cftc_contract_market_code": "239742",
            "open_interest_all": "200",
            "lev_money_positions_long": "130",
            "lev_money_positions_short": "70",
            "asset_mgr_positions_long": "100",
            "asset_mgr_positions_short": "100",
        },
        {
            "report_date_as_yyyy_mm_dd": "2017-08-22T00:00:00.000",
            "cftc_contract_market_code": "239742",
            "open_interest_all": "0",
            "lev_money_positions_long": "1",
            "lev_money_positions_short": "1",
            "asset_mgr_positions_long": "1",
            "asset_mgr_positions_short": "1",
        },
    ]
    result = parse_cftc_records(records, "RTY")
    assert len(result) == 2
    assert result.loc[0, "source_code"] == "239742"
    assert result.loc[0, "lev_net_oi"] == pytest.approx(0.30)
    assert pd.isna(result.loc[1, "lev_net_oi"])


def test_align_released_series_never_backfills_and_expires_after_limit() -> None:
    calendar = pd.date_range("2024-01-01", periods=8, freq="B")
    released = pd.Series([2.0], index=[pd.Timestamp("2024-01-03")])
    aligned = align_released_series(released, calendar, max_sessions=3)
    assert pd.isna(aligned.loc["2024-01-02"])
    assert aligned.loc["2024-01-03"] == 2.0
    assert aligned.loc["2024-01-05"] == 2.0
    assert pd.isna(aligned.loc["2024-01-08"])
