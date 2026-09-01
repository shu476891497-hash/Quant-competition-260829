"""Build compact CFTC trader-participation and concentration data for QC."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

DATASET = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CODES = ("13874+", "20974+")
FIELDS = (
    "report_date_as_yyyy_mm_dd",
    "open_interest_all",
    "asset_mgr_positions_long",
    "asset_mgr_positions_short",
    "lev_money_positions_long",
    "lev_money_positions_short",
    "traders_asset_mgr_long_all",
    "traders_asset_mgr_short_all",
    "traders_lev_money_long_all",
    "traders_lev_money_short_all",
    "conc_net_le_4_tdr_long_all",
    "conc_net_le_4_tdr_short_all",
    "conc_net_le_8_tdr_long_all",
    "conc_net_le_8_tdr_short_all",
    "conc_gross_le_4_tdr_long",
    "conc_gross_le_4_tdr_short",
    "conc_gross_le_8_tdr_long",
    "conc_gross_le_8_tdr_short",
)
BASE_DATE = date(2010, 1, 1)


def fetch(code: str) -> list[dict[str, str]]:
    where = quote(
        f"cftc_contract_market_code='{code}' AND "
        "report_date_as_yyyy_mm_dd>='2010-01-01T00:00:00.000'",
        safe="='>",
    )
    url = (
        f"{DATASET}?$limit=5000&$select={','.join(FIELDS)}"
        f"&$where={where}&$order=report_date_as_yyyy_mm_dd"
    )
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def factor_values(row: dict[str, str]) -> list[int]:
    oi = float(row["open_interest_all"])
    asset_count = float(row["traders_asset_mgr_long_all"]) + float(
        row["traders_asset_mgr_short_all"]
    )
    lev_count = float(row["traders_lev_money_long_all"]) + float(
        row["traders_lev_money_short_all"]
    )
    asset_per_trader = (
        float(row["asset_mgr_positions_long"])
        - float(row["asset_mgr_positions_short"])
    ) / oi / asset_count
    lev_per_trader = (
        float(row["lev_money_positions_long"])
        - float(row["lev_money_positions_short"])
    ) / oi / lev_count
    c4net = (
        float(row["conc_net_le_4_tdr_long_all"])
        - float(row["conc_net_le_4_tdr_short_all"])
    ) / 100.0
    c8net = (
        float(row["conc_net_le_8_tdr_long_all"])
        - float(row["conc_net_le_8_tdr_short_all"])
    ) / 100.0
    gross4 = float(row["conc_gross_le_4_tdr_long"]) + float(
        row["conc_gross_le_4_tdr_short"]
    )
    gross8 = float(row["conc_gross_le_8_tdr_long"]) + float(
        row["conc_gross_le_8_tdr_short"]
    )
    c4ratio = gross4 / gross8
    return [
        round(asset_per_trader * 1_000_000),
        round(lev_per_trader * 1_000_000),
        round(c4net * 10_000),
        round(c8net * 10_000),
        round(c4ratio * 10_000),
    ]


def pack(rows: list[dict[str, str]]) -> str:
    payload = bytearray()
    previous_day = None
    previous_values = None
    for row in rows:
        report_date = datetime.fromisoformat(row[FIELDS[0]].replace("Z", "+00:00")).date()
        day = (report_date - BASE_DATE).days
        values = factor_values(row)
        if previous_day is None:
            payload.extend(struct.pack("<Hiihhh", day, *values))
        else:
            changes = [value - previous for value, previous in zip(values, previous_values)]
            payload.extend(struct.pack("<Biihhh", day - previous_day, *changes))
        previous_day = day
        previous_values = values
    return base64.b64encode(zlib.compress(bytes(payload), level=9)).decode()


def render(grouped: dict[str, list[dict[str, str]]]) -> str:
    blobs = {code: pack(rows) for code, rows in grouped.items()}
    return f'''"""Embedded official CFTC participation/concentration factors."""

import base64
import struct
import zlib
from datetime import date

_ES_B64 = """{blobs["13874+"]}"""
_NQ_B64 = """{blobs["20974+"]}"""


def load_participation_rows():
    base_date = date(2010, 1, 1)
    grouped = {{}}
    for code, payload in (("13874+", _ES_B64), ("20974+", _NQ_B64)):
        raw = zlib.decompress(base64.b64decode(payload))
        day, *current = struct.unpack_from("<Hiihhh", raw, 0)
        offset = 16
        decoded = [(day, current)]
        while offset < len(raw):
            day_change, *changes = struct.unpack_from("<Biihhh", raw, offset)
            day += day_change
            current = [value + change for value, change in zip(current, changes)]
            decoded.append((day, current.copy()))
            offset += 15
        rows = []
        for day, values in decoded:
            asset, lev, c4net, c8net, c4ratio = values
            rows.append({{
                "report_date": base_date.fromordinal(base_date.toordinal() + day),
                "asset_per_trader": asset / 1_000_000.0,
                "lev_per_trader": lev / 1_000_000.0,
                "c4net": c4net / 10_000.0,
                "c8net": c8net / 10_000.0,
                "c4ratio": c4ratio / 10_000.0,
            }})
        grouped[code] = rows
    return grouped
'''


def main() -> None:
    destination = (
        Path(__file__).parents[1]
        / "quantconnect"
        / "cftc_participation_embedded.py"
    )
    grouped = {code: fetch(code) for code in CODES}
    destination.write_text(render(grouped), encoding="utf-8")
    print({code: len(rows) for code, rows in grouped.items()})
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
