from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SYMBOLS = {"ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F"}
CASH_BENCHMARKS = {"GSPC": "^GSPC"}
IS_SYMBOLS = ("ES", "NQ", "YM")
FRED_SERIES = ("WRESBAL", "WALCL", "WTREGEN", "RRPONTSYD")
CASH_RATE_SERIES = "DGS3MO"
CFTC_CODES = {
    "ES": ("13874+",),
    "NQ": ("20974+",),
    "YM": ("12460+",),
    "RTY": ("23977A", "239742"),
}
FAMILIES = {
    "put_call": ("pcr_level", "pcr_deviation", "pcr_change"),
    "liquidity": ("liq_reserves", "liq_net", "liq_net_rrp"),
    "positioning": ("cot_lev", "cot_asset", "cot_divergence"),
}
PERIODS = {
    "IS": ("2010-06-15", "2018-12-31"),
    "VALIDATION": ("2019-01-01", "2022-12-31"),
    "OOS": ("2023-01-01", "2099-12-31"),
    "FULL": ("2010-06-15", "2099-12-31"),
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    def ensure(self) -> None:
        for path in (self.raw, self.processed, self.outputs, self.figures):
            path.mkdir(parents=True, exist_ok=True)


def default_paths() -> ProjectPaths:
    return ProjectPaths(Path(__file__).resolve().parents[1])
