"""Auditable factor-result persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def save_factor_results(
    results: pd.DataFrame,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Save factor statistics and the assumptions required to reproduce them."""

    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("results must be a non-empty DataFrame")
    required = {
        "factor",
        "symbol",
        "horizon",
        "n",
        "ic",
        "rank_ic",
        "newey_west_t",
        "q5_minus_q1_bps",
        "availability_lag",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "factor_metrics.csv", index=False)
    manifest = {
        "schema": "quantcta-factor-research-v1",
        "rows": len(results),
        "factors": sorted(results["factor"].astype(str).unique().tolist()),
        "symbols": sorted(results["symbol"].astype(str).unique().tolist()),
        "horizons": sorted(int(value) for value in results["horizon"].unique()),
        "availability_lags": sorted(
            int(value) for value in results["availability_lag"].unique()
        ),
        "metadata": dict(metadata),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return output
