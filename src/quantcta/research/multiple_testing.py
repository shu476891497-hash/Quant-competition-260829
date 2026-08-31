"""Multiple-testing controls for batches of factor hypotheses."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd


def adjust_multiple_tests(
    results: pd.DataFrame,
    *,
    t_column: str = "newey_west_t",
    false_discovery_rate: float = 0.10,
) -> pd.DataFrame:
    """Add normal-approximation p-values and Benjamini-Hochberg q-values.

    Pass the complete hypothesis family, including failed factors and horizons.
    Applying this function only to the winner after selection is invalid.
    """

    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("results must be a non-empty DataFrame")
    if t_column not in results:
        raise ValueError(f"results must contain {t_column!r}")
    if not 0 < false_discovery_rate < 1:
        raise ValueError("false_discovery_rate must be between zero and one")

    output = results.copy()
    statistics = pd.to_numeric(output[t_column], errors="coerce").to_numpy(dtype=float)
    p_values = np.asarray(
        [
            erfc(abs(value) / sqrt(2.0)) if np.isfinite(value) else np.nan
            for value in statistics
        ],
        dtype=float,
    )
    q_values = np.full(len(output), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(p_values))
    if len(valid_positions):
        sorted_positions = valid_positions[
            np.argsort(p_values[valid_positions], kind="mergesort")
        ]
        ranks = np.arange(1, len(sorted_positions) + 1, dtype=float)
        adjusted = p_values[sorted_positions] * len(sorted_positions) / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q_values[sorted_positions] = np.minimum(adjusted, 1.0)

    output["p_value"] = p_values
    output["bh_q_value"] = q_values
    output["passes_fdr"] = q_values <= false_discovery_rate
    output["fdr_level"] = false_discovery_rate
    output["hypothesis_family_size"] = len(valid_positions)
    return output
