"""Minimal example: one factor function, one uniform evaluation call."""

import numpy as np
import pandas as pd

from quantcta.factors import annualized_curve_curvature
from quantcta.research import (
    adjust_multiple_tests,
    causal_winsorize,
    evaluate_factor,
    evaluate_factor_by_year,
    factor_autocorrelation,
    save_factor_results,
)

index = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
trend = 100.0 * np.exp(np.arange(len(index)) * 0.0002)
front = pd.DataFrame({"NQ": trend}, index=index)
middle = pd.DataFrame(
    {"NQ": trend * (1.01 + 0.004 * np.sin(np.arange(len(index)) / 17.0))},
    index=index,
)
back = pd.DataFrame(
    {"NQ": trend * (1.02 + 0.006 * np.sin(np.arange(len(index)) / 31.0))},
    index=index,
)
front_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=30)}, index=index)
middle_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=90)}, index=index)
back_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=180)}, index=index)
contract_ids = pd.DataFrame({"NQ": "NQ_SYNTHETIC"}, index=index)

factor = annualized_curve_curvature(
    front, middle, back, front_expiry, middle_expiry, back_expiry
)
result = evaluate_factor(
    factor,
    front,
    horizons=(1, 5, 21),
    availability_lag=1,
    contract_ids=contract_ids,
    factor_name="curve_curvature",
)
adjusted_result = adjust_multiple_tests(result)
print(adjusted_result.to_string(index=False))

winsorized = causal_winsorize(factor, window=126, min_periods=60)
yearly = evaluate_factor_by_year(
    winsorized,
    front,
    horizons=(1, 5, 21),
    availability_lag=1,
    contract_ids=contract_ids,
    factor_name="curve_curvature_winsorized",
)
autocorrelation = factor_autocorrelation(factor)
output = save_factor_results(
    adjusted_result,
    "runs/example_factor",
    metadata={
        "data_source": "synthetic",
        "entry_rule": "next bar",
        "roll_rule": "same contract from decision through exit",
        "hypothesis_family": "all rows in this example run",
    },
)
print("\nYear-by-year robustness:")
print(yearly.to_string(index=False))
print("\nFactor autocorrelation:")
print(autocorrelation.to_string(index=False))
print(f"\nSaved auditable result to {output}")
