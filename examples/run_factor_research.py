"""Minimal example: one factor function, one uniform evaluation call."""

import numpy as np
import pandas as pd

from quantcta.factors import annualized_curve_curvature
from quantcta.research import evaluate_factor

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
print(result.to_string(index=False))
