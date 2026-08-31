"""Point-in-time-safe factor research helpers."""

from quantcta.research.artifacts import save_factor_results
from quantcta.research.diagnostics import causal_winsorize, factor_autocorrelation
from quantcta.research.factor_evaluation import (
    evaluate_factor,
    evaluate_factor_by_year,
    forward_returns,
)
from quantcta.research.multiple_testing import adjust_multiple_tests

__all__ = [
    "adjust_multiple_tests",
    "causal_winsorize",
    "evaluate_factor",
    "evaluate_factor_by_year",
    "factor_autocorrelation",
    "forward_returns",
    "save_factor_results",
]
