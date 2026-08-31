import numpy as np
import pandas as pd

from quantcta.research import adjust_multiple_tests


def test_benjamini_hochberg_adjustment_is_monotone() -> None:
    results = pd.DataFrame(
        {
            "factor": ["a", "b", "c"],
            "newey_west_t": [3.0, 2.0, 0.0],
        }
    )

    adjusted = adjust_multiple_tests(results, false_discovery_rate=0.10)

    assert np.all(np.diff(adjusted["bh_q_value"]) >= 0)
    assert adjusted["passes_fdr"].tolist() == [True, True, False]
    assert adjusted["hypothesis_family_size"].unique().tolist() == [3]


def test_adjustment_keeps_nan_statistics_out_of_family() -> None:
    results = pd.DataFrame(
        {
            "factor": ["valid", "missing"],
            "newey_west_t": [2.5, np.nan],
        }
    )

    adjusted = adjust_multiple_tests(results)

    assert adjusted.loc[0, "hypothesis_family_size"] == 1
    assert np.isnan(adjusted.loc[1, "p_value"])
    assert not adjusted.loc[1, "passes_fdr"]
