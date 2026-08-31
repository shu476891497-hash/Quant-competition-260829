import json

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantcta.research import (
    causal_winsorize,
    evaluate_factor,
    factor_autocorrelation,
    save_factor_results,
)


def test_causal_winsorize_cannot_see_future_mutation() -> None:
    index = pd.date_range("2020-01-01", periods=160, freq="D", tz="UTC")
    factor = pd.DataFrame({"NQ": np.sin(np.arange(160) / 7.0)}, index=index)
    cutoff = index[120]
    mutated = factor.copy()
    mutated.loc[mutated.index > cutoff] *= 1_000

    original_result = causal_winsorize(factor, window=60, min_periods=20)
    mutated_result = causal_winsorize(mutated, window=60, min_periods=20)

    pdt.assert_frame_equal(original_result.loc[:cutoff], mutated_result.loc[:cutoff])


def test_factor_autocorrelation_reports_requested_lags() -> None:
    index = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    factor = pd.DataFrame({"ES": np.arange(30, dtype=float)}, index=index)

    result = factor_autocorrelation(factor, lags=(1, 5))

    assert result["lag"].tolist() == [1, 5]
    assert result["n"].tolist() == [29, 25]
    assert np.allclose(result["autocorrelation"], 1.0)


def test_save_factor_results_writes_metrics_and_manifest(tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D", tz="UTC")
    prices = pd.DataFrame({"NQ": 100 * np.exp(np.arange(80) * 0.001)}, index=index)
    factor = pd.DataFrame({"NQ": np.sin(np.arange(80) / 5.0)}, index=index)
    results = evaluate_factor(
        factor, prices, horizons=(1,), min_samples=20, factor_name="curve"
    )

    output = save_factor_results(
        results,
        tmp_path / "run",
        metadata={"data_source": "synthetic", "contract_rule": "same-contract"},
    )

    restored = pd.read_csv(output / "factor_metrics.csv")
    pdt.assert_frame_equal(restored, results.reset_index(drop=True))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "quantcta-factor-research-v1"
    assert manifest["availability_lags"] == [1]
    assert not manifest["multiple_testing_adjusted"]
    assert manifest["metadata"]["contract_rule"] == "same-contract"
