import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantcta.factors import (
    align_cot_publication,
    annualized_curve_carry,
    annualized_curve_curvature,
    cot_crowding_zscore,
    curve_carry_zscore,
    front_open_interest_share,
    next_to_front_open_interest_ratio,
    normalized_curve_curvature,
    open_interest_weighted_maturity,
    price_oi_confirmation,
    volume_minus_open_interest_maturity,
    volume_to_open_interest,
)


def test_curve_carry_is_positive_in_backwardation() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    front = pd.DataFrame({"ES": [102.0, 99.0]}, index=index)
    back = pd.DataFrame({"ES": [100.0, 100.0]}, index=index)
    front_expiry = pd.DataFrame(
        {"ES": pd.to_datetime(["2025-03-21", "2025-03-21"], utc=True)}, index=index
    )
    back_expiry = pd.DataFrame(
        {"ES": pd.to_datetime(["2025-06-20", "2025-06-20"], utc=True)}, index=index
    )

    carry = annualized_curve_carry(front, back, front_expiry, back_expiry)

    assert carry.iloc[0, 0] > 0
    assert carry.iloc[1, 0] < 0


def test_curve_factor_cannot_see_future_mutation() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    front = pd.DataFrame({"ES": np.linspace(100.0, 110.0, len(index))}, index=index)
    back = front * 1.01
    front_expiry = pd.DataFrame(
        {"ES": index + pd.Timedelta(days=45)}, index=index
    )
    back_expiry = pd.DataFrame(
        {"ES": index + pd.Timedelta(days=135)}, index=index
    )
    cutoff = index[90]
    mutated_back = back.copy()
    mutated_back.loc[mutated_back.index > cutoff] *= 1.5

    original = curve_carry_zscore(
        front, back, front_expiry, back_expiry, window=40, min_periods=20
    )
    mutated = curve_carry_zscore(
        front, mutated_back, front_expiry, back_expiry, window=40, min_periods=20
    )

    pdt.assert_frame_equal(original.loc[:cutoff], mutated.loc[:cutoff])


def test_curve_curvature_normalizes_both_expiry_gaps() -> None:
    index = pd.date_range("2025-01-01", periods=1, tz="UTC")
    front = pd.DataFrame({"NQ": [100.0]}, index=index)
    middle = pd.DataFrame({"NQ": [99.0]}, index=index)
    back = pd.DataFrame({"NQ": [98.5]}, index=index)
    front_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=30)}, index=index)
    middle_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=90)}, index=index)
    back_expiry = pd.DataFrame({"NQ": index + pd.Timedelta(days=180)}, index=index)

    curvature = annualized_curve_curvature(
        front, middle, back, front_expiry, middle_expiry, back_expiry
    )
    expected = np.log(100.0 / 99.0) * 365 / 60 - np.log(99.0 / 98.5) * 365 / 90

    assert np.isclose(curvature.iloc[0, 0], expected)


def test_volume_to_open_interest() -> None:
    index = pd.date_range("2025-01-01", periods=2, tz="UTC")
    volume = pd.DataFrame({"ES": [200.0, 300.0]}, index=index)
    oi = pd.DataFrame({"ES": [1_000.0, 1_200.0]}, index=index)
    result = volume_to_open_interest(volume, oi)
    assert result["ES"].tolist() == [0.2, 0.25]


def test_normalized_curvature_is_dimensionless() -> None:
    index = pd.date_range("2025-01-01", periods=1, tz="UTC")
    front = pd.DataFrame({"NQ": [100.0]}, index=index)
    middle = pd.DataFrame({"NQ": [99.0]}, index=index)
    back = pd.DataFrame({"NQ": [98.0]}, index=index)
    expiries = [
        pd.DataFrame({"NQ": index + pd.Timedelta(days=days)}, index=index)
        for days in (30, 90, 180)
    ]
    result = normalized_curve_curvature(
        front, middle, back, expiries[0], expiries[1], expiries[2]
    )
    assert -1 <= result.iloc[0, 0] <= 1


def test_curve_position_maturity_factors() -> None:
    index = pd.date_range("2025-01-01", periods=1, tz="UTC")
    dte = pd.DataFrame([[30.0, 90.0, 180.0]], index=index, columns=["F1", "F2", "F3"])
    oi = pd.DataFrame([[60.0, 30.0, 10.0]], index=index, columns=dte.columns)
    volume = pd.DataFrame([[10.0, 30.0, 60.0]], index=index, columns=dte.columns)

    maturity = open_interest_weighted_maturity(dte, oi)
    front_share = front_open_interest_share(oi)
    ratio = next_to_front_open_interest_ratio(oi)
    flow_gap = volume_minus_open_interest_maturity(dte, volume, oi)

    assert np.isclose(maturity.iloc[0], 63.0)
    assert np.isclose(front_share.iloc[0], 0.6)
    assert np.isclose(ratio.iloc[0], np.log(0.5))
    assert flow_gap.iloc[0] > 0


def test_price_oi_confirmation_direction() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    close = pd.DataFrame(
        {
            "UP": np.linspace(100.0, 130.0, len(index)),
            "DOWN": np.linspace(130.0, 100.0, len(index)),
        },
        index=index,
    )
    variation = np.sin(np.arange(len(index)) / 4.0) + np.arange(len(index)) * 0.08
    oi = pd.DataFrame(
        {"UP": 10_000.0 + variation * 100.0, "DOWN": 12_000.0 + variation * 100.0},
        index=index,
    )

    factor = price_oi_confirmation(
        close,
        oi,
        price_lookback=5,
        oi_lookback=2,
        scale_window=30,
        min_periods=15,
    ).dropna()

    rising_oi = oi["UP"].pct_change(2).loc[factor.index] > 0
    assert (factor.loc[rising_oi, "UP"] > 0).all()
    assert (factor.loc[rising_oi, "DOWN"] < 0).all()


def test_cot_factor_cannot_see_future_mutation() -> None:
    index = pd.date_range("2020-01-03", periods=100, freq="W-FRI", tz="UTC")
    long_positions = pd.DataFrame(
        {"ES": 100_000.0 + np.arange(len(index)) * 100.0}, index=index
    )
    short_positions = pd.DataFrame(
        {"ES": 80_000.0 + np.sin(np.arange(len(index)) / 5.0) * 2_000.0}, index=index
    )
    total_oi = pd.DataFrame({"ES": 1_000_000.0}, index=index)
    cutoff = index[75]
    mutated_long = long_positions.copy()
    mutated_long.loc[mutated_long.index > cutoff] *= 5.0

    original = cot_crowding_zscore(
        long_positions, short_positions, total_oi, window=40, min_periods=20
    )
    mutated = cot_crowding_zscore(
        mutated_long, short_positions, total_oi, window=40, min_periods=20
    )

    pdt.assert_frame_equal(original.loc[:cutoff], mutated.loc[:cutoff])


def test_cot_report_is_shifted_to_publication_date() -> None:
    report_dates = pd.DatetimeIndex(["2025-01-07", "2025-01-14"], tz="UTC")
    reports = pd.DataFrame({"ES": [0.1, 0.2]}, index=report_dates)
    published = align_cot_publication(reports)
    assert published.index.weekday.tolist() == [4, 4]
    assert published.iloc[:, 0].tolist() == [0.1, 0.2]
