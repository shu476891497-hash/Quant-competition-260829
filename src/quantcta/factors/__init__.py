"""Factor functions. New factors do not require changes to the engine."""

from quantcta.factors.curve import (
    annualized_curve_carry,
    annualized_curve_curvature,
    curve_carry_zscore,
    front_open_interest_share,
    next_to_front_open_interest_ratio,
    normalized_curve_curvature,
    open_interest_weighted_maturity,
    volume_minus_open_interest_maturity,
    volume_to_open_interest,
)
from quantcta.factors.positioning import (
    align_cot_publication,
    cot_crowding_zscore,
    cot_net_share,
    open_interest_change,
    price_oi_confirmation,
)
from quantcta.factors.trend import dual_ema_momentum

__all__ = [
    "align_cot_publication",
    "annualized_curve_carry",
    "annualized_curve_curvature",
    "cot_crowding_zscore",
    "cot_net_share",
    "curve_carry_zscore",
    "dual_ema_momentum",
    "front_open_interest_share",
    "next_to_front_open_interest_ratio",
    "normalized_curve_curvature",
    "open_interest_change",
    "open_interest_weighted_maturity",
    "price_oi_confirmation",
    "volume_minus_open_interest_maturity",
    "volume_to_open_interest",
]
