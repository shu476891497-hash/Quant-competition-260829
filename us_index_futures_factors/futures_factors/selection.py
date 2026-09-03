from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


def _correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 20 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    function = spearmanr if method == "spearman" else pearsonr
    return float(function(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def _hac_tvalue(x: pd.Series, y: pd.Series, maxlags: int) -> float:
    valid = pd.concat([x.rename("factor"), y.rename("forward_return")], axis=1).dropna()
    if len(valid) < max(30, maxlags * 4) or valid["factor"].nunique() < 2:
        return np.nan
    design = sm.add_constant(valid[["factor"]])
    model = sm.OLS(valid["forward_return"], design).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(model.tvalues["factor"])


def select_family_candidates(
    panel: pd.DataFrame,
    family: str,
    candidates: Sequence[str],
    is_start: str,
    is_end: str,
    symbols: Sequence[str] = ("ES", "NQ", "YM"),
    horizon: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    start, end = pd.Timestamp(is_start), pd.Timestamp(is_end)
    sample = panel.loc[
        panel["symbol"].isin(symbols) & panel["date"].between(start, end)
    ].copy()
    sample["forward_return"] = sample.groupby("symbol", sort=False)["close"].transform(
        lambda values: np.log(values.shift(-horizon) / values)
    )
    detail_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for order, candidate in enumerate(candidates):
        per_symbol: list[dict[str, object]] = []
        for symbol in symbols:
            group = sample.loc[sample["symbol"] == symbol]
            spearman = _correlation(group[candidate], group["forward_return"], "spearman")
            pearson = _correlation(group[candidate], group["forward_return"], "pearson")
            hac_t = _hac_tvalue(group[candidate], group["forward_return"], horizon)
            coverage = float(group[[candidate, "forward_return"]].dropna().shape[0] / max(1, len(group)))
            turnover = float(group[candidate].diff().abs().mean())
            row = {
                "family": family,
                "candidate": candidate,
                "symbol": symbol,
                "spearman_ic_raw": spearman,
                "pearson_ic_raw": pearson,
                "hac_t_raw": hac_t,
                "coverage": coverage,
                "factor_turnover": turnover,
            }
            per_symbol.append(row)
            detail_rows.append(row)
        ic_values = pd.Series([row["spearman_ic_raw"] for row in per_symbol], dtype=float)
        raw_median_ic = float(ic_values.median()) if ic_values.notna().any() else np.nan
        orientation = 1.0 if pd.isna(raw_median_ic) or raw_median_ic >= 0 else -1.0
        oriented_t = pd.Series([row["hac_t_raw"] for row in per_symbol], dtype=float) * orientation
        oriented_ic = ic_values * orientation
        consistency = int((oriented_ic > 0).sum())
        summaries.append(
            {
                "family": family,
                "candidate": candidate,
                "symbol": "SUMMARY",
                "orientation": int(orientation),
                "median_spearman_ic": float(oriented_ic.median()) if oriented_ic.notna().any() else np.nan,
                "median_hac_t": float(oriented_t.median()) if oriented_t.notna().any() else np.nan,
                "consistent_symbols": consistency,
                "coverage": float(pd.Series([row["coverage"] for row in per_symbol]).median()),
                "factor_turnover": float(pd.Series([row["factor_turnover"] for row in per_symbol]).median()),
                "candidate_order": order,
            }
        )
    summary_frame = pd.DataFrame(summaries)
    eligible = summary_frame.loc[summary_frame["consistent_symbols"] >= 2].copy()
    if eligible.empty:
        eligible = summary_frame.copy()
    eligible["rank_t"] = eligible["median_hac_t"].fillna(-np.inf)
    eligible = eligible.sort_values(
        ["rank_t", "coverage", "factor_turnover", "candidate_order"],
        ascending=[False, False, True, True],
    )
    winner = eligible.iloc[0]
    chosen = {
        "family": family,
        "candidate": str(winner["candidate"]),
        "orientation": int(winner["orientation"]),
        "selection_period": f"{is_start}/{is_end}",
        "selection_symbols": list(symbols),
        "horizon": horizon,
    }
    evidence = pd.concat([pd.DataFrame(detail_rows), summary_frame], ignore_index=True, sort=False)
    evidence["selected"] = evidence["candidate"].eq(chosen["candidate"])
    return evidence, chosen
