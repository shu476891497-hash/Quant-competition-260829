from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAMILIES, IS_SYMBOLS, PERIODS, ProjectPaths
from .data import data_quality_report, load_processed
from .evaluation import backtest_symbol, extreme_state_diagnostics, factor_diagnostics, performance_metrics
from .factors import build_candidate_panel
from .reporting import create_figures, write_chinese_report
from .selection import select_family_candidates
from .utils import sha256_file, write_json


FINAL_NAMES = {
    "put_call": "put_call_factor",
    "liquidity": "liquidity_factor",
    "positioning": "positioning_factor",
}


def _coverage(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol"):
        for column in columns:
            valid = group.loc[group[column].notna(), "date"]
            rows.append(
                {
                    "symbol": symbol,
                    "field": column,
                    "start": valid.min() if not valid.empty else pd.NaT,
                    "end": valid.max() if not valid.empty else pd.NaT,
                    "observations": int(valid.size),
                    "coverage": float(valid.size / max(1, len(group))),
                }
            )
    return pd.DataFrame(rows)


def build_factors(paths: ProjectPaths, reselect: bool = False) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    paths.ensure()
    data = load_processed(paths)
    candidates = build_candidate_panel(data["prices"], data["pcr"], data["fred"], data["cftc"])
    candidates.to_parquet(paths.processed / "candidate_factors.parquet", index=False)
    selection_path = paths.outputs / "frozen_selection.json"
    evidence_frames: list[pd.DataFrame] = []
    if selection_path.exists() and not reselect:
        selections = json.loads(selection_path.read_text(encoding="utf-8"))
        if len(selections) != 3:
            raise RuntimeError("Frozen selection must contain exactly three factors")
    else:
        selections = []
        for family, family_candidates in FAMILIES.items():
            evidence, chosen = select_family_candidates(
                candidates,
                family=family,
                candidates=family_candidates,
                is_start=PERIODS["IS"][0],
                is_end=PERIODS["IS"][1],
                symbols=IS_SYMBOLS,
                horizon=5,
            )
            evidence_frames.append(evidence)
            selections.append(chosen)
        write_json(selection_path, selections)
        pd.concat(evidence_frames, ignore_index=True).to_csv(paths.outputs / "candidate_selection_evidence.csv", index=False)

    final = candidates[["date", "symbol", "close", "log_return"]].copy()
    for selected in selections:
        family = str(selected["family"])
        candidate = str(selected["candidate"])
        if candidate not in candidates:
            raise KeyError(f"Frozen candidate {candidate} is absent from candidate panel")
        final[FINAL_NAMES[family]] = candidates[candidate] * int(selected["orientation"])
    factor_columns = list(FINAL_NAMES.values())
    final["composite"] = final[factor_columns].mean(axis=1, skipna=False).clip(-1.0, 1.0)
    final.to_parquet(paths.processed / "final_factor_panel.parquet", index=False)
    final.to_csv(paths.outputs / "final_factor_panel.csv", index=False)
    coverage = _coverage(final, factor_columns + ["composite"])
    coverage.to_csv(paths.outputs / "factor_coverage.csv", index=False)
    hashes = {
        "candidate_factors_sha256": sha256_file(paths.processed / "candidate_factors.parquet"),
        "final_factor_panel_sha256": sha256_file(paths.processed / "final_factor_panel.parquet"),
        "selection_sha256": sha256_file(selection_path),
    }
    write_json(paths.outputs / "factor_build_hashes.json", hashes)
    return final, selections


def _run_backtests(final: pd.DataFrame, costs: tuple[float, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cost in costs:
        symbol_frames: list[pd.DataFrame] = []
        for symbol, group in final.groupby("symbol", sort=True):
            result = backtest_symbol(group, cost_bps=cost)
            result["cost_bps"] = cost
            symbol_frames.append(result)
        individual = pd.concat(symbol_frames, ignore_index=True)
        frames.append(individual)
        portfolio = individual.groupby("date", as_index=False).agg(
            net_log_return=("net_log_return", "mean"),
            gross_log_return=("gross_log_return", "mean"),
            benchmark_log_return=("benchmark_log_return", "mean"),
            turnover=("turnover", "mean"),
        )
        portfolio["symbol"] = "PORTFOLIO"
        portfolio["cost_bps"] = cost
        frames.append(portfolio)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["cost_bps", "symbol", "date"])


def _metric_table(backtests: pd.DataFrame, cash_benchmarks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cost, symbol), group in backtests.groupby(["cost_bps", "symbol"]):
        for period, (start, end) in PERIODS.items():
            sample = group.loc[group["date"].between(start, end)]
            strategy = performance_metrics(sample["net_log_return"], sample["turnover"])
            rows.append({"period": period, "symbol": symbol, "kind": "strategy", "cost_bps": cost, **strategy})
            if cost == 0.0:
                benchmark = performance_metrics(sample["benchmark_log_return"])
                rows.append({"period": period, "symbol": symbol, "kind": "benchmark", "cost_bps": 1.0, **benchmark})
    es_active = backtests.loc[
        (backtests["symbol"] == "ES") & (backtests["cost_bps"] == 1.0) & backtests["net_log_return"].notna(),
        ["date"],
    ].drop_duplicates()
    for symbol, group in cash_benchmarks.groupby("symbol"):
        matched = es_active.merge(group[["date", "log_return"]], on="date", how="inner")
        for period, (start, end) in PERIODS.items():
            sample = matched.loc[matched["date"].between(start, end)]
            benchmark = performance_metrics(sample["log_return"])
            rows.append({"period": period, "symbol": symbol, "kind": "cash_benchmark", "cost_bps": 1.0, **benchmark})
    return pd.DataFrame(rows)


def evaluate(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    paths.ensure()
    final_path = paths.processed / "final_factor_panel.parquet"
    if not final_path.exists():
        raise FileNotFoundError("Run build-factors before evaluate")
    final = pd.read_parquet(final_path)
    selections = json.loads((paths.outputs / "frozen_selection.json").read_text(encoding="utf-8"))
    if len(selections) != 3:
        raise RuntimeError("Evaluation requires exactly three frozen factors")
    factor_columns = list(FINAL_NAMES.values())
    ic, quintiles, yearly = factor_diagnostics(final, factor_columns, PERIODS)
    extremes = extreme_state_diagnostics(final, factor_columns, PERIODS)
    ic.to_csv(paths.outputs / "factor_ic.csv", index=False)
    quintiles.to_csv(paths.outputs / "factor_quintiles.csv", index=False)
    yearly.to_csv(paths.outputs / "factor_yearly_stability.csv", index=False)
    extremes.to_csv(paths.outputs / "factor_extreme_states.csv", index=False)

    backtests = _run_backtests(final, costs=(0.0, 1.0, 2.0, 5.0))
    backtests.to_parquet(paths.outputs / "backtest_daily.parquet", index=False)
    backtests.to_csv(paths.outputs / "backtest_daily.csv", index=False)
    processed = load_processed(paths)
    metrics = _metric_table(backtests, processed["cash_benchmarks"])
    metrics.to_csv(paths.outputs / "performance_metrics.csv", index=False)
    create_figures(backtests, final, paths.figures)

    quality_path = paths.outputs / "data_quality.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else data_quality_report(
        processed["prices"], processed["pcr"], processed["fred"], processed["cftc"], processed["cash_benchmarks"]
    )
    coverage = pd.read_csv(paths.outputs / "factor_coverage.csv")
    latest_date = pd.Timestamp(final["date"].max())
    write_chinese_report(paths.root / "REPORT_CN.md", selections, metrics, ic, quality, coverage, latest_date)
    write_json(
        paths.outputs / "run_metadata.json",
        {
            "label": "PSEUDO_OOS_CURRENT_VINTAGE",
            "latest_complete_session": latest_date,
            "is_period": PERIODS["IS"],
            "validation_period": PERIODS["VALIDATION"],
            "oos_period": (PERIODS["OOS"][0], latest_date),
            "selected_factors": selections,
        },
    )
    deliverables = [
        path
        for path in paths.root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and "data" not in path.relative_to(paths.root).parts
        and path.name != "delivery_manifest.json"
    ]
    write_json(
        paths.outputs / "delivery_manifest.json",
        [
            {
                "path": str(path.relative_to(paths.root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(deliverables)
        ],
    )
    return {"metrics": metrics, "ic": ic, "quintiles": quintiles, "extremes": extremes, "yearly": yearly, "backtests": backtests}
