from __future__ import annotations

import argparse

import pandas as pd

from futures_factors.config import default_paths
from futures_factors.data import download_all
from futures_factors.low_turnover import evaluate_low_turnover
from futures_factors.pipeline import build_factors, evaluate


def _add_download_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", default="2006-01-01", help="Download start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Last allowed completed session (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true", help="Redownload existing raw cache")
    parser.add_argument("--cboe-workers", type=int, default=8, help="Parallel Cboe page downloads")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="US index futures three-factor research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_options(subparsers.add_parser("download", help="Download and clean source data"))
    build = subparsers.add_parser("build-factors", help="Build candidates and freeze one factor per family")
    build.add_argument("--reselect", action="store_true", help="Explicitly replace frozen IS selection")
    subparsers.add_parser("evaluate", help="Evaluate frozen factors and generate report")
    low_turnover = subparsers.add_parser(
        "evaluate-low-turnover",
        help="Run the independent low-turnover overlay and OOS target audit",
    )
    low_turnover.add_argument("--cost-bps", type=float, default=1.0)
    all_parser = subparsers.add_parser("all", help="Run download, factor build, and evaluation")
    _add_download_options(all_parser)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    paths = default_paths()
    if args.command in {"download", "all"}:
        data = download_all(
            paths,
            start=args.start,
            end=pd.Timestamp(args.end) if args.end else None,
            refresh=args.refresh,
            cboe_workers=max(1, args.cboe_workers),
        )
        print(data["quality"].to_string(index=False))
    if args.command in {"build-factors", "all"}:
        final, selections = build_factors(paths, reselect=getattr(args, "reselect", False))
        print(f"Built {len(final):,} panel rows; frozen factors: {selections}")
    if args.command in {"evaluate", "all"}:
        results = evaluate(paths)
        oos = results["metrics"].loc[
            (results["metrics"]["period"] == "OOS")
            & (results["metrics"]["cost_bps"] == 1.0)
            & (results["metrics"]["kind"] == "strategy")
        ]
        print(oos[["symbol", "annual_return", "sharpe", "max_drawdown"]].to_string(index=False))
    if args.command == "evaluate-low-turnover":
        results = evaluate_low_turnover(paths, cost_bps=args.cost_bps)
        oos = results["metrics"].loc[results["metrics"]["period"] == "OOS"]
        print(
            oos[
                [
                    "return_basis",
                    "annual_return",
                    "sharpe",
                    "max_drawdown",
                    "annual_turnover",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
