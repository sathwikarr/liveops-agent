"""CLI entry point — runs the eval corpus and exits with a non-zero status if
the pass rate falls below `--min-pass-rate` (or below the baseline).

Examples:

    python -m analyst.evals
    python -m analyst.evals --backend llm
    python -m analyst.evals --tag retention --tag products
    python -m analyst.evals --json eval_results.json
    python -m analyst.evals --min-pass-rate 0.95
    python -m analyst.evals --baseline tests/fixtures/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from analyst.evals import (
    ALL_CASES, EvalReport, load_cases, load_holdout_cases, run_all,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "analyst" / "sample_data" / "retail_orders.csv"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analyst.evals",
        description="Run the LLM agent eval harness against a sample dataset.",
    )
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help=f"CSV/Parquet to run evals against (default: {DEFAULT_DATA.name}).",
    )
    p.add_argument(
        "--backend", choices=["heuristic", "llm", "auto"], default="heuristic",
        help="Agent planning backend. CI should use 'heuristic' for determinism.",
    )
    p.add_argument(
        "--tag", action="append", default=None,
        help="Filter by tag (repeatable). OR semantics across tags.",
    )
    p.add_argument(
        "--id", action="append", default=None,
        help="Filter by case id (repeatable).",
    )
    p.add_argument(
        "--json", type=Path, default=None,
        help="Write the full report to this file as JSON.",
    )
    p.add_argument(
        "--min-pass-rate", type=float, default=0.0,
        help="Exit non-zero if pass rate falls below this threshold.",
    )
    p.add_argument(
        "--baseline", type=Path, default=None,
        help=("Compare to a previous run's JSON. Exit non-zero if pass_rate "
              "drops by more than 0.02."),
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress per-case logs.",
    )
    p.add_argument(
        "--holdout", action="store_true",
        help=("Use the held-out corpus (paraphrases the heuristic was NOT "
              "tuned on). The honest generalization signal."),
    )
    return p


def _print_report(report: EvalReport, *, quiet: bool = False) -> None:
    print(f"\n=== eval report (backend={report.backend}) ===")
    print(f"pass_rate     : {report.pass_rate * 100:.1f}%  "
          f"({report.n_passed}/{report.n_cases})")
    print(f"mean_overall  : {report.mean_overall:.3f}")
    print(f"  tool_match  : {report.mean_tool_match:.3f}")
    print(f"  args_match  : {report.mean_args_match:.3f}")
    print(f"  no_forbidden: {report.mean_no_forbidden:.3f}")
    print(f"  success_match:{report.mean_success_match:.3f}")

    if report.per_tool:
        print("\nper-tool:")
        for tool, stats in sorted(report.per_tool.items(),
                                   key=lambda kv: kv[1]["pass_rate"]):
            print(f"  {tool:<22s}  {stats['pass_rate']*100:5.1f}%  "
                  f"(n={int(stats['n'])}, mean={stats['mean_overall']:.3f})")

    if not quiet:
        failures = [c for c in report.cases if not c.passed]
        if failures:
            print(f"\nfailures ({len(failures)}):")
            for r in failures:
                print(f"  ✗ {r.case_id}  {r.question[:70]}")
                for reason in r.failed_reasons:
                    print(f"      • {reason}")


def _maybe_compare_baseline(report: EvalReport, baseline_path: Path) -> int:
    if not baseline_path.exists():
        print(f"\n(no baseline at {baseline_path} — skipping comparison)")
        return 0
    baseline = json.loads(baseline_path.read_text())
    base_pass = float(baseline.get("pass_rate", 0.0))
    delta = report.pass_rate - base_pass
    print(f"\nbaseline pass_rate : {base_pass*100:.1f}%")
    print(f"current  pass_rate : {report.pass_rate*100:.1f}%")
    print(f"delta              : {delta*100:+.1f}pp")
    if delta < -0.02:   # >2 percentage point regression
        print("✗ REGRESSION — pass rate dropped by more than 2pp")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.data.exists():
        print(f"error: data file not found: {args.data}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.data) if args.data.suffix == ".csv" else pd.read_parquet(args.data)
    print(f"loaded {len(df):,} rows from {args.data.name}")

    if args.holdout:
        cases = load_holdout_cases(tags=args.tag, ids=args.id)
        corpus_label = "held-out"
    else:
        cases = load_cases(tags=args.tag, ids=args.id)
        corpus_label = "main"
    if not cases:
        print("error: no cases matched the filters", file=sys.stderr)
        return 1
    print(f"running {len(cases)} {corpus_label} cases on backend={args.backend}")

    report = run_all(cases, df, backend=args.backend)
    _print_report(report, quiet=args.quiet)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), default=str, indent=2))
        print(f"\nwrote {args.json}")

    exit_code = 0
    if args.baseline:
        exit_code = max(exit_code, _maybe_compare_baseline(report, args.baseline))

    if report.pass_rate < args.min_pass_rate:
        print(f"\n✗ pass rate {report.pass_rate*100:.1f}% < min "
              f"{args.min_pass_rate*100:.1f}%")
        exit_code = max(exit_code, 1)
    elif args.min_pass_rate > 0:
        print(f"\n✓ pass rate {report.pass_rate*100:.1f}% ≥ min "
              f"{args.min_pass_rate*100:.1f}%")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
