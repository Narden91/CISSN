#!/usr/bin/env python
"""Verify on-disk dataset CSVs against their registered integrity fingerprints.

Usage
-----
    uv run python scripts/verify_datasets.py
    uv run python scripts/verify_datasets.py --datasets ETTh1,ETTh2

Exits non-zero if any dataset fails a structural check (wrong row count,
wrong date range, pre-standardized values, or noise substitution).
"""

import argparse
import sys
from pathlib import Path

from cissn.data.registry import supported_datasets, verify_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(supported_datasets()),
        help="Comma-separated dataset names to verify. Default: all registered datasets.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override the dataset root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"{'Dataset':<16} {'Status':<8} Detail")
    print(f"{'-'*16} {'-'*8} {'-'*50}")

    any_failed = False
    for name in names:
        report = verify_dataset(name, data_root=args.data_root, strict=False)
        status = "OK" if report["passed"] else "FAIL"
        any_failed = any_failed or not report["passed"]
        detail = "; ".join(report["failures"]) or "; ".join(report["warnings"]) or "-"
        print(f"{name:<16} {status:<8} {detail}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
