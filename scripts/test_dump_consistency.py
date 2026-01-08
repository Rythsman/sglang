#!/usr/bin/env python3
"""Compare consistency between two dump directories.

This script is a thin CLI wrapper around `sglang.test.dump_consistency`.
"""

from __future__ import annotations

import argparse
import os
import sys

from sglang.test.dump_consistency import DumpConsistencyTester


def main(argv: list[str]) -> int:
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Compare consistency between two dump directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("dir1", help="First dump directory")
    parser.add_argument("dir2", help="Second dump directory")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for tensor comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--max-tensors-per-file",
        type=int,
        default=64,
        help="Max tensors compared per tensor file (default: 64)",
    )
    parser.add_argument(
        "--no-verify-values",
        action="store_true",
        help="Only verify tensor keys/shape/dtype, skip value comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    if not os.path.isdir(args.dir1):
        print(f"Directory 1 does not exist: {args.dir1}")
        return 1
    if not os.path.isdir(args.dir2):
        print(f"Directory 2 does not exist: {args.dir2}")
        return 1

    tester = DumpConsistencyTester(
        tolerance=args.tolerance,
        max_tensors_per_file=args.max_tensors_per_file,
        verify_tensor_values=not args.no_verify_values,
    )

    try:
        ok = tester.compare_directories(args.dir1, args.dir2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during comparison: {e}")
        return 1

    if ok:
        print("Dump directories are consistent.")
        return 0

    print("Dump directories are NOT consistent.")
    if args.verbose:
        for err in tester.errors[:200]:
            print(f"- {err}")
        if len(tester.errors) > 200:
            print(f"... truncated, total_errors={len(tester.errors)}")
        print(f"stats={tester.stats}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

