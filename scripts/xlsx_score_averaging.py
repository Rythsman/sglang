#!/usr/bin/env python3
"""
Compute average scores from multiple XLSX files in a folder.

The script scans a directory for .xlsx files and computes per-file averages for
these columns (if present):
  - 正确性_分数
  - 相关性_分数
  - 全面性_分数
  - 流畅度_分数
  - 创造性_分数
  - 整体评估_分数

Output is printed to stdout in a formatted table.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tabulate import tabulate


SCORE_COLUMNS: Tuple[str, ...] = (
    "正确性_分数",
    "相关性_分数",
    "全面性_分数",
    "流畅度_分数",
    "创造性_分数",
    "整体评估_分数",
)


@dataclass(frozen=True)
class FileAverages:
    """Averages computed for a single XLSX file."""

    file_name: str
    averages: Dict[str, Optional[float]]
    counts: Dict[str, int]


@dataclass(frozen=True)
class ObjectIterKey:
    """Parsed object name and iter tag from a filename."""

    object_name: str
    iter_tag: str


def _iter_xlsx_files(input_dir: str) -> List[str]:
    """Return sorted .xlsx file paths (excluding temporary office files)."""
    files: List[str] = []
    for entry in os.listdir(input_dir):
        if not entry.lower().endswith(".xlsx"):
            continue
        # Exclude Excel temp/lock files (e.g. "~$foo.xlsx").
        if entry.startswith("~$"):
            continue
        files.append(os.path.join(input_dir, entry))
    files.sort()
    return files


def _read_xlsx_as_dataframe(
    xlsx_path: str,
    sheet: Optional[str],
    read_all_sheets: bool,
) -> pd.DataFrame:
    """Read an XLSX file into a DataFrame (optionally concatenating sheets)."""
    if read_all_sheets:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
        if not sheets:
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for _, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # Default behavior: read the first sheet (sheet_name=0). If sheet is provided,
    # allow either a sheet name or an integer index (e.g. "0", "1").
    sheet_name: object = 0
    if sheet is not None:
        sheet_str = str(sheet).strip()
        sheet_name = int(sheet_str) if sheet_str.isdigit() else sheet_str
    return pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a column to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def compute_file_averages(
    xlsx_path: str,
    score_columns: Sequence[str],
    sheet: Optional[str],
    read_all_sheets: bool,
    only_evaluated: bool,
    evaluated_column: str,
) -> FileAverages:
    """Compute averages for a single XLSX file."""
    df = _read_xlsx_as_dataframe(
        xlsx_path=xlsx_path,
        sheet=sheet,
        read_all_sheets=read_all_sheets,
    )

    if df.empty:
        return FileAverages(
            file_name=os.path.basename(xlsx_path),
            averages={col: None for col in score_columns},
            counts={col: 0 for col in score_columns},
        )

    if only_evaluated and evaluated_column in df.columns:
        mask = (
            df[evaluated_column]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )
        df = df.loc[mask]

    averages: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}

    for col in score_columns:
        if col not in df.columns:
            averages[col] = None
            counts[col] = 0
            continue
        numeric = _to_numeric_series(df[col])
        valid = numeric.dropna()
        counts[col] = int(valid.shape[0])
        averages[col] = float(valid.mean()) if counts[col] > 0 else None

    return FileAverages(
        file_name=os.path.basename(xlsx_path),
        averages=averages,
        counts=counts,
    )


def _format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_table_rows(results: Iterable[FileAverages], digits: int) -> List[List[str]]:
    rows: List[List[str]] = []
    for r in results:
        row: List[str] = [r.file_name]
        for col in SCORE_COLUMNS:
            row.append(_format_float(r.averages.get(col), digits))
        rows.append(row)
    return rows


def _parse_object_iter_from_filename(file_name: str) -> Optional[ObjectIterKey]:
    """Parse object name and iter tag (iter0/iter1/...) from a file name.

    The iter tag is detected by substring "iter<digits>" (case-insensitive).
    The object name is the remaining part of the filename stem after removing
    the iter token and adjacent separators.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    match = re.search(r"(?i)iter(\d+)", stem)
    if match is None:
        return None

    iter_tag = f"iter{match.group(1)}"
    object_name = re.sub(r"(?i)([_\-.]?iter\d+[_\-.]?)", "_", stem)
    object_name = object_name.strip("_-. ")
    if not object_name:
        object_name = stem
    return ObjectIterKey(object_name=object_name, iter_tag=iter_tag)


def _group_results_by_object_and_iter(
    results: Sequence[FileAverages],
) -> Tuple[Dict[str, Dict[str, FileAverages]], List[str], List[str]]:
    """Group per-file results by object name and iter tag."""
    grouped: Dict[str, Dict[str, FileAverages]] = {}
    objects: List[str] = []
    iters: List[str] = []

    for r in results:
        key = _parse_object_iter_from_filename(r.file_name)
        if key is None:
            continue
        if key.object_name not in grouped:
            grouped[key.object_name] = {}
            objects.append(key.object_name)
        grouped[key.object_name][key.iter_tag] = r
        if key.iter_tag not in iters:
            iters.append(key.iter_tag)

    def _iter_sort_key(tag: str) -> Tuple[int, str]:
        suffix = tag.replace("iter", "", 1)
        return (int(suffix), tag) if suffix.isdigit() else (10**9, tag)

    objects.sort()
    iters.sort(key=_iter_sort_key)
    return grouped, objects, iters


def _build_object_pivot_table(
    grouped: Dict[str, Dict[str, FileAverages]],
    objects: Sequence[str],
    iters: Sequence[str],
    digits: int,
    include_iter_mean: bool,
) -> Tuple[List[str], List[List[str]]]:
    """Build a pivot table (one row per object, columns per iter)."""
    headers: List[str] = ["object"]
    for iter_tag in iters:
        for col in SCORE_COLUMNS:
            headers.append(f"{iter_tag}_{col}")
    if include_iter_mean:
        for col in SCORE_COLUMNS:
            headers.append(f"mean_{col}")

    rows: List[List[str]] = []
    for obj in objects:
        row: List[str] = [obj]
        for iter_tag in iters:
            r = grouped.get(obj, {}).get(iter_tag)
            for col in SCORE_COLUMNS:
                value = r.averages.get(col) if r is not None else None
                row.append(_format_float(value, digits))

        if include_iter_mean:
            for col in SCORE_COLUMNS:
                values: List[float] = []
                for iter_tag in iters:
                    r = grouped.get(obj, {}).get(iter_tag)
                    v = r.averages.get(col) if r is not None else None
                    if v is not None:
                        values.append(float(v))
                mean_value = sum(values) / len(values) if values else None
                row.append(_format_float(mean_value, digits))

        rows.append(row)

    return headers, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average scores from XLSX files in a folder."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Folder path containing .xlsx files",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Decimal digits to display (default: 3)",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Excel sheet name or index to read when not using --all-sheets",
    )
    parser.add_argument(
        "--all-sheets",
        action="store_true",
        help="If set, read and concatenate all sheets (default: off)",
    )
    parser.add_argument(
        "--only-evaluated",
        action="store_true",
        help="If set, only include rows where evaluated column is true-like (default: off)",
    )
    parser.add_argument(
        "--evaluated-column",
        type=str,
        default="_evaluated",
        help="Column name used by --only-evaluated (default: _evaluated)",
    )
    parser.add_argument(
        "--group-by-object",
        action="store_true",
        help="If set, pivot files with iter0/iter1/... in filename into one row per object",
    )
    parser.add_argument(
        "--include-iter-mean",
        action="store_true",
        help="When using --group-by-object, append mean columns across iters",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    xlsx_files = _iter_xlsx_files(input_dir)
    if not xlsx_files:
        raise SystemExit(f"No .xlsx files found under: {input_dir}")

    results: List[FileAverages] = []
    errors: List[Tuple[str, str]] = []

    for xlsx_path in xlsx_files:
        try:
            results.append(
                compute_file_averages(
                    xlsx_path=xlsx_path,
                    score_columns=SCORE_COLUMNS,
                    sheet=args.sheet,
                    read_all_sheets=args.all_sheets,
                    only_evaluated=args.only_evaluated,
                    evaluated_column=args.evaluated_column,
                )
            )
        except ImportError as e:
            # Common case: openpyxl not installed for pandas.read_excel(engine="openpyxl").
            hint = (
                "Missing dependency. Install with: pip install openpyxl\n"
                f"Original error: {e}"
            )
            raise SystemExit(hint) from e
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append((os.path.basename(xlsx_path), str(e)))

    if args.group_by_object:
        grouped, objects, iters = _group_results_by_object_and_iter(results)
        if not grouped:
            raise SystemExit(
                "No files matched iter pattern in filename. Expected '*iter0*.xlsx'."
            )
        headers, rows = _build_object_pivot_table(
            grouped=grouped,
            objects=objects,
            iters=iters,
            digits=args.digits,
            include_iter_mean=args.include_iter_mean,
        )
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))
    else:
        headers = ["file"] + list(SCORE_COLUMNS)
        rows = _build_table_rows(results, digits=args.digits)
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))

    if errors:
        print("\nErrors:")
        print(
            tabulate(errors, headers=["file", "error"], tablefmt="grid", stralign="left")
        )


if __name__ == "__main__":
    main()

