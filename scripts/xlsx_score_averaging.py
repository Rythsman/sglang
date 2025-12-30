#!/usr/bin/env python3
"""
Compute average scores from multiple XLSX files in a folder.

The tool supports:
  - Per-file averaging for the configured score columns.
  - Group-by-object mode: files containing iter tags (iter0/iter1/...) are
    grouped into one row per object, with list-like values such as [v0, v1, v2].

Output is printed to stdout in a formatted table.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

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


DEFAULT_MODEL_TYPES: Tuple[str, ...] = ("sglang", "vllm")
DEFAULT_WEIGHTS_TYPES: Tuple[str, ...] = ("opensource", "rl", "sft")


@dataclass(frozen=True)
class FileAverages:
    """Averages computed for a single XLSX file."""

    file_name: str
    averages: Dict[str, Optional[float]]


@dataclass(frozen=True)
class ObjectIterKey:
    """A parsed grouping key from filename."""

    object_name: str
    iter_tag: str


class WorkbookLoader(Protocol):
    """Loads an XLSX workbook into a DataFrame."""

    def load(self, xlsx_path: str, sheet: Optional[str], all_sheets: bool) -> pd.DataFrame:
        """Loads the workbook and returns a DataFrame."""


class ObjectIterParser(Protocol):
    """Parses (object_name, iter_tag) from a filename."""

    def parse(self, file_name: str) -> Optional[ObjectIterKey]:
        """Returns parsed key, or None if not match."""


class PandasOpenpyxlWorkbookLoader:
    """Workbook loader based on pandas + openpyxl engine."""

    def load(self, xlsx_path: str, sheet: Optional[str], all_sheets: bool) -> pd.DataFrame:
        if all_sheets:
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

        sheet_name: object = 0
        if sheet is not None:
            sheet_str = str(sheet).strip()
            sheet_name = int(sheet_str) if sheet_str.isdigit() else sheet_str
        return pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")


class RegexFilenameObjectIterParser:
    """Parse object name and iter tag from filename.

    Behavior:
      - Strips a trailing timestamp suffix like "_20251230_194805" (or '-' / '.').
      - Detects iter by substring "iter<digits>".
      - Derives object name by removing iter token from the stem.
      - Optionally normalizes object name into "<model>_<weights>" when both are present.
    """

    def __init__(
        self,
        model_types: Sequence[str] = DEFAULT_MODEL_TYPES,
        weights_types: Sequence[str] = DEFAULT_WEIGHTS_TYPES,
    ):
        self._model_types = tuple(m.lower() for m in model_types)
        self._weights_types = tuple(w.lower() for w in weights_types)

    def parse(self, file_name: str) -> Optional[ObjectIterKey]:
        stem = os.path.splitext(os.path.basename(file_name))[0]
        stem = re.sub(r"([_\-.])\d{8}([_\-.])\d{6}$", "", stem)

        match = re.search(r"(?i)iter(\d+)", stem)
        if match is None:
            return None

        iter_tag = f"iter{match.group(1)}"
        object_name = re.sub(r"(?i)([_\-.]?iter\d+[_\-.]?)", "_", stem).strip("_-. ")
        if not object_name:
            object_name = stem

        normalized = self._normalize_object_name(object_name)
        return ObjectIterKey(object_name=normalized, iter_tag=iter_tag)

    def _normalize_object_name(self, object_name: str) -> str:
        lowered = object_name.lower()
        model = next((m for m in self._model_types if m in lowered), None)
        weights = next((w for w in self._weights_types if w in lowered), None)
        if model is not None and weights is not None:
            return f"{model}_{weights}"
        return object_name


class ScoreAverager:
    """Computes average score columns from a DataFrame."""

    def __init__(self, score_columns: Sequence[str]):
        self._score_columns = tuple(score_columns)

    def compute(
        self,
        df: pd.DataFrame,
        only_evaluated: bool,
        evaluated_column: str,
    ) -> Dict[str, Optional[float]]:
        if df.empty:
            return {col: None for col in self._score_columns}

        if only_evaluated and evaluated_column in df.columns:
            mask = (
                df[evaluated_column].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            )
            df = df.loc[mask]

        averages: Dict[str, Optional[float]] = {}
        for col in self._score_columns:
            if col not in df.columns:
                averages[col] = None
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid = numeric.dropna()
            averages[col] = float(valid.mean()) if not valid.empty else None
        return averages


class XlsxScoreService:
    """Orchestrates reading, averaging and optional grouping."""

    def __init__(
        self,
        loader: WorkbookLoader,
        parser: ObjectIterParser,
        averager: ScoreAverager,
    ):
        self._loader = loader
        self._parser = parser
        self._averager = averager

    def list_xlsx_files(self, input_dir: str) -> List[str]:
        files: List[str] = []
        for entry in os.listdir(input_dir):
            if not entry.lower().endswith(".xlsx"):
                continue
            if entry.startswith("~$"):
                continue
            files.append(os.path.join(input_dir, entry))
        files.sort()
        return files

    def compute_file_averages(
        self,
        xlsx_path: str,
        sheet: Optional[str],
        all_sheets: bool,
        only_evaluated: bool,
        evaluated_column: str,
    ) -> FileAverages:
        df = self._loader.load(xlsx_path=xlsx_path, sheet=sheet, all_sheets=all_sheets)
        averages = self._averager.compute(
            df=df,
            only_evaluated=only_evaluated,
            evaluated_column=evaluated_column,
        )
        return FileAverages(file_name=os.path.basename(xlsx_path), averages=averages)

    def group_by_object_and_iter(
        self, results: Sequence[FileAverages]
    ) -> Tuple[Dict[str, Dict[str, FileAverages]], List[str], List[str]]:
        grouped: Dict[str, Dict[str, FileAverages]] = {}
        objects: List[str] = []
        iters: List[str] = []

        for r in results:
            key = self._parser.parse(r.file_name)
            if key is None:
                continue
            if key.object_name not in grouped:
                grouped[key.object_name] = {}
                objects.append(key.object_name)
            grouped[key.object_name][key.iter_tag] = r
            if key.iter_tag not in iters:
                iters.append(key.iter_tag)

        objects.sort()
        iters.sort(key=_iter_sort_key)
        return grouped, objects, iters


def _iter_sort_key(tag: str) -> Tuple[int, str]:
    suffix = tag.replace("iter", "", 1)
    return (int(suffix), tag) if suffix.isdigit() else (10**9, tag)


def _format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_list(values: Sequence[Optional[float]], digits: int) -> str:
    inner = ", ".join(_format_float(v, digits) for v in values)
    return f"[{inner}]"


def _build_per_file_table(results: Iterable[FileAverages], digits: int) -> Tuple[List[str], List[List[str]]]:
    headers = ["file"] + list(SCORE_COLUMNS)
    rows: List[List[str]] = []
    for r in results:
        row = [r.file_name] + [_format_float(r.averages.get(col), digits) for col in SCORE_COLUMNS]
        rows.append(row)
    return headers, rows


def _build_grouped_table(
    grouped: Dict[str, Dict[str, FileAverages]],
    objects: Sequence[str],
    iters: Sequence[str],
    digits: int,
    include_iter_mean: bool,
) -> Tuple[List[str], List[List[str]]]:
    headers: List[str] = ["object"] + list(SCORE_COLUMNS)
    if include_iter_mean:
        headers.extend([f"mean_{col}" for col in SCORE_COLUMNS])

    rows: List[List[str]] = []
    for obj in objects:
        row: List[str] = [obj]
        for col in SCORE_COLUMNS:
            per_iter: List[Optional[float]] = []
            for iter_tag in iters:
                r = grouped.get(obj, {}).get(iter_tag)
                per_iter.append(r.averages.get(col) if r is not None else None)
            row.append(_format_list(per_iter, digits))

        if include_iter_mean:
            for col in SCORE_COLUMNS:
                values: List[float] = []
                for iter_tag in iters:
                    r = grouped.get(obj, {}).get(iter_tag)
                    v = r.averages.get(col) if r is not None else None
                    if v is not None:
                        values.append(float(v))
                mean_value = (sum(values) / len(values)) if values else None
                row.append(_format_float(mean_value, digits))

        rows.append(row)
    return headers, rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average scores from XLSX files.")
    parser.add_argument("input_dir", type=str, help="Folder path containing .xlsx files")
    parser.add_argument("--digits", type=int, default=3, help="Decimal digits to display (default: 3)")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name or index")
    parser.add_argument("--all-sheets", action="store_true", help="Read and concatenate all sheets")
    parser.add_argument(
        "--only-evaluated",
        action="store_true",
        help="Only include rows where evaluated column is true-like",
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
        help="Group files with iter0/iter1/... in filename into one row per object",
    )
    parser.add_argument(
        "--include-iter-mean",
        action="store_true",
        help="When using --group-by-object, append mean columns across iters",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    service = XlsxScoreService(
        loader=PandasOpenpyxlWorkbookLoader(),
        parser=RegexFilenameObjectIterParser(),
        averager=ScoreAverager(score_columns=SCORE_COLUMNS),
    )

    xlsx_files = service.list_xlsx_files(input_dir)
    if not xlsx_files:
        raise SystemExit(f"No .xlsx files found under: {input_dir}")

    results: List[FileAverages] = []
    errors: List[Tuple[str, str]] = []

    for xlsx_path in xlsx_files:
        try:
            results.append(
                service.compute_file_averages(
                    xlsx_path=xlsx_path,
                    sheet=args.sheet,
                    all_sheets=args.all_sheets,
                    only_evaluated=args.only_evaluated,
                    evaluated_column=args.evaluated_column,
                )
            )
        except ImportError as e:
            hint = (
                "Missing dependency. Install with: pip install openpyxl\n"
                f"Original error: {e}"
            )
            raise SystemExit(hint) from e
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append((os.path.basename(xlsx_path), str(e)))

    if args.group_by_object:
        grouped, objects, iters = service.group_by_object_and_iter(results)
        if not grouped:
            raise SystemExit("No files matched iter pattern in filename. Expected '*iter0*.xlsx'.")
        headers, rows = _build_grouped_table(
            grouped=grouped,
            objects=objects,
            iters=iters,
            digits=args.digits,
            include_iter_mean=args.include_iter_mean,
        )
    else:
        headers, rows = _build_per_file_table(results=results, digits=args.digits)

    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))

    if errors:
        print("\nErrors:")
        print(tabulate(errors, headers=["file", "error"], tablefmt="grid", stralign="left"))


if __name__ == "__main__":
    main()

