#!/usr/bin/env python3
"""
Copy a HuggingFace-style model folder, but symlink weight files.

This script copies all non-weight files (configs, tokenizers, indexes, etc.)
with metadata preserved, while creating `ln -s`-style symlinks for weight files
to avoid duplicating large checkpoints.

Typical weight shards look like:
  - model-00136-of-00337.safetensors
  - pytorch_model-00001-of-000xx.bin

Examples:
  # Copy non-weights, symlink weights (absolute symlinks by default)
  python3 scripts/copy_hf_folder_with_weight_symlinks.py --src /path/to/model --dst /path/to/model_copy

  # Create relative symlinks (portable if you move the dst folder)
  python3 scripts/copy_hf_folder_with_weight_symlinks.py --src /path/to/model --dst /path/to/model_copy --relative

  # Dry run (print actions only)
  python3 scripts/copy_hf_folder_with_weight_symlinks.py --src /path/to/model --dst /path/to/model_copy --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_WEIGHT_GLOBS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.onnx",
)


def _matches_any_glob(name: str, globs: Sequence[str]) -> bool:
    """Return True if name matches any fnmatch glob."""
    for pattern in globs:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def _is_probably_weight_file(path: Path, weight_globs: Sequence[str]) -> bool:
    """
    Heuristic: treat common checkpoint artifacts as weight files.

    Notes:
    - We intentionally do NOT treat `*.index.json` as weights (they are small and
      should be copied).
    - Users can override with --weights-glob and --exclude-weights-glob.
    """
    name = path.name
    if name.endswith(".index.json"):
        return False
    if _matches_any_glob(name, weight_globs):
        return True

    # Extra shard naming patterns sometimes don't fit simple globs.
    # Example: model-00001-of-00010.safetensors (handled by *.safetensors),
    # but keep this for clarity/extendability.
    if "-of-" in name and (name.endswith(".safetensors") or name.endswith(".bin")):
        return True

    return False


def _is_within(child: Path, parent: Path) -> bool:
    """Return True if child is within parent (after resolving)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _safe_unlink(path: Path) -> None:
    """Remove a file/symlink if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _ensure_empty_dir(dst_dir: Path, overwrite: bool) -> None:
    """Create an empty destination directory (optionally overwriting)."""
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_dir}. "
                "Use --overwrite to remove it first."
            )
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


def _make_symlink_target(src_file: Path, dst_file: Path, relative: bool) -> str:
    """Compute symlink target string (absolute or relative)."""
    if not relative:
        return str(src_file.resolve())
    return os.path.relpath(str(src_file.resolve()), start=str(dst_file.parent.resolve()))


@dataclass
class Stats:
    copied_files: int = 0
    linked_files: int = 0
    copied_bytes: int = 0


def copy_hf_folder_with_weight_symlinks(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    relative_links: bool,
    weight_globs: Sequence[str],
    exclude_weight_globs: Sequence[str],
    min_weight_size_bytes: int,
) -> Stats:
    """Copy src_dir to dst_dir; symlink weight files; return copy stats."""
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_dir}")
    if _is_within(dst_dir, src_dir):
        raise ValueError(
            f"Destination must not be inside source. src={src_dir} dst={dst_dir}"
        )

    if dry_run:
        if dst_dir.exists() and overwrite:
            print(f"[dry-run] Would remove: {dst_dir}")
        print(f"[dry-run] Would create: {dst_dir}")
    else:
        _ensure_empty_dir(dst_dir, overwrite=overwrite)

    stats = Stats()

    for root, dirnames, filenames in os.walk(src_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        dst_root = dst_dir / rel_root

        if dry_run:
            print(f"[dry-run] Would ensure dir: {dst_root}")
        else:
            dst_root.mkdir(parents=True, exist_ok=True)

        # Keep directory traversal deterministic.
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            src_file = root_path / filename
            dst_file = dst_root / filename

            # Preserve existing symlinks in source as symlinks in destination.
            if src_file.is_symlink():
                link_target = os.readlink(src_file)
                if dry_run:
                    print(f"[dry-run] Would symlink (preserve): {dst_file} -> {link_target}")
                    continue
                _safe_unlink(dst_file)
                os.symlink(link_target, dst_file)
                stats.linked_files += 1
                continue

            if not src_file.is_file():
                # Skip special files.
                continue

            is_weight = _is_probably_weight_file(src_file, weight_globs=weight_globs)
            if is_weight and exclude_weight_globs and _matches_any_glob(
                src_file.name, exclude_weight_globs
            ):
                is_weight = False

            if is_weight and min_weight_size_bytes > 0:
                try:
                    if src_file.stat().st_size < min_weight_size_bytes:
                        is_weight = False
                except FileNotFoundError:
                    # Source disappeared during walk; ignore.
                    continue

            if is_weight:
                target = _make_symlink_target(
                    src_file=src_file, dst_file=dst_file, relative=relative_links
                )
                if dry_run:
                    print(f"[dry-run] Would symlink: {dst_file} -> {target}")
                    stats.linked_files += 1
                    continue
                _safe_unlink(dst_file)
                os.symlink(target, dst_file)
                stats.linked_files += 1
                continue

            if dry_run:
                print(f"[dry-run] Would copy: {src_file} -> {dst_file}")
                stats.copied_files += 1
                try:
                    stats.copied_bytes += src_file.stat().st_size
                except FileNotFoundError:
                    pass
                continue

            shutil.copy2(src_file, dst_file)
            stats.copied_files += 1
            try:
                stats.copied_bytes += src_file.stat().st_size
            except FileNotFoundError:
                pass

    return stats


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a HuggingFace-style model folder: copy non-weights; "
            "symlink weight files."
        )
    )
    parser.add_argument("--src", required=True, help="Source model folder.")
    parser.add_argument("--dst", required=True, help="Destination folder.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying the filesystem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove destination folder if it already exists.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Create relative symlinks instead of absolute symlinks.",
    )
    parser.add_argument(
        "--weights-glob",
        action="append",
        default=[],
        help=(
            "Extra glob pattern for weight files (repeatable). "
            f"Default patterns: {', '.join(DEFAULT_WEIGHT_GLOBS)}"
        ),
    )
    parser.add_argument(
        "--exclude-weights-glob",
        action="append",
        default=[],
        help="Exclude glob pattern from weight detection (repeatable).",
    )
    parser.add_argument(
        "--min-weight-size-bytes",
        type=int,
        default=0,
        help=(
            "Only treat a file as weight if size >= this threshold. "
            "0 means no size threshold."
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    src_dir = Path(os.path.expanduser(args.src))
    dst_dir = Path(os.path.expanduser(args.dst))

    weight_globs = list(DEFAULT_WEIGHT_GLOBS) + list(args.weights_glob or [])
    exclude_weight_globs = list(args.exclude_weights_glob or [])

    try:
        stats = copy_hf_folder_with_weight_symlinks(
            src_dir=src_dir,
            dst_dir=dst_dir,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            relative_links=bool(args.relative),
            weight_globs=weight_globs,
            exclude_weight_globs=exclude_weight_globs,
            min_weight_size_bytes=int(args.min_weight_size_bytes),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("-" * 70)
        print(
            "Dry-run summary (estimated): "
            f"copied_files={stats.copied_files}, linked_files={stats.linked_files}, "
            f"copied_bytes={stats.copied_bytes}"
        )
        return 0

    print("-" * 70)
    print(
        "Done: "
        f"copied_files={stats.copied_files}, linked_files={stats.linked_files}, "
        f"copied_bytes={stats.copied_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
