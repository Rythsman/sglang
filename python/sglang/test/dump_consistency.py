"""Utilities for validating dump directory consistency.

This module is intentionally independent from CLI so it can be reused by tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import json
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclasses.dataclass
class DumpConsistencyStats:
    """Statistics collected during dump directory comparison."""

    compared_files: int = 0
    compared_tensors: int = 0
    skipped_tensors: int = 0
    mismatched_files: int = 0
    mismatched_tensors: int = 0
    errors: int = 0


class DumpConsistencyTester:
    """Compare two dump directories for consistency.

    The tester supports a pragmatic, scalable approach:
    - File set must match exactly (relative paths).
    - Non-tensor files are compared by SHA256.
    - Tensor container files (.pth/.pt/.bin/.safetensors) are compared by:
        - Keys/metadata equality
        - Tensor shape/dtype equality
        - Values equality with numerical tolerance on a bounded subset

    This keeps the test fast enough for large models while still catching bugs.
    """

    _TENSOR_EXTS = (".pth", ".pt", ".bin", ".safetensors")
    _TEXT_EXTS = (".json", ".txt", ".md", ".yaml", ".yml", ".ini", ".cfg")

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_tensors_per_file: int = 64,
        verify_tensor_values: bool = True,
        json_ignore_keys: Optional[Sequence[str]] = None,
    ):
        self._tolerance = float(tolerance)
        self._max_tensors_per_file = int(max_tensors_per_file)
        self._verify_tensor_values = bool(verify_tensor_values)
        self._json_ignore_keys = set(
            json_ignore_keys
            if json_ignore_keys is not None
            else (
                "timestamp",
                "time",
                "date",
                "dump_time",
                "run_id",
                "pid",
                "hostname",
            )
        )
        self._stats = DumpConsistencyStats()
        self._errors: List[str] = []

    @property
    def stats(self) -> DumpConsistencyStats:
        return self._stats

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def compare_directories(self, dir1: str, dir2: str) -> bool:
        """Compare two directories and return True if consistent."""
        files1 = self._collect_files(dir1)
        files2 = self._collect_files(dir2)

        if files1.keys() != files2.keys():
            missing_in_2 = sorted(set(files1.keys()) - set(files2.keys()))
            missing_in_1 = sorted(set(files2.keys()) - set(files1.keys()))
            self._errors.append(f"File set mismatch: missing_in_dir2={missing_in_2}")
            self._errors.append(f"File set mismatch: missing_in_dir1={missing_in_1}")
            return False

        ok = True
        for rel_path in sorted(files1.keys()):
            path1 = files1[rel_path]
            path2 = files2[rel_path]
            try:
                if rel_path.endswith(self._TENSOR_EXTS):
                    ok = self._compare_tensor_file(path1, path2) and ok
                else:
                    ok = self._compare_bytes_file(path1, path2) and ok
                self._stats.compared_files += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                ok = False
                self._stats.errors += 1
                self._errors.append(f"Error comparing {rel_path}: {e}")
        return ok

    def _collect_files(self, root_dir: str) -> Dict[str, str]:
        root_dir = os.path.abspath(root_dir)
        results: Dict[str, str] = {}
        for cur_dir, _, filenames in os.walk(root_dir):
            for filename in filenames:
                abs_path = os.path.join(cur_dir, filename)
                rel_path = os.path.relpath(abs_path, root_dir)
                results[rel_path] = abs_path
        return results

    def _compare_bytes_file(self, path1: str, path2: str) -> bool:
        if path1.endswith(".json") and path2.endswith(".json"):
            return self._compare_json_file(path1, path2)

        return self._compare_raw_bytes_file(path1, path2)

    def _compare_raw_bytes_file(self, path1: str, path2: str) -> bool:
        if os.path.getsize(path1) != os.path.getsize(path2):
            self._stats.mismatched_files += 1
            self._errors.append(
                f"File size mismatch: {path1} ({os.path.getsize(path1)}) vs "
                f"{path2} ({os.path.getsize(path2)})"
            )
            return False

        h1 = self._sha256(path1)
        h2 = self._sha256(path2)
        if h1 != h2:
            self._stats.mismatched_files += 1
            self._errors.append(f"File hash mismatch: {path1} vs {path2}")
            return False
        return True

    def _compare_json_file(self, path1: str, path2: str) -> bool:
        try:
            with open(path1, "r", encoding="utf-8") as f:
                obj1 = json.load(f)
            with open(path2, "r", encoding="utf-8") as f:
                obj2 = json.load(f)
        except Exception:
            # Fallback to byte-level comparison if parsing fails.
            return self._compare_raw_bytes_file(path1, path2)

        norm1 = self._normalize_json(obj1)
        norm2 = self._normalize_json(obj2)
        if norm1 != norm2:
            self._stats.mismatched_files += 1
            self._errors.append(f"JSON content mismatch: {path1} vs {path2}")
            return False
        return True

    def _normalize_json(self, obj):
        if isinstance(obj, dict):
            items = {}
            for k, v in obj.items():
                if isinstance(k, str) and k in self._json_ignore_keys:
                    continue
                items[k] = self._normalize_json(v)
            return {k: items[k] for k in sorted(items.keys(), key=str)}
        if isinstance(obj, list):
            return [self._normalize_json(x) for x in obj]
        return obj

    def _compare_tensor_file(self, path1: str, path2: str) -> bool:
        if path1.endswith(".safetensors"):
            return self._compare_safetensors(path1, path2)
        return self._compare_torch_save(path1, path2)

    def _compare_safetensors(self, path1: str, path2: str) -> bool:
        try:
            from safetensors import safe_open
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._errors.append(f"safetensors import failed: {e}")
            self._stats.errors += 1
            return False

        ok = True
        with safe_open(path1, framework="pt", device="cpu") as f1, safe_open(
            path2, framework="pt", device="cpu"
        ) as f2:
            keys1 = sorted(list(f1.keys()))
            keys2 = sorted(list(f2.keys()))
            if keys1 != keys2:
                self._stats.mismatched_files += 1
                self._errors.append(f"Safetensors keys mismatch: {path1} vs {path2}")
                return False

            keys_to_check = keys1[: self._max_tensors_per_file]
            self._stats.skipped_tensors += max(0, len(keys1) - len(keys_to_check))
            for key in keys_to_check:
                t1 = f1.get_tensor(key)
                t2 = f2.get_tensor(key)
                ok = self._compare_tensor(key, t1, t2) and ok
        return ok

    def _compare_torch_save(self, path1: str, path2: str) -> bool:
        torch = self._import_torch()
        obj1 = torch.load(path1, map_location="cpu")
        obj2 = torch.load(path2, map_location="cpu")

        tensors1 = self._extract_tensor_dict(obj1)
        tensors2 = self._extract_tensor_dict(obj2)
        if tensors1 is None or tensors2 is None:
            return self._compare_bytes_file(path1, path2)

        keys1 = sorted(list(tensors1.keys()))
        keys2 = sorted(list(tensors2.keys()))
        if keys1 != keys2:
            self._stats.mismatched_files += 1
            self._errors.append(f"Tensor dict keys mismatch: {path1} vs {path2}")
            return False

        ok = True
        keys_to_check = keys1[: self._max_tensors_per_file]
        self._stats.skipped_tensors += max(0, len(keys1) - len(keys_to_check))
        for key in keys_to_check:
            ok = self._compare_tensor(key, tensors1[key], tensors2[key]) and ok
        return ok

    def _compare_tensor(self, name: str, t1, t2) -> bool:
        torch = self._import_torch()
        self._stats.compared_tensors += 1

        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            self._stats.mismatched_tensors += 1
            self._errors.append(f"Non-tensor entry for key={name}")
            return False

        if t1.shape != t2.shape or t1.dtype != t2.dtype:
            self._stats.mismatched_tensors += 1
            self._errors.append(
                f"Tensor meta mismatch for key={name}: "
                f"{t1.shape}/{t1.dtype} vs {t2.shape}/{t2.dtype}"
            )
            return False

        if not self._verify_tensor_values:
            return True

        try:
            if not torch.allclose(
                t1, t2, rtol=self._tolerance, atol=self._tolerance, equal_nan=True
            ):
                self._stats.mismatched_tensors += 1
                self._errors.append(f"Tensor values mismatch for key={name}")
                return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._stats.errors += 1
            self._errors.append(f"Error comparing tensor key={name}: {e}")
            return False

        return True

    def _extract_tensor_dict(self, obj) -> Optional[Dict[str, "torch.Tensor"]]:
        torch = self._import_torch()
        if isinstance(obj, dict) and obj:
            if all(isinstance(k, str) for k in obj.keys()) and any(
                isinstance(v, torch.Tensor) for v in obj.values()
            ):
                # Typical state_dict-like container.
                return obj
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
        return None

    def _sha256(self, path: str, chunk_bytes: int = 8 * 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _import_torch(self):
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                "torch is required for tensor dump consistency checks"
            ) from e
        return torch

