"""Run an RL regression test pipeline described in YAML.

This runner is intentionally lightweight:
- Steps are executed sequentially.
- Each step runs a single Python test file (unittest/pytest-compatible).
- Optional per-step env and timeout are supported.

Design goals:
- KISS: minimal behavior, minimal schema.
- Reuse: reuse SGLang's existing "python3 <test_file>" execution style.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Step:
    """A single pipeline step."""

    name: str
    file: str
    timeout_s: int
    env: Dict[str, str]


@dataclass(frozen=True)
class Pipeline:
    """Parsed pipeline configuration."""

    name: str
    version: int
    continue_on_error: bool
    pythonpath: Sequence[str]
    steps: Sequence[Step]


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "PyYAML is required to load pipeline YAML. "
            "Please install it (e.g., pip install pyyaml)."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML root must be a mapping.")
    return data


def _as_str_dict(obj: Any) -> Dict[str, str]:
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("env must be a mapping.")
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError("env keys must be strings.")
        out[k] = "" if v is None else str(v)
    return out


def _parse_pipeline(data: Dict[str, Any]) -> Pipeline:
    name = str(data.get("name", "rl-regression"))
    version = int(data.get("version", 1))
    defaults = data.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")

    default_timeout_s = int(defaults.get("timeout_s", 1800))
    continue_on_error = bool(defaults.get("continue_on_error", False))
    pythonpath = defaults.get("pythonpath", ["python", "test"])
    if not isinstance(pythonpath, list) or not all(isinstance(x, str) for x in pythonpath):
        raise ValueError("defaults.pythonpath must be a list of strings.")

    raw_steps = data.get("steps", []) or []
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list.")

    steps: List[Step] = []
    for idx, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid step at index {idx}.")
        step_name = str(raw.get("name", f"step-{idx+1}"))
        file_path = raw.get("file")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(f"Step '{step_name}' must set a non-empty 'file'.")
        timeout_s = int(raw.get("timeout_s", default_timeout_s))
        env = _as_str_dict(raw.get("env"))
        steps.append(Step(name=step_name, file=file_path, timeout_s=timeout_s, env=env))

    if not steps:
        raise ValueError("Pipeline must contain at least one step.")

    return Pipeline(
        name=name,
        version=version,
        continue_on_error=continue_on_error,
        pythonpath=pythonpath,
        steps=steps,
    )


def _prepend_pythonpath(repo_root: Path, extra_entries: Sequence[str]):
    entries = [str((repo_root / x).resolve()) for x in extra_entries]
    existing = os.environ.get("PYTHONPATH", "")
    combined = ":".join([*entries, existing]) if existing else ":".join(entries)
    os.environ["PYTHONPATH"] = combined


def _run_python_file(path: str, timeout_s: int, env: Dict[str, str]) -> int:
    """Run `python3 <path>` with a timeout and return exit code."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    proc = subprocess.Popen(
        ["python3", path],
        stdout=None,
        stderr=None,
        env=env,
        start_new_session=True,  # Create a new process group on Linux.
    )
    try:
        proc.wait(timeout=timeout_s)
        return int(proc.returncode or 0)
    except subprocess.TimeoutExpired:
        try:
            # Try a graceful stop first.
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 124


def _run_step(step: Step, repo_root: Path) -> int:
    print(f"\n=== Running step: {step.name} ===", flush=True)
    print(f"file={step.file}", flush=True)
    print(f"timeout_s={step.timeout_s}", flush=True)

    env = dict(os.environ)
    env.update(step.env)

    abs_path = str((repo_root / step.file).resolve())
    start = time.perf_counter()
    try:
        code = _run_python_file(abs_path, timeout_s=step.timeout_s, env=env)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"exception={e}", flush=True)
        code = 1
    elapsed = time.perf_counter() - start
    print(f"exit_code={code} elapsed_s={elapsed:.1f}", flush=True)
    return 0 if code == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = _REPO_ROOT
    default_config = repo_root / "test" / "srt" / "rl" / "rl_regression_pipeline.yaml"

    parser = argparse.ArgumentParser(description="Run RL regression pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining steps even if one fails.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    data = _load_yaml(args.config)
    pipeline = _parse_pipeline(data)

    # Ensure local imports work when running from source.
    _prepend_pythonpath(repo_root, pipeline.pythonpath)

    print(f"Pipeline: {pipeline.name} (version={pipeline.version})", flush=True)
    print(f"Steps: {[s.name for s in pipeline.steps]}", flush=True)

    any_failed = False
    passed_steps: List[str] = []
    failed_steps: List[str] = []
    for step in pipeline.steps:
        ret = _run_step(step, repo_root=repo_root)
        if ret != 0:
            any_failed = True
            failed_steps.append(step.name)
            if not (args.continue_on_error or pipeline.continue_on_error):
                break
        else:
            passed_steps.append(step.name)

    print("\n" + "=" * 60, flush=True)
    print(f"Summary: {len(passed_steps)}/{len(pipeline.steps)} passed", flush=True)
    if passed_steps:
        print(f"PASSED: {passed_steps}", flush=True)
    if failed_steps:
        print(f"FAILED: {failed_steps}", flush=True)
    print("=" * 60, flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

