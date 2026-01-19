"""
Unit test script for _read_video_decord function in keye.py.

This script validates the output of _read_video_decord given a video input.

Usage:
    python test_read_video_decord.py --video_path /path/to/video.mp4 [options]

Options:
    --video_path    Path to the video file (required)
    --fps           Target FPS for frame extraction (default: 2.0)
    --max_frames    Maximum number of frames (default: None)
    --min_pixels    Minimum pixels per frame (default: 37632)
    --max_pixels    Maximum pixels per frame (default: 602112)
    --min_frame_similarity  Threshold for slow/fast frame classification (default: 0.9)
    --verbose       Print detailed output
    --save_output   Save output tensors to a file

Example:
    python test_read_video_decord.py --video_path test.mp4 --fps 2.0 --verbose
"""

import argparse
import os
import sys
from typing import Dict, Any, Tuple

import numpy as np
import torch


def load_video_reader(video_path: str):
    """Load a video using decord.VideoReader."""
    from decord import VideoReader, cpu, gpu

    try:
        from decord.bridge import decord_bridge
        ctx = gpu(0)
        _ = decord_bridge.get_ctx_device(ctx)
    except Exception:
        ctx = cpu(0)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vr = VideoReader(video_path, ctx=ctx)
    return vr


def test_read_video_decord(
    video_path: str,
    ele: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test _read_video_decord function with given video and config.

    Args:
        video_path: Path to the video file
        ele: Configuration dict for video processing
        verbose: Whether to print detailed output

    Returns:
        Tuple of (frames, timestamps, frame_types)
    """
    from sglang.srt.multimodal.processors.keye import _read_video_decord

    # Load video
    vr = load_video_reader(video_path)

    # Get video info
    nframes = len(vr)
    video_fps = vr.get_avg_fps()

    if verbose:
        print("=" * 60)
        print("VIDEO INFO")
        print("=" * 60)
        print(f"  Video path: {video_path}")
        print(f"  Total frames: {nframes}")
        print(f"  FPS: {video_fps:.2f}")
        print(f"  Duration: {nframes / video_fps:.2f} seconds")
        print()
        print("CONFIG (ele)")
        print("-" * 40)
        for k, v in ele.items():
            print(f"  {k}: {v}")
        print()

    # Call _read_video_decord
    frames, timestamps, frame_types = _read_video_decord(vr, ele)

    if verbose:
        print("=" * 60)
        print("OUTPUT VALIDATION")
        print("=" * 60)

        # Validate frames
        print("\n[1] FRAMES TENSOR")
        print("-" * 40)
        print(f"  Shape: {frames.shape}")
        print(f"  Expected format: (T, C, H, W)")
        print(f"    T (num frames): {frames.shape[0]}")
        print(f"    C (channels): {frames.shape[1]}")
        print(f"    H (height): {frames.shape[2]}")
        print(f"    W (width): {frames.shape[3]}")
        print(f"  Dtype: {frames.dtype}")
        print(f"  Min value: {frames.min().item()}")
        print(f"  Max value: {frames.max().item()}")
        print(f"  Mean value: {frames.float().mean().item():.2f}")

        # Validate timestamps
        print("\n[2] TIMESTAMPS TENSOR")
        print("-" * 40)
        print(f"  Shape: {timestamps.shape}")
        print(f"  Dtype: {timestamps.dtype}")
        print(f"  Values: {timestamps.tolist()}")
        print(f"  Time range: [{timestamps[0].item():.3f}s, {timestamps[-1].item():.3f}s]")

        # Validate frame_types
        print("\n[3] FRAME_TYPES TENSOR")
        print("-" * 40)
        print(f"  Shape: {frame_types.shape}")
        print(f"  Dtype: {frame_types.dtype}")
        print(f"  Values: {frame_types.tolist()}")
        slow_count = (frame_types == 0).sum().item()
        fast_count = (frame_types == 1).sum().item()
        print(f"  Slow frames (type=0): {slow_count}")
        print(f"  Fast frames (type=1): {fast_count}")
        print(f"  Slow ratio: {slow_count / len(frame_types) * 100:.1f}%")

        # Consistency checks
        print("\n[4] CONSISTENCY CHECKS")
        print("-" * 40)
        checks_passed = True

        # Check 1: Same number of frames
        if frames.shape[0] == timestamps.shape[0] == frame_types.shape[0]:
            print(f"  ✅ Frame count consistency: {frames.shape[0]} frames")
        else:
            print(f"  ❌ Frame count mismatch: frames={frames.shape[0]}, "
                  f"timestamps={timestamps.shape[0]}, frame_types={frame_types.shape[0]}")
            checks_passed = False

        # Check 2: Timestamps are monotonically increasing
        if torch.all(timestamps[1:] >= timestamps[:-1]):
            print("  ✅ Timestamps are monotonically increasing")
        else:
            print("  ❌ Timestamps are NOT monotonically increasing")
            checks_passed = False

        # Check 3: Frame types are 0 or 1
        if torch.all((frame_types == 0) | (frame_types == 1)):
            print("  ✅ Frame types are all 0 or 1")
        else:
            unique_types = frame_types.unique().tolist()
            print(f"  ❌ Unexpected frame types: {unique_types}")
            checks_passed = False

        # Check 4: Frames shape is valid (T, C, H, W)
        if len(frames.shape) == 4 and frames.shape[1] == 3:
            print("  ✅ Frames shape is valid (T, C, H, W) with C=3")
        else:
            print(f"  ❌ Invalid frames shape: {frames.shape}")
            checks_passed = False

        # Check 5: First frame is always slow (type=0)
        if frame_types[0] == 0:
            print("  ✅ First frame is slow (type=0)")
        else:
            print(f"  ⚠️  First frame is fast (type={frame_types[0].item()})")

        print("\n" + "=" * 60)
        if checks_passed:
            print("ALL CHECKS PASSED ✅")
        else:
            print("SOME CHECKS FAILED ❌")
        print("=" * 60)

    return frames, timestamps, frame_types


def compare_with_expected(
    frames: torch.Tensor,
    timestamps: torch.Tensor,
    frame_types: torch.Tensor,
    expected: Dict[str, Any],
) -> bool:
    """
    Compare output with expected values.

    Args:
        frames, timestamps, frame_types: Output from _read_video_decord
        expected: Dict with expected values, e.g.:
            {
                "num_frames": 10,
                "frame_shape": (10, 3, 480, 640),
                "timestamps": [0.0, 0.5, 1.0, ...],
                "frame_types": [0, 1, 1, ...],
            }

    Returns:
        True if all checks pass
    """
    all_passed = True

    if "num_frames" in expected:
        if frames.shape[0] == expected["num_frames"]:
            print(f"✅ num_frames: {frames.shape[0]} == {expected['num_frames']}")
        else:
            print(f"❌ num_frames: {frames.shape[0]} != {expected['num_frames']}")
            all_passed = False

    if "frame_shape" in expected:
        if tuple(frames.shape) == tuple(expected["frame_shape"]):
            print(f"✅ frame_shape: {tuple(frames.shape)} == {expected['frame_shape']}")
        else:
            print(f"❌ frame_shape: {tuple(frames.shape)} != {expected['frame_shape']}")
            all_passed = False

    if "timestamps" in expected:
        expected_ts = torch.tensor(expected["timestamps"], dtype=torch.float32)
        if torch.allclose(timestamps, expected_ts, atol=1e-3):
            print("✅ timestamps match expected values")
        else:
            print(f"❌ timestamps mismatch:")
            print(f"   Got:      {timestamps.tolist()}")
            print(f"   Expected: {expected['timestamps']}")
            all_passed = False

    if "frame_types" in expected:
        expected_ft = torch.tensor(expected["frame_types"], dtype=torch.int32)
        if torch.equal(frame_types, expected_ft):
            print("✅ frame_types match expected values")
        else:
            print(f"❌ frame_types mismatch:")
            print(f"   Got:      {frame_types.tolist()}")
            print(f"   Expected: {expected['frame_types']}")
            all_passed = False

    return all_passed


def save_output(
    frames: torch.Tensor,
    timestamps: torch.Tensor,
    frame_types: torch.Tensor,
    output_path: str,
):
    """Save output tensors to a file."""
    torch.save({
        "frames": frames,
        "timestamps": timestamps,
        "frame_types": frame_types,
    }, output_path)
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test _read_video_decord function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for frame extraction (default: 2.0)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames (default: None)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=37632,  # VIDEO_MIN_PIXELS
        help="Minimum pixels per frame (default: 37632)",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=602112,  # VIDEO_MAX_PIXELS
        help="Maximum pixels per frame (default: 602112)",
    )
    parser.add_argument(
        "--video_total_pixels",
        type=int,
        default=6422528,  # VIDEO_TOTAL_PIXELS
        help="Total pixels for all frames (default: 6422528)",
    )
    parser.add_argument(
        "--min_frame_similarity",
        type=float,
        default=0.9,
        help="Threshold for slow/fast frame classification (default: 0.9)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--save_output",
        type=str,
        default=None,
        help="Save output tensors to this file (e.g., output.pt)",
    )
    parser.add_argument(
        "--expected_num_frames",
        type=int,
        default=None,
        help="Expected number of frames for validation",
    )
    parser.add_argument(
        "--expected_frame_types",
        type=str,
        default=None,
        help="Expected frame types as comma-separated values (e.g., '0,1,1,0,1')",
    )

    args = parser.parse_args()

    # Build config dict (ele)
    ele = {
        "fps": args.fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "video_total_pixels": args.video_total_pixels,
        "min_frame_similarity": args.min_frame_similarity,
    }
    if args.max_frames is not None:
        ele["max_frames"] = args.max_frames

    # Run test
    frames, timestamps, frame_types = test_read_video_decord(
        video_path=args.video_path,
        ele=ele,
        verbose=args.verbose,
    )

    # Compare with expected if provided
    if args.expected_num_frames or args.expected_frame_types:
        print("\n" + "=" * 60)
        print("EXPECTED VALUE COMPARISON")
        print("=" * 60)

        expected = {}
        if args.expected_num_frames:
            expected["num_frames"] = args.expected_num_frames
        if args.expected_frame_types:
            expected["frame_types"] = [
                int(x) for x in args.expected_frame_types.split(",")
            ]

        compare_with_expected(frames, timestamps, frame_types, expected)

    # Save output if requested
    if args.save_output:
        save_output(frames, timestamps, frame_types, args.save_output)

    # Print summary for non-verbose mode
    if not args.verbose:
        print(f"Frames: {tuple(frames.shape)}")
        print(f"Timestamps: {timestamps.shape[0]} values, "
              f"range [{timestamps[0].item():.3f}s, {timestamps[-1].item():.3f}s]")
        slow_count = (frame_types == 0).sum().item()
        fast_count = (frame_types == 1).sum().item()
        print(f"Frame types: {slow_count} slow, {fast_count} fast")

    return frames, timestamps, frame_types


# For programmatic usage
def run_test(
    video_path: str,
    fps: float = 2.0,
    max_frames: int = None,
    min_pixels: int = 37632,
    max_pixels: int = 602112,
    video_total_pixels: int = 6422528,
    min_frame_similarity: float = 0.9,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Programmatic interface to run the test.

    Example:
        from test_read_video_decord import run_test
        frames, timestamps, frame_types = run_test(
            video_path="/path/to/video.mp4",
            fps=2.0,
            verbose=True,
        )
    """
    ele = {
        "fps": fps,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "video_total_pixels": video_total_pixels,
        "min_frame_similarity": min_frame_similarity,
    }
    if max_frames is not None:
        ele["max_frames"] = max_frames

    return test_read_video_decord(
        video_path=video_path,
        ele=ele,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
