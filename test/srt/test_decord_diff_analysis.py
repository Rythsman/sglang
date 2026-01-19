"""
Analysis script to compare decord frame outputs and identify differences.

This script helps analyze the root cause of frame value differences
between decord==0.6.0 and decord2==3.0.0.

Usage:
    # Compare saved outputs from two versions
    python test_decord_diff_analysis.py --old old_frames.pt --new new_frames.pt

    # Analyze current decord output
    python test_decord_diff_analysis.py --video /path/to/video.mp4 --analyze

    # Save current output for later comparison
    python test_decord_diff_analysis.py --video /path/to/video.mp4 --save output.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

import numpy as np


def analyze_single_output(frames: np.ndarray, name: str = "frames"):
    """Analyze a single frame output."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    print(f"Shape: {frames.shape}")
    print(f"Dtype: {frames.dtype}")
    print(f"Min: {frames.min()}")
    print(f"Max: {frames.max()}")
    print(f"Mean: {frames.mean():.6f}")
    print(f"Std: {frames.std():.6f}")

    # Per-channel statistics
    if len(frames.shape) == 4 and frames.shape[3] == 3:
        # Shape is (T, H, W, C)
        for c, name in enumerate(['R', 'G', 'B']):
            channel = frames[..., c]
            print(f"  {name} channel: mean={channel.mean():.4f}, std={channel.std():.4f}")
    elif len(frames.shape) == 4 and frames.shape[1] == 3:
        # Shape is (T, C, H, W)
        for c, name in enumerate(['R', 'G', 'B']):
            channel = frames[:, c, :, :]
            print(f"  {name} channel: mean={channel.mean():.4f}, std={channel.std():.4f}")

    return {
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "min": float(frames.min()),
        "max": float(frames.max()),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
    }


def compare_outputs(frames_old: np.ndarray, frames_new: np.ndarray):
    """Compare two frame outputs and analyze differences."""
    print(f"\n{'='*60}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*60}")

    # Shape check
    if frames_old.shape != frames_new.shape:
        print(f"❌ Shape mismatch: {frames_old.shape} vs {frames_new.shape}")
        return

    print(f"✅ Shape: {frames_old.shape}")

    # Convert to float for precise comparison
    old_f = frames_old.astype(np.float64)
    new_f = frames_new.astype(np.float64)

    # Compute difference
    diff = new_f - old_f
    abs_diff = np.abs(diff)

    print(f"\n[Difference Statistics]")
    print(f"  Min diff: {diff.min():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Std diff: {diff.std():.6f}")
    print(f"  Mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6f}")

    # Count pixels with different values
    diff_mask = abs_diff > 0
    diff_count = diff_mask.sum()
    total_count = diff_mask.size
    diff_ratio = diff_count / total_count * 100

    print(f"\n[Difference Distribution]")
    print(f"  Pixels with diff > 0: {diff_count:,} / {total_count:,} ({diff_ratio:.2f}%)")

    for threshold in [1, 2, 3, 5, 10]:
        count = (abs_diff > threshold).sum()
        ratio = count / total_count * 100
        print(f"  Pixels with diff > {threshold}: {count:,} ({ratio:.4f}%)")

    # Histogram of differences
    print(f"\n[Difference Histogram]")
    hist_bins = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
    hist, edges = np.histogram(diff.flatten(), bins=hist_bins + [np.inf])
    for i, (low, high) in enumerate(zip(hist_bins, hist_bins[1:] + [float('inf')])):
        print(f"  [{low:+3}, {high:+3}): {hist[i]:>10,}")

    # Per-frame analysis
    print(f"\n[Per-Frame Analysis]")
    num_frames = frames_old.shape[0]
    for f in range(min(num_frames, 5)):  # First 5 frames
        frame_diff = abs_diff[f]
        print(f"  Frame {f}: mean_diff={frame_diff.mean():.4f}, max_diff={frame_diff.max():.1f}")

    # Sample specific pixels
    print(f"\n[Sample Pixel Values (Frame 0, top-left 3x3, R channel)]")
    if len(frames_old.shape) == 4:
        if frames_old.shape[1] == 3:  # (T, C, H, W)
            old_sample = frames_old[0, 0, :3, :3]
            new_sample = frames_new[0, 0, :3, :3]
        else:  # (T, H, W, C)
            old_sample = frames_old[0, :3, :3, 0]
            new_sample = frames_new[0, :3, :3, 0]

        print(f"  Old:\n{old_sample}")
        print(f"  New:\n{new_sample}")
        print(f"  Diff:\n{new_sample.astype(float) - old_sample.astype(float)}")

    # Quality metrics
    mse = np.mean((old_f - new_f) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')

    print(f"\n[Quality Metrics]")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: (requires scipy)")

    # Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if abs_diff.max() == 0:
        print("✅ Outputs are IDENTICAL")
    elif abs_diff.max() <= 1:
        print("✅ Outputs have NEGLIGIBLE differences (max diff ≤ 1)")
        print("   Likely cause: Floating-point rounding in YUV→RGB conversion")
    elif abs_diff.max() <= 3:
        print("⚠️  Outputs have MINOR differences (max diff ≤ 3)")
        print("   Likely cause: FFmpeg version difference in color conversion")
    elif abs_diff.max() <= 10:
        print("⚠️  Outputs have MODERATE differences (max diff ≤ 10)")
        print("   Likely cause: Different decode algorithm or scaling method")
    else:
        print("❌ Outputs have SIGNIFICANT differences (max diff > 10)")
        print("   Likely cause: Major algorithm change or bug")

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "max_diff": float(abs_diff.max()),
        "mean_diff": float(abs_diff.mean()),
        "diff_ratio": float(diff_ratio),
    }


def load_frames_from_video(video_path: str, indices: list = None):
    """Load frames from video using current decord version."""
    from decord import VideoReader, cpu

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0))
    print(f"Video: {video_path}")
    print(f"Total frames: {len(vr)}")
    print(f"FPS: {vr.get_avg_fps():.2f}")

    if indices is None:
        # Sample 10 frames evenly
        total = len(vr)
        indices = np.linspace(0, total - 1, min(10, total), dtype=int).tolist()

    print(f"Extracting frames: {indices}")
    frames = vr.get_batch(indices)

    # Convert to numpy
    if hasattr(frames, 'asnumpy'):
        frames_np = frames.asnumpy()
    else:
        frames_np = np.array(frames)

    return frames_np, indices


def save_frames(frames: np.ndarray, indices: list, output_path: str, metadata: dict = None):
    """Save frames to file for later comparison."""
    import pickle

    data = {
        "frames": frames,
        "indices": indices,
        "metadata": metadata or {},
    }

    # Try torch.save if available
    try:
        import torch
        torch.save(data, output_path)
        print(f"Saved to: {output_path} (torch format)")
    except ImportError:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to: {output_path} (pickle format)")


def load_frames(path: str):
    """Load frames from saved file."""
    import pickle

    try:
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
    except:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    frames = data["frames"]
    if hasattr(frames, 'numpy'):
        frames = frames.numpy()

    return np.array(frames), data.get("indices", []), data.get("metadata", {})


def main():
    parser = argparse.ArgumentParser(description="Analyze decord frame differences")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--old", type=str, help="Path to old version output file")
    parser.add_argument("--new", type=str, help="Path to new version output file")
    parser.add_argument("--save", type=str, help="Save current output to file")
    parser.add_argument("--analyze", action="store_true", help="Analyze current output")
    parser.add_argument("--indices", type=str, help="Frame indices (comma-separated)")

    args = parser.parse_args()

    indices = None
    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]

    # Compare two saved outputs
    if args.old and args.new:
        print("Loading old version output...")
        frames_old, indices_old, meta_old = load_frames(args.old)
        print(f"  Shape: {frames_old.shape}, indices: {indices_old}")
        if meta_old:
            print(f"  Metadata: {meta_old}")

        print("\nLoading new version output...")
        frames_new, indices_new, meta_new = load_frames(args.new)
        print(f"  Shape: {frames_new.shape}, indices: {indices_new}")
        if meta_new:
            print(f"  Metadata: {meta_new}")

        analyze_single_output(frames_old, "Old Version")
        analyze_single_output(frames_new, "New Version")
        compare_outputs(frames_old, frames_new)
        return

    # Analyze or save current output
    if args.video:
        frames, frame_indices = load_frames_from_video(args.video, indices)

        if args.analyze:
            analyze_single_output(frames, "Current Output")

        if args.save:
            # Get decord version
            try:
                import decord
                version = getattr(decord, '__version__', 'unknown')
            except:
                version = 'unknown'

            metadata = {
                "decord_version": version,
                "video_path": args.video,
            }
            save_frames(frames, frame_indices, args.save, metadata)

        if not args.analyze and not args.save:
            analyze_single_output(frames, "Current Output")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
